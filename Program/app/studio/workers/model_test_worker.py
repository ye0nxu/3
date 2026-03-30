from __future__ import annotations

import contextlib
import csv
import json
import logging
import math
import os
import queue
import random
import re
import shlex
import shutil
import stat
import subprocess
import sys
import textwrap
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from config import apply_remote_env_defaults
from utils.env import env_flag as _env_flag, env_int as _env_int
try:
    from core.dataset import BoxAnnotation, FrameAnnotation
except ModuleNotFoundError:
    PROJECT_DIR = Path(__file__).resolve().parents[2]
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from core.dataset import BoxAnnotation, FrameAnnotation

from backend.pipelines.preview_postprocess import postprocess_preview_items
from backend.storage.remote_storage import remote_storage_enabled, sync_local_tree_to_remote
from core.paths import (
    CROP_SAVE_BASE_DIR,
    DATASET_SAVE_DIR,
    MERGED_DATASET_SAVE_DIR,
    TEAM_MODEL_DIR,
    TRAIN_RTDETR_MODELS_DIR,
    TRAIN_RUNS_DIR,
    TRAIN_YOLO_MODELS_DIR,
    ensure_storage_directories,
    model_storage_dir_for_name,
)
from app.studio.config import (
    DEFAULT_CLASS_NAMES,
    TEAM_MODEL_PATH,
    TRAIN_DEFAULT_FREEZE,
    TRAIN_DEFAULT_LR0,
    TRAIN_REPLAY_RATIO_DEFAULT,
    TRAIN_RETRAIN_SEED_DEFAULT,
    TRAIN_STAGE1_EPOCHS_DEFAULT,
    TRAIN_STAGE2_EPOCHS_DEFAULT,
    TRAIN_STAGE2_LR_FACTOR_DEFAULT,
    TRAIN_STAGE_UNFREEZE_BACKBONE_LAST,
    TRAIN_STAGE_UNFREEZE_CHOICES,
    TRAIN_STAGE_UNFREEZE_NECK_ONLY,
)
from core.models import (
    ExportRunSummary,
    PreviewThumbnail,
    ProgressEvent,
    WorkerOutput,
    WorkerStoppedError,
)
from app.studio.runtime import (
    FilterConfig,
    SampleCandidate,
    SampleFilterEngine,
    TeamTrackObject,
    TqdmFormatter,
    ULTRALYTICS_VERSION,
    UltralyticsRTDETR,
    UltralyticsYOLO,
    cv2,
    torch,
    yaml,
)
from app.studio.utils import (
    _build_component_from_yaml,
    _build_unique_path,
    _collect_split_counts_from_yaml,
    _dedupe_preserve_order,
    _guess_label_path_for_image,
    _load_yaml_dict,
    _metric_to_unit_interval,
    _normalize_split_entries,
    _parse_version_tuple,
    _parse_names_from_yaml_payload,
    _read_image_list_file,
    _resolve_dataset_root_from_yaml_payload,
    _resolve_managed_model_source,
    _resolve_split_images,
    _scan_images_in_directory,
    _sync_ultralytics_datasets_dir,
    _temporary_working_directory,
    _try_autofix_data_yaml_path,
    _try_autofix_data_yaml_splits,
    _write_replay_dataset_readme,
    build_dataset_folder_name,
    collect_original_components_from_yaml,
    extract_training_metrics_and_losses,
    sanitize_class_name,
    write_provenance_readme,
)

class ModelTestWorker(QObject):
    """학습된 모델로 테스트 영상을 순차 추론하며 프리뷰 프레임을 발행하는 워커입니다."""

    log_message = pyqtSignal(str)
    frame_ready = pyqtSignal(object, object, int, int)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        model_path: Path,
        video_path: Path,
        conf: float,
        iou: float,
        imgsz: int,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path)
        self.video_path = Path(video_path)
        self.conf = max(0.01, min(0.99, float(conf)))
        self.iou = max(0.05, min(0.99, float(iou)))
        self.imgsz = max(320, int(imgsz))
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    @pyqtSlot()
    def run(self) -> None:
        cap: cv2.VideoCapture | None = None
        try:
            if UltralyticsYOLO is None:
                raise RuntimeError("ultralytics 패키지를 불러올 수 없습니다.")
            if not self.model_path.is_file():
                raise RuntimeError(f"모델 파일이 없습니다: {self.model_path}")
            if not self.video_path.is_file():
                raise RuntimeError(f"영상 파일이 없습니다: {self.video_path}")

            self.log_message.emit(
                f"test start: model={self.model_path.name}, video={self.video_path.name}, conf={self.conf:.2f}, iou={self.iou:.2f}, imgsz={self.imgsz}"
            )
            model = UltralyticsYOLO(str(self.model_path))
            # 모델의 실제 클래스명 맵을 추출합니다 (class_id → class_name).
            _raw_names = getattr(model, "names", {})
            model_class_names: dict[int, str] = (
                {int(k): str(v) for k, v in _raw_names.items()}
                if isinstance(_raw_names, dict)
                else {i: str(v) for i, v in enumerate(_raw_names)}
            )
            if model_class_names:
                self.log_message.emit(
                    "모델 클래스: " + ", ".join(f"{k}={v}" for k, v in sorted(model_class_names.items()))
                )
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"영상을 열 수 없습니다: {self.video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = max(1, total_frames)
            frame_index = 0

            while not self._stop_requested:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_index += 1

                result = model.predict(
                    source=frame,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    verbose=False,
                )[0]

                overlay_boxes: list[dict[str, Any]] = []
                boxes = getattr(result, "boxes", None)
                if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                    try:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.zeros(len(xyxy))
                        classes = boxes.cls.cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None else np.zeros(len(xyxy), dtype=np.int32)
                        for i in range(len(xyxy)):
                            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                            score = float(confs[i]) if i < len(confs) else 0.0
                            class_id = int(classes[i]) if i < len(classes) else 0
                            overlay_boxes.append(
                                {
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2,
                                    "class_id": class_id,
                                    "class_name": model_class_names.get(class_id, f"cls_{class_id}"),
                                    "score": score,
                                    "status": "keep",
                                }
                            )
                    except Exception:
                        pass

                self.frame_ready.emit(frame.copy(), overlay_boxes, int(frame_index), int(total_frames))

                if frame_index % 120 == 0 or frame_index >= total_frames:
                    self.log_message.emit(
                        f"test progress: frame {frame_index}/{total_frames}, det={len(overlay_boxes)}"
                    )
            if self._stop_requested:
                self.log_message.emit("test stopped")
            else:
                self.log_message.emit("test finish")
            self.finished.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()

