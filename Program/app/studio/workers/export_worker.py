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

class DatasetExportWorker(QObject):
    """데이터셋 내보내기를 UI 스레드와 분리해 백그라운드로 수행하는 워커입니다."""

    progress = pyqtSignal(str, int, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        video_path: Path,
        dataset_root: Path,
        video_stem: str,
        preview_items: Sequence[PreviewThumbnail],
        frame_annotations: Sequence[FrameAnnotation],
        class_names: Sequence[str],
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        shuffle_seed: int | None = None,
        roi_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        """내보내기에 필요한 입력을 보관하고 실행 준비 상태를 초기화합니다."""
        super().__init__()
        self.video_path = Path(video_path)
        self.dataset_root = Path(dataset_root)
        self.video_stem = str(video_stem)
        self.preview_items = list(preview_items)
        self.frame_annotations = list(frame_annotations)
        self.class_names = [str(name).strip() for name in class_names if str(name).strip()]
        self.split_ratio = tuple(float(v) for v in split_ratio)
        self.shuffle_seed = shuffle_seed
        self.roi_rect = self._sanitize_roi_rect(roi_rect)

    @pyqtSlot()
    def run(self) -> None:
        """백그라운드 내보내기를 실행하고 완료/실패 시그널을 발행합니다."""
        try:
            summary = self._run_export()
            if summary is None:
                raise RuntimeError("keep 상태로 확정된 객체가 없어 내보낼 데이터가 없습니다.")
            self.finished.emit(summary)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _run_export(self) -> ExportRunSummary | None:
        """preview 또는 annotation 정보를 이용해 직접 YOLO 구조로 저장합니다."""
        boxes_by_frame = self._collect_boxes_by_frame()
        frame_indices = sorted(idx for idx, boxes in boxes_by_frame.items() if boxes)
        if not frame_indices:
            return None

        train_ratio, valid_ratio, _ = self._normalize_split_ratio(self.split_ratio)
        split_by_frame = self._build_split_map(frame_indices, train_ratio, valid_ratio, self.shuffle_seed)
        shuffled_for_name = [int(idx) for idx in frame_indices]
        if self.shuffle_seed is None:
            random.shuffle(shuffled_for_name)
        else:
            random.Random(int(self.shuffle_seed)).shuffle(shuffled_for_name)
        save_rank_by_frame = {int(frame_idx): int(rank) for rank, frame_idx in enumerate(shuffled_for_name, start=1)}
        split_dirs = self._prepare_dirs(self.dataset_root)
        class_map: dict[str, int] = {}
        for frame_boxes in boxes_by_frame.values():
            for box in frame_boxes:
                parsed = self._extract_xyxy(box)
                if parsed is None:
                    continue
                _, _, _, _, class_id = parsed
                label = self._label_for_export_class_id(int(class_id))
                if not label:
                    continue
                if label not in class_map:
                    class_map[label] = len(class_map)
        if not class_map:
            return None

        total_boxes = 0
        total_crop_images = 0
        train_images = valid_images = test_images = 0
        train_labels = valid_labels = test_labels = 0

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"영상을 열 수 없습니다: {self.video_path}")

        try:
            first_idx = int(frame_indices[0])
            last_idx = int(frame_indices[-1])
            density = float(len(frame_indices)) / float(max(1, (last_idx - first_idx + 1)))
            sequential = density >= 0.35
            processed = 0
            total = len(frame_indices)

            if sequential:
                needed = set(frame_indices)
                cap.set(cv2.CAP_PROP_POS_FRAMES, first_idx)
                current = first_idx
                while current <= last_idx and needed:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    frame = self._apply_roi_to_frame(frame)
                    if current in needed:
                        split_key = split_by_frame.get(current, "test")
                        lines = self._build_yolo_lines(
                            boxes_by_frame.get(current, []),
                            width=int(frame.shape[1]),
                            height=int(frame.shape[0]),
                            class_map=class_map,
                        )
                        if lines:
                            save_rank = int(save_rank_by_frame.get(int(current), processed))
                            file_name = f"{self.video_stem}_{save_rank:06d}_f{current:06d}.jpg"
                            img_path = split_dirs[split_key]["images"] / file_name
                            if cv2.imwrite(str(img_path), frame):
                                lbl_path = split_dirs[split_key]["labels"] / f"{Path(file_name).stem}.txt"
                                lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                                total_crop_images += self._save_crops_for_frame(
                                    frame_bgr=frame,
                                    file_stem=Path(file_name).stem,
                                    boxes=boxes_by_frame.get(current, []),
                                    class_map=class_map,
                                )
                                total_boxes += len(lines)
                                if split_key == "train":
                                    train_images += 1
                                    train_labels += 1
                                elif split_key == "valid":
                                    valid_images += 1
                                    valid_labels += 1
                                else:
                                    test_images += 1
                                    test_labels += 1
                        needed.discard(current)
                        processed += 1
                        if processed % 60 == 0 or processed == total:
                            self.progress.emit("내보내기 진행", processed, total)
                    current += 1
            else:
                for processed, frame_index in enumerate(frame_indices, start=1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    frame = self._apply_roi_to_frame(frame)
                    split_key = split_by_frame.get(frame_index, "test")
                    lines = self._build_yolo_lines(
                        boxes_by_frame.get(frame_index, []),
                        width=int(frame.shape[1]),
                        height=int(frame.shape[0]),
                        class_map=class_map,
                    )
                    if not lines:
                        continue
                    save_rank = int(save_rank_by_frame.get(int(frame_index), processed))
                    file_name = f"{self.video_stem}_{save_rank:06d}_f{frame_index:06d}.jpg"
                    img_path = split_dirs[split_key]["images"] / file_name
                    if cv2.imwrite(str(img_path), frame):
                        lbl_path = split_dirs[split_key]["labels"] / f"{Path(file_name).stem}.txt"
                        lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                        total_crop_images += self._save_crops_for_frame(
                            frame_bgr=frame,
                            file_stem=Path(file_name).stem,
                            boxes=boxes_by_frame.get(frame_index, []),
                            class_map=class_map,
                        )
                        total_boxes += len(lines)
                        if split_key == "train":
                            train_images += 1
                            train_labels += 1
                        elif split_key == "valid":
                            valid_images += 1
                            valid_labels += 1
                        else:
                            test_images += 1
                            test_labels += 1
                    if processed % 60 == 0 or processed == len(frame_indices):
                        self.progress.emit("내보내기 진행", processed, len(frame_indices))
        finally:
            cap.release()

        total_images = train_images + valid_images + test_images
        if total_images <= 0:
            return None

        self._write_classes_txt(self.dataset_root, class_map)
        self._write_data_yaml(
            self.dataset_root,
            class_map,
            split_counts={
                "train": train_images,
                "valid": valid_images,
                "test": test_images,
            },
        )
        if remote_storage_enabled():
            self.progress.emit("Remote Sync", 0, 1)
            sync_local_tree_to_remote(self.dataset_root)
            crop_root = CROP_SAVE_BASE_DIR / self.dataset_root.name
            if crop_root.is_dir():
                sync_local_tree_to_remote(crop_root)
            self.progress.emit("Remote Sync", 1, 1)
        sorted_items = sorted(((int(idx), str(name)) for name, idx in class_map.items()), key=lambda item: item[0])
        ordered_class_names = [name for _, name in sorted_items]
        return ExportRunSummary(
            dataset_root=self.dataset_root,
            train_images=train_images,
            valid_images=valid_images,
            test_images=test_images,
            train_labels=train_labels,
            valid_labels=valid_labels,
            test_labels=test_labels,
            class_count=len(class_map),
            total_boxes=total_boxes,
            class_names=ordered_class_names,
            crop_images=total_crop_images,
            crop_root=CROP_SAVE_BASE_DIR / self.dataset_root.name,
        )

    def _collect_boxes_by_frame(self) -> dict[int, list[Any]]:
        """preview 우선, 없으면 annotation에서 keep 박스를 프레임별로 수집해 반환합니다."""
        result: dict[int, list[Any]] = {}
        if self.preview_items:
            for item in self.preview_items:
                category = str(item.category).strip().lower()
                if category != "keep":
                    continue
                frame_index = int(item.frame_index)
                if frame_index < 0:
                    continue
                bucket = result.setdefault(frame_index, [])
                for box in item.boxes:
                    status = self._extract_status(box)
                    if status != "keep":
                        continue
                    if self._extract_xyxy(box) is None:
                        continue
                    bucket.append(box)
            return result

        for ann in self.frame_annotations:
            frame_index = int(ann.frame_index)
            if frame_index < 0:
                continue
            bucket = result.setdefault(frame_index, [])
            for box in ann.boxes:
                status = self._extract_status(box)
                if status != "keep":
                    continue
                if self._extract_xyxy(box) is None:
                    continue
                bucket.append(box)
        return result

    def _build_yolo_lines(
        self,
        boxes: Sequence[Any],
        width: int,
        height: int,
        class_map: dict[str, int],
    ) -> list[str]:
        """프레임 박스 목록을 YOLO 문자열 라인 목록으로 변환합니다."""
        if width <= 0 or height <= 0:
            return []
        lines: list[str] = []
        for box in boxes:
            parsed = self._extract_xyxy(box)
            if parsed is None:
                continue
            x1, y1, x2, y2, class_id = parsed
            label = self._label_for_export_class_id(int(class_id))
            class_idx = class_map.get(label)
            if class_idx is None:
                class_idx = len(class_map)
                class_map[label] = class_idx

            cx = ((x1 + x2) * 0.5) / float(width)
            cy = ((y1 + y2) * 0.5) / float(height)
            bw = abs(x2 - x1) / float(width)
            bh = abs(y2 - y1) / float(height)
            cx = max(0.0, min(1.0, float(cx)))
            cy = max(0.0, min(1.0, float(cy)))
            bw = max(0.0, min(1.0, float(bw)))
            bh = max(0.0, min(1.0, float(bh)))
            lines.append(f"{int(class_idx)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return lines

    def _save_crops_for_frame(
        self,
        frame_bgr: np.ndarray,
        file_stem: str,
        boxes: Sequence[Any],
        class_map: dict[str, int],
    ) -> int:
        """프레임 bbox별 crop 이미지를 저장하고 crop 기준 YOLO 라벨(txt)을 생성합니다."""
        if frame_bgr is None:
            return 0
        if not isinstance(frame_bgr, np.ndarray):
            return 0
        h, w = frame_bgr.shape[:2]
        if w <= 0 or h <= 0:
            return 0

        crop_root = CROP_SAVE_BASE_DIR / self.dataset_root.name
        crop_img_dir = crop_root / "images"
        crop_lbl_dir = crop_root / "labels"
        crop_img_dir.mkdir(parents=True, exist_ok=True)
        crop_lbl_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i, box in enumerate(boxes):
            status = self._extract_status(box)
            if status != "keep":
                continue
            parsed = self._extract_xyxy(box)
            if parsed is None:
                continue
            x1, y1, x2, y2, class_id = parsed

            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w))
            y2 = max(0, min(int(y2), h))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame_bgr[y1:y2, x1:x2].copy()
            if crop is None or crop.size <= 0:
                continue

            interp = cv2.INTER_AREA if (crop.shape[1] > 640 or crop.shape[0] > 640) else cv2.INTER_LINEAR
            crop = cv2.resize(crop, (640, 640), interpolation=interp)

            label = self._label_for_export_class_id(int(class_id))
            class_idx = class_map.get(label)
            if class_idx is None:
                class_idx = len(class_map)
                class_map[label] = class_idx

            crop_name = f"{file_stem}_c{i:03d}.jpg"
            crop_path = crop_img_dir / crop_name
            if not cv2.imwrite(str(crop_path), crop):
                continue

            crop_lbl_path = crop_lbl_dir / f"{Path(crop_name).stem}.txt"
            crop_lbl_path.write_text(
                f"{int(class_idx)} 0.500000 0.500000 1.000000 1.000000\n",
                encoding="utf-8",
            )
            saved += 1
        return saved

    def _extract_status(self, box: Any) -> str | None:
        """박스 데이터에서 status(keep/hold/drop)를 추출해 반환합니다."""
        if isinstance(box, dict):
            status = str(box.get("status", "")).strip().lower()
            return status or None
        return None

    def _sanitize_roi_rect(self, roi: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
        """ROI 입력을 정수 사각형(x, y, w, h)으로 정규화해 저장 가능한 형태로 반환합니다."""
        if roi is None:
            return None
        try:
            x, y, w, h = [int(v) for v in roi]
        except Exception:
            return None
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)

    def _normalize_roi_for_shape(
        self,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        """현재 ROI를 프레임 크기에 맞게 보정해 반환하고 전체 프레임이면 None을 반환합니다."""
        if self.roi_rect is None or width <= 0 or height <= 0:
            return None
        x, y, w, h = self.roi_rect
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        if x == 0 and y == 0 and w >= width and h >= height:
            return None
        return (x, y, w, h)

    def _apply_roi_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임에 ROI가 설정되어 있으면 해당 영역으로 크롭한 결과를 반환합니다."""
        if frame is None:
            return frame
        height, width = frame.shape[:2]
        roi = self._normalize_roi_for_shape(width, height)
        if roi is None:
            return frame
        x, y, w, h = roi
        cropped = frame[y : y + h, x : x + w]
        if cropped.size <= 0:
            return frame
        return cropped

    def _extract_xyxy(self, box: Any) -> tuple[int, int, int, int, int] | None:
        """박스 데이터에서 xyxy/class_id를 추출해 정수 좌표로 반환합니다."""
        if isinstance(box, BoxAnnotation):
            return (
                int(round(box.x1)),
                int(round(box.y1)),
                int(round(box.x2)),
                int(round(box.y2)),
                int(box.class_id),
            )
        if isinstance(box, dict):
            class_id = int(box.get("class_id", box.get("cls", 0)))
            if {"x1", "y1", "x2", "y2"}.issubset(box):
                return (
                    int(round(float(box["x1"]))),
                    int(round(float(box["y1"]))),
                    int(round(float(box["x2"]))),
                    int(round(float(box["y2"]))),
                    class_id,
                )
        if isinstance(box, Sequence) and not isinstance(box, (str, bytes, bytearray)):
            values = list(box)
            if len(values) >= 5:
                try:
                    return (
                        int(round(float(values[1]))),
                        int(round(float(values[2]))),
                        int(round(float(values[3]))),
                        int(round(float(values[4]))),
                        int(values[0]),
                    )
                except Exception:
                    return None
        return None

    def _label_for_export_class_id(self, class_id: int) -> str:
        """class_id를 영문 slug 라벨로 변환해 반환합니다."""
        raw = f"class_{class_id}"
        if 0 <= class_id < len(self.class_names):
            raw = str(self.class_names[class_id])
        raw = raw.strip()
        if raw.casefold() in {"객체", "object"}:
            return "object"
        normalized = self._to_english_slug(raw, fallback=f"class_{class_id}")
        if normalized in {"class", "objects"}:
            return f"class_{class_id}"
        return normalized

    def _to_english_slug(self, text: str, fallback: str = "item") -> str:
        """문자열을 영문/숫자/밑줄 slug로 정규화합니다."""
        value = str(text or "").strip()
        if not value:
            return fallback
        decomposed = unicodedata.normalize("NFKD", value)
        ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
        lowered = ascii_text.casefold()
        slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
        slug = re.sub(r"_+", "_", slug)
        if not slug:
            return fallback
        if slug[0].isdigit():
            return f"{fallback}_{slug}"
        return slug

    def _normalize_split_ratio(self, ratio: tuple[float, float, float]) -> tuple[float, float, float]:
        """train/valid/test 비율을 0 이상, 합 1.0으로 정규화합니다."""
        values = [max(0.0, float(ratio[0])), max(0.0, float(ratio[1])), max(0.0, float(ratio[2]))]
        total = sum(values)
        if total <= 0.0:
            return (0.8, 0.1, 0.1)
        return (values[0] / total, values[1] / total, values[2] / total)

    def _build_split_map(
        self,
        frame_indices: Sequence[int],
        train_ratio: float,
        valid_ratio: float,
        seed: int | None,
    ) -> dict[int, str]:
        """프레임 인덱스를 비율에 맞춰 split(train/valid/test)으로 매핑합니다."""
        shuffled = [int(idx) for idx in frame_indices]
        if seed is None:
            random.shuffle(shuffled)
        else:
            random.Random(int(seed)).shuffle(shuffled)
        total = len(shuffled)
        if total <= 0:
            return {}

        if total == 1:
            return {shuffled[0]: "train"}

        train_count = max(1, int(round(total * train_ratio)))
        valid_count = max(1, int(round(total * valid_ratio)))
        if train_count >= total:
            train_count = total - 1
        if train_count + valid_count > total:
            overflow = (train_count + valid_count) - total
            reducible_train = max(0, train_count - 1)
            shift_from_train = min(overflow, reducible_train)
            train_count -= shift_from_train
            overflow -= shift_from_train
            if overflow > 0:
                valid_count = max(1, valid_count - overflow)

        split_map: dict[int, str] = {}
        for idx, frame_idx in enumerate(shuffled):
            if idx < train_count:
                split_map[frame_idx] = "train"
            elif idx < train_count + valid_count:
                split_map[frame_idx] = "valid"
            else:
                split_map[frame_idx] = "test"
        return split_map

    def _prepare_dirs(self, dataset_root: Path) -> dict[str, dict[str, Path]]:
        """내보내기 대상의 train/valid/test 이미지/라벨 폴더를 생성합니다."""
        split_dirs: dict[str, dict[str, Path]] = {}
        for split in ("train", "valid", "test"):
            img_dir = dataset_root / split / "images"
            lbl_dir = dataset_root / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            split_dirs[split] = {"images": img_dir, "labels": lbl_dir}
        return split_dirs

    def _write_classes_txt(self, dataset_root: Path, class_map: Mapping[str, int]) -> None:
        """클래스 인덱스와 이름 매핑을 classes.txt로 기록합니다."""
        sorted_items = sorted(((int(idx), str(name)) for name, idx in class_map.items()), key=lambda item: item[0])
        lines = [f"{idx}: {name}" for idx, name in sorted_items]
        (dataset_root / "classes.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _write_data_yaml(
        self,
        dataset_root: Path,
        class_map: Mapping[str, int],
        split_counts: Mapping[str, int] | None = None,
    ) -> None:
        """YOLO 학습용 data.yaml 파일을 생성합니다."""
        sorted_items = sorted(((int(idx), str(name)) for name, idx in class_map.items()), key=lambda item: item[0])
        names_block = "\n".join([f"  {idx}: {name}" for idx, name in sorted_items])
        train_images = int((split_counts or {}).get("train", 0))
        valid_images = int((split_counts or {}).get("valid", 0))
        test_images = int((split_counts or {}).get("test", 0))
        train_entry = "train/images"
        valid_entry = "valid/images"
        test_entry = "test/images"
        if train_images <= 0:
            if valid_images > 0:
                train_entry = valid_entry
            elif test_images > 0:
                train_entry = test_entry
        if valid_images <= 0:
            if train_images > 0:
                valid_entry = train_entry
            elif test_images > 0:
                valid_entry = test_entry
        yaml_text = (
            f"path: {dataset_root.as_posix()}\n"
            f"train: {train_entry}\n"
            f"val: {valid_entry}\n"
            f"test: {test_entry}\n"
            f"nc: {len(sorted_items)}\n"
            "names:\n"
            f"{names_block}\n"
        )
        (dataset_root / "data.yaml").write_text(yaml_text, encoding="utf-8")

