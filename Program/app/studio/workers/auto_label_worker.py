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

class AutoLabelWorker(QObject):
    """영상 분석 파이프라인을 백그라운드 스레드에서 실행하고 진행/결과 시그널을 발행하는 워커입니다."""
    TEAM_BORDER_MARGIN = 5
    TEAM_REID_RESET_GAP = 30
    # Training-oriented defaults:
    # - Duplicate: stricter only for near-identical frames
    # - Detection score: keep only confident boxes
    DUPLICATE_PHASH_THRESHOLD = 1
    YOLO_KEEP_SCORE_THRESHOLD = 0.70
    YOLO_HOLD_SCORE_THRESHOLD = 0.35

    stage_changed = pyqtSignal(str)
    progress_changed = pyqtSignal(object)
    log_message = pyqtSignal(str)
    preview_ready = pyqtSignal(object, object, int)
    pause_state_changed = pyqtSignal(bool)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        video_path: Path,
        sample_count: int,
        class_names: Sequence[str],
        rgb_bits: int = 24,
        start_frame_index: int = 0,
        prefer_team_pipeline: bool = True,
        realtime_mode: bool = True,
        roi_rect: tuple[int, int, int, int] | None = None,
        team_model_path: Path | None = None,
        yolo_conf: float = 0.10,
        yolo_iou: float = 0.45,
        yolo_imgsz: int = 320,
        track_conf_high: float = 0.20,
        track_validation_frames: int = 5,
        track_iou_threshold: float = 0.50,
        track_size_diff_threshold: float = 0.20,
        track_area_change_limit: float = 0.30,
        track_ratio_change_limit: float = 0.30,
        track_hold_frames: int = 5,
        border_margin: int = 5,
    ) -> None:
        """객체 생성 시 필요한 의존성, 기본값, 내부 상태를 초기화합니다."""
        super().__init__()
        self.video_path = Path(video_path)
        self.sample_count = int(sample_count)
        self.sample_limit: int | None = self.sample_count if self.sample_count > 0 else None
        self.start_frame_index = max(0, int(start_frame_index))
        self.class_names = [name for name in class_names if str(name).strip()]
        # 빈 리스트 = 모델의 모든 클래스를 그대로 사용하는 all-class 모드
        self.rgb_bits = self._sanitize_rgb_bits(rgb_bits)
        self.prefer_team_pipeline = bool(prefer_team_pipeline)
        self.realtime_mode = bool(realtime_mode)
        self.fast_mode = not self.realtime_mode
        self.frame_stride = 1
        self.preview_emit_interval = 1 if self.realtime_mode else 999999
        self.progress_emit_interval = 1 if self.realtime_mode else 6
        self.log_emit_interval = 5 if self.realtime_mode else 24
        self._last_progress_emit_at = 0.0
        self._last_log_emit_at = 0.0
        self._last_preview_emit_at = 0.0
        # Sync mode: do not throttle preview frames (UI follows worker frame-by-frame).
        self.preview_min_interval = 0.0
        self._stop_requested = False
        self._pause_requested = False
        self._is_paused = False
        self._seek_frame_index: int | None = None
        self._preview_item_serial = 0
        self._last_filter_summary: dict[str, Any] = {}
        self.roi_rect = self._sanitize_roi_rect(roi_rect)
        self.team_model_path = Path(team_model_path) if team_model_path is not None else TEAM_MODEL_PATH
        self.yolo_conf = max(0.01, min(1.0, float(yolo_conf)))
        self.yolo_iou = max(0.01, min(1.0, float(yolo_iou)))
        self.yolo_imgsz = max(32, int(yolo_imgsz))
        self.track_conf_high          = max(0.01, min(1.0, float(track_conf_high)))
        self.track_validation_frames  = max(1, int(track_validation_frames))
        self.track_iou_threshold      = max(0.01, min(1.0, float(track_iou_threshold)))
        self.track_size_diff_threshold = max(0.01, min(1.0, float(track_size_diff_threshold)))
        self.track_area_change_limit  = max(0.01, min(1.0, float(track_area_change_limit)))
        self.track_ratio_change_limit = max(0.01, min(1.0, float(track_ratio_change_limit)))
        self.track_hold_frames        = max(1, int(track_hold_frames))
        self.border_margin            = max(0, int(border_margin))
        self.sample_filter_engine: Any | None = None
        if SampleFilterEngine is not None and FilterConfig is not None:
            try:
                self.sample_filter_engine = SampleFilterEngine(FilterConfig())
            except Exception:
                self.sample_filter_engine = None

    @pyqtSlot()
    def run(self) -> None:
        """처리 파이프라인을 실행하고 진행/로그/완료/실패 시그널을 발생시킵니다."""
        cap: cv2.VideoCapture | None = None
        started_at = time.monotonic()
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"영상을 열 수 없습니다: {self.video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise RuntimeError("영상 프레임 수가 올바르지 않습니다.")

            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                fps = 30.0

            self.start_frame_index = max(0, int(self.start_frame_index))
            if self.start_frame_index >= total_frames:
                raise RuntimeError("영상 끝 지점이라 추가로 처리할 프레임이 없습니다.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.start_frame_index))

            annotations: list[FrameAnnotation] = []
            preview_items: list[PreviewThumbnail] = []
            class_names_out: list[str] = list(self.class_names)
            pipeline_meta: dict[str, Any] = {"pipeline": "basic_fallback"}

            use_team, reason = self._can_use_team_pipeline()
            if use_team:
                try:
                    annotations, class_names_out, pipeline_meta, preview_items = self._run_team_pipeline(
                        cap=cap,
                        total_frames=total_frames,
                        fps=fps,
                        started_at=started_at,
                    )
                    self.log_message.emit("팀 자동 라벨링 파이프라인 사용 중")
                except WorkerStoppedError:
                    raise
                except Exception as exc:
                    self.log_message.emit(f"팀 파이프라인 예외: {exc}")
                    if not self._basic_pipeline_can_respect_class_filter():
                        raise RuntimeError(
                            "팀 파이프라인 실패로 처리를 중단했습니다. "
                            f"입력한 클래스만 저장하려면 팀 파이프라인이 필요합니다. 원인: {exc}"
                        ) from exc
                    self.log_message.emit("팀 파이프라인 실패, 기본 파이프라인으로 전환")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.start_frame_index))
                    annotations, class_names_out, pipeline_meta, preview_items = self._run_basic_pipeline(
                        cap=cap,
                        total_frames=total_frames,
                        fps=fps,
                        started_at=started_at,
                    )
            else:
                if reason:
                    self.log_message.emit(f"팀 파이프라인 사용 불가: {reason}")
                if not self._basic_pipeline_can_respect_class_filter():
                    raise RuntimeError(
                        "팀 파이프라인을 사용할 수 없어 처리를 중단했습니다. "
                        "입력한 클래스만 저장하려면 팀 파이프라인이 필요합니다."
                    )
                annotations, class_names_out, pipeline_meta, preview_items = self._run_basic_pipeline(
                    cap=cap,
                    total_frames=total_frames,
                    fps=fps,
                    started_at=started_at,
                )

            preview_items = self._postprocess_preview_items(
                cap=cap,
                preview_items=preview_items,
                annotations=annotations,
            )

            if not annotations:
                raise RuntimeError("라벨링 가능한 프레임이 없습니다.")

            self.log_message.emit(f"라벨링 완료: 주석 프레임 {len(annotations)}개")
            self.stage_changed.emit("완료")
            output = WorkerOutput(
                frame_annotations=annotations,
                preview_items=preview_items,
                class_names=list(class_names_out),
                video_info={
                    "video_path": str(self.video_path),
                    "fps": fps,
                    "frame_count": total_frames,
                    "start_frame_index": self.start_frame_index,
                    "sampled_frame_count": len(annotations),
                },
                run_config={
                    "sample_count_requested": self.sample_count,
                    "sample_count_used": len(annotations),
                    "rgb_bits": self.rgb_bits,
                    "start_frame_index": self.start_frame_index,
                    "roi_rect": list(self.roi_rect) if self.roi_rect is not None else None,
                    "team_model_path": str(self.team_model_path),
                    "filter_summary": dict(self._last_filter_summary),
                    "stages": ["필터링", "객체 검출", "객체 추적", "라벨링"],
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    **pipeline_meta,
                },
            )
            self.finished.emit(output)
        except WorkerStoppedError:
            self.failed.emit("사용자 요청으로 처리가 중지되었습니다.")
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if self._is_paused:
                self._is_paused = False
                self.pause_state_changed.emit(False)
            if cap is not None:
                cap.release()

    def _postprocess_preview_items(
        self,
        cap: cv2.VideoCapture,
        preview_items: Sequence[PreviewThumbnail],
        annotations: Sequence[FrameAnnotation],
    ) -> list[PreviewThumbnail]:
        """오토라벨링 결과를 track_id/프레임 순으로 정렬하고 ID를 재부여해 반환합니다."""
        self._last_filter_summary = {}
        if not preview_items:
            return []
        return postprocess_preview_items(
            preview_items,
            annotations=annotations,
            log_callback=self.log_message.emit,
        )

        sorted_items = self._sort_preview_items_by_track_and_frame(preview_items)
        # 롤백 모드: 내부 자동 품질 필터(중복/블러/노출/점수)는 실행하지 않습니다.
        processed_items = self._dedupe_keep_preview_items(sorted_items)
        self._renumber_preview_items_by_track(processed_items, annotations)
        self._sync_annotation_status_from_preview_items(annotations, processed_items)

        keep_count = sum(1 for item in processed_items if str(item.category).strip().lower() == "keep")
        hold_count = sum(1 for item in processed_items if str(item.category).strip().lower() == "hold")
        drop_count = sum(1 for item in processed_items if str(item.category).strip().lower() == "drop")
        self.log_message.emit(
            f"후처리 결과: keep={keep_count}, hold={hold_count}, drop={drop_count}"
        )
        return processed_items

    def _dedupe_keep_preview_items(
        self,
        preview_items: Sequence[PreviewThumbnail],
    ) -> list[PreviewThumbnail]:
        """keep 카테고리의 중복 샘플(동일 track/frame 또는 연속 동일 박스)을 제거합니다."""
        deduped: list[PreviewThumbnail] = []
        key_to_index: dict[tuple[int, int], int] = {}
        last_keep_by_track: dict[int, tuple[int, tuple[float, float, float, float]]] = {}
        dropped = 0

        def _first_box_dict(item: PreviewThumbnail) -> dict[str, Any] | None:
            if not item.boxes:
                return None
            first = item.boxes[0]
            if not isinstance(first, dict):
                return None
            return first

        def _box_xyxy(box: dict[str, Any]) -> tuple[float, float, float, float] | None:
            try:
                x1 = float(box.get("x1", 0.0))
                y1 = float(box.get("y1", 0.0))
                x2 = float(box.get("x2", 0.0))
                y2 = float(box.get("y2", 0.0))
            except Exception:
                return None
            if x2 <= x1 or y2 <= y1:
                return None
            return (x1, y1, x2, y2)

        def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0.0:
                return 0.0
            aa = max(0.0, (ax2 - ax1) * (ay2 - ay1))
            bb = max(0.0, (bx2 - bx1) * (by2 - by1))
            denom = aa + bb - inter
            if denom <= 0.0:
                return 0.0
            return inter / denom

        for item in preview_items:
            category = str(item.category).strip().lower()
            if category != "keep":
                deduped.append(item)
                continue

            box = _first_box_dict(item)
            if box is None:
                deduped.append(item)
                continue
            raw_track = box.get("track_id")
            if raw_track is None:
                deduped.append(item)
                continue
            try:
                track_id = int(raw_track)
            except Exception:
                deduped.append(item)
                continue

            frame_idx = int(item.frame_index)
            xyxy = _box_xyxy(box)
            if xyxy is None:
                deduped.append(item)
                continue

            # 1) 동일 track/frame 중복 제거 (점수 높은 항목만 유지)
            key = (track_id, frame_idx)
            if key in key_to_index:
                prev_idx = key_to_index[key]
                prev_item = deduped[prev_idx]
                prev_box = _first_box_dict(prev_item)
                try:
                    prev_score = float(prev_box.get("score", 0.0)) if prev_box is not None else 0.0
                except Exception:
                    prev_score = 0.0
                try:
                    cur_score = float(box.get("score", 0.0))
                except Exception:
                    cur_score = 0.0
                if cur_score > prev_score:
                    deduped[prev_idx] = item
                dropped += 1
                continue

            # 2) 같은 track의 직전 keep과 사실상 동일 박스(연속 프레임)면 중복으로 제거
            prev_track = last_keep_by_track.get(track_id)
            if prev_track is not None:
                prev_frame_idx, prev_xyxy = prev_track
                if abs(frame_idx - prev_frame_idx) <= 1 and _iou_xyxy(prev_xyxy, xyxy) >= 0.995:
                    dropped += 1
                    continue

            key_to_index[key] = len(deduped)
            last_keep_by_track[track_id] = (frame_idx, xyxy)
            deduped.append(item)

        if dropped > 0:
            self.log_message.emit(f"중복 제거(keep): {dropped}개")
        return deduped

    def _renumber_preview_items_by_track(
        self,
        preview_items: Sequence[PreviewThumbnail],
        annotations: Sequence[FrameAnnotation],
    ) -> None:
        """객체(track_id)별 프레임 순서로 preview item ID를 id{track}_{seq} 형태로 재부여합니다."""
        remap: dict[str, str] = {}
        track_seq: dict[int, int] = {}
        unknown_seq = 0

        for item in preview_items:
            old_item_id = str(item.item_id).strip()
            track_id: int | None = None
            if item.boxes and isinstance(item.boxes[0], dict):
                raw_track = item.boxes[0].get("track_id")
                if raw_track is not None:
                    try:
                        track_id = int(raw_track)
                    except Exception:
                        track_id = None

            if track_id is None:
                unknown_seq += 1
                new_item_id = f"id_unknown_{unknown_seq:03d}"
            else:
                next_seq = int(track_seq.get(track_id, 0)) + 1
                track_seq[track_id] = next_seq
                new_item_id = f"id{track_id}_{next_seq:03d}"

            item.item_id = new_item_id
            for box in item.boxes:
                if isinstance(box, dict):
                    box["preview_item_id"] = new_item_id
            if old_item_id:
                remap[old_item_id] = new_item_id

        if not remap:
            return

        for ann in annotations:
            for box in ann.boxes:
                if not isinstance(box, dict):
                    continue
                prev = str(box.get("preview_item_id", "")).strip()
                if not prev:
                    continue
                mapped = remap.get(prev)
                if mapped is not None:
                    box["preview_item_id"] = mapped

    def _sort_preview_items_by_track_and_frame(
        self,
        preview_items: Sequence[PreviewThumbnail],
    ) -> list[PreviewThumbnail]:
        """preview 항목을 track_id 우선, frame_index 차순으로 정렬합니다."""
        cloned: list[PreviewThumbnail] = []
        for item in preview_items:
            boxes: list[Any] = []
            for box in item.boxes:
                if isinstance(box, dict):
                    boxes.append(dict(box))
                else:
                    boxes.append(box)
            cloned.append(
                PreviewThumbnail(
                    frame_index=int(item.frame_index),
                    image=None,
                    boxes=boxes,
                    category=str(item.category).strip().lower(),
                    item_id=str(item.item_id).strip(),
                    image_path=None,
                    thumb_path=None,
                    manifest_path=None,
                )
            )

        def _item_sort_key(item: PreviewThumbnail) -> tuple[int, int, str]:
            track_key = 2_147_483_647
            if item.boxes and isinstance(item.boxes[0], dict):
                raw_track = item.boxes[0].get("track_id")
                if raw_track is not None:
                    try:
                        track_key = int(raw_track)
                    except Exception:
                        track_key = 2_147_483_647
            return (track_key, int(item.frame_index), str(item.item_id))

        cloned.sort(key=_item_sort_key)
        return cloned

    def _crop_box_region(self, frame: np.ndarray, box: dict[str, Any]) -> np.ndarray | None:
        """프레임에서 박스 영역을 안전하게 crop 하여 반환합니다."""
        if frame is None or frame.size <= 0:
            return None
        try:
            x1 = int(round(float(box.get("x1", 0.0))))
            y1 = int(round(float(box.get("y1", 0.0))))
            x2 = int(round(float(box.get("x2", 0.0))))
            y2 = int(round(float(box.get("y2", 0.0))))
        except Exception:
            return None
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size <= 0:
            return None
        return crop

    def _evaluate_prefilter_candidate(
        self,
        frame_idx: int,
        fps: float,
        track_id: int,
        class_id: int,
        bbox: tuple[float, float, float, float],
        crop: np.ndarray,
    ) -> tuple[bool, str, Any | None]:
        """스트리밍 품질/중복 엔진으로 사전 필터를 평가합니다."""
        if self.sample_filter_engine is None or SampleCandidate is None:
            return True, "PASS", None
        try:
            candidate = SampleCandidate(
                frame_idx=int(frame_idx),
                timestamp_ms=(float(frame_idx) / max(1e-6, float(fps))) * 1000.0,
                track_id=str(track_id),
                bbox=bbox,
                crop_image=crop,
                sample_id=f"trk{int(track_id)}_frm{int(frame_idx):06d}_cls{int(class_id)}",
                meta={"class_id": int(class_id)},
            )
            result = self.sample_filter_engine.evaluate(candidate)
            if bool(getattr(result, "passed", False)):
                return True, "PASS", candidate
            reason = str(getattr(result, "reason", "DROP_INVALID_SAMPLE")).strip().upper() or "DROP_INVALID_SAMPLE"
            return False, reason, candidate
        except Exception:
            return True, "PASS", None

    def _register_prefilter_final_keep(self, candidate: Any | None) -> None:
        """최종 keep 확정 샘플만 사전 필터 엔진의 기준 상태로 등록합니다."""
        if self.sample_filter_engine is None or candidate is None:
            return
        try:
            self.sample_filter_engine.on_final_keep(candidate)
        except Exception:
            pass

    def _merge_quality_stage(
        self,
        original: str,
        blur_stage: str,
        exposure_stage: str,
        fallback_stage: str = "",
    ) -> str:
        """원래 상태와 품질 필터 결과를 병합해 최종 keep/hold/drop을 반환합니다."""
        stages = [
            str(original).strip().lower(),
            str(blur_stage).strip().lower(),
            str(exposure_stage).strip().lower(),
        ]
        if fallback_stage:
            stages.append(str(fallback_stage).strip().lower())
        drop_count = sum(1 for stage in stages if stage == "drop")
        if drop_count >= 2:
            return "drop"
        if drop_count == 1:
            return "hold"
        if any(stage == "hold" for stage in stages):
            return "hold"
        return "keep"

    def _set_preview_item_category(
        self,
        item: PreviewThumbnail,
        category: str,
        drop_reason: str | None = None,
    ) -> None:
        """preview 항목과 내부 박스 status를 동일 카테고리로 동기화합니다."""
        normalized = str(category).strip().lower()
        if normalized not in {"keep", "hold", "drop"}:
            normalized = "hold"
        item.category = normalized
        for box in item.boxes:
            if isinstance(box, dict):
                box["status"] = normalized
                if normalized == "drop":
                    reason = str(drop_reason or "").strip().lower()
                    if reason:
                        box["drop_reason"] = reason
                    else:
                        box.pop("drop_reason", None)
                else:
                    box.pop("drop_reason", None)

    def _sync_annotation_status_from_preview_items(
        self,
        annotations: Sequence[FrameAnnotation],
        preview_items: Sequence[PreviewThumbnail],
    ) -> None:
        """preview item의 최종 카테고리를 같은 preview_item_id를 가진 annotation 박스 상태에 반영합니다."""
        status_by_item_id: dict[str, str] = {}
        for item in preview_items:
            item_id = str(item.item_id).strip()
            category = str(item.category).strip().lower()
            if not item_id or category not in {"keep", "hold", "drop"}:
                continue
            status_by_item_id[item_id] = category

        if not status_by_item_id:
            return

        for ann in annotations:
            for box in ann.boxes:
                if not isinstance(box, dict):
                    continue
                item_id = str(box.get("preview_item_id", "")).strip()
                if not item_id:
                    continue
                updated = status_by_item_id.get(item_id)
                if updated is not None:
                    box["status"] = updated

    def request_stop(self) -> None:
        """중단 요청 플래그를 설정해 워커가 안전한 체크 지점에서 작업을 멈추도록 합니다."""
        self._pause_requested = False
        self._seek_frame_index = None
        self._stop_requested = True

    def request_pause(self) -> None:
        """일시정지 요청 플래그를 설정합니다."""
        if self._stop_requested:
            return
        self._pause_requested = True

    def request_resume(self) -> None:
        """일시정지 상태를 해제해 작업을 재개합니다."""
        self._pause_requested = False

    def request_seek(self, frame_index: int) -> None:
        """다음 루프 체크 시 지정 프레임으로 점프하도록 요청합니다."""
        if self._stop_requested:
            return
        try:
            target = int(frame_index)
        except Exception:
            return
        self._seek_frame_index = max(0, target)

    def _consume_seek_request(self, total_frames: int) -> int | None:
        """대기 중인 seek 요청을 1회 소비해 유효한 프레임 인덱스로 반환합니다."""
        requested = self._seek_frame_index
        self._seek_frame_index = None
        if requested is None:
            return None
        if total_frames <= 0:
            return max(0, int(requested))
        return max(0, min(int(total_frames) - 1, int(requested)))

    def _check_stop(self) -> None:
        """중단 요청 플래그를 확인하고 요청된 경우 WorkerStoppedError를 발생시켜 상위 루프를 안전 종료합니다."""
        if self._stop_requested:
            raise WorkerStoppedError()
        if self._pause_requested:
            if not self._is_paused:
                self._is_paused = True
                self.pause_state_changed.emit(True)
            while self._pause_requested and (not self._stop_requested):
                time.sleep(0.05)
            if self._stop_requested:
                raise WorkerStoppedError()
            if self._is_paused:
                self._is_paused = False
                self.pause_state_changed.emit(False)

    def _assign_ids_by_iou(
        self,
        boxes: Any,
        track_history: dict[int, Any],
        frame_idx: int,
        iou_threshold: float = 0.3,
        max_gap: int = 5,
    ) -> list[int]:
        """boxes.id가 None일 때 이전 프레임 tracker와 IOU 매칭으로 연속 track_id를 부여해 반환합니다."""
        next_id = max(track_history.keys(), default=0) + 1
        assigned: list[int] = []
        used: set[int] = set()

        for bi, box in enumerate(boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            best_id: int | None = None
            best_iou = iou_threshold

            for tid, tracker in track_history.items():
                if tid in used:
                    continue
                last_seen = int(getattr(tracker, "last_seen_frame", -999))
                if frame_idx - last_seen > max_gap:
                    continue
                buf = getattr(tracker, "buffer", [])
                last_bbox = None
                if buf and isinstance(buf[-1], dict):
                    last_bbox = buf[-1].get("bbox")
                if last_bbox is None:
                    continue
                ix1 = max(x1, float(last_bbox[0]))
                iy1 = max(y1, float(last_bbox[1]))
                ix2 = min(x2, float(last_bbox[2]))
                iy2 = min(y2, float(last_bbox[3]))
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter <= 0.0:
                    continue
                a1 = (x2 - x1) * (y2 - y1)
                a2 = (float(last_bbox[2]) - float(last_bbox[0])) * (float(last_bbox[3]) - float(last_bbox[1]))
                iou = inter / max(a1 + a2 - inter, 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is None:
                best_id = next_id
                next_id += 1
            used.add(best_id)
            assigned.append(best_id)

        return assigned

    def _can_use_team_pipeline(self) -> tuple[bool, str]:
        """모델 파일/라이브러리/옵션 상태를 점검해 팀 파이프라인 사용 가능 여부와 사유를 반환합니다."""
        if not self.prefer_team_pipeline:
            return False, "작업자 옵션에서 비활성화됨"
        if UltralyticsYOLO is None:
            return False, "추론 라이브러리를 불러올 수 없습니다."
        if TeamTrackObject is None:
            return False, "추적 모듈을 불러올 수 없습니다."
        if not self.team_model_path.is_file():
            return False, f"모델 파일 없음 ({self.team_model_path})"
        return True, ""

    def _basic_pipeline_can_respect_class_filter(self) -> bool:
        """기본 파이프라인이 현재 클래스 필터 조건을 만족하며 동작 가능한지 여부를 반환합니다."""
        allowed = {str(name).strip().casefold() for name in self.class_names if str(name).strip()}
        if not allowed:
            return True
        # Basic fallback is class-agnostic; allow it only for generic "object" style labeling.
        return allowed.issubset({"object", "객체"})

    def _is_generic_object_class_mode(self) -> bool:
        """클래스 입력이 일반 객체(object/객체) 모드인지 검사해 불리언으로 반환합니다."""
        allowed = {str(name).strip().casefold() for name in self.class_names if str(name).strip()}
        if not allowed:
            return True
        return allowed.issubset({"object", "객체"})

    def _run_basic_pipeline(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
        fps: float,
        started_at: float,
    ) -> tuple[list[FrameAnnotation], list[str], dict[str, Any], list[PreviewThumbnail]]:
        """내부 파이프라인 단계를 실행하고 처리 결과를 반환합니다."""
        available_frames = max(0, int(total_frames) - int(self.start_frame_index))
        sample_count = available_frames if self.sample_limit is None else min(int(self.sample_limit), available_frames)
        if sample_count <= 0:
            return [], list(self.class_names), {"pipeline": "basic_fallback"}, []
        indices = self._sample_frame_indices(total_frames, sample_count, start_frame=self.start_frame_index)
        self.stage_changed.emit("필터링")
        self.log_message.emit(f"필터링 완료: 선택 프레임 {len(indices)}개")

        total_units = max(1, len(indices) * 3)
        processed_units = 0

        self.stage_changed.emit("객체 검출")
        detected_items: list[dict[str, Any]] = []
        detect_step = 0
        detect_cursor = 0
        while detect_cursor < len(indices):
            self._check_stop()
            seek_target = self._consume_seek_request(total_frames)
            if seek_target is not None:
                while detect_cursor < len(indices) and int(indices[detect_cursor]) < int(seek_target):
                    detect_cursor += 1
                if detect_cursor >= len(indices):
                    break
            frame_idx = int(indices[detect_cursor])
            detect_cursor += 1
            frame = self._read_frame(cap, frame_idx)
            if frame is None:
                self.log_message.emit(f"프레임 {frame_idx} 건너뜀: 읽기 실패")
                continue
            frame = self._apply_rgb_bits(frame)

            boxes = self._detect_boxes(frame)
            if self.sample_filter_engine is not None and boxes:
                filtered_boxes: list[BoxAnnotation] = []
                for box in boxes:
                    crop = self._crop_box_region(
                        frame,
                        {
                            "x1": float(box.x1),
                            "y1": float(box.y1),
                            "x2": float(box.x2),
                            "y2": float(box.y2),
                        },
                    )
                    if crop is None:
                        continue
                    passed, _reason_code, _candidate = self._evaluate_prefilter_candidate(
                        frame_idx=frame_idx,
                        fps=fps,
                        track_id=int(box.track_id) if box.track_id is not None else int(box.class_id),
                        class_id=int(box.class_id),
                        bbox=(float(box.x1), float(box.y1), float(box.x2), float(box.y2)),
                        crop=crop,
                    )
                    if passed:
                        filtered_boxes.append(box)
                boxes = filtered_boxes

            detected_items.append(
                {
                    "frame_index": int(frame_idx),
                    "boxes": boxes,
                }
            )
            detect_step += 1
            processed_units += 1
            self._apply_processing_pacing(processed_frames=detect_step, fps=fps, started_at=started_at)
            if self._should_emit_preview(detect_step, len(indices)):
                self.preview_ready.emit(frame.copy(), boxes, int(frame_idx))
            if self._should_emit_progress(detect_step, len(indices)):
                self.progress_changed.emit(
                    self._build_progress(
                        stage="객체 검출",
                        processed_units=processed_units,
                        total_units=total_units,
                        current_frame=frame_idx,
                        total_frames=total_frames,
                        remaining_items=max(0, len(indices) - detect_step),
                        started_at=started_at,
                    )
                )
            if self._should_emit_step_log(detect_step, len(indices)):
                self.log_message.emit(f"객체 검출 진행: {detect_step}/{len(indices)}")

        self.stage_changed.emit("객체 추적")
        tracked_items = self._track_sequence(detected_items)
        for idx, item in enumerate(tracked_items, start=1):
            self._check_stop()
            processed_units += 1
            frame_idx = int(item["frame_index"])
            preview_frame = self._read_frame(cap, frame_idx)
            if preview_frame is not None:
                preview_frame = self._apply_rgb_bits(preview_frame)
                if self._should_emit_preview(idx, len(tracked_items)):
                    self.preview_ready.emit(preview_frame.copy(), item["boxes"], int(frame_idx))
            if self._should_emit_progress(idx, len(tracked_items)):
                self.progress_changed.emit(
                    self._build_progress(
                        stage="객체 추적",
                        processed_units=processed_units,
                        total_units=total_units,
                        current_frame=frame_idx,
                        total_frames=total_frames,
                        remaining_items=len(tracked_items) - idx,
                        started_at=started_at,
                    )
                )
            if self._should_emit_step_log(idx, len(tracked_items)):
                self.log_message.emit(f"객체 추적 진행: {idx}/{len(tracked_items)}")

        self.stage_changed.emit("라벨링")
        annotations: list[FrameAnnotation] = []
        for idx, item in enumerate(tracked_items, start=1):
            self._check_stop()
            frame_idx = int(item["frame_index"])
            image_name = f"{self.video_path.stem}_{frame_idx:06d}.jpg"
            annotation = FrameAnnotation(
                frame_index=frame_idx,
                image=None,
                image_name=image_name,
                split="train",
                boxes=item["boxes"],
                timestamp_sec=frame_idx / fps,
            )
            annotations.append(annotation)
            processed_units += 1
            if self._should_emit_progress(idx, len(tracked_items)):
                self.progress_changed.emit(
                    self._build_progress(
                        stage="라벨링",
                        processed_units=processed_units,
                        total_units=total_units,
                        current_frame=frame_idx,
                        total_frames=total_frames,
                        remaining_items=len(tracked_items) - idx,
                        started_at=started_at,
                    )
                )

        preview_items = self._build_preview_items_from_annotations(annotations, default_category="keep")
        return annotations, list(self.class_names), {"pipeline": "basic_fallback"}, preview_items

    def _run_team_pipeline(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
        fps: float,
        started_at: float,
    ) -> tuple[list[FrameAnnotation], list[str], dict[str, Any], list[PreviewThumbnail]]:
        """내부 파이프라인 단계를 실행하고 처리 결과를 반환합니다."""
        if UltralyticsYOLO is None or TeamTrackObject is None:
            raise RuntimeError("팀 파이프라인 의존성을 사용할 수 없습니다.")

        self.stage_changed.emit("필터링")
        self.log_message.emit("팀 자동 라벨링 모델 초기화 중...")
        model = UltralyticsYOLO(str(self.team_model_path))

        # 모델의 실제 클래스명을 로그로 출력해 사용자가 확인할 수 있도록 합니다.
        _model_names_dict = getattr(model, "names", {})
        _model_cls_list = list(_model_names_dict.values()) if isinstance(_model_names_dict, dict) else []
        if _model_cls_list:
            self.log_message.emit(f"모델 클래스: {', '.join(str(n) for n in _model_cls_list)}")

        self.stage_changed.emit("객체 검출")
        track_history: dict[int, Any] = {}
        frame_annotations: dict[int, FrameAnnotation] = {}
        class_name_to_id: dict[str, int] = {name: idx for idx, name in enumerate(self.class_names)}

        def _norm_cls(name: str) -> str:
            """클래스명 비교용 정규화: 소문자 + 언더스코어→공백 치환."""
            return str(name).strip().casefold().replace("_", " ")

        allowed_class_name_by_key: dict[str, str] = {
            _norm_cls(name): str(name).strip()
            for name in self.class_names
            if str(name).strip()
        }
        # 클래스명이 비어있으면 모델의 모든 클래스를 그대로 사용하는 all-class 모드
        all_class_mode = not self.class_names
        generic_class_mode = self._is_generic_object_class_mode() and not all_class_mode
        generic_class_name = self.class_names[0] if self.class_names else "객체"
        filtered_out_by_class_count = 0
        keep_saved_count = 0
        hold_saved_count = 0
        drop_filtered_count = 0
        prefilter_drop_by_reason: dict[str, int] = {}
        drop_preview_items: list[PreviewThumbnail] = []
        tracking_announced = False
        labeling_announced = False

        processed_frames_in_window = 0
        current_frame_idx = int(self.start_frame_index)
        frame_span = max(1, total_frames - int(self.start_frame_index))
        total_units = max(1, frame_span)

        while current_frame_idx < total_frames:
            self._check_stop()
            seek_target = self._consume_seek_request(total_frames)
            if seek_target is not None:
                current_frame_idx = int(seek_target)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_idx = int(current_frame_idx)
            frame = self._apply_roi_to_frame(frame)
            frame = self._apply_rgb_bits(frame)

            results = model.track(
                frame,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
                conf=self.yolo_conf,
                iou=self.yolo_iou,
                imgsz=self.yolo_imgsz,
            )
            preview_boxes: list[Any] = []

            has_tracking_items = bool(results) and results[0].boxes is not None and len(results[0].boxes.xyxy) > 0
            if has_tracking_items:
                if not tracking_announced:
                    self.stage_changed.emit("객체 추적")
                    tracking_announced = True

                boxes = results[0].boxes.xyxy.cpu().numpy()
                # boxes.id가 None이면(tracker ID 미할당) IOU 기반으로 이전 tracker와 매칭해 연속 ID 부여
                if results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                else:
                    ids = self._assign_ids_by_iou(boxes, track_history, frame_idx)
                confs = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                masks = results[0].masks
                masks_xy = masks.xy if masks is not None else []
                height, width = frame.shape[:2]
                use_polygon = True

                for i, track_id in enumerate(ids):
                    bbox = boxes[i]
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    score = float(confs[i])
                    cls_name = self._resolve_model_class_name(model, int(classes[i]))
                    if all_class_mode:
                        # 클래스명 미입력: 모델의 실제 클래스명을 그대로 사용
                        allowed_name = cls_name
                    elif generic_class_mode:
                        allowed_name = generic_class_name
                    else:
                        allowed_name = allowed_class_name_by_key.get(_norm_cls(cls_name))
                        if allowed_name is None:
                            filtered_out_by_class_count += 1
                            continue
                    raw_mask = masks_xy[i] if i < len(masks_xy) else None
                    polygon_points = (
                        self._mask_to_polygon_points(raw_mask, bbox, width=width, height=height)
                        if use_polygon
                        else None
                    )

                    if (
                        x1 < self.border_margin
                        or y1 < self.border_margin
                        or x2 > width - self.border_margin
                        or y2 > height - self.border_margin
                    ):
                        continue

                    # all_class_mode에서는 클래스명이 동적으로 추가될 수 있음
                    if allowed_name not in class_name_to_id:
                        class_name_to_id[allowed_name] = len(class_name_to_id)
                    class_id = class_name_to_id[allowed_name]
                    prefilter_candidate: Any | None = None
                    crop = self._crop_box_region(
                        frame,
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                    )
                    if crop is not None:
                        passed, reason_code, prefilter_candidate = self._evaluate_prefilter_candidate(
                            frame_idx=frame_idx,
                            fps=fps,
                            track_id=int(track_id),
                            class_id=int(class_id),
                            bbox=(x1, y1, x2, y2),
                            crop=crop,
                        )
                        if not passed:
                            drop_filtered_count += 1
                            reason_label = str(reason_code).strip().upper()
                            reason_suffix = reason_label.replace("DROP_", "").lower()
                            prefilter_drop_by_reason[reason_label] = int(
                                prefilter_drop_by_reason.get(reason_label, 0)
                            ) + 1
                            preview_boxes.append(
                                self._build_box_payload(
                                    x1=x1,
                                    y1=y1,
                                    x2=x2,
                                    y2=y2,
                                    class_id=int(class_id),
                                    score=score,
                                    track_id=int(track_id),
                                    status="drop",
                                    polygon_points=polygon_points,
                                    drop_reason=reason_suffix,
                                )
                            )
                            self._append_preview_item(
                                preview_items=drop_preview_items,
                                frame_index=frame_idx,
                                frame_shape=(int(height), int(width)),
                                bbox=bbox,
                                class_id=int(class_id),
                                score=score,
                                track_id=int(track_id),
                                category="drop",
                                polygon_points=polygon_points,
                                drop_reason=reason_suffix,
                            )
                            continue

                    tracker = track_history.get(int(track_id))
                    if tracker is None:
                        tracker = TeamTrackObject(
                            int(track_id), frame_idx,
                            conf_low=self.yolo_conf,
                            conf_high=self.track_conf_high,
                            validation_frames=self.track_validation_frames,
                            iou_threshold=self.track_iou_threshold,
                            size_diff_threshold=self.track_size_diff_threshold,
                            area_change_limit=self.track_area_change_limit,
                            ratio_change_limit=self.track_ratio_change_limit,
                            hold_frames=self.track_hold_frames,
                        )
                        track_history[int(track_id)] = tracker

                    if frame_idx - int(getattr(tracker, "last_seen_frame", frame_idx)) > self.TEAM_REID_RESET_GAP:
                        tracker.state = "CANDIDATE"
                        tracker.buffer = []
                        tracker.status_msg = "재식별 초기화"

                    action, _reason = tracker.process(bbox, score, raw_mask, frame_idx, (height, width))
                    # Keep prefilter candidate on the buffered frame item so that
                    # buffer flush can deduplicate and register final-keep in order.
                    if prefilter_candidate is not None:
                        try:
                            buffer_ref = getattr(tracker, "buffer", None)
                            if isinstance(buffer_ref, list) and buffer_ref:
                                last_item = buffer_ref[-1]
                                if isinstance(last_item, dict) and int(last_item.get("frame", -1)) == int(frame_idx):
                                    last_item["prefilter_candidate"] = prefilter_candidate
                        except Exception:
                            pass
                    preview_boxes.append(
                        self._build_box_payload(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            class_id=int(class_id),
                            score=score,
                            track_id=int(track_id),
                            status=self._status_from_tracker_action(action),
                            polygon_points=polygon_points,
                        )
                    )
                    if action in {"DELETE_TRACK", "DROP"}:
                        drop_filtered_count += 1
                        self._append_preview_item(
                            preview_items=drop_preview_items,
                            frame_index=frame_idx,
                            frame_shape=(int(height), int(width)),
                            bbox=bbox,
                            class_id=int(class_id),
                            score=score,
                            track_id=int(track_id),
                            category="drop",
                            polygon_points=polygon_points,
                        )
                        track_history.pop(int(track_id), None)
                        continue
                    if action in {"SKIP", "BUFFERING"}:
                        continue

                    if not labeling_announced:
                        self.stage_changed.emit("라벨링")
                        labeling_announced = True

                    if action in {"SAVE", "KEEP", "HOLD"}:
                        status = "hold" if action == "HOLD" else "keep"
                        if status == "keep":
                            self._register_prefilter_final_keep(prefilter_candidate)
                        self._append_annotation(
                            frame_annotations=frame_annotations,
                            frame_index=frame_idx,
                            frame_shape=(int(height), int(width)),
                            bbox=bbox,
                            class_id=int(class_id),
                            score=score,
                            track_id=int(track_id),
                            fps=fps,
                            status=status,
                            polygon_points=polygon_points,
                        )
                        if action == "HOLD":
                            hold_saved_count += 1
                        else:
                            keep_saved_count += 1
                    elif action in {"SAVE_BUFFER", "KEEP_BUFFER", "HOLD_BUFFER"}:
                        status = "hold" if action == "HOLD_BUFFER" else "keep"
                        buffered_by_frame: dict[int, dict[str, Any]] = {}
                        for raw_item in list(getattr(tracker, "buffer", [])):
                            if not isinstance(raw_item, dict):
                                continue
                            try:
                                bf_idx = int(raw_item.get("frame", frame_idx))
                            except Exception:
                                bf_idx = int(frame_idx)
                            prev = buffered_by_frame.get(bf_idx)
                            if prev is None:
                                buffered_by_frame[bf_idx] = raw_item
                                continue
                            try:
                                prev_score = float(prev.get("score", -1.0))
                            except Exception:
                                prev_score = -1.0
                            try:
                                new_score = float(raw_item.get("score", -1.0))
                            except Exception:
                                new_score = -1.0
                            if new_score >= prev_score:
                                buffered_by_frame[bf_idx] = raw_item

                        current_frame_in_buffer = int(frame_idx) in buffered_by_frame
                        for bf_idx in sorted(buffered_by_frame.keys()):
                            item = buffered_by_frame[bf_idx]
                            candidate_for_item = item.get("prefilter_candidate")
                            if status == "keep" and candidate_for_item is not None and self.sample_filter_engine is not None:
                                try:
                                    recheck = self.sample_filter_engine.evaluate(candidate_for_item)
                                    if not bool(getattr(recheck, "passed", False)):
                                        continue
                                except Exception:
                                    pass
                            buffer_bbox = item.get("bbox", bbox)
                            buffer_mask = item.get("mask")
                            buffer_polygon = (
                                self._mask_to_polygon_points(
                                    buffer_mask,
                                    buffer_bbox,
                                    width=width,
                                    height=height,
                                )
                                if use_polygon
                                else None
                            )
                            self._append_annotation(
                                frame_annotations=frame_annotations,
                                frame_index=int(bf_idx),
                                frame_shape=(int(height), int(width)),
                                bbox=buffer_bbox,
                                class_id=int(class_id),
                                score=float(item.get("score", score)),
                                track_id=int(track_id),
                                fps=fps,
                                status=status,
                                polygon_points=buffer_polygon,
                            )
                            if status == "keep":
                                self._register_prefilter_final_keep(candidate_for_item)
                        if not current_frame_in_buffer:
                            self._append_annotation(
                                frame_annotations=frame_annotations,
                                frame_index=frame_idx,
                                frame_shape=(int(height), int(width)),
                                bbox=bbox,
                                class_id=int(class_id),
                                score=score,
                                track_id=int(track_id),
                                fps=fps,
                                status=status,
                                polygon_points=polygon_points,
                            )
                            if status == "keep":
                                self._register_prefilter_final_keep(prefilter_candidate)
                        if action == "HOLD_BUFFER":
                            hold_saved_count += 1
                        else:
                            keep_saved_count += 1
                        tracker.buffer = []

            processed_frames_in_window += 1
            if self._should_emit_preview(processed_frames_in_window, total_units):
                # Keep preview playback smooth at source FPS even when nothing is detected on this frame.
                self.preview_ready.emit(frame.copy(), preview_boxes, int(frame_idx))
            current_frame_idx += 1
            self._apply_processing_pacing(processed_frames=processed_frames_in_window, fps=fps, started_at=started_at)
            if self._should_emit_progress(processed_frames_in_window, total_units):
                self.progress_changed.emit(
                    self._build_progress(
                        stage="객체 추적" if tracking_announced else "객체 검출",
                        processed_units=processed_frames_in_window,
                        total_units=total_units,
                        current_frame=frame_idx,
                        total_frames=total_frames,
                        remaining_items=(
                            max(0, (total_frames - int(self.start_frame_index)) - processed_frames_in_window)
                            if self.sample_limit is None
                            else max(0, int(self.sample_limit) - len(frame_annotations))
                        ),
                        started_at=started_at,
                    )
                )

            if self.sample_limit is not None and len(frame_annotations) >= int(self.sample_limit):
                break

        annotations = sorted(frame_annotations.values(), key=lambda item: item.frame_index)
        if self.sample_limit is not None and len(annotations) > int(self.sample_limit):
            keep = self._sample_frame_indices(len(annotations), int(self.sample_limit))
            annotations = [annotations[i] for i in keep]

        if filtered_out_by_class_count > 0:
            self.log_message.emit(
                f"클래스 필터 적용: 입력 클래스에 없는 검출 {filtered_out_by_class_count}건 제외됨"
            )
        self.log_message.emit(
            f"추적 필터 결과: keep={keep_saved_count}, hold={hold_saved_count}, drop={drop_filtered_count}"
        )
        if prefilter_drop_by_reason:
            reason_text = ", ".join(
                [f"{key}={prefilter_drop_by_reason[key]}" for key in sorted(prefilter_drop_by_reason.keys())]
            )
            self.log_message.emit(f"사전 필터 드롭: {reason_text}")

        preview_items = self._build_preview_items_from_annotations(annotations, default_category="keep")
        preview_items.extend(drop_preview_items)
        if all_class_mode:
            # all-class 모드: 실제 탐지된 클래스명을 인덱스 순서로 반환
            sorted_names = sorted(class_name_to_id.items(), key=lambda x: x[1])
            class_names_out = [name for name, _ in sorted_names]
        else:
            class_names_out = list(self.class_names)
        return annotations, class_names_out, {"pipeline": "team_autolabeling"}, preview_items

    def _should_emit_preview(self, step: int, total: int) -> bool:
        """미리보기 프레임 전달 주기를 제어해 고속 모드 UI 오버헤드를 줄입니다."""
        if self.fast_mode:
            return False
        step_i = max(1, int(step))
        total_i = max(1, int(total))
        if step_i == 1 or step_i >= total_i:
            self._last_preview_emit_at = time.monotonic()
            return True
        if (step_i % max(1, int(self.preview_emit_interval))) != 0:
            return False
        now = time.monotonic()
        min_interval = float(self.preview_min_interval)
        if min_interval > 0.0 and (now - float(self._last_preview_emit_at)) < min_interval:
            return False
        self._last_preview_emit_at = now
        return True

    def _should_emit_progress(self, step: int, total: int) -> bool:
        """진행률 이벤트 전달 주기를 제어해 스레드 간 시그널 부하를 줄입니다."""
        step_i = max(1, int(step))
        total_i = max(1, int(total))
        if step_i == 1 or step_i >= total_i:
            self._last_progress_emit_at = time.monotonic()
            return True
        if (step_i % max(1, int(self.progress_emit_interval))) == 0:
            self._last_progress_emit_at = time.monotonic()
            return True
        if self.fast_mode:
            now = time.monotonic()
            if (now - float(self._last_progress_emit_at)) >= 0.18:
                self._last_progress_emit_at = now
                return True
        return False

    def _should_emit_step_log(self, step: int, total: int) -> bool:
        """중간 로그 출력 빈도를 조절해 고속 모드에서 불필요한 문자열 처리량을 줄입니다."""
        step_i = max(1, int(step))
        total_i = max(1, int(total))
        if step_i == 1 or step_i >= total_i:
            self._last_log_emit_at = time.monotonic()
            return True
        if (step_i % max(1, int(self.log_emit_interval))) == 0:
            self._last_log_emit_at = time.monotonic()
            return True
        if self.fast_mode:
            now = time.monotonic()
            if (now - float(self._last_log_emit_at)) >= 0.9:
                self._last_log_emit_at = now
                return True
        return False

    def _apply_processing_pacing(self, processed_frames: int, fps: float, started_at: float) -> None:
        """설정값 또는 필터를 대상 데이터/상태에 적용하고 후속 상태를 갱신합니다."""
        if (not self.realtime_mode) or fps <= 0.0 or processed_frames <= 0:
            return
        target_elapsed = float(processed_frames) / float(fps)
        delay = target_elapsed - max(0.0, (time.monotonic() - started_at))
        while delay > 0.0:
            self._check_stop()
            chunk = min(0.05, delay)
            time.sleep(chunk)
            delay -= chunk

    def _sample_frame_indices(self, total_frames: int, sample_count: int, start_frame: int = 0) -> list[int]:
        """전체 데이터에서 처리 대상 샘플 인덱스를 계산해 반환합니다."""
        if total_frames <= 0:
            return []

        start = max(0, min(int(start_frame), total_frames - 1))
        available = total_frames - start
        if available <= 0:
            return []

        sample_count = max(1, min(int(sample_count), available))
        if sample_count >= available:
            return list(range(start, total_frames))

        values = np.linspace(start, total_frames - 1, num=sample_count, dtype=np.int64)
        unique = sorted({int(v) for v in values})
        if len(unique) == sample_count:
            return unique

        needed = sample_count - len(unique)
        unique_set = set(unique)
        for idx in range(start, total_frames):
            if idx in unique_set:
                continue
            unique.append(idx)
            unique_set.add(idx)
            needed -= 1
            if needed <= 0:
                break
        return sorted(unique)[:sample_count]

    def _read_frame(self, cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
        """파일/프레임 등 외부 소스에서 데이터를 읽고 실패 시 안전한 기본 결과를 반환합니다."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return self._apply_roi_to_frame(frame)

    def _sanitize_roi_rect(self, roi: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
        """ROI 입력값을 내부 처리용 정수 사각형으로 정규화합니다."""
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
        """현재 프레임 크기에 맞춰 ROI를 보정하고 전체 프레임이면 None을 반환합니다."""
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
        """설정된 ROI가 있으면 프레임을 ROI 기준으로 크롭해 반환합니다."""
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

    def _sanitize_rgb_bits(self, value: int) -> int:
        """입력 RGB 비트 값을 허용 값(8/16/24/32)으로 보정해 반환합니다."""
        try:
            numeric = int(value)
        except Exception:
            return 24
        allowed = [8, 16, 24, 32]
        return min(allowed, key=lambda v: abs(v - numeric))

    def _apply_rgb_bits(self, frame: np.ndarray) -> np.ndarray:
        """설정값 또는 필터를 대상 데이터/상태에 적용하고 후속 상태를 갱신합니다."""
        if self.rgb_bits <= 8:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return frame

    def _detect_boxes(self, frame: np.ndarray) -> list[BoxAnnotation]:
        """프레임에서 객체를 검출하고 박스/점수 정보를 반환합니다."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        height, width = frame.shape[:2]
        min_area = max(120.0, float(width * height) * 0.0015)
        boxes: list[BoxAnnotation] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 12 or h < 12:
                continue

            class_id = self._pick_class_id(w, h)
            boxes.append(
                BoxAnnotation(
                    class_id=class_id,
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                    score=0.8,
                )
            )
            if len(boxes) >= 4:
                break

        if boxes:
            return boxes

        # Fallback box to keep V1 preview and E2E flow available even when contour detection fails.
        fallback_w = max(40, width // 4)
        fallback_h = max(40, height // 4)
        center_x = width // 2
        center_y = height // 2
        x1 = max(0, center_x - fallback_w // 2)
        y1 = max(0, center_y - fallback_h // 2)
        x2 = min(width, x1 + fallback_w)
        y2 = min(height, y1 + fallback_h)
        return [BoxAnnotation(class_id=0, x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), score=0.5)]

    def _pick_class_id(self, box_w: int, box_h: int) -> int:
        """후보 중 우선순위 규칙에 따라 대상 값을 선택해 반환합니다."""
        if len(self.class_names) <= 1:
            return 0
        return 0 if box_h >= box_w else 1

    def _track_sequence(self, items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """연속 프레임 데이터에서 추적 상태를 계산해 결과 시퀀스를 구성합니다."""
        tracked_items: list[dict[str, Any]] = []
        prev_tracks: list[tuple[int, BoxAnnotation]] = []
        next_track_id = 1

        for item in items:
            current_tracks: list[tuple[int, BoxAnnotation]] = []
            boxes: list[BoxAnnotation] = item["boxes"]
            for box in boxes:
                track_id = None
                best_iou = 0.0
                for prev_id, prev_box in prev_tracks:
                    iou = self._iou(box, prev_box)
                    if iou > best_iou:
                        best_iou = iou
                        track_id = prev_id

                if track_id is None or best_iou < 0.35:
                    track_id = next_track_id
                    next_track_id += 1

                box.track_id = int(track_id)
                current_tracks.append((int(track_id), box))

            tracked_items.append(
                {
                    "frame_index": item["frame_index"],
                    "boxes": [box for _, box in current_tracks],
                }
            )
            prev_tracks = current_tracks

        return tracked_items

    def _iou(self, box_a: BoxAnnotation, box_b: BoxAnnotation) -> float:
        """두 바운딩 박스의 교집합 대비 합집합 비율(IoU)을 계산해 반환합니다."""
        x1 = max(box_a.x1, box_b.x1)
        y1 = max(box_a.y1, box_b.y1)
        x2 = min(box_a.x2, box_b.x2)
        y2 = min(box_a.y2, box_b.y2)

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = max(0.0, (box_a.x2 - box_a.x1) * (box_a.y2 - box_a.y1))
        area_b = max(0.0, (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1))
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _resolve_model_class_name(self, model: Any, cls_idx: int) -> str:
        """입력 후보를 실제로 사용할 확정 값으로 해석해 반환합니다."""
        names = getattr(model, "names", {})
        if isinstance(names, dict):
            return str(names.get(cls_idx, f"클래스_{cls_idx}"))
        if isinstance(names, list) and 0 <= cls_idx < len(names):
            return str(names[cls_idx])
        return f"클래스_{cls_idx}"

    def _append_annotation(
        self,
        frame_annotations: dict[int, FrameAnnotation],
        frame_index: int,
        frame_shape: tuple[int, int] | None,
        bbox: Any,
        class_id: int,
        score: float,
        track_id: int,
        fps: float,
        status: str = "keep",
        polygon_points: Sequence[Any] | None = None,
    ) -> None:
        """결과/로그/목록 버퍼에 항목을 추가하고 관련 상태를 갱신합니다."""
        if frame_shape is None:
            return

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            return

        try:
            height, width = int(frame_shape[0]), int(frame_shape[1])
        except Exception:
            return
        if height <= 0 or width <= 0:
            return

        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            return

        ann = frame_annotations.get(frame_index)
        if ann is None:
            ann = FrameAnnotation(
                frame_index=frame_index,
                image=None,
                image_name=f"{self.video_path.stem}_{frame_index:06d}.jpg",
                split="train",
                boxes=[],
                timestamp_sec=frame_index / fps,
            )
            frame_annotations[frame_index] = ann

        def _parse_existing_box(item_box: Any) -> tuple[float, float, float, float, int, int | None, str] | None:
            if not isinstance(item_box, dict):
                return None
            try:
                ex_x1 = float(item_box.get("x1", 0.0))
                ex_y1 = float(item_box.get("y1", 0.0))
                ex_x2 = float(item_box.get("x2", 0.0))
                ex_y2 = float(item_box.get("y2", 0.0))
                ex_cls = int(item_box.get("class_id", 0))
            except Exception:
                return None
            raw_track = item_box.get("track_id")
            ex_track = None
            if raw_track is not None:
                try:
                    ex_track = int(raw_track)
                except Exception:
                    ex_track = None
            ex_status = str(item_box.get("status", "")).strip().lower()
            return ex_x1, ex_y1, ex_x2, ex_y2, ex_cls, ex_track, ex_status

        def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0.0:
                return 0.0
            aa = max(0.0, (ax2 - ax1) * (ay2 - ay1))
            bb = max(0.0, (bx2 - bx1) * (by2 - by1))
            denom = aa + bb - inter
            if denom <= 0.0:
                return 0.0
            return inter / denom

        new_status = str(status).strip().lower()
        new_track = int(track_id) if track_id is not None else None
        new_box = (float(x1), float(y1), float(x2), float(y2))
        for existing in ann.boxes:
            parsed_existing = _parse_existing_box(existing)
            if parsed_existing is None:
                continue
            ex_x1, ex_y1, ex_x2, ex_y2, ex_cls, ex_track, ex_status = parsed_existing
            if ex_cls != int(class_id):
                continue
            if ex_track != new_track:
                continue
            if ex_status != new_status:
                continue
            exact = (
                abs(ex_x1 - new_box[0]) <= 1.0
                and abs(ex_y1 - new_box[1]) <= 1.0
                and abs(ex_x2 - new_box[2]) <= 1.0
                and abs(ex_y2 - new_box[3]) <= 1.0
            )
            if exact:
                return
            if _iou_xyxy((ex_x1, ex_y1, ex_x2, ex_y2), new_box) >= 0.995:
                return

        ann.boxes.append(
            self._build_box_payload(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                class_id=int(class_id),
                score=float(score),
                track_id=int(track_id),
                status=status,
                polygon_points=polygon_points,
            )
        )

    def _build_box_payload(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        class_id: int,
        score: float,
        track_id: int | None,
        status: str = "keep",
        preview_item_id: str | None = None,
        polygon_points: Sequence[Any] | None = None,
        drop_reason: str | None = None,
    ) -> dict[str, Any]:
        """박스 좌표/클래스/상태 정보를 표준 딕셔너리 payload 형태로 구성해 반환합니다."""
        resolved_preview_item_id = str(preview_item_id).strip() if preview_item_id is not None else ""
        if not resolved_preview_item_id:
            resolved_preview_item_id = self._next_preview_item_id()
        payload: dict[str, Any] = {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "class_id": int(class_id),
            "score": float(score),
            "track_id": int(track_id) if track_id is not None else None,
            "status": str(status).strip().lower(),
            "preview_item_id": resolved_preview_item_id,
        }
        if drop_reason is not None:
            reason = str(drop_reason).strip().lower()
            if reason:
                payload["drop_reason"] = reason
        if polygon_points:
            normalized: list[list[int]] = []
            for pt in polygon_points:
                try:
                    px, py = float(pt[0]), float(pt[1])
                except Exception:
                    continue
                normalized.append([int(round(px)), int(round(py))])
            if len(normalized) >= 3:
                payload["polygon"] = normalized
        return payload

    def _mask_to_polygon_points(
        self,
        mask_points: Any,
        bbox: Any,
        width: int,
        height: int,
    ) -> list[tuple[int, int]] | None:
        """마스크 데이터를 포인트/폴리곤 좌표 형태로 변환합니다."""
        polygon: list[tuple[int, int]] = []
        if mask_points is not None:
            try:
                for pt in mask_points:
                    x = int(round(float(pt[0])))
                    y = int(round(float(pt[1])))
                    x = max(0, min(int(width) - 1, x))
                    y = max(0, min(int(height) - 1, y))
                    polygon.append((x, y))
            except Exception:
                polygon = []
        if len(polygon) >= 3:
            return polygon

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except Exception:
            return None
        x1i = max(0, min(int(width) - 1, int(round(x1))))
        y1i = max(0, min(int(height) - 1, int(round(y1))))
        x2i = max(0, min(int(width) - 1, int(round(x2))))
        y2i = max(0, min(int(height) - 1, int(round(y2))))
        if x2i <= x1i or y2i <= y1i:
            return None
        return [(x1i, y1i), (x2i, y1i), (x2i, y2i), (x1i, y2i)]

    def _status_from_tracker_action(self, action: str) -> str:
        """내부 액션/상태 값을 UI 표시용 상태 코드로 매핑합니다."""
        key = str(action).strip().upper()
        if key in {"KEEP", "KEEP_BUFFER", "SAVE", "SAVE_BUFFER"}:
            return "keep"
        if key in {"HOLD", "HOLD_BUFFER"}:
            return "hold"
        if key in {"DROP", "DELETE_TRACK"}:
            return "drop"
        return "candidate"

    def _next_preview_item_id(self) -> str:
        """세션 내에서 고유한 preview item 식별자를 생성해 반환합니다."""
        self._preview_item_serial += 1
        return f"pv_{self._preview_item_serial:08d}"

    def _ensure_preview_item_id(self, box_payload: Any) -> str:
        """박스 payload에 preview item ID가 없으면 생성해 주입하고 ID 문자열을 반환합니다."""
        if isinstance(box_payload, dict):
            existing = str(box_payload.get("preview_item_id", "")).strip()
            if existing:
                return existing
            created = self._next_preview_item_id()
            box_payload["preview_item_id"] = created
            return created
        return self._next_preview_item_id()

    def _append_preview_item(
        self,
        preview_items: list[PreviewThumbnail],
        frame_index: int,
        frame_shape: tuple[int, int] | None,
        bbox: Any,
        class_id: int,
        score: float,
        track_id: int,
        category: str,
        polygon_points: Sequence[Any] | None = None,
        drop_reason: str | None = None,
    ) -> None:
        """프레임/박스 정보를 기반으로 썸네일 메타 항목(이미지는 디스크 캐시 단계에서 로딩)을 생성합니다."""
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except Exception:
            return

        if frame_shape is None:
            return
        try:
            h, w = int(frame_shape[0]), int(frame_shape[1])
        except Exception:
            return
        if h <= 0 or w <= 0:
            return

        box_payload = self._build_box_payload(
            x1=max(0.0, min(float(w), x1)),
            y1=max(0.0, min(float(h), y1)),
            x2=max(0.0, min(float(w), x2)),
            y2=max(0.0, min(float(h), y2)),
            class_id=int(class_id),
            score=float(score),
            track_id=int(track_id),
            status=category,
            polygon_points=polygon_points,
            drop_reason=drop_reason,
        )
        preview_item_id = self._ensure_preview_item_id(box_payload)
        preview_items.append(
            PreviewThumbnail(
                frame_index=int(frame_index),
                boxes=[box_payload],
                category=str(category).strip().lower(),
                item_id=preview_item_id,
            )
        )

    def _build_preview_items_from_annotations(
        self,
        annotations: Sequence[FrameAnnotation],
        default_category: str = "keep",
    ) -> list[PreviewThumbnail]:
        """주석 결과를 썸네일 미리보기 리스트 형태로 변환해 반환합니다."""
        preview_items: list[PreviewThumbnail] = []
        fallback_category = str(default_category).strip().lower() or "keep"
        for ann in annotations:
            if not ann.boxes:
                continue
            for box in ann.boxes:
                status = fallback_category
                preview_item_id = ""
                if isinstance(box, dict):
                    raw_status = str(box.get("status", "")).strip().lower()
                    if raw_status in {"keep", "hold", "drop"}:
                        status = raw_status
                    preview_item_id = self._ensure_preview_item_id(box)
                else:
                    preview_item_id = self._next_preview_item_id()
                preview_items.append(
                    PreviewThumbnail(
                        frame_index=int(ann.frame_index),
                        boxes=[box],
                        category=status,
                        item_id=preview_item_id,
                    )
                )
        return preview_items

    def _build_progress(
        self,
        stage: str,
        processed_units: int,
        total_units: int,
        current_frame: int,
        total_frames: int,
        remaining_items: int,
        started_at: float,
    ) -> ProgressEvent:
        """현재 처리량과 경과 시간을 바탕으로 ETA를 계산해 ProgressEvent 객체를 생성합니다."""
        elapsed = max(0.001, time.monotonic() - started_at)
        if processed_units <= 0:
            eta = 0.0
        else:
            eta = (elapsed / float(processed_units)) * max(0, total_units - processed_units)
        return ProgressEvent(
            stage=stage,
            processed_units=int(processed_units),
            total_units=int(total_units),
            current_frame=int(current_frame),
            total_frames=int(total_frames),
            remaining_items=max(0, int(remaining_items)),
            eta_seconds=max(0.0, float(eta)),
        )
