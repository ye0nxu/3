from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QFileDialog, QMessageBox

import sys
try:
    from core.dataset import FrameAnnotation
except ModuleNotFoundError:
    PROJECT_DIR = Path(__file__).resolve().parents[2]
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from core.dataset import FrameAnnotation

from core.paths import DATASET_SAVE_DIR
from backend.storage.remote_storage import remote_path_for_local, remote_storage_enabled
from core.models import PreviewThumbnail
from app.studio.utils import (
    _build_unique_path,
    _load_yaml_dict,
    _parse_names_from_yaml_payload,
    _try_autofix_data_yaml_path,
    _try_autofix_data_yaml_splits,
    build_dataset_folder_name,
)
from app.studio.workers import DatasetExportWorker

def _on_export_dataset_ops(self) -> None:
    """내보내기를 백그라운드 스레드로 실행해 UI 멈춤 없이 처리합니다."""
    if self.is_processing or self.is_exporting:
        return
    if self.video_path is None or self.latest_result is None:
        QMessageBox.information(self, "데이터셋 내보내기", "내보낼 처리 결과가 없습니다.")
        return

    default_dir = str(DATASET_SAVE_DIR if DATASET_SAVE_DIR.is_dir() else Path.home())
    base_dir = QFileDialog.getExistingDirectory(self, "데이터셋 저장 위치 선택", default_dir)
    if not base_dir:
        return

    raw_video_stem = self.video_path.stem if self.video_path is not None else "video"
    video_stem = self._to_english_slug(raw_video_stem, fallback="video")

    source_items = self.thumbnail_items if self.thumbnail_items else self.pending_thumbnail_items
    preview_items = self._clone_preview_items_for_export(source_items)
    annotations = self._clone_frame_annotations_for_export(self.latest_result.frame_annotations)
    if (not preview_items) and (not annotations):
        QMessageBox.information(self, "데이터셋 내보내기", "내보낼 처리 결과가 없습니다.")
        return

    class_names = list(self.latest_result.class_names) if self.latest_result.class_names else list(self.active_class_names)
    if not class_names:
        class_names = ["object"]
    export_roi = self._result_run_roi()
    if export_roi is None:
        export_roi = self._normalize_video_roi(
            self.video_roi,
            int(self.video_meta.get("width", 0)),
            int(self.video_meta.get("height", 0)),
        )
    dataset_name = build_dataset_folder_name(class_names, kind="export", created_at=datetime.now())
    dataset_root = _build_unique_path(Path(base_dir) / dataset_name)
    logging.getLogger(__name__).info("dataset export started: target_dir=%s", dataset_root)

    self.export_thread = QThread(self)
    self.export_worker = DatasetExportWorker(
        video_path=self.video_path,
        dataset_root=dataset_root,
        video_stem=video_stem,
        preview_items=preview_items,
        frame_annotations=annotations,
        class_names=class_names,
        split_ratio=(0.8, 0.1, 0.1),
        shuffle_seed=None,
        roi_rect=export_roi,
    )
    self.export_worker.moveToThread(self.export_thread)

    self.export_thread.started.connect(self.export_worker.run)
    self.export_worker.progress.connect(self._on_export_progress)
    self.export_worker.finished.connect(self._on_export_finished)
    self.export_worker.failed.connect(self._on_export_failed)

    self.export_worker.finished.connect(self.export_thread.quit)
    self.export_worker.failed.connect(self.export_thread.quit)
    self.export_thread.finished.connect(self._on_export_thread_finished)
    self.export_thread.finished.connect(self.export_worker.deleteLater)
    self.export_thread.finished.connect(self.export_thread.deleteLater)

    self.is_exporting = True
    self._set_session_state("내보내기 중")
    self._append_log("start export")
    self._begin_loading("데이터셋을 내보내는 중입니다...")
    self._update_button_state()
    self.export_thread.start()

def _clone_preview_items_for_export_ops(self, items: Sequence[PreviewThumbnail]) -> list[PreviewThumbnail]:
    """스레드 안전을 위해 preview 항목을 복제해 반환합니다."""
    cloned: list[PreviewThumbnail] = []
    for item in items:
        cloned_boxes: list[Any] = []
        for box in item.boxes:
            if isinstance(box, dict):
                cloned_boxes.append(dict(box))
            else:
                cloned_boxes.append(box)
        preview_item_cls = type(item)
        cloned.append(
            preview_item_cls(
                frame_index=int(item.frame_index),
                image=None,
                boxes=cloned_boxes,
                category=str(item.category),
                item_id=str(item.item_id),
                image_path=None,
                thumb_path=None,
                manifest_path=None,
            )
        )
    return cloned

def _clone_frame_annotations_for_export_ops(self, annotations: Sequence[FrameAnnotation]) -> list[FrameAnnotation]:
    """스레드 안전을 위해 프레임 주석을 최소 필드만 복제해 반환합니다."""
    cloned: list[FrameAnnotation] = []
    for ann in annotations:
        boxes: list[Any] = []
        for box in ann.boxes:
            if isinstance(box, dict):
                boxes.append(dict(box))
            else:
                boxes.append(box)
        annotation_cls = type(ann)
        cloned.append(
            annotation_cls(
                frame_index=int(ann.frame_index),
                image=None,
                image_path=None,
                image_name=str(ann.image_name),
                split=str(ann.split),
                boxes=boxes,
                timestamp_sec=float(ann.timestamp_sec),
            )
        )
    return cloned

def _on_export_progress_ops(self, message: str, done: int, total: int) -> None:
    """백그라운드 내보내기 진행 상태를 로딩 패널 텍스트에 반영합니다."""
    if self.loading_label is not None:
        self.loading_label.setText(f"{message}: {done}/{max(1, total)}")
    if self.loading_sub_label is not None:
        self.loading_sub_label.setText("백그라운드 내보내기 실행 중")

def _on_export_finished_ops(self, summary_obj: object) -> None:
    """내보내기 성공 시 결과 요약을 표시하고 상태를 정리합니다."""
    self._end_loading()
    required_attrs = (
        "dataset_root",
        "train_images",
        "valid_images",
        "test_images",
        "train_labels",
        "valid_labels",
        "test_labels",
        "class_count",
        "total_boxes",
    )
    if not all(hasattr(summary_obj, attr) for attr in required_attrs):
        self._on_export_failed("내보내기 결과 형식이 올바르지 않습니다.")
        return

    summary = summary_obj
    exported_root = Path(summary.dataset_root).resolve()
    ordered_class_names = [str(name).strip() for name in getattr(summary, "class_names", []) if str(name).strip()]
    if not ordered_class_names:
        data_yaml_for_name = exported_root / "data.yaml"
        payload = _load_yaml_dict(data_yaml_for_name)
        ordered_class_names = _parse_names_from_yaml_payload(payload)
    if not ordered_class_names:
        ordered_class_names = ["class"]

    target_name = build_dataset_folder_name(ordered_class_names, kind="export", created_at=datetime.now())
    target_root = _build_unique_path(exported_root.parent / target_name)
    if exported_root.name.startswith("_tmp_export_") and target_root != exported_root:
        try:
            exported_root.rename(target_root)
            logging.getLogger(__name__).info(
                "dataset export renamed: old=%s new=%s",
                exported_root,
                target_root,
            )
            crop_root_raw = getattr(summary, "crop_root", None)
            if crop_root_raw is not None:
                crop_root = Path(crop_root_raw)
                if crop_root.is_dir():
                    renamed_crop_root = crop_root.with_name(target_root.name)
                    renamed_crop_root = _build_unique_path(renamed_crop_root)
                    try:
                        crop_root.rename(renamed_crop_root)
                        summary.crop_root = renamed_crop_root
                    except Exception:
                        summary.crop_root = crop_root
            summary.dataset_root = target_root
            exported_root = target_root
        except Exception as exc:
            logging.getLogger(__name__).warning("dataset export rename failed: %s", exc)

    self.last_export_dataset_root = exported_root
    if remote_storage_enabled():
        remote_dataset_root = remote_path_for_local(exported_root)
        if remote_dataset_root:
            self._append_log(f"remote dataset mirror: {remote_dataset_root}")
    data_yaml_path = self.last_export_dataset_root / "data.yaml"
    if data_yaml_path.is_file():
        if _try_autofix_data_yaml_path(data_yaml_path):
            self._append_log(f"data.yaml 경로 자동 보정: {data_yaml_path}")
        if _try_autofix_data_yaml_splits(data_yaml_path):
            self._append_log(f"data.yaml split 자동 보정: {data_yaml_path}")
        try:
            alias_yaml = self._create_retrain_merged_alias_for_yaml(data_yaml_path)
            if alias_yaml is not None:
                self._append_log(f"재학습용 merged YAML 준비: {alias_yaml}")
        except Exception as exc:
            self._append_log(f"warning: 재학습용 merged YAML 생성 실패 ({exc})")
        self._set_training_dataset_path(data_yaml_path)
    total_images = int(summary.train_images + summary.valid_images + summary.test_images)
    total_label_files = int(summary.train_labels + summary.valid_labels + summary.test_labels)
    total_box_annotations = int(summary.total_boxes)
    total_crop_images = int(getattr(summary, "crop_images", 0))
    crop_root = getattr(summary, "crop_root", None)
    crop_root_text = str(crop_root) if crop_root else "-"
    self._set_session_state("내보내기 완료")
    self._append_log(
        "finish export: "
        f"이미지={total_images}, 라벨파일={total_label_files}, 객체박스={total_box_annotations}, "
        f"crop={total_crop_images}, crop경로={crop_root_text}, 경로={self.last_export_dataset_root}",
    )
    QMessageBox.information(
        self,
        "내보내기 완료",
        f"데이터셋 저장 위치:\n{self.last_export_dataset_root}\n\n"
        f"이미지: {total_images}\n라벨 파일: {total_label_files}\n객체 박스: {total_box_annotations}\n"
        f"bbox crop 이미지: {total_crop_images}\n"
        f"bbox crop 경로: {crop_root_text}\n"
        f"학습/검증/테스트: {summary.train_images}/{summary.valid_images}/{summary.test_images}\n"
        f"클래스 수: {summary.class_count}",
    )
    self._delete_preview_cache_storage_after_export()
    self._update_button_state()

def _on_export_failed_ops(self, message: str) -> None:
    """내보내기 실패 시 로딩 상태를 해제하고 오류 메시지를 표시합니다."""
    self._end_loading()
    self._set_session_state("내보내기 실패")
    self._append_log(f"error: export failed ({message})")
    QMessageBox.critical(self, "내보내기 오류", f"데이터셋 내보내기에 실패했습니다.\n{message}")
    self._update_button_state()

def _on_export_thread_finished_ops(self) -> None:
    """내보내기 워커 스레드 종료 후 핸들을 정리합니다."""
    if self.loading_depth > 0:
        self._end_loading()
    self.is_exporting = False
    self.export_worker = None
    self.export_thread = None
    self._update_button_state()

def _on_load_cached_preview_ops(self) -> None:
    """keep/hold/drop 대기 결과를 화면에 수동 반영하고 임시 버퍼를 정리합니다."""
    if self.is_processing or self.is_exporting:
        return
    cache_count = self._count_cached_preview_items()
    if cache_count <= 0:
        QMessageBox.information(self, "미리보기 불러오기", "불러올 keep/hold/drop 결과가 없습니다.")
        return

    loaded_count = 0
    self._begin_loading("keep/hold/drop 미리보기를 불러오는 중입니다...")
    try:
        self.thumbnail_filter = "all"
        if self.tabAll is not None:
            self.tabAll.setChecked(True)
        self._load_thumbnail_items_from_cache()
        loaded_count = len(self.thumbnail_items)
        self._delete_preview_cache_storage_after_load()
    except Exception as exc:
        self._append_log(f"error: preview cache 불러오기 실패 ({exc})")
        QMessageBox.warning(self, "미리보기 불러오기", f"미리보기 불러오기에 실패했습니다.\n{exc}")
        return
    finally:
        self._end_loading()

    self._set_session_state("미리보기 로드 완료")
    self._append_log(f"user action: preview loaded ({loaded_count})")
    self._append_log("user action: preview pending buffer cleared")
    if loaded_count <= 0:
        QMessageBox.information(self, "미리보기 불러오기", "불러온 항목이 없습니다.")
    self._update_button_state()

def _label_for_export_class_id_ops(self, class_id: int) -> str:
    """내부 클래스/ID 값을 내보내기용 라벨 규칙에 맞게 변환해 반환합니다."""
    raw_label = f"class_{class_id}"
    if self.latest_result is not None and self.latest_result.class_names:
        if 0 <= class_id < len(self.latest_result.class_names):
            raw_label = str(self.latest_result.class_names[class_id])
    elif 0 <= class_id < len(self.active_class_names):
        raw_label = str(self.active_class_names[class_id])

    raw_label = str(raw_label).strip()
    if raw_label.casefold() in {"객체", "object"}:
        return "object"
    normalized = self._to_english_slug(raw_label, fallback=f"class_{class_id}")
    if normalized in {"class", "objects"}:
        return f"class_{class_id}"
    return normalized

def _to_english_slug_ops(self, text: str, fallback: str = "item") -> str:
    """입력 문자열을 영문 소문자/숫자/밑줄 기반 slug로 정규화해 반환합니다."""
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
