from __future__ import annotations

import json
import queue
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QListWidgetItem, QPushButton, QVBoxLayout

from core.dataset import BoxAnnotation
from core.models import PreviewThumbnail, WorkerOutput
from app.studio.runtime import cv2

def _clear_preview_cache_ops(self) -> None:
    """현재 미리보기 캐시/감시 리소스를 정리하고 메모리 목록을 초기화합니다."""
    self._stop_preview_cache_watchdog()
    self._clear_thumbnail_source_cache()
    old_root = self.preview_cache_root
    self.preview_cache_root = None
    self.thumbnail_items = []
    self.pending_thumbnail_items = []
    self.thumbnail_visible_indices = []
    self.thumb_preview_source_indices = []
    if self.listThumbs is not None:
        self.listThumbs.clear()
    self._update_thumbnail_filter_tab_counts()
    if self.thumb_preview_dialog is not None and self.thumb_preview_dialog.isVisible():
        self.thumb_preview_dialog.close()
    try:
        while True:
            self.preview_cache_event_queue.get_nowait()
    except queue.Empty:
        pass
    if old_root is not None and old_root.is_dir():
        try:
            shutil.rmtree(old_root, ignore_errors=True)
        except Exception:
            pass
    self._update_button_state()

def _delete_preview_cache_storage_after_load_ops(self) -> None:
    """캐시를 메모리로 올린 뒤 디스크 임시 폴더를 삭제하고 메모리 참조만 유지합니다."""
    old_root = self.preview_cache_root
    self._stop_preview_cache_watchdog()
    self._clear_thumbnail_source_cache()
    self.preview_cache_root = None
    self.pending_thumbnail_items = []
    for item in self.thumbnail_items:
        item.manifest_path = None
        item.image_path = None
        item.thumb_path = None
    if old_root is not None and old_root.is_dir():
        try:
            shutil.rmtree(old_root, ignore_errors=True)
        except Exception:
            pass
    self._update_button_state()

def _delete_preview_cache_storage_after_export_ops(self) -> None:
    """내보내기 완료 후 디스크 임시 캐시 폴더만 정리하고 메모리 상태는 유지합니다."""
    old_root = self.preview_cache_root
    self._stop_preview_cache_watchdog()
    self._clear_thumbnail_source_cache()
    self.preview_cache_root = None
    self.pending_thumbnail_items = []
    if old_root is not None and old_root.is_dir():
        try:
            shutil.rmtree(old_root, ignore_errors=True)
        except Exception:
            pass
    self._update_button_state()

def _stop_preview_cache_watchdog_ops(self) -> None:
    """동작 중인 watchdog 관찰자를 종료하고 핸들을 정리합니다."""
    self.preview_watch_timer.stop()
    observer = self.preview_watchdog_observer
    self.preview_watchdog_observer = None
    self.preview_watchdog_handler = None
    if observer is None:
        return
    try:
        observer.stop()
        observer.join(timeout=1.5)
    except Exception:
        pass

def _drain_preview_cache_events_ops(self) -> None:
    """주기적으로 이벤트 큐를 비우고 변경이 있으면 썸네일 목록을 디스크에서 다시 로드합니다."""
    if self.preview_cache_root is None:
        return
    changed = False
    while True:
        try:
            self.preview_cache_event_queue.get_nowait()
            changed = True
        except queue.Empty:
            break
    if changed:
        self._load_thumbnail_items_from_cache()

def _normalize_thumbnail_category_ops(self, raw: str) -> str:
    """입력 카테고리 문자열을 keep/hold/drop 중 하나로 정규화해 반환합니다."""
    key = str(raw).strip().lower()
    if key in {"keep", "hold", "drop"}:
        return key
    return "hold"

def _persist_preview_items_to_cache_ops(self, payload: WorkerOutput) -> int:
    """워커 결과를 디스크 없이 메모리 대기 목록으로 정규화해 저장하고 항목 수를 반환합니다."""

    self._clear_preview_cache()
    self.preview_cache_root = None
    self.pending_thumbnail_items = []

    for ann in payload.frame_annotations:
        ann.image = None
        ann.image_path = None

    normalized_items: list[PreviewThumbnail] = []
    for serial, item in enumerate(payload.preview_items, start=1):
        category = self._normalize_thumbnail_category(item.category)
        frame_index = int(item.frame_index)

        normalized_boxes: list[dict[str, Any]] = []
        for box_idx, box in enumerate(item.boxes, start=1):
            copied: dict[str, Any] | None = None
            if isinstance(box, dict):
                copied = dict(box)
            elif isinstance(box, BoxAnnotation):
                copied = {
                    "x1": float(box.x1),
                    "y1": float(box.y1),
                    "x2": float(box.x2),
                    "y2": float(box.y2),
                    "class_id": int(box.class_id),
                    "score": float(box.score),
                    "track_id": int(box.track_id) if box.track_id is not None else None,
                }
            if copied is None:
                continue
            preview_item_id = str(copied.get("preview_item_id", "")).strip()
            if not preview_item_id:
                preview_item_id = str(item.item_id).strip() or f"pv_mem_{serial:08d}_{box_idx:02d}"
                copied["preview_item_id"] = preview_item_id
            copied["status"] = category
            normalized_boxes.append(copied)
        if not normalized_boxes:
            continue

        item_id = str(item.item_id).strip() or str(normalized_boxes[0].get("preview_item_id", "")).strip()
        if not item_id:
            item_id = f"pv_mem_{serial:08d}"

        normalized_items.append(
            PreviewThumbnail(
                frame_index=frame_index,
                image=None,
                boxes=normalized_boxes,
                category=category,
                item_id=item_id,
                image_path=None,
                thumb_path=None,
                manifest_path=None,
            )
        )
        if serial % 400 == 0:
            self._process_ui_keep_alive()

    normalized_items.sort(
        key=lambda item: (
            int(item.boxes[0].get("track_id"))
            if item.boxes and isinstance(item.boxes[0], dict) and item.boxes[0].get("track_id") is not None
            else 2_147_483_647,
            int(item.frame_index),
            str(item.item_id),
        )
    )
    self.pending_thumbnail_items = normalized_items
    item_count = len(self.pending_thumbnail_items)
    self._append_log(f"preview ready in memory: items={item_count}")
    return item_count

def _load_thumbnail_items_from_cache_ops(self) -> None:
    """디스크 캐시의 keep/hold/drop manifest를 읽어 썸네일 목록 메모리를 재구성합니다."""

    self._clear_thumbnail_source_cache()
    if self.pending_thumbnail_items:
        loaded_items: list[PreviewThumbnail] = []
        for source in self.pending_thumbnail_items:
            cloned_boxes: list[Any] = []
            for box in source.boxes:
                if isinstance(box, dict):
                    cloned_boxes.append(dict(box))
                else:
                    cloned_boxes.append(box)
            loaded_items.append(
                PreviewThumbnail(
                    frame_index=int(source.frame_index),
                    image=None,
                    boxes=cloned_boxes,
                    category=self._normalize_thumbnail_category(source.category),
                    item_id=str(source.item_id),
                    image_path=None,
                    thumb_path=None,
                    manifest_path=None,
                )
            )
        self.pending_thumbnail_items = []
        self.thumbnail_items = sorted(
            loaded_items,
            key=lambda item: (
                int(item.boxes[0].get("track_id"))
                if item.boxes and isinstance(item.boxes[0], dict) and item.boxes[0].get("track_id") is not None
                else 2_147_483_647,
                int(item.frame_index),
                str(item.item_id),
            ),
        )
        self._update_thumbnail_filter_tab_counts()
        self._refresh_thumbnail_list()
        self._update_button_state()
        if self.thumb_preview_dialog is not None and self.thumb_preview_dialog.isVisible():
            self.thumb_preview_source_indices = list(self.thumbnail_visible_indices)
            if not self.thumb_preview_source_indices:
                self.thumb_preview_dialog.close()
            else:
                self.thumb_preview_index = max(
                    0,
                    min(self.thumb_preview_index, len(self.thumb_preview_source_indices) - 1),
                )
                self._refresh_thumbnail_preview_dialog()
        return

    if self.preview_cache_root is None:
        self.thumbnail_items = []
        if self.listThumbs is not None:
            self.listThumbs.clear()
        return

    loaded_items: list[PreviewThumbnail] = []
    read_count = 0
    for category in ("keep", "hold", "drop"):
        category_dir = self.preview_cache_root / category
        if not category_dir.is_dir():
            continue
        for manifest_path in sorted(category_dir.glob("*.json")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            frame_index = int(payload.get("frame_index", -1))
            image_path = str(payload.get("image_path", "")).strip()
            thumb_path = str(payload.get("thumb_path", "")).strip()
            if frame_index < 0:
                continue
            raw_boxes = payload.get("boxes")
            if not isinstance(raw_boxes, list):
                continue
            boxes: list[Any] = []
            for box in raw_boxes:
                if not isinstance(box, dict):
                    continue
                copied = dict(box)
                copied["status"] = category
                boxes.append(copied)
            if not boxes:
                continue
            item_id = str(payload.get("item_id", "")).strip() or manifest_path.stem
            loaded_items.append(
                PreviewThumbnail(
                    frame_index=frame_index,
                    image=None,
                    boxes=boxes,
                    category=category,
                    item_id=item_id,
                    image_path=image_path,
                    thumb_path=thumb_path or None,
                    manifest_path=str(manifest_path),
                )
            )
            read_count += 1
            if read_count % 120 == 0:
                self._process_ui_keep_alive()

    self.thumbnail_items = sorted(
        loaded_items,
        key=lambda item: (
            int(item.boxes[0].get("track_id"))
            if item.boxes and isinstance(item.boxes[0], dict) and item.boxes[0].get("track_id") is not None
            else 2_147_483_647,
            int(item.frame_index),
            str(item.item_id),
        ),
    )
    self._update_thumbnail_filter_tab_counts()
    self._refresh_thumbnail_list()
    self._update_button_state()
    if self.thumb_preview_dialog is not None and self.thumb_preview_dialog.isVisible():
        self.thumb_preview_source_indices = list(self.thumbnail_visible_indices)
        if not self.thumb_preview_source_indices:
            self.thumb_preview_dialog.close()
        else:
            self.thumb_preview_index = max(
                0,
                min(self.thumb_preview_index, len(self.thumb_preview_source_indices) - 1),
            )
            self._refresh_thumbnail_preview_dialog()

def _load_thumbnail_source_image_ops(self, item_data: PreviewThumbnail) -> np.ndarray | None:
    """썸네일/팝업 렌더링에 사용할 원본 이미지를 메모리 또는 디스크에서 읽어 반환합니다."""
    preview_roi = self._result_run_roi()
    if preview_roi is None:
        preview_roi = self.video_roi
    roi_key = "full" if preview_roi is None else f"{preview_roi[0]}_{preview_roi[1]}_{preview_roi[2]}_{preview_roi[3]}"
    cache_key = (int(item_data.frame_index), roi_key)
    cached_frame = self._thumb_source_cache.get(cache_key)
    if isinstance(cached_frame, np.ndarray):
        self._thumb_source_cache.move_to_end(cache_key, last=True)
        return cached_frame

    if isinstance(item_data.image, np.ndarray):
        resolved = self._apply_roi_to_frame(item_data.image, preview_roi)
        self._remember_thumbnail_source_cache(cache_key, resolved)
        return resolved
    image_path = str(item_data.image_path or "").strip()
    if not image_path:
        frame = None
    else:
        frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if frame is not None and isinstance(frame, np.ndarray):
            resolved = self._apply_roi_to_frame(frame, preview_roi)
            self._remember_thumbnail_source_cache(cache_key, resolved)
            return resolved

    if self.video_path is None:
        return None
    cap = self._ensure_preview_image_cap()
    if cap is None:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(item_data.frame_index))
    ok, frame = cap.read()
    if ok and frame is not None:
        resolved = self._apply_roi_to_frame(frame, preview_roi)
        self._remember_thumbnail_source_cache(cache_key, resolved)
        return resolved

    self._release_preview_image_cap()
    cap = self._ensure_preview_image_cap()
    if cap is None:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(item_data.frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    resolved = self._apply_roi_to_frame(frame, preview_roi)
    self._remember_thumbnail_source_cache(cache_key, resolved)
    return resolved

def _populate_thumbnails_ops(self, preview_items: Sequence[PreviewThumbnail]) -> int:
    """워커 썸네일 메타를 메모리 대기 목록에 저장하고 저장된 항목 수를 반환합니다."""
    _ = preview_items
    if self.latest_result is None:
        return 0
    self.thumbnail_filter = "all"
    if self.tabAll is not None:
        self.tabAll.setChecked(True)
    written_count = self._persist_preview_items_to_cache(self.latest_result)
    if written_count <= 0:
        self.thumbnail_items = []
        self.thumbnail_visible_indices = []
        if self.listThumbs is not None:
            self.listThumbs.clear()
        self._update_thumbnail_filter_tab_counts()
        return 0

    # 미리보기 결과를 즉시 목록에 반영해 수동 불러오기 없이 바로 확인할 수 있도록 합니다.
    self._load_thumbnail_items_from_cache()
    self._delete_preview_cache_storage_after_load()
    return len(self.thumbnail_items)

def _update_thumbnail_filter_tab_counts_ops(self) -> None:
    """현재 데이터와 상태를 기준으로 UI 표시값 또는 내부 상태를 동기화합니다."""
    keep_count = 0
    hold_count = 0
    drop_count = 0
    for item_data in self.thumbnail_items:
        category = str(item_data.category).strip().lower()
        if category == "keep":
            keep_count += 1
        elif category == "hold":
            hold_count += 1
        elif category == "drop":
            drop_count += 1

    if self.tabAll is not None:
        self.tabAll.setText(f"전체(keep/hold) ({keep_count + hold_count})")
    if self.tabKeep is not None:
        self.tabKeep.setText(f"keep ({keep_count})")
    if self.tabHold is not None:
        self.tabHold.setText(f"hold ({hold_count})")
    if self.tabDrop is not None:
        self.tabDrop.setText(f"drop ({drop_count})")

def _set_thumbnail_filter_ops(self, key: str) -> None:
    """썸네일 필터 키를 설정하고 필터 조건에 맞게 목록을 다시 구성합니다."""
    normalized = str(key).strip().lower()
    if normalized not in {"all", "keep", "hold", "drop"}:
        normalized = "all"
    self.thumbnail_filter = normalized
    self._refresh_thumbnail_list()

def _refresh_thumbnail_list_ops(self) -> None:
    """현재 상태를 기준으로 화면/목록/미리보기를 다시 그려 최신 상태로 갱신합니다."""
    if self.listThumbs is None:
        return

    self.listThumbs.setUpdatesEnabled(False)
    try:
        self.listThumbs.clear()
        self.thumbnail_visible_indices = []
        for idx, item_data in enumerate(self.thumbnail_items):
            category = str(item_data.category).strip().lower()
            if not self._category_matches_filter(category):
                continue
            drop_reason = ""
            if category == "drop" and item_data.boxes and isinstance(item_data.boxes[0], dict):
                drop_reason = str(item_data.boxes[0].get("drop_reason", "")).strip().lower()
            category_label = f"{category}:{drop_reason}" if drop_reason else category
            item = QListWidgetItem(f"[{category_label}] 프레임 {int(item_data.frame_index):06d}")
            item.setToolTip(
                f"카테고리={category_label} | 프레임={int(item_data.frame_index):06d} | 박스={len(item_data.boxes)}"
            )
            item.setData(Qt.ItemDataRole.UserRole, int(len(self.thumbnail_visible_indices)))
            self.thumbnail_visible_indices.append(idx)
            self.listThumbs.addItem(item)
            if len(self.thumbnail_visible_indices) % 240 == 0:
                self._process_ui_keep_alive()
    finally:
        self.listThumbs.setUpdatesEnabled(True)

def _set_thumbnail_item_category_ops(self, source_idx: int, category: str) -> bool:
    """썸네일 항목 상태를 갱신합니다(캐시가 있으면 manifest 이동, 없으면 메모리 상태만 갱신)."""
    if not (0 <= source_idx < len(self.thumbnail_items)):
        return False
    normalized = self._normalize_thumbnail_category(category)
    if normalized not in {"keep", "hold", "drop"}:
        return False

    item_data = self.thumbnail_items[source_idx]
    current_category = self._normalize_thumbnail_category(item_data.category)
    if current_category == normalized:
        return False

    manifest_path_raw = str(item_data.manifest_path or "").strip()
    if self.preview_cache_root is None or not manifest_path_raw:
        item_data.category = normalized
        for box in item_data.boxes:
            if isinstance(box, dict):
                box["status"] = normalized
                box.pop("drop_reason", None)
        return True

    source_manifest_path = Path(manifest_path_raw)
    if not source_manifest_path.is_file():
        item_data.category = normalized
        for box in item_data.boxes:
            if isinstance(box, dict):
                box["status"] = normalized
                box.pop("drop_reason", None)
        return True

    target_dir = self.preview_cache_root / normalized
    target_dir.mkdir(parents=True, exist_ok=True)
    target_manifest_path = target_dir / source_manifest_path.name

    try:
        payload = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    payload["category"] = normalized
    boxes = payload.get("boxes")
    if isinstance(boxes, list):
        for idx, box in enumerate(boxes):
            if not isinstance(box, dict):
                continue
            copied = dict(box)
            copied["status"] = normalized
            copied.pop("drop_reason", None)
            boxes[idx] = copied
    try:
        target_manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if source_manifest_path.resolve() != target_manifest_path.resolve():
            source_manifest_path.unlink(missing_ok=True)
    except Exception:
        return False

    item_data.category = normalized
    item_data.manifest_path = str(target_manifest_path)
    for box in item_data.boxes:
        if isinstance(box, dict):
            box["status"] = normalized
            box.pop("drop_reason", None)
    return True

def _category_matches_filter_ops(self, category: str) -> bool:
    """썸네일 카테고리가 현재 선택된 필터 조건과 일치하는지 판정합니다."""
    key = str(category).strip().lower()
    if self.thumbnail_filter == "all":
        return key in {"keep", "hold"}
    return key == self.thumbnail_filter

def _on_thumbnail_double_clicked_ops(self, item: QListWidgetItem) -> None:
    """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
    if not self.thumbnail_visible_indices:
        return

    visible_pos_raw = item.data(Qt.ItemDataRole.UserRole)
    try:
        visible_pos = int(visible_pos_raw)
    except Exception:
        visible_pos = self.listThumbs.row(item) if self.listThumbs is not None else 0

    visible_pos = max(0, min(visible_pos, len(self.thumbnail_visible_indices) - 1))
    self._open_thumbnail_preview_dialog(visible_pos)

def _open_thumbnail_preview_dialog_ops(self, start_visible_pos: int) -> None:
    """파일/리소스를 열고 로드 결과에 맞춰 초기 상태를 설정합니다."""
    if not self.thumbnail_visible_indices:
        return

    if self.thumb_preview_dialog is None:
        dialog = QDialog(self)
        dialog.setWindowTitle("이미지 미리보기")
        dialog.resize(1120, 760)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.thumb_preview_info_label = QLabel(dialog)
        self.thumb_preview_info_label.setWordWrap(True)
        layout.addWidget(self.thumb_preview_info_label)

        self.thumb_preview_image_label = QLabel(dialog)
        self.thumb_preview_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_preview_image_label.setMinimumSize(960, 620)
        layout.addWidget(self.thumb_preview_image_label, 1)

        nav_layout = QHBoxLayout()
        self.thumb_preview_move_keep_button = QPushButton("keep으로 보내기", dialog)
        self.thumb_preview_move_keep_button.clicked.connect(
            lambda: self._move_current_thumbnail_in_preview("keep")
        )
        self.thumb_preview_move_drop_button = QPushButton("drop으로 보내기", dialog)
        self.thumb_preview_move_drop_button.clicked.connect(
            lambda: self._move_current_thumbnail_in_preview("drop")
        )
        nav_layout.addWidget(self.thumb_preview_move_keep_button)
        nav_layout.addWidget(self.thumb_preview_move_drop_button)
        nav_layout.addStretch(1)
        self.thumb_preview_prev_button = QPushButton("이전", dialog)
        self.thumb_preview_prev_button.clicked.connect(lambda: self._step_thumbnail_preview_dialog(-1))
        self.thumb_preview_next_button = QPushButton("다음", dialog)
        self.thumb_preview_next_button.clicked.connect(lambda: self._step_thumbnail_preview_dialog(1))
        close_button = QPushButton("닫기", dialog)
        close_button.clicked.connect(dialog.close)
        nav_layout.addWidget(self.thumb_preview_prev_button)
        nav_layout.addWidget(self.thumb_preview_next_button)
        nav_layout.addWidget(close_button)
        layout.addLayout(nav_layout)

        self.thumb_preview_dialog = dialog

    self.thumb_preview_source_indices = list(self.thumbnail_visible_indices)
    self.thumb_preview_index = max(0, min(int(start_visible_pos), len(self.thumb_preview_source_indices) - 1))
    self._refresh_thumbnail_preview_dialog()
    self.thumb_preview_dialog.show()
    self.thumb_preview_dialog.raise_()
    self.thumb_preview_dialog.activateWindow()

def _step_thumbnail_preview_dialog_ops(self, delta: int) -> None:
    """미리보기/탐색 인덱스를 한 단계 이동한 뒤 화면 갱신을 수행합니다."""
    if not self.thumb_preview_source_indices:
        return
    next_index = self.thumb_preview_index + int(delta)
    next_index = max(0, min(next_index, len(self.thumb_preview_source_indices) - 1))
    if next_index == self.thumb_preview_index:
        return
    self.thumb_preview_index = next_index
    self._refresh_thumbnail_preview_dialog()

def _move_current_thumbnail_in_preview_ops(self, target_category: str) -> None:
    """팝업에서 현재 표시 중인 썸네일 상태를 keep/drop으로 전환하고 목록/카운트를 동기화합니다."""
    normalized_target = str(target_category).strip().lower()
    if normalized_target not in {"keep", "drop"}:
        return
    if not self.thumb_preview_source_indices:
        return

    source_idx = self.thumb_preview_source_indices[self.thumb_preview_index]
    if not (0 <= source_idx < len(self.thumbnail_items)):
        return

    item_data = self.thumbnail_items[source_idx]
    current_category = str(item_data.category).strip().lower()
    if current_category == "keep":
        allowed_targets = {"drop"}
    elif current_category == "hold":
        allowed_targets = {"keep", "drop"}
    elif current_category == "drop":
        allowed_targets = {"keep"}
    else:
        allowed_targets = set()
    if normalized_target not in allowed_targets:
        return

    previous_preview_index = int(self.thumb_preview_index)
    current_item_id = str(item_data.item_id).strip()
    frame_index = int(item_data.frame_index)
    moved = self._set_thumbnail_item_category(source_idx, normalized_target)
    if not moved:
        return
    if self.preview_cache_root is not None:
        self._load_thumbnail_items_from_cache()
    else:
        self._update_thumbnail_filter_tab_counts()
        self._refresh_thumbnail_list()

    self.thumb_preview_source_indices = list(self.thumbnail_visible_indices)
    if not self.thumb_preview_source_indices:
        if self.thumb_preview_dialog is not None:
            self.thumb_preview_dialog.close()
        self._append_log(
            f"user action: {current_category} -> {normalized_target} (frame {frame_index:06d})"
        )
        return

    target_visible_pos: int | None = None
    if current_item_id:
        for visible_pos, current_source_idx in enumerate(self.thumb_preview_source_indices):
            if not (0 <= current_source_idx < len(self.thumbnail_items)):
                continue
            candidate_item = self.thumbnail_items[current_source_idx]
            if str(candidate_item.item_id).strip() == current_item_id:
                target_visible_pos = visible_pos
                break

    if target_visible_pos is not None:
        self.thumb_preview_index = target_visible_pos
    else:
        self.thumb_preview_index = max(0, min(previous_preview_index, len(self.thumb_preview_source_indices) - 1))
    self._refresh_thumbnail_preview_dialog()
    self._append_log(
        f"user action: {current_category} -> {normalized_target} (frame {frame_index:06d})"
    )

def _refresh_thumbnail_preview_dialog_ops(self) -> None:
    """현재 상태를 기준으로 화면/목록/미리보기를 다시 그려 최신 상태로 갱신합니다."""
    if (
        not self.thumb_preview_source_indices
        or self.thumb_preview_image_label is None
        or self.thumb_preview_info_label is None
    ):
        return

    source_idx = self.thumb_preview_source_indices[self.thumb_preview_index]
    if not (0 <= source_idx < len(self.thumbnail_items)):
        return
    item_data = self.thumbnail_items[source_idx]
    source_frame = self._load_thumbnail_source_image(item_data)
    if source_frame is None:
        return

    frame = self._draw_boxes(source_frame, item_data.boxes)
    category = str(item_data.category).strip().lower()
    drop_reason = ""
    if category == "drop" and item_data.boxes and isinstance(item_data.boxes[0], dict):
        drop_reason = str(item_data.boxes[0].get("drop_reason", "")).strip().lower()
    category_label = f"{category}:{drop_reason}" if drop_reason else category
    self.thumb_preview_image_label.setPixmap(self._frame_to_pixmap(frame, max_width=1040, max_height=640))
    self.thumb_preview_info_label.setText(
        f"{self.thumb_preview_index + 1}/{len(self.thumb_preview_source_indices)} | "
        f"카테고리={category_label} | 프레임={int(item_data.frame_index):06d} | 박스={len(item_data.boxes)}"
    )
    if self.thumb_preview_prev_button is not None:
        self.thumb_preview_prev_button.setEnabled(self.thumb_preview_index > 0)
    if self.thumb_preview_next_button is not None:
        self.thumb_preview_next_button.setEnabled(
            self.thumb_preview_index < len(self.thumb_preview_source_indices) - 1
        )
    if self.thumb_preview_move_keep_button is not None:
        show_keep_move = category in {"hold", "drop"}
        self.thumb_preview_move_keep_button.setVisible(show_keep_move)
        self.thumb_preview_move_keep_button.setEnabled(show_keep_move and (not self.is_processing))
    if self.thumb_preview_move_drop_button is not None:
        show_drop_move = category in {"keep", "hold"}
        self.thumb_preview_move_drop_button.setVisible(show_drop_move)
        self.thumb_preview_move_drop_button.setEnabled(show_drop_move and (not self.is_processing))
