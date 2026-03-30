from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.models import PreviewThumbnail


class ResultThumbnailBrowser(QWidget):
    countsChanged = pyqtSignal(int, int, int)

    def __init__(
        self,
        parent: QWidget | None = None,
        log_callback: Callable[[str], None] | None = None,
        busy_getter: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self._log_callback = log_callback
        self._busy_getter = busy_getter
        self._video_path: str | None = None
        self.thumbnail_items: list[PreviewThumbnail] = []
        self.thumbnail_visible_indices: list[int] = []
        self.thumbnail_filter = "all"
        self.thumb_preview_source_indices: list[int] = []
        self.thumb_preview_index = 0
        self.thumb_preview_dialog: QDialog | None = None
        self.thumb_preview_info_label: QLabel | None = None
        self.thumb_preview_image_label: QLabel | None = None
        self.thumb_preview_prev_button: QPushButton | None = None
        self.thumb_preview_next_button: QPushButton | None = None
        self.thumb_preview_move_keep_button: QPushButton | None = None
        self.thumb_preview_move_drop_button: QPushButton | None = None
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        self.tab_row = QHBoxLayout()
        self.tab_row.setContentsMargins(0, 0, 0, 0)
        self.tab_row.setSpacing(8)
        self.tabGroup = QButtonGroup(self)
        self.tabGroup.setExclusive(True)
        self.tabAll = self._make_tab("tabAll", "전체(keep/hold)")
        self.tabKeep = self._make_tab("tabOk", "keep")
        self.tabHold = self._make_tab("tabHold", "hold")
        self.tabDrop = self._make_tab("tabDrop", "drop")
        for button in (self.tabAll, self.tabKeep, self.tabHold, self.tabDrop):
            self.tabGroup.addButton(button)
            self.tab_row.addWidget(button)
        self.tab_row.addStretch(1)
        root.addLayout(self.tab_row)

        self.listThumbs = QListWidget(self)
        self.listThumbs.setObjectName("listThumbs")
        root.addWidget(self.listThumbs, 1)

        self.tabAll.setChecked(True)

    def _connect_signals(self) -> None:
        self.tabAll.clicked.connect(lambda: self._set_thumbnail_filter("all"))
        self.tabKeep.clicked.connect(lambda: self._set_thumbnail_filter("keep"))
        self.tabHold.clicked.connect(lambda: self._set_thumbnail_filter("hold"))
        self.tabDrop.clicked.connect(lambda: self._set_thumbnail_filter("drop"))
        self.listThumbs.itemDoubleClicked.connect(self._on_thumbnail_double_clicked)

    def _make_tab(self, object_name: str, text: str) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName(object_name)
        button.setText(text)
        button.setCheckable(True)
        return button

    def clear(self) -> None:
        self.thumbnail_items = []
        self.thumbnail_visible_indices = []
        self.listThumbs.clear()
        self._update_counts()

    def set_video_path(self, video_path: str | None) -> None:
        self._video_path = str(video_path).strip() if video_path else None

    def set_items(self, items: Sequence[PreviewThumbnail], video_path: str | None = None) -> None:
        if video_path is not None:
            self._video_path = str(video_path).strip() or None
        self.thumbnail_items = [self._normalize_item(item) for item in items]
        self.thumbnail_filter = "all"
        self.tabAll.setChecked(True)
        self._update_counts()
        self._refresh_list()

    def append_items(self, items: Sequence[PreviewThumbnail]) -> None:
        self.thumbnail_items.extend(self._normalize_item(item) for item in items)
        self._update_counts()
        self._refresh_list()

    def _normalize_item(self, item: PreviewThumbnail) -> PreviewThumbnail:
        category = self._normalize_category(item.category)
        item_id = str(item.item_id).strip() or uuid.uuid4().hex
        return PreviewThumbnail(
            frame_index=int(item.frame_index),
            image=item.image.copy() if isinstance(item.image, np.ndarray) else item.image,
            boxes=list(item.boxes),
            category=category,
            item_id=item_id,
            image_path=item.image_path,
            thumb_path=item.thumb_path,
            manifest_path=item.manifest_path,
        )

    def _normalize_category(self, raw: str) -> str:
        key = str(raw).strip().lower()
        return key if key in {"keep", "hold", "drop"} else "hold"

    def _update_counts(self) -> None:
        keep_count = sum(1 for item in self.thumbnail_items if item.category == "keep")
        hold_count = sum(1 for item in self.thumbnail_items if item.category == "hold")
        drop_count = sum(1 for item in self.thumbnail_items if item.category == "drop")
        self.tabAll.setText(f"전체(keep/hold) ({keep_count + hold_count})")
        self.tabKeep.setText(f"keep ({keep_count})")
        self.tabHold.setText(f"hold ({hold_count})")
        self.tabDrop.setText(f"drop ({drop_count})")
        self.countsChanged.emit(keep_count, hold_count, drop_count)

    def _set_thumbnail_filter(self, key: str) -> None:
        normalized = str(key).strip().lower()
        self.thumbnail_filter = normalized if normalized in {"all", "keep", "hold", "drop"} else "all"
        self._refresh_list()

    def _refresh_list(self) -> None:
        self.listThumbs.setUpdatesEnabled(False)
        try:
            self.listThumbs.clear()
            self.thumbnail_visible_indices = []
            added_count = 0
            for idx, item_data in enumerate(self.thumbnail_items):
                category = self._normalize_category(item_data.category)
                if (self.thumbnail_filter == "all" and category not in {"keep", "hold"}) or (
                    self.thumbnail_filter != "all" and category != self.thumbnail_filter
                ):
                    continue
                item = QListWidgetItem(f"[{category}] 프레임 {int(item_data.frame_index):06d}")
                item.setData(Qt.ItemDataRole.UserRole, int(len(self.thumbnail_visible_indices)))
                self.thumbnail_visible_indices.append(idx)
                self.listThumbs.addItem(item)
                added_count += 1
                if (added_count % 200) == 0:
                    self.listThumbs.setUpdatesEnabled(True)
                    QApplication.processEvents()
                    self.listThumbs.setUpdatesEnabled(False)
        finally:
            self.listThumbs.setUpdatesEnabled(True)

    def _on_thumbnail_double_clicked(self, item: QListWidgetItem) -> None:
        if not self.thumbnail_visible_indices:
            return
        try:
            visible_pos = int(item.data(Qt.ItemDataRole.UserRole))
        except Exception:
            visible_pos = max(0, self.listThumbs.row(item))
        self._open_thumbnail_preview_dialog(visible_pos)

    def _open_thumbnail_preview_dialog(self, start_visible_pos: int) -> None:
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
            nav = QHBoxLayout()
            self.thumb_preview_move_keep_button = QPushButton("keep으로 보내기", dialog)
            self.thumb_preview_move_keep_button.clicked.connect(lambda: self._move_current_thumbnail_in_preview("keep"))
            self.thumb_preview_move_drop_button = QPushButton("drop으로 보내기", dialog)
            self.thumb_preview_move_drop_button.clicked.connect(lambda: self._move_current_thumbnail_in_preview("drop"))
            self.thumb_preview_prev_button = QPushButton("이전", dialog)
            self.thumb_preview_prev_button.clicked.connect(lambda: self._step_thumbnail_preview_dialog(-1))
            self.thumb_preview_next_button = QPushButton("다음", dialog)
            self.thumb_preview_next_button.clicked.connect(lambda: self._step_thumbnail_preview_dialog(1))
            close_button = QPushButton("닫기", dialog)
            close_button.clicked.connect(dialog.close)
            nav.addWidget(self.thumb_preview_move_keep_button)
            nav.addWidget(self.thumb_preview_move_drop_button)
            nav.addStretch(1)
            nav.addWidget(self.thumb_preview_prev_button)
            nav.addWidget(self.thumb_preview_next_button)
            nav.addWidget(close_button)
            layout.addLayout(nav)
            self.thumb_preview_dialog = dialog
        self.thumb_preview_source_indices = list(self.thumbnail_visible_indices)
        self.thumb_preview_index = max(0, min(int(start_visible_pos), len(self.thumb_preview_source_indices) - 1))
        self._refresh_thumbnail_preview_dialog()
        self.thumb_preview_dialog.show()
        self.thumb_preview_dialog.raise_()
        self.thumb_preview_dialog.activateWindow()

    def _step_thumbnail_preview_dialog(self, delta: int) -> None:
        if not self.thumb_preview_source_indices:
            return
        self.thumb_preview_index = max(
            0,
            min(self.thumb_preview_index + int(delta), len(self.thumb_preview_source_indices) - 1),
        )
        self._refresh_thumbnail_preview_dialog()

    def _move_current_thumbnail_in_preview(self, target_category: str) -> None:
        if not self.thumb_preview_source_indices:
            return
        source_idx = self.thumb_preview_source_indices[self.thumb_preview_index]
        if not (0 <= source_idx < len(self.thumbnail_items)):
            return
        item_data = self.thumbnail_items[source_idx]
        current_category = self._normalize_category(item_data.category)
        allowed = {"drop"} if current_category == "keep" else {"keep", "drop"} if current_category == "hold" else {"keep"} if current_category == "drop" else set()
        target = self._normalize_category(target_category)
        if target not in allowed:
            return
        item_data.category = target
        for box in item_data.boxes:
            if isinstance(box, dict):
                box["status"] = target
                box.pop("drop_reason", None)
        self._update_counts()
        self._refresh_list()
        self.thumb_preview_source_indices = list(self.thumbnail_visible_indices)
        if not self.thumb_preview_source_indices:
            if self.thumb_preview_dialog is not None:
                self.thumb_preview_dialog.close()
            return
        self.thumb_preview_index = max(0, min(self.thumb_preview_index, len(self.thumb_preview_source_indices) - 1))
        self._refresh_thumbnail_preview_dialog()
        self._append_log(f"user action: {current_category} -> {target} (frame {item_data.frame_index:06d})")

    def _refresh_thumbnail_preview_dialog(self) -> None:
        if not self.thumb_preview_source_indices or self.thumb_preview_image_label is None or self.thumb_preview_info_label is None:
            return
        source_idx = self.thumb_preview_source_indices[self.thumb_preview_index]
        if not (0 <= source_idx < len(self.thumbnail_items)):
            return
        item_data = self.thumbnail_items[source_idx]
        source_frame = self._load_thumbnail_source_image(item_data)
        if source_frame is None:
            return
        frame = self._draw_boxes(source_frame, item_data.boxes)
        category = self._normalize_category(item_data.category)
        self.thumb_preview_image_label.setPixmap(self._frame_to_pixmap(frame, 1040, 640))
        self.thumb_preview_info_label.setText(
            f"{self.thumb_preview_index + 1}/{len(self.thumb_preview_source_indices)} | "
            f"카테고리={category} | 프레임={int(item_data.frame_index):06d} | 박스={len(item_data.boxes)}"
        )
        if self.thumb_preview_prev_button is not None:
            self.thumb_preview_prev_button.setEnabled(self.thumb_preview_index > 0)
        if self.thumb_preview_next_button is not None:
            self.thumb_preview_next_button.setEnabled(self.thumb_preview_index < len(self.thumb_preview_source_indices) - 1)
        if self.thumb_preview_move_keep_button is not None:
            show_keep = category in {"hold", "drop"}
            self.thumb_preview_move_keep_button.setVisible(show_keep)
            self.thumb_preview_move_keep_button.setEnabled(show_keep and (not self._is_busy()))
        if self.thumb_preview_move_drop_button is not None:
            show_drop = category in {"keep", "hold"}
            self.thumb_preview_move_drop_button.setVisible(show_drop)
            self.thumb_preview_move_drop_button.setEnabled(show_drop and (not self._is_busy()))

    def _load_thumbnail_source_image(self, item_data: PreviewThumbnail) -> np.ndarray | None:
        if isinstance(item_data.image, np.ndarray):
            return item_data.image.copy()
        for path_value in (item_data.image_path, item_data.thumb_path):
            path = Path(str(path_value or "")).expanduser()
            if path.is_file():
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
        if self._video_path:
            cap = cv2.VideoCapture(self._video_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(item_data.frame_index))
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    return frame
        return None

    def _draw_boxes(self, frame: np.ndarray, boxes: Sequence[Any]) -> np.ndarray:
        canvas = frame.copy()
        for box in boxes:
            xyxy = self._extract_box_xyxy(box)
            if xyxy is None:
                continue
            x1, y1, x2, y2 = xyxy
            status = self._extract_box_status(box)
            color = self._box_color_for_status(status)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.putText(canvas, str(status or "item"), (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 255, 240), 2, cv2.LINE_AA)
        return canvas

    def _extract_box_xyxy(self, box: Any) -> tuple[int, int, int, int] | None:
        if isinstance(box, dict) and {"x1", "y1", "x2", "y2"}.issubset(box):
            return (
                int(round(float(box["x1"]))),
                int(round(float(box["y1"]))),
                int(round(float(box["x2"]))),
                int(round(float(box["y2"]))),
            )
        return None

    def _extract_box_status(self, box: Any) -> str | None:
        if isinstance(box, dict):
            return self._normalize_category(str(box.get("status", "")).strip().lower())
        return None

    def _box_color_for_status(self, status: str | None) -> tuple[int, int, int]:
        key = str(status or "").strip().lower()
        if key == "keep":
            return (80, 240, 120)
        if key == "hold":
            return (0, 165, 255)
        if key == "drop":
            return (50, 50, 255)
        return (120, 220, 240)

    def _frame_to_pixmap(self, frame: np.ndarray, max_width: int, max_height: int) -> QPixmap:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(image).scaled(
            max_width,
            max_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _append_log(self, message: str) -> None:
        if self._log_callback is not None:
            self._log_callback(message)

    def _is_busy(self) -> bool:
        return bool(self._busy_getter()) if self._busy_getter is not None else False
