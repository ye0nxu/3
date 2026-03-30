from __future__ import annotations

from typing import Any, Type

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget


def _build_video_load_page_page(self: Any, hover_preview_label_cls: Type[QLabel]) -> QWidget:
    """영상 로드, 메타 정보, 호버 미리보기 UI를 포함한 페이지를 구성해 반환합니다."""
    page = QWidget(self)
    root_layout = QHBoxLayout(page)
    root_layout.setContentsMargins(0, 0, 0, 0)
    root_layout.setSpacing(14)

    left_card = QFrame(page)
    left_card.setProperty("pageCard", True)
    left_layout = QVBoxLayout(left_card)
    left_layout.setContentsMargins(18, 16, 18, 16)
    left_layout.setSpacing(10)

    title = QLabel("영상입력", left_card)
    title.setObjectName("labelPageCardTitle")
    left_layout.addWidget(title)

    subtitle = QLabel(
        "미리보기 영역에 마우스를 올리면 약 10초간 자동 재생됩니다.",
        left_card,
    )
    subtitle.setWordWrap(True)
    subtitle.setStyleSheet("color: rgb(113, 126, 149); font-size: 9pt;")
    left_layout.addWidget(subtitle)

    self.labelVideoLoadPathValue = QLabel("선택된 영상이 없습니다.", left_card)
    self.labelVideoLoadPathValue.setWordWrap(True)
    self.labelVideoLoadMetaValue = QLabel("해상도: - | 초당 프레임: - | 프레임: -", left_card)
    self.labelVideoLoadRoiValue = QLabel("ROI: 전체 프레임", left_card)
    left_layout.addWidget(self.labelVideoLoadPathValue)
    left_layout.addWidget(self.labelVideoLoadMetaValue)
    left_layout.addWidget(self.labelVideoLoadRoiValue)

    preview_frame = QFrame(left_card)
    preview_frame.setObjectName("videoLoadPreviewFrame")
    preview_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    preview_frame.setMinimumHeight(520)
    preview_frame.setMaximumHeight(520)
    preview_layout = QVBoxLayout(preview_frame)
    preview_layout.setContentsMargins(0, 0, 0, 0)
    preview_layout.setSpacing(0)

    self.labelVideoLoadPreview = hover_preview_label_cls("마우스를 올려 10초 미리보기", preview_frame)
    self.labelVideoLoadPreview.setObjectName("labelVideoLoadPreview")
    self.labelVideoLoadPreview.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self.labelVideoLoadPreview.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    self.labelVideoLoadPreview.setMinimumSize(920, 520)
    self.labelVideoLoadPreview.setMaximumSize(920, 520)
    self.labelVideoLoadPreview.hover_entered.connect(self._start_video_load_hover_preview)
    self.labelVideoLoadPreview.hover_left.connect(self._stop_video_load_hover_preview)
    preview_layout.addWidget(self.labelVideoLoadPreview, 0, Qt.AlignmentFlag.AlignCenter)
    left_layout.addWidget(preview_frame, 1)

    buttons = QHBoxLayout()
    buttons.setSpacing(8)
    btn_open = QPushButton("영상 열기", left_card)
    btn_open.clicked.connect(self.on_open_video)
    self.btnVideoLoadSetRoi = QPushButton("ROI 설정", left_card)
    self.btnVideoLoadSetRoi.clicked.connect(self.on_set_video_roi)
    self.btnVideoLoadResetRoi = QPushButton("ROI 해제", left_card)
    self.btnVideoLoadResetRoi.clicked.connect(self.on_reset_video_roi)
    btn_to_stage1 = QPushButton("기존 객체 라벨링으로 이동", left_card)
    btn_to_stage1.clicked.connect(lambda: self._navigate_to_page("advanced_1"))
    btn_to_stage2 = QPushButton("신규 객체 라벨링으로 이동", left_card)
    btn_to_stage2.clicked.connect(lambda: self._navigate_to_page("advanced_2"))
    buttons.addWidget(btn_open)
    buttons.addWidget(self.btnVideoLoadSetRoi)
    buttons.addWidget(self.btnVideoLoadResetRoi)
    buttons.addWidget(btn_to_stage1)
    buttons.addWidget(btn_to_stage2)
    buttons.addStretch(1)
    left_layout.addLayout(buttons)

    root_layout.addWidget(left_card, 1)
    return page


def _refresh_video_load_page_page(self: Any) -> None:
    """현재 상태를 기준으로 화면/목록/미리보기를 다시 그려 최신 상태로 갱신합니다."""
    if (
        self.labelVideoLoadPathValue is None
        or self.labelVideoLoadMetaValue is None
        or self.labelVideoLoadRoiValue is None
        or self.labelVideoLoadPreview is None
    ):
        return
    if self.video_path is None:
        self._stop_video_load_hover_preview()
        self.last_video_load_preview_frame = None
        self.labelVideoLoadPathValue.setText("선택된 영상이 없습니다.")
        self.labelVideoLoadMetaValue.setText("해상도: - | 초당 프레임: - | 프레임: -")
        self.labelVideoLoadRoiValue.setText("ROI: -")
        self.labelVideoLoadPreview.clear()
        self.labelVideoLoadPreview.setText("마우스를 올려 10초 미리보기")
        return

    width = int(self.video_meta.get("width", 0))
    height = int(self.video_meta.get("height", 0))
    fps = float(self.video_meta.get("fps", 0.0))
    frame_count = int(self.video_meta.get("frame_count", 0))
    self.labelVideoLoadPathValue.setText(str(self.video_path))
    self.labelVideoLoadMetaValue.setText(
        f"해상도: {width}x{height} | 초당 프레임: {fps:.2f} | 프레임: {frame_count}"
    )
    self._update_video_load_roi_label()

    if self.video_load_hover_active:
        return

    frame_index = 0
    if self.sliderTimeline is not None:
        frame_index = int(self.sliderTimeline.value())
    frame = self._read_video_frame(frame_index)
    if frame is None:
        self.last_video_load_preview_frame = None
        self.labelVideoLoadPreview.clear()
        self.labelVideoLoadPreview.setText("마우스를 올려 10초 미리보기")
        return
    self._set_video_load_preview_frame(frame)
