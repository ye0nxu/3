from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from app.widgets.labeling_widget import NewObjectLabelingWidget
except Exception:  # pragma: no cover - optional dependency
    NewObjectLabelingWidget = None  # type: ignore[assignment]


def _build_second_advanced_page_page(self: Any) -> QWidget:
    """Build the second-stage labeling/training flow page."""
    page = QWidget(self)
    root_layout = QVBoxLayout(page)
    root_layout.setContentsMargins(0, 0, 0, 0)
    root_layout.setSpacing(12)

    self.labelSecondStageStepState = None
    self.labelSecondStageStatus = None

    self.second_stage_stack = QStackedWidget(page)
    self.second_stage_stack.setObjectName("secondStageStack")
    self.second_stage_stack.addWidget(self._build_second_stage_labeling_step(page))
    root_layout.addWidget(self.second_stage_stack, 1)

    self._set_second_stage_step(0)
    self._reset_second_stage_progress()
    return page


def _build_second_stage_labeling_step_page(self: Any, parent: QWidget) -> QWidget:
    """Build the interactive labeling step UI."""
    if NewObjectLabelingWidget is not None:
        self.newObjectLabelingWidget = NewObjectLabelingWidget(parent)
        self.newObjectLabelingWidget.statusChanged.connect(
            lambda text: self.labelSecondStageStatus.setText(f"상태: {text}")
            if self.labelSecondStageStatus is not None
            else None
        )
        if self.labelSecondStageStatus is not None:
            self.labelSecondStageStatus.setText("상태: 라벨링 완료 대기")
        return self.newObjectLabelingWidget

    page = QWidget(parent)
    root_layout = QHBoxLayout(page)
    root_layout.setContentsMargins(0, 0, 0, 0)
    root_layout.setSpacing(14)

    left_col = QVBoxLayout()
    left_col.setContentsMargins(0, 0, 0, 0)
    left_col.setSpacing(14)

    video_card = QFrame(page)
    video_card.setProperty("pageCard", True)
    video_layout = QVBoxLayout(video_card)
    video_layout.setContentsMargins(18, 16, 18, 16)
    video_layout.setSpacing(10)

    title = QLabel("대화형 객체 프롬프팅 (사무라이 + 에스에이엠2)", video_card)
    title.setObjectName("labelPageCardTitle")
    video_layout.addWidget(title)

    self.labelSecondVideoPath = QLabel("영상: -", video_card)
    self.labelSecondVideoPath.setWordWrap(True)
    self.labelSecondVideoMeta = QLabel("해상도: - | 초당 프레임: - | 프레임: -", video_card)
    video_layout.addWidget(self.labelSecondVideoPath)
    video_layout.addWidget(self.labelSecondVideoMeta)

    preview_frame = QFrame(video_card)
    preview_frame.setObjectName("secondVideoView")
    preview_layout = QVBoxLayout(preview_frame)
    preview_layout.setContentsMargins(0, 0, 0, 0)
    preview_layout.setSpacing(0)
    self.labelSecondVideoPreview = QLabel("2차 단계 영상 미리보기", preview_frame)
    self.labelSecondVideoPreview.setObjectName("labelSecondVideoPlaceholder")
    self.labelSecondVideoPreview.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self.labelSecondVideoPreview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
    self.labelSecondVideoPreview.setMinimumSize(0, 0)
    preview_layout.addWidget(self.labelSecondVideoPreview)
    video_layout.addWidget(preview_frame, 1)

    timeline_row = QHBoxLayout()
    timeline_row.setSpacing(8)
    self.labelSecondTimeline = QLabel("00:00:00.0 / 00:00:00.0", video_card)
    self.sliderSecondTimeline = QSlider(Qt.Orientation.Horizontal, video_card)
    self.sliderSecondTimeline.setRange(0, 0)
    self.sliderSecondTimeline.valueChanged.connect(self._on_second_timeline_changed)
    timeline_row.addWidget(self.labelSecondTimeline)
    timeline_row.addWidget(self.sliderSecondTimeline, 1)
    video_layout.addLayout(timeline_row)

    action_row = QHBoxLayout()
    action_row.setSpacing(8)
    btn_open = QPushButton("영상 열기", video_card)
    btn_open.clicked.connect(self.on_open_video)
    btn_pos = QPushButton("양성 포인트", video_card)
    btn_pos.clicked.connect(
        lambda: self._show_placeholder_dialog(
            "프롬프트 모드",
            "양성 클릭 프롬프트 화면 연결 지점입니다.",
        )
    )
    btn_neg = QPushButton("음성 포인트", video_card)
    btn_neg.clicked.connect(
        lambda: self._show_placeholder_dialog(
            "프롬프트 모드",
            "음성 클릭 프롬프트 화면 연결 지점입니다.",
        )
    )
    btn_object = QPushButton("객체 생성", video_card)
    btn_object.clicked.connect(
        lambda: self._show_placeholder_dialog(
            "객체 생성",
            "사무라이/에스에이엠2 객체 제안 및 확정 흐름 연결 지점입니다.",
        )
    )
    for button in [btn_open, btn_pos, btn_neg, btn_object]:
        action_row.addWidget(button)
    action_row.addStretch(1)
    video_layout.addLayout(action_row)
    left_col.addWidget(video_card, 1)

    prompt_card = QFrame(page)
    prompt_card.setProperty("pageCard", True)
    prompt_layout = QVBoxLayout(prompt_card)
    prompt_layout.setContentsMargins(18, 16, 18, 16)
    prompt_layout.setSpacing(8)
    prompt_title = QLabel("프롬프트 세션 큐", prompt_card)
    prompt_title.setObjectName("labelPageCardTitle")
    prompt_layout.addWidget(prompt_title)
    prompt_list = QListWidget(prompt_card)
    prompt_list.addItems(
        [
            "트랙 #12 - 클릭 프롬프트: 4 (저장 준비)",
            "트랙 #21 - 클릭 프롬프트: 2 (추가 보정 필요)",
            "트랙 #35 - 박스 프롬프트: 1 (신규 객체 후보)",
        ]
    )
    prompt_layout.addWidget(prompt_list)
    left_col.addWidget(prompt_card, 1)

    right_col = QVBoxLayout()
    right_col.setContentsMargins(0, 0, 0, 0)
    right_col.setSpacing(14)

    config_card = QFrame(page)
    config_card.setProperty("pageCard", True)
    config_layout = QVBoxLayout(config_card)
    config_layout.setContentsMargins(18, 16, 18, 16)
    config_layout.setSpacing(8)
    config_title = QLabel("분할 엔진 설정", config_card)
    config_title.setObjectName("labelPageCardTitle")
    config_layout.addWidget(config_title)

    model_form = QFormLayout()
    model_form.setHorizontalSpacing(10)
    model_form.setVerticalSpacing(7)
    combo_engine = QComboBox(config_card)
    combo_engine.addItems(
        [
            "사무라이 + 에스에이엠2 혼합",
            "사무라이 우선, 에스에이엠2 정밀 보정",
            "에스에이엠2 우선, 사무라이 검증",
        ]
    )
    combo_click_policy = QComboBox(config_card)
    combo_click_policy.addItems(["단일 클릭 시드", "다중 클릭 정밀화", "박스 + 클릭 혼합"])
    combo_save_format = QComboBox(config_card)
    combo_save_format.addItems(["검출 형식", "분할 형식", "제이슨 형식"])
    model_form.addRow("추론 전략", combo_engine)
    model_form.addRow("프롬프트 정책", combo_click_policy)
    model_form.addRow("자동 저장 형식", combo_save_format)
    config_layout.addLayout(model_form)

    flags_row = QHBoxLayout()
    flags_row.setSpacing(10)
    flags_row.addWidget(QCheckBox("다음 프레임 자동 추적", config_card))
    flags_row.addWidget(QCheckBox("마스크 + 박스 저장", config_card))
    flags_row.addWidget(QCheckBox("불확실 샘플 유지", config_card))
    flags_row.addStretch(1)
    config_layout.addLayout(flags_row)
    right_col.addWidget(config_card)

    finalize_card = QFrame(page)
    finalize_card.setProperty("pageCard", True)
    finalize_layout = QVBoxLayout(finalize_card)
    finalize_layout.setContentsMargins(18, 16, 18, 16)
    finalize_layout.setSpacing(8)
    finalize_title = QLabel("라벨링 마무리", finalize_card)
    finalize_title.setObjectName("labelPageCardTitle")
    finalize_layout.addWidget(finalize_title)

    finalize_form = QFormLayout()
    finalize_form.setHorizontalSpacing(10)
    finalize_form.setVerticalSpacing(8)
    self.editSecondStageClassName = QLineEdit(finalize_card)
    self.editSecondStageClassName.setPlaceholderText("클래스 이름 입력 (예: 지게차)")
    finalize_form.addRow("클래스 이름", self.editSecondStageClassName)
    finalize_layout.addLayout(finalize_form)

    btn_done = QPushButton("라벨링 완료", finalize_card)
    btn_done.setObjectName("btnSecondStageDone")
    btn_done.clicked.connect(self._on_second_stage_labeling_complete)
    finalize_layout.addWidget(btn_done)

    hint = QLabel(
        "라벨링이 완료되면 -> 버튼이 나타나고 2단계(다중 모델 학습)로 이동할 수 있습니다.",
        finalize_card,
    )
    hint.setWordWrap(True)
    hint.setStyleSheet("color: rgb(113, 126, 149); font-size: 9pt;")
    finalize_layout.addWidget(hint)
    right_col.addWidget(finalize_card)
    right_col.addStretch(1)

    root_layout.addLayout(left_col, 3)
    root_layout.addLayout(right_col, 2)
    return page


def _on_second_stage_labeling_complete_page(self: Any) -> None:
    """Handle completion of the second-stage interactive labeling step."""
    if self.video_path is None:
        QMessageBox.information(self, "1단계", "먼저 영상을 불러오세요.")
        return

    class_name = ""
    if self.editSecondStageClassName is not None:
        class_name = self.editSecondStageClassName.text().strip()
    if not class_name:
        QMessageBox.information(self, "1단계", "라벨링 완료 전에 클래스 이름을 입력하세요.")
        return

    self.second_stage_labeling_done = True
    if self.labelSecondStageStatus is not None:
        self.labelSecondStageStatus.setText(f"상태: 라벨링 완료 ({class_name})")
    self._append_log(f"2차 고도화 1단계 완료: 클래스={class_name}")


def _set_second_stage_step_page(self: Any, index: int) -> None:
    """Sync the current second-stage step and stack index."""
    if self.second_stage_stack is None:
        return
    target = max(0, min(index, self.second_stage_stack.count() - 1))
    self.second_stage_stack.setCurrentIndex(target)


def _reset_second_stage_progress_page(self: Any) -> None:
    """Reset the second-stage progress UI to its initial state."""
    self.second_stage_labeling_done = False
    if self.labelSecondStageStatus is not None:
        self.labelSecondStageStatus.setText("상태: 라벨링 완료 대기")
    if getattr(self, "newObjectLabelingWidget", None) is not None:
        self._set_second_stage_step(0)
        return
    if self.editSecondStageClassName is not None:
        self.editSecondStageClassName.clear()
    self._set_second_stage_step(0)


def _on_second_timeline_changed_page(self: Any, value: int) -> None:
    """Handle timeline scrubbing in the second-stage preview."""
    if self.is_playing:
        return

    if self.video_cap is None or (self.is_processing and (not self.is_processing_paused)):
        return
    frame_index = max(0, int(value))
    self.processing_next_start_frame = frame_index
    if self.is_processing_paused and self.worker is not None:
        self.worker.request_seek(frame_index)
    frame = self._read_video_frame(frame_index)
    if frame is None:
        return
    self._show_frame(frame, [])
    self._update_time_label(frame_index)
    if not self.video_load_hover_active:
        self._set_video_load_preview_frame(frame)
    self.last_second_preview_frame = frame
    if self.labelSecondVideoPreview is not None:
        pixmap = self._frame_to_pixmap(frame, max_width=960, max_height=560)
        self.labelSecondVideoPreview.setPixmap(pixmap)
        self.labelSecondVideoPreview.setText("")

    if self.sliderTimeline is not None and self.sliderTimeline.value() != frame_index:
        self.sliderTimeline.blockSignals(True)
        self.sliderTimeline.setValue(frame_index)
        self.sliderTimeline.blockSignals(False)
    self._update_second_timeline_label(frame_index)


def _update_second_timeline_label_page(self: Any, frame_index: int) -> None:
    """Update the formatted timeline label for the second-stage preview."""
    if self.labelSecondTimeline is None:
        return
    fps = float(self.video_meta.get("fps", 0.0))
    total_frames = int(self.video_meta.get("frame_count", 0))
    if fps <= 0 or total_frames <= 0:
        self.labelSecondTimeline.setText("00:00:00.0 / 00:00:00.0")
        return

    current_sec = max(0.0, float(frame_index) / fps)
    total_sec = max(0.0, float(total_frames - 1) / fps)
    self.labelSecondTimeline.setText(f"{self._format_hms(current_sec)} / {self._format_hms(total_sec)}")


def _refresh_second_advanced_page_page(self: Any) -> None:
    """Refresh the second-stage page based on the current video state."""
    widget = getattr(self, "newObjectLabelingWidget", None)
    if widget is not None and hasattr(widget, "refresh_view"):
        # 영상 입력 페이지에서 선택된 영상/ROI를 위젯에 자동 주입
        if self.video_path is not None and hasattr(widget, "load_video_from_path"):
            roi = getattr(self, "video_roi", None)
            widget.load_video_from_path(str(self.video_path), roi)
        widget.refresh_view()
        return

    if (
        self.labelSecondVideoPath is None
        or self.labelSecondVideoMeta is None
        or self.labelSecondVideoPreview is None
        or self.sliderSecondTimeline is None
    ):
        return

    if self.video_path is None:
        self.labelSecondVideoPath.setText("영상: -")
        self.labelSecondVideoMeta.setText("해상도: - | 초당 프레임: - | 프레임: -")
        self.sliderSecondTimeline.blockSignals(True)
        self.sliderSecondTimeline.setRange(0, 0)
        self.sliderSecondTimeline.setValue(0)
        self.sliderSecondTimeline.blockSignals(False)
        self.labelSecondVideoPreview.clear()
        self.labelSecondVideoPreview.setText("2차 단계 영상 미리보기")
        self._update_second_timeline_label(0)
        return

    width = int(self.video_meta.get("width", 0))
    height = int(self.video_meta.get("height", 0))
    fps = float(self.video_meta.get("fps", 0.0))
    frame_count = int(self.video_meta.get("frame_count", 0))
    self.labelSecondVideoPath.setText(f"영상: {self.video_path}")
    self.labelSecondVideoMeta.setText(f"해상도: {width}x{height} | 초당 프레임: {fps:.2f} | 프레임: {frame_count}")

    self.sliderSecondTimeline.blockSignals(True)
    self.sliderSecondTimeline.setRange(0, max(0, frame_count - 1))

    sync_index = 0
    if self.sliderTimeline is not None:
        sync_index = int(self.sliderTimeline.value())
    sync_index = max(0, min(max(0, frame_count - 1), sync_index))
    self.sliderSecondTimeline.setValue(sync_index)
    self.sliderSecondTimeline.blockSignals(False)

    frame = self._read_video_frame(sync_index)
    if frame is not None:
        self.last_second_preview_frame = frame
        self.labelSecondVideoPreview.setPixmap(self._frame_to_pixmap(frame, max_width=960, max_height=560))
        self.labelSecondVideoPreview.setText("")
    else:
        self.labelSecondVideoPreview.clear()
        self.labelSecondVideoPreview.setText("2차 단계 영상 미리보기")
    self._update_second_timeline_label(sync_index)
