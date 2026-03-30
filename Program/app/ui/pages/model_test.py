from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def _build_model_test_page_page(self: Any) -> QWidget:
    """학습된 모델(best.pt)로 신규 영상 테스트 추론을 실행하는 페이지를 구성합니다."""
    page = QWidget(self)
    root_layout = QHBoxLayout(page)
    root_layout.setContentsMargins(0, 0, 0, 0)
    root_layout.setSpacing(14)

    left_col = QWidget(page)
    left_layout = QVBoxLayout(left_col)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(14)

    control_card = QFrame(left_col)
    control_card.setProperty("pageCard", True)
    control_layout = QVBoxLayout(control_card)
    control_layout.setContentsMargins(18, 16, 18, 16)
    control_layout.setSpacing(8)

    title = QLabel("모델 테스트 (best.pt 추론)", control_card)
    title.setObjectName("labelPageCardTitle")
    control_layout.addWidget(title)

    self.labelModelTestStatus = QLabel("상태: 대기", control_card)
    control_layout.addWidget(self.labelModelTestStatus)

    model_row = QHBoxLayout()
    self.editModelTestModel = QLineEdit(control_card)
    self.editModelTestModel.setReadOnly(True)
    self.editModelTestModel.setPlaceholderText("테스트 모델(.pt) 경로")
    self.btnModelTestPickModel = QPushButton("모델 선택", control_card)
    self.btnModelTestPickModel.clicked.connect(self._on_pick_model_test_model)
    model_row.addWidget(self.editModelTestModel, 1)
    model_row.addWidget(self.btnModelTestPickModel, 0)
    control_layout.addLayout(model_row)

    video_row = QHBoxLayout()
    self.editModelTestVideo = QLineEdit(control_card)
    self.editModelTestVideo.setReadOnly(True)
    self.editModelTestVideo.setPlaceholderText("테스트 영상 경로")
    self.btnModelTestPickVideo = QPushButton("영상 선택", control_card)
    self.btnModelTestPickVideo.clicked.connect(self._on_pick_model_test_video)
    video_row.addWidget(self.editModelTestVideo, 1)
    video_row.addWidget(self.btnModelTestPickVideo, 0)
    control_layout.addLayout(video_row)

    option_row = QHBoxLayout()
    option_row.setSpacing(8)
    self.spinModelTestConf = QDoubleSpinBox(control_card)
    self.spinModelTestConf.setRange(0.01, 0.99)
    self.spinModelTestConf.setSingleStep(0.01)
    self.spinModelTestConf.setValue(0.25)
    self.spinModelTestConf.setPrefix("conf ")
    self.spinModelTestIou = QDoubleSpinBox(control_card)
    self.spinModelTestIou.setRange(0.05, 0.99)
    self.spinModelTestIou.setSingleStep(0.01)
    self.spinModelTestIou.setValue(0.45)
    self.spinModelTestIou.setPrefix("iou ")
    self.spinModelTestImgsz = QSpinBox(control_card)
    self.spinModelTestImgsz.setRange(320, 1920)
    self.spinModelTestImgsz.setSingleStep(32)
    self.spinModelTestImgsz.setValue(960)
    self.spinModelTestImgsz.setPrefix("imgsz ")
    option_row.addWidget(self.spinModelTestConf)
    option_row.addWidget(self.spinModelTestIou)
    option_row.addWidget(self.spinModelTestImgsz)
    option_row.addStretch(1)
    control_layout.addLayout(option_row)

    progress_row = QHBoxLayout()
    self.progressModelTest = QProgressBar(control_card)
    self.progressModelTest.setRange(0, 100)
    self.progressModelTest.setValue(0)
    self.progressModelTest.setFormat("추론 진행률 %p%")
    self.labelModelTestFrame = QLabel("frame: - / -", control_card)
    progress_row.addWidget(self.progressModelTest, 1)
    progress_row.addWidget(self.labelModelTestFrame, 0)
    control_layout.addLayout(progress_row)

    action_row = QHBoxLayout()
    self.btnModelTestStart = QPushButton("테스트 시작", control_card)
    self.btnModelTestStart.clicked.connect(self.on_start_model_test)
    self.btnModelTestStop = QPushButton("테스트 중지", control_card)
    self.btnModelTestStop.clicked.connect(self.on_stop_model_test)
    action_row.addWidget(self.btnModelTestStart)
    action_row.addWidget(self.btnModelTestStop)
    action_row.addStretch(1)
    control_layout.addLayout(action_row)
    left_layout.addWidget(control_card, 0)

    log_card = QFrame(left_col)
    log_card.setProperty("pageCard", True)
    log_layout = QVBoxLayout(log_card)
    log_layout.setContentsMargins(18, 16, 18, 16)
    log_layout.setSpacing(8)
    log_title = QLabel("테스트 로그", log_card)
    log_title.setObjectName("labelPageCardTitle")
    log_layout.addWidget(log_title)
    self.textModelTestLog = QTextEdit(log_card)
    self.textModelTestLog.setReadOnly(True)
    self.textModelTestLog.setPlainText("[안내] 모델과 영상을 선택한 뒤 테스트를 시작하세요.")
    log_layout.addWidget(self.textModelTestLog, 1)
    left_layout.addWidget(log_card, 1)

    preview_card = QFrame(page)
    preview_card.setProperty("pageCard", True)
    preview_layout = QVBoxLayout(preview_card)
    preview_layout.setContentsMargins(12, 12, 12, 12)
    preview_layout.setSpacing(8)
    preview_title = QLabel("추론 미리보기", preview_card)
    preview_title.setObjectName("labelPageCardTitle")
    preview_layout.addWidget(preview_title)

    self.labelModelTestPreview = QLabel("모델 테스트 프리뷰", preview_card)
    self.labelModelTestPreview.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self.labelModelTestPreview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
    self.labelModelTestPreview.setMinimumSize(0, 0)
    preview_layout.addWidget(self.labelModelTestPreview, 1)

    root_layout.addWidget(left_col, 2)
    root_layout.addWidget(preview_card, 3)

    self._update_model_test_ui_state()
    return page
