from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
from PyQt6.QtCore import QEvent, QObject, QSettings, QSize, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QKeyEvent, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,

    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from backend.llm.client import LLMApiClient
from backend.labeling.artifacts import prepare_output_dir
from backend.labeling.sam3_runner import (
    RemoteSam3Request,
    RemoteSam3Worker,
    _clear_preview_cache_root,
    _draw_preview_box,
    _persist_preview_items_to_cache,
    _preview_cache_root_for_experiment,
    summarize_preview_items,
)
from backend.llm.prompting import (
    build_display_payload,
    build_sam_prompt_candidates,
    extract_ranked_prompt_candidates,
    format_nlp_output_for_display,
    heuristic_english_candidates,
    normalize_user_text,
)
from core.models import PreviewThumbnail
from app.widgets.result_thumbnail_browser import ResultThumbnailBrowser
from core.paths import DATASET_SAVE_DIR
from app.studio.utils import _build_unique_path, build_dataset_folder_name
from app.studio.workers import DatasetExportWorker
from app.ui.widgets.studio_support import FuturisticSpinner
VIDEO_EXTENSIONS = "*.mp4 *.avi *.mov *.mkv *.wmv"


@dataclass(slots=True)
class VideoMeta:
    filename: str
    width: int
    height: int
    fps: float
    totalFrames: int
    durationSec: float


@dataclass(slots=True)
class SamRunConfig:
    className: str
    imagePath: str
    videoPath: str
    experimentId: str
    promptMode: str = "text"
    promptText: str = ""


@dataclass(slots=True)
class SamProgress:
    framesProcessed: int
    newSamples: int
    numImages: int
    numLabels: int
    message: str
    level: str


class NlpWarmupWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, client: LLMApiClient) -> None:
        super().__init__()
        self._client = client

    def run(self) -> None:
        try:
            self.finished.emit(self._client.warmup())
        except Exception as exc:
            self.failed.emit(str(exc))


class NlpRunWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, client: LLMApiClient, user_text: str, class_name: str, n: int = 3) -> None:
        super().__init__()
        self._client = client
        self._user_text = str(user_text)
        self._class_name = str(class_name)
        self._top_n = max(1, int(n))

    def run(self) -> None:
        try:
            response = self._client.rank_prompts(user_text=self._user_text, n=self._top_n, debug=False)
            self.finished.emit(
                {
                    "user_text": self._user_text,
                    "class_name": self._class_name,
                    "response": response,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class PromptTextEdit(QPlainTextEdit):
    submitRequested = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        enter = event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        modified = bool(
            event.modifiers()
            & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)
        )
        if enter and (not modified):
            self.submitRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class AspectRatioFrame(QFrame):
    def __init__(
        self,
        ratio_width: int,
        ratio_height: int,
        min_height: int,
        max_height: int,
        default_width: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._ratio_width = max(1, int(ratio_width))
        self._ratio_height = max(1, int(ratio_height))
        self._min_height = max(1, int(min_height))
        self._max_height = max(self._min_height, int(max_height))
        self._default_width = max(1, int(default_width))
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setMinimumWidth(0)
        self.setMinimumHeight(self._min_height)
        self.setMaximumHeight(self._max_height)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        scaled = int(round(max(1, width) * self._ratio_height / self._ratio_width))
        return max(self._min_height, min(scaled, self._max_height))

    def sizeHint(self) -> QSize:
        width = self.width() if self.width() > 0 else self._default_width
        return QSize(max(0, width), self.heightForWidth(width))

    def minimumSizeHint(self) -> QSize:
        return QSize(0, self._min_height)

def _load_preview_items_from_cache(cache_root: Path) -> list[PreviewThumbnail]:
    loaded: list[PreviewThumbnail] = []
    for category in ("keep", "hold", "drop"):
        category_dir = cache_root / category
        if not category_dir.is_dir():
            continue
        for manifest_path in sorted(category_dir.glob("*.json")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            frame_index = int(payload.get("frame_index", -1))
            boxes = payload.get("boxes")
            if frame_index < 0 or not isinstance(boxes, list):
                continue
            normalized_boxes = [dict(box) for box in boxes if isinstance(box, Mapping)]
            if not normalized_boxes:
                continue
            base_category = str(payload.get("category", category)).strip().lower() or category
            base_item_id = str(payload.get("item_id", manifest_path.stem)).strip() or manifest_path.stem
            image_path = str(payload.get("image_path", "") or "").strip() or None
            thumb_path = str(payload.get("thumb_path", "") or "").strip() or None
            for box_index, box in enumerate(normalized_boxes, start=1):
                item_category = str(box.get("status", base_category)).strip().lower() or base_category
                if item_category not in {"keep", "hold", "drop"}:
                    item_category = base_category if base_category in {"keep", "hold", "drop"} else category
                item_id = str(box.get("preview_item_id", "")).strip()
                if not item_id:
                    item_id = base_item_id if len(normalized_boxes) == 1 else f"{base_item_id}_{box_index:02d}"
                    box["preview_item_id"] = item_id
                loaded.append(
                    PreviewThumbnail(
                        frame_index=frame_index,
                        image=None,
                        boxes=[box],
                        category=item_category,
                        item_id=item_id,
                        image_path=image_path,
                        thumb_path=thumb_path,
                        manifest_path=str(manifest_path),
                    )
                )
    loaded.sort(
        key=lambda item: (
            int(item.boxes[0].get("track_id"))
            if item.boxes and isinstance(item.boxes[0], Mapping) and item.boxes[0].get("track_id") is not None
            else 2_147_483_647,
            int(item.frame_index),
            str(item.item_id),
        )
    )
    return loaded


class NewObjectLabelingWidget(QWidget):
    statusChanged = pyqtSignal(str)

    VIDEO_ASPECT_WIDTH = 16
    VIDEO_ASPECT_HEIGHT = 9
    VIDEO_MIN_HEIGHT = 380
    VIDEO_MAX_HEIGHT = 520
    ACTION_W = 118
    ACTION_H = 40
    THUMB = 72

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.video_path: str | None = None
        self.videoMeta = VideoMeta(filename="-", width=0, height=0, fps=0.0, totalFrames=0, durationSec=0.0)
        self.currentFrame = 0
        self.timeline = {"currentTimeSec": 0.0, "totalTimeSec": 0.0}
        self.className = ""
        self.promptText = ""
        self.promptCandidates: list[str] = []
        self.isPlaying = False
        self.isRunning = False
        self.outputDir = "-"
        self.numImages = 0
        self.numLabels = 0
        self.framesProcessed = 0
        self.newSamples = 0
        self.logs: list[str] = []
        self.experimentId = "-"
        self.modelsText = "SAM3 Text Tracker"
        self._video_pixmap: QPixmap | None = None
        self._current_video_frame: Any | None = None
        self._video_capture: cv2.VideoCapture | None = None
        self._display_video_path: str | None = None
        self._run_config: SamRunConfig | None = None
        self._sam3_thread: QThread | None = None
        self._sam3_worker: RemoteSam3Worker | None = None
        self._export_thread: QThread | None = None
        self._export_worker: DatasetExportWorker | None = None
        self._isExporting = False
        self.lastExportDatasetRoot: Path | None = None
        self._nlp_warmup_thread: QThread | None = None
        self._nlp_warmup_worker: NlpWarmupWorker | None = None
        self._nlp_run_thread: QThread | None = None
        self._nlp_run_worker: NlpRunWorker | None = None
        self._nlpWarmupRunning = False
        self._nlpWarmupDone = False
        self._nlpRunRunning = False
        self._previewCacheRoot: Path | None = None
        self.previewItems: list[PreviewThumbnail] = []
        self._bbox_overlay: dict[int, list[dict]] = {}
        self._lastSamProgressLogFrame = -1
        self._dividers: list[QFrame] = []
        self.loadingOverlay: QWidget | None = None
        self.loadingPanel: QFrame | None = None
        self.loadingSpinner: FuturisticSpinner | None = None
        self.loadingLabel: QLabel | None = None
        self.loadingSubLabel: QLabel | None = None
        self.llm_client = LLMApiClient()

        self.playbackTimer = QTimer(self)
        self.playbackTimer.timeout.connect(self._advance_playback)

        self._build_ui()
        self._connect_signals()
        self.refresh_theme()
        self._update_meta_labels()
        self._update_summary_labels()
        self.promptCandidates = build_sam_prompt_candidates(
            prompt_text=self.promptText,
            class_name=self.inputClassName.text().strip(),
        )
        self._update_button_state()
        QTimer.singleShot(0, self._sync_responsive_layouts)
        QTimer.singleShot(150, self._start_nlp_warmup)
        self.statusChanged.emit("샘플링 대기")

    def _build_ui(self) -> None:
        page = QVBoxLayout(self)
        page.setContentsMargins(14, 14, 14, 14)
        page.setSpacing(0)

        self.videoCard = self._make_card("newObjectVideoCard")
        self.videoCard.setMinimumWidth(0)
        self.videoCard.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        vbox = QVBoxLayout(self.videoCard)
        vbox.setContentsMargins(18, 18, 18, 18)
        vbox.setSpacing(12)
        self.videoPreviewBox = AspectRatioFrame(
            self.VIDEO_ASPECT_WIDTH,
            self.VIDEO_ASPECT_HEIGHT,
            self.VIDEO_MIN_HEIGHT,
            self.VIDEO_MAX_HEIGHT,
            960,
            self.videoCard,
        )
        self.videoPreviewBox.setObjectName("videoPreviewBox")
        self.videoPreviewBox.setCursor(Qt.CursorShape.PointingHandCursor)
        preview_layout = QVBoxLayout(self.videoPreviewBox)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        self.videoPreview = QLabel("영상 로드", self.videoPreviewBox)
        self.videoPreview.setObjectName("videoPreview")
        self.videoPreview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.videoPreview.setMinimumSize(0, 0)
        self.videoPreview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.videoPreview.installEventFilter(self)
        self.videoPreviewBox.installEventFilter(self)
        preview_layout.addWidget(self.videoPreview)
        self._build_loading_overlay()
        vbox.addWidget(self.videoPreviewBox, 0)
        timeline = QHBoxLayout()
        self.lblTime = QLabel("00:00:00.0 / 00:00:00.0", self.videoCard)
        self.lblTime.setObjectName("labelTime")
        self.sliderTimeline = QSlider(Qt.Orientation.Horizontal, self.videoCard)
        self.sliderTimeline.setObjectName("sliderTimeline")
        self.sliderTimeline.setRange(0, 0)
        timeline.addWidget(self.lblTime)
        timeline.addWidget(self.sliderTimeline, 1)
        vbox.addLayout(timeline)
        controls = QHBoxLayout()
        controls.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        controls.setSpacing(10)
        self.btnSeekBack = self._make_tool("[]", "정지")
        self.btnPrevFrame = self._make_tool("|<", "이전 프레임")
        self.btnPlayPause = self._make_tool(">", "재생")
        self.btnNextFrame = self._make_tool(">|", "다음 프레임")
        self.btnPrev = self.btnPrevFrame
        self.btnNext = self.btnNextFrame
        for button in (self.btnSeekBack, self.btnPrevFrame, self.btnPlayPause, self.btnNextFrame):
            controls.addWidget(button)
        vbox.addLayout(controls)

        self.promptCard = self._make_card("newObjectPromptCard")
        self.promptCard.setMinimumWidth(0)
        self.promptCard.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        pbox = QVBoxLayout(self.promptCard)
        pbox.setContentsMargins(14, 14, 14, 14)
        pbox.setSpacing(8)
        title = QLabel("프롬프트", self.promptCard)
        title.setObjectName("labelNewObjectSectionTitle")
        pbox.addWidget(title)

        self.labelTextPromptHint = QLabel("텍스트로 찾고 싶은 객체를 설명하세요.", self.promptCard)
        self.labelTextPromptHint.setProperty("promptHelper", True)
        self.labelTextPromptHint.setWordWrap(True)
        pbox.addWidget(self.labelTextPromptHint)

        self.promptInputFrame = QFrame(self.promptCard)
        self.promptInputFrame.setObjectName("promptInputFrame")
        self.promptInputFrame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        prompt_frame = QVBoxLayout(self.promptInputFrame)
        prompt_frame.setContentsMargins(10, 8, 10, 8)
        prompt_frame.setSpacing(6)
        self.promptInput = PromptTextEdit(self.promptInputFrame)
        self.promptInput.setObjectName("promptTextInput")
        self.promptInput.setPlaceholderText("찾을 객체를 설명해주세요. (예: 하얀색 승용차)")
        self.promptInput.setMinimumHeight(44)
        self.promptInput.setMaximumHeight(52)
        self.promptInput.setTabChangesFocus(True)
        prompt_frame.addWidget(self.promptInput, 0)
        self.labelNlpStatus = QLabel("NLP 상태: 대기", self.promptInputFrame)
        self.labelNlpStatus.setObjectName("labelNlpStatus")
        self.labelNlpStatus.setProperty("promptHelper", True)
        prompt_frame.addWidget(self.labelNlpStatus)

        nlp_actions = QHBoxLayout()
        nlp_actions.setContentsMargins(0, 0, 0, 0)
        nlp_actions.setSpacing(8)
        self.btnNlpHealth = QPushButton("Health", self.promptInputFrame)
        self.btnNlpHealth.setObjectName("btnNlpHealth")
        self.btnNlpRun = QPushButton("NLP 실행", self.promptInputFrame)
        self.btnNlpRun.setObjectName("btnNlpRun")
        self.btnNlpClear = QPushButton("결과 지우기", self.promptInputFrame)
        self.btnNlpClear.setObjectName("btnNlpClear")
        nlp_actions.addWidget(self.btnNlpHealth)
        nlp_actions.addWidget(self.btnNlpRun)
        nlp_actions.addWidget(self.btnNlpClear)
        nlp_actions.addStretch(1)
        prompt_frame.addLayout(nlp_actions)

        self.nlpOutput = QPlainTextEdit(self.promptInputFrame)
        self.nlpOutput.setObjectName("nlpOutput")
        self.nlpOutput.setReadOnly(True)
        self.nlpOutput.setPlaceholderText("NLP 결과가 여기에 표시됩니다.")
        self.nlpOutput.setMinimumHeight(96)
        self.nlpOutput.setMaximumHeight(140)
        prompt_frame.addWidget(self.nlpOutput, 0)
        pbox.addWidget(self.promptInputFrame, 0)

        cls = QHBoxLayout()
        cls.setSpacing(10)
        self.classLabel = QLabel("클래스 이름 설정", self.promptCard)
        self.classLabel.setObjectName("labelPromptInputCaption")
        self.classLabel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.inputClassName = QLineEdit(self.promptCard)
        self.inputClassName.setObjectName("inputClassName")
        self.inputClassName.setPlaceholderText("예: cone")
        self.inputClassName.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        cls.addWidget(self.classLabel)
        cls.addWidget(self.inputClassName, 1)
        pbox.addLayout(cls)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(10)
        actions.addStretch(1)
        self.btnRun = QPushButton("실행", self.promptCard)
        self.btnRun.setObjectName("btnRun")
        self.btnStop = QPushButton("중지", self.promptCard)
        self.btnStop.setObjectName("btnStop")
        for button in (self.btnRun, self.btnStop):
            button.setFixedSize(self.ACTION_W, self.ACTION_H)
            actions.addWidget(button)
        pbox.addLayout(actions)

        self.resultCard = QFrame(self)
        self.resultCard.setProperty("pageCard", True)
        self.resultCard.setObjectName("newObjectBottomLogSection")
        self.resultCard.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        rbox = QVBoxLayout(self.resultCard)
        rbox.setContentsMargins(18, 18, 18, 18)
        rbox.setSpacing(12)
        result_header = QHBoxLayout()
        result_header.setContentsMargins(0, 0, 0, 0)
        result_header.setSpacing(8)
        result_title = QLabel("로그", self.resultCard)
        result_title.setObjectName("labelResultLogTitle")
        result_header.addWidget(result_title)
        result_header.addStretch(1)
        rbox.addLayout(result_header)
        rbox.addWidget(self._divider(self.resultCard))
        self.logArea = QPlainTextEdit(self.resultCard)
        self.logArea.setObjectName("logArea")
        self.logArea.setReadOnly(True)
        self.logArea.setMinimumHeight(80)
        rbox.addWidget(self.logArea, 1)

        self.bottomPanelDivider = QFrame(self)
        self.bottomPanelDivider.setObjectName("bottomPanelDivider")
        self._dividers.append(self.bottomPanelDivider)
        self._style_divider(self.bottomPanelDivider)

        self.resultBrowserCard = QFrame(self)
        self.resultBrowserCard.setProperty("pageCard", True)
        self.resultBrowserCard.setObjectName("newObjectBottomResultSection")
        self.resultBrowserCard.setMinimumWidth(0)
        self.resultBrowserCard.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        ebox = QVBoxLayout(self.resultBrowserCard)
        ebox.setContentsMargins(18, 18, 18, 18)
        ebox.setSpacing(8)
        self.resultBrowser = ResultThumbnailBrowser(
            self.resultBrowserCard,
            log_callback=self._append_log,
            busy_getter=lambda: self.isRunning,
        )
        self.resultBrowser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.btnExport = QPushButton("결과 내보내기", self.resultBrowserCard)
        self.btnExport.setObjectName("btnExport")
        self.btnExport.setFixedHeight(self.ACTION_H)
        self.resultBrowser.tab_row.addWidget(self.btnExport)
        ebox.addWidget(self.resultBrowser, 1)

        leftCol = QVBoxLayout()
        leftCol.setContentsMargins(0, 0, 0, 0)
        leftCol.setSpacing(12)
        leftCol.addWidget(self.videoCard, 3)
        leftCol.addWidget(self.resultCard, 1)

        rightCol = QVBoxLayout()
        rightCol.setContentsMargins(0, 0, 0, 0)
        rightCol.setSpacing(12)
        rightCol.addWidget(self.promptCard, 0)
        rightCol.addWidget(self.resultBrowserCard, 1)

        mainSplit = QHBoxLayout()
        mainSplit.setContentsMargins(0, 0, 0, 0)
        mainSplit.setSpacing(0)
        mainSplit.addLayout(leftCol, 6)
        mainSplit.addWidget(self.bottomPanelDivider)
        mainSplit.addLayout(rightCol, 4)
        page.addLayout(mainSplit, 1)

    def _connect_signals(self) -> None:
        self.inputClassName.textChanged.connect(self._on_class_name_changed)
        self.promptInput.textChanged.connect(self._on_prompt_text_changed)
        self.promptInput.submitRequested.connect(self.submit_prompt)
        self.btnNlpHealth.clicked.connect(self._on_nlp_health)
        self.btnNlpRun.clicked.connect(self._on_nlp_run)
        self.btnNlpClear.clicked.connect(self._on_nlp_clear)
        self.sliderTimeline.valueChanged.connect(self.timeline_changed)
        self.btnSeekBack.clicked.connect(self.on_stop_playback)
        self.btnPrevFrame.clicked.connect(lambda: self._seek_relative(-1))
        self.btnPlayPause.clicked.connect(self.toggle_play_pause)
        self.btnNextFrame.clicked.connect(lambda: self._seek_relative(1))
        self.btnRun.clicked.connect(self.run_process)
        self.btnStop.clicked.connect(self.stop_process)
        self.btnExport.clicked.connect(self.export_results)
        self.resultBrowser.countsChanged.connect(self._on_result_browser_counts_changed)

    def _build_loading_overlay(self) -> None:
        overlay = QWidget(self.videoPreviewBox)
        overlay.setObjectName("sam3LoadingOverlay")
        overlay.hide()
        overlay.setStyleSheet(
            """
QWidget#sam3LoadingOverlay {
    background: rgba(15, 23, 42, 188);
    border-radius: 14px;
}
QFrame#sam3LoadingPanel {
    background: rgba(255, 255, 255, 244);
    border: 1px solid rgba(191, 203, 224, 220);
    border-radius: 18px;
}
QLabel#sam3LoadingTitle {
    color: #0F172A;
    font: 900 15px "Pretendard";
}
QLabel#sam3LoadingSub {
    color: #475569;
    font: 600 10pt "Pretendard";
}
"""
        )
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setSpacing(0)
        overlay_layout.addStretch(1)

        panel = QFrame(overlay)
        panel.setObjectName("sam3LoadingPanel")
        panel.setFixedWidth(340)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 24, 24, 24)
        panel_layout.setSpacing(10)

        spinner = FuturisticSpinner(panel, size=72)
        spinner_row = QHBoxLayout()
        spinner_row.setContentsMargins(0, 0, 0, 0)
        spinner_row.addStretch(1)
        spinner_row.addWidget(spinner, 0, Qt.AlignmentFlag.AlignCenter)
        spinner_row.addStretch(1)
        panel_layout.addLayout(spinner_row)

        title = QLabel("원격 SAM3 추론 중", panel)
        title.setObjectName("sam3LoadingTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(title)

        sub_label = QLabel("A100 서버에서 객체를 추적하고 있습니다...", panel)
        sub_label.setObjectName("sam3LoadingSub")
        sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub_label.setWordWrap(True)
        panel_layout.addWidget(sub_label)

        overlay_layout.addWidget(panel, 0, Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addStretch(1)

        self.loadingOverlay = overlay
        self.loadingPanel = panel
        self.loadingSpinner = spinner
        self.loadingLabel = title
        self.loadingSubLabel = sub_label
        self._sync_loading_overlay()

    def _sync_loading_overlay(self) -> None:
        if self.loadingOverlay is None:
            return
        self.loadingOverlay.setGeometry(self.videoPreviewBox.rect())
        self.loadingOverlay.raise_()

    def _show_loading_overlay(self, title: str, sub_message: str = "") -> None:
        if self.loadingOverlay is None:
            return
        if self.loadingLabel is not None:
            self.loadingLabel.setText(str(title).strip() or "원격 SAM3 추론 중")
        if self.loadingSubLabel is not None:
            self.loadingSubLabel.setText(str(sub_message).strip() or "A100 서버에서 작업 중입니다...")
        self._sync_loading_overlay()
        self.loadingOverlay.show()
        if self.loadingSpinner is not None:
            self.loadingSpinner.start()

    def _update_loading_overlay(self, sub_message: str) -> None:
        if self.loadingOverlay is None or not self.loadingOverlay.isVisible():
            return
        if self.loadingSubLabel is not None:
            self.loadingSubLabel.setText(str(sub_message).strip() or "A100 서버에서 작업 중입니다...")

    def _hide_loading_overlay(self) -> None:
        if self.loadingSpinner is not None:
            self.loadingSpinner.stop()
        if self.loadingOverlay is not None:
            self.loadingOverlay.hide()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.MouseButtonRelease and obj in (self.videoPreview, self.videoPreviewBox):
            self.load_video()
            return True
        if event.type() == QEvent.Type.Resize and obj is self.videoPreview:
            self._render_video_preview()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._sync_responsive_layouts()
        self._sync_loading_overlay()

    def onLoadVideo(self) -> None: self.load_video()
    def onTimelineChange(self, value: int) -> None: self.timeline_changed(value)
    def onPlayPause(self) -> None: self.toggle_play_pause()
    def onRun(self) -> None: self.run_process()
    def onStop(self) -> None: self.stop_process()
    def onExport(self) -> None: self.export_results()

    def on_stop_playback(self) -> None:
        if self._video_capture is None:
            return
        self.playbackTimer.stop()
        self.isPlaying = False
        self.sliderTimeline.setValue(0)
        self._update_play_pause_button_icon()
        self._update_button_state()

    def _open_video_capture(
        self,
        path: str,
        *,
        preserve_source_video: bool,
        autoplay: bool,
        failure_log: str,
    ) -> bool:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            self._append_log(failure_log)
            return False
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            self._append_log("[ERROR] 첫 프레임을 읽지 못했습니다.")
            return False
        self.playbackTimer.stop()
        self.isPlaying = False
        self._release_video_capture()
        self._video_capture = capture
        self._display_video_path = path
        if not preserve_source_video:
            self.video_path = path
        self.videoMeta = VideoMeta(Path(path).name, width, height, fps, total, float(total) / max(1.0, fps))
        self.currentFrame = 0
        self.timeline.update({"currentTimeSec": 0.0, "totalTimeSec": self.videoMeta.durationSec})
        self.sliderTimeline.blockSignals(True)
        self.sliderTimeline.setRange(0, max(0, total - 1))
        self.sliderTimeline.setValue(0)
        self.sliderTimeline.blockSignals(False)
        self._set_video_frame(frame, 0)
        if autoplay:
            self.playbackTimer.start(max(33, int(round(1000.0 / max(1.0, self.videoMeta.fps)))))
            self.isPlaying = True
        self._update_play_pause_button_icon()
        self._update_button_state()
        return True

    def load_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "영상 선택", str(Path.home()), f"영상 파일 ({VIDEO_EXTENSIONS});;모든 파일 (*)")
        if not path:
            return
        if not self._open_video_capture(
            path,
            preserve_source_video=False,
            autoplay=False,
            failure_log="[ERROR] 영상을 불러오지 못했습니다.",
        ):
            return
        self._append_log(f"[INFO] video loaded: {self.videoMeta.filename}")
        self._update_button_state()
        return
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            self._append_log("[ERROR] 영상을 불러오지 못했습니다.")
            return
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            self._append_log("[ERROR] 첫 프레임을 읽지 못했습니다.")
            return
        self._release_video_capture()
        self._video_capture = capture
        self.video_path = path
        self.videoMeta = VideoMeta(Path(path).name, width, height, fps, total, float(total) / max(1.0, fps))
        self.currentFrame = 0
        self.timeline.update({"currentTimeSec": 0.0, "totalTimeSec": self.videoMeta.durationSec})
        self.sliderTimeline.blockSignals(True)
        self.sliderTimeline.setRange(0, max(0, total - 1))
        self.sliderTimeline.setValue(0)
        self.sliderTimeline.blockSignals(False)
        self._set_video_frame(frame, 0)
        self._append_log(f"[INFO] video loaded: {self.videoMeta.filename}")
        self._update_button_state()

    def timeline_changed(self, value: int) -> None:
        if self._video_capture is None:
            return
        index = max(0, min(int(value), self.videoMeta.totalFrames - 1))
        frame = self._read_frame(index)
        if frame is None:
            self._append_log("[WARN] 프레임 이동에 실패했습니다.")
            return
        self._set_video_frame(frame, index)

    def toggle_play_pause(self) -> None:
        if self.isRunning:
            self._append_log("[WARN] 추론 중에는 결과 영상 재생만 가능합니다.")
            return
        if self._video_capture is None:
            self._append_log("[WARN] 먼저 영상을 불러오세요.")
            return
        if self.isPlaying:
            self.playbackTimer.stop()
            self.isPlaying = False
            self._append_log("[INFO] 영상 재생을 일시정지했습니다.")
        else:
            self.playbackTimer.start(max(33, int(round(1000.0 / max(1.0, self.videoMeta.fps)))))
            self.isPlaying = True
            self._append_log("[INFO] 영상 재생을 시작했습니다.")
        self._update_play_pause_button_icon()
        self._update_button_state()

    def _start_playback_if_ready(self) -> None:
        if self._video_capture is None or self.isPlaying or self.isRunning:
            return
        self.playbackTimer.start(max(33, int(round(1000.0 / max(1.0, self.videoMeta.fps)))))
        self.isPlaying = True
        self._append_log("[INFO] 영상 재생을 시작했습니다.")
        self._update_play_pause_button_icon()
        self._update_button_state()

    def submit_prompt(self) -> None:
        self.promptText = normalize_user_text(self.promptInput.toPlainText())
        if not self.promptText:
            self._append_log("[WARN] 텍스트 프롬프트를 입력하세요.")
            return
        self._append_log(f"[INFO] 프롬프트를 적용했습니다: {self.promptText}")
        self.statusChanged.emit("프롬프트 준비 완료")
        self._update_button_state()

    def _set_nlp_status(self, text: str) -> None:
        self.labelNlpStatus.setText(text)

    def _on_nlp_health(self) -> None:
        self._set_nlp_status("NLP 상태: 확인 중...")
        QApplication.processEvents()
        try:
            payload = self.llm_client.health()
            model_id = str(payload.get("model_id", "-"))
            load_mode = str(payload.get("load_mode", "-"))
            device = str(payload.get("device", "-"))
            self.nlpOutput.setPlainText(
                "\n".join(
                    [
                        f"status: {payload.get('status', '-')}",
                        f"model_id: {model_id}",
                        f"load_mode: {load_mode}",
                        f"device: {device}",
                    ]
                )
            )
            self._set_nlp_status("NLP 상태: 정상")
        except Exception as exc:
            self.nlpOutput.setPlainText(f"Health check failed:\n{exc}")
            self._set_nlp_status("NLP 상태: 실패")
            self._append_log(f"[ERROR] NLP health check failed: {exc}")
        self._update_button_state()

    def _start_nlp_warmup(self) -> None:
        if self._nlpWarmupRunning or self._nlpWarmupDone:
            return
        self._cleanup_nlp_warmup_thread(wait=False)
        self._nlpWarmupRunning = True
        self._set_nlp_status("NLP 상태: 서버 모델 로딩 중...")
        self._append_log("[INFO] NLP 서버 모델 로딩을 시작합니다.")
        thread = QThread(self)
        worker = NlpWarmupWorker(self.llm_client)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_nlp_warmup_finished)
        worker.failed.connect(self._on_nlp_warmup_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_nlp_warmup_thread_finished)
        self._nlp_warmup_thread = thread
        self._nlp_warmup_worker = worker
        self._update_button_state()
        thread.start()

    def _cleanup_nlp_warmup_thread(self, wait: bool) -> None:
        thread = self._nlp_warmup_thread
        if thread is None:
            return
        try:
            thread.quit()
        except Exception:
            pass
        if wait:
            try:
                thread.wait(3000)
            except Exception:
                pass

    def _on_nlp_warmup_finished(self, payload: object) -> None:
        result = dict(payload) if isinstance(payload, Mapping) else {}
        self._nlpWarmupRunning = False
        self._nlpWarmupDone = True
        model_id = str(result.get("model_id", "-"))
        load_mode = str(result.get("load_mode", "-"))
        device = str(result.get("device", "-"))
        self._set_nlp_status("NLP 상태: 준비 완료")
        self._append_log(f"[INFO] NLP 서버 준비 완료: model={model_id}, mode={load_mode}, device={device}")
        self._update_button_state()

    def _on_nlp_warmup_failed(self, message: str) -> None:
        self._nlpWarmupRunning = False
        self._nlpWarmupDone = False
        self._set_nlp_status("NLP 상태: 초기화 실패")
        self._append_log(f"[ERROR] NLP warmup failed: {message}")
        self._update_button_state()

    def _on_nlp_warmup_thread_finished(self) -> None:
        if self._nlp_warmup_thread is not None:
            self._nlp_warmup_thread.deleteLater()
        self._nlp_warmup_thread = None
        self._nlp_warmup_worker = None

    def _render_nlp_response(
        self,
        *,
        user_text: str,
        class_name: str,
        response: Mapping[str, Any],
        provisional: bool,
    ) -> None:
        meta = dict(response.get("_meta") or {})
        ranked_candidates = extract_ranked_prompt_candidates(response)
        self.promptCandidates = build_sam_prompt_candidates(
            prompt_text=user_text,
            class_name=class_name,
            ranked_candidates=ranked_candidates,
        )
        fallback_mode = str(meta.get("fallback_mode", "")).strip()

        best_prompt = self.promptCandidates[0] if self.promptCandidates else ""
        if best_prompt:
            if not provisional:
                self._append_log(
                    f"[INFO] nlp prompt applied: {best_prompt}"
                    + (f" (cache={str(bool(meta.get('cache_hit', False))).lower()})" if meta else "")
                )
                if fallback_mode:
                    self._append_log(f"[WARN] NLP fallback used: {fallback_mode}")

        # 휴리스틱 후보를 1순위로, LLM 결과를 보완으로 통합한 표시 payload 생성
        display_payload = build_display_payload(
            user_text=user_text,
            class_name=class_name,
            llm_payload=response,
        )
        self.nlpOutput.setPlainText(format_nlp_output_for_display(display_payload))

    def _show_nlp_heuristic_preview(self, user_text: str, class_name: str) -> None:
        heuristic_candidates = heuristic_english_candidates(user_text, class_name)
        if not heuristic_candidates:
            return
        total = float(sum(range(1, len(heuristic_candidates) + 1))) or 1.0
        items: list[dict[str, Any]] = []
        for index, candidate in enumerate(heuristic_candidates, start=1):
            items.append(
                {
                    "english_prompt": candidate,
                    "korean_gloss": "",
                    "probability": round((len(heuristic_candidates) - index + 1) / total, 6),
                    "loss": float(index - 1),
                }
            )
        self._render_nlp_response(
            user_text=user_text,
            class_name=class_name,
            response={
                "model_id": "heuristic-preview",
                "load_mode": "local-heuristic-preview",
                "device": "local",
                "items": items,
                "_meta": {"cache_hit": False, "elapsed_ms": 0.0},
            },
            provisional=True,
        )
        if self._nlpWarmupRunning:
            self._append_log("[INFO] NLP warmup is still running. Heuristic preview applied first.")

    def _start_nlp_run(self, user_text: str, class_name: str) -> None:
        self._cleanup_nlp_run_thread(wait=False)
        thread = QThread(self)
        worker = NlpRunWorker(self.llm_client, user_text=user_text, class_name=class_name, n=3)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_nlp_run_finished)
        worker.failed.connect(self._on_nlp_run_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_nlp_run_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._nlp_run_thread = thread
        self._nlp_run_worker = worker
        self._nlpRunRunning = True
        self._update_button_state()
        thread.start()

    def _cleanup_nlp_run_thread(self, wait: bool) -> None:
        thread = self._nlp_run_thread
        if thread is None:
            return
        try:
            thread.quit()
        except Exception:
            pass
        if wait:
            try:
                thread.wait(3000)
            except Exception:
                pass

    def _on_nlp_run_finished(self, payload: object) -> None:
        result = dict(payload) if isinstance(payload, Mapping) else {}
        response = dict(result.get("response") or {})
        user_text = normalize_user_text(str(result.get("user_text", "") or ""))
        class_name = str(result.get("class_name", self.inputClassName.text().strip()) or "").strip()
        self._nlpRunRunning = False
        if response:
            self._render_nlp_response(
                user_text=user_text or normalize_user_text(self.promptInput.toPlainText()),
                class_name=class_name or self.inputClassName.text().strip(),
                response=response,
                provisional=False,
            )
            self._set_nlp_status("NLP 상태: 완료")
        else:
            self.nlpOutput.setPlainText("Run failed:\nNLP returned an empty payload.")
            self._set_nlp_status("NLP 상태: 실패")
            self._append_log("[ERROR] NLP run failed: empty response")
        self._update_button_state()

    def _on_nlp_run_failed(self, message: str) -> None:
        self._nlpRunRunning = False
        current = self.nlpOutput.toPlainText().strip()
        if current:
            self.nlpOutput.setPlainText(f"{current}\n\nRun failed:\n{message}")
        else:
            self.nlpOutput.setPlainText(f"Run failed:\n{message}")
        self._set_nlp_status("NLP 상태: 실패")
        self._append_log(f"[ERROR] NLP run failed: {message}")
        self._update_button_state()

    def _on_nlp_run_thread_finished(self) -> None:
        if self._nlp_run_thread is not None:
            self._nlp_run_thread.deleteLater()
        self._nlp_run_thread = None
        self._nlp_run_worker = None

    def _on_nlp_run(self) -> None:
        user_text = normalize_user_text(self.promptInput.toPlainText())
        if not user_text:
            self.nlpOutput.setPlainText("Input is empty.")
            self._set_nlp_status("NLP 상태: 입력 필요")
            self._update_button_state()
            return
        if self._nlpRunRunning:
            return
        class_name = self.inputClassName.text().strip()
        self._show_nlp_heuristic_preview(user_text, class_name)
        self._set_nlp_status("NLP 상태: 실행 중...")
        self._append_log("[INFO] NLP run started in background.")
        self._start_nlp_run(user_text, class_name)

    def _on_nlp_clear(self) -> None:
        self.nlpOutput.clear()
        self.promptCandidates = build_sam_prompt_candidates(
            prompt_text=self.promptInput.toPlainText(),
            class_name=self.inputClassName.text().strip(),
        )
        self._set_nlp_status("NLP 상태: 대기")
        self._update_button_state()

    def run_process(self) -> None:
        self.className = self.inputClassName.text().strip()
        self.promptText = normalize_user_text(self.className)
        self.promptCandidates = build_sam_prompt_candidates(
            prompt_text=self.promptText,
            class_name=self.className,
            ranked_candidates=self.promptCandidates,
        )
        if self.isRunning:
            self._append_log("[WARN] 이미 실행 중입니다.")
            return
        if not self.video_path:
            self._append_log("[WARN] 영상을 먼저 선택하세요.")
            return
        if not self.className:
            self._append_log("[WARN] 클래스 이름을 입력하세요.")
            return
        if not self._has_prompt_source():
            self._append_log("[WARN] 텍스트 프롬프트를 입력하세요.")
            return
        self.logs = []
        self.logArea.clear()
        self.previewItems = []
        self.resultBrowser.clear()
        _clear_preview_cache_root(self._previewCacheRoot)
        self._previewCacheRoot = None
        self.framesProcessed = self.newSamples = self.numImages = self.numLabels = 0
        self._lastSamProgressLogFrame = -1
        self.outputDir = "-"
        self.experimentId = datetime.now().strftime("sam_%Y%m%d_%H%M%S")
        self._run_config = SamRunConfig(
            className=self.className,
            imagePath="",
            videoPath=self.video_path,
            experimentId=self.experimentId,
            promptMode="text",
            promptText=self.promptText,
        )
        self.outputDir = str(prepare_output_dir(self.experimentId))
        self.playbackTimer.stop()
        self.isPlaying = False
        self.isRunning = True
        self._update_summary_labels()
        self._append_log(f"[INFO] 실험 시작: {self.experimentId}")
        self._append_log(f"[INFO] 프롬프트: {self.promptText}")
        self._append_log(f"[INFO] 모델: {self.modelsText}")
        self.statusChanged.emit("SAM 실행 중")
        self._show_loading_overlay("원격 SAM3 추론 중", "A100 서버에서 객체 탐지와 추적을 수행하고 있습니다...")
        self._start_remote_sam3_run()
        self._update_play_pause_button_icon()
        self._update_button_state()

    def stop_process(self) -> None:
        if not self.isRunning:
            self._append_log("[WARN] 실행 중인 작업이 없습니다.")
            return
        if self._sam3_worker is not None:
            self._sam3_worker.request_stop()
        self._update_loading_overlay("중단 요청을 전송했습니다...")
        self._append_log("[INFO] 중지 요청을 보냈습니다.")
        self.statusChanged.emit("SAM 중지 요청")
        self._update_button_state()

    def _run_dataset_export(self) -> None:
        from app.studio.workers import DatasetExportWorker

        worker = DatasetExportWorker(
            video_path=Path(self.video_path or ""),
            dataset_root=Path(self.outputDir),
            video_stem=Path(self.video_path or "video").stem or "video",
            preview_items=self.resultBrowser.thumbnail_items,
            frame_annotations=[],
            class_names=[self.className or (self._run_config.className if self._run_config else "object")],
        )
        summary = worker._run_export()
        if summary is None:
            raise RuntimeError("keep 상태 결과가 없어 내보낼 데이터셋을 만들 수 없습니다.")
        self.numImages = int(summary.train_images + summary.valid_images + summary.test_images)
        self.numLabels = int(summary.total_boxes)

    def export_results(self) -> None:
        self._start_dataset_export()
        return
        if self.isRunning or (self.numImages == 0 and self.numLabels == 0):
            self._append_log("[WARN] 현재 내보낼 결과가 없습니다.")
            return
        export_dir = QFileDialog.getExistingDirectory(self, "내보내기 폴더 선택", str(Path.home()))
        if not export_dir:
            return
        self._run_dataset_export()
        write_manifest(
            self.outputDir,
            config=self._run_config or {},
            progress=SamProgress(
                framesProcessed=int(self.framesProcessed),
                newSamples=int(self.newSamples),
                numImages=int(self.numImages),
                numLabels=int(self.numLabels),
                message="exported",
                level="INFO",
            ),
        )
        write_config_snapshot(self.outputDir, config=self._run_config or {})
        export_output_tree(self.outputDir, export_dir)
        _clear_preview_cache_root(self._previewCacheRoot)
        self._previewCacheRoot = None
        self._append_log(f"[INFO] 내보내기 완료: {Path(export_dir).resolve() / Path(self.outputDir).name}")
        self.statusChanged.emit("내보내기 완료")

    def _clone_preview_items_for_export(self) -> list[PreviewThumbnail]:
        cloned: list[PreviewThumbnail] = []
        for item in self.resultBrowser.thumbnail_items:
            cloned_boxes: list[Any] = []
            for box in list(getattr(item, "boxes", []) or []):
                if isinstance(box, Mapping):
                    cloned_boxes.append(dict(box))
                else:
                    cloned_boxes.append(box)
            cloned.append(
                PreviewThumbnail(
                    frame_index=int(getattr(item, "frame_index", -1)),
                    image=None,
                    boxes=cloned_boxes,
                    category=str(getattr(item, "category", "hold")),
                    item_id=str(getattr(item, "item_id", "")),
                    image_path=None,
                    thumb_path=None,
                    manifest_path=None,
                )
            )
        return cloned

    def _build_export_dataset_root(self, base_dir: Path) -> Path:
        class_names = [self.className or (self._run_config.className if self._run_config else "object")]
        dataset_name = build_dataset_folder_name(class_names, kind="export", created_at=datetime.now())
        return _build_unique_path(base_dir / dataset_name)

    def _start_dataset_export(self) -> None:
        if self.isRunning or self._isExporting or (self.numImages == 0 and self.numLabels == 0):
            self._append_log("[WARN] 현재 내보낼 결과가 없습니다.")
            return
        if not self.video_path:
            self._append_log("[WARN] 먼저 영상을 선택하세요.")
            return
        base_dir = QFileDialog.getExistingDirectory(
            self,
            "데이터셋 저장 위치 선택",
            str(DATASET_SAVE_DIR if DATASET_SAVE_DIR.is_dir() else Path.home()),
        )
        if not base_dir:
            return
        preview_items = self._clone_preview_items_for_export()
        if not preview_items:
            self._append_log("[WARN] keep/hold/drop 결과가 없어 내보낼 수 없습니다.")
            return

        dataset_root = self._build_export_dataset_root(Path(base_dir).resolve())
        worker = DatasetExportWorker(
            video_path=Path(self.video_path),
            dataset_root=dataset_root,
            video_stem=Path(self.video_path).stem or "video",
            preview_items=preview_items,
            frame_annotations=[],
            class_names=[self.className or (self._run_config.className if self._run_config else "object")],
            split_ratio=(0.8, 0.1, 0.1),
            shuffle_seed=None,
            roi_rect=None,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_export_progress)
        worker.finished.connect(self._on_export_finished)
        worker.failed.connect(self._on_export_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_export_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._export_thread = thread
        self._export_worker = worker
        self._isExporting = True
        self._append_log(f"[INFO] export start: {dataset_root}")
        self.statusChanged.emit("내보내기 중")
        self._show_loading_overlay("YOLO 데이터셋 내보내기 중", "SAM keep 결과를 기존 결과 내보내기 로직으로 저장하고 있습니다...")
        self._update_button_state()
        thread.start()

    def _on_export_progress(self, message: str, done: int, total: int) -> None:
        self._update_loading_overlay(f"{message}: {int(done)}/{max(1, int(total))}")

    def _on_export_finished(self, summary_obj: object) -> None:
        self._hide_loading_overlay()
        self._isExporting = False
        summary = summary_obj
        if not hasattr(summary, "dataset_root"):
            self._on_export_failed("내보내기 결과 형식이 올바르지 않습니다.")
            return
        self.lastExportDatasetRoot = Path(summary.dataset_root).resolve()
        self.numImages = int(summary.train_images + summary.valid_images + summary.test_images)
        self.numLabels = int(summary.total_boxes)
        self._append_log(
            f"[INFO] 내보내기 완료: {self.lastExportDatasetRoot} "
            f"(images={self.numImages}, labels={self.numLabels}, classes={int(summary.class_count)})"
        )
        QMessageBox.information(
            self,
            "내보내기 완료",
            f"YOLO 데이터셋 저장 위치:\n{self.lastExportDatasetRoot}\n\n"
            f"이미지: {self.numImages}\n"
            f"라벨 파일: {int(summary.train_labels + summary.valid_labels + summary.test_labels)}\n"
            f"객체 박스: {self.numLabels}\n"
            f"클래스 수: {int(summary.class_count)}",
        )
        self.statusChanged.emit("내보내기 완료")
        self._update_button_state()

    def _on_export_failed(self, message: str) -> None:
        self._hide_loading_overlay()
        self._isExporting = False
        self._append_log(f"[ERROR] export failed: {message}")
        QMessageBox.warning(self, "내보내기 실패", str(message))
        self.statusChanged.emit("내보내기 실패")
        self._update_button_state()

    def _on_export_thread_finished(self) -> None:
        if self._export_thread is not None:
            self._export_thread.deleteLater()
        self._export_thread = None
        self._export_worker = None

    def _start_remote_sam3_run(self) -> None:
        self._cleanup_remote_sam3_thread(wait=False)
        request = RemoteSam3Request(
            class_name=self.className,
            prompt_text=self.promptText,
            prompt_candidates=list(self.promptCandidates),
            video_path=self.video_path or "",
            experiment_id=self.experimentId,
            output_dir=self.outputDir,
        )
        thread = QThread(self)
        worker = RemoteSam3Worker(request)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._handle_progress)
        worker.log_message.connect(self._append_log)
        worker.finished.connect(self._on_remote_sam3_finished)
        worker.failed.connect(self._on_remote_sam3_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_remote_sam3_thread_finished)
        self._sam3_thread = thread
        self._sam3_worker = worker
        thread.start()

    def _cleanup_remote_sam3_thread(self, wait: bool) -> None:
        worker = self._sam3_worker
        thread = self._sam3_thread
        if worker is not None:
            try:
                worker.request_stop()
            except Exception:
                pass
        if thread is not None:
            try:
                thread.quit()
            except Exception:
                pass
            if wait:
                try:
                    thread.wait(3000)
                except Exception:
                    pass

    def _on_remote_sam3_finished(self, payload: object) -> None:
        result = dict(payload) if isinstance(payload, Mapping) else {}
        self.isRunning = False
        self._hide_loading_overlay()
        self.outputDir = str(result.get("output_dir", self.outputDir))
        if bool(result.get("stopped", False)):
            self._append_log("[INFO] 원격 SAM3 작업이 중지되었습니다.")
            self.statusChanged.emit("작업 중지")
            self._update_button_state()
            return
        preview_items = list(result.get("preview_items", []) or [])
        self.previewItems = [item for item in preview_items if isinstance(item, PreviewThumbnail)]
        preview_cache_root_raw = str(result.get("preview_cache_root", "") or "").strip()
        self._previewCacheRoot = (
            Path(preview_cache_root_raw).resolve()
            if preview_cache_root_raw
            else _preview_cache_root_for_experiment(self.experimentId or Path(self.outputDir).name)
        )
        cached_count = int(result.get("preview_item_count", 0) or 0)
        if preview_cache_root_raw:
            cached_items = _load_preview_items_from_cache(self._previewCacheRoot)
        else:
            cached_count = _persist_preview_items_to_cache(
                self._previewCacheRoot,
                self.previewItems,
                video_path=self.video_path,
            )
            cached_items = _load_preview_items_from_cache(self._previewCacheRoot)
        if cached_items:
            self.previewItems = cached_items
            if cached_count <= 0:
                cached_count = len(cached_items)
        self._rebuild_bbox_overlay()
        self.resultBrowser.set_items(self.previewItems, video_path=self.video_path)
        summary = dict(result.get("summary", {}) or {})
        self.framesProcessed = int(summary.get("processed_frames", self.framesProcessed or 0))
        counts = dict(result.get("counts", {}) or {})
        self.numImages = int(counts.get("num_images", self.numImages or 0))
        self.numLabels = int(counts.get("num_labels", self.numLabels or 0))
        keep_count = int(summary.get("keep_items", 0) or 0)
        hold_count = int(summary.get("hold_items", 0) or 0)
        drop_count = int(summary.get("drop_items", 0) or 0)

        # 모든 프레임에서 객체가 하나도 검출되지 않은 경우 경고 메세지 표시
        if keep_count == 0 and hold_count == 0 and drop_count == 0:
            class_name_input = ""
            if self._sam3_worker is not None:
                class_name_input = str(self._sam3_worker.request.class_name or "").strip()
            label = class_name_input if class_name_input else "(클래스 미입력)"
            self._append_log(f"[WARN] SAM3 추론 결과: 모든 프레임에서 객체 미검출 (입력: {label})")
            QMessageBox.warning(
                self,
                "객체 미검출",
                f"입력된 클래스명 : [ {label} ]\n\n"
                "- 검출된 객체가 없습니다. 아래 사항을 확인하세요.\n\n"
                "  1. 영상 내 입력한 클래스가 존재하는지 확인하세요.\n\n"
                "  2. 영상 내에 입력한 클래스가 존재한다면\n"
                "     → 클래스명을 SAM3 프롬프트 형식에 맞게 수정하세요.\n"
                '     → SAM3 프롬프트 형식 : "명확한 영어 명사구"',
            )

        unique_track_count = int(summary.get("filtered_unique_track_count", summary.get("unique_track_count", 0)) or 0)
        self.newSamples = max(0, keep_count + hold_count)
        tracker_backend = str(summary.get("tracker_backend", "-")).strip() or "-"
        keyframe_interval = int(summary.get("keyframe_interval", 0) or 0)
        pipeline_fps = float(summary.get("pipeline_fps", 0.0) or 0.0)
        preview_video_path = str(result.get("preview_video_path", "") or "").strip()
        if preview_video_path:
            preview_path = Path(preview_video_path)
            if preview_path.is_file():
                loaded = self._open_video_capture(
                    str(preview_path),
                    preserve_source_video=True,
                    autoplay=True,
                    failure_log="[WARN] 결과 미리보기 영상을 불러오지 못했습니다.",
                )
                if loaded:
                    self._append_log(f"[INFO] 결과 영상 재생 시작: {preview_path.name}")
        if cached_count > 0:
            self._append_log(f"[INFO] keep/hold/drop 임시 캐시에 저장됨: {cached_count}개")
        self._append_log(
            f"[INFO] 원격 SAM3 완료: 항목={len(self.previewItems)}, keep 이미지={self.numImages}, "
            f"id={unique_track_count}, tracker={tracker_backend}, keyframe={keyframe_interval}, fps={pipeline_fps:.2f}"
        )
        self.statusChanged.emit("SAM 완료")
        self._update_summary_labels()
        self._update_button_state()

    def _on_remote_sam3_failed(self, message: str) -> None:
        self.isRunning = False
        self._hide_loading_overlay()
        self._append_log(f"[ERROR] {message}")
        self.statusChanged.emit("SAM 실패")
        self._update_button_state()

    def _on_remote_sam3_thread_finished(self) -> None:
        if self._sam3_thread is not None:
            self._sam3_thread.deleteLater()
        self._sam3_thread = None
        self._sam3_worker = None

    def load_video_from_path(
        self,
        path: str,
        roi: tuple[int, int, int, int] | None = None,
    ) -> bool:
        """영상 입력 페이지에서 선택된 영상을 다이얼로그 없이 직접 로드합니다."""
        if self.isRunning or self._isExporting:
            return False
        # 동일 영상이 이미 열려 있으면 재로드 생략
        if self.video_path == path and self._video_capture is not None:
            return True
        ok = self._open_video_capture(
            path,
            preserve_source_video=False,
            autoplay=False,
            failure_log=f"[ERROR] 영상을 불러오지 못했습니다: {path}",
        )
        if ok:
            self._append_log(f"[INFO] 영상 자동 로드: {Path(path).name}")
            self._update_button_state()
        return ok

    def refresh_view(self) -> None:
        self._render_video_preview()
        self._update_meta_labels()
        self._update_summary_labels()
        self._update_button_state()

    def refresh_theme(self) -> None:
        self._style_playback_controls()
        self._style_log_area()
        self._apply_page_styles()
        for divider in self._dividers:
            self._style_divider(divider)
        self._sync_responsive_layouts()
        self._update_play_pause_button_icon()
        self._update_button_state()

    def _on_class_name_changed(self, text: str) -> None:
        self.className = text.strip()
        self.promptCandidates = build_sam_prompt_candidates(
            prompt_text=self.promptText,
            class_name=self.className,
            ranked_candidates=self.promptCandidates,
        )
        self._update_button_state()

    def _on_prompt_text_changed(self) -> None:
        self.promptText = normalize_user_text(self.promptInput.toPlainText())
        self.promptCandidates = build_sam_prompt_candidates(
            prompt_text=self.promptText,
            class_name=self.inputClassName.text().strip(),
            ranked_candidates=self.promptCandidates,
        )
        self._update_button_state()

    def _on_result_browser_counts_changed(self, keep_count: int, hold_count: int, drop_count: int) -> None:
        self.previewItems = list(self.resultBrowser.thumbnail_items)
        _ = (keep_count, hold_count, drop_count)
        self._rebuild_bbox_overlay()
        summary = summarize_preview_items(self.previewItems)
        self.numImages = int(summary.get("num_images", 0))
        self.numLabels = int(summary.get("num_labels", 0))
        self.newSamples = int(summary.get("keep_items", 0)) + int(summary.get("hold_items", 0))
        self._update_summary_labels()
        self._update_button_state()

    def _advance_playback(self) -> None:
        next_index = self.currentFrame + 1
        if self._video_capture is None or next_index >= self.videoMeta.totalFrames:
            self.playbackTimer.stop()
            self.isPlaying = False
            self._update_play_pause_button_icon()
            self._update_button_state()
            return
        self.sliderTimeline.setValue(next_index)

    def _handle_progress(self, progress: Any) -> None:
        if isinstance(progress, Mapping):
            progress = SamProgress(
                framesProcessed=int(progress.get("framesProcessed", 0)),
                newSamples=int(progress.get("newSamples", 0)),
                numImages=int(progress.get("numImages", 0)),
                numLabels=int(progress.get("numLabels", 0)),
                message=str(progress.get("message", "")),
                level=str(progress.get("level", "INFO")),
            )
        self.framesProcessed = int(getattr(progress, "framesProcessed", 0))
        self.newSamples = int(getattr(progress, "newSamples", 0))
        self.numImages = int(getattr(progress, "numImages", 0))
        self.numLabels = int(getattr(progress, "numLabels", 0))
        self._update_summary_labels()
        message = str(getattr(progress, "message", "")).strip()
        level = str(getattr(progress, "level", "INFO")).strip() or "INFO"
        if self.isRunning:
            if message:
                self._update_loading_overlay(message)
            else:
                self._update_loading_overlay(f"원격 처리 프레임 수: {self.framesProcessed}")
        should_append_log = bool(message)
        if message.startswith("원격 SAM3 처리 프레임 수:"):
            if self.framesProcessed <= 1 or (self.framesProcessed - self._lastSamProgressLogFrame) >= 50:
                self._lastSamProgressLogFrame = self.framesProcessed
            else:
                should_append_log = False
        if should_append_log:
            self._append_log(f"[{level}] {message}")

    def _update_button_state(self) -> None:
        run_enabled = bool(
            self.video_path and self.inputClassName.text().strip() and self._has_prompt_source() and (not self.isRunning) and (not self._isExporting)
        )
        export_enabled = (not self.isRunning) and (not self._isExporting) and (self.numImages > 0 or self.numLabels > 0)
        playback = self._video_capture is not None and (not self.isRunning) and (not self._isExporting)
        send = bool(self.promptInput.toPlainText().strip()) and (not self.isRunning) and (not self._isExporting)
        nlp_enabled = (
            (not self.isRunning)
            and (not self._isExporting)
            and (not self._nlpWarmupRunning)
            and (not self._nlpRunRunning)
        )
        nlp_clear = bool(self.nlpOutput.toPlainText().strip()) and nlp_enabled
        self.btnRun.setEnabled(run_enabled)
        self.btnStop.setEnabled(self.isRunning)
        self.btnExport.setEnabled(export_enabled)
        self.btnNlpHealth.setEnabled(nlp_enabled)
        self.btnNlpRun.setEnabled(send and nlp_enabled)
        self.btnNlpClear.setEnabled(nlp_clear)
        self.sliderTimeline.setEnabled(playback)
        for button in (self.btnSeekBack, self.btnPrevFrame, self.btnPlayPause, self.btnNextFrame):
            button.setEnabled(playback)
        for button in (
            self.btnRun,
            self.btnStop,
            self.btnExport,
            self.btnNlpHealth,
            self.btnNlpRun,
            self.btnNlpClear,
            self.btnSeekBack,
            self.btnPrevFrame,
            self.btnPlayPause,
            self.btnNextFrame,
        ):
            button.setCursor(Qt.CursorShape.PointingHandCursor if button.isEnabled() else Qt.CursorShape.ForbiddenCursor)

    def _update_meta_labels(self) -> None:
        self.lblTime.setText(f"{self._format_hms(self.timeline['currentTimeSec'])} / {self._format_hms(self.timeline['totalTimeSec'])}")

    def _update_summary_labels(self) -> None:
        return

    def _render_video_preview(self) -> None:
        self._apply_scaled_pixmap(self.videoPreview, self._video_pixmap, "영상 로드")

    def _apply_scaled_pixmap(self, label: QLabel, pixmap: QPixmap | None, placeholder: str) -> None:
        if pixmap is None or pixmap.isNull():
            label.clear()
            label.setText(placeholder)
            return
        label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        label.setText("")

    def _style_playback_controls(self) -> None:
        light = self._is_light_theme()
        style = (
            "QToolButton{min-width:38px;max-width:38px;min-height:38px;max-height:38px;border-radius:19px;"
            f"border:1px solid {'rgba(17,24,39,30)' if light else 'rgba(173,190,217,42)'};"
            f"background:{'rgba(255,255,255,245)' if light else 'rgba(24,34,52,230)'};"
            f"color:{'rgb(17,24,39)' if light else 'rgb(232,239,255)'};font-size:14px;font-weight:700;"
            "}"
            f"QToolButton:hover{{background:{'rgb(239,244,251)' if light else 'rgba(35,48,71,238)'};}}"
            f"QToolButton:pressed{{background:{'rgb(227,235,247)' if light else 'rgba(52,67,94,240)'};}}"
            f"QToolButton:disabled{{background:{'rgb(239,242,247)' if light else 'rgba(44,52,66,210)'};color:{'rgb(147,157,173)' if light else 'rgb(139,147,165)'};}}"
        )
        for button in (self.btnSeekBack, self.btnPrevFrame, self.btnPlayPause, self.btnNextFrame):
            button.setAutoRaise(False)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet(style)

    def _style_log_area(self) -> None:
        light = self._is_light_theme()
        self.logArea.setStyleSheet(
            "QPlainTextEdit{"
            f"background-color:{'#EEF2F7' if light else '#091224'};"
            f"color:{'#1E293B' if light else '#DBEAFE'};"
            f"border:1px solid {'#D3DDEB' if light else '#36455F'};border-radius:12px;padding:8px;"
            "selection-background-color:#2563EB;selection-color:#FFFFFF;font-family:Consolas,'D2Coding',monospace;}"
        )

    def _apply_page_styles(self) -> None:
        light = self._is_light_theme()
        c = {
            "surface": "#F7F9FC" if light else "rgba(10,18,34,0.72)",
            "border": "#D8E2F0" if light else "rgba(93,118,156,0.60)",
            "border_strong": "#D3DDEB" if light else "rgba(93,118,156,0.62)",
            "muted": "#94A3B8" if light else "#A5B4CF",
            "text": "#0F172A" if light else "#E5EEF8",
            "subtle": "#5B6B81" if light else "#B8C5DB",
            "field": "#FFFFFF" if light else "rgba(7,16,30,0.82)",
            "tab_bg": "#F2F5F9" if light else "rgba(17,27,46,0.76)",
            "tab_fg": "#475569" if light else "#C4D3E9",
            "hover_bg": "#EEF4FF" if light else "rgba(23,36,61,0.92)",
            "accent": "#2563EB" if light else "#7AA2FF",
            "accent_hover": "#1D4ED8" if light else "#7AA2FF",
            "accent_text": "#FFFFFF" if light else "#1A2230",
            "accent_disabled": "rgba(37,99,235,0.42)" if light else "rgba(122,162,255,0.42)",
        }
        self.setStyleSheet(
            f"""
QFrame#videoPreviewBox,QFrame#promptInputFrame{{background:{c["surface"]};border:1px solid {c["border"]};border-radius:16px;}}

QLabel#videoPreview{{color:{c["muted"]};font:800 15px "Pretendard";}}
QLabel#labelNewObjectSectionTitle{{color:{c["text"]};font:900 12pt "Pretendard";}}
QLabel[promptHelper="true"],QLabel#labelPromptInputCaption{{color:{c["subtle"]};font:600 10pt "Pretendard";}}
QLineEdit#inputClassName{{min-height:38px;padding:0 12px;background:{c["field"]};border:1px solid {c["border_strong"]};border-radius:12px;color:{c["text"]};}}
QToolButton#tabAll,QToolButton#tabOk,QToolButton#tabHold,QToolButton#tabDrop{{border:1px solid {c["border_strong"]};border-radius:12px;background:{c["tab_bg"]};color:{c["tab_fg"]};min-height:30px;padding:0 10px;font:800 9pt "Pretendard";}}
QToolButton#tabAll:hover,QToolButton#tabOk:hover,QToolButton#tabHold:hover,QToolButton#tabDrop:hover{{border-color:{c["accent"]};background:{c["hover_bg"] if light else '#2B3B56'};}}
QToolButton#tabAll:checked,QToolButton#tabOk:checked,QToolButton#tabHold:checked,QToolButton#tabDrop:checked{{background:{c["accent"]};border-color:{c["accent"]};color:{c["accent_text"]};}}
QListWidget#listThumbs{{background:{c["field"]};border:1px solid {c["border_strong"]};border-radius:12px;padding:6px;color:{c["text"]};selection-background-color:{c["accent"]};selection-color:{c["accent_text"]};}}
QListWidget#listThumbs::item{{padding:8px;border-radius:10px;}}
QListWidget#listThumbs::item:hover{{background:{'rgba(37,99,235,0.16)' if light else 'rgba(122,162,255,0.16)'};}}
QListWidget#listThumbs::item:selected{{background:{'rgba(37,99,235,0.24)' if light else 'rgba(122,162,255,0.24)'};color:{c["accent_text"] if not light else c["text"]};}}
QPushButton#btnNlpHealth,QPushButton#btnNlpRun,QPushButton#btnNlpClear{{min-height:30px;padding:0 10px;border-radius:10px;font:700 9pt "Pretendard";}}
QPushButton#btnNlpHealth,QPushButton#btnNlpClear{{background:{c["tab_bg"]};border:1px solid {c["border_strong"]};color:{c["text"]};}}
QPushButton#btnNlpHealth:hover,QPushButton#btnNlpClear:hover{{border-color:{c["accent"]};}}
QPushButton#btnNlpRun{{background:{c["accent"]};border:1px solid {c["accent"]};color:{c["accent_text"]};}}
QPushButton#btnNlpRun:hover{{background:{c["accent_hover"]};border-color:{c["accent_hover"]};}}
QPushButton#btnNlpHealth:disabled,QPushButton#btnNlpRun:disabled,QPushButton#btnNlpClear:disabled{{background:{c["accent_disabled"]};border-color:{c["accent_disabled"]};color:{c["accent_text"]};}}
QFrame#newObjectPromptCard QPushButton#btnRun,QFrame#newObjectPromptCard QPushButton#btnStop,QPushButton#btnExport{{color:#FFFFFF;font:900 10pt "Pretendard";min-height:38px;padding:0 14px;}}
QFrame#newObjectPromptCard QPushButton#btnRun{{background:#22C55E;border:1px solid #22C55E;}}
QFrame#newObjectPromptCard QPushButton#btnRun:hover{{background:#16A34A;border-color:#16A34A;}}
QFrame#newObjectPromptCard QPushButton#btnRun:disabled{{background:rgba(34,197,94,0.42);border-color:rgba(34,197,94,0.42);}}
QFrame#newObjectPromptCard QPushButton#btnStop{{background:#EF4444;border:1px solid #EF4444;}}
QFrame#newObjectPromptCard QPushButton#btnStop:hover{{background:#DC2626;border-color:#DC2626;}}
QFrame#newObjectPromptCard QPushButton#btnStop:disabled{{background:rgba(239,68,68,0.42);border-color:rgba(239,68,68,0.42);}}
QPushButton#btnExport{{background:{c["accent"]};border:1px solid {c["accent"]};color:{c["accent_text"]};border-radius:12px;}}
QPushButton#btnExport:hover{{background:{c["accent_hover"]};border-color:{c["accent_hover"]};}}
QPushButton#btnExport:disabled{{background:{c["accent_disabled"]};border-color:{c["accent_disabled"]};color:{c["accent_text"]};}}
QPlainTextEdit#promptTextInput{{background:transparent;border:none;color:{c["text"]};font:500 10.5pt "Pretendard";}}
QPlainTextEdit#nlpOutput{{background:{c["field"]};border:1px solid {c["border_strong"]};border-radius:12px;padding:8px;color:{c["text"]};font:500 9.5pt "Consolas";}}
"""
        )

    def _style_divider(self, divider: QFrame) -> None:
        divider.setStyleSheet(
            f"background-color:{'rgba(160,176,204,0.85)' if self._is_light_theme() else 'rgba(120,154,209,0.60)'};border:none;"
        )

    def _update_play_pause_button_icon(self) -> None:
        if self.btnPlayPause is None:
            return
        if self.isPlaying:
            self.btnPlayPause.setText("||")
            self.btnPlayPause.setToolTip("일시정지")
        else:
            self.btnPlayPause.setText(">")
            self.btnPlayPause.setToolTip("재생")

    def _sync_responsive_layouts(self) -> None:
        if not hasattr(self, "bottomPanelDivider"):
            return
        self.bottomPanelDivider.setFixedWidth(1)
        self.bottomPanelDivider.setMaximumWidth(1)
        self.bottomPanelDivider.setMinimumHeight(0)
        self.bottomPanelDivider.setMaximumHeight(16777215)
        self.bottomPanelDivider.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def _rebuild_bbox_overlay(self) -> None:
        overlay: dict[int, list[dict]] = {}
        for item in self.previewItems:
            fi = int(item.frame_index)
            for box in item.boxes:
                if isinstance(box, dict):
                    overlay.setdefault(fi, []).append(box)
        self._bbox_overlay = overlay
        if self._current_video_frame is not None:
            self._set_video_frame(self._current_video_frame, self.currentFrame)

    def _set_video_frame(self, frame: Any, frame_index: int) -> None:
        raw_frame = frame
        overlay_boxes = self._bbox_overlay.get(frame_index)
        if overlay_boxes and frame is not None:
            frame = frame.copy()
            for box in overlay_boxes:
                _draw_preview_box(frame, box)
        pixmap = self._frame_to_pixmap(frame)
        if pixmap is None:
            return
        self.currentFrame = max(0, frame_index)
        self.timeline["currentTimeSec"] = self.currentFrame / max(1.0, self.videoMeta.fps)
        self._current_video_frame = raw_frame.copy() if hasattr(raw_frame, "copy") else raw_frame
        self._video_pixmap = pixmap
        self._render_video_preview()
        self._update_meta_labels()

    def _append_log(self, message: str) -> None:
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        self.logs.append(entry)
        self.logArea.appendPlainText(entry)
        sb = self.logArea.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def _seek_relative(self, delta: int) -> None:
        if self._video_capture is not None:
            self.sliderTimeline.setValue(max(0, min(self.currentFrame + delta, self.videoMeta.totalFrames - 1)))

    def _read_frame(self, frame_index: int) -> Any | None:
        if self._video_capture is None:
            return None
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = self._video_capture.read()
        return frame if ok else None

    def _release_video_capture(self) -> None:
        if self._video_capture is not None:
            try:
                self._video_capture.release()
            except Exception:
                pass
            self._video_capture = None

    def _frame_to_pixmap(self, frame: Any) -> QPixmap | None:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            return None
        h, w, c = rgb.shape
        image = QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image.copy())

    def _format_hms(self, seconds: float) -> str:
        total = max(0.0, float(seconds))
        hours = int(total // 3600)
        minutes = int((total % 3600) // 60)
        secs = total % 60.0
        return f"{hours:02d}:{minutes:02d}:{secs:04.1f}"

    def _make_card(self, object_name: str) -> QFrame:
        card = QFrame(self)
        card.setProperty("pageCard", True)
        card.setObjectName(object_name)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return card

    def _make_tool(self, text: str, tooltip: str) -> QToolButton:
        button = QToolButton(self.videoCard)
        button.setText(text)
        button.setToolTip(tooltip)
        return button

    def _divider(self, parent: QWidget) -> QFrame:
        divider = QFrame(parent)
        divider.setFixedHeight(1)
        self._style_divider(divider)
        self._dividers.append(divider)
        return divider

    def _has_prompt_source(self) -> bool:
        return bool(self.inputClassName.text().strip())

    def _is_light_theme(self) -> bool:
        return str(QSettings("SL_TEAM", "AutoLabelStudio").value("theme", "dark")).strip().lower() == "light"

    ## =====================================
    ## 함수 기능 : 위젯 종료 시 모든 QThread를 안전하게 정리하는 함수
    ## 매개 변수 : 없음
    ## 반환 결과 : 없음 (모든 실행 중인 스레드에 quit 요청 후 최대 3초 대기)
    ## =====================================
    def cleanup_threads(self) -> None:
        self._cleanup_nlp_warmup_thread(wait=True)
        self._cleanup_nlp_run_thread(wait=True)
        self._cleanup_remote_sam3_thread(wait=True)
        thread = self._export_thread
        if thread is not None and thread.isRunning():
            try:
                thread.quit()
                thread.wait(3000)
            except Exception:
                pass

    def __del__(self) -> None:
        self.playbackTimer.stop()
        self._release_video_capture()
