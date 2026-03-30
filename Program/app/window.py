from __future__ import annotations

import os
import json
import logging
import queue
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Reduce OpenCV/FFmpeg console noise for broken/corrupt H.264 streams.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_FFMPEG_DEBUG", "0")

import numpy as np
from PyQt6 import uic
from PyQt6.QtCore import (
    QEventLoop,
    QPropertyAnimation,
    QRect,
    QThread,
    QTimer,
    Qt,
    pyqtSlot,
)
from PyQt6.QtGui import QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QListView,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QSizePolicy,
    QTableWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:
    from core.dataset import FrameAnnotation
except ModuleNotFoundError:
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from core.dataset import FrameAnnotation

from core.paths import (
    DATASET_SAVE_DIR,
    MERGED_DATASET_SAVE_DIR,
    TEAM_MODEL_DIR,
    TRAIN_RTDETR_MODELS_DIR,
    TRAIN_RUNS_DIR,
    TRAIN_YOLO_MODELS_DIR,
    ensure_storage_directories,
)
from app.ui.pages import (
    _build_home_page_page,
    _build_model_test_page_page,
    _build_second_advanced_page_page,
    _build_second_stage_labeling_step_page,
    _build_training_page_page,
    _build_video_load_page_page,
    _install_theme_toggle_page,
    _navigate_to_page_page,
    _on_page_fade_out_finished_page,
    _on_second_stage_labeling_complete_page,
    _on_second_timeline_changed_page,
    _refresh_advanced1_preview_fit_page,
    _refresh_second_advanced_page_page,
    _refresh_video_load_page_page,
    _relocate_topbar_actions_to_nav_page,
    _reset_second_stage_progress_page,
    _schedule_advanced1_preview_refresh_page,
    _set_nav_checked_page,
    _set_second_stage_step_page,
    _setup_page_navigation_shell_page,
    _switch_page_index_page,
    _update_second_timeline_label_page,
)

from app.studio.config import DEFAULT_CLASS_NAMES
from app.studio.export_ops import (
    _clone_frame_annotations_for_export_ops,
    _clone_preview_items_for_export_ops,
    _label_for_export_class_id_ops,
    _on_export_dataset_ops,
    _on_export_failed_ops,
    _on_export_finished_ops,
    _on_export_progress_ops,
    _on_export_thread_finished_ops,
    _on_load_cached_preview_ops,
    _to_english_slug_ops,
)
from core.models import PreviewThumbnail, ProgressEvent, WorkerOutput
from app.studio.preview_ops import (
    _category_matches_filter_ops,
    _clear_preview_cache_ops,
    _delete_preview_cache_storage_after_export_ops,
    _delete_preview_cache_storage_after_load_ops,
    _drain_preview_cache_events_ops,
    _load_thumbnail_items_from_cache_ops,
    _load_thumbnail_source_image_ops,
    _move_current_thumbnail_in_preview_ops,
    _normalize_thumbnail_category_ops,
    _on_thumbnail_double_clicked_ops,
    _open_thumbnail_preview_dialog_ops,
    _persist_preview_items_to_cache_ops,
    _populate_thumbnails_ops,
    _refresh_thumbnail_list_ops,
    _refresh_thumbnail_preview_dialog_ops,
    _set_thumbnail_filter_ops,
    _set_thumbnail_item_category_ops,
    _step_thumbnail_preview_dialog_ops,
    _stop_preview_cache_watchdog_ops,
    _update_thumbnail_filter_tab_counts_ops,
)
from app.studio.runtime import (
    TqdmFormatter,
    cv2,
    yaml,
)
from app.studio.utils import (
    _build_unique_path,
    _dedupe_preserve_order,
    _load_yaml_dict,
    _metric_to_unit_interval,
    _parse_names_from_yaml_payload,
    _resolve_dataset_root_from_yaml_payload,
    _sync_ultralytics_datasets_dir,
    _to_korean_class_name,
    _to_korean_stage,
    _try_autofix_data_yaml_path,
    _try_autofix_data_yaml_splits,
    build_dataset_folder_name,
    collect_original_components_from_yaml,
    extract_training_metrics_and_losses,
    sanitize_class_name,
    write_provenance_readme,
)
from app.studio.video_ops import (
    _box_color_for_status_ops,
    _class_name_ops,
    _clear_thumbnail_source_cache_ops,
    _clear_video_frame_cache_ops,
    _draw_boxes_ops,
    _ensure_preview_image_cap_ops,
    _extract_box_status_ops,
    _extract_box_xyxy_ops,
    _extract_polygon_points_ops,
    _format_eta_ops,
    _format_hms_ops,
    _frame_to_pixmap_ops,
    _read_video_frame_ops,
    _read_video_frame_raw_ops,
    _release_preview_image_cap_ops,
    _remember_thumbnail_source_cache_ops,
    _show_frame_ops,
    _update_time_label_ops,
)
from app.studio.workers import (
    AutoLabelWorker,
    DatasetExportWorker,
    ModelTestWorker,
    MultiDatasetMergeWorker,
    ReplayDatasetMergeWorker,
    YoloTrainWorker,
)
from app.ui.dialogs.training_yaml_picker import TrainingYamlPickerDialog
from app.ui.widgets.studio_support import FuturisticSpinner, HoverPreviewLabel, MetricChartWidget, RoiSelectionLabel

from app.studio.mixins.session_processing import StudioSessionProcessingMixin
from app.studio.mixins.setup_layout import StudioSetupLayoutMixin
from app.studio.mixins.training_flow import StudioTrainingFlowMixin

class AutoLabelStudioWindow(
    StudioSetupLayoutMixin,
    StudioTrainingFlowMixin,
    StudioSessionProcessingMixin,
    QMainWindow,
):
    """?? ???? UI ??? ????? ???? ?? ??????."""

    def __init__(self, ui_path: Path) -> None:
        """객체 생성 시 필요한 의존성, 기본값, 내부 상태를 초기화합니다."""
        super().__init__()
        ensure_storage_directories()
        uic.loadUi(str(ui_path), self)

        self.play_index = 0
        self.video_path: Path | None = None
        self.video_cap: cv2.VideoCapture | None = None
        self.preview_image_cap: cv2.VideoCapture | None = None
        self.preview_image_cap_path: str = ""
        self._video_frame_cache_index: int = -1
        self._video_frame_cache_frame: np.ndarray | None = None
        self._thumb_source_cache: OrderedDict[tuple[int, str], np.ndarray] = OrderedDict()
        self._thumb_source_cache_limit: int = 24
        self._overlay_boxes_by_frame: OrderedDict[int, list[Any]] = OrderedDict()
        self._overlay_boxes_cache_limit = 720
        self.video_meta: dict[str, Any] = {}
        self.is_processing = False
        self.is_processing_paused = False
        self.is_playing = False
        self.last_worker_frame_index = -1
        self.processing_next_start_frame = 0
        self.worker_thread: QThread | None = None
        self.worker: AutoLabelWorker | None = None
        self.latest_result: WorkerOutput | None = None
        self.available_class_names: list[str] = list(DEFAULT_CLASS_NAMES)
        self.active_class_names: list[str] = list(self.available_class_names)
        self.active_sample_target = 0
        self.last_preview_frame: np.ndarray | None = None
        self.thumbnail_items: list[PreviewThumbnail] = []
        self.pending_thumbnail_items: list[PreviewThumbnail] = []
        self.thumbnail_filter: str = "all"
        self.thumbnail_visible_indices: list[int] = []
        self.btnLoadPreviewCache: QPushButton | None = None
        self.thumb_preview_source_indices: list[int] = []
        self.thumb_preview_dialog: QDialog | None = None
        self.thumb_preview_image_label: QLabel | None = None
        self.thumb_preview_info_label: QLabel | None = None
        self.thumb_preview_prev_button: QPushButton | None = None
        self.thumb_preview_next_button: QPushButton | None = None
        self.thumb_preview_move_keep_button: QPushButton | None = None
        self.thumb_preview_move_drop_button: QPushButton | None = None
        self.thumb_preview_index: int = 0
        self.primary_workbench_widget: QWidget | None = None
        self.page_shell_widget: QWidget | None = None
        self.page_stack: QStackedWidget | None = None
        self.page_button_group: QButtonGroup | None = None
        self.nav_buttons: dict[str, QPushButton] = {}
        self.page_indices: dict[str, int] = {}
        self.current_page_key = "home"
        self.session_status_text = "작업대기"
        self.session_status_raw = ""
        self._page_fade_effect: QGraphicsOpacityEffect | None = None
        self._page_fade_out: QPropertyAnimation | None = None
        self._page_fade_in: QPropertyAnimation | None = None
        self._pending_page_index: int | None = None
        self.labelVideoLoadPathValue: QLabel | None = None
        self.labelVideoLoadMetaValue: QLabel | None = None
        self.labelVideoLoadRoiValue: QLabel | None = None
        self.labelVideoLoadPreview: HoverPreviewLabel | None = None
        self.btnVideoLoadSetRoi: QPushButton | None = None
        self.btnVideoLoadResetRoi: QPushButton | None = None
        self.video_load_hover_timer: QTimer | None = None
        self.video_load_hover_cap: cv2.VideoCapture | None = None
        self.video_load_hover_active = False
        self.video_load_hover_remaining_frames = 0
        self.last_video_load_preview_frame: np.ndarray | None = None
        self.video_roi: tuple[int, int, int, int] | None = None
        self.labelSecondVideoPath: QLabel | None = None
        self.labelSecondVideoMeta: QLabel | None = None
        self.labelSecondVideoPreview: QLabel | None = None
        self.sliderSecondTimeline: QSlider | None = None
        self.labelSecondTimeline: QLabel | None = None
        self.last_second_preview_frame: np.ndarray | None = None
        self.second_stage_stack: QStackedWidget | None = None
        self.labelSecondStageStepState: QLabel | None = None
        self.labelSecondStageStatus: QLabel | None = None
        self.editSecondStageClassName: QLineEdit | None = None
        self.newObjectLabelingWidget: QWidget | None = None
        self.second_stage_labeling_done = False
        self.preview_cache_root: Path | None = None
        self.preview_cache_event_queue: queue.Queue[int] = queue.Queue()
        self.preview_watchdog_observer: Any | None = None
        self.preview_watchdog_handler: Any | None = None
        self.preview_watch_timer = QTimer(self)
        self.preview_watch_timer.setInterval(250)
        self.preview_watch_timer.timeout.connect(self._drain_preview_cache_events)
        self.loading_dialog: QWidget | None = None
        self.loading_label: QLabel | None = None
        self.loading_sub_label: QLabel | None = None
        self.loading_spinner: FuturisticSpinner | None = None
        self.loading_locked_widgets: list[tuple[QWidget, bool]] = []
        self.loading_depth = 0
        self.filter_loading_active = False
        self.progress_target_value = 0
        self.progress_smooth_timer = QTimer(self)
        self.progress_smooth_timer.setInterval(16)
        self.progress_smooth_timer.timeout.connect(self._on_progress_smooth_tick)
        self.preview_render_timer = QTimer(self)
        self.preview_render_timer.setInterval(33)
        self.preview_render_timer.timeout.connect(self._flush_pending_worker_preview)
        self._pending_worker_preview_frame: np.ndarray | None = None
        self._pending_worker_preview_boxes: list[Any] = []
        self.is_exporting = False
        self.export_thread: QThread | None = None
        self.export_worker: DatasetExportWorker | None = None
        self.is_training = False
        self.training_thread: QThread | None = None
        self.training_worker: YoloTrainWorker | None = None
        self.training_merge_thread: QThread | None = None
        self.training_merge_worker: MultiDatasetMergeWorker | None = None
        self.training_merge_active = False
        self.processing_live_playback_enabled = False
        self.last_export_dataset_root: Path | None = None
        self.training_data_yaml_path: Path | None = None
        self.training_data_yaml_paths: list[Path] = []
        self.training_model_source: str = ""
        self.training_engine_name: str = "yolo"
        self.training_mode_name: str = "new"
        self.training_task_name: str = "detect"
        self.training_loading_active = False
        self.training_waiting_epoch_start = False
        self.training_started_at: datetime | None = None
        self.training_args_snapshot: dict[str, Any] = {}
        self._pending_training_request: dict[str, Any] | None = None
        self._training_settings_syncing = False
        self.labelTrainDataset: QLabel | None = None
        self.labelTrainMetricNow: QLabel | None = None
        self.editTrainDataYaml: QLineEdit | None = None
        self.checkTrainNew: QCheckBox | None = None
        self.checkTrainRetrain: QCheckBox | None = None
        self.comboTrainEngine: QComboBox | None = None
        self.comboTrainModel: QComboBox | None = None
        self._training_mode_syncing = False
        self.btnTrainPickDataYaml: QPushButton | None = None
        self.frameTrainRetrainYamlPanel: QFrame | None = None
        self.frameTrainAdvancedOptions: QFrame | None = None
        self.frameTrainFreezeOption: QFrame | None = None
        self.frameTrainRetrainRecipeOptions: QFrame | None = None
        self.labelTrainDataYamlCount: QLabel | None = None
        self.btnTrainAddDataYamls: QPushButton | None = None
        self.btnTrainRemoveDataYaml: QPushButton | None = None
        self.btnTrainClearDataYamls: QPushButton | None = None
        self.listTrainDataYamls: QListWidget | None = None
        self.spinTrainEpochs: QSpinBox | None = None
        self.spinTrainFreeze: QSpinBox | None = None
        self.spinTrainLr0: QDoubleSpinBox | None = None
        self.spinTrainReplayRatio: QDoubleSpinBox | None = None
        self.spinTrainStage1Epochs: QSpinBox | None = None
        self.spinTrainStage2Epochs: QSpinBox | None = None
        self.spinTrainStage2LrFactor: QDoubleSpinBox | None = None
        self.comboTrainUnfreezeMode: QComboBox | None = None
        self.spinTrainReplaySeed: QSpinBox | None = None
        self.spinTrainImgsz: QSpinBox | None = None
        self.spinTrainBatch: QSpinBox | None = None
        self.progressTrainTotal: QProgressBar | None = None
        self.btnTrainStart: QPushButton | None = None
        self.btnTrainStop: QPushButton | None = None
        # Feature 1: 로컬/서버 학습 전환
        self.checkTrainLocal: QCheckBox | None = None
        self.labelTrainLocalDevice: QLabel | None = None
        self._yolo_extra_params: dict[str, Any] = {}
        # Feature 2: 고급 파라미터 다이얼로그 버튼
        self.btnTrainAdvancedParams: QPushButton | None = None
        self.textTrainLog: QTextEdit | None = None
        self.trainingBoxLossChart: MetricChartWidget | None = None
        self.trainingClsLossChart: MetricChartWidget | None = None
        self.trainingDflLossChart: MetricChartWidget | None = None
        self.trainingAccChart: MetricChartWidget | None = None
        self._train_progress_anchor_pos: int | None = None
        self._train_progress_line_text: str = ""
        self._train_progress_last_key: tuple[int, int, int, int] | None = None
        self.is_model_testing = False
        self.model_test_thread: QThread | None = None
        self.model_test_worker: ModelTestWorker | None = None
        self.model_test_model_path: Path | None = None
        self.model_test_video_path: Path | None = None
        self.labelModelTestStatus: QLabel | None = None
        self.labelModelTestFrame: QLabel | None = None
        self.editModelTestModel: QLineEdit | None = None
        self.editModelTestVideo: QLineEdit | None = None
        self.btnModelTestPickModel: QPushButton | None = None
        self.btnModelTestPickVideo: QPushButton | None = None
        self.btnModelTestStart: QPushButton | None = None
        self.btnModelTestStop: QPushButton | None = None
        self.spinModelTestConf: QDoubleSpinBox | None = None
        self.spinModelTestIou: QDoubleSpinBox | None = None
        self.spinModelTestImgsz: QSpinBox | None = None
        self.progressModelTest: QProgressBar | None = None
        self.textModelTestLog: QTextEdit | None = None
        self.labelModelTestPreview: QLabel | None = None

        self._bind_widgets()
        self._remove_model_setting_card()
        self._setup_class_selector()
        self._setup_user_input_fields()
        self._setup_processing_mode_buttons()
        self._setup_thumbnail_filter_controls()
        self._relocate_log_card_under_start_stop()
        self._rebuild_content_layout_for_full_height_sidebar()
        self._setup_progress_panel()
        self._setup_play_timer()
        self._setup_video_load_hover_preview()
        self._setup_page_navigation_shell()
        self._connect_signals()
        self._apply_initial_state()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        """창 크기 변경 시 현재 프레임/미리보기를 새 크기에 맞게 다시 표시"""
        super().resizeEvent(event)
        if self.loading_dialog is not None and self.loading_dialog.isVisible():
            self.loading_dialog.setGeometry(self.rect())
        if self.last_preview_frame is not None and self.labelVideoPlaceholder is not None:
            self.labelVideoPlaceholder.setPixmap(self._frame_to_pixmap(self.last_preview_frame))
        if self.last_video_load_preview_frame is not None and self.labelVideoLoadPreview is not None:
            self.labelVideoLoadPreview.setPixmap(
                self._frame_to_pixmap(self.last_video_load_preview_frame, max_width=920, max_height=520)
            )
        if self.last_second_preview_frame is not None and self.labelSecondVideoPreview is not None:
            self.labelSecondVideoPreview.setPixmap(
                self._frame_to_pixmap(self.last_second_preview_frame, max_width=960, max_height=560)
            )

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        """키보드 단축키 입력을 처리하고 재생/탐색 등 대응 동작을 실행"""
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_F11):
            event.ignore()
            return
        super().keyPressEvent(event)

    def _release_video_cap(self) -> None:
        """열려 있는 비디오 캡처 객체를 해제하고 핸들을 초기화"""
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self._clear_video_frame_cache()
        self._release_preview_image_cap()

    def _quit_thread(self, thread: QThread | None, timeout_ms: int = 5000) -> None:
        ## =====================================
        ## 함수 기능 : QThread를 안전하게 종료하고 지정 시간(ms)만큼 대기합니다
        ## 매개 변수 : thread(QThread|None), timeout_ms(int)
        ## 반환 결과 : None
        ## =====================================
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(timeout_ms)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """창 종료 시 타이머/비디오 캡처/워커 리소스를 안전하게 정리한 뒤 종료"""
        widget = getattr(self, "newObjectLabelingWidget", None)
        if widget is not None and hasattr(widget, "cleanup_threads"):
            widget.cleanup_threads()
        for worker in (self.worker, self.training_worker, self.model_test_worker):
            if worker is not None and hasattr(worker, "request_stop"):
                worker.request_stop()
        for thread in (
            self.worker_thread,
            self.export_thread,
            self.training_thread,
            self.training_merge_thread,
            self.model_test_thread,
        ):
            self._quit_thread(thread)

        # 스레드가 아직 살아있으면 창을 닫지 않고 재시도하게 해 QThread destroy 에러를 방지합니다.
        all_threads = (
            self.worker_thread,
            self.export_thread,
            self.training_thread,
            self.training_merge_thread,
            self.model_test_thread,
        )
        if any(t is not None and t.isRunning() for t in all_threads):
            self._append_log("error: 종료 대기 중입니다. 잠시 후 다시 종료하세요.")
            event.ignore()
            return

        self.play_timer.stop()
        self.preview_watch_timer.stop()
        if self.loading_spinner is not None:
            self.loading_spinner.stop()
        if self.loading_dialog is not None:
            self.loading_dialog.hide()
        self._unlock_interactions_after_loading()
        self.preview_render_timer.stop()
        self._pending_worker_preview_frame = None
        self._pending_worker_preview_boxes = []
        if QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        self._stop_video_load_hover_preview()
        self._release_preview_image_cap()
        self._release_video_cap()
        self._clear_preview_cache()
        super().closeEvent(event)

