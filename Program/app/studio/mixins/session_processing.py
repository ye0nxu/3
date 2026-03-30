from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PyQt6.QtCore import QCoreApplication, QEventLoop, QPoint, QRect, QThread, Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QSlider,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.studio.config import TEAM_MODEL_PATH
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
from app.studio.runtime import cv2
from app.studio.utils import _to_korean_class_name, _to_korean_stage
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
from app.studio.workers import AutoLabelWorker
from app.ui.widgets.studio_support import FuturisticSpinner


class StudioSessionProcessingMixin:
    def _apply_initial_state(self) -> None:
        """설정값 또는 필터를 대상 데이터/상태에 적용하고 후속 상태를 갱신합니다."""
        self._set_session_state("세션 준비")
        self.is_processing_paused = False
        self.last_worker_frame_index = -1
        self.processing_next_start_frame = 0
        self._update_button_state()
        self._update_play_pause_button_icon()
        if self.sliderTimeline is not None:
            self.sliderTimeline.setRange(0, 0)
            self.sliderTimeline.setValue(0)

        self._refresh_video_load_page()
        self._refresh_second_advanced_page()
        self._reset_second_stage_progress()
        self._load_training_ui_settings()
        self._update_training_mode_ui()
        self._navigate_to_page("home", animate=False)

    def _update_button_state(self) -> None:
        """현재 데이터와 상태를 기준으로 UI 표시값 또는 내부 상태를 동기화합니다."""
        has_video = self.video_path is not None
        has_result = self.latest_result is not None and bool(self.latest_result.frame_annotations)
        has_cached_preview = self._has_cached_preview_items()
        has_export_folder = bool(
            self.last_export_dataset_root is not None
            and Path(self.last_export_dataset_root).is_dir()
        )
        busy = self.is_processing or self.is_exporting or self.is_training or self.is_model_testing

        if self.btnOpenVideo is not None:
            self.btnOpenVideo.setEnabled((not busy) and has_export_folder)
        if self.btnStart is not None:
            self.btnStart.setEnabled(has_video and (not busy))
        if self.btnStartFast is not None:
            self.btnStartFast.setEnabled(has_video and (not busy))
        if self.btnStop is not None:
            if self.is_processing:
                self.btnStop.setEnabled(True)
                self.btnStop.setText("재개" if self.is_processing_paused else "중지")
            else:
                self.btnStop.setEnabled(False)
                self.btnStop.setText("중지")
        if self.btnExportResult is not None:
            self.btnExportResult.setEnabled(has_result and (not busy))
        if self.btnLoadPreviewCache is not None:
            self.btnLoadPreviewCache.setEnabled((not busy) and has_cached_preview)
        if self.btnVideoLoadSetRoi is not None:
            self.btnVideoLoadSetRoi.setEnabled(has_video and (not busy) and (not self.is_playing))
        if self.btnVideoLoadResetRoi is not None:
            self.btnVideoLoadResetRoi.setEnabled(has_video and (not busy) and (not self.is_playing))
        if self.btnPickTeamModel is not None:
            self.btnPickTeamModel.setEnabled(not busy)

        playback_enabled = has_video and (not self.is_exporting) and ((not self.is_processing) or self.is_processing_paused)
        for button in [self.btnSeekBack, self.btnPrevFrame, self.btnNextFrame]:
            if button is not None:
                button.setEnabled(playback_enabled)
        if self.btnPlayPause is not None:
            self.btnPlayPause.setEnabled(has_video and (not busy))
        self._update_play_pause_button_icon()
        if self.sliderTimeline is not None:
            self.sliderTimeline.setEnabled(playback_enabled)
        if self.sliderSecondTimeline is not None:
            self.sliderSecondTimeline.setEnabled(playback_enabled)
        if self.loading_depth > 0:
            self._lock_interactions_for_loading()
        self._update_training_ui_state()
        self._update_model_test_ui_state()

    def _has_cached_preview_items(self) -> bool:
        """불러오기 대기(메모리/디스크) keep/hold/drop 항목이 하나라도 있으면 True를 반환합니다."""
        if self.pending_thumbnail_items:
            return True
        if self.preview_cache_root is None or (not self.preview_cache_root.is_dir()):
            return False
        for category in ("keep", "hold", "drop"):
            category_dir = self.preview_cache_root / category
            if not category_dir.is_dir():
                continue
            if any(category_dir.glob("*.json")):
                return True
        return False

    def _count_cached_preview_items(self) -> int:
        """불러오기 대기(메모리/디스크) keep/hold/drop 항목 총 개수를 집계해 반환합니다."""
        if self.pending_thumbnail_items:
            return int(len(self.pending_thumbnail_items))
        if self.preview_cache_root is None or (not self.preview_cache_root.is_dir()):
            return 0
        count = 0
        for category in ("keep", "hold", "drop"):
            category_dir = self.preview_cache_root / category
            if not category_dir.is_dir():
                continue
            count += sum(1 for _ in category_dir.glob("*.json"))
        return count

    def _lock_interactions_for_loading(self) -> None:
        """로딩 중에는 입력 위젯 상호작용을 잠급니다."""
        lockable_types = (
            QAbstractButton,
            QLineEdit,
            QPlainTextEdit,
            QTextEdit,
            QComboBox,
            QSlider,
            QListWidget,
            QTableWidget,
            QDoubleSpinBox,
        )

        for widget in self.findChildren(QWidget):
            if not isinstance(widget, lockable_types):
                continue
            if self.loading_dialog is not None:
                if widget is self.loading_dialog or self.loading_dialog.isAncestorOf(widget):
                    continue
            already_tracked = any(locked_widget is widget for locked_widget, _ in self.loading_locked_widgets)
            if not already_tracked:
                self.loading_locked_widgets.append((widget, bool(widget.isEnabled())))
            if widget.isEnabled():
                widget.setEnabled(False)

    def _unlock_interactions_after_loading(self) -> None:
        """로딩 종료 후 입력 위젯의 원래 활성화 상태를 복원합니다."""
        if not self.loading_locked_widgets:
            return
        for widget, was_enabled in reversed(self.loading_locked_widgets):
            try:
                widget.setEnabled(bool(was_enabled))
            except Exception:
                pass
        self.loading_locked_widgets = []

    def _process_ui_keep_alive(self) -> None:
        """장시간 작업 중 사용자 입력을 제외한 UI 이벤트만 처리해 무응답 표시를 방지합니다."""
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def _begin_loading(self, message: str) -> None:
        """장시간 작업 중 메인 UI 위 오버레이 로딩 패널을 표시합니다."""
        self.loading_depth += 1
        if self.loading_depth > 1:
            if self.loading_label is not None:
                self.loading_label.setText(str(message))
            self._lock_interactions_for_loading()
            self._process_ui_keep_alive()
            return

        if self.loading_dialog is None:
            overlay = QWidget(self)
            overlay.setObjectName("loadingOverlay")
            overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            overlay.setStyleSheet(
                """
                QWidget#loadingOverlay {
                    background-color: rgba(6, 10, 18, 152);
                }
                QFrame#loadingPanel {
                    background-color: rgba(11, 17, 28, 242);
                    border: 1px solid rgba(82, 226, 255, 160);
                    border-radius: 18px;
                }
                QLabel#futuristicLoadingTitle {
                    color: rgb(227, 248, 255);
                    font-size: 12pt;
                    font-weight: 700;
                }
                QLabel#futuristicLoadingMessage {
                    color: rgba(197, 234, 255, 215);
                    font-size: 10pt;
                }
                """
            )
            overlay_layout = QVBoxLayout(overlay)
            overlay_layout.setContentsMargins(0, 0, 0, 0)
            overlay_layout.setSpacing(0)
            overlay_layout.addStretch(1)

            panel = QFrame(overlay)
            panel.setObjectName("loadingPanel")
            panel.setFixedSize(392, 214)
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(18, 16, 18, 16)
            panel_layout.setSpacing(10)

            spinner = FuturisticSpinner(panel, size=82)
            title = QLabel("SYSTEM ACTIVE", panel)
            title.setObjectName("futuristicLoadingTitle")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            panel_layout.addWidget(title)

            spinner_row = QHBoxLayout()
            spinner_row.addStretch(1)
            spinner_row.addWidget(spinner, 0, Qt.AlignmentFlag.AlignCenter)
            spinner_row.addStretch(1)
            panel_layout.addLayout(spinner_row)

            label = QLabel(panel)
            label.setObjectName("futuristicLoadingMessage")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            panel_layout.addWidget(label)

            sub_label = QLabel("작업이 끝날 때까지 잠시만 기다려주세요.", panel)
            sub_label.setObjectName("futuristicLoadingMessage")
            sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            panel_layout.addWidget(sub_label)

            overlay_layout.addWidget(panel, 0, Qt.AlignmentFlag.AlignCenter)
            overlay_layout.addStretch(1)

            self.loading_dialog = overlay
            self.loading_label = label
            self.loading_sub_label = sub_label
            self.loading_spinner = spinner

        if self.loading_label is not None:
            self.loading_label.setText(str(message))
        if self.loading_sub_label is not None:
            self.loading_sub_label.setText("엔진 상태 확인 중...")
        if self.loading_spinner is not None:
            self.loading_spinner.start()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        if self.loading_dialog is not None:
            self.loading_dialog.setGeometry(self.rect())
            self.loading_dialog.show()
            self.loading_dialog.raise_()
        self._lock_interactions_for_loading()
        self._process_ui_keep_alive()

    def _end_loading(self) -> None:
        """중첩 로딩 상태를 해제하고 마지막 작업 종료 시 로딩 다이얼로그를 닫습니다."""
        if self.loading_depth <= 0:
            return
        self.loading_depth -= 1
        if self.loading_depth > 0:
            self._process_ui_keep_alive()
            return
        self.loading_depth = 0
        if self.loading_spinner is not None:
            self.loading_spinner.stop()
        if self.loading_dialog is not None:
            self.loading_dialog.hide()
        self._unlock_interactions_after_loading()
        if QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        self._update_button_state()
        self._process_ui_keep_alive()

    def _append_log(self, message: str) -> None:
        """현재 시각 타임스탬프를 붙여 로그 텍스트 패널에 한 줄을 추가합니다."""
        if self.textLog is None:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        self.textLog.append(f"[{stamp}] {message}")

    def _show_modal_message(self, title: str, text: str, level: str = "info") -> None:
        """메시지박스를 전면/모달로 표시해 사용자가 완료 상태를 놓치지 않게 합니다."""
        dialog = QMessageBox(self)
        dialog.setWindowTitle(str(title))
        dialog.setText(str(text))
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        level_key = str(level).strip().lower()
        if level_key in {"warn", "warning"}:
            dialog.setIcon(QMessageBox.Icon.Warning)
        elif level_key in {"error", "critical"}:
            dialog.setIcon(QMessageBox.Icon.Critical)
        else:
            dialog.setIcon(QMessageBox.Icon.Information)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        dialog.exec()

    def _format_class_list_for_ui(self, class_names: Sequence[str]) -> str:
        """내부 값을 사용자 표시용 문자열 형식으로 변환해 반환합니다."""
        labels = [_to_korean_class_name(str(name)) for name in class_names if str(name).strip()]
        if not labels:
            return "-"
        if len(labels) <= 8:
            return ", ".join(labels)
        return f"{', '.join(labels[:8])} 외 {len(labels) - 8}개"

    def _normalize_session_status_text(self, raw_state: str) -> str:
        """내부 상태 문자열을 UI 표준 상태(작업대기/작업 중/작업 완료)로 정규화합니다."""
        text = str(raw_state).strip()
        if not text:
            return "작업대기"
        if "완료" in text:
            return "작업 완료"
        idle_keywords = ("대기", "준비", "중지", "정지")
        if any(keyword in text for keyword in idle_keywords):
            return "작업대기"
        busy_keywords = ("중", "요청", "단계", "로드", "저장", "실행", "병합", "검증", "학습")
        if any(keyword in text for keyword in busy_keywords):
            return "작업 중"
        return "작업대기"

    def _session_status_color(self, normalized_state: str) -> str:
        key = str(normalized_state).strip()
        if key == "작업 중":
            return "#FF3B30"
        if key == "작업 완료":
            return "#2ECC71"
        return "#1E6FFF"

    def _reapply_session_status_style(self) -> None:
        if self.labelSessionState is None:
            return
        status_text = str(self.session_status_text or "작업대기")
        color = self._session_status_color(status_text)
        self.labelSessionState.setText(status_text)
        self.labelSessionState.setStyleSheet(f"color: {color}; font-weight: 700;")

    def set_session_status(self, state: str) -> None:
        """세션 상태 텍스트와 색상을 함께 갱신합니다."""
        self.session_status_raw = str(state)
        self.session_status_text = self._normalize_session_status_text(self.session_status_raw)
        if self.labelSessionState is not None:
            self.labelSessionState.setToolTip(self.session_status_raw)
        self._reapply_session_status_style()

    def _set_session_state(self, text: str) -> None:
        """기존 호출부 호환을 위한 세션 상태 업데이트 래퍼입니다."""
        self.set_session_status(str(text))

    def _reset_progress_panel(self) -> None:
        """작업 중 누적된 상태를 초기값으로 되돌려 다음 실행을 준비합니다."""
        self.labelProgressStage.setText("현재 단계: 대기")
        self.labelProgressFrame.setText("진행 - / -")
        self.labelProgressRemain.setText("남은 예상: -")
        self._set_progress_target(0, allow_decrease=True)

    def _set_progress_target(self, percent: int, allow_decrease: bool = False) -> None:
        """진행률 목표값을 갱신하고 스무딩 타이머를 통해 자연스럽게 반영합니다."""
        target = max(0, min(100, int(percent)))
        if (not allow_decrease) and target < self.progress_target_value:
            target = self.progress_target_value
        self.progress_target_value = target
        if self.progressPipeline is None:
            return
        if allow_decrease:
            self.progressPipeline.setValue(target)
        if not self.progress_smooth_timer.isActive():
            self.progress_smooth_timer.start()

    def _on_progress_smooth_tick(self) -> None:
        """현재 진행률 표시값을 목표값으로 점진 이동시켜 부드러운 애니메이션을 만듭니다."""
        if self.progressPipeline is None:
            self.progress_smooth_timer.stop()
            return
        current = float(self.progressPipeline.value())
        target = float(self.progress_target_value)
        diff = target - current
        if abs(diff) < 0.6:
            self.progressPipeline.setValue(int(round(target)))
            if int(round(target)) == self.progress_target_value:
                self.progress_smooth_timer.stop()
            return
        step = max(0.45, abs(diff) * 0.22)
        next_value = current + (step if diff > 0 else -step)
        if diff > 0 and next_value > target:
            next_value = target
        if diff < 0 and next_value < target:
            next_value = target
        self.progressPipeline.setValue(int(round(next_value)))

    def _compute_worker_progress_percent(self, event: ProgressEvent) -> int:
        """워커 이벤트를 바탕으로 진행률 퍼센트를 계산합니다(샘플 목표 기반 보정 포함)."""
        total_units = max(1, int(event.total_units))
        processed_units = max(0, min(int(event.processed_units), total_units))
        unit_ratio = float(processed_units) / float(total_units)

        sample_target = max(0, int(self.active_sample_target))
        remaining_items = max(0, int(event.remaining_items))
        sample_ratio: float | None = None
        if sample_target > 0 and remaining_items <= sample_target:
            sample_ratio = float(sample_target - remaining_items) / float(sample_target)

        if sample_ratio is not None and total_units >= sample_target * 4:
            ratio = max(unit_ratio, sample_ratio)
        else:
            ratio = unit_ratio

        return int(round(max(0.0, min(1.0, ratio)) * 100.0))

    def on_open_video(self) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "영상 선택",
            str(Path.home()),
            "영상 파일 (*.mp4 *.avi *.mov *.mkv);;모든 파일 (*)",
        )
        if not file_path:
            return

        self._load_video(Path(file_path))

    def on_open_export_folder(self) -> None:
        """최근 내보내기 저장 폴더를 파일 탐색기로 엽니다."""
        folder = self.last_export_dataset_root
        if folder is None:
            QMessageBox.information(self, "저장 폴더 열기", "먼저 결과 내보내기를 실행하세요.")
            return
        target = Path(folder)
        if not target.is_dir():
            QMessageBox.warning(self, "저장 폴더 열기", f"폴더가 없습니다:\n{target}")
            return

        opened = False
        if os.name == "nt":
            try:
                os.startfile(str(target))  # type: ignore[attr-defined]
                opened = True
            except Exception:
                opened = False
        if not opened:
            try:
                from PyQt6.QtCore import QUrl
                from PyQt6.QtGui import QDesktopServices
                opened = bool(QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))))
            except Exception:
                opened = False
        if not opened:
            QMessageBox.warning(self, "저장 폴더 열기", f"탐색기를 열지 못했습니다:\n{target}")
            return
        self._append_log(f"user action: export folder opened ({target})")

    def _load_video(self, path: Path) -> None:
        """선택한 영상 파일을 로드하고 메타데이터, 타임라인, 초기 프레임 표시 상태를 초기화합니다."""
        self._stop_video_load_hover_preview()
        self._clear_video_frame_cache()
        self._clear_thumbnail_source_cache()
        self._release_preview_image_cap()
        self.play_timer.stop()
        self.is_playing = False
        self._update_play_pause_button_icon()

        self._release_video_cap()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            QMessageBox.warning(self, "영상 열기", f"영상을 열지 못했습니다:\n{path}")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            fps = 30.0

        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            cap.release()
            QMessageBox.warning(self, "영상 열기", f"첫 프레임을 읽지 못했습니다:\n{path}")
            return

        self.video_path = path
        self.video_cap = cap
        self.video_meta = {
            "frame_count": max(1, frame_count),
            "fps": fps,
            "width": width,
            "height": height,
        }
        self.video_roi = None
        self._sync_play_timer_to_video_fps()
        self.latest_result = None
        self.last_preview_frame = None
        self.is_processing_paused = False
        self.last_worker_frame_index = -1
        self.processing_next_start_frame = 0
        self._clear_preview_cache()
        self._reset_progress_panel()
        if self.listThumbs is not None:
            self.listThumbs.clear()

        if self.editVideoFileName is not None:
            self.editVideoFileName.setText(str(path))
        if self.sliderTimeline is not None:
            self.sliderTimeline.blockSignals(True)
            self.sliderTimeline.setRange(0, max(0, self.video_meta["frame_count"] - 1))
            self.sliderTimeline.setValue(0)
            self.sliderTimeline.blockSignals(False)

        self._show_frame(self._apply_video_roi_to_frame(first_frame), [])
        self._update_time_label(0)
        self._set_session_state("영상 로드 완료")
        self._update_video_load_roi_label()
        self._refresh_video_load_page()
        self._refresh_second_advanced_page()
        self._reset_second_stage_progress()
        self._update_button_state()

    def on_timeline_changed(self, value: int) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        if self.is_playing:
            return

        if self.video_cap is None or (self.is_processing and (not self.is_processing_paused)):
            return
        index = max(0, int(value))
        self.processing_next_start_frame = index
        if self.is_processing_paused and self.worker is not None:
            self.worker.request_seek(index)
        frame = self._read_video_frame(index)
        if frame is None:
            return
        self._show_frame(frame, [])
        self._update_time_label(index)
        if not self.video_load_hover_active:
            self._set_video_load_preview_frame(frame)

        self.last_second_preview_frame = frame
        if self.labelSecondVideoPreview is not None:
            self.labelSecondVideoPreview.setPixmap(self._frame_to_pixmap(frame, max_width=960, max_height=560))
            self.labelSecondVideoPreview.setText("")
        if self.sliderSecondTimeline is not None and self.sliderSecondTimeline.value() != index:
            self.sliderSecondTimeline.blockSignals(True)
            self.sliderSecondTimeline.setValue(index)
            self.sliderSecondTimeline.blockSignals(False)
        self._update_second_timeline_label(index)

    def on_stop_playback(self) -> None:
        """비디오 재생을 완전히 멈추고 타임라인을 시작 위치(0프레임)로 이동합니다."""
        if self.video_cap is None:
            return
        if self.is_processing and (not self.is_processing_paused):
            return

        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
        self.play_index = 0
        self.processing_next_start_frame = 0
        self._update_play_pause_button_icon()

        if self.sliderTimeline is not None:
            start_idx = int(self.sliderTimeline.minimum())
            self.sliderTimeline.setValue(start_idx)
            self.on_timeline_changed(start_idx)

    def on_toggle_playback(self) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        if self.video_cap is None or self.is_processing:
            return

        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
            self._update_play_pause_button_icon()
            return
        
        # ✅ 재생 시작 시: 현재 슬라이더 위치로 VideoCapture 위치 맞추기
        if self.sliderTimeline is not None:
            self.play_index = int(self.sliderTimeline.value())
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.play_index))

        self.play_timer.start()
        self.is_playing = True
        self._update_play_pause_button_icon()

    def _on_play_tick(self) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        if self.video_cap is None or self.sliderTimeline is None:
            self.play_timer.stop()
            self.is_playing = False
            return

        maximum = int(self.sliderTimeline.maximum())
        if self.play_index >= maximum:
            self.play_timer.stop()
            self.is_playing = False
            self._update_play_pause_button_icon()
            return

        current_frame_index = int(self.play_index)
        ok, frame = self.video_cap.read()  # ✅ 순차 읽기 (cap.set 금지)
        if not ok or frame is None:
            self.play_timer.stop()
            self.is_playing = False
            return
        self._video_frame_cache_index = current_frame_index
        self._video_frame_cache_frame = frame
        frame = self._apply_video_roi_to_frame(frame)

        self.play_index += 1
        self.processing_next_start_frame = int(self.play_index)

        # ✅ 화면 갱신
        overlay_boxes = self._get_overlay_boxes_for_frame(current_frame_index)
        self._show_frame(frame, overlay_boxes)
        self._update_time_label(self.play_index)

        # ✅ 슬라이더는 값만 반영(시킹 슬롯 호출 금지)
        self.sliderTimeline.blockSignals(True)
        self.sliderTimeline.setValue(self.play_index)
        self.sliderTimeline.blockSignals(False)

        # ✅ 2차 페이지 슬라이더도 같이 이동(프레임 로드는 하지 않음)
        if self.sliderSecondTimeline is not None:
            self.sliderSecondTimeline.blockSignals(True)
            self.sliderSecondTimeline.setValue(self.play_index)
            self.sliderSecondTimeline.blockSignals(False)
            self._update_second_timeline_label(self.play_index)

    def _seek_relative(self, delta: int) -> None:
        """현재 타임라인 위치에서 상대 오프셋만큼 프레임을 이동합니다."""
        if self.sliderTimeline is None or self.video_cap is None or (self.is_processing and (not self.is_processing_paused)):
            return
        current = int(self.sliderTimeline.value())
        new_value = max(int(self.sliderTimeline.minimum()), min(int(self.sliderTimeline.maximum()), current + int(delta)))
        self.sliderTimeline.setValue(new_value)

    def _get_overlay_boxes_for_frame(self, frame_index: int, max_lag: int = 300) -> list[Any]:
        """현재 재생 프레임에 대응하는 오버레이를 반환(정확 일치 우선, 최근/근접 프레임 보정)."""
        idx = max(0, int(frame_index))
        exact = self._overlay_boxes_by_frame.get(idx)
        if isinstance(exact, list):
            return exact
        if not self._overlay_boxes_by_frame:
            return []
        keys = sorted(int(k) for k in self._overlay_boxes_by_frame.keys())

        # 1) prefer nearest past frame (stable for playback)
        past = [k for k in keys if k <= idx]
        if past:
            best_key = past[-1]
            if (idx - best_key) <= max(0, int(max_lag)):
                candidate = self._overlay_boxes_by_frame.get(best_key)
                if isinstance(candidate, list):
                    return candidate

        # 2) fallback to nearest future frame if processing is ahead/behind
        future = [k for k in keys if k > idx]
        if future:
            near_future = future[0]
            if (near_future - idx) <= max(0, int(max_lag)):
                candidate = self._overlay_boxes_by_frame.get(near_future)
                if isinstance(candidate, list):
                    return candidate

        # 3) last resort: most recent cached overlay
        candidate = self._overlay_boxes_by_frame.get(keys[-1])
        return candidate if isinstance(candidate, list) else []

    def on_start_processing(self, fast_mode: bool = False) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        self._end_filter_loading_if_active()
        if self.video_path is None:
            QMessageBox.information(self, "처리 시작", "먼저 영상을 선택하세요.")
            return
        if self.is_processing:
            return
        self._stop_video_load_hover_preview()

        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
            self._update_play_pause_button_icon()

        sample_count = self._parse_sample_count()
        rgb_bits = self._parse_rgb_bits()
        team_model_path = self._parse_team_model_path()
        total_frames = int(self.video_meta.get("frame_count", 0))
        start_frame_index = max(0, int(self.processing_next_start_frame))
        if total_frames > 0 and start_frame_index >= total_frames:
            QMessageBox.information(self, "처리 시작", "이미 영상 끝 지점까지 처리되었습니다.")
            return
        if self.editSampleCount is not None:
            if sample_count > 0:
                self.editSampleCount.setText(str(sample_count))
            else:
                self.editSampleCount.clear()
        if self.editRgbBits is not None:
            rgb_index = self.editRgbBits.findData(rgb_bits)
            if rgb_index >= 0:
                self.editRgbBits.setCurrentIndex(rgb_index)
        self.active_class_names = self._collect_class_names()
        self.active_sample_target = int(sample_count)
        self.last_worker_frame_index = max(-1, start_frame_index - 1)
        self.is_processing_paused = False
        self.processing_live_playback_enabled = False
        self._overlay_boxes_by_frame.clear()
        self._pending_worker_preview_frame = None
        self._pending_worker_preview_boxes = []
        self.preview_render_timer.stop()
        self.latest_result = None
        self._clear_preview_cache()
        if self.listThumbs is not None:
            self.listThumbs.clear()
        self._reset_progress_panel()

        self._append_log("start")
        self._append_log(f"start frame: {start_frame_index + 1}")
        self._append_log(f"mode: {'fast' if fast_mode else 'sync'}")
        if fast_mode:
            self._append_log("fast profile: preview off, full-frame processing")
        if sample_count <= 0:
            self._append_log("sample target: end-of-video")
        else:
            self._append_log(f"sample target: {sample_count}")
        self._append_log(f"detect target: {self._format_class_list_for_ui(self.active_class_names)}")
        self._append_log(f"model: {team_model_path.name}")
        processing_roi = self._normalize_video_roi(
            self.video_roi,
            int(self.video_meta.get("width", 0)),
            int(self.video_meta.get("height", 0)),
        )
        if processing_roi is None:
            self._append_log("roi: full frame")
        else:
            rx, ry, rw, rh = processing_roi
            self._append_log(f"roi: x={rx}, y={ry}, w={rw}, h={rh}")
        self._set_session_state("처리 중")

        # 고급 설정 다이얼로그에서 모든 임계치 읽기
        adv = {}
        if getattr(self, "advancedSettingsDialog", None) is not None:
            adv = self.advancedSettingsDialog.get_values()
        yolo_conf   = float(adv.get("yolo_conf",   0.10))
        yolo_iou    = float(adv.get("yolo_iou",    0.45))
        yolo_imgsz  = int(adv.get("yolo_imgsz",    320))
        self._append_log(f"yolo: conf={yolo_conf:.2f}, iou={yolo_iou:.2f}, imgsz={yolo_imgsz}")
        self._append_log(
            f"track: conf_high={adv.get('track_conf_high', 0.20):.2f}, "
            f"val_frames={adv.get('track_validation_frames', 5)}, "
            f"border={adv.get('border_margin', 5)}"
        )

        self.worker_thread = QThread(self)
        self.worker = AutoLabelWorker(
            video_path=self.video_path,
            sample_count=sample_count,
            class_names=self.active_class_names,
            rgb_bits=rgb_bits,
            start_frame_index=start_frame_index,
            realtime_mode=(not fast_mode),
            roi_rect=processing_roi,
            team_model_path=team_model_path,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            yolo_imgsz=yolo_imgsz,
            track_conf_high=float(adv.get("track_conf_high", 0.20)),
            track_validation_frames=int(adv.get("track_validation_frames", 5)),
            track_iou_threshold=float(adv.get("track_iou_threshold", 0.50)),
            track_size_diff_threshold=float(adv.get("track_size_diff_threshold", 0.20)),
            track_area_change_limit=float(adv.get("track_area_change_limit", 0.30)),
            track_ratio_change_limit=float(adv.get("track_ratio_change_limit", 0.30)),
            track_hold_frames=int(adv.get("track_hold_frames", 5)),
            border_margin=int(adv.get("border_margin", 5)),
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.stage_changed.connect(self._on_worker_stage_changed)
        self.worker.progress_changed.connect(self._on_worker_progress)
        self.worker.log_message.connect(self._on_worker_log_message)
        self.worker.preview_ready.connect(self._on_worker_preview)
        self.worker.pause_state_changed.connect(self._on_worker_pause_state_changed)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.failed.connect(self._on_worker_failed)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._on_worker_thread_finished)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.is_processing = True
        self._update_button_state()
        self.worker_thread.start()

    def on_stop_processing(self) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        if self.worker is None or (not self.is_processing):
            return
        if self.is_processing_paused:
            self._append_log("user action: resume requested")
            self.is_processing_paused = False
            self._set_session_state("재개 요청")
            self._update_button_state()
            self.worker.request_seek(max(0, int(self.processing_next_start_frame)))
            self.worker.request_resume()
        else:
            self._append_log("user action: pause requested")
            self.is_processing_paused = True
            self._set_session_state("일시정지 요청")
            self._update_button_state()
            self.worker.request_pause()

    def _on_worker_stage_changed(self, stage: str) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        stage_text = _to_korean_stage(stage)
        self.labelProgressStage.setText(f"현재 단계: {stage_text}")
        self._set_session_state(f"단계: {stage_text}")

    def _on_worker_pause_state_changed(self, paused: bool) -> None:
        """워커 일시정지 상태 변화에 맞춰 UI 버튼/세션 상태를 동기화합니다."""
        self.is_processing_paused = bool(paused)
        if self.is_processing_paused:
            self._set_session_state("일시정지")
            self._append_log("pause")
        else:
            if self.is_processing:
                self._set_session_state("처리 중")
            self._append_log("resume")
        self._update_button_state()

    def _on_worker_progress(self, event: object) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        if not isinstance(event, ProgressEvent):
            return 

        percent = self._compute_worker_progress_percent(event)
        self._set_progress_target(percent)
        current_frame = max(0, int(event.current_frame))
        total_frames = max(0, int(event.total_frames))
        if current_frame > self.last_worker_frame_index:
            self.last_worker_frame_index = current_frame
        if total_frames > 0:
            self.processing_next_start_frame = min(total_frames, current_frame + 1)

        self.labelProgressFrame.setText(
            f"진행 {int(event.processed_units)}/{max(1, int(event.total_units))} | "
            f"프레임 {current_frame + 1}/{total_frames}"
        )
        self.labelProgressRemain.setText(
            f"남은 예상: {event.remaining_items} | 예상 시간 {self._format_eta(event.eta_seconds)}"
        )
        self._update_time_label(current_frame)

        if self.sliderTimeline is not None and total_frames > 0:
            display_index = max(0, min(total_frames - 1, current_frame))
            if self.sliderTimeline.value() != display_index:
                self.sliderTimeline.blockSignals(True)
                self.sliderTimeline.setValue(display_index)
                self.sliderTimeline.blockSignals(False)
        if self.sliderSecondTimeline is not None and total_frames > 0:
            display_index = max(0, min(total_frames - 1, current_frame))
            if self.sliderSecondTimeline.value() != display_index:
                self.sliderSecondTimeline.blockSignals(True)
                self.sliderSecondTimeline.setValue(display_index)
                self.sliderSecondTimeline.blockSignals(False)
            self._update_second_timeline_label(display_index)

    def _sync_processing_cursor_after_worker(self) -> None:
        """워커 종료/중단 후 다음 시작 프레임 커서를 마지막 처리 위치 기준으로 보정합니다."""
        total_frames = int(self.video_meta.get("frame_count", 0))
        if total_frames <= 0:
            return

        if self.last_worker_frame_index >= 0:
            self.processing_next_start_frame = min(total_frames, self.last_worker_frame_index + 1)
        else:
            self.processing_next_start_frame = max(
                0,
                min(total_frames, int(self.processing_next_start_frame)),
            )

        display_index = min(total_frames - 1, max(0, self.processing_next_start_frame))
        if self.sliderTimeline is not None and self.sliderTimeline.value() != display_index:
            self.sliderTimeline.blockSignals(True)
            self.sliderTimeline.setValue(display_index)
            self.sliderTimeline.blockSignals(False)
        if self.sliderSecondTimeline is not None and self.sliderSecondTimeline.value() != display_index:
            self.sliderSecondTimeline.blockSignals(True)
            self.sliderSecondTimeline.setValue(display_index)
            self.sliderSecondTimeline.blockSignals(False)
        self._update_time_label(display_index)
        self._update_second_timeline_label(display_index)

    def _on_worker_log_message(self, message: str) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        text = str(message or "").strip()
        if not text:
            return

        if text == "[필터링] duplicate start":
            if not self.filter_loading_active:
                self.filter_loading_active = True
                self._begin_loading("중복 제거 처리 중입니다...")
            self._append_log(text)
            return

        if text == "[필터링] duplicate done":
            if self.filter_loading_active:
                self.filter_loading_active = False
                self._end_loading()
            self._append_log(text)
            return

        if text.startswith("[필터링]") or "추적 필터 결과" in text:
            self._append_log(text)
            return

        if "예외" in text or "실패" in text or "오류" in text:
            self._append_log(f"error: {text}")
            return

        # 기본적으로 워커 로그는 표시해 흐름을 확인할 수 있도록 유지합니다.
        self._append_log(text)

    def _end_filter_loading_if_active(self) -> None:
        """중복 제거 로딩 오버레이가 남아있으면 종료 시점에 안전하게 해제합니다."""
        if not self.filter_loading_active:
            return
        self.filter_loading_active = False
        self._end_loading()

    def _on_worker_preview(self, frame_obj: object, boxes_obj: object, frame_index: int) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        boxes = boxes_obj if isinstance(boxes_obj, list) else []
        idx = max(0, int(frame_index))
        self._overlay_boxes_by_frame[idx] = boxes
        self._overlay_boxes_by_frame.move_to_end(idx)
        while len(self._overlay_boxes_by_frame) > int(self._overlay_boxes_cache_limit):
            self._overlay_boxes_by_frame.popitem(last=False)

        if not isinstance(frame_obj, np.ndarray):
            return
        # Keep strict sync with worker output: render every emitted frame.
        self._show_frame(frame_obj, boxes)

    def _flush_pending_worker_preview(self) -> None:
        """워커에서 들어온 최신 프리뷰 프레임만 주기적으로 렌더링해 UI 끊김을 줄입니다."""
        frame = self._pending_worker_preview_frame
        if frame is None:
            if not self.is_processing:
                self.preview_render_timer.stop()
            return
        boxes = self._pending_worker_preview_boxes
        self._pending_worker_preview_frame = None
        self._pending_worker_preview_boxes = []
        self._show_frame(frame, boxes)

    def _on_worker_finished(self, payload: object) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        self._end_filter_loading_if_active()
        self.processing_live_playback_enabled = False
        self._overlay_boxes_by_frame.clear()
        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
            self._update_play_pause_button_icon()
        self.preview_render_timer.stop()
        self._flush_pending_worker_preview()
        if not isinstance(payload, WorkerOutput):
            self._on_worker_failed("작업자 결과 형식이 올바르지 않습니다.")
            return

        self.is_processing_paused = False
        self.latest_result = payload
        self._sync_processing_cursor_after_worker()
        self._set_progress_target(100)
        self.labelProgressRemain.setText("남은 예상: 0 | 예상 시간 00:00")
        self._set_session_state("결과 저장 중")
        self._append_log("finish")
        cached_count = 0
        cache_failed: Exception | None = None
        self._begin_loading("라벨링 결과를 정리하고 있습니다...")
        try:
            cached_count = self._populate_thumbnails(payload.preview_items)
        except Exception as exc:
            cache_failed = exc
        finally:
            self._end_loading()

        if self.latest_result is not None:
            self.latest_result.preview_items = []

        filter_summary_text = ""
        try:
            run_cfg = payload.run_config if isinstance(payload.run_config, dict) else {}
            fs = run_cfg.get("filter_summary", {}) if isinstance(run_cfg, dict) else {}
            if isinstance(fs, dict) and fs:
                blur = fs.get("blur", {}) if isinstance(fs.get("blur", {}), dict) else {}
                exposure = fs.get("exposure", {}) if isinstance(fs.get("exposure", {}), dict) else {}
                yolo_score = fs.get("yolo_score", {}) if isinstance(fs.get("yolo_score", {}), dict) else {}
                final = fs.get("final", {}) if isinstance(fs.get("final", {}), dict) else {}
                dup_drop = int(fs.get("duplicate_drop", 0))
                target_count = int(fs.get("target_count", 0))
                filter_summary_text = (
                    "\n\n[필터링 결과]\n"
                    f"- 대상: {target_count}\n"
                    f"- yolo-score: keep={int(yolo_score.get('keep', 0))}, hold={int(yolo_score.get('hold', 0))}, drop={int(yolo_score.get('drop', 0))}\n"
                    f"- blur: keep={int(blur.get('keep', 0))}, hold={int(blur.get('hold', 0))}, drop={int(blur.get('drop', 0))}\n"
                    f"- exposure: keep={int(exposure.get('keep', 0))}, hold={int(exposure.get('hold', 0))}, drop={int(exposure.get('drop', 0))}\n"
                    f"- duplicate drop: {dup_drop}\n"
                    f"- final: keep={int(final.get('keep', 0))}, hold={int(final.get('hold', 0))}, drop={int(final.get('drop', 0))}"
                )
        except Exception:
            filter_summary_text = ""

        if cache_failed is not None:
            self._set_session_state("처리 완료 (캐시 저장 실패)")
            self._append_log(f"error: preview cache 저장 실패 ({cache_failed})")
            self._show_modal_message(
                "처리 완료",
                "라벨링은 완료되었지만 미리보기 캐시 저장에 실패했습니다.\n"
                f"오류: {cache_failed}{filter_summary_text}",
                level="warning",
            )
        else:
            self._set_session_state("처리 완료")
            if cached_count > 0:
                self._show_modal_message(
                    "라벨링 완료",
                    "객체 라벨링이 완료되었습니다.\n"
                    "keep/hold/drop 미리보기가 자동으로 반영되었습니다.\n\n"
                    f"저장된 항목: {cached_count}개{filter_summary_text}",
                    level="info",
                )
            else:
                self._show_modal_message(
                    "라벨링 완료",
                    "객체 라벨링이 완료되었습니다.\n"
                    "저장된 미리보기 항목이 없어 불러올 목록은 없습니다."
                    f"{filter_summary_text}",
                    level="info",
                )
        self._update_button_state()

    def _on_worker_failed(self, message: str) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        self._end_filter_loading_if_active()
        self.processing_live_playback_enabled = False
        self._overlay_boxes_by_frame.clear()
        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
            self._update_play_pause_button_icon()
        self.preview_render_timer.stop()
        self._pending_worker_preview_frame = None
        self._pending_worker_preview_boxes = []
        self.is_processing_paused = False
        self._set_session_state("중지됨")
        self.active_sample_target = 0
        if "사용자 요청으로 처리가 중지되었습니다." in str(message):
            self._append_log("finish (stopped by user)")
            self._sync_processing_cursor_after_worker()
        else:
            self._append_log(f"error: {message}")
        if "사용자 요청으로 처리가 중지되었습니다." not in message:
            self._show_modal_message("처리 오류", message, level="warning")

    def _on_worker_thread_finished(self) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        self._end_filter_loading_if_active()
        self.processing_live_playback_enabled = False
        self._overlay_boxes_by_frame.clear()
        self.preview_render_timer.stop()
        self._pending_worker_preview_frame = None
        self._pending_worker_preview_boxes = []
        self.is_processing = False
        self.is_processing_paused = False
        self.active_sample_target = 0
        self.worker = None
        self.worker_thread = None
        self._update_button_state()

    def on_export_dataset(self) -> None:
        """내보내기를 백그라운드 스레드로 실행해 UI 멈춤 없이 처리합니다."""
        _on_export_dataset_ops(self)

    def _clone_preview_items_for_export(self, items: Sequence[PreviewThumbnail]) -> list[PreviewThumbnail]:
        """스레드 안전을 위해 preview 항목을 복제해 반환합니다."""
        return _clone_preview_items_for_export_ops(self, items)

    def _clone_frame_annotations_for_export(self, annotations: Sequence[FrameAnnotation]) -> list[FrameAnnotation]:
        """스레드 안전을 위해 프레임 주석을 최소 필드만 복제해 반환합니다."""
        return _clone_frame_annotations_for_export_ops(self, annotations)

    def _on_export_progress(self, message: str, done: int, total: int) -> None:
        """백그라운드 내보내기 진행 상태를 로딩 패널 텍스트에 반영합니다."""
        _on_export_progress_ops(self, message, done, total)

    def _on_export_finished(self, summary_obj: object) -> None:
        """내보내기 성공 시 결과 요약을 표시하고 상태를 정리합니다."""
        _on_export_finished_ops(self, summary_obj)

    def _on_export_failed(self, message: str) -> None:
        """내보내기 실패 시 로딩 상태를 해제하고 오류 메시지를 표시합니다."""
        _on_export_failed_ops(self, message)

    def _on_export_thread_finished(self) -> None:
        """내보내기 워커 스레드 종료 후 핸들을 정리합니다."""
        _on_export_thread_finished_ops(self)

    def on_load_cached_preview(self) -> None:
        """keep/hold/drop 대기 결과를 화면에 수동 반영하고 임시 버퍼를 정리합니다."""
        _on_load_cached_preview_ops(self)

    def _label_for_export_class_id(self, class_id: int) -> str:
        """내부 클래스/ID 값을 내보내기용 라벨 규칙에 맞게 변환해 반환합니다."""
        return _label_for_export_class_id_ops(self, class_id)

    def _to_english_slug(self, text: str, fallback: str = "item") -> str:
        """입력 문자열을 영문 소문자/숫자/밑줄 기반 slug로 정규화해 반환합니다."""
        return _to_english_slug_ops(self, text, fallback)

    def _parse_sample_count(self) -> int:
        """사용자 입력 또는 문자열 값을 파싱하고 허용 범위로 보정해 반환합니다."""
        default_count = 20
        if self.editSampleCount is not None:
            text = self.editSampleCount.text().strip()
            if not text:
                return 0
            try:
                numeric = int(text)
                if numeric <= 0:
                    return 0
                return numeric
            except ValueError:
                pass
        return default_count

    def _parse_rgb_bits(self) -> int:
        """사용자 입력 또는 문자열 값을 파싱하고 허용 범위로 보정해 반환합니다."""
        allowed = [8, 16, 24, 32]
        default_value = 24

        if self.editRgbBits is not None:
            current_data = self.editRgbBits.currentData()
            if current_data is not None:
                try:
                    numeric = int(current_data)
                    return numeric if numeric in allowed else default_value
                except (TypeError, ValueError):
                    pass

            text = self.editRgbBits.currentText().strip()
            if text:
                matched = re.search(r"\d+", text)
                if matched:
                    try:
                        numeric = int(matched.group(0))
                        return numeric if numeric in allowed else min(allowed, key=lambda v: abs(v - numeric))
                    except ValueError:
                        pass
        return default_value

    def _parse_team_model_path(self) -> Path:
        """UI에서 선택한 팀 자동 라벨링 모델 경로를 해석해 반환합니다."""
        selected = getattr(self, "selected_team_model_path", TEAM_MODEL_PATH)
        if isinstance(selected, Path):
            return selected
        try:
            return Path(str(selected))
        except Exception:
            return TEAM_MODEL_PATH

    def _collect_class_names(self) -> list[str]:
        """여러 소스에서 필요한 항목을 수집하고 중복/공백을 정리해 반환합니다."""
        names: list[str] = []
        seen: set[str] = set()
        for _row, edit, _btn in self.class_name_rows:
            if edit is None:
                continue
            label = edit.text().strip()
            if not label:
                continue
            key = label.casefold()
            if key in seen:
                continue
            seen.add(key)
            names.append(label)
        # 클래스명 미입력 시 빈 리스트 반환 → 워커에서 모델의 모든 클래스 사용 (all-class 모드)
        return names

    def _clear_preview_cache(self) -> None:
        """현재 미리보기 캐시/감시 리소스를 정리하고 메모리 목록을 초기화합니다."""
        _clear_preview_cache_ops(self)

    def _delete_preview_cache_storage_after_load(self) -> None:
        """캐시를 메모리로 올린 뒤 디스크 임시 폴더를 삭제하고 메모리 참조만 유지합니다."""
        _delete_preview_cache_storage_after_load_ops(self)

    def _delete_preview_cache_storage_after_export(self) -> None:
        """내보내기 완료 후 디스크 임시 캐시 폴더만 정리하고 메모리 상태는 유지합니다."""
        _delete_preview_cache_storage_after_export_ops(self)

    def _stop_preview_cache_watchdog(self) -> None:
        """동작 중인 watchdog 관찰자를 종료하고 핸들을 정리합니다."""
        _stop_preview_cache_watchdog_ops(self)

    def _drain_preview_cache_events(self) -> None:
        """주기적으로 이벤트 큐를 비우고 변경이 있으면 썸네일 목록을 디스크에서 다시 로드합니다."""
        _drain_preview_cache_events_ops(self)

    def _normalize_thumbnail_category(self, raw: str) -> str:
        """입력 카테고리 문자열을 keep/hold/drop 중 하나로 정규화해 반환합니다."""
        return _normalize_thumbnail_category_ops(self, raw)

    def _persist_preview_items_to_cache(self, payload: WorkerOutput) -> int:
        """워커 결과를 디스크 없이 메모리 대기 목록으로 정규화해 저장하고 항목 수를 반환합니다."""
        return _persist_preview_items_to_cache_ops(self, payload)

    def _load_thumbnail_items_from_cache(self) -> None:
        """디스크 캐시의 keep/hold/drop manifest를 읽어 썸네일 목록 메모리를 재구성합니다."""
        _load_thumbnail_items_from_cache_ops(self)

    def _load_thumbnail_source_image(self, item_data: PreviewThumbnail) -> np.ndarray | None:
        """썸네일/팝업 렌더링에 사용할 원본 이미지를 메모리 또는 디스크에서 읽어 반환합니다."""
        return _load_thumbnail_source_image_ops(self, item_data)

    def _populate_thumbnails(self, preview_items: Sequence[PreviewThumbnail]) -> int:
        """워커 썸네일 메타를 메모리 대기 목록에 저장하고 저장된 항목 수를 반환합니다."""
        return _populate_thumbnails_ops(self, preview_items)

    def _update_thumbnail_filter_tab_counts(self) -> None:
        """현재 데이터와 상태를 기준으로 UI 표시값 또는 내부 상태를 동기화합니다."""
        _update_thumbnail_filter_tab_counts_ops(self)

    def _set_thumbnail_filter(self, key: str) -> None:
        """썸네일 필터 키를 설정하고 필터 조건에 맞게 목록을 다시 구성합니다."""
        _set_thumbnail_filter_ops(self, key)

    def _refresh_thumbnail_list(self) -> None:
        """현재 상태를 기준으로 화면/목록/미리보기를 다시 그려 최신 상태로 갱신합니다."""
        _refresh_thumbnail_list_ops(self)

    def _set_thumbnail_item_category(self, source_idx: int, category: str) -> bool:
        """썸네일 항목 상태를 갱신합니다(캐시가 있으면 manifest 이동, 없으면 메모리 상태만 갱신)."""
        return _set_thumbnail_item_category_ops(self, source_idx, category)

    def _category_matches_filter(self, category: str) -> bool:
        """썸네일 카테고리가 현재 선택된 필터 조건과 일치하는지 판정합니다."""
        return _category_matches_filter_ops(self, category)

    def _on_thumbnail_double_clicked(self, item: QListWidgetItem) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        _on_thumbnail_double_clicked_ops(self, item)

    def _open_thumbnail_preview_dialog(self, start_visible_pos: int) -> None:
        """파일/리소스를 열고 로드 결과에 맞춰 초기 상태를 설정합니다."""
        _open_thumbnail_preview_dialog_ops(self, start_visible_pos)

    def _step_thumbnail_preview_dialog(self, delta: int) -> None:
        """미리보기/탐색 인덱스를 한 단계 이동한 뒤 화면 갱신을 수행합니다."""
        _step_thumbnail_preview_dialog_ops(self, delta)

    def _move_current_thumbnail_in_preview(self, target_category: str) -> None:
        """팝업에서 현재 표시 중인 썸네일 상태를 keep/drop으로 전환하고 목록/카운트를 동기화합니다."""
        _move_current_thumbnail_in_preview_ops(self, target_category)

    def _refresh_thumbnail_preview_dialog(self) -> None:
        """현재 상태를 기준으로 화면/목록/미리보기를 다시 그려 최신 상태로 갱신합니다."""
        _refresh_thumbnail_preview_dialog_ops(self)

    def _show_frame(self, frame: np.ndarray, boxes: Sequence[Any]) -> None:
        """대상 프레임 또는 대화상자를 사용자 화면에 표시합니다."""
        _show_frame_ops(self, frame, boxes)

    def _draw_boxes(self, frame: np.ndarray, boxes: Sequence[Any]) -> np.ndarray:
        """프레임 이미지 위에 박스/폴리곤/클래스 라벨을 오버레이해 시각화 결과를 반환합니다."""
        return _draw_boxes_ops(self, frame, boxes)

    def _extract_box_status(self, box: Any) -> str | None:
        """원본 데이터 구조에서 필요한 필드만 추출해 표준 형식으로 반환합니다."""
        return _extract_box_status_ops(self, box)

    def _extract_polygon_points(self, box: Any) -> list[tuple[int, int]] | None:
        """원본 데이터 구조에서 필요한 필드만 추출해 표준 형식으로 반환"""
        return _extract_polygon_points_ops(self, box)

    def _box_color_for_status(self, status: str | None) -> tuple[int, int, int]:
        """박스 상태(keep/hold/drop 등)에 대응하는 표시 색상을 반환"""
        return _box_color_for_status_ops(self, status)

    def _extract_box_xyxy(self, box: Any) -> tuple[int, int, int, int, int, int | None] | None:
        """원본 데이터 구조에서 필요한 필드만 추출해 표준 형식으로 반환"""
        return _extract_box_xyxy_ops(self, box)

    def _class_name(self, class_id: int) -> str:
        """클래스 ID를 현재 결과/활성 클래스 목록에서 사용자 표시용 이름으로 변환해 반환"""
        return _class_name_ops(self, class_id)

    def _frame_to_pixmap(
        self,
        frame: np.ndarray,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> QPixmap:
        """OpenCV BGR 프레임을 Qt Pixmap으로 변환하고 최대 크기에 맞춰 스케일링"""
        return _frame_to_pixmap_ops(self, frame, max_width, max_height)

    def _read_video_frame_raw(self, frame_index: int) -> np.ndarray | None:
        """원본 VideoCapture에서 지정 프레임을 읽어 그대로 반환합니다."""
        return _read_video_frame_raw_ops(self, frame_index)

    def _read_video_frame(self, frame_index: int) -> np.ndarray | None:
        """지정 프레임을 읽고 현재 선택 ROI를 적용한 화면/처리용 프레임을 반환합니다."""
        return _read_video_frame_ops(self, frame_index)

    def _ensure_preview_image_cap(self) -> cv2.VideoCapture | None:
        """미리보기 팝업용 전용 VideoCapture를 준비해 반환합니다."""
        return _ensure_preview_image_cap_ops(self)

    def _release_preview_image_cap(self) -> None:
        """미리보기 팝업 전용 VideoCapture를 해제합니다."""
        _release_preview_image_cap_ops(self)

    def _clear_video_frame_cache(self) -> None:
        """랜덤 접근용 최근 프레임 캐시를 초기화합니다."""
        _clear_video_frame_cache_ops(self)

    def _clear_thumbnail_source_cache(self) -> None:
        """썸네일/팝업 원본 프레임 캐시를 비웁니다."""
        _clear_thumbnail_source_cache_ops(self)

    def _remember_thumbnail_source_cache(self, cache_key: tuple[int, str], frame: np.ndarray) -> None:
        """썸네일 소스 프레임을 LRU 캐시에 저장하고 용량 상한을 유지합니다."""
        _remember_thumbnail_source_cache_ops(self, cache_key, frame)

    def _update_time_label(self, frame_index: int) -> None:
        """현재 데이터와 상태를 기준으로 UI 표시값 또는 내부 상태를 동기화"""
        _update_time_label_ops(self, frame_index)

    def _format_hms(self, sec: float) -> str:
        """내부 값을 사용자 표시용 문자열 형식으로 변환해 반환"""
        return _format_hms_ops(self, sec)

    def _format_eta(self, sec: float) -> str:
        """내부 값을 사용자 표시용 문자열 형식으로 변환해 반환"""
        return _format_eta_ops(self, sec)

