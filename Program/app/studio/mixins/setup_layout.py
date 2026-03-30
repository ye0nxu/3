from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PyQt6.QtCore import QRect, QSettings, QTimer, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QDialog,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.studio.config import (
    MAIN_PAGE_ICON_PATH,
    TEAM_MODEL_PATH,
    TRAIN_DEFAULT_FREEZE,
    TRAIN_DEFAULT_LR0,
    TRAIN_REPLAY_RATIO_DEFAULT,
    TRAIN_RETRAIN_SEED_DEFAULT,
    TRAIN_STAGE1_EPOCHS_DEFAULT,
    TRAIN_STAGE2_EPOCHS_DEFAULT,
    TRAIN_STAGE2_LR_FACTOR_DEFAULT,
    TRAIN_STAGE_UNFREEZE_BACKBONE_LAST,
    TRAIN_STAGE_UNFREEZE_NECK_ONLY,
    apply_theme,
)
from core.paths import TEAM_MODEL_DIR
from app.studio.runtime import cv2
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
from app.ui.widgets.studio_support import HoverPreviewLabel, MetricChartWidget, RoiSelectionLabel


class StudioSetupLayoutMixin:
    def _bind_widgets(self) -> None:
        """UI 객체를 멤버 변수에 바인딩하고 초기 속성을 설정합니다."""
        self.btnOpenVideo = self.findChild(QPushButton, "btnOpenVideo")
        self.btnExportResult = self.findChild(QPushButton, "btnExportResult")
        self.btnStart = self.findChild(QPushButton, "btnStart")
        self.btnStop = self.findChild(QPushButton, "btnStop")
        self.btnExit = self.findChild(QPushButton, "btnExit")

        self.btnSeekBack = self.findChild(QToolButton, "btnSeekBack")
        self.btnPrevFrame = self.findChild(QToolButton, "btnPrevFrame")
        self.btnPlayPause = self.findChild(QToolButton, "btnPlayPause")
        self.btnNextFrame = self.findChild(QToolButton, "btnNextFrame")

        self.editVideoFileName = self.findChild(QLineEdit, "editVideoFileName")
        self.textLog = self.findChild(QTextEdit, "textLog")
        self.textUserInput = self.findChild(QPlainTextEdit, "textUserInput")
        self.listThumbs = self.findChild(QListWidget, "listThumbs")
        self.thumbFilterLayout = self.findChild(QHBoxLayout, "thumbFilterLayout")
        self.tabAll = self.findChild(QToolButton, "tabAll")
        self.tabKeep = self.findChild(QToolButton, "tabOk")
        self.tabHold = self.findChild(QToolButton, "tabHold")
        self.tabDrop: QToolButton | None = None
        self.thumbFilterGroup: QButtonGroup | None = None
        self.btnLoadPreviewCache = self.findChild(QPushButton, "btnLoadPreviewCache")
        self.sliderTimeline = self.findChild(QSlider, "sliderTimeline")
        self.labelTime = self.findChild(QLabel, "labelTime")
        self.labelVideoPlaceholder = self.findChild(QLabel, "labelVideoPlaceholder")
        self.labelSessionState = self.findChild(QLabel, "labelSessionState")
        self.labelAppTitle = self.findChild(QLabel, "labelAppTitle")

        self.rootLayout = self.findChild(QVBoxLayout, "rootLayout")
        self.topBarLayout = self.findChild(QHBoxLayout, "topBarLayout")
        self.topRowLayout = self.findChild(QHBoxLayout, "topRowLayout")
        self.startStopLayout = self.findChild(QHBoxLayout, "startStopLayout")
        self.cardLog = self.findChild(QWidget, "cardLog")
        self.sidebarLayout = self.findChild(QVBoxLayout, "sidebarLayout")
        self.bottomRowLayout = self.findChild(QHBoxLayout, "bottomRowLayout")
        self.cardVideo = self.findChild(QWidget, "cardVideo")
        self.cardThumbs = self.findChild(QWidget, "cardThumbs")
        self.cardLabelSetting = self.findChild(QWidget, "cardLabelSetting")
        self.cardClassList = self.findChild(QWidget, "cardClassList")
        self.labelClassListTitle = self.findChild(QLabel, "labelClassListTitle")
        self.classListLayout = self.findChild(QVBoxLayout, "classListLayout")
        self.cardUserInput = self.findChild(QWidget, "cardUserInput")
        self.userInputLayout = self.findChild(QVBoxLayout, "userInputLayout")

        self.editSampleCount: QLineEdit | None = None
        self.editRgbBits: QComboBox | None = None
        self.editTeamModelName: QLineEdit | None = None
        self.btnPickTeamModel: QPushButton | None = None
        self.spinLabelConf: QDoubleSpinBox | None = None
        self.spinLabelIou: QDoubleSpinBox | None = None
        self.spinLabelImgsz: QSpinBox | None = None
        self.selected_team_model_path: Path = TEAM_MODEL_PATH
        self.class_name_rows: list[tuple[QWidget, QLineEdit, QPushButton]] = []
        self.class_rows_scroll: QScrollArea | None = None
        self.class_rows_container: QWidget | None = None
        self.class_rows_layout: QVBoxLayout | None = None
        self.btnAddClassRow: QPushButton | None = None
        self.btnStartFast: QPushButton | None = None

        if self.textLog is not None:
            self.textLog.setReadOnly(True)
            self.textLog.setMinimumHeight(60)
            self.textLog.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if self.editVideoFileName is not None:
            self.editVideoFileName.setReadOnly(True)
        if self.labelVideoPlaceholder is not None:
            self.labelVideoPlaceholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.labelVideoPlaceholder.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            self.labelVideoPlaceholder.setMinimumSize(0, 0)
        if self.listThumbs is not None:
            self.listThumbs.setLayoutMode(QListView.LayoutMode.Batched)
            self.listThumbs.setBatchSize(48)
            self.listThumbs.setUniformItemSizes(True)

        if self.labelAppTitle is not None:
            self.labelAppTitle.setText("Auto Labeling Tool")
        if self.btnOpenVideo is not None:
            self.btnOpenVideo.setText("저장 폴더 열기")
        if self.btnExportResult is not None:
            self.btnExportResult.setText("결과 내보내기")
        if self.btnExit is not None:
            self.btnExit.setText("종료")
        if self.btnStart is not None:
            self.btnStart.setText("기본 처리 시작")
        if self.btnStop is not None:
            self.btnStop.setText("중지")
        if self.btnSeekBack is not None:
            self.btnSeekBack.setText("■")
            self.btnSeekBack.setToolTip("정지")
        if self.btnPrevFrame is not None:
            self.btnPrevFrame.setText("|◀")
            self.btnPrevFrame.setToolTip("이전 프레임")
        if self.btnPlayPause is not None:
            self.btnPlayPause.setText("▶")
            self.btnPlayPause.setToolTip("재생")
        if self.btnNextFrame is not None:
            self.btnNextFrame.setText("▶|")
            self.btnNextFrame.setToolTip("다음 프레임")
        self._style_advanced1_playback_controls()
        self._update_play_pause_button_icon()
        if self.btnLoadPreviewCache is not None:
            self.btnLoadPreviewCache.setText("keep/hold/drop 불러오기")
        if self.cardLog is not None:
            log_title = self.cardLog.findChild(QLabel, "labelLogTitle")
            if log_title is not None:
                log_title.setText("처리 로그")

    def _style_home_hero_section(self) -> None:
        """홈 화면 상단 히어로 섹션을 현재 테마(라이트/다크)에 맞게 스타일링합니다."""
        hero = self.findChild(QFrame, "homeHeroCard")
        if hero is None:
            return

        title = self.findChild(QLabel, "labelHomeHeroTitle")
        badge = self.findChild(QLabel, "labelHomeSessionBadge")
        process_badge = self.findChild(QLabel, "labelHomeProcessBadge")
        process_divider = self.findChild(QFrame, "homeHeroProcessDivider")
        process_text = self.findChild(QLabel, "labelHomeProcessText")
        chips = [label for label in self.findChildren(QLabel) if bool(label.property("homeHeroChip"))]

        theme = str(QSettings("SL_TEAM", "AutoLabelStudio").value("theme", "dark")).strip().lower()
        is_light = theme == "light"

        if is_light:
            hero.setStyleSheet(
                "QFrame#homeHeroCard {"
                "  border: 1px solid rgba(17, 24, 39, 22);"
                "  border-radius: 16px;"
                "  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
                "    stop:0 rgba(250, 252, 255, 255), stop:1 rgba(237, 243, 251, 255));"
                "}"
                "QLabel { border: none; background: transparent; }"
            )
            if badge is not None:
                badge.setStyleSheet(
                    "padding: 5px 12px; border-radius: 10px; font-size: 12px; font-weight: 700;"
                    "color: rgb(35, 85, 175);"
                    "border: 1px solid rgba(59, 130, 246, 90);"
                    "background: rgba(59, 130, 246, 18);"
                )
            if title is not None:
                title.setStyleSheet("font-size: 48px; font-weight: 850; color: rgb(17, 24, 39);")
            if process_badge is not None:
                process_badge.setStyleSheet(
                    "padding: 5px 12px; border-radius: 10px; font-size: 12px; font-weight: 700;"
                    "color: rgb(29, 78, 216);"
                    "border: 1px solid rgba(59, 130, 246, 95);"
                    "background: rgba(59, 130, 246, 20);"
                )
            if process_divider is not None:
                process_divider.setStyleSheet("background: rgba(56, 89, 138, 105); border: none;")
            if process_text is not None:
                process_text.setStyleSheet("font-size: 15px; font-weight: 700; color: rgb(30, 41, 59);")
            chip_style = (
                "padding: 5px 12px; border-radius: 10px;"
                "font-size: 12px; font-weight: 700;"
                "color: rgb(29, 78, 216);"
                "border: 1px solid rgba(59, 130, 246, 95);"
                "background: rgba(59, 130, 246, 18);"
            )
        else:
            hero.setStyleSheet(
                "QFrame#homeHeroCard {"
                "  border: 1px solid rgba(106, 149, 241, 140);"
                "  border-radius: 16px;"
                "  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
                "    stop:0 rgba(24, 34, 54, 248), stop:1 rgba(18, 28, 44, 248));"
                "}"
                "QLabel { border: none; background: transparent; }"
            )
            if badge is not None:
                badge.setStyleSheet(
                    "padding: 5px 12px; border-radius: 10px; font-size: 12px; font-weight: 700;"
                    "color: rgb(163, 208, 255);"
                    "border: 1px solid rgba(118, 173, 255, 180);"
                    "background: rgba(71, 113, 178, 72);"
                )
            if title is not None:
                title.setStyleSheet("font-size: 48px; font-weight: 850; color: rgb(238, 246, 255);")
            if process_badge is not None:
                process_badge.setStyleSheet(
                    "padding: 5px 12px; border-radius: 10px; font-size: 12px; font-weight: 700;"
                    "color: rgb(228, 241, 255);"
                    "border: 1px solid rgba(118, 173, 255, 170);"
                    "background: rgba(71, 113, 178, 54);"
                )
            if process_divider is not None:
                process_divider.setStyleSheet("background: rgba(120, 154, 209, 165); border: none;")
            if process_text is not None:
                process_text.setStyleSheet("font-size: 15px; font-weight: 700; color: rgb(225, 235, 251);")
            chip_style = (
                "padding: 5px 12px; border-radius: 10px;"
                "font-size: 12px; font-weight: 700;"
                "color: rgb(163, 208, 255);"
                "border: 1px solid rgba(118, 173, 255, 170);"
                "background: rgba(71, 113, 178, 52);"
            )

        for chip in chips:
            chip.setStyleSheet(chip_style)

    def _style_home_guide_section(self) -> None:
        """홈 화면 사용 가이드 카드를 현재 테마(라이트/다크)에 맞게 스타일링합니다."""
        cards = [frame for frame in self.findChildren(QFrame) if bool(frame.property("homeGuideFlowCard"))]
        if not cards:
            return

        theme = str(QSettings("SL_TEAM", "AutoLabelStudio").value("theme", "dark")).strip().lower()
        is_light = theme == "light"

        for card in cards:
            variant = str(card.property("guideVariant") or "").strip().lower()
            badge = card.findChild(QLabel, "labelHomeGuideBadge")
            title = card.findChild(QLabel, "labelHomeGuideTitle")
            desc = card.findChild(QLabel, "labelHomeGuideDesc")
            flow_title = card.findChild(QLabel, "labelHomeGuideFlowTitle")
            tip_title = card.findChild(QLabel, "labelHomeGuideTipTitle")
            tip_desc = card.findChild(QLabel, "labelHomeGuideTipDesc")
            step_labels = [lbl for lbl in card.findChildren(QLabel) if bool(lbl.property("homeGuideStep"))]
            dividers = [fr for fr in card.findChildren(QFrame) if bool(fr.property("homeGuideDivider"))]

            if is_light:
                card.setStyleSheet(
                    "QFrame#homeGuideFlowCard {"
                    "  border: 1px solid rgba(17, 24, 39, 20);"
                    "  border-radius: 18px;"
                    "  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
                    "    stop:0 rgba(252, 253, 255, 255), stop:1 rgba(241, 246, 255, 255));"
                    "}"
                    "QLabel { border: none; background: transparent; }"
                )
                if badge is not None:
                    if variant == "new":
                        badge.setStyleSheet(
                            "padding: 4px 10px; border-radius: 11px; font-size: 11px; font-weight: 700;"
                            "color: rgb(8, 117, 134); border: 1px solid rgba(6, 182, 212, 95);"
                            "background: rgba(6, 182, 212, 16);"
                        )
                    else:
                        badge.setStyleSheet(
                            "padding: 4px 10px; border-radius: 11px; font-size: 11px; font-weight: 700;"
                            "color: rgb(29, 78, 216); border: 1px solid rgba(59, 130, 246, 95);"
                            "background: rgba(59, 130, 246, 16);"
                        )
                if title is not None:
                    title.setStyleSheet("font-size: 18px; font-weight: 800; color: rgb(17, 24, 39);")
                if desc is not None:
                    desc.setStyleSheet("font-size: 13px; color: rgb(71, 85, 105);")
                if flow_title is not None:
                    flow_title.setStyleSheet("font-size: 13px; font-weight: 800; color: rgb(17, 24, 39);")
                for step_label in step_labels:
                    step_label.setStyleSheet("font-size: 13px; font-weight: 700; color: rgb(30, 41, 59);")
                if tip_title is not None:
                    tip_title.setStyleSheet("font-size: 13px; font-weight: 800; color: rgb(30, 41, 59);")
                if tip_desc is not None:
                    tip_desc.setStyleSheet("font-size: 13px; color: rgb(71, 85, 105);")
                for divider in dividers:
                    divider.setStyleSheet("background: rgba(71, 85, 105, 58); border: none;")
            else:
                card.setStyleSheet(
                    "QFrame#homeGuideFlowCard {"
                    "  border: 1px solid rgba(112, 153, 238, 120);"
                    "  border-radius: 18px;"
                    "  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
                    "    stop:0 rgba(20, 30, 52, 245), stop:1 rgba(17, 26, 45, 245));"
                    "}"
                    "QLabel { border: none; background: transparent; }"
                )
                if badge is not None:
                    if variant == "new":
                        badge.setStyleSheet(
                            "padding: 4px 10px; border-radius: 11px; font-size: 11px; font-weight: 700;"
                            "color: rgb(122, 218, 232); border: 1px solid rgba(70, 202, 224, 140);"
                            "background: rgba(45, 130, 162, 45);"
                        )
                    else:
                        badge.setStyleSheet(
                            "padding: 4px 10px; border-radius: 11px; font-size: 11px; font-weight: 700;"
                            "color: rgb(128, 176, 255); border: 1px solid rgba(106, 149, 241, 150);"
                            "background: rgba(57, 93, 160, 48);"
                        )
                if title is not None:
                    title.setStyleSheet("font-size: 18px; font-weight: 800; color: rgb(230, 238, 255);")
                if desc is not None:
                    desc.setStyleSheet("font-size: 13px; color: rgb(166, 181, 212);")
                if flow_title is not None:
                    flow_title.setStyleSheet("font-size: 13px; font-weight: 800; color: rgb(229, 236, 251);")
                for step_label in step_labels:
                    step_label.setStyleSheet("font-size: 13px; font-weight: 700; color: rgb(226, 234, 250);")
                if tip_title is not None:
                    tip_title.setStyleSheet("font-size: 13px; font-weight: 800; color: rgb(196, 228, 246);")
                if tip_desc is not None:
                    tip_desc.setStyleSheet("font-size: 13px; color: rgb(175, 191, 220);")
                for divider in dividers:
                    divider.setStyleSheet("background: rgba(120, 154, 209, 95); border: none;")

    def _style_advanced1_playback_controls(self) -> None:
        """기존 객체 라벨링 비디오 컨트롤 버튼을 현재 테마에 맞게 스타일링합니다."""
        controls_layout = self.findChild(QHBoxLayout, "controlsLayout")
        if controls_layout is not None:
            for i in range(controls_layout.count() - 1, -1, -1):
                if controls_layout.itemAt(i).spacerItem() is not None:
                    controls_layout.takeAt(i)
            controls_layout.setSpacing(10)
            controls_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        theme = str(QSettings("SL_TEAM", "AutoLabelStudio").value("theme", "dark")).strip().lower()
        is_light = theme == "light"
        if is_light:
            style = (
                "QToolButton {"
                "  min-width: 38px; max-width: 38px;"
                "  min-height: 38px; max-height: 38px;"
                "  border-radius: 19px;"
                "  border: 1px solid rgba(17, 24, 39, 30);"
                "  background: rgba(255, 255, 255, 245);"
                "  color: rgb(17, 24, 39);"
                "  font-size: 14px;"
                "  font-weight: 700;"
                "}"
                "QToolButton:hover { background: rgb(239, 244, 251); }"
                "QToolButton:pressed { background: rgb(227, 235, 247); }"
                "QToolButton:disabled { background: rgb(239, 242, 247); color: rgb(147, 157, 173); }"
            )
        else:
            style = (
                "QToolButton {"
                "  min-width: 38px; max-width: 38px;"
                "  min-height: 38px; max-height: 38px;"
                "  border-radius: 19px;"
                "  border: 1px solid rgba(173, 190, 217, 42);"
                "  background: rgba(24, 34, 52, 230);"
                "  color: rgb(232, 239, 255);"
                "  font-size: 14px;"
                "  font-weight: 700;"
                "}"
                "QToolButton:hover { background: rgba(35, 48, 71, 238); }"
                "QToolButton:pressed { background: rgba(52, 67, 94, 240); }"
                "QToolButton:disabled { background: rgba(44, 52, 66, 210); color: rgb(139, 147, 165); }"
            )

        buttons = [self.btnSeekBack, self.btnPrevFrame, self.btnPlayPause, self.btnNextFrame]
        for button in buttons:
            if button is None:
                continue
            button.setAutoRaise(False)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet(style)

    def _update_play_pause_button_icon(self) -> None:
        """재생 상태에 따라 재생/일시정지 버튼 아이콘과 툴팁을 갱신합니다."""
        if self.btnPlayPause is None:
            return
        if self.is_playing:
            self.btnPlayPause.setText("⏸")
            self.btnPlayPause.setToolTip("일시정지")
        else:
            self.btnPlayPause.setText("▶")
            self.btnPlayPause.setToolTip("재생")

    def _remove_model_setting_card(self) -> None:
        """대상 항목 또는 위젯을 제거하고 연관 상태를 정리합니다."""
        if self.cardLabelSetting is None:
            return
        if self.sidebarLayout is not None:
            self.sidebarLayout.removeWidget(self.cardLabelSetting)
        self.cardLabelSetting.setParent(None)
        self.cardLabelSetting.deleteLater()
        self.cardLabelSetting = None

    def _setup_processing_mode_buttons(self) -> None:
        """UI/타이머/시그널 등 초기 구성을 수행하고 기본 상태를 연결합니다."""
        if self.startStopLayout is None:
            return
        if self.btnStartFast is not None:
            return

        self.btnStartFast = QPushButton("빠른 처리 시작", self)
        self.btnStartFast.setObjectName("btnStartFast")
        self.btnStartFast.setMinimumHeight(34)
        self.startStopLayout.insertWidget(1, self.btnStartFast)

    def _setup_thumbnail_filter_controls(self) -> None:
        """UI/타이머/시그널 등 초기 구성을 수행하고 기본 상태를 연결합니다."""
        if self.tabAll is None or self.tabKeep is None or self.tabHold is None:
            return

        self.tabAll.setText("전체(keep/hold)")
        self.tabKeep.setText("keep")
        self.tabHold.setText("hold")

        for tab in (self.tabAll, self.tabKeep, self.tabHold):
            tab.setCheckable(True)

        if self.thumbFilterLayout is not None and self.tabDrop is None:
            self.tabDrop = QToolButton(self)
            self.tabDrop.setObjectName("tabDrop")
            self.tabDrop.setText("drop")
            self.tabDrop.setCheckable(True)
            insert_index = max(0, self.thumbFilterLayout.count() - 1)
            self.thumbFilterLayout.insertWidget(insert_index, self.tabDrop)

        if self.thumbFilterLayout is not None:
            if self.btnLoadPreviewCache is None:
                self.btnLoadPreviewCache = QPushButton("keep/hold/drop 불러오기", self)
                self.btnLoadPreviewCache.setObjectName("btnLoadPreviewCache")
                self.btnLoadPreviewCache.setMinimumHeight(28)
                insert_index = max(0, self.thumbFilterLayout.count() - 1)
                self.thumbFilterLayout.insertWidget(insert_index, self.btnLoadPreviewCache)
            else:
                self.btnLoadPreviewCache.setText("keep/hold/drop 불러오기")
                self.btnLoadPreviewCache.setMinimumHeight(28)
            self.btnLoadPreviewCache.hide()

        self.thumbFilterGroup = QButtonGroup(self)
        self.thumbFilterGroup.setExclusive(True)
        for tab in (self.tabAll, self.tabKeep, self.tabHold, self.tabDrop):
            if tab is None:
                continue
            self.thumbFilterGroup.addButton(tab)

        self.tabAll.setChecked(True)
        self._update_thumbnail_filter_tab_counts()

    def _setup_class_selector(self) -> None:
        """UI/타이머/시그널 등 초기 구성을 수행하고 기본 상태를 연결합니다."""
        if self.classListLayout is None:
            return

        self.classListLayout.setSpacing(5)

        while self.classListLayout.count() > 0:
            item = self.classListLayout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        if self.labelClassListTitle is not None:
            self.labelClassListTitle.setText("객체 클래스 입력")

        class_box_title = QLabel("객체 클래스 설정", self)
        class_box_title.setStyleSheet("font-weight: 700;")
        self.classListLayout.addWidget(class_box_title)

        class_input_hint = QLabel(
            "비우면 기본값 '객체'를 사용합니다.",
            self,
        )
        class_input_hint.setWordWrap(False)
        class_input_hint.setStyleSheet("color: rgb(113, 126, 149); font-size: 8.8pt;")
        self.classListLayout.addWidget(class_input_hint)

        self.class_rows_container = QWidget(self)
        self.class_rows_layout = QVBoxLayout(self.class_rows_container)
        self.class_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.class_rows_layout.setSpacing(6)
        self.class_rows_scroll = QScrollArea(self)
        self.class_rows_scroll.setObjectName("classRowsScroll")
        self.class_rows_scroll.setWidgetResizable(True)
        self.class_rows_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.class_rows_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.class_rows_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.class_rows_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.class_rows_scroll.setWidget(self.class_rows_container)
        self.classListLayout.addWidget(self.class_rows_scroll)

        self.btnAddClassRow = QPushButton("+ 라벨 추가", self)
        self.btnAddClassRow.setObjectName("btnAddClassRow")
        self.btnAddClassRow.setMinimumHeight(30)
        self.btnAddClassRow.clicked.connect(lambda: self._add_class_input_row())
        self.classListLayout.addWidget(self.btnAddClassRow)

        self.class_name_rows = []
        self._add_class_input_row()

        if self.cardClassList is not None:
            self.cardClassList.setMinimumHeight(200)
            self.cardClassList.setMaximumHeight(260)
            self.cardClassList.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def _add_class_input_row(self, value: str = "") -> None:
        """새 항목 또는 위젯을 생성해 컬렉션과 레이아웃에 추가합니다."""
        if self.class_rows_layout is None:
            return

        row_widget = QWidget(self.class_rows_container)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        edit = QLineEdit(row_widget)
        edit.setPlaceholderText("예: car")
        edit.setText(str(value).strip())
        edit.setMinimumHeight(28)

        btn_delete = QPushButton("삭제", row_widget)
        btn_delete.setMinimumHeight(28)
        btn_delete.setMinimumWidth(56)
        btn_delete.clicked.connect(lambda _checked=False, target=row_widget: self._remove_class_input_row(target))

        row_layout.addWidget(edit, 1)
        row_layout.addWidget(btn_delete, 0)

        self.class_rows_layout.addWidget(row_widget)
        self.class_name_rows.append((row_widget, edit, btn_delete))
        self._sync_class_row_delete_state()
        self._update_class_rows_scroll_height()

    def _remove_class_input_row(self, target: QWidget) -> None:
        """대상 항목 또는 위젯을 제거하고 연관 상태를 정리합니다."""
        if not self.class_name_rows:
            return

        if len(self.class_name_rows) <= 1:
            self.class_name_rows[0][1].clear()
            return

        remove_index = -1
        for idx, (row_widget, _edit, _btn) in enumerate(self.class_name_rows):
            if row_widget is target:
                remove_index = idx
                break
        if remove_index < 0:
            return

        row_widget, _edit, _btn = self.class_name_rows.pop(remove_index)
        if self.class_rows_layout is not None:
            self.class_rows_layout.removeWidget(row_widget)
        row_widget.setParent(None)
        row_widget.deleteLater()
        self._sync_class_row_delete_state()
        self._update_class_rows_scroll_height()

    def _sync_class_row_delete_state(self) -> None:
        """서로 다른 컴포넌트의 값을 동일하게 맞춰 상태 일관성을 유지합니다."""
        can_delete = len(self.class_name_rows) > 1
        for _row, _edit, btn in self.class_name_rows:
            btn.setEnabled(can_delete)

    def _update_class_rows_scroll_height(self) -> None:
        """현재 데이터와 상태를 기준으로 UI 표시값 또는 내부 상태를 동기화합니다."""
        if self.class_rows_scroll is None:
            return
        visible_rows = max(1, min(4, len(self.class_name_rows)))
        row_height = 34
        spacing = 4
        target_height = (visible_rows * row_height) + ((visible_rows - 1) * spacing) + 3
        self.class_rows_scroll.setMinimumHeight(target_height)
        self.class_rows_scroll.setMaximumHeight(target_height)

    def _setup_user_input_fields(self) -> None:
        """UI/타이머/시그널 등 초기 구성을 수행하고 기본 상태를 연결합니다."""
        if self.userInputLayout is None:
            return

        if self.textUserInput is not None:
            self.userInputLayout.removeWidget(self.textUserInput)
            self.textUserInput.setMaximumHeight(1)
            self.textUserInput.hide()

        form_widget = QWidget(self)
        form_layout = QFormLayout(form_widget)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.editSampleCount = QLineEdit(form_widget)
        self.editSampleCount.setObjectName("editSampleCount")
        self.editSampleCount.setText("")
        self.editSampleCount.setPlaceholderText("비우면 끝까지 (예: 500)")
        self.editSampleCount.setMaximumWidth(120)

        self.editRgbBits = QComboBox(form_widget)
        self.editRgbBits.setObjectName("editRgbBits")
        self.editRgbBits.setMaximumWidth(120)
        self.editRgbBits.setEditable(False)
        self.editRgbBits.addItem("8 (흑백)", 8)
        self.editRgbBits.addItem("16", 16)
        self.editRgbBits.addItem("24", 24)
        self.editRgbBits.addItem("32", 32)
        default_rgb_index = self.editRgbBits.findData(24)
        if default_rgb_index >= 0:
            self.editRgbBits.setCurrentIndex(default_rgb_index)

        self.editTeamModelName = QLineEdit(form_widget)
        self.editTeamModelName.setObjectName("editTeamModelName")
        self.editTeamModelName.setMaximumWidth(220)
        self.editTeamModelName.setReadOnly(True)
        self.selected_team_model_path = TEAM_MODEL_PATH
        self.editTeamModelName.setText(self.selected_team_model_path.name)

        self.btnPickTeamModel = QPushButton("파일 선택", form_widget)
        self.btnPickTeamModel.setObjectName("btnPickTeamModel")
        self.btnPickTeamModel.setMinimumHeight(28)
        self.btnPickTeamModel.clicked.connect(self._on_pick_team_model_file)

        model_row = QWidget(form_widget)
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        model_row_layout.setSpacing(6)
        model_row_layout.addWidget(self.editTeamModelName, 1)
        model_row_layout.addWidget(self.btnPickTeamModel, 0)

        # 고급 설정 다이얼로그 인스턴스 생성
        self.advancedSettingsDialog = AdvancedLabelingSettingsDialog(self)

        # 고급 설정 열기 버튼
        self.btnAdvancedSettings = QPushButton("⚙ 고급 설정", form_widget)
        self.btnAdvancedSettings.setObjectName("btnAdvancedSettings")
        self.btnAdvancedSettings.setMinimumHeight(30)
        self.btnAdvancedSettings.setToolTip("검출·추적·필터 임계치를 세부 조정합니다.")
        self.btnAdvancedSettings.clicked.connect(self._on_open_advanced_settings)

        form_layout.addRow("이미지 장수", self.editSampleCount)
        form_layout.addRow("색상 비트", self.editRgbBits)
        form_layout.addRow("팀 모델", model_row)
        form_layout.addRow("", self.btnAdvancedSettings)
        self.userInputLayout.addWidget(form_widget)

        if self.cardUserInput is not None:
            self.cardUserInput.setMinimumHeight(160)
            self.cardUserInput.setMaximumHeight(240)

    def _on_open_advanced_settings(self) -> None:
        """고급 설정 다이얼로그를 열고 확인 시 값을 반영합니다."""
        if hasattr(self, "advancedSettingsDialog"):
            self.advancedSettingsDialog.exec()

    def _on_pick_team_model_file(self) -> None:
        """사용자가 로컬 .pt 모델 파일을 직접 선택해 모델 이름 표시와 경로를 갱신합니다."""
        if self.is_processing or self.is_exporting:
            return
        if self.editTeamModelName is None:
            return

        start_dir = TEAM_MODEL_DIR if TEAM_MODEL_DIR.is_dir() else Path.home()
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "팀 모델 파일 선택",
            str(start_dir),
            "PyTorch 모델 (*.pt);;모든 파일 (*)",
        )
        if not selected_path:
            return

        model_path = Path(selected_path).resolve()
        if not model_path.is_file():
            QMessageBox.warning(self, "모델 선택", f"모델 파일을 찾을 수 없습니다.\n{model_path}")
            return

        self.selected_team_model_path = model_path
        self.editTeamModelName.setText(model_path.name)
        self._append_log(f"user action: model selected ({model_path.name})")

    def _relocate_log_card_under_start_stop(self) -> None:
        """사이드바 레이아웃에서 로그 카드를 START/STOP 영역 아래로 재배치합니다."""
        if self.cardLog is None or self.sidebarLayout is None:
            return

        if self.bottomRowLayout is not None:
            self.bottomRowLayout.removeWidget(self.cardLog)

        # Remove the spacer under START/STOP so the log card sits right below it.
        idx = 0
        while idx < self.sidebarLayout.count():
            item = self.sidebarLayout.itemAt(idx)
            if item is not None and item.spacerItem() is not None:
                self.sidebarLayout.takeAt(idx)
                break
            idx += 1

        self.cardLog.setMinimumHeight(170)
        self.cardLog.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.sidebarLayout.addWidget(self.cardLog)
        self.sidebarLayout.setStretchFactor(self.cardLog, 1)

    def _rebuild_content_layout_for_full_height_sidebar(self) -> None:
        """구형 2열 레이아웃을 재구성해 사이드바가 전체 높이를 활용하도록 배치 구조를 변경합니다."""
        if self.rootLayout is None:
            return
        if self.topRowLayout is None or self.bottomRowLayout is None:
            return
        if self.sidebarLayout is None or self.cardVideo is None or self.cardThumbs is None:
            return
        if getattr(self, "_content_rebuilt_for_full_height_sidebar", False):
            return

        # Detach widgets/layouts from old two-row structure.
        self.topRowLayout.removeWidget(self.cardVideo)
        self.bottomRowLayout.removeWidget(self.cardThumbs)
        for idx in range(self.topRowLayout.count()):
            item = self.topRowLayout.itemAt(idx)
            if item is not None and item.layout() is self.sidebarLayout:
                self.topRowLayout.takeAt(idx)
                break

        # Remove old row layouts from the root vertical layout.
        for target_layout in (self.bottomRowLayout, self.topRowLayout):
            for idx in range(self.rootLayout.count()):
                item = self.rootLayout.itemAt(idx)
                if item is not None and item.layout() is target_layout:
                    self.rootLayout.takeAt(idx)
                    break

        content_widget = QWidget(self)
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)

        left_widget = QWidget(content_widget)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)
        left_layout.addWidget(self.cardVideo)
        left_layout.addWidget(self.cardThumbs)
        left_layout.setStretch(0, 3)
        left_layout.setStretch(1, 1)

        content_layout.addWidget(left_widget, 3)
        content_layout.addLayout(self.sidebarLayout, 2)

        self.primary_workbench_widget = content_widget
        self.rootLayout.insertWidget(1, content_widget, 1)
        self.rootLayout.setStretch(1, 1)
        self._content_rebuilt_for_full_height_sidebar = True

    def _setup_progress_panel(self) -> None:
        """UI/타이머/시그널 등 초기 구성을 수행하고 기본 상태를 연결합니다."""
        self.labelProgressStage = QLabel("현재 단계: 대기", self)
        self.labelProgressStage.setObjectName("labelProgressStage")
        self.labelProgressFrame = QLabel("프레임 - / -", self)
        self.labelProgressFrame.setObjectName("labelProgressFrame")
        self.labelProgressRemain = QLabel("남은 예상: -", self)
        self.labelProgressRemain.setObjectName("labelProgressRemain")

        self.progressPipeline = QProgressBar(self)
        self.progressPipeline.setObjectName("progressPipeline")
        self.progressPipeline.setRange(0, 100)
        self.progressPipeline.setValue(0)

        panel = QWidget(self)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(4)
        panel_layout.addWidget(self.labelProgressStage)
        panel_layout.addWidget(self.labelProgressFrame)
        panel_layout.addWidget(self.labelProgressRemain)
        panel_layout.addWidget(self.progressPipeline)
        panel_height = max(84, panel.sizeHint().height() + 4)
        panel.setMinimumHeight(panel_height)
        panel.setMaximumHeight(panel_height)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        log_layout = self.findChild(QVBoxLayout, "logLayout")
        if log_layout is None:
            candidate = getattr(self, "logLayout", None)
            if isinstance(candidate, QVBoxLayout):
                log_layout = candidate
        if log_layout is not None:
            log_layout.insertWidget(1, panel)
            # Keep a small visual gap between progress panel and log text area.
            log_layout.insertSpacing(2, 8)
            if self.textLog is not None:
                log_layout.setStretchFactor(self.textLog, 1)

    def _setup_play_timer(self) -> None:
        """UI/타이머/시그널 등 초기 구성을 수행하고 기본 상태를 연결합니다."""
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(33)
        self.play_timer.timeout.connect(self._on_play_tick)

    def _sync_play_timer_to_video_fps(self) -> None:
        """서로 다른 컴포넌트의 값을 동일하게 맞춰 상태 일관성을 유지합니다."""
        fps = float(self.video_meta.get("fps", 0.0))
        if fps <= 0.0:
            self.play_timer.setInterval(33)
            return
        interval_ms = max(1, int(round(1000.0 / fps)))
        self.play_timer.setInterval(interval_ms)

    def _setup_video_load_hover_preview(self) -> None:
        """UI/타이머/시그널 등 초기 구성을 수행하고 기본 상태를 연결합니다."""
        self.video_load_hover_timer = QTimer(self)
        self.video_load_hover_timer.setInterval(33)
        self.video_load_hover_timer.timeout.connect(self._on_video_load_hover_tick)

    def _start_video_load_hover_preview(self) -> None:
        """영상 로드 페이지에서 마우스 호버 시 약 10초 미리보기 재생을 시작합니다."""
        if self.video_path is None or self.is_processing:
            return
        if self.video_load_hover_timer is None:
            return
        if self.is_playing:
            return

        self._stop_video_load_hover_preview()
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return

        fps = float(self.video_meta.get("fps", 0.0))
        if fps <= 0:
            fps = 30.0
        total_frames = int(self.video_meta.get("frame_count", 0))
        if total_frames <= 0:
            cap.release()
            return

        start_index = 0
        if self.sliderTimeline is not None:
            start_index = int(self.sliderTimeline.value())
        start_index = max(0, min(total_frames - 1, start_index))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        self.video_load_hover_cap = cap
        self.video_load_hover_remaining_frames = max(1, min(int(round(fps * 10.0)), total_frames - start_index))
        self.video_load_hover_active = True
        self.video_load_hover_timer.setInterval(max(15, int(round(1000.0 / fps))))
        self.video_load_hover_timer.start()

    def _stop_video_load_hover_preview(self) -> None:
        """호버 미리보기 재생을 중지하고 임시 캡처 리소스를 정리합니다."""
        self.video_load_hover_active = False
        self.video_load_hover_remaining_frames = 0
        if self.video_load_hover_timer is not None:
            self.video_load_hover_timer.stop()
        if self.video_load_hover_cap is not None:
            self.video_load_hover_cap.release()
            self.video_load_hover_cap = None

    def _on_video_load_hover_tick(self) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        if not self.video_load_hover_active:
            self._stop_video_load_hover_preview()
            return
        if self.video_load_hover_cap is None:
            self._stop_video_load_hover_preview()
            return
        if self.video_load_hover_remaining_frames <= 0:
            self._stop_video_load_hover_preview()
            return

        ok, frame = self.video_load_hover_cap.read()
        if not ok or frame is None:
            self._stop_video_load_hover_preview()
            return

        self.video_load_hover_remaining_frames -= 1
        self._set_video_load_preview_frame(self._apply_video_roi_to_frame(frame))
        if self.video_load_hover_remaining_frames <= 0:
            self._stop_video_load_hover_preview()

    def _set_video_load_preview_frame(self, frame: np.ndarray) -> None:
        """영상 로드 페이지 미리보기 라벨에 지정 프레임을 표시하고 마지막 프레임 캐시를 갱신합니다."""
        if self.labelVideoLoadPreview is None:
            return
        self.last_video_load_preview_frame = frame
        self.labelVideoLoadPreview.setPixmap(self._frame_to_pixmap(frame, max_width=920, max_height=520))
        self.labelVideoLoadPreview.setText("")

    def _normalize_video_roi(
        self,
        roi: tuple[int, int, int, int] | None,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        """입력 ROI를 프레임 경계 안으로 보정해 반환하고 전체 프레임이면 None을 반환합니다."""
        if roi is None:
            return None
        if width <= 0 or height <= 0:
            return None
        try:
            x, y, w, h = [int(v) for v in roi]
        except Exception:
            return None
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        if x == 0 and y == 0 and w >= width and h >= height:
            return None
        return (x, y, w, h)

    def _parse_roi_payload(self, payload: object) -> tuple[int, int, int, int] | None:
        """외부 입력(run_config 포함)에서 ROI 값을 읽어 정수 사각형으로 변환합니다."""
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
            return None
        values = list(payload)
        if len(values) < 4:
            return None
        try:
            x = int(round(float(values[0])))
            y = int(round(float(values[1])))
            w = int(round(float(values[2])))
            h = int(round(float(values[3])))
        except Exception:
            return None
        return (x, y, w, h)

    def _result_run_roi(self) -> tuple[int, int, int, int] | None:
        """최근 처리 결과(run_config)에 기록된 ROI를 반환합니다."""
        if self.latest_result is None:
            return None
        run_config = self.latest_result.run_config
        if not isinstance(run_config, dict):
            return None
        parsed = self._parse_roi_payload(run_config.get("roi_rect"))
        width = int(self.video_meta.get("width", 0))
        height = int(self.video_meta.get("height", 0))
        return self._normalize_video_roi(parsed, width, height)

    def _apply_roi_to_frame(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        """전달된 ROI를 프레임에 적용해 크롭 결과를 반환합니다."""
        if frame is None:
            return frame
        height, width = frame.shape[:2]
        normalized = self._normalize_video_roi(roi, width, height)
        if normalized is None:
            return frame
        x, y, w, h = normalized
        cropped = frame[y : y + h, x : x + w]
        if cropped.size <= 0:
            return frame
        return cropped

    def _apply_video_roi_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """현재 선택된 ROI가 있으면 프레임을 해당 영역으로 크롭해 반환합니다."""
        return self._apply_roi_to_frame(frame, self.video_roi)

    def _update_video_load_roi_label(self) -> None:
        """영상 로드 페이지의 ROI 상태 라벨을 현재 값으로 갱신합니다."""
        if self.labelVideoLoadRoiValue is None:
            return
        width = int(self.video_meta.get("width", 0))
        height = int(self.video_meta.get("height", 0))
        roi = self._normalize_video_roi(self.video_roi, width, height)
        if self.video_path is None:
            self.labelVideoLoadRoiValue.setText("ROI: -")
            return
        if roi is None:
            self.labelVideoLoadRoiValue.setText("ROI: 전체 프레임")
            return
        x, y, w, h = roi
        self.labelVideoLoadRoiValue.setText(f"ROI: x={x}, y={y}, w={w}, h={h}")

    def _show_video_roi_dialog(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """현재 프레임 스냅샷에서 ROI를 드래그 선택하고 원본 좌표 기준 ROI를 반환합니다."""
        if not isinstance(frame, np.ndarray):
            return None
        src_h, src_w = frame.shape[:2]
        if src_w <= 0 or src_h <= 0:
            return None

        pixmap = self._frame_to_pixmap(frame, max_width=1200, max_height=720)
        if pixmap.isNull():
            return None

        dlg = QDialog(self)
        dlg.setWindowTitle("ROI 선택")
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        help_label = QLabel("마우스로 영역을 드래그해서 ROI를 선택하세요.", dlg)
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        canvas = RoiSelectionLabel(dlg)
        canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas.setPixmap(pixmap)
        canvas.setFixedSize(pixmap.size())
        canvas.setStyleSheet("background:#000; border:1px solid rgba(120, 160, 190, 120);")
        layout.addWidget(canvas, 0, Qt.AlignmentFlag.AlignCenter)

        disp_w = max(1, int(pixmap.width()))
        disp_h = max(1, int(pixmap.height()))
        current_roi = self._normalize_video_roi(self.video_roi, src_w, src_h)
        if current_roi is not None:
            rx, ry, rw, rh = current_roi
            sx = int(round((rx * disp_w) / float(src_w)))
            sy = int(round((ry * disp_h) / float(src_h)))
            sw = max(1, int(round((rw * disp_w) / float(src_w))))
            sh = max(1, int(round((rh * disp_h) / float(src_h))))
            canvas.set_selection_rect(QRect(sx, sy, sw, sh))

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        btn_apply = QPushButton("적용", dlg)
        btn_full = QPushButton("전체 프레임", dlg)
        btn_cancel = QPushButton("취소", dlg)
        button_row.addStretch(1)
        button_row.addWidget(btn_apply)
        button_row.addWidget(btn_full)
        button_row.addWidget(btn_cancel)
        layout.addLayout(button_row)

        result: dict[str, tuple[int, int, int, int] | None] = {"roi": None}
        selected_full = {"value": False}

        def _on_apply() -> None:
            rect = canvas.selected_rect()
            if rect is None or rect.width() < 2 or rect.height() < 2:
                QMessageBox.information(dlg, "ROI 선택", "ROI 영역을 드래그해서 선택하세요.")
                return
            x = int(round((rect.x() * src_w) / float(disp_w)))
            y = int(round((rect.y() * src_h) / float(disp_h)))
            w = int(round((rect.width() * src_w) / float(disp_w)))
            h = int(round((rect.height() * src_h) / float(disp_h)))
            result["roi"] = (x, y, max(1, w), max(1, h))
            dlg.accept()

        def _on_full() -> None:
            selected_full["value"] = True
            dlg.accept()

        btn_apply.clicked.connect(_on_apply)
        btn_full.clicked.connect(_on_full)
        btn_cancel.clicked.connect(dlg.reject)

        if dlg.exec() != int(QDialog.DialogCode.Accepted):
            return None
        if selected_full["value"]:
            return (0, 0, src_w, src_h)
        return result["roi"]

    def _refresh_views_after_roi_change(self) -> None:
        """ROI 변경 후 영상 로드/1차/2차 미리보기를 현재 프레임 기준으로 다시 렌더링합니다."""
        self._refresh_video_load_page()
        self._refresh_second_advanced_page()
        if self.video_cap is None:
            return
        index = 0
        if self.sliderTimeline is not None:
            index = int(self.sliderTimeline.value())
        frame = self._read_video_frame(index)
        if frame is None:
            return
        overlay_boxes = self._get_overlay_boxes_for_frame(index) if self.is_processing else []
        self._show_frame(frame, overlay_boxes)
        self._update_time_label(index)
        if not self.video_load_hover_active:
            self._set_video_load_preview_frame(frame)
        self.last_second_preview_frame = frame
        if self.labelSecondVideoPreview is not None:
            self.labelSecondVideoPreview.setPixmap(self._frame_to_pixmap(frame, max_width=960, max_height=560))
            self.labelSecondVideoPreview.setText("")

    def on_set_video_roi(self) -> None:
        """영상 로드 단계에서 현재 프레임 기준 ROI를 선택해 이후 처리에 반영합니다."""
        if self.video_path is None or self.video_cap is None:
            QMessageBox.information(self, "ROI 설정", "먼저 영상을 불러오세요.")
            return
        if self.is_processing or self.is_exporting:
            return
        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
            self._update_play_pause_button_icon()

        frame_idx = 0
        if self.sliderTimeline is not None:
            frame_idx = int(self.sliderTimeline.value())
        frame = self._read_video_frame_raw(frame_idx)
        if frame is None:
            QMessageBox.warning(self, "ROI 설정", "현재 프레임을 불러오지 못했습니다.")
            return

        selected = self._show_video_roi_dialog(frame)
        if selected is None:
            return
        normalized = self._normalize_video_roi(selected, int(frame.shape[1]), int(frame.shape[0]))
        self.video_roi = normalized
        self._clear_thumbnail_source_cache()
        self._update_video_load_roi_label()
        self._refresh_views_after_roi_change()
        if normalized is None:
            self._append_log("user action: ROI reset to full frame")
        else:
            x, y, w, h = normalized
            self._append_log(f"user action: ROI set ({x},{y},{w},{h})")

    def on_reset_video_roi(self) -> None:
        """선택된 ROI를 해제하고 전체 프레임 기준으로 되돌립니다."""
        if self.video_path is None:
            return
        if self.is_processing or self.is_exporting:
            return
        self.video_roi = None
        self._clear_thumbnail_source_cache()
        self._update_video_load_roi_label()
        self._refresh_views_after_roi_change()
        self._append_log("user action: ROI reset to full frame")

    def _setup_page_navigation_shell(self) -> None:
        """UI/타이머/시그널 등 초기 구성을 수행하고 기본 상태를 연결합니다."""
        _setup_page_navigation_shell_page(self, MAIN_PAGE_ICON_PATH)

    def _relocate_topbar_actions_to_nav(self, nav_layout: QVBoxLayout, nav_panel: QWidget) -> None:
        """상단 액션 버튼을 내비게이션 패널의 빠른 작업 카드로 이동 배치합니다."""
        _relocate_topbar_actions_to_nav_page(self, nav_layout, nav_panel)

    def _build_home_page(self) -> QWidget:
        """홈 화면 카드(히어로/가이드)를 구성한 페이지 위젯을 생성해 반환합니다."""
        return _build_home_page_page(self, MAIN_PAGE_ICON_PATH)

    def _build_video_load_page(self) -> QWidget:
        """영상 로드, 메타 정보, 호버 미리보기 UI를 포함한 페이지를 구성해 반환합니다."""
        return _build_video_load_page_page(self, HoverPreviewLabel)

    def _build_second_advanced_page(self) -> QWidget:
        """2차 고도화의 단계 헤더, 스택 페이지, 네비게이션 버튼을 포함한 루트 페이지를 생성합니다."""
        return _build_second_advanced_page_page(self)

    def _build_second_stage_labeling_step(self, parent: QWidget) -> QWidget:
        """2차 고도화 1단계(인터랙티브 라벨링) 화면 위젯을 구성해 반환합니다."""
        return _build_second_stage_labeling_step_page(self, parent)

    def _install_theme_toggle(self, nav_layout: QVBoxLayout) -> None:
        """라이트/다크 테마 토글 버튼을 생성하고 클릭 시 테마 적용 및 설정 저장을 연결합니다."""
        _install_theme_toggle_page(self, nav_layout, apply_theme)

    def _on_second_stage_labeling_complete(self) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        _on_second_stage_labeling_complete_page(self)

    def _set_second_stage_step(self, index: int) -> None:
        """2차 고도화 단계 인덱스를 전환하고 단계별 상태/버튼 활성화를 동기화합니다."""
        _set_second_stage_step_page(self, index)

    def _reset_second_stage_progress(self) -> None:
        """작업 중 누적된 상태를 초기값으로 되돌려 다음 실행을 준비합니다."""
        _reset_second_stage_progress_page(self)

    def _on_second_timeline_changed(self, value: int) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        _on_second_timeline_changed_page(self, value)

    def _update_second_timeline_label(self, frame_index: int) -> None:
        """현재 데이터와 상태를 기준으로 UI 표시값 또는 내부 상태를 동기화합니다."""
        _update_second_timeline_label_page(self, frame_index)

    def _refresh_second_advanced_page(self) -> None:
        """현재 상태를 기준으로 화면/목록/미리보기를 다시 그려 최신 상태로 갱신합니다."""
        _refresh_second_advanced_page_page(self)

    def _build_training_page(self) -> QWidget:
        """학습 진행 요약과 제어 버튼을 포함한 학습 페이지를 구성해 반환합니다."""
        return _build_training_page_page(
            self,
            metric_chart_widget_cls=MetricChartWidget,
            train_default_freeze=TRAIN_DEFAULT_FREEZE,
            train_default_lr0=TRAIN_DEFAULT_LR0,
            train_replay_ratio_default=TRAIN_REPLAY_RATIO_DEFAULT,
            train_stage1_epochs_default=TRAIN_STAGE1_EPOCHS_DEFAULT,
            train_stage2_epochs_default=TRAIN_STAGE2_EPOCHS_DEFAULT,
            train_stage2_lr_factor_default=TRAIN_STAGE2_LR_FACTOR_DEFAULT,
            train_retrain_seed_default=TRAIN_RETRAIN_SEED_DEFAULT,
            train_stage_unfreeze_neck_only=TRAIN_STAGE_UNFREEZE_NECK_ONLY,
            train_stage_unfreeze_backbone_last=TRAIN_STAGE_UNFREEZE_BACKBONE_LAST,
        )

    def _build_model_test_page(self) -> QWidget:
        """학습된 모델 테스트 페이지를 구성해 반환합니다."""
        return _build_model_test_page_page(self)
    def _navigate_to_page(self, page_key: str, animate: bool = True) -> None:
        """요청된 화면으로 이동하고 전환에 필요한 부가 상태를 함께 갱신합니다."""
        _navigate_to_page_page(self, page_key, animate)

    def _set_nav_checked(self, page_key: str) -> None:
        """내비게이션 버튼 그룹에서 현재 페이지 버튼의 체크 상태를 갱신합니다."""
        _set_nav_checked_page(self, page_key)

    def _switch_page_index(self, target_index: int, animate: bool = True) -> None:
        """활성 페이지 또는 모드를 전환하고 동기화 동작을 수행합니다."""
        _switch_page_index_page(self, target_index, animate)

    def _on_page_fade_out_finished(self) -> None:
        """사용자 입력 또는 비동기 이벤트를 수신해 후속 처리 흐름을 실행합니다."""
        _on_page_fade_out_finished_page(self)

    def _schedule_advanced1_preview_refresh(self) -> None:
        """1차 고도화 페이지 진입 직후 프리뷰가 작게 보이지 않도록 리사이즈를 보정합니다."""
        _schedule_advanced1_preview_refresh_page(self)

    def _refresh_advanced1_preview_fit(self) -> None:
        """메인 영상 프리뷰를 현재 라벨 크기에 맞춰 다시 렌더링합니다."""
        _refresh_advanced1_preview_fit_page(self)

    def _refresh_video_load_page(self) -> None:
        """현재 상태를 기준으로 화면/목록/미리보기를 다시 그려 최신 상태로 갱신합니다."""
        _refresh_video_load_page_page(self)

    def _show_placeholder_dialog(self, title: str, message: str) -> None:
        """대상 프레임 또는 대화상자를 사용자 화면에 표시합니다."""
        QMessageBox.information(self, title, message)

    def _connect_signals(self) -> None:
        """버튼/슬라이더/탭의 시그널을 각 이벤트 처리 슬롯에 연결합니다."""
        if self.btnOpenVideo is not None:
            self.btnOpenVideo.clicked.connect(self.on_open_export_folder)
        if self.btnStart is not None:
            self.btnStart.clicked.connect(lambda: self.on_start_processing(fast_mode=False))
        if self.btnStartFast is not None:
            self.btnStartFast.clicked.connect(lambda: self.on_start_processing(fast_mode=True))
        if self.btnStop is not None:
            self.btnStop.clicked.connect(self.on_stop_processing)
        if self.btnExportResult is not None:
            self.btnExportResult.clicked.connect(self.on_export_dataset)
        if self.btnExit is not None:
            self.btnExit.clicked.connect(self.close)

        if self.sliderTimeline is not None:
            self.sliderTimeline.valueChanged.connect(self.on_timeline_changed)
        if self.listThumbs is not None:
            self.listThumbs.itemDoubleClicked.connect(self._on_thumbnail_double_clicked)
        if self.tabAll is not None:
            self.tabAll.clicked.connect(lambda: self._set_thumbnail_filter("all"))
        if self.tabKeep is not None:
            self.tabKeep.clicked.connect(lambda: self._set_thumbnail_filter("keep"))
        if self.tabHold is not None:
            self.tabHold.clicked.connect(lambda: self._set_thumbnail_filter("hold"))
        if self.tabDrop is not None:
            self.tabDrop.clicked.connect(lambda: self._set_thumbnail_filter("drop"))
        if self.btnLoadPreviewCache is not None:
            self.btnLoadPreviewCache.clicked.connect(self.on_load_cached_preview)

        if self.btnSeekBack is not None:
            self.btnSeekBack.clicked.connect(self.on_stop_playback)
        if self.btnPrevFrame is not None:
            self.btnPrevFrame.clicked.connect(lambda: self._seek_relative(-1))
        if self.btnNextFrame is not None:
            self.btnNextFrame.clicked.connect(lambda: self._seek_relative(1))
        if self.btnPlayPause is not None:
            self.btnPlayPause.clicked.connect(self.on_toggle_playback)
        if self.comboTrainEngine is not None:
            self.comboTrainEngine.currentIndexChanged.connect(self._on_training_engine_changed)
        if self.comboTrainModel is not None:
            self.comboTrainModel.currentIndexChanged.connect(self._on_training_model_changed)


## =====================================
## 함수 기능 : 기존객체라벨링 고급 설정 다이얼로그
## 매개 변수 : parent (QWidget)
## 반환 결과 : 없음 (exec()로 모달 실행)
## =====================================
class AdvancedLabelingSettingsDialog(QDialog):
    """검출·추적·필터 임계치를 한 곳에서 조정하는 고급 설정 팝업."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("고급 설정 — 기존 객체 라벨링")
        self.setMinimumWidth(480)
        self.setModal(True)

        root = QVBoxLayout(self)
        root.setSpacing(14)

        # ── 섹션 헬퍼 ────────────────────────────────────────────────
        def _section(title: str) -> tuple[QFormLayout, QWidget]:
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(4)
            lbl = QLabel(f"<b>{title}</b>")
            v.addWidget(lbl)
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            v.addWidget(line)
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            form.setHorizontalSpacing(12)
            v.addLayout(form)
            root.addWidget(box)
            return form, box

        def _dspin(lo, hi, step, val, tip) -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(2)
            s.setValue(val)
            s.setToolTip(tip)
            s.setFixedWidth(100)
            return s

        def _ispin(lo, hi, step, val, tip) -> QSpinBox:
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setValue(val)
            s.setToolTip(tip)
            s.setFixedWidth(100)
            return s

        def _row(form, label, widget, desc):
            lbl = QLabel(label)
            lbl.setToolTip(desc)
            widget.setToolTip(desc)
            form.addRow(lbl, widget)
            # 설명 라벨 (회색 소자)
            desc_lbl = QLabel(f"<small style='color:#888'>{desc}</small>")
            desc_lbl.setWordWrap(True)
            form.addRow("", desc_lbl)

        # ── 1. 검출 설정 ──────────────────────────────────────────────
        f1, _ = _section("🔍 검출 설정 (YOLO)")
        self.spinConf = _dspin(0.01, 1.0, 0.05, 0.10,
            "검출 신뢰도 하한값. 이 값 미만의 탐지는 무시됩니다. 낮을수록 더 많이 검출되지만 오탐도 증가합니다.")
        self.spinIou = _dspin(0.01, 1.0, 0.05, 0.45,
            "NMS(비최대억제) IoU 임계값. 겹치는 박스 제거 기준입니다. 낮을수록 더 많은 중복 박스를 제거합니다.")
        self.spinImgsz = _ispin(32, 4096, 32, 320,
            "추론 이미지 크기(픽셀). 모델 학습 시 사용한 크기와 맞추면 정확도가 높아집니다.")
        _row(f1, "검출 conf", self.spinConf,
             "검출 신뢰도 하한값. 이 값 미만의 탐지는 무시됩니다. 낮을수록 더 많이 검출되지만 오탐도 증가합니다.")
        _row(f1, "NMS iou", self.spinIou,
             "NMS(비최대억제) IoU 임계값. 겹치는 박스 제거 기준입니다. 낮을수록 더 많은 중복 박스를 제거합니다.")
        _row(f1, "추론 imgsz", self.spinImgsz,
             "추론 이미지 크기(픽셀). 모델 학습 시 사용한 크기와 맞추면 정확도가 높아집니다.")

        # ── 2. 추적 필터 설정 ─────────────────────────────────────────
        f2, _ = _section("🎯 추적 필터 설정 (TrackObject)")
        self.spinConfHigh = _dspin(0.01, 1.0, 0.05, 0.20,
            "KEEP 판정 신뢰도 기준. 이 값 이상이면 KEEP, 미만이면 HOLD로 분류됩니다. 검출 conf보다 높게 설정하세요.")
        self.spinValidFrames = _ispin(1, 30, 1, 5,
            "VALID 상태로 승격되기 위해 연속으로 탐지되어야 하는 프레임 수. 낮을수록 빨리 저장되지만 오탐이 늘어납니다.")
        self.spinTrackIou = _dspin(0.01, 1.0, 0.05, 0.50,
            "연속 프레임 간 박스 IoU 검증 임계값. 이 값 미만이면 같은 객체로 인정하지 않고 버퍼를 초기화합니다.")
        self.spinSizeDiff = _dspin(0.01, 1.0, 0.05, 0.20,
            "연속 프레임 간 박스 크기 변화 허용 비율. 초과 시 다른 객체로 판단해 버퍼를 초기화합니다.")
        self.spinAreaChange = _dspin(0.01, 1.0, 0.05, 0.30,
            "VALID 상태에서 이전 프레임 대비 면적 변화 허용 비율. 초과 시 해당 프레임은 SKIP됩니다.")
        self.spinRatioChange = _dspin(0.01, 1.0, 0.05, 0.30,
            "VALID 상태에서 이전 프레임 대비 가로세로 비율 변화 허용 비율. 초과 시 해당 프레임은 SKIP됩니다.")
        self.spinHoldFrames = _ispin(1, 60, 1, 5,
            "KEEP 이후 신뢰도가 낮아져도 HOLD를 유지하는 프레임 수. 짧은 신뢰도 저하를 허용합니다.")
        _row(f2, "KEEP 상한 conf", self.spinConfHigh,
             "KEEP 판정 신뢰도 기준. 이 값 이상이면 KEEP, 미만이면 HOLD로 분류됩니다.")
        _row(f2, "검증 프레임 수", self.spinValidFrames,
             "VALID 상태로 승격되기 위해 연속으로 탐지되어야 하는 프레임 수.")
        _row(f2, "추적 IoU 임계값", self.spinTrackIou,
             "연속 프레임 간 박스 IoU 검증 임계값. 이 값 미만이면 버퍼를 초기화합니다.")
        _row(f2, "크기 변화 허용", self.spinSizeDiff,
             "연속 프레임 간 박스 크기 변화 허용 비율.")
        _row(f2, "면적 변화 한도", self.spinAreaChange,
             "VALID 상태에서 면적 변화 허용 비율. 초과 시 SKIP됩니다.")
        _row(f2, "비율 변화 한도", self.spinRatioChange,
             "VALID 상태에서 가로세로 비율 변화 허용 비율. 초과 시 SKIP됩니다.")
        _row(f2, "HOLD 유지 프레임", self.spinHoldFrames,
             "KEEP 이후 신뢰도가 낮아져도 HOLD를 유지하는 프레임 수.")

        # ── 3. 경계 설정 ──────────────────────────────────────────────
        f3, _ = _section("📐 경계 설정")
        self.spinBorderMargin = _ispin(0, 200, 1, 5,
            "이미지 경계에서 이 픽셀 이내에 걸치는 박스는 경계 잡음으로 간주해 제외합니다. 0이면 모든 박스 허용.")
        _row(f3, "경계 여백(px)", self.spinBorderMargin,
             "이미지 경계에서 이 픽셀 이내에 걸치는 박스는 경계 잡음으로 간주해 제외합니다. 0이면 모든 박스 허용.")

        # ── 확인/취소 버튼 ────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_ok = QPushButton("확인")
        btn_ok.setFixedWidth(80)
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("취소")
        btn_cancel.setFixedWidth(80)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        root.addLayout(btn_row)

        # 스크롤 래퍼
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner.setLayout(root)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    ## =====================================
    ## 함수 기능 : 현재 다이얼로그 설정값을 딕셔너리로 반환
    ## 매개 변수 : 없음
    ## 반환 결과 : dict — 모든 임계치 값
    ## =====================================
    def get_values(self) -> dict:
        return {
            "yolo_conf":               self.spinConf.value(),
            "yolo_iou":                self.spinIou.value(),
            "yolo_imgsz":              self.spinImgsz.value(),
            "track_conf_high":         self.spinConfHigh.value(),
            "track_validation_frames": self.spinValidFrames.value(),
            "track_iou_threshold":     self.spinTrackIou.value(),
            "track_size_diff_threshold": self.spinSizeDiff.value(),
            "track_area_change_limit": self.spinAreaChange.value(),
            "track_ratio_change_limit": self.spinRatioChange.value(),
            "track_hold_frames":       self.spinHoldFrames.value(),
            "border_margin":           self.spinBorderMargin.value(),
        }

