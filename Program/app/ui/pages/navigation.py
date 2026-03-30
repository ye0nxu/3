from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QSettings, QTimer, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


def _setup_page_navigation_shell_page(self: Any, main_page_icon_path: Path) -> None:
    """Compose the page shell and navigation stack."""
    if self.rootLayout is None or self.primary_workbench_widget is None:
        return
    if getattr(self, "_page_shell_ready", False):
        return

    for idx in range(self.rootLayout.count()):
        item = self.rootLayout.itemAt(idx)
        if item is not None and item.widget() is self.primary_workbench_widget:
            self.rootLayout.takeAt(idx)
            break

    self.page_shell_widget = QWidget(self)
    shell_layout = QHBoxLayout(self.page_shell_widget)
    shell_layout.setContentsMargins(0, 0, 0, 0)
    shell_layout.setSpacing(14)

    nav_panel = QFrame(self.page_shell_widget)
    nav_panel.setObjectName("pageNavPanel")
    nav_panel.setMinimumWidth(220)
    nav_panel.setMaximumWidth(260)
    nav_layout = QVBoxLayout(nav_panel)
    nav_layout.setContentsMargins(12, 12, 12, 12)
    nav_layout.setSpacing(8)

    brand_icon = QLabel(nav_panel)
    brand_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if main_page_icon_path.is_file():
        icon_pixmap = QPixmap(str(main_page_icon_path))
        if not icon_pixmap.isNull():
            brand_icon.setPixmap(
                icon_pixmap.scaled(
                    92,
                    92,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
    nav_layout.addWidget(brand_icon)

    brand_title = QLabel("작업 네비게이션", nav_panel)
    brand_title.setObjectName("labelNavBrandTitle")
    brand_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    nav_layout.addWidget(brand_title)

    self.page_button_group = QButtonGroup(self)
    self.page_button_group.setExclusive(True)

    page_defs = [
        ("home", "홈"),
        ("video_load", "영상입력"),
        ("advanced_1", "기존 객체 라벨링"),
        ("advanced_2", "신규 객체 라벨링"),
        ("training", "모델 학습"),
        ("model_test", "모델 테스트"),
    ]
    page_tooltips = {
        "home": "프로그램 개요와 사용 흐름(기존/신규 라벨링 경로)을 확인합니다.",
        "video_load": (
            "영상 입력 페이지입니다.\n"
            "- 분석할 영상을 선택하고 ROI를 설정합니다.\n"
            "- 지원 포맷: mp4 / avi / mov / mkv\n"
            "- 해상도/FPS/프레임 정보, 10초 미리보기를 제공합니다.\n"
            "- 화면 하단 버튼으로 기존/신규 라벨링 페이지로 이동합니다."
        ),
        "advanced_1": (
            "기존 객체 라벨링 단계입니다.\n"
            "- 기본 처리: 화면 확인형\n"
            "- 빠른 처리: 고속 데이터셋 생성\n"
            "- 결과는 Keep / Hold / Drop으로 분류됩니다."
        ),
        "advanced_2": (
            "신규 객체 오토라벨링 단계입니다.\n"
            "- 새로운 클래스 기준으로 라벨링\n"
            "- 신규 학습용 데이터셋 생성"
        ),
        "training": (
            "모델 학습 단계입니다.\n"
            "- YOLO / RT-DETR 엔진 선택\n"
            "- 신규학습 / 재학습 실행\n"
            "- EPOCH, IMGSZ, BATCH 설정 가능"
        ),
        "model_test": (
            "학습된 모델 검증 단계입니다.\n"
            "- best.pt 모델과 테스트 영상 선택\n"
            "- 프레임별 추론 결과를 실시간 확인"
        ),
    }
    for key, text in page_defs:
        btn = QPushButton(text, nav_panel)
        btn.setCheckable(True)
        btn.setMinimumHeight(38)
        btn.setProperty("navButton", True)
        btn.setToolTip(page_tooltips.get(key, ""))
        btn.clicked.connect(lambda _checked, page_key=key: self._navigate_to_page(page_key, animate=True))
        self.page_button_group.addButton(btn)
        self.nav_buttons[key] = btn
        nav_layout.addWidget(btn)

    self._relocate_topbar_actions_to_nav(nav_layout=nav_layout, nav_panel=nav_panel)
    nav_layout.addStretch(1)
    self._install_theme_toggle(nav_layout)
    nav_hint = QLabel("페이지 전환 시 현재 작업 상태는 유지됩니다.", nav_panel)
    nav_hint.setWordWrap(True)
    nav_hint.setStyleSheet("color: rgb(113, 126, 149); font-size: 9pt;")
    nav_layout.addWidget(nav_hint)

    self.page_stack = QStackedWidget(self.page_shell_widget)
    self.page_stack.setObjectName("pageStack")

    pages: list[tuple[str, QWidget]] = [
        ("home", self._build_home_page()),
        ("video_load", self._build_video_load_page()),
        ("advanced_1", self.primary_workbench_widget),
        ("advanced_2", self._build_second_advanced_page()),
        ("training", self._build_training_page()),
        ("model_test", self._build_model_test_page()),
    ]
    for key, page in pages:
        self.page_indices[key] = self.page_stack.addWidget(page)
    if hasattr(self, "_style_home_hero_section"):
        self._style_home_hero_section()
    if hasattr(self, "_style_home_guide_section"):
        self._style_home_guide_section()

    self._page_fade_effect = QGraphicsOpacityEffect(self.page_stack)
    self._page_fade_effect.setOpacity(1.0)
    self.page_stack.setGraphicsEffect(self._page_fade_effect)

    shell_layout.addWidget(nav_panel, 0)
    shell_layout.addWidget(self.page_stack, 1)

    self.rootLayout.insertWidget(1, self.page_shell_widget, 1)
    self.rootLayout.setStretch(1, 1)
    self._page_shell_ready = True
    self._navigate_to_page("home", animate=False)
    self._refresh_video_load_page()


def _relocate_topbar_actions_to_nav_page(self: Any, nav_layout: QVBoxLayout, nav_panel: QWidget) -> None:
    """Move quick actions into the navigation panel."""
    if getattr(self, "_topbar_actions_relocated", False):
        return

    buttons = [self.btnOpenVideo, self.btnExportResult]
    if self.topBarLayout is not None:
        for button in buttons:
            if button is not None:
                self.topBarLayout.removeWidget(button)

    actions_card = QFrame(nav_panel)
    actions_card.setObjectName("quickActionsCard")
    actions_card.setProperty("pageCard", True)
    actions_layout = QVBoxLayout(actions_card)
    actions_layout.setContentsMargins(10, 10, 10, 10)
    actions_layout.setSpacing(6)

    actions_title = QLabel("빠른 작업", actions_card)
    actions_title.setObjectName("labelQuickActionsTitle")
    actions_layout.addWidget(actions_title)

    for button in buttons:
        if button is None:
            continue
        button.setParent(actions_card)
        button.setProperty("quickAction", True)
        button.setMinimumHeight(32)
        if button is self.btnOpenVideo:
            button.setToolTip("최근 내보낸 데이터셋 저장 폴더를 엽니다.")
        elif button is self.btnExportResult:
            button.setToolTip("현재 라벨링 결과를 학습용 데이터셋으로 내보냅니다.")
        actions_layout.addWidget(button)

    nav_layout.addWidget(actions_card)
    self._topbar_actions_relocated = True


def _install_theme_toggle_page(
    self: Any,
    nav_layout: QVBoxLayout,
    apply_theme_fn: Callable[[QApplication, str], str],
) -> None:
    """Install the light/dark theme toggle row in the nav panel."""
    app = QApplication.instance()
    if app is None:
        return

    settings = QSettings("SL_TEAM", "AutoLabelStudio")
    theme = str(settings.value("theme", "dark")).strip().lower()
    if theme not in ("dark", "light"):
        theme = "dark"

    row = QFrame(self)
    row.setProperty("pageCard", True)
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(10, 8, 10, 8)
    row_layout.setSpacing(8)

    label = QLabel("테마", row)
    label.setStyleSheet("font-weight: 600;")
    row_layout.addWidget(label)

    btn = QToolButton(row)
    btn.setObjectName("btnThemeToggle")
    btn.setCheckable(True)
    btn.setChecked(theme == "light")
    btn.setText("☀ Light" if theme == "light" else "🌙 Dark")
    btn.setToolTip("라이트/다크 전환")
    btn.setMinimumHeight(32)

    def _apply(new_theme: str) -> None:
        applied = apply_theme_fn(app, new_theme)
        settings.setValue("theme", applied)
        btn.setChecked(applied == "light")
        btn.setText("☀ Light" if applied == "light" else "🌙 Dark")
        if hasattr(self, "_reapply_session_status_style"):
            self._reapply_session_status_style()
        if hasattr(self, "_style_home_hero_section"):
            self._style_home_hero_section()
        if hasattr(self, "_style_home_guide_section"):
            self._style_home_guide_section()
        if hasattr(self, "_style_advanced1_playback_controls"):
            self._style_advanced1_playback_controls()
        if hasattr(self, "_update_play_pause_button_icon"):
            self._update_play_pause_button_icon()
        widget = getattr(self, "newObjectLabelingWidget", None)
        if widget is not None and hasattr(widget, "refresh_theme"):
            widget.refresh_theme()

    btn.toggled.connect(lambda checked: _apply("light" if checked else "dark"))

    row_layout.addStretch(1)
    row_layout.addWidget(btn)
    nav_layout.addWidget(row)
    _apply(theme)


def _navigate_to_page_page(self: Any, page_key: str, animate: bool = True) -> None:
    """Navigate to a page in the main stack."""
    if self.page_stack is None:
        return
    target_index = self.page_indices.get(page_key)
    if target_index is None:
        return
    if page_key != "video_load":
        self._stop_video_load_hover_preview()
    self.current_page_key = page_key
    self._set_nav_checked(page_key)
    self._switch_page_index(target_index, animate=animate)


def _set_nav_checked_page(self: Any, page_key: str) -> None:
    """Sync the checked nav button to the current page."""
    for key, button in self.nav_buttons.items():
        should_check = key == page_key
        if button.isChecked() == should_check:
            continue
        button.blockSignals(True)
        button.setChecked(should_check)
        button.blockSignals(False)


def _switch_page_index_page(self: Any, target_index: int, animate: bool = True) -> None:
    """Switch the stacked page and refresh dependent previews."""
    if self.page_stack is None:
        return
    if self.page_stack.currentIndex() == target_index:
        self._refresh_video_load_page()
        self._refresh_second_advanced_page()
        self._schedule_advanced1_preview_refresh()
        return

    if (not animate) or self._page_fade_effect is None:
        self.page_stack.setCurrentIndex(target_index)
        self._refresh_video_load_page()
        self._refresh_second_advanced_page()
        self._schedule_advanced1_preview_refresh()
        return

    if self._page_fade_out is not None:
        self._page_fade_out.stop()
    if self._page_fade_in is not None:
        self._page_fade_in.stop()

    self._pending_page_index = target_index
    self._page_fade_out = QPropertyAnimation(self._page_fade_effect, b"opacity", self)
    self._page_fade_out.setDuration(150)
    self._page_fade_out.setEasingCurve(QEasingCurve.Type.InOutCubic)
    self._page_fade_out.setStartValue(float(self._page_fade_effect.opacity()))
    self._page_fade_out.setEndValue(0.06)
    self._page_fade_out.finished.connect(self._on_page_fade_out_finished)
    self._page_fade_out.start()


def _on_page_fade_out_finished_page(self: Any) -> None:
    """Complete the fade-out transition and fade back in."""
    if self.page_stack is None or self._page_fade_effect is None:
        return
    if self._pending_page_index is None:
        return

    target_index = int(self._pending_page_index)
    self._pending_page_index = None
    self.page_stack.setCurrentIndex(target_index)
    self._refresh_video_load_page()
    self._refresh_second_advanced_page()
    self._schedule_advanced1_preview_refresh()

    self._page_fade_in = QPropertyAnimation(self._page_fade_effect, b"opacity", self)
    self._page_fade_in.setDuration(220)
    self._page_fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)
    self._page_fade_in.setStartValue(0.06)
    self._page_fade_in.setEndValue(1.0)
    self._page_fade_in.start()


def _schedule_advanced1_preview_refresh_page(self: Any) -> None:
    """Refresh advanced page 1 preview after layout settles."""
    if self.current_page_key != "advanced_1":
        return
    QTimer.singleShot(0, self._refresh_advanced1_preview_fit)
    QTimer.singleShot(40, self._refresh_advanced1_preview_fit)


def _refresh_advanced1_preview_fit_page(self: Any) -> None:
    """Re-fit the primary preview pixmap to the label size."""
    if self.current_page_key != "advanced_1":
        return
    if self.last_preview_frame is None or self.labelVideoPlaceholder is None:
        return
    if self.labelVideoPlaceholder.width() <= 1 or self.labelVideoPlaceholder.height() <= 1:
        return
    self.labelVideoPlaceholder.setPixmap(self._frame_to_pixmap(self.last_preview_frame))
    self.labelVideoPlaceholder.setText("")
