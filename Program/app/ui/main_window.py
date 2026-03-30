from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QMainWindow


def create_main_window(project_root: Path) -> QMainWindow:
    ## =====================================
    ## 함수 기능 : QSettings에서 마지막 테마를 읽어 적용 후 메인 윈도우를 생성해 반환합니다
    ## 매개 변수 : project_root(Path) -> 프로젝트 루트 경로
    ## 반환 결과 : QMainWindow -> 생성된 메인 윈도우 인스턴스
    ## =====================================
    # 순환 임포트 방지를 위해 함수 내부에서 임포트
    from app.window import AutoLabelStudioWindow
    from app.studio.config import apply_theme

    ui_path = project_root / "assets" / "ui" / "main_window.ui"
    app = QApplication.instance()
    if app is not None:
        settings = QSettings("SL_TEAM", "AutoLabelStudio")
        theme = str(settings.value("theme", "dark")).strip().lower()
        applied = apply_theme(app, theme)
        settings.setValue("theme", applied)
    return AutoLabelStudioWindow(ui_path)
