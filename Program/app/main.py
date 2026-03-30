from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import QCoreApplication, QLibraryInfo
from PyQt6.QtWidgets import QApplication

from backend.llm.manager import LLMServerManager
from app.ui.main_window import create_main_window


class MainController:
    """앱 부트스트랩 및 메인 윈도우 실행을 담당합니다."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        auto_manage_llm = str(os.getenv("LLM_SERVER_AUTOSTART", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.llm_server_manager = LLMServerManager() if auto_manage_llm else None

    def run(self) -> int:
        if self.llm_server_manager is not None:
            self.llm_server_manager.start()
        plugins_path = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
        if plugins_path.is_dir():
            QCoreApplication.setLibraryPaths([str(plugins_path)])

        app = QApplication(sys.argv)
        window = create_main_window(self.project_root)
        screen = app.primaryScreen()
        if screen is not None:
            window.setFixedSize(screen.geometry().size())
        window.showFullScreen()
        try:
            return app.exec()
        finally:
            if self.llm_server_manager is not None:
                self.llm_server_manager.stop()


def main() -> None:
    controller = MainController()
    raise SystemExit(controller.run())
