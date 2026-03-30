from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import QLibraryInfo

from core.paths import APP_ROOT as PROJECT_ROOT

ULTRALYTICS_LOCAL_ROOT = PROJECT_ROOT.parent / "ultralytics-main"
if ULTRALYTICS_LOCAL_ROOT.is_dir():
    local_ultra_path = str(ULTRALYTICS_LOCAL_ROOT)
    if local_ultra_path not in sys.path:
        sys.path.insert(0, local_ultra_path)

def _configure_qt_plugin_env() -> None:
    """
    Windows 환경에서 Qt 플러그인 경로를 PyQt6 기준으로 재설정합니다.

    OpenCV가 설정한 Qt 경로와 충돌할 때 발생하는
    `Qt platform plugin 'windows'` 로드 오류를 방지하기 위한 초기화입니다.
    """
    for key in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        value = os.environ.get(key, "")
        if "cv2" in value.lower() or "opencv" in value.lower():
            os.environ.pop(key, None)

    plugins_path = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
    binaries_path = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.BinariesPath))
    os.environ["QT_PLUGIN_PATH"] = str(plugins_path)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins_path / "platforms")
    os.environ.setdefault("QT_QPA_PLATFORM", "windows")
    os.environ.setdefault("QT_OPENGL", "software")

    if os.name == "nt":
        # Ensure Qt6 DLLs from PyQt6 are preferred over cv2-shipped binaries.
        for path in (binaries_path, plugins_path, plugins_path / "platforms"):
            if path.is_dir():
                try:
                    os.add_dll_directory(str(path))
                except Exception:
                    pass

        path_entries = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        filtered_entries = [p for p in path_entries if ("opencv" not in p.lower() and "cv2" not in p.lower())]
        if str(binaries_path) not in filtered_entries:
            filtered_entries.insert(0, str(binaries_path))
        os.environ["PATH"] = os.pathsep.join(filtered_entries)


_configure_qt_plugin_env()
import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

try:
    import ultralytics as ultralytics_package  # type: ignore
except Exception:  # pragma: no cover - runtime dependency check
    ultralytics_package = None  # type: ignore[assignment]

try:
    from ultralytics import YOLO as UltralyticsYOLO  # type: ignore
except Exception:  # pragma: no cover - runtime dependency check
    UltralyticsYOLO = None  # type: ignore[assignment]

try:
    from ultralytics import RTDETR as UltralyticsRTDETR  # type: ignore
except Exception:  # pragma: no cover - runtime dependency check
    UltralyticsRTDETR = None  # type: ignore[assignment]

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:
    from tqdm import tqdm as TqdmFormatter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TqdmFormatter = None  # type: ignore[assignment]

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

ULTRALYTICS_VERSION = str(getattr(ultralytics_package, "__version__", "")).strip()

try:
    from core.tracking import TrackObject as TeamTrackObject
except Exception:  # pragma: no cover - runtime dependency check
    TeamTrackObject = None  # type: ignore[assignment]

try:
    from backend.filters import FilterConfig, SampleCandidate, SampleFilterEngine
except Exception:  # pragma: no cover - optional dependency
    FilterConfig = None  # type: ignore[assignment]
    SampleCandidate = None  # type: ignore[assignment]
    SampleFilterEngine = None  # type: ignore[assignment]

STAGE_NAME_KR: dict[str, str] = {
    "Filtering": "필터링",
    "Detection": "객체 검출",
    "Tracking": "객체 추적",
    "Labeling": "라벨링",
    "Completed": "완료",
}


def _to_korean_stage(stage: str) -> str:
    """영문 단계 키를 UI 표시용 한글 단계명으로 변환해 반환합니다."""
    return STAGE_NAME_KR.get(str(stage), str(stage))


def _to_korean_class_name(name: str) -> str:
    """클래스 이름 표시값을 정리(공백 제거)해 반환합니다."""
    return str(name).strip()


