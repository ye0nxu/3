from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from core.paths import APP_ASSETS_ROOT, TRAIN_YOLO_MODELS_DIR

ASSETS_ROOT = APP_ASSETS_ROOT
# =========================
# Theme (Light/Dark) Toggle
# =========================
THEME_DARK_QSS = ASSETS_ROOT / "styles" / "main_window.qss"          # ✅ 기존 qss = 다크로 사용
THEME_LIGHT_QSS = ASSETS_ROOT / "styles" / "main_window_light.qss"  # ✅ 없으면 자동 라이트 오버라이드로 대체

# 라이트 파일이 없을 때도 "확실히 밝아지게" 하는 최소 오버라이드
LIGHT_FALLBACK_OVERRIDE_QSS = """
/* ---- light fallback override ---- */
QWidget { background: #F5F7FA; color: #111827; }
QFrame[property="pageCard"], QFrame[pageCard="true"], QWidget[pageCard="true"] {
    background: #FFFFFF;
    border: 1px solid rgba(17,24,39,0.12);
    border-radius: 12px;
}
QLabel { color: #111827; }
QLineEdit, QPlainTextEdit, QTextEdit, QListWidget, QTableWidget, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #FFFFFF;
    color: #111827;
    border: 1px solid rgba(17,24,39,0.18);
    border-radius: 8px;
}
QPushButton, QToolButton {
    background: #FFFFFF;
    color: #111827;
    border: 1px solid rgba(17,24,39,0.20);
    border-radius: 10px;
    padding: 6px 10px;
}
QPushButton:hover, QToolButton:hover { background: #EEF2F7; }
QProgressBar { background: #FFFFFF; border: 1px solid rgba(17,24,39,0.18); border-radius: 8px; }
"""

def _read_text(path: Path) -> str:
    """파일/프레임 등 외부 소스에서 데이터를 읽고 실패 시 안전한 기본 결과를 반환합니다."""
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except Exception:
        pass
    return ""

def apply_theme(app: QApplication, theme: str) -> str:
    """
    앱 전역 테마를 적용합니다.

    - `dark`: `main_window.qss`를 그대로 적용합니다.
    - `light`: `main_window_light.qss`가 있으면 적용하고, 없으면 라이트 오버라이드로 대체합니다.
    """
    t = (theme or "").strip().lower()
    if t not in ("dark", "light"):
        t = "dark"

    dark_qss = _read_text(THEME_DARK_QSS)
    if t == "dark":
        app.setStyleSheet(dark_qss)
        return "dark"

    light_qss = _read_text(THEME_LIGHT_QSS)
    if light_qss:
        app.setStyleSheet(light_qss)
        return "light"

    # 라이트 파일이 없으면: 기존 다크 QSS + 라이트 오버라이드로 "밝게" 보이게
    app.setStyleSheet((dark_qss or "") + "\n" + LIGHT_FALLBACK_OVERRIDE_QSS)
    return "light"

DEFAULT_TEAM_MODEL_NAME = "yolo11n-seg.pt"
TEAM_MODEL_PATH = TRAIN_YOLO_MODELS_DIR / DEFAULT_TEAM_MODEL_NAME
MAIN_PAGE_ICON_PATH = ASSETS_ROOT / "icons" / "main_page_icon.png"
DEFAULT_CLASS_NAMES: list[str] = ["객체"]
PROVENANCE_JSON_BEGIN = "---PROVENANCE_JSON_BEGIN---"
PROVENANCE_JSON_END = "---PROVENANCE_JSON_END---"
IMAGE_EXTENSIONS_LOWER = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
INVALID_WIN_PATH_CHARS_RE = re.compile(r'[\\/:*?"<>|]+')
TRAIN_DEFAULT_FREEZE = 10
TRAIN_DEFAULT_LR0 = 0.002
TRAIN_REPLAY_RATIO_DEFAULT = 0.10
TRAIN_STAGE1_EPOCHS_DEFAULT = 10
TRAIN_STAGE2_EPOCHS_DEFAULT = 40
TRAIN_STAGE2_LR_FACTOR_DEFAULT = 0.3
TRAIN_RETRAIN_SEED_DEFAULT = 42
TRAIN_STAGE_UNFREEZE_NECK_ONLY = "neck_only"
TRAIN_STAGE_UNFREEZE_BACKBONE_LAST = "backbone_last"
TRAIN_STAGE_UNFREEZE_CHOICES = (
    TRAIN_STAGE_UNFREEZE_NECK_ONLY,
    TRAIN_STAGE_UNFREEZE_BACKBONE_LAST,
)
