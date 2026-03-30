from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_log = logging.getLogger(__name__)

# ─── 파라미터 그룹 정의 ─────────────────────────────────────────────────────
# 각 항목: (key, 한국어 레이블, 한국어 설명, 기본값, 위젯종류, min, max, decimals, step)
# widget_type: "int" → QSpinBox | "float" → QDoubleSpinBox | "combo" → QComboBox
_PARAM_GROUPS: list[tuple[str, list[tuple[str, str, str, Any, str, Any, Any, Any, Any]]]] = [
    (
        "기본 학습",
        [
            ("epochs",  "학습 에포크 수",    "전체 데이터셋을 몇 번 반복 학습할지 설정합니다.",                          100,   "int",   1,      1000,  None, 1),
            ("imgsz",   "입력 이미지 크기",   "학습에 사용할 이미지 해상도(픽셀). 클수록 정확도↑, 속도↓",               640,   "int",   32,     1280,  None, 32),
            ("batch",   "배치 크기",          "한 번에 처리할 이미지 수. GPU 메모리에 맞게 조정하세요.",                  16,    "int",   1,      512,   None, 1),
            ("device",  "학습 장치",          "학습에 사용할 하드웨어를 선택합니다.",                                      "0",   "combo", None,   None,  None, None),
        ],
    ),
    (
        "학습률",
        [
            ("lr0",             "초기 학습률",          "학습 시작 시 가중치를 업데이트하는 보폭 크기입니다.",              0.01,   "float", 0.0001, 1.0,   6, 0.001),
            ("lrf",             "최종 학습률 비율",      "학습 종료 시 lr0 대비 최종 학습률 비율 (lr0 × lrf).",            0.01,   "float", 0.0001, 1.0,   6, 0.001),
            ("momentum",        "모멘텀",                "이전 업데이트 방향을 얼마나 유지할지 결정합니다.",                0.937,  "float", 0.0,    0.999, 4, 0.001),
            ("weight_decay",    "가중치 감쇠",           "과적합 방지를 위해 큰 가중치에 패널티를 줍니다.",                 0.0005, "float", 0.0,    0.1,   6, 0.0001),
            ("warmup_epochs",   "워밍업 에포크",         "학습 초반에 학습률을 서서히 올리는 구간(에포크 수).",             3.0,    "float", 0.0,    10.0,  2, 0.5),
            ("warmup_momentum", "워밍업 모멘텀",         "워밍업 구간에서 사용하는 초기 모멘텀 값입니다.",                  0.8,    "float", 0.0,    0.999, 3, 0.01),
            ("warmup_bias_lr",  "워밍업 바이어스 학습률","워밍업 구간에서 바이어스에 적용하는 학습률입니다.",               0.1,    "float", 0.0,    1.0,   4, 0.01),
        ],
    ),
    (
        "데이터 증강",
        [
            ("hsv_h",       "색조 변환 강도",       "이미지 색조(Hue)를 무작위로 변경하는 증강 강도입니다.",                0.015, "float", 0.0, 1.0,   4, 0.005),
            ("hsv_s",       "채도 변환 강도",       "이미지 채도(Saturation)를 무작위로 변경하는 증강 강도입니다.",        0.7,   "float", 0.0, 1.0,   3, 0.05),
            ("hsv_v",       "밝기 변환 강도",       "이미지 밝기(Value)를 무작위로 변경하는 증강 강도입니다.",              0.4,   "float", 0.0, 1.0,   3, 0.05),
            ("degrees",     "회전 증강",            "이미지를 무작위로 회전하는 최대 각도(°)입니다.",                       0.0,   "float", 0.0, 180.0, 2, 1.0),
            ("translate",   "이동 증강",            "이미지를 가로/세로로 무작위 이동하는 비율입니다.",                     0.1,   "float", 0.0, 0.9,   3, 0.05),
            ("scale",       "크기 증강",            "이미지를 무작위로 확대/축소하는 비율 범위입니다.",                     0.5,   "float", 0.0, 0.9,   3, 0.05),
            ("shear",       "전단 변환",            "이미지를 평행사변형 방향으로 비트는 최대 각도(°).",                    0.0,   "float", 0.0, 90.0,  2, 1.0),
            ("perspective", "원근 변환",            "이미지에 원근감 왜곡을 적용하는 강도입니다.",                          0.0,   "float", 0.0, 0.001, 6, 0.0001),
            ("flipud",      "상하 반전 확률",       "이미지를 위아래로 뒤집을 확률입니다.",                                 0.0,   "float", 0.0, 1.0,   3, 0.05),
            ("fliplr",      "좌우 반전 확률",       "이미지를 좌우로 뒤집을 확률입니다. (수평 대칭 물체에 권장)",           0.5,   "float", 0.0, 1.0,   3, 0.05),
            ("mosaic",      "모자이크 증강 확률",   "4개 이미지를 합쳐 1장으로 만드는 증강 확률입니다.",                    1.0,   "float", 0.0, 1.0,   3, 0.05),
            ("mixup",       "믹스업 증강 확률",     "두 이미지를 투명하게 겹쳐 혼합하는 증강 확률입니다.",                  0.0,   "float", 0.0, 1.0,   3, 0.05),
            ("copy_paste",  "복사-붙여넣기 증강",   "객체를 다른 이미지에 무작위로 붙여넣는 증강 확률입니다.",              0.0,   "float", 0.0, 1.0,   3, 0.05),
            ("close_mosaic","모자이크 종료 에포크", "마지막 N 에포크에서 모자이크 증강을 끄는 시점입니다.",                 10,    "int",   0,   100,   None, 1),
        ],
    ),
    (
        "손실 가중치",
        [
            ("box", "박스 손실 가중치",  "바운딩박스 위치 손실의 반영 비중입니다.",        7.5, "float", 0.0, 100.0, 2, 0.5),
            ("cls", "분류 손실 가중치",  "클래스 분류 손실의 반영 비중입니다.",            0.5, "float", 0.0, 100.0, 2, 0.1),
            ("dfl", "DFL 손실 가중치",   "Distribution Focal Loss 반영 비중입니다.",       1.5, "float", 0.0, 100.0, 2, 0.1),
        ],
    ),
    (
        "임계치 & 추론",
        [
            ("conf",    "신뢰도 임계치",      "이 값 이상인 탐지 결과만 최종 출력합니다.",                                     0.25, "float", 0.0, 1.0,   3, 0.05),
            ("iou",     "IoU 임계치 (NMS)",   "겹치는 박스를 제거할 때 기준이 되는 IoU 값입니다.",                             0.7,  "float", 0.0, 1.0,   3, 0.05),
            ("max_det", "최대 탐지 수",       "이미지 한 장에서 출력할 수 있는 최대 탐지 객체 수입니다.",                      300,  "int",   1,   10000, None, 10),
        ],
    ),
]

# conf / iou 추가 설명 블록
_EXTRA_NOTES: dict[str, str] = {
    "conf": (
        "낮은 값 (예: 0.1) → 오탐(False Positive) 증가, 작은 객체 검출 유리\n"
        "높은 값 (예: 0.7) → 미탐(False Negative) 증가, 정밀도 향상\n"
        "권장 범위: 0.20 ~ 0.35"
    ),
    "iou": (
        "낮은 값 (예: 0.3) → 겹친 박스를 더 공격적으로 제거 (밀집 객체에 불리)\n"
        "높은 값 (예: 0.7) → 박스를 더 많이 남김 (중복 탐지 허용)\n"
        "권장 범위: 0.45 ~ 0.70"
    ),
}

# 기본값 딕셔너리 (키 → 기본값)
_DEFAULTS: dict[str, Any] = {}
for _gname, _params in _PARAM_GROUPS:
    for _p in _params:
        _DEFAULTS[_p[0]] = _p[3]


class YoloParamsDialog(QDialog):
    ## =====================================
    ## 함수 기능 : YOLO 학습 하이퍼파라미터 설정 다이얼로그
    ## 매개 변수 : parent(QWidget | None)
    ## 반환 결과 : None
    ## =====================================
    """
    YOLO 학습 하이퍼파라미터 전체를 그룹별로 조회·편집하는 모달 다이얼로그.

    사용 예::

        dialog = YoloParamsDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.current_params()
    """

    _SETTINGS_ORG = "SL_TEAM"
    _SETTINGS_APP = "AutoLabelStudio"
    _SETTINGS_PREFIX = "yolo_params"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("YOLO 학습 파라미터")
        self.setModal(True)
        self.resize(700, 740)

        # 키 → 위젯 매핑
        self._widgets: dict[str, QSpinBox | QDoubleSpinBox | QComboBox] = {}

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 14, 14, 10)
        root_layout.setSpacing(10)

        # ── 스크롤 영역 ──────────────────────────────────────────────────────
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget(scroll)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(14)
        scroll.setWidget(content)
        root_layout.addWidget(scroll, 1)

        # ── 파라미터 그룹 생성 ───────────────────────────────────────────────
        for group_name, params in _PARAM_GROUPS:
            group_box = QGroupBox(group_name, content)
            form = QFormLayout(group_box)
            form.setContentsMargins(12, 14, 12, 12)
            form.setSpacing(10)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            for p in params:
                key, label_text, desc_text, default, wtype, vmin, vmax, decimals, step = p
                widget = self._build_widget(key, default, wtype, vmin, vmax, decimals, step)
                self._widgets[key] = widget

                cell = QWidget(group_box)
                cell_layout = QVBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 4)
                cell_layout.setSpacing(3)
                cell_layout.addWidget(widget)

                desc_lbl = QLabel(desc_text, cell)
                desc_lbl.setObjectName("labelParamDesc")
                desc_lbl.setWordWrap(True)
                cell_layout.addWidget(desc_lbl)

                if key in _EXTRA_NOTES:
                    note_lbl = QLabel(_EXTRA_NOTES[key], cell)
                    note_lbl.setObjectName("labelParamNote")
                    note_lbl.setWordWrap(True)
                    cell_layout.addWidget(note_lbl)

                row_label = QLabel(label_text, group_box)
                form.addRow(row_label, cell)

            content_layout.addWidget(group_box)

        content_layout.addStretch(1)

        # ── 버튼 행 ──────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        btn_reset = QPushButton("기본값 복원", self)
        btn_reset.setObjectName("btnResetDefaults")
        btn_reset.clicked.connect(self._reset_to_defaults)
        btn_row.addWidget(btn_reset, 0)
        btn_row.addStretch(1)

        btn_apply = QPushButton("적용", self)
        btn_apply.setObjectName("btnApplyParams")
        btn_apply.setDefault(True)
        btn_apply.clicked.connect(self.accept)

        btn_cancel = QPushButton("취소", self)
        btn_cancel.setObjectName("btnCancelParams")
        btn_cancel.clicked.connect(self.reject)

        btn_row.addWidget(btn_apply, 0)
        btn_row.addWidget(btn_cancel, 0)
        root_layout.addLayout(btn_row)

        # 저장된 값 복원
        self._load_from_settings()

    # ── 위젯 생성 ─────────────────────────────────────────────────────────────

    def _build_widget(
        self,
        key: str,
        default: Any,
        widget_type: str,
        vmin: Any,
        vmax: Any,
        decimals: int | None,
        step: Any,
    ) -> QSpinBox | QDoubleSpinBox | QComboBox:
        ## =====================================
        ## 함수 기능 : 파라미터 종류에 맞는 입력 위젯을 생성합니다
        ## 매개 변수 : key(str), default(Any), widget_type(str), vmin, vmax, decimals, step
        ## 반환 결과 : QSpinBox | QDoubleSpinBox | QComboBox
        ## =====================================
        if widget_type == "combo":
            w = QComboBox(self)
            w.addItem("0  (GPU 0)", "0")
            w.addItem("cpu", "cpu")
            idx = w.findData(str(default))
            w.setCurrentIndex(max(0, idx))
            return w

        if widget_type == "int":
            w = QSpinBox(self)
            w.setRange(int(vmin), int(vmax))
            if step is not None:
                w.setSingleStep(int(step))
            w.setValue(int(default))
            return w

        # float
        w = QDoubleSpinBox(self)
        w.setRange(float(vmin), float(vmax))
        if decimals is not None:
            w.setDecimals(int(decimals))
        if step is not None:
            w.setSingleStep(float(step))
        w.setValue(float(default))
        return w

    # ── 값 접근 헬퍼 ──────────────────────────────────────────────────────────

    def _get_value(self, key: str) -> Any:
        ## =====================================
        ## 함수 기능 : 지정 키의 위젯 현재 값을 반환합니다
        ## 매개 변수 : key(str)
        ## 반환 결과 : Any -> int | float | str
        ## =====================================
        w = self._widgets.get(key)
        if isinstance(w, QComboBox):
            return str(w.currentData() or w.currentText())
        if isinstance(w, QSpinBox):
            return int(w.value())
        if isinstance(w, QDoubleSpinBox):
            return float(w.value())
        return _DEFAULTS.get(key)

    def _set_value(self, key: str, value: Any) -> None:
        ## =====================================
        ## 함수 기능 : 지정 키의 위젯 값을 설정합니다
        ## 매개 변수 : key(str), value(Any)
        ## 반환 결과 : None
        ## =====================================
        w = self._widgets.get(key)
        if w is None:
            return
        try:
            if isinstance(w, QComboBox):
                idx = w.findData(str(value))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif isinstance(w, QSpinBox):
                w.setValue(int(value))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(float(value))
        except Exception:
            pass

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def current_params(self) -> dict[str, Any]:
        ## =====================================
        ## 함수 기능 : 현재 다이얼로그의 모든 파라미터 값을 딕셔너리로 반환합니다
        ## 매개 변수 : 없음
        ## 반환 결과 : dict[str, Any] -> 파라미터 키-값 쌍
        ## =====================================
        return {key: self._get_value(key) for key in self._widgets}

    # ── 기본값 / QSettings 처리 ───────────────────────────────────────────────

    def _reset_to_defaults(self) -> None:
        ## =====================================
        ## 함수 기능 : 모든 위젯 값을 기본값으로 초기화합니다
        ## 매개 변수 : 없음
        ## 반환 결과 : None
        ## =====================================
        for key, default in _DEFAULTS.items():
            self._set_value(key, default)
        _log.info("YOLO 파라미터 기본값으로 초기화")

    def _load_from_settings(self) -> None:
        ## =====================================
        ## 함수 기능 : QSettings에서 저장된 파라미터 값을 읽어 위젯에 복원합니다
        ## 매개 변수 : 없음
        ## 반환 결과 : None
        ## =====================================
        settings = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        for key, default in _DEFAULTS.items():
            raw = settings.value(f"{self._SETTINGS_PREFIX}/{key}", default)
            try:
                self._set_value(key, raw)
            except Exception:
                self._set_value(key, default)

    def _save_to_settings(self, params: dict[str, Any]) -> None:
        ## =====================================
        ## 함수 기능 : 파라미터 딕셔너리를 QSettings에 영속 저장합니다
        ## 매개 변수 : params(dict[str, Any])
        ## 반환 결과 : None
        ## =====================================
        settings = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        for key, value in params.items():
            settings.setValue(f"{self._SETTINGS_PREFIX}/{key}", value)

    # ── QDialog 오버라이드 ────────────────────────────────────────────────────

    def accept(self) -> None:
        ## =====================================
        ## 함수 기능 : "적용" 클릭 시 파라미터를 저장하고 다이얼로그를 닫습니다
        ## 매개 변수 : 없음
        ## 반환 결과 : None
        ## =====================================
        params = self.current_params()
        self._save_to_settings(params)
        _log.info("YOLO 파라미터 적용됨: %s", params)
        super().accept()
