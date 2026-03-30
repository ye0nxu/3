from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget


def _build_home_page_page(self: Any, main_page_icon_path: Path) -> QWidget:
    """홈 화면을 프로그램 소개 + 사용 흐름 카드 형태로 구성해 반환합니다."""
    _ = main_page_icon_path
    page = QWidget(self)
    root_layout = QVBoxLayout(page)
    root_layout.setContentsMargins(0, 0, 0, 0)
    root_layout.setSpacing(0)

    scroll = QScrollArea(page)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidgetResizable(True)
    root_layout.addWidget(scroll, 1)

    content = QWidget(scroll)
    scroll.setWidget(content)
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(14)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    hero = QFrame(content)
    hero.setObjectName("homeHeroCard")
    hero.setProperty("pageCard", True)
    hero_layout = QHBoxLayout(hero)
    hero_layout.setContentsMargins(32, 22, 32, 22)
    hero_layout.setSpacing(14)

    hero_text_col = QVBoxLayout()
    hero_text_col.setContentsMargins(0, 0, 0, 0)
    hero_text_col.setSpacing(7)

    session_badge = QLabel("SESSION READY", hero)
    session_badge.setObjectName("labelHomeSessionBadge")
    hero_text_col.addWidget(session_badge, 0, Qt.AlignmentFlag.AlignLeft)

    title = QLabel("Auto Labeling Tool", hero)
    title.setObjectName("labelHomeHeroTitle")
    title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    hero_text_col.addWidget(title)

    chip_row = QHBoxLayout()
    chip_row.setContentsMargins(0, 6, 0, 0)
    chip_row.setSpacing(8)
    for text in ("AUTO LABELING", "MERGE DATASET", "TRAINING"):
        chip = QLabel(text, hero)
        chip.setProperty("homeHeroChip", True)
        chip_row.addWidget(chip, 0)
    chip_row.addStretch(1)
    hero_text_col.addLayout(chip_row)

    process_row = QHBoxLayout()
    process_row.setContentsMargins(0, 6, 0, 0)
    process_row.setSpacing(10)

    process_badge = QLabel("PROCESS", hero)
    process_badge.setObjectName("labelHomeProcessBadge")
    process_row.addWidget(process_badge, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    process_divider = QFrame(hero)
    process_divider.setObjectName("homeHeroProcessDivider")
    process_divider.setFixedWidth(1)
    process_divider.setMinimumHeight(20)
    process_row.addWidget(process_divider, 0, Qt.AlignmentFlag.AlignVCenter)

    process_text = QLabel(
        "영상 입력 -> 라벨링(기존/신규) -> 결과 내보내기 -> 학습/재학습",
        hero,
    )
    process_text.setObjectName("labelHomeProcessText")
    process_text.setWordWrap(True)
    process_row.addWidget(process_text, 1, Qt.AlignmentFlag.AlignVCenter)
    hero_text_col.addLayout(process_row)

    hero_layout.addLayout(hero_text_col, 1)
    hero.setMinimumHeight(240)
    hero.setMaximumHeight(280)
    layout.addWidget(hero, 0)

    guide = QFrame(content)
    guide.setObjectName("homeGuideSection")
    guide.setProperty("pageCard", True)
    guide.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    guide_layout = QVBoxLayout(guide)
    guide_layout.setContentsMargins(26, 20, 26, 20)
    guide_layout.setSpacing(12)

    guide_title = QLabel("사용 가이드", guide)
    guide_title.setObjectName("labelPageCardTitle")
    guide_layout.addWidget(guide_title, 0, Qt.AlignmentFlag.AlignLeft)

    def _build_guide_card(
        badge_text: str,
        title_text: str,
        desc_text: str,
        step_lines: Sequence[str],
        tip_text: str,
        variant: str,
    ) -> QFrame:
        card = QFrame(guide)
        card.setObjectName("homeGuideFlowCard")
        card.setProperty("pageCard", True)
        card.setProperty("homeGuideFlowCard", True)
        card.setProperty("guideVariant", variant)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 18, 20, 18)
        card_layout.setSpacing(10)

        badge = QLabel(badge_text, card)
        badge.setObjectName("labelHomeGuideBadge")
        badge.setProperty("guideVariant", variant)
        card_layout.addWidget(badge, 0, Qt.AlignmentFlag.AlignLeft)

        card_title = QLabel(title_text, card)
        card_title.setObjectName("labelHomeGuideTitle")
        card_layout.addWidget(card_title)

        desc_label = QLabel(desc_text, card)
        desc_label.setObjectName("labelHomeGuideDesc")
        desc_label.setWordWrap(True)
        card_layout.addWidget(desc_label)

        divider_top = QFrame(card)
        divider_top.setFrameShape(QFrame.Shape.HLine)
        divider_top.setProperty("homeGuideDivider", True)
        card_layout.addWidget(divider_top)

        flow_title = QLabel("진행 흐름", card)
        flow_title.setObjectName("labelHomeGuideFlowTitle")
        card_layout.addWidget(flow_title)

        circled_nums = ("①", "②", "③", "④", "⑤")
        for idx, step in enumerate(step_lines):
            prefix = circled_nums[idx] if idx < len(circled_nums) else f"{idx + 1}."
            step_label = QLabel(f"{prefix}  {step}", card)
            step_label.setProperty("homeGuideStep", True)
            step_label.setWordWrap(True)
            card_layout.addWidget(step_label)

        divider_bottom = QFrame(card)
        divider_bottom.setFrameShape(QFrame.Shape.HLine)
        divider_bottom.setProperty("homeGuideDivider", True)
        card_layout.addWidget(divider_bottom)

        tip_title = QLabel("Tip", card)
        tip_title.setObjectName("labelHomeGuideTipTitle")
        card_layout.addWidget(tip_title)

        tip_desc = QLabel(tip_text, card)
        tip_desc.setObjectName("labelHomeGuideTipDesc")
        tip_desc.setWordWrap(True)
        card_layout.addWidget(tip_desc)
        card.setMinimumHeight(470)
        return card

    cards_row = QHBoxLayout()
    cards_row.setContentsMargins(0, 0, 0, 0)
    cards_row.setSpacing(18)
    cards_row.setAlignment(Qt.AlignmentFlag.AlignTop)
    cards_row.addWidget(
        _build_guide_card(
            "MODEL UPGRADE",
            "기존 객체 라벨링",
            "이미 인식 가능한 객체 모델이 있고, 성능을 개선하려는 경우 사용합니다.",
            ("영상입력", "기존 객체 라벨링", "결과 내보내기", "학습 실행", "재학습 실행"),
            "데이터를 추가 수집한 후 재학습하면 성능을 지속적으로 향상할 수 있습니다.",
            "upgrade",
        ),
        1,
    )
    cards_row.addWidget(
        _build_guide_card(
            "NEW CLASS",
            "신규 객체 라벨링",
            "인식 모델이 없거나 새로운 객체를 처음 학습하는 경우 사용합니다.",
            ("영상입력", "프롬프트 및 클래스 입력", "오토라벨링", "결과 내보내기", "신규학습 실행"),
            "초기 학습 후 기존 객체 라벨링 -> 재학습으로 점진적 고도화가 가능합니다.",
            "new",
        ),
        1,
    )
    guide_layout.addLayout(cards_row, 0)

    layout.addWidget(guide, 0)
    if hasattr(self, "_style_home_hero_section"):
        self._style_home_hero_section()
    if hasattr(self, "_style_home_guide_section"):
        self._style_home_guide_section()
    return page
