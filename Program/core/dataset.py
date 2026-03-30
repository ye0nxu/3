from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class BoxAnnotation:
    """단일 객체의 클래스 ID와 바운딩 박스 좌표를 표현하는 주석 데이터입니다."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    track_id: int | None = None


@dataclass
class FrameAnnotation:
    """한 프레임의 이미지 정보와 객체 박스 목록을 저장하는 프레임 단위 주석 데이터입니다."""

    frame_index: int
    image: np.ndarray | None = None
    image_path: str | None = None
    image_name: str = ""
    split: str = "train"
    boxes: list[BoxAnnotation | Mapping[str, Any] | Sequence[Any]] = field(default_factory=list)
    timestamp_sec: float = 0.0

