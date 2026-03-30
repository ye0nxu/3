from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.dataset import FrameAnnotation

@dataclass(slots=True)
class ProgressEvent:
    """작업 단계, 처리량, 프레임 위치, ETA 등을 UI로 전달하기 위한 진행 이벤트 데이터입니다."""
    stage: str
    processed_units: int
    total_units: int
    current_frame: int
    total_frames: int
    remaining_items: int
    eta_seconds: float

@dataclass(slots=True)
class WorkerOutput:
    """워커 처리 결과(주석/미리보기/클래스/메타)를 한 번에 전달하기 위한 결과 컨테이너입니다."""
    frame_annotations: list[FrameAnnotation]
    preview_items: list["PreviewThumbnail"]
    class_names: list[str]
    video_info: dict[str, Any]
    run_config: dict[str, Any]

@dataclass(slots=True)
class PreviewThumbnail:
    """썸네일 미리보기에 필요한 프레임 이미지, 박스 정보, 분류 상태를 담는 데이터입니다."""
    frame_index: int
    image: np.ndarray | None = None
    boxes: list[Any] = field(default_factory=list)
    category: str = "keep"
    item_id: str = ""
    image_path: str | None = None
    thumb_path: str | None = None
    manifest_path: str | None = None

@dataclass(slots=True)
class ExportRunSummary:
    """직접 내보내기 실행 결과(분할별 이미지/라벨 개수, 클래스 맵, 총 박스 수) 요약입니다."""
    dataset_root: Path
    train_images: int
    valid_images: int
    test_images: int
    train_labels: int
    valid_labels: int
    test_labels: int
    class_count: int
    total_boxes: int
    class_names: list[str] = field(default_factory=list)
    crop_images: int = 0
    crop_root: Path | None = None

class WorkerStoppedError(Exception):
    """사용자 중단 요청으로 워커 파이프라인을 정상 종료할 때 사용하는 예외입니다."""
    pass
