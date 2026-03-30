from __future__ import annotations

from enum import Enum


class TrackState(str, Enum):
    ## =====================================
    ## 함수 기능 : 추적 객체의 내부 상태를 나타내는 열거형
    ## =====================================
    CANDIDATE = "CANDIDATE"
    VALID = "VALID"


class TrackAction(str, Enum):
    ## =====================================
    ## 함수 기능 : process() 반환 액션을 나타내는 열거형
    ## =====================================
    KEEP = "KEEP"
    KEEP_BUFFER = "KEEP_BUFFER"
    HOLD = "HOLD"
    HOLD_BUFFER = "HOLD_BUFFER"
    DROP = "DROP"
    SKIP = "SKIP"
    BUFFERING = "BUFFERING"


# 전역 기본값 (UI 고급설정에서 덮어씀)
CONF_LOW           = 0.10
CONF_HIGH          = 0.20
HOLD_FRAMES        = 5
VALIDATION_FRAMES  = 5
IOU_THRESHOLD      = 0.5
SIZE_DIFF_THRESHOLD = 0.2
AREA_CHANGE_LIMIT  = 0.3
RATIO_CHANGE_LIMIT = 0.3


class TrackObject:
    """
    KEEP / HOLD / DROP 상태를 판단하는 추적 필터 클래스입니다.

    사용 API:
    - `process(bbox, score, mask_points, frame_idx, img_shape) -> (action, reason)`
    - 워커에서 참조하는 상태 필드: `state`, `buffer`, `last_seen_frame`, `status_msg`
    """
    __slots__ = (
        "id",
        "state",
        "buffer",
        "prev_area",
        "prev_ratio",
        "last_seen_high",
        "last_seen_frame",
        "has_graduated",
        "status_msg",
        "conf_low",
        "conf_high",
        "validation_frames",
        "iou_threshold",
        "size_diff_threshold",
        "area_change_limit",
        "ratio_change_limit",
        "hold_frames",
    )

    def __init__(
        self,
        obj_id: int,
        frame_idx: int,
        conf_low: float | None = None,
        conf_high: float | None = None,
        validation_frames: int | None = None,
        iou_threshold: float | None = None,
        size_diff_threshold: float | None = None,
        area_change_limit: float | None = None,
        ratio_change_limit: float | None = None,
        hold_frames: int | None = None,
    ) -> None:
        """객체 생성 시 필요한 의존성, 기본값, 내부 상태를 초기화합니다."""
        self.id = int(obj_id)
        self.state = TrackState.CANDIDATE
        self.buffer: list[dict[str, object]] = []
        self.prev_area: float | None = None
        self.prev_ratio: float | None = None
        self.last_seen_high = int(frame_idx)
        self.last_seen_frame = int(frame_idx)
        self.has_graduated = False
        self.status_msg = "New"
        self.conf_low         = float(conf_low)          if conf_low          is not None else CONF_LOW
        self.conf_high        = float(conf_high)         if conf_high         is not None else CONF_HIGH
        self.validation_frames = int(validation_frames)  if validation_frames is not None else VALIDATION_FRAMES
        self.iou_threshold    = float(iou_threshold)     if iou_threshold     is not None else IOU_THRESHOLD
        self.size_diff_threshold = float(size_diff_threshold) if size_diff_threshold is not None else SIZE_DIFF_THRESHOLD
        self.area_change_limit = float(area_change_limit) if area_change_limit is not None else AREA_CHANGE_LIMIT
        self.ratio_change_limit = float(ratio_change_limit) if ratio_change_limit is not None else RATIO_CHANGE_LIMIT
        self.hold_frames      = int(hold_frames)         if hold_frames       is not None else HOLD_FRAMES

    def process(
        self,
        bbox,
        score: float,
        mask_points,
        frame_idx: int,
        img_shape,
    ) -> tuple[str, str]:
        """검출 박스와 점수를 상태 머신 규칙으로 평가해 KEEP/HOLD/DROP 액션과 사유를 반환합니다."""
        del img_shape

        self.last_seen_frame = int(frame_idx)
        parsed = self._parse_bbox(bbox)
        if parsed is None:
            self.status_msg = "Drop (Invalid BBox)"
            return "DROP", "Invalid BBox"

        x1, y1, x2, y2 = parsed
        current_area = self._area(parsed)
        current_ratio = (x2 - x1) / max(1e-6, (y2 - y1))
        conf = float(score)

        if conf < self.conf_low:
            self.status_msg = "Drop (Low Score)"
            return "DROP", "Low Score"

        if self.state == TrackState.CANDIDATE:
            candidate_item = {
                "bbox": [x1, y1, x2, y2],
                "score": conf,
                "frame": int(frame_idx),
                "mask": mask_points,
            }
            if self.buffer and isinstance(self.buffer[-1], dict):
                try:
                    last_frame = int(self.buffer[-1].get("frame", -1))
                except Exception:
                    last_frame = -1
                if last_frame == int(frame_idx):
                    try:
                        prev_score = float(self.buffer[-1].get("score", -1.0))
                    except Exception:
                        prev_score = -1.0
                    if conf >= prev_score:
                        carry_prefilter = self.buffer[-1].get("prefilter_candidate")
                        self.buffer[-1] = candidate_item
                        if carry_prefilter is not None:
                            self.buffer[-1]["prefilter_candidate"] = carry_prefilter
                else:
                    self.buffer.append(candidate_item)
            else:
                self.buffer.append(candidate_item)
            self.status_msg = f"Buffering({len(self.buffer)}/{self.validation_frames})"

            if len(self.buffer) < self.validation_frames:
                return "BUFFERING", "Wait Validation Frames"

            prev_bbox = self.buffer[-2]["bbox"]
            iou = self._iou(prev_bbox, [x1, y1, x2, y2])
            prev_area = self._area(prev_bbox)
            size_diff = abs(current_area - prev_area) / max(prev_area, 1e-6)
            if iou < self.iou_threshold or size_diff > self.size_diff_threshold:
                self.buffer = []
                self.status_msg = "Drop (Validation Fail)"
                return "DROP", f"Validation Fail (IoU:{iou:.2f}, SizeDiff:{size_diff:.2f})"

            self.state = TrackState.VALID
            self.prev_area = current_area
            self.prev_ratio = current_ratio
            if conf >= self.conf_high:
                self.has_graduated = True
                self.last_seen_high = int(frame_idx)
                self.status_msg = "Keep (Validated)"
                return "KEEP_BUFFER", "Validated High Score"

            self.status_msg = "Hold (Validated)"
            return "HOLD_BUFFER", "Validated Hold Range"

        if self.state == TrackState.VALID:
            if self.prev_area is not None:
                area_change = abs(current_area - self.prev_area) / max(self.prev_area, 1e-6)
                if area_change > self.area_change_limit:
                    self.status_msg = "Skip (Area Anomaly)"
                    return "SKIP", "Area Anomaly"

            if self.prev_ratio is not None:
                ratio_change = abs(current_ratio - self.prev_ratio) / max(abs(self.prev_ratio), 1e-6)
                if ratio_change > self.ratio_change_limit:
                    self.status_msg = "Skip (Ratio Anomaly)"
                    return "SKIP", "Ratio Anomaly"

            self.prev_area = current_area
            self.prev_ratio = current_ratio

            if conf >= self.conf_high:
                self.has_graduated = True
                self.last_seen_high = int(frame_idx)
                self.status_msg = "Keep"
                return "KEEP", "High Confidence"

            if self.conf_low <= conf < self.conf_high:
                if not self.has_graduated:
                    self.status_msg = "Hold (Pre-Graduation)"
                    return "HOLD", "Mid Confidence"
                if frame_idx - self.last_seen_high <= self.hold_frames:
                    self.status_msg = f"Hold (Grace {frame_idx - self.last_seen_high})"
                    return "HOLD", "Post-High Grace"
                self.status_msg = "Skip (Hold Timeout)"
                return "SKIP", "Hold Timeout"

        self.status_msg = "Skip (Unknown State)"
        return "SKIP", "Unknown State"

    def _parse_bbox(self, bbox) -> tuple[float, float, float, float] | None:
        """입력 박스를 float형 xyxy 좌표로 변환하고 유효성(x2>x1, y2>y1)을 검증해 반환합니다."""
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _iou(self, box1, box2) -> float:
        """두 바운딩 박스의 교집합 대비 합집합 비율(IoU)을 계산해 반환합니다."""
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0

        area1 = self._area(box1)
        area2 = self._area(box2)
        union = area1 + area2 - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def _area(self, box) -> float:
        """입력 박스 좌표(x1,y1,x2,y2)로부터 너비와 높이를 계산해 면적을 반환합니다."""
        return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
