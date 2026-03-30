from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .detector import Detection


def _bbox_center(b: np.ndarray) -> Tuple[float, float]:
    return float((b[0] + b[2]) * 0.5), float((b[1] + b[3]) * 0.5)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _hungarian_assign(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Hungarian assignment for rectangular matrix.
    Returns list of (row_idx, col_idx).
    """
    if cost_matrix.size == 0:
        return []
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be 2D")

    n_rows, n_cols = cost_matrix.shape
    transposed = False
    c = np.asarray(cost_matrix, dtype=np.float64)
    if n_rows > n_cols:
        c = c.T
        n_rows, n_cols = c.shape
        transposed = True

    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int32)
    way = np.zeros(n_cols + 1, dtype=np.int32)

    inf = 1e18
    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = inf
            j1 = 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = c[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: List[Tuple[int, int]] = []
    for j in range(1, n_cols + 1):
        i = int(p[j])
        if i > 0:
            r = i - 1
            col = j - 1
            if transposed:
                # cost matrix was transposed: row=orig col, col=orig row
                pairs.append((col, r))
            else:
                pairs.append((r, col))
    return pairs


@dataclass
class Track:
    track_id: int
    bbox_xyxy: np.ndarray
    score: float
    prompt: str
    hits: int = 1
    age: int = 1
    misses: int = 0
    vx: float = 0.0
    vy: float = 0.0
    predicted_frames: int = 0
    updated_this_frame: bool = False
    last_mask: Optional[np.ndarray] = None

    def predict_bbox(self) -> np.ndarray:
        x0, y0, x1, y1 = self.bbox_xyxy.astype(np.float32)
        return np.asarray([x0 + self.vx, y0 + self.vy, x1 + self.vx, y1 + self.vy], dtype=np.float32)


class BaseTracker:
    def update(
        self,
        detections: Sequence[Detection],
        detection_performed: bool = True,
    ) -> List[Track]:
        raise NotImplementedError


class IoUByteLikeTracker(BaseTracker):
    def __init__(
        self,
        iou_match_threshold: float,
        max_missing_frames: int,
        min_hits: int,
        fallback_cfg: Optional[Dict] = None,
    ) -> None:
        self.iou_match_threshold = float(iou_match_threshold)
        self.max_missing_frames = int(max_missing_frames)
        self.min_hits = int(min_hits)
        cfg = fallback_cfg or {}
        self.center_distance_weight = float(cfg.get("center_distance_weight", 0.35))
        self.max_center_distance_ratio = float(cfg.get("max_center_distance_ratio", 1.2))
        self.prompt_mismatch_penalty = float(cfg.get("prompt_mismatch_penalty", 0.2))
        self.prompt_match_bonus = float(cfg.get("prompt_match_bonus", 0.05))
        self.min_match_score = float(cfg.get("min_match_score", 0.05))
        self.bbox_smooth_alpha = float(cfg.get("bbox_smooth_alpha", 0.85))
        self.velocity_smooth_alpha = float(cfg.get("velocity_smooth_alpha", 0.7))
        self.velocity_decay = float(cfg.get("velocity_decay", 0.92))
        self.prediction_velocity_decay = float(cfg.get("prediction_velocity_decay", 0.82))
        self.prediction_score_decay = float(cfg.get("prediction_score_decay", 0.9))
        self.max_velocity_ratio = float(cfg.get("max_velocity_ratio", 0.35))
        self._next_track_id = 1
        self._tracks: List[Track] = []

    def _new_track(self, det: Detection) -> Track:
        t = Track(
            track_id=self._next_track_id,
            bbox_xyxy=np.asarray(det.bbox_xyxy, dtype=np.float32),
            score=float(det.score),
            prompt=det.prompt,
            hits=1,
            age=1,
            misses=0,
            vx=0.0,
            vy=0.0,
            predicted_frames=0,
            updated_this_frame=True,
            last_mask=det.mask,
        )
        self._next_track_id += 1
        return t

    def _match_with_hungarian(
        self,
        tracks: Sequence[Track],
        track_boxes: Sequence[np.ndarray],
        detections: Sequence[Detection],
    ) -> List[Tuple[int, int]]:
        if len(track_boxes) == 0 or len(detections) == 0:
            return []

        invalid_cost = 1e6
        costs = np.full((len(track_boxes), len(detections)), invalid_cost, dtype=np.float64)

        for ti, tbox in enumerate(track_boxes):
            tcx, tcy = _bbox_center(tbox)
            tw = max(1.0, float(tbox[2] - tbox[0]))
            th = max(1.0, float(tbox[3] - tbox[1]))
            tdiag = max(1.0, float(np.hypot(tw, th)))
            track_prompt = str(getattr(tracks[ti], "prompt", "")).strip().lower()
            for di, det in enumerate(detections):
                det_box = np.asarray(det.bbox_xyxy, dtype=np.float32)
                iou = _iou_xyxy(tbox, det_box)
                dcx, dcy = _bbox_center(det_box)
                center_dist = float(np.hypot(dcx - tcx, dcy - tcy))
                center_dist_ratio = center_dist / tdiag
                if iou < self.iou_match_threshold and center_dist_ratio > self.max_center_distance_ratio:
                    continue

                det_prompt = str(det.prompt).strip().lower()
                prompt_score = (
                    self.prompt_match_bonus
                    if track_prompt and det_prompt and track_prompt == det_prompt
                    else -self.prompt_mismatch_penalty
                )
                match_score = iou - (self.center_distance_weight * center_dist_ratio) + prompt_score
                if match_score < self.min_match_score and iou < self.iou_match_threshold:
                    continue
                quality = float(np.clip(match_score, -1.0, 1.0))
                costs[ti, di] = 1.0 - quality

        assignment = _hungarian_assign(costs)
        matched: List[Tuple[int, int]] = []
        for ti, di in assignment:
            if ti < 0 or di < 0 or ti >= costs.shape[0] or di >= costs.shape[1]:
                continue
            if costs[ti, di] >= invalid_cost:
                continue
            matched.append((ti, di))
        return matched

    def update(
        self,
        detections: Sequence[Detection],
        detection_performed: bool = True,
    ) -> List[Track]:
        predicted_boxes: List[np.ndarray] = []
        for t in self._tracks:
            t.updated_this_frame = False
            t.age += 1
            t.last_mask = None
            t.bbox_xyxy = t.predict_bbox()
            t.predicted_frames += 1
            predicted_boxes.append(t.bbox_xyxy.copy())

        if not detection_performed:
            for tr in self._tracks:
                tr.vx *= float(np.clip(self.prediction_velocity_decay, 0.0, 1.0))
                tr.vy *= float(np.clip(self.prediction_velocity_decay, 0.0, 1.0))
                tr.score *= float(np.clip(self.prediction_score_decay, 0.0, 1.0))
            return [t for t in self._tracks if t.hits >= self.min_hits]

        matches = self._match_with_hungarian(self._tracks, predicted_boxes, detections)
        matched_track_indices = {ti for ti, _ in matches}
        matched_det_indices = {di for _, di in matches}

        for ti, di in matches:
            track = self._tracks[ti]
            det = detections[di]
            det_box = np.asarray(det.bbox_xyxy, dtype=np.float32)
            pred_box = track.bbox_xyxy.copy()
            px, py = _bbox_center(pred_box)
            dx, dy = _bbox_center(det_box)
            measured_vx = dx - px
            measured_vy = dy - py
            tw = max(1.0, float(pred_box[2] - pred_box[0]))
            th = max(1.0, float(pred_box[3] - pred_box[1]))
            tdiag = max(1.0, float(np.hypot(tw, th)))
            max_speed = max(0.0, self.max_velocity_ratio) * tdiag
            speed = float(np.hypot(measured_vx, measured_vy))
            if max_speed > 0.0 and speed > max_speed and speed > 1e-6:
                scale = max_speed / speed
                measured_vx *= scale
                measured_vy *= scale
            v_alpha = float(np.clip(self.velocity_smooth_alpha, 0.0, 1.0))
            track.vx = (v_alpha * measured_vx) + ((1.0 - v_alpha) * track.vx)
            track.vy = (v_alpha * measured_vy) + ((1.0 - v_alpha) * track.vy)
            b_alpha = float(np.clip(self.bbox_smooth_alpha, 0.0, 1.0))
            track.bbox_xyxy = ((b_alpha * det_box) + ((1.0 - b_alpha) * pred_box)).astype(np.float32)
            track.score = float(det.score)
            track.prompt = det.prompt
            track.hits += 1
            track.misses = 0
            track.predicted_frames = 0
            track.updated_this_frame = True
            track.last_mask = det.mask

        new_tracks: List[Track] = []
        for idx, tr in enumerate(self._tracks):
            if idx not in matched_track_indices:
                tr.misses += 1
                tr.vx *= float(np.clip(self.velocity_decay, 0.0, 1.0))
                tr.vy *= float(np.clip(self.velocity_decay, 0.0, 1.0))
                tr.score *= float(np.clip(self.prediction_score_decay, 0.0, 1.0))
            if tr.misses <= self.max_missing_frames:
                new_tracks.append(tr)
        self._tracks = new_tracks

        for di, det in enumerate(detections):
            if di in matched_det_indices:
                continue
            self._tracks.append(self._new_track(det))

        return [t for t in self._tracks if t.hits >= self.min_hits]


class SupervisionByteTracker(BaseTracker):
    """
    ByteTrack backend via supervision package.
    - Keyframe detections: ByteTrack assigns stable IDs
    - Non-keyframe detections(empty): lightweight linear prediction maintains live preview
    """

    def __init__(
        self,
        frame_rate: float,
        iou_match_threshold: float,
        max_missing_frames: int,
        min_hits: int,
        bytetrack_cfg: Optional[Dict] = None,
    ) -> None:
        self.frame_rate = float(frame_rate)
        self.iou_match_threshold = float(iou_match_threshold)
        self.max_missing_frames = int(max_missing_frames)
        self.min_hits = int(min_hits)
        self.bytetrack_cfg = bytetrack_cfg or {}
        self._enable_unmatched_spawn = bool(self.bytetrack_cfg.get("enable_unmatched_spawn", True))
        self._spawn_min_score = float(self.bytetrack_cfg.get("spawn_min_score", 0.2))
        self._spawn_iou_threshold = float(self.bytetrack_cfg.get("spawn_iou_threshold", 0.45))
        self._det_match_iou = float(self.bytetrack_cfg.get("det_match_iou_threshold", 0.2))
        self._prediction_velocity_decay = float(self.bytetrack_cfg.get("prediction_velocity_decay", 0.82))
        self._prediction_score_decay = float(self.bytetrack_cfg.get("prediction_score_decay", 0.9))

        try:
            import supervision as sv
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ByteTrack backend requires `supervision` package. "
                "Install with: pip install supervision"
            ) from exc

        self._sv = sv
        self._tracker = self._build_bytetrack()
        self._tracks: Dict[int, Track] = {}
        self._fallback_next_id = 1_000_000

    def _build_bytetrack(self):
        import inspect

        sig = inspect.signature(self._sv.ByteTrack)
        params = sig.parameters
        cfg = dict(self.bytetrack_cfg)

        kwargs = {}
        if "frame_rate" in params:
            kwargs["frame_rate"] = self.frame_rate

        aliases = [
            ("track_activation_threshold", "track_activation_threshold"),
            ("track_activation_threshold", "track_thresh"),
            ("lost_track_buffer", "lost_track_buffer"),
            ("lost_track_buffer", "track_buffer"),
            ("minimum_matching_threshold", "minimum_matching_threshold"),
            ("minimum_matching_threshold", "match_thresh"),
            ("minimum_consecutive_frames", "minimum_consecutive_frames"),
        ]
        for src, dst in aliases:
            if src in cfg and dst in params:
                kwargs[dst] = cfg[src]

        return self._sv.ByteTrack(**kwargs)

    def _detections_to_sv(self, detections: Sequence[Detection]):
        if len(detections) == 0:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confidence = np.empty((0,), dtype=np.float32)
            class_id = np.empty((0,), dtype=np.int32)
            data = {"prompt": np.asarray([], dtype=object)}
            return self._sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id, data=data)

        xyxy = np.asarray([d.bbox_xyxy for d in detections], dtype=np.float32)
        confidence = np.asarray([d.score for d in detections], dtype=np.float32)
        class_id = np.zeros((len(detections),), dtype=np.int32)
        prompts = np.asarray([d.prompt for d in detections], dtype=object)
        data = {"prompt": prompts}
        return self._sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id, data=data)

    def _resolve_prompt(self, bbox_xyxy: np.ndarray, detections: Sequence[Detection], default: str) -> str:
        best_iou = 0.0
        best_prompt = default
        for d in detections:
            iou = _iou_xyxy(bbox_xyxy, np.asarray(d.bbox_xyxy, dtype=np.float32))
            if iou > best_iou:
                best_iou = iou
                best_prompt = d.prompt
        return best_prompt

    def _upsert_track(
        self,
        track_id: int,
        bbox_xyxy: np.ndarray,
        score: float,
        prompt: str,
        updated_this_frame: bool,
        mask: Optional[np.ndarray],
    ) -> None:
        if track_id in self._tracks:
            tr = self._tracks[track_id]
            px, py = _bbox_center(tr.bbox_xyxy)
            dx, dy = _bbox_center(bbox_xyxy)
            tr.vx = dx - px
            tr.vy = dy - py
            tr.bbox_xyxy = bbox_xyxy
            tr.score = float(score)
            tr.prompt = prompt
            tr.hits += 1
            tr.misses = 0 if updated_this_frame else tr.misses
            tr.predicted_frames = 0 if updated_this_frame else (tr.predicted_frames + 1)
            tr.updated_this_frame = updated_this_frame
            tr.last_mask = mask if updated_this_frame else None
            tr.age += 1
        else:
            self._tracks[track_id] = Track(
                track_id=track_id,
                bbox_xyxy=bbox_xyxy.copy(),
                score=float(score),
                prompt=prompt,
                hits=1,
                age=1,
                misses=0,
                vx=0.0,
                vy=0.0,
                predicted_frames=0 if updated_this_frame else 1,
                updated_this_frame=updated_this_frame,
                last_mask=mask,
            )

    def _predict_unmatched(
        self,
        matched_ids: set[int],
        increment_misses: bool = True,
        apply_prediction_decay: bool = True,
    ) -> None:
        to_delete = []
        for tid, tr in self._tracks.items():
            if tid in matched_ids:
                continue
            tr.updated_this_frame = False
            tr.last_mask = None
            tr.bbox_xyxy = tr.predict_bbox()
            tr.predicted_frames += 1
            if increment_misses:
                tr.misses += 1
            if apply_prediction_decay:
                tr.vx *= float(np.clip(self._prediction_velocity_decay, 0.0, 1.0))
                tr.vy *= float(np.clip(self._prediction_velocity_decay, 0.0, 1.0))
                tr.score *= float(np.clip(self._prediction_score_decay, 0.0, 1.0))
            tr.age += 1
            if increment_misses and tr.misses > self.max_missing_frames:
                to_delete.append(tid)
        for tid in to_delete:
            del self._tracks[tid]

    def update(
        self,
        detections: Sequence[Detection],
        detection_performed: bool = True,
    ) -> List[Track]:
        if not detection_performed:
            self._predict_unmatched(set(), increment_misses=False, apply_prediction_decay=True)
            return [t for t in self._tracks.values() if t.hits >= self.min_hits]

        sv_dets = self._detections_to_sv(detections)
        tracked = self._tracker.update_with_detections(sv_dets)

        matched_ids: set[int] = set()
        matched_det_indices: set[int] = set()
        if len(tracked) > 0:
            tracked_xyxy = np.asarray(tracked.xyxy, dtype=np.float32)
            tracked_conf = np.asarray(tracked.confidence, dtype=np.float32)
            tracked_ids = np.asarray(tracked.tracker_id)

            prompts_data = None
            if hasattr(tracked, "data") and isinstance(tracked.data, dict):
                prompts_data = tracked.data.get("prompt")

            for i in range(len(tracked_xyxy)):
                tid_raw = tracked_ids[i]
                if tid_raw is None:
                    tid = self._fallback_next_id
                    self._fallback_next_id += 1
                else:
                    tid = int(tid_raw)
                matched_ids.add(tid)

                prompt = "object."
                if prompts_data is not None and i < len(prompts_data):
                    prompt = str(prompts_data[i])
                else:
                    prompt = self._resolve_prompt(tracked_xyxy[i], detections, prompt)

                mask = None
                best_det_idx = -1
                if len(detections) > 0:
                    best_iou = 0.0
                    for di, d in enumerate(detections):
                        iou = _iou_xyxy(tracked_xyxy[i], np.asarray(d.bbox_xyxy, dtype=np.float32))
                        if iou > best_iou:
                            best_iou = iou
                            mask = d.mask
                            best_det_idx = di
                    if best_det_idx >= 0 and best_iou >= self._det_match_iou:
                        matched_det_indices.add(best_det_idx)

                self._upsert_track(
                    track_id=tid,
                    bbox_xyxy=tracked_xyxy[i],
                    score=float(tracked_conf[i]),
                    prompt=prompt,
                    updated_this_frame=True,
                    mask=mask,
                )

        self._predict_unmatched(matched_ids, increment_misses=True, apply_prediction_decay=True)

        # ByteTrack can delay/ignore low-confidence new entrants.
        # Spawn fallback tracks directly from unmatched SAM detections to improve new-object recall.
        if self._enable_unmatched_spawn and len(detections) > 0:
            current_boxes = [np.asarray(t.bbox_xyxy, dtype=np.float32) for t in self._tracks.values()]
            for di, det in enumerate(detections):
                if di in matched_det_indices:
                    continue
                if float(det.score) < self._spawn_min_score:
                    continue
                det_box = np.asarray(det.bbox_xyxy, dtype=np.float32)
                overlaps_existing = False
                for tbox in current_boxes:
                    if _iou_xyxy(det_box, tbox) >= self._spawn_iou_threshold:
                        overlaps_existing = True
                        break
                if overlaps_existing:
                    continue
                tid = self._fallback_next_id
                self._fallback_next_id += 1
                self._upsert_track(
                    track_id=tid,
                    bbox_xyxy=det_box,
                    score=float(det.score),
                    prompt=str(det.prompt),
                    updated_this_frame=True,
                    mask=det.mask,
                )
                current_boxes.append(det_box)

        return [t for t in self._tracks.values() if t.hits >= self.min_hits]


def build_tracker(
    backend: str,
    frame_rate: float,
    iou_match_threshold: float,
    max_missing_frames: int,
    min_hits: int,
    bytetrack_cfg: Optional[Dict] = None,
    iou_fallback_cfg: Optional[Dict] = None,
) -> BaseTracker:
    b = backend.strip().lower()
    if b == "bytetrack":
        return SupervisionByteTracker(
            frame_rate=frame_rate,
            iou_match_threshold=iou_match_threshold,
            max_missing_frames=max_missing_frames,
            min_hits=min_hits,
            bytetrack_cfg=bytetrack_cfg,
        )
    return IoUByteLikeTracker(
        iou_match_threshold=iou_match_threshold,
        max_missing_frames=max_missing_frames,
        min_hits=min_hits,
        fallback_cfg=iou_fallback_cfg,
    )
