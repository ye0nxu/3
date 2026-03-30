from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

from .config import FilterConfig
from .global_index import GlobalDedupIndex
from .hash import dhash64, hamming_distance64, make_thumb_gray
from .quality import QualityMetrics, evaluate_quality
from .track_state import TrackKeepState, TrackStateStore


class FilterReason(str, Enum):
    ## =====================================
    ## 함수 기능 : 필터 판정 사유를 나타내는 열거형 (str 상속으로 기존 문자열 비교 호환)
    ## =====================================
    PASS = "PASS"
    DROP_BLUR = "DROP_BLUR"
    DROP_OVEREXPOSED = "DROP_OVEREXPOSED"
    DROP_UNDEREXPOSED = "DROP_UNDEREXPOSED"
    DROP_DUP_TRACK = "DROP_DUP_TRACK"
    DROP_DUP_GLOBAL = "DROP_DUP_GLOBAL"
    DROP_INVALID_SAMPLE = "DROP_INVALID_SAMPLE"


@dataclass(slots=True)
class SampleCandidate:
    frame_idx: int
    timestamp_ms: float
    track_id: str | int
    bbox: tuple[float, float, float, float] | None
    crop_image: np.ndarray
    sample_id: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FilterResult:
    passed: bool
    reason: str
    metrics: dict[str, Any]


class SampleFilterEngine:
    """Streaming quality + dedup engine. State updates happen only on final keep."""

    def __init__(self, config: FilterConfig | None = None) -> None:
        self.config = config or FilterConfig()
        self.track_store = TrackStateStore(ttl_frames=self.config.track_ttl_frames)
        self.global_index = GlobalDedupIndex(
            prefix_bits=self.config.global_bucket_prefix_bits,
            ttl_frames=self.config.global_ttl_frames,
            max_entries=self.config.global_max_entries,
            compare_neighbor_buckets=self.config.compare_neighbor_buckets,
        )

    def evaluate(self, sample: SampleCandidate) -> FilterResult:
        if sample.crop_image is None or sample.crop_image.size == 0:
            return FilterResult(False, FilterReason.DROP_INVALID_SAMPLE, {"sample_id": sample.sample_id})

        try:
            image = sample.crop_image
            if image.ndim != 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        except Exception:
            return FilterResult(False, FilterReason.DROP_INVALID_SAMPLE, {"sample_id": sample.sample_id})

        quality_reason, quality_metrics = evaluate_quality(image, self.config)
        hash64 = dhash64(image)
        thumb = make_thumb_gray(image, self.config.thumb_size)
        metrics = self._base_metrics(sample, quality_metrics, hash64)

        if quality_reason != FilterReason.PASS:
            return FilterResult(False, quality_reason, metrics)

        track_key = str(sample.track_id)
        track_state = self.track_store.get(track_key, sample.frame_idx)
        if track_state is not None:
            frame_gap = int(sample.frame_idx) - int(track_state.frame_idx)
            track_dist = hamming_distance64(hash64, track_state.hash64)
            metrics["track_frame_gap"] = frame_gap
            metrics["track_hash_dist"] = track_dist
            if frame_gap <= int(self.config.frame_gap_thr):
                if track_dist <= int(self.config.hash_dist_thr):
                    return FilterResult(False, FilterReason.DROP_DUP_TRACK, metrics)
                if self._is_refine_duplicate(track_dist, thumb, track_state.thumb_gray):
                    return FilterResult(False, FilterReason.DROP_DUP_TRACK, metrics)

        global_match, global_dist = self.global_index.find_near(
            hash64=hash64,
            dist_thr=self.config.hash_dist_thr,
            current_frame_idx=sample.frame_idx,
        )
        metrics["global_hash_dist"] = global_dist
        if global_match is not None:
            metrics["global_match_sample_id"] = global_match.sample_id
            return FilterResult(False, FilterReason.DROP_DUP_GLOBAL, metrics)

        return FilterResult(True, FilterReason.PASS, metrics)

    def on_final_keep(self, sample: SampleCandidate) -> None:
        """Call this only when sample becomes FINAL KEEP."""
        if sample.crop_image is None or sample.crop_image.size == 0:
            return
        image = sample.crop_image
        if image.ndim != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        hash64 = dhash64(image)
        thumb = make_thumb_gray(image, self.config.thumb_size)

        track_key = str(sample.track_id)
        track_state = TrackKeepState(
            track_id=track_key,
            sample_id=str(sample.sample_id),
            frame_idx=int(sample.frame_idx),
            hash64=int(hash64),
            thumb_gray=thumb,
            meta=dict(sample.meta),
        )
        self.track_store.update(track_state)
        self.track_store.evict(sample.frame_idx)

        self.global_index.create_and_add(
            sample_id=str(sample.sample_id),
            track_id=track_key,
            frame_idx=int(sample.frame_idx),
            hash64=int(hash64),
            thumb_gray=thumb,
            meta=dict(sample.meta),
            current_frame_idx=int(sample.frame_idx),
        )

    def _base_metrics(self, sample: SampleCandidate, q: QualityMetrics, hash64: int) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "frame_idx": int(sample.frame_idx),
            "track_id": str(sample.track_id),
            "blur_score": float(q.blur_score),
            "white_ratio": float(q.white_ratio),
            "black_ratio": float(q.black_ratio),
            "hash64": int(hash64),
        }

    def _is_refine_duplicate(self, hash_dist: int, thumb_a: np.ndarray, thumb_b: np.ndarray) -> bool:
        low = int(self.config.hash_dist_thr) + 1
        high = int(self.config.hash_dist_thr) + int(self.config.refine_band)
        if hash_dist < low or hash_dist > high:
            return False
        score = self._ncc_similarity(thumb_a, thumb_b)
        return score >= float(self.config.refine_ncc_thr)

    @staticmethod
    def _ncc_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a32 = a.astype(np.float32)
        b32 = b.astype(np.float32)
        a_center = a32 - float(np.mean(a32))
        b_center = b32 - float(np.mean(b32))
        denom = float(np.linalg.norm(a_center) * np.linalg.norm(b_center))
        if denom <= 1e-6:
            return 1.0
        return float(np.sum(a_center * b_center) / denom)

