from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class TrackKeepState:
    track_id: str
    sample_id: str
    frame_idx: int
    hash64: int
    thumb_gray: np.ndarray
    meta: dict[str, Any]


class TrackStateStore:
    """Per-track last FINAL KEEP state with TTL eviction."""

    def __init__(self, ttl_frames: int = 3000) -> None:
        self.ttl_frames = max(1, int(ttl_frames))
        self._states: dict[str, TrackKeepState] = {}

    def get(self, track_id: str, current_frame_idx: int) -> TrackKeepState | None:
        self.evict(current_frame_idx)
        return self._states.get(str(track_id))

    def update(self, state: TrackKeepState) -> None:
        self._states[str(state.track_id)] = state

    def evict(self, current_frame_idx: int) -> None:
        cutoff = int(current_frame_idx) - self.ttl_frames
        stale_keys = [
            key
            for key, value in self._states.items()
            if int(value.frame_idx) < cutoff
        ]
        for key in stale_keys:
            self._states.pop(key, None)

