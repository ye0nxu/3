from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .hash import hash_prefix, hamming_distance64


@dataclass(slots=True)
class GlobalHashEntry:
    entry_id: str
    sample_id: str
    track_id: str
    frame_idx: int
    hash64: int
    thumb_gray: np.ndarray
    meta: dict[str, Any]


class GlobalDedupIndex:
    """Bucketed hash index with LRU + TTL eviction."""

    def __init__(
        self,
        prefix_bits: int = 16,
        ttl_frames: int = 3000,
        max_entries: int = 20000,
        compare_neighbor_buckets: bool = True,
    ) -> None:
        self.prefix_bits = max(1, min(63, int(prefix_bits)))
        self.ttl_frames = max(1, int(ttl_frames))
        self.max_entries = max(100, int(max_entries))
        self.compare_neighbor_buckets = bool(compare_neighbor_buckets)
        self._entries: OrderedDict[str, GlobalHashEntry] = OrderedDict()
        self._bucket_to_ids: dict[int, set[str]] = defaultdict(set)
        self._next_serial = 0

    def add(self, entry: GlobalHashEntry, current_frame_idx: int) -> None:
        self.evict(current_frame_idx)
        self._entries[entry.entry_id] = entry
        self._entries.move_to_end(entry.entry_id, last=True)
        bucket = hash_prefix(entry.hash64, self.prefix_bits)
        self._bucket_to_ids[bucket].add(entry.entry_id)
        self._evict_by_size()

    def create_and_add(
        self,
        sample_id: str,
        track_id: str,
        frame_idx: int,
        hash64: int,
        thumb_gray: np.ndarray,
        meta: dict[str, Any],
        current_frame_idx: int,
    ) -> GlobalHashEntry:
        self._next_serial += 1
        entry = GlobalHashEntry(
            entry_id=f"g_{self._next_serial:09d}",
            sample_id=str(sample_id),
            track_id=str(track_id),
            frame_idx=int(frame_idx),
            hash64=int(hash64),
            thumb_gray=thumb_gray,
            meta=dict(meta),
        )
        self.add(entry, current_frame_idx=current_frame_idx)
        return entry

    def find_near(
        self,
        hash64: int,
        dist_thr: int,
        current_frame_idx: int,
    ) -> tuple[GlobalHashEntry | None, int]:
        self.evict(current_frame_idx)
        best_entry: GlobalHashEntry | None = None
        best_dist = 10**9
        for entry in self._iter_bucket_candidates(hash64):
            dist = hamming_distance64(hash64, entry.hash64)
            if dist < best_dist:
                best_dist = dist
                best_entry = entry
        if best_entry is None or best_dist > int(dist_thr):
            return None, best_dist if best_entry is not None else -1
        self._entries.move_to_end(best_entry.entry_id, last=True)
        return best_entry, best_dist

    def evict(self, current_frame_idx: int) -> None:
        cutoff = int(current_frame_idx) - self.ttl_frames
        stale_ids = [
            entry_id
            for entry_id, entry in self._entries.items()
            if int(entry.frame_idx) < cutoff
        ]
        for entry_id in stale_ids:
            self._remove(entry_id)

    def _evict_by_size(self) -> None:
        while len(self._entries) > self.max_entries:
            oldest_id = next(iter(self._entries))
            self._remove(oldest_id)

    def _remove(self, entry_id: str) -> None:
        entry = self._entries.pop(entry_id, None)
        if entry is None:
            return
        bucket = hash_prefix(entry.hash64, self.prefix_bits)
        ids = self._bucket_to_ids.get(bucket)
        if ids is not None:
            ids.discard(entry_id)
            if not ids:
                self._bucket_to_ids.pop(bucket, None)

    def _iter_bucket_candidates(self, hash64: int) -> Iterable[GlobalHashEntry]:
        bucket = hash_prefix(hash64, self.prefix_bits)
        bucket_ids = set(self._bucket_to_ids.get(bucket, set()))
        if self.compare_neighbor_buckets:
            bucket_ids.update(self._bucket_to_ids.get(bucket - 1, set()))
            bucket_ids.update(self._bucket_to_ids.get(bucket + 1, set()))
        for entry_id in bucket_ids:
            entry = self._entries.get(entry_id)
            if entry is not None:
                yield entry

