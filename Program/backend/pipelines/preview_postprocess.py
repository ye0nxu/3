from __future__ import annotations

from typing import Any, Callable, Sequence

from core.dataset import FrameAnnotation

from core.models import PreviewThumbnail


def preview_category_summary(preview_items: Sequence[PreviewThumbnail]) -> dict[str, int]:
    keep_count = sum(1 for item in preview_items if str(item.category).strip().lower() == "keep")
    hold_count = sum(1 for item in preview_items if str(item.category).strip().lower() == "hold")
    drop_count = sum(1 for item in preview_items if str(item.category).strip().lower() == "drop")
    return {"keep": int(keep_count), "hold": int(hold_count), "drop": int(drop_count)}


def _clone_preview_item(item: PreviewThumbnail) -> PreviewThumbnail:
    boxes: list[Any] = []
    for box in list(item.boxes or []):
        if isinstance(box, dict):
            boxes.append(dict(box))
        else:
            boxes.append(box)
    return PreviewThumbnail(
        frame_index=int(item.frame_index),
        image=None,
        boxes=boxes,
        category=str(item.category).strip().lower() or "hold",
        item_id=str(item.item_id).strip(),
        image_path=str(item.image_path).strip() if item.image_path else None,
        thumb_path=str(item.thumb_path).strip() if item.thumb_path else None,
        manifest_path=str(item.manifest_path).strip() if item.manifest_path else None,
    )


def _sort_preview_items_by_track_and_frame(preview_items: Sequence[PreviewThumbnail]) -> list[PreviewThumbnail]:
    cloned = [_clone_preview_item(item) for item in preview_items]

    def _item_sort_key(item: PreviewThumbnail) -> tuple[int, int, str]:
        track_key = 2_147_483_647
        if item.boxes and isinstance(item.boxes[0], dict):
            raw_track = item.boxes[0].get("track_id")
            if raw_track is not None:
                try:
                    track_key = int(raw_track)
                except Exception:
                    track_key = 2_147_483_647
        return (track_key, int(item.frame_index), str(item.item_id))

    cloned.sort(key=_item_sort_key)
    return cloned


def _dedupe_keep_preview_items(
    preview_items: Sequence[PreviewThumbnail],
    log_callback: Callable[[str], None] | None = None,
) -> list[PreviewThumbnail]:
    deduped: list[PreviewThumbnail] = []
    key_to_index: dict[tuple[int, int], int] = {}
    last_keep_by_track: dict[int, tuple[int, tuple[float, float, float, float]]] = {}
    dropped = 0

    def _first_box_dict(item: PreviewThumbnail) -> dict[str, Any] | None:
        if not item.boxes:
            return None
        first = item.boxes[0]
        if not isinstance(first, dict):
            return None
        return first

    def _box_xyxy(box: dict[str, Any]) -> tuple[float, float, float, float] | None:
        try:
            x1 = float(box.get("x1", 0.0))
            y1 = float(box.get("y1", 0.0))
            x2 = float(box.get("x2", 0.0))
            y2 = float(box.get("y2", 0.0))
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        aa = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        bb = max(0.0, (bx2 - bx1) * (by2 - by1))
        denom = aa + bb - inter
        if denom <= 0.0:
            return 0.0
        return inter / denom

    for item in preview_items:
        category = str(item.category).strip().lower()
        if category != "keep":
            deduped.append(item)
            continue

        box = _first_box_dict(item)
        if box is None:
            deduped.append(item)
            continue
        raw_track = box.get("track_id")
        if raw_track is None:
            deduped.append(item)
            continue
        try:
            track_id = int(raw_track)
        except Exception:
            deduped.append(item)
            continue

        frame_idx = int(item.frame_index)
        xyxy = _box_xyxy(box)
        if xyxy is None:
            deduped.append(item)
            continue

        key = (track_id, frame_idx)
        if key in key_to_index:
            prev_idx = key_to_index[key]
            prev_item = deduped[prev_idx]
            prev_box = _first_box_dict(prev_item)
            try:
                prev_score = float(prev_box.get("score", 0.0)) if prev_box is not None else 0.0
            except Exception:
                prev_score = 0.0
            try:
                cur_score = float(box.get("score", 0.0))
            except Exception:
                cur_score = 0.0
            if cur_score > prev_score:
                deduped[prev_idx] = item
            dropped += 1
            continue

        prev_track = last_keep_by_track.get(track_id)
        if prev_track is not None:
            prev_frame_idx, prev_xyxy = prev_track
            if abs(frame_idx - prev_frame_idx) <= 1 and _iou_xyxy(prev_xyxy, xyxy) >= 0.995:
                dropped += 1
                continue

        key_to_index[key] = len(deduped)
        last_keep_by_track[track_id] = (frame_idx, xyxy)
        deduped.append(item)

    if dropped > 0 and log_callback is not None:
        log_callback(f"중복 제거(keep): {dropped}개")
    return deduped


def _renumber_preview_items_by_track(
    preview_items: Sequence[PreviewThumbnail],
    annotations: Sequence[FrameAnnotation] | None = None,
) -> None:
    remap: dict[str, str] = {}
    track_seq: dict[int, int] = {}
    unknown_seq = 0

    for item in preview_items:
        old_item_id = str(item.item_id).strip()
        track_id: int | None = None
        if item.boxes and isinstance(item.boxes[0], dict):
            raw_track = item.boxes[0].get("track_id")
            if raw_track is not None:
                try:
                    track_id = int(raw_track)
                except Exception:
                    track_id = None

        if track_id is None:
            unknown_seq += 1
            new_item_id = f"id_unknown_{unknown_seq:03d}"
        else:
            next_seq = int(track_seq.get(track_id, 0)) + 1
            track_seq[track_id] = next_seq
            new_item_id = f"id{track_id}_{next_seq:03d}"

        item.item_id = new_item_id
        for box in item.boxes:
            if isinstance(box, dict):
                box["preview_item_id"] = new_item_id
        if old_item_id:
            remap[old_item_id] = new_item_id

    if not remap or not annotations:
        return

    for ann in annotations:
        for box in ann.boxes:
            if not isinstance(box, dict):
                continue
            prev = str(box.get("preview_item_id", "")).strip()
            if not prev:
                continue
            mapped = remap.get(prev)
            if mapped is not None:
                box["preview_item_id"] = mapped


def _sync_annotation_status_from_preview_items(
    annotations: Sequence[FrameAnnotation] | None,
    preview_items: Sequence[PreviewThumbnail],
) -> None:
    if not annotations:
        return
    status_by_item_id: dict[str, str] = {}
    for item in preview_items:
        item_id = str(item.item_id).strip()
        category = str(item.category).strip().lower()
        if not item_id or category not in {"keep", "hold", "drop"}:
            continue
        status_by_item_id[item_id] = category

    if not status_by_item_id:
        return

    for ann in annotations:
        for box in ann.boxes:
            if not isinstance(box, dict):
                continue
            item_id = str(box.get("preview_item_id", "")).strip()
            if not item_id:
                continue
            updated = status_by_item_id.get(item_id)
            if updated is not None:
                box["status"] = updated


def postprocess_preview_items(
    preview_items: Sequence[PreviewThumbnail],
    *,
    annotations: Sequence[FrameAnnotation] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> list[PreviewThumbnail]:
    if not preview_items:
        return []
    sorted_items = _sort_preview_items_by_track_and_frame(preview_items)
    processed_items = _dedupe_keep_preview_items(sorted_items, log_callback=log_callback)
    _renumber_preview_items_by_track(processed_items, annotations)
    _sync_annotation_status_from_preview_items(annotations, processed_items)
    if log_callback is not None:
        summary = preview_category_summary(processed_items)
        log_callback(
            f"후처리 결과: keep={summary['keep']}, hold={summary['hold']}, drop={summary['drop']}"
        )
    return processed_items
