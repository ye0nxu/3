from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from .detector import Detection, Sam3TextDetector
from .tracker import build_tracker


def _resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_prompts(raw_prompts: Optional[str], fallback: Sequence[str]) -> List[str]:
    if raw_prompts is None:
        return [p.strip() for p in fallback if p.strip()]
    if not raw_prompts.strip():
        return [p.strip() for p in fallback if p.strip()]
    return [p.strip() for p in raw_prompts.split(",") if p.strip()]


def _track_color(track_id: int) -> tuple[int, int, int]:
    return (
        int((37 * track_id) % 255),
        int((17 * track_id) % 255),
        int((97 * track_id) % 255),
    )


def _sanitize_bbox_xyxy(
    bbox_xyxy: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> Optional[np.ndarray]:
    box = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
    if box.size != 4:
        return None
    if not np.isfinite(box).all():
        return None
    box[0] = float(np.clip(box[0], 0, max(0, frame_w - 1)))
    box[1] = float(np.clip(box[1], 0, max(0, frame_h - 1)))
    box[2] = float(np.clip(box[2], 0, max(0, frame_w - 1)))
    box[3] = float(np.clip(box[3], 0, max(0, frame_h - 1)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _point_in_box(x: float, y: float, box: np.ndarray) -> bool:
    return float(box[0]) <= x <= float(box[2]) and float(box[1]) <= y <= float(box[3])


def _filter_detections_for_new_objects(
    detections: Sequence[Detection],
    active_tracks: Sequence,
    frame_w: int,
    frame_h: int,
    iou_threshold: float,
) -> tuple[List[Detection], int]:
    if len(detections) == 0:
        return [], 0

    active_boxes: List[np.ndarray] = []
    for tr in active_tracks:
        if not hasattr(tr, "bbox_xyxy"):
            continue
        box = _sanitize_bbox_xyxy(tr.bbox_xyxy, frame_w, frame_h)
        if box is None:
            continue
        active_boxes.append(box)

    if len(active_boxes) == 0:
        return list(detections), 0

    kept: List[Detection] = []
    dropped = 0
    for det in detections:
        det_box = _sanitize_bbox_xyxy(np.asarray(det.bbox_xyxy, dtype=np.float32), frame_w, frame_h)
        if det_box is None:
            continue
        cx = float((det_box[0] + det_box[2]) * 0.5)
        cy = float((det_box[1] + det_box[3]) * 0.5)
        is_existing = False
        for active_box in active_boxes:
            iou = _iou_xyxy(det_box, active_box)
            # Suppress only when it is strongly overlapping and center is inside
            # an existing active track, to reduce accidental removal of nearby objects.
            if iou >= iou_threshold and _point_in_box(cx, cy, active_box):
                is_existing = True
                break
        if is_existing:
            dropped += 1
            continue
        kept.append(det)
    return kept, dropped


def _draw_tracks(
    frame_bgr: np.ndarray,
    tracks: Iterable,
    draw_tracker_predictions: bool,
) -> np.ndarray:
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    for tr in tracks:
        box = _sanitize_bbox_xyxy(tr.bbox_xyxy, w, h)
        if box is None:
            continue
        x0, y0, x1, y1 = box.astype(int).tolist()
        color = _track_color(tr.track_id)
        source = "sam3" if tr.updated_this_frame else "track"
        if source == "track" and not draw_tracker_predictions:
            continue
        thickness = 2 if source == "sam3" else 1
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, thickness)
        prompt_label = str(getattr(tr, "prompt", "")).strip() or "object"
        label = f"id{tr.track_id} {prompt_label} {tr.score:.2f}"
        cv2.putText(
            vis,
            label,
            (x0, max(18, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def _resize_for_ui_preview(frame_bgr: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    if max_w <= 0 or max_h <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return frame_bgr
    scale = min(float(max_w) / float(w), float(max_h) / float(h), 1.0)
    if scale >= 0.999:
        return frame_bgr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _refine_tracks_with_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    tracks: Iterable,
    frame_w: int,
    frame_h: int,
    max_corners: int,
    quality_level: float,
    min_distance: float,
    min_points: int,
    max_track_misses: int,
) -> int:
    refined_count = 0
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
    )

    for tr in tracks:
        if tr.updated_this_frame:
            continue
        if int(getattr(tr, "misses", 0)) > int(max_track_misses):
            continue
        box = _sanitize_bbox_xyxy(tr.bbox_xyxy, frame_w, frame_h)
        if box is None:
            continue
        x0, y0, x1, y1 = box.astype(int).tolist()
        if (x1 - x0) < 8 or (y1 - y0) < 8:
            continue

        roi_prev = prev_gray[y0:y1, x0:x1]
        if roi_prev.size == 0:
            continue

        p0 = cv2.goodFeaturesToTrack(
            roi_prev,
            maxCorners=max(8, int(max_corners)),
            qualityLevel=max(0.001, float(quality_level)),
            minDistance=max(1.0, float(min_distance)),
            blockSize=5,
        )
        if p0 is None or len(p0) < min_points:
            continue

        p0[:, 0, 0] += x0
        p0[:, 0, 1] += y0

        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
        if p1 is None or st is None:
            continue

        good = st.reshape(-1) == 1
        if int(np.count_nonzero(good)) < int(min_points):
            continue

        src = p0.reshape(-1, 2)[good]
        dst = p1.reshape(-1, 2)[good]
        flow = dst - src
        if flow.size == 0:
            continue

        dx = float(np.median(flow[:, 0]))
        dy = float(np.median(flow[:, 1]))
        if not np.isfinite(dx) or not np.isfinite(dy):
            continue
        if abs(dx) > frame_w * 0.2 or abs(dy) > frame_h * 0.2:
            continue

        tr.bbox_xyxy = np.asarray(
            [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy], dtype=np.float32
        )
        # Update motion hint but do not reset misses; detector refresh should own lifecycle.
        tr.vx = dx
        tr.vy = dy
        refined_count += 1

    return refined_count


def run_pipeline(
    config_path: str,
    video_path: str,
    prompts_override: Optional[str] = None,
    run_name: Optional[str] = None,
    max_frames_override: Optional[int] = None,
    keyframe_interval_override: Optional[int] = None,
    show_progress_override: Optional[bool] = None,
    on_preview_frame: Optional[Callable[[np.ndarray, int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict:
    cfg_path = Path(config_path).resolve()
    cfg_base = cfg_path.parent
    cfg = _load_config(cfg_path)

    video = Path(video_path).resolve()
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    run_root = _resolve_path(cfg_base, cfg.get("run_root", "../runs"))
    run_root.mkdir(parents=True, exist_ok=True)
    run_id = run_name or f"sam3_text_track_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts = _parse_prompts(prompts_override, cfg.get("prompts", ["car."]))
    if not prompts:
        raise ValueError("Prompt list is empty.")

    vcfg = cfg.get("video", {})
    keyframe_interval = int(vcfg.get("keyframe_interval", 15))
    max_frames = int(vcfg.get("max_frames", 0))
    force_detect_when_empty = bool(vcfg.get("force_detect_when_empty", False))
    detect_new_only_for_active_ids = bool(vcfg.get("detect_new_only_for_active_ids", True))
    active_track_iou_threshold = float(vcfg.get("active_track_iou_threshold", 0.5))
    active_track_refresh_interval = int(vcfg.get("active_track_refresh_interval", 4))
    if max_frames_override is not None:
        max_frames = int(max_frames_override)
    if keyframe_interval_override is not None:
        keyframe_interval = max(1, int(keyframe_interval_override))
    detect_new_only_effective = detect_new_only_for_active_ids and keyframe_interval > 1

    scfg = cfg.get("sam3", {})
    infer_size = scfg.get("infer_size")
    infer_size_tuple = None
    if isinstance(infer_size, list) and len(infer_size) == 2:
        infer_size_tuple = (int(infer_size[0]), int(infer_size[1]))

    detector = Sam3TextDetector(
        config_base=cfg_base,
        sam3_root=cfg.get("sam3_root", "../sam3"),
        checkpoint=scfg.get("checkpoint", "sam3/checkpoints/sam3.pt"),
        bpe=scfg.get("bpe", "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        device=cfg.get("device", "cuda"),
        confidence_threshold=float(scfg.get("confidence_threshold", 0.45)),
        infer_size=infer_size_tuple,
        nms_iou_threshold=float(cfg.get("detections", {}).get("nms_iou_threshold", 0.5)),
        min_area=float(cfg.get("detections", {}).get("min_area", 120)),
        max_per_prompt=int(cfg.get("detections", {}).get("max_per_prompt", 0)),
        max_total=int(cfg.get("detections", {}).get("max_total", 0)),
        use_amp=bool(scfg.get("use_amp", True)),
    )

    tcfg = cfg.get("tracker", {})
    tracker_backend = str(tcfg.get("backend", "bytetrack"))
    bytetrack_requested = tracker_backend.strip().lower() == "bytetrack"
    bytetrack_available = True
    max_prediction_frames = int(tcfg.get("max_prediction_frames", keyframe_interval + 1))
    max_missing_frames = int(tcfg.get("max_missing_frames", 20))
    prediction_min_score = float(tcfg.get("prediction_min_score", 0.0))
    flow_cfg = tcfg.get("optical_flow", {})
    flow_enabled = bool(flow_cfg.get("enabled", True))
    flow_max_corners = int(flow_cfg.get("max_corners", 28))
    flow_quality_level = float(flow_cfg.get("quality_level", 0.02))
    flow_min_distance = float(flow_cfg.get("min_distance", 4.0))
    flow_min_points = int(flow_cfg.get("min_points", 6))
    flow_max_track_misses = int(flow_cfg.get("max_track_misses", 3))

    ocfg = cfg.get("output", {})
    save_masks = bool(ocfg.get("save_masks", False))
    save_preview_video = bool(ocfg.get("save_preview_video", True))
    draw_tracker_predictions = bool(ocfg.get("draw_tracker_predictions", True))
    preview_fps_cfg = float(ocfg.get("preview_fps", 15))
    live_preview = bool(ocfg.get("live_preview", False))
    live_preview_window = str(ocfg.get("live_preview_window", "SAM3 Text Prompt Tracking Preview"))
    live_preview_wait_ms = int(ocfg.get("live_preview_wait_ms", 1))
    live_preview_stop_on_q = bool(ocfg.get("live_preview_stop_on_q", True))
    show_progress = bool(ocfg.get("show_progress", False))
    ui_emit_interval = max(1, int(ocfg.get("ui_emit_interval", 1)))
    ui_preview_max_size = ocfg.get("ui_preview_max_size")
    ui_preview_max_w = 0
    ui_preview_max_h = 0
    if isinstance(ui_preview_max_size, list) and len(ui_preview_max_size) == 2:
        ui_preview_max_w = max(0, int(ui_preview_max_size[0]))
        ui_preview_max_h = max(0, int(ui_preview_max_size[1]))
    if show_progress_override is not None:
        show_progress = bool(show_progress_override)

    masks_dir = run_dir / "masks"
    if save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    preview_fps = preview_fps_cfg if preview_fps_cfg > 0 else src_fps

    tracker_fallback_reason = None
    try:
        tracker = build_tracker(
            backend=tracker_backend,
            frame_rate=src_fps,
            iou_match_threshold=float(tcfg.get("iou_match_threshold", 0.25)),
            max_missing_frames=max_missing_frames,
            min_hits=int(tcfg.get("min_hits", 1)),
            bytetrack_cfg=tcfg.get("bytetrack", {}),
            iou_fallback_cfg=tcfg.get("iou_fallback", {}),
        )
    except ModuleNotFoundError as exc:
        tracker_backend = "iou_fallback"
        bytetrack_available = False
        tracker_fallback_reason = str(exc)
        tracker = build_tracker(
            backend="iou",
            frame_rate=src_fps,
            iou_match_threshold=float(tcfg.get("iou_match_threshold", 0.25)),
            max_missing_frames=max_missing_frames,
            min_hits=int(tcfg.get("min_hits", 1)),
            bytetrack_cfg=tcfg.get("bytetrack", {}),
            iou_fallback_cfg=tcfg.get("iou_fallback", {}),
        )

    preview_path = run_dir / "preview.mp4"
    writer = None
    if save_preview_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(preview_path), fourcc, preview_fps, (width, height))

    tracks_csv_path = run_dir / "tracks.csv"
    csv_file = tracks_csv_path.open("w", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "frame_idx",
            "track_id",
            "x0",
            "y0",
            "x1",
            "y1",
            "score",
            "source",
            "prompt",
        ]
    )

    processed = 0
    keyframes = 0
    unique_track_ids = set()
    stop_requested = False
    last_visible_track_count = 0
    flow_refined_tracks_total = 0
    active_track_filtered_detections_total = 0
    active_track_refresh_keyframes = 0
    active_tracks_for_detection = []
    prev_gray = None

    progress_total = total_frames if total_frames > 0 else None
    pbar = tqdm(total=progress_total, desc="SAM3 text tracking", unit="frame", disable=not show_progress)
    frame_idx = 0
    started_at = time.perf_counter()
    detector_calls = 0
    detector_time_total_sec = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames > 0 and frame_idx >= max_frames:
                break

            is_keyframe = (frame_idx % max(1, keyframe_interval)) == 0
            should_detect = is_keyframe or (force_detect_when_empty and last_visible_track_count == 0)
            if should_detect:
                detect_t0 = time.perf_counter()
                detections = detector.detect(frame, prompts, include_masks=save_masks)
                detector_time_total_sec += max(0.0, time.perf_counter() - detect_t0)
                detector_calls += 1
                keyframe_index = frame_idx // max(1, keyframe_interval)
                refresh_active_tracks = (
                    detect_new_only_effective
                    and is_keyframe
                    and active_track_refresh_interval > 0
                    and (keyframe_index % active_track_refresh_interval == 0)
                )
                if refresh_active_tracks:
                    active_track_refresh_keyframes += 1
                if (
                    detect_new_only_effective
                    and (not refresh_active_tracks)
                    and len(active_tracks_for_detection) > 0
                ):
                    detections, dropped = _filter_detections_for_new_objects(
                        detections=detections,
                        active_tracks=active_tracks_for_detection,
                        frame_w=frame.shape[1],
                        frame_h=frame.shape[0],
                        iou_threshold=max(0.0, active_track_iou_threshold),
                    )
                    active_track_filtered_detections_total += dropped
                if is_keyframe:
                    keyframes += 1
            else:
                detections = []

            tracks = tracker.update(detections, detection_performed=should_detect)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if flow_enabled else None
            if flow_enabled and prev_gray is not None and curr_gray is not None and len(tracks) > 0:
                flow_refined_tracks_total += _refine_tracks_with_optical_flow(
                    prev_gray=prev_gray,
                    curr_gray=curr_gray,
                    tracks=tracks,
                    frame_w=frame.shape[1],
                    frame_h=frame.shape[0],
                    max_corners=flow_max_corners,
                    quality_level=flow_quality_level,
                    min_distance=flow_min_distance,
                    min_points=flow_min_points,
                    max_track_misses=flow_max_track_misses,
                )
            visible_tracks = []
            frame_h, frame_w = frame.shape[:2]
            for tr in tracks:
                pred_frames = int(getattr(tr, "predicted_frames", int(getattr(tr, "misses", 0))))
                if not tr.updated_this_frame and pred_frames > max_prediction_frames:
                    continue
                if not tr.updated_this_frame and float(tr.score) < prediction_min_score:
                    continue
                sanitized = _sanitize_bbox_xyxy(tr.bbox_xyxy, frame_w, frame_h)
                if sanitized is None:
                    continue
                tr.bbox_xyxy = sanitized
                visible_tracks.append(tr)

            for tr in visible_tracks:
                source = "sam3" if tr.updated_this_frame else "track"
                x0, y0, x1, y1 = tr.bbox_xyxy.tolist()
                unique_track_ids.add(int(tr.track_id))
                csv_writer.writerow(
                    [
                        frame_idx,
                        int(tr.track_id),
                        round(float(x0), 2),
                        round(float(y0), 2),
                        round(float(x1), 2),
                        round(float(y1), 2),
                        round(float(tr.score), 5),
                        source,
                        tr.prompt,
                    ]
                )
                if save_masks and tr.updated_this_frame and tr.last_mask is not None:
                    mask_u8 = (tr.last_mask.astype(np.uint8)) * 255
                    mask_path = masks_dir / f"frame_{frame_idx:06d}_id_{tr.track_id:04d}.png"
                    cv2.imwrite(str(mask_path), mask_u8)

            need_vis = (
                writer is not None
                or live_preview
                or (on_preview_frame is not None and (frame_idx % ui_emit_interval == 0))
            )
            if need_vis:
                vis = _draw_tracks(frame, visible_tracks, draw_tracker_predictions)
                if writer is not None:
                    writer.write(vis)
                if live_preview:
                    cv2.imshow(live_preview_window, vis)
                    k = cv2.waitKey(live_preview_wait_ms) & 0xFF
                    if live_preview_stop_on_q and k == ord("q"):
                        stop_requested = True
                        break
                if on_preview_frame is not None and (frame_idx % ui_emit_interval == 0):
                    ui_frame = _resize_for_ui_preview(vis, ui_preview_max_w, ui_preview_max_h)
                    on_preview_frame(ui_frame, frame_idx, processed)
            elif on_preview_frame is not None:
                on_preview_frame(frame, frame_idx, processed)
            if should_stop is not None and should_stop():
                stop_requested = True
                break

            last_visible_track_count = len(visible_tracks)
            active_tracks_for_detection = list(visible_tracks)
            if flow_enabled:
                prev_gray = curr_gray
            processed += 1
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()
        csv_file.close()
        if live_preview:
            cv2.destroyAllWindows()

    elapsed_total_sec = max(0.0, time.perf_counter() - started_at)

    quality_warning = None
    if tracker_backend == "iou_fallback" and tracker_fallback_reason:
        quality_warning = "ByteTrack unavailable; using iou_fallback."
        if keyframe_interval >= 6:
            quality_warning += " Keyframe interval >= 6 may reduce ID stability."

    summary = {
        "run_dir": str(run_dir),
        "video_path": str(video),
        "source_fps": src_fps,
        "preview_fps": preview_fps,
        "video_width": width,
        "video_height": height,
        "total_frames_in_video": total_frames,
        "processed_frames": processed,
        "keyframes_processed": keyframes,
        "keyframe_interval": keyframe_interval,
        "force_detect_when_empty": force_detect_when_empty,
        "detect_new_only_for_active_ids": detect_new_only_for_active_ids,
        "detect_new_only_effective": detect_new_only_effective,
        "active_track_iou_threshold": active_track_iou_threshold,
        "active_track_refresh_interval": active_track_refresh_interval,
        "active_track_refresh_keyframes": active_track_refresh_keyframes,
        "active_track_filtered_detections_total": active_track_filtered_detections_total,
        "tracker_backend": tracker_backend,
        "bytetrack_requested": bytetrack_requested,
        "bytetrack_available": bytetrack_available,
        "tracker_fallback_reason": tracker_fallback_reason,
        "quality_warning": quality_warning,
        "max_missing_frames": max_missing_frames,
        "max_prediction_frames": max_prediction_frames,
        "prediction_min_score": prediction_min_score,
        "optical_flow_enabled": flow_enabled,
        "optical_flow_max_track_misses": flow_max_track_misses,
        "optical_flow_refined_tracks_total": flow_refined_tracks_total,
        "ui_emit_interval": ui_emit_interval,
        "ui_preview_max_size": [ui_preview_max_w, ui_preview_max_h],
        "detector_calls": detector_calls,
        "detector_time_total_sec": round(detector_time_total_sec, 4),
        "detector_avg_ms": round((detector_time_total_sec * 1000.0 / max(1, detector_calls)), 3),
        "pipeline_elapsed_sec": round(elapsed_total_sec, 4),
        "pipeline_fps": round((processed / max(1e-6, elapsed_total_sec)), 3),
        "prompts": prompts,
        "unique_track_count": len(unique_track_ids),
        "tracks_csv": str(tracks_csv_path),
        "preview_video": str(preview_path) if save_preview_video else None,
        "masks_dir": str(masks_dir) if save_masks else None,
        "live_preview_stopped_by_user": stop_requested,
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
