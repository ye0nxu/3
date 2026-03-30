from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
from PyQt6.QtCore import QObject, pyqtSignal

from config import apply_remote_env_defaults
from config.settings import missing_remote_ssh_fields
from backend.pipelines.preview_postprocess import postprocess_preview_items, preview_category_summary
from backend.llm.prompting import build_sam_prompt_candidates
try:
    from backend.filters import FilterConfig, SampleCandidate, SampleFilterEngine
except Exception:
    FilterConfig = None
    SampleCandidate = None
    SampleFilterEngine = None

from core.models import PreviewThumbnail
from core.paths import APP_ROOT, PREVIEW_CACHE_BASE, PROGRAM_ROOT
from utils.env import env_flag, env_int


SAM3_PROGRESS_PREFIX = "__SAM3_PROGRESS__="
SAM3_SUMMARY_PREFIX = "__SAM3_SUMMARY__="
# 하드코딩 폴백: config.local.json 으로 반드시 덮어쓰기 권장
_DEFAULT_REMOTE_PROGRAM_ROOT = os.getenv(
    "APP_REMOTE_STORAGE_PROGRAM_ROOT",
    f"G:/KDT10_3_1team_KLIK/0_Program_/{PROGRAM_ROOT.name}",
)
_DEFAULT_REMOTE_SAM3_ROOT = "G:/models/sam3"
VENDORED_SAM3_PROJECT_ROOT = APP_ROOT / "vendor" / "sam_3"


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _normalize_remote_device(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "cuda"
    lowered = text.lower()
    if lowered in {"cpu", "cuda"} or lowered.startswith("cuda:"):
        return lowered
    if re.fullmatch(r"\d+", text):
        return f"cuda:{text}"
    return text


def _normalize_discovered_windows_python_candidate(raw_line: str) -> str:
    text = str(raw_line or "").strip().replace("\\", "/")
    if not text or text.startswith("#"):
        return ""
    lowered = text.lower()
    if "windowsapps/python.exe" in lowered:
        return ""
    python_match = re.search(r"([A-Za-z]:/[^\s]*python\.exe)\s*$", text, flags=re.IGNORECASE)
    if python_match:
        candidate = python_match.group(1).strip()
    else:
        env_match = re.search(r"([A-Za-z]:/[^\s]+)\s*$", text, flags=re.IGNORECASE)
        if not env_match:
            return ""
        env_path = env_match.group(1).strip().rstrip("/")
        lowered_env = env_path.lower()
        if lowered_env in {"g:/anaconda3", "c:/programdata/anaconda3"}:
            candidate = f"{env_path}/python.exe"
        elif any(token in lowered_env for token in ("/envs/", "/conda_envs/", "/miniconda3/envs/", "/anaconda3/envs/")):
            candidate = f"{env_path}/python.exe"
        else:
            return ""
    lowered_candidate = candidate.lower()
    if not any(keyword in lowered_candidate for keyword in ("pj_310", "sam", "llm", "anaconda3/python.exe")):
        return ""
    return candidate


def _is_absolute_remote_path(value: str) -> bool:
    text = str(value or "").strip().replace("\\", "/")
    return bool(re.match(r"^[A-Za-z]:/", text) or text.startswith("/"))


def _decode_remote_output(data: bytes, is_remote_windows: bool) -> str:
    raw = bytes(data or b"")
    encodings = ["utf-8"]
    if is_remote_windows:
        encodings.extend(["cp949", "euc-kr"])
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except Exception:
            continue
    if is_remote_windows:
        return raw.decode("cp949", errors="replace")
    return raw.decode("utf-8", errors="replace")


@dataclass(slots=True)
class RemoteSam3Config:
    enabled: bool
    required: bool
    ssh_host: str
    ssh_port: int
    ssh_user: str
    ssh_password: str
    python_cmd: str
    remote_work_root: str
    remote_sam3_root: str
    device: str
    connect_timeout_sec: float
    keep_remote_files: bool

    @classmethod
    def from_env(cls) -> "RemoteSam3Config":
        apply_remote_env_defaults()
        preferred_device = str(os.getenv("NEW_OBJECT_REMOTE_DEVICE", os.getenv("TRAIN_REMOTE_DEVICE", "cuda"))).strip()
        remote_work_root = str(
            os.getenv("NEW_OBJECT_REMOTE_WORKDIR", os.getenv("TRAIN_REMOTE_WORKDIR", _DEFAULT_REMOTE_PROGRAM_ROOT))
        ).strip() or _DEFAULT_REMOTE_PROGRAM_ROOT
        if not _is_absolute_remote_path(remote_work_root):
            remote_work_root = _DEFAULT_REMOTE_PROGRAM_ROOT
        return cls(
            enabled=env_flag("NEW_OBJECT_REMOTE_ENABLE", env_flag("TRAIN_REMOTE_ENABLE", True)),
            required=env_flag("NEW_OBJECT_REMOTE_REQUIRED", env_flag("TRAIN_REMOTE_REQUIRED", True)),
            ssh_host=str(os.getenv("NEW_OBJECT_REMOTE_SSH_HOST", os.getenv("TRAIN_REMOTE_SSH_HOST", ""))).strip(),
            ssh_port=env_int("NEW_OBJECT_REMOTE_SSH_PORT", env_int("TRAIN_REMOTE_SSH_PORT", 8875)),
            ssh_user=str(os.getenv("NEW_OBJECT_REMOTE_SSH_USER", os.getenv("TRAIN_REMOTE_SSH_USER", ""))).strip(),
            ssh_password=str(os.getenv("NEW_OBJECT_REMOTE_SSH_PASSWORD", os.getenv("TRAIN_REMOTE_SSH_PASSWORD", ""))),
            python_cmd=str(
                os.getenv("NEW_OBJECT_REMOTE_PYTHON_CMD", os.getenv("TRAIN_REMOTE_PYTHON_CMD", "G:/conda/envs/PJ_310_LLM_SAM3/python.exe"))
            ).strip()
            or "G:/conda/envs/PJ_310_LLM_SAM3/python.exe",
            remote_work_root=remote_work_root,
            remote_sam3_root=str(os.getenv("NEW_OBJECT_REMOTE_SAM3_ROOT", _DEFAULT_REMOTE_SAM3_ROOT)).strip()
            or _DEFAULT_REMOTE_SAM3_ROOT,
            device=_normalize_remote_device(preferred_device),
            connect_timeout_sec=max(
                3.0,
                _env_float("NEW_OBJECT_REMOTE_CONNECT_TIMEOUT", _env_float("TRAIN_REMOTE_CONNECT_TIMEOUT", 15.0)),
            ),
            keep_remote_files=env_flag("NEW_OBJECT_REMOTE_KEEP_FILES", env_flag("TRAIN_REMOTE_KEEP_FILES", False)),
        )


@dataclass(slots=True)
class RemoteSam3Request:
    class_name: str
    prompt_text: str
    prompt_candidates: Sequence[str]
    video_path: str
    experiment_id: str
    output_dir: str


def _build_prompt_variants(
    prompt_text: str,
    class_name: str,
    ranked_candidates: Sequence[str] | None = None,
) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        value = str(raw or "").strip()
        if not value:
            return
        key = value.casefold()
        if key in seen:
            return
        seen.add(key)
        variants.append(value)

    base_prompt = str(prompt_text or "").strip()
    if base_prompt:
        add(base_prompt)
        if not base_prompt.endswith("."):
            add(f"{base_prompt}.")
    for candidate in ranked_candidates or ():
        if len(variants) >= 2:
            break
        candidate_text = str(candidate or "").strip()
        if not candidate_text:
            continue
        add(candidate_text)
    if not variants:
        fallback_candidates = build_sam_prompt_candidates(
            prompt_text=prompt_text,
            class_name=class_name,
            ranked_candidates=ranked_candidates,
            limit=2,
        )
        for candidate in fallback_candidates:
            add(candidate)
    if not variants:
        add("object.")
    return variants[:2]


class WorkerStoppedError(RuntimeError):
    pass


def resolve_local_sam3_project_root() -> Path:
    candidates: list[Path] = []
    env_root = str(os.getenv("NEW_OBJECT_SAM3_PROJECT_ROOT", "")).strip()
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(VENDORED_SAM3_PROJECT_ROOT)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_dir() and (resolved / "src" / "sam3_text_prompt_tracker").is_dir():
            return resolved
    joined = "\n".join(str(path.expanduser().resolve()) for path in candidates)
    raise RuntimeError(
        "Local sam_3 project not found. "
        "Set NEW_OBJECT_SAM3_PROJECT_ROOT or restore the vendored sam_3 runtime.\n"
        f"{joined}".rstrip()
    )


def resolve_local_sam3_config_path(project_root: Path) -> Path:
    env_path = str(os.getenv("NEW_OBJECT_SAM3_CONFIG", "")).strip()
    if env_path:
        config_path = Path(env_path).expanduser().resolve()
        if not config_path.is_file():
            raise RuntimeError(f"SAM3 config not found: {config_path}")
        return config_path
    for name in ("a100.json", "default.json"):
        candidate = (project_root / "configs" / name).resolve()
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"SAM3 config not found under: {project_root / 'configs'}")


def summarize_preview_items(preview_items: Sequence[PreviewThumbnail]) -> dict[str, int]:
    frame_to_count: dict[int, int] = {}
    keep_items = 0
    hold_items = 0
    for item in preview_items:
        category = str(getattr(item, "category", "")).strip().lower()
        if category == "hold":
            hold_items += 1
        if category != "keep":
            continue
        keep_items += 1
        frame_index = int(getattr(item, "frame_index", -1))
        if frame_index < 0:
            continue
        for box in list(getattr(item, "boxes", []) or []):
            if not isinstance(box, Mapping):
                continue
            if str(box.get("status", "")).strip().lower() != "keep":
                continue
            frame_to_count[frame_index] = frame_to_count.get(frame_index, 0) + 1
    return {
        "num_images": len(frame_to_count),
        "num_labels": sum(frame_to_count.values()),
        "keep_items": keep_items,
        "hold_items": hold_items,
    }


def _normalize_preview_category(value: str, default: str = "hold") -> str:
    key = str(value or "").strip().lower()
    if key in {"keep", "hold", "drop"}:
        return key
    return str(default or "hold").strip().lower() or "hold"


def _build_single_box_preview_item(
    *,
    experiment_id: str,
    frame_index: int,
    category: str,
    box_payload: Mapping[str, Any],
    sample_index: int,
) -> PreviewThumbnail:
    copied_box = dict(box_payload)
    normalized_category = _normalize_preview_category(category)
    track_label = "unknown"
    try:
        raw_track = copied_box.get("track_id")
        if raw_track is not None:
            track_label = f"trk{int(raw_track):04d}"
    except Exception:
        track_label = "unknown"
    item_id = str(copied_box.get("preview_item_id", "")).strip()
    if not item_id:
        item_id = f"{experiment_id}_{track_label}_{int(frame_index):06d}_{int(sample_index):05d}"
    copied_box["preview_item_id"] = item_id
    copied_box["status"] = normalized_category
    return PreviewThumbnail(
        frame_index=int(frame_index),
        image_path=None,
        category=normalized_category,
        item_id=item_id,
        boxes=[copied_box],
    )


def _preview_frame_counts_by_category(preview_items: Sequence[PreviewThumbnail]) -> dict[str, int]:
    frames_by_category: dict[str, set[int]] = {
        "keep": set(),
        "hold": set(),
        "drop": set(),
    }
    for item in preview_items:
        category = _normalize_preview_category(getattr(item, "category", "hold"))
        if category not in frames_by_category:
            continue
        try:
            frame_index = int(getattr(item, "frame_index", -1))
        except Exception:
            continue
        if frame_index < 0:
            continue
        frames_by_category[category].add(frame_index)
    return {key: len(value) for key, value in frames_by_category.items()}


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _preview_cache_root_for_experiment(experiment_id: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(experiment_id or "").strip())
    return (PREVIEW_CACHE_BASE / "new_object_labeling" / (safe_name or "latest")).resolve()


def _clear_preview_cache_root(cache_root: Path | None) -> None:
    if cache_root is None:
        return
    try:
        shutil.rmtree(cache_root, ignore_errors=True)
    except Exception:
        pass


def _extract_frames_from_video(video_path: str, frame_indices: Sequence[int]) -> dict[int, Any]:
    wanted = sorted({int(idx) for idx in frame_indices if int(idx) >= 0})
    resolved: dict[int, Any] = {}
    if not wanted:
        return resolved
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        next_expected_frame = -1
        for frame_index in wanted:
            if frame_index != next_expected_frame:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                next_expected_frame = -1
                continue
            resolved[frame_index] = frame.copy()
            next_expected_frame = int(frame_index) + 1
    finally:
        capture.release()
    return resolved


def _persist_preview_items_to_cache(
    cache_root: Path,
    items: Sequence[PreviewThumbnail],
    *,
    video_path: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    ## =====================================
    ## 함수 기능 : 미리보기 아이템을 디스크 캐시에 저장 (프레임은 스트리밍 방식으로 1장씩 처리하여 RAM 사용 최소화)
    ## 매개 변수 : cache_root(Path), items(Sequence[PreviewThumbnail]), video_path(str|None), progress_callback(callable|None), should_stop(callable|None)
    ## 반환 결과 : count(int) -> 저장된 아이템 수
    ## =====================================
    _clear_preview_cache_root(cache_root)
    total_items = len(items)
    if should_stop is not None and should_stop():
        raise WorkerStoppedError()
    frame_thumb_paths: dict[int, str] = {}
    if video_path and items:
        wanted_indices = sorted({int(getattr(item, "frame_index", -1)) for item in items if int(getattr(item, "frame_index", -1)) >= 0})
        if wanted_indices:
            frame_dir = cache_root / "_frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            capture = cv2.VideoCapture(str(video_path))
            if capture.isOpened():
                try:
                    next_expected_frame = -1
                    for frame_index in wanted_indices:
                        if should_stop is not None and should_stop():
                            raise WorkerStoppedError()
                        if frame_index != next_expected_frame:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                        ok, frame = capture.read()
                        if not ok or frame is None:
                            next_expected_frame = -1
                            continue
                        thumb_candidate = frame_dir / f"frame_{int(frame_index):06d}.jpg"
                        if cv2.imwrite(str(thumb_candidate), frame):
                            frame_thumb_paths[int(frame_index)] = str(thumb_candidate)
                        next_expected_frame = int(frame_index) + 1
                finally:
                    capture.release()
    count = 0
    for item in items:
        if should_stop is not None and should_stop():
            raise WorkerStoppedError()
        category = str(getattr(item, "category", "")).strip().lower()
        if category not in {"keep", "hold", "drop"}:
            category = "hold"
        item_id = str(getattr(item, "item_id", "")).strip() or f"frame_{int(getattr(item, 'frame_index', 0)):06d}_{count:06d}"
        target_dir = cache_root / category
        target_dir.mkdir(parents=True, exist_ok=True)
        frame_index = int(getattr(item, "frame_index", -1))
        payload = {
            "frame_index": frame_index,
            "category": category,
            "item_id": item_id,
            "image_path": "",
            "thumb_path": str(frame_thumb_paths.get(frame_index, "")),
            "boxes": [dict(box) for box in list(getattr(item, "boxes", []) or []) if isinstance(box, Mapping)],
        }
        (target_dir / f"{item_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1
        if progress_callback is not None and ((count % 250) == 0 or count >= total_items):
            progress_callback(count, total_items)
    return count


def _preview_box_color(box: Mapping[str, Any]) -> tuple[int, int, int]:
    status = str(box.get("status", "")).strip().lower()
    if status == "keep":
        return (64, 220, 96)
    if status == "drop":
        return (64, 96, 255)
    return (0, 196, 255)


def _draw_preview_box(frame: Any, box: Mapping[str, Any]) -> None:
    if frame is None:
        return
    try:
        x1 = int(round(float(box.get("x1", 0))))
        y1 = int(round(float(box.get("y1", 0))))
        x2 = int(round(float(box.get("x2", 0))))
        y2 = int(round(float(box.get("y2", 0))))
    except Exception:
        return
    if x2 <= x1 or y2 <= y1:
        return
    color = _preview_box_color(box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    status = str(box.get("status", "hold")).strip().lower() or "hold"
    track_id = str(box.get("track_id", "")).strip()
    score = box.get("score")
    label = status
    if track_id:
        label += f" #{track_id}"
    try:
        label += f" {float(score):.2f}"
    except Exception:
        pass
    text_scale = 0.5
    text_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
    text_y = max(text_h + 6, y1 - 6)
    text_x = max(0, x1)
    cv2.rectangle(
        frame,
        (text_x, max(0, text_y - text_h - baseline - 4)),
        (text_x + text_w + 8, text_y + 2),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (text_x + 4, text_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (10, 18, 28),
        text_thickness,
        cv2.LINE_AA,
    )


def _render_preview_video_locally(
    *,
    output_dir: Path,
    preview_items: Sequence[PreviewThumbnail],
    summary: Mapping[str, Any],
    video_path: str,
) -> str:
    remote_run_dir = (output_dir / "_remote_run").resolve()
    preview_path = remote_run_dir / "preview.mp4"
    boxes_by_frame: dict[int, list[dict[str, Any]]] = {}
    for item in preview_items:
        try:
            frame_index = int(getattr(item, "frame_index", -1))
        except Exception:
            continue
        if frame_index < 0:
            continue
        bucket = boxes_by_frame.setdefault(frame_index, [])
        for box in list(getattr(item, "boxes", []) or []):
            if isinstance(box, Mapping):
                bucket.append(dict(box))

    frame_indices = sorted(boxes_by_frame)
    frames_by_index = _extract_frames_from_video(video_path, frame_indices)
    if not frames_by_index:
        return ""

    first_frame = next((frame for frame in frames_by_index.values() if frame is not None), None)
    if first_frame is None:
        return ""
    height, width = first_frame.shape[:2]
    fps = float(summary.get("preview_fps", summary.get("source_fps", 15.0)) or 15.0)
    writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (width, height))
    if not writer.isOpened():
        return ""

    try:
        for frame_index in frame_indices:
            frame = frames_by_index.get(frame_index)
            if frame is None:
                continue
            frame = frame.copy()
            for box in boxes_by_frame.get(frame_index, []):
                _draw_preview_box(frame, box)
            writer.write(frame)
    finally:
        writer.release()
    return str(preview_path) if preview_path.is_file() else ""


def _crop_box_region(frame: Any, box: Mapping[str, Any]) -> Any | None:
    if frame is None or not hasattr(frame, "shape"):
        return None
    try:
        height, width = frame.shape[:2]
        x1 = int(round(float(box.get("x1", 0))))
        y1 = int(round(float(box.get("y1", 0))))
        x2 = int(round(float(box.get("x2", 0))))
        y2 = int(round(float(box.get("y2", 0))))
    except Exception:
        return None
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop is None or getattr(crop, "size", 0) <= 0:
        return None
    return crop


def _normalize_filter_drop_reason(reason: str) -> str:
    text = str(reason or "").strip().upper()
    if not text:
        return "invalid_sample"
    if text.startswith("DROP_"):
        text = text[5:]
    return text.lower()


def build_preview_items_from_remote_run(
    *,
    output_dir: Path,
    video_path: str,
    experiment_id: str,
    progress_callback: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[PreviewThumbnail], dict[str, Any]]:
    remote_run_dir = (output_dir / "_remote_run").resolve()
    tracks_csv_path = remote_run_dir / "tracks.csv"
    summary_path = remote_run_dir / "summary.json"
    if not tracks_csv_path.is_file():
        raise RuntimeError(f"Remote tracks.csv not found: {tracks_csv_path}")

    rows: list[dict[str, Any]] = []
    frame_indices: list[int] = []
    unique_track_ids: set[int] = set()
    seen_rows: set[tuple[Any, ...]] = set()
    with tracks_csv_path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if not isinstance(row, Mapping):
                continue
            try:
                frame_index = int(float(row.get("frame_idx", "0")))
                track_id = int(float(row.get("track_id", "0")))
                x0 = int(round(float(row.get("x0", "0"))))
                y0 = int(round(float(row.get("y0", "0"))))
                x1 = int(round(float(row.get("x1", "0"))))
                y1 = int(round(float(row.get("y1", "0"))))
                score = float(row.get("score", "0"))
            except Exception:
                continue
            if frame_index < 0 or x1 <= x0 or y1 <= y0:
                continue
            source = str(row.get("source", "track")).strip().lower() or "track"
            prompt = str(row.get("prompt", "")).strip()
            row_key = (frame_index, track_id, x0, y0, x1, y1, round(score, 5), source, prompt)
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            unique_track_ids.add(track_id)
            rows.append(
                {
                    "frame_index": frame_index,
                    "track_id": track_id,
                    "x1": x0,
                    "y1": y0,
                    "x2": x1,
                    "y2": y1,
                    "score": score,
                    "source": source,
                    "prompt": prompt,
                }
            )
            frame_indices.append(frame_index)

    summary = _read_json_file(summary_path)
    rows.sort(
        key=lambda item: (
            int(item["frame_index"]),
            0 if str(item["source"]) == "sam3" else 1,
            int(item["track_id"]),
            -float(item["score"]),
        )
    )
    total_rows = len(rows)
    raw_preview_items: list[PreviewThumbnail] = []
    filter_engine = None
    if SampleFilterEngine is not None and FilterConfig is not None:
        try:
            filter_engine = SampleFilterEngine(FilterConfig())
        except Exception:
            filter_engine = None
    source_fps = float(summary.get("source_fps", 30.0) or 30.0)
    hold_frame_gap = max(1, env_int("NEW_OBJECT_HOLD_SAMPLE_GAP", 1))
    hold_max_per_track = max(0, env_int("NEW_OBJECT_HOLD_MAX_PER_TRACK", 0))
    hold_sample_state: dict[int, tuple[int, int]] = {}
    keep_count = 0
    hold_count = 0
    drop_count = 0
    drop_by_reason: dict[str, int] = {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    current_frame_index = -1
    current_frame = None
    try:
        for index, row in enumerate(rows, start=1):
            if should_stop is not None and should_stop():
                raise WorkerStoppedError()
            if progress_callback is not None and (index == 1 or (index % 250) == 0 or index >= total_rows):
                progress_callback(index, total_rows)
            frame_index = int(row["frame_index"])
            if frame_index != current_frame_index:
                if frame_index != current_frame_index + 1:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    current_frame_index = -1
                    current_frame = None
                    continue
                current_frame_index = frame_index
                current_frame = frame
            frame = current_frame
            if frame is None:
                continue
            source = str(row["source"]).strip().lower()
            base_category = "keep" if source == "sam3" else "hold"
            track_id = int(row["track_id"])
            if base_category == "hold":
                last_frame, sampled_count = hold_sample_state.get(track_id, (-10**9, 0))
                if hold_max_per_track > 0 and sampled_count >= hold_max_per_track:
                    continue
                if (frame_index - last_frame) < hold_frame_gap:
                    continue
                hold_sample_state[track_id] = (frame_index, sampled_count + 1)
            category = base_category
            drop_reason = ""
            metrics: dict[str, Any] = {}
            if filter_engine is not None and SampleCandidate is not None:
                crop = _crop_box_region(frame, row)
                try:
                    candidate = SampleCandidate(
                        frame_idx=int(frame_index),
                        timestamp_ms=(float(frame_index) / max(1e-6, float(source_fps))) * 1000.0,
                        track_id=str(track_id),
                        bbox=(
                            float(row["x1"]),
                            float(row["y1"]),
                            float(row["x2"]),
                            float(row["y2"]),
                        ),
                        crop_image=crop,
                        sample_id=(
                            f"sam3_trk{track_id:04d}_frm{int(frame_index):06d}_"
                            f"{str(source or 'track')}_{index:05d}"
                        ),
                        meta={
                            "score": float(row["score"]),
                            "prompt": str(row["prompt"]),
                            "source": source,
                        },
                    )
                    result = filter_engine.evaluate(candidate)
                    metrics = dict(getattr(result, "metrics", {}) or {})
                    if bool(getattr(result, "passed", False)):
                        if base_category == "keep":
                            try:
                                filter_engine.on_final_keep(candidate)
                            except Exception:
                                pass
                    else:
                        category = "drop"
                        drop_reason = _normalize_filter_drop_reason(getattr(result, "reason", "DROP_INVALID_SAMPLE"))
                except Exception:
                    category = base_category

            if category == "keep":
                keep_count += 1
            elif category == "hold":
                hold_count += 1
            else:
                drop_count += 1
                if drop_reason:
                    drop_by_reason[drop_reason] = int(drop_by_reason.get(drop_reason, 0)) + 1

            box_payload: dict[str, Any] = {
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "x2": int(row["x2"]),
                "y2": int(row["y2"]),
                "status": str(category).strip().lower(),
                "class_id": 0,
                "track_id": int(row["track_id"]),
                "score": float(row["score"]),
                "prompt": str(row["prompt"]),
                "source": source,
            }
            if drop_reason:
                box_payload["drop_reason"] = drop_reason
            for metric_key in ("blur_score", "white_ratio", "black_ratio", "track_hash_dist", "global_hash_dist"):
                if metric_key in metrics:
                    box_payload[metric_key] = metrics[metric_key]
            raw_preview_items.append(
                _build_single_box_preview_item(
                    experiment_id=experiment_id,
                    frame_index=frame_index,
                    category=category,
                    box_payload=box_payload,
                    sample_index=index,
                )
            )
    finally:
        capture.release()
    if progress_callback is not None:
        progress_callback(total_rows, total_rows)
    processed_preview_items = postprocess_preview_items(raw_preview_items)
    category_summary = preview_category_summary(processed_preview_items)
    frame_counts = _preview_frame_counts_by_category(processed_preview_items)
    preview_frame_total = len(
        {
            int(getattr(item, "frame_index", -1))
            for item in processed_preview_items
            if int(getattr(item, "frame_index", -1)) >= 0
        }
    )
    preview_video_local = (remote_run_dir / "preview.mp4").resolve()
    summary["rows_total"] = len(rows)
    summary["keep_boxes"] = int(keep_count)
    summary["hold_boxes"] = int(hold_count)
    summary["drop_boxes"] = int(drop_count)
    summary["keep_items"] = int(category_summary["keep"])
    summary["hold_items"] = int(category_summary["hold"])
    summary["drop_items"] = int(category_summary["drop"])
    summary["keep_frames"] = int(frame_counts["keep"])
    summary["hold_frames"] = int(frame_counts["hold"])
    summary["drop_frames"] = int(frame_counts["drop"])
    summary["preview_frames"] = int(preview_frame_total)
    summary["filtered_unique_track_count"] = int(len(unique_track_ids))
    summary["filter_drop_by_reason"] = dict(drop_by_reason)
    summary["preview_video_local"] = str(preview_video_local) if preview_video_local.is_file() else ""
    return processed_preview_items, summary


def _sftp_mkdir_p(sftp: Any, remote_dir: str) -> None:
    target = str(remote_dir).replace("\\", "/").strip()
    if not target:
        return
    parts = target.split("/")
    current = "/" if target.startswith("/") else ""
    for part in parts:
        part = str(part).strip()
        if not part:
            continue
        current = f"{current.rstrip('/')}/{part}" if current else part
        try:
            sftp.stat(current)
        except Exception:
            sftp.mkdir(current)


def _sftp_path_exists(sftp: Any, remote_path: str) -> bool:
    try:
        sftp.stat(str(remote_path).replace("\\", "/"))
        return True
    except Exception:
        return False


def _iter_uploadable_files(local_root: Path) -> list[tuple[Path, str]]:
    local_root = local_root.resolve()
    skip_dirs = {".git", ".pytest_cache", "__pycache__", ".mypy_cache", "runs", "tests"}
    skip_suffixes = {".pyc", ".pyo"}
    files: list[tuple[Path, str]] = []
    for root, dirs, names in os.walk(local_root):
        dirs[:] = [name for name in dirs if name not in skip_dirs]
        root_path = Path(root)
        for name in sorted(names):
            if any(name.endswith(suffix) for suffix in skip_suffixes):
                continue
            local_file = root_path / name
            relative = local_file.relative_to(local_root).as_posix()
            files.append((local_file, relative))
    return files


def _project_tree_signature(
    local_root: Path,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    digest = hashlib.sha256()
    for local_file, relative in _iter_uploadable_files(local_root):
        if should_stop is not None and should_stop():
            raise WorkerStoppedError()
        digest.update(relative.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(local_file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sftp_put_file(
    sftp: Any,
    local_path: Path,
    remote_path: str,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    def _transfer_callback(_sent: int, _total: int) -> None:
        if should_stop is not None and should_stop():
            raise WorkerStoppedError()

    _sftp_mkdir_p(sftp, str(remote_path).replace("\\", "/").rsplit("/", 1)[0])
    sftp.put(str(local_path), str(remote_path).replace("\\", "/"), callback=_transfer_callback)


def _sftp_get_file(
    sftp: Any,
    remote_path: str,
    local_path: Path,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> bool:
    normalized_remote = str(remote_path).replace("\\", "/")
    if not _sftp_path_exists(sftp, normalized_remote):
        return False

    def _transfer_callback(_sent: int, _total: int) -> None:
        if should_stop is not None and should_stop():
            raise WorkerStoppedError()

    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(normalized_remote, str(local_path), callback=_transfer_callback)
    return True


def _sftp_upload_tree_filtered(
    sftp: Any,
    local_root: Path,
    remote_root: str,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    local_root = local_root.resolve()
    remote_root = str(remote_root).replace("\\", "/")
    _sftp_mkdir_p(sftp, remote_root)
    for local_file, relative in _iter_uploadable_files(local_root):
        if should_stop is not None and should_stop():
            raise WorkerStoppedError()
        remote_file = f"{remote_root.rstrip('/')}/{relative}"
        _sftp_put_file(sftp, local_file, remote_file, should_stop=should_stop)

def _sftp_download_tree(
    sftp: Any,
    remote_root: str,
    local_root: Path,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    def _transfer_callback(_sent: int, _total: int) -> None:
        if should_stop is not None and should_stop():
            raise WorkerStoppedError()

    local_root.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(str(remote_root)):
        if should_stop is not None and should_stop():
            raise WorkerStoppedError()
        remote_path = f"{str(remote_root).rstrip('/')}/{entry.filename}"
        local_path = local_root / str(entry.filename)
        if stat.S_ISDIR(int(entry.st_mode)):
            _sftp_download_tree(sftp, remote_path, local_path, should_stop=should_stop)
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_path, str(local_path), callback=_transfer_callback)


def _sftp_download_selected_files(
    sftp: Any,
    remote_root: str,
    local_root: Path,
    names: Sequence[str],
    *,
    should_stop: Callable[[], bool] | None = None,
) -> list[str]:
    downloaded: list[str] = []
    local_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        if should_stop is not None and should_stop():
            raise WorkerStoppedError()
        filename = str(name).strip()
        if not filename:
            continue
        remote_path = f"{str(remote_root).rstrip('/')}/{filename}"
        local_path = local_root / filename
        if _sftp_get_file(sftp, remote_path, local_path, should_stop=should_stop):
            downloaded.append(filename)
    return downloaded


def _ensure_remote_cached_project(
    sftp: Any,
    *,
    local_root: Path,
    remote_cache_root: str,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[str, bool]:
    project_signature = _project_tree_signature(local_root, should_stop=should_stop)
    remote_project_root = f"{str(remote_cache_root).rstrip('/')}/{local_root.name}_{project_signature[:16]}"
    marker_path = f"{remote_project_root}/.upload_complete"
    if _sftp_path_exists(sftp, marker_path) and _sftp_path_exists(
        sftp,
        f"{remote_project_root}/src/sam3_text_prompt_tracker/pipeline.py",
    ):
        return remote_project_root, True

    _sftp_upload_tree_filtered(
        sftp,
        local_root,
        remote_project_root,
        should_stop=should_stop,
    )
    with sftp.file(marker_path, "wb") as fp:
        fp.write(project_signature.encode("utf-8"))
    return remote_project_root, False


def _fingerprint_file(
    local_path: Path,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    source = Path(local_path).resolve()
    stat_result = source.stat()
    sample_size = 1024 * 1024
    digest = hashlib.sha256()
    digest.update(source.name.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(str(int(stat_result.st_size)).encode("ascii", errors="ignore"))
    digest.update(b"\0")
    digest.update(str(int(stat_result.st_mtime_ns)).encode("ascii", errors="ignore"))
    digest.update(b"\0")
    with source.open("rb") as fp:
        if should_stop is not None and should_stop():
            raise WorkerStoppedError()
        digest.update(fp.read(sample_size))
        if stat_result.st_size > sample_size:
            tail_size = min(sample_size, int(stat_result.st_size))
            fp.seek(max(0, int(stat_result.st_size) - tail_size))
            if should_stop is not None and should_stop():
                raise WorkerStoppedError()
            digest.update(fp.read(tail_size))
    return digest.hexdigest()


def _ensure_remote_cached_input_file(
    sftp: Any,
    *,
    local_path: Path,
    remote_cache_root: str,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[str, bool]:
    source = Path(local_path).resolve()
    file_signature = _fingerprint_file(source, should_stop=should_stop)
    remote_name = f"{source.stem}_{file_signature[:16]}{source.suffix}"
    remote_path = f"{str(remote_cache_root).rstrip('/')}/{remote_name}"
    if _sftp_path_exists(sftp, remote_path):
        return remote_path, True
    _sftp_put_file(sftp, source, remote_path, should_stop=should_stop)
    return remote_path, False


def _build_runtime_config(
    *,
    local_config_path: Path,
    remote_sam3_root: str,
    remote_device: str,
    remote_runs_root: str,
    prompts: Sequence[str],
) -> str:
    payload = _read_json_file(local_config_path)
    if not payload:
        raise RuntimeError(f"Invalid SAM3 config: {local_config_path}")
    payload["sam3_root"] = str(remote_sam3_root).replace("\\", "/")
    payload["run_root"] = str(remote_runs_root).replace("\\", "/")
    payload["device"] = str(remote_device).strip() or "cuda"
    payload["prompts"] = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
    sam3_cfg = dict(payload.get("sam3") or {})
    payload["sam3"] = sam3_cfg
    output_cfg = dict(payload.get("output") or {})
    output_cfg["save_preview_video"] = env_flag(
        "NEW_OBJECT_REMOTE_SAVE_PREVIEW_VIDEO",
        bool(output_cfg.get("save_preview_video", False)),
    )
    output_cfg["save_masks"] = False
    output_cfg["show_progress"] = False
    output_cfg["live_preview"] = False
    output_cfg["ui_emit_interval"] = max(1, int(output_cfg.get("ui_emit_interval", 4)))
    payload["output"] = output_cfg
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_remote_script_text() -> str:
    return (
        "from __future__ import annotations\n"
        "import argparse\n"
        "import json\n"
        "import sys\n"
        "import types\n"
        "from pathlib import Path\n\n"
        f"SAM3_PROGRESS_PREFIX = {SAM3_PROGRESS_PREFIX!r}\n"
        f"SAM3_SUMMARY_PREFIX = {SAM3_SUMMARY_PREFIX!r}\n\n"
        "def _parse() -> argparse.Namespace:\n"
        "    p = argparse.ArgumentParser(description='Run remote SAM3 text tracking')\n"
        "    p.add_argument('--project-root', required=True)\n"
        "    p.add_argument('--config', required=True)\n"
        "    p.add_argument('--video', required=True)\n"
        "    p.add_argument('--run-name', required=True)\n"
        "    return p.parse_args()\n\n"
        "def _install_pkg_resources_shim() -> None:\n"
        "    if 'pkg_resources' in sys.modules:\n"
        "        return\n"
        "    shim = types.ModuleType('pkg_resources')\n"
        "    def resource_filename(package_or_requirement, resource_name):\n"
        "        import importlib\n"
        "        package = importlib.import_module(str(package_or_requirement)) if isinstance(package_or_requirement, str) else package_or_requirement\n"
        "        package_path = Path(getattr(package, '__file__', '')).resolve().parent\n"
        "        return str((package_path / str(resource_name)).resolve())\n"
        "    shim.resource_filename = resource_filename\n"
        "    sys.modules['pkg_resources'] = shim\n\n"
        "def _emit_progress(_frame, frame_idx: int, processed: int) -> None:\n"
        "    payload = {\n"
        "        'frame_idx': int(frame_idx),\n"
        "        'frames_processed': int(max(frame_idx + 1, processed + 1)),\n"
        "    }\n"
        "    print(SAM3_PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=True), flush=True)\n\n"
        "def main() -> None:\n"
        "    args = _parse()\n"
        "    project_root = Path(args.project_root).resolve()\n"
        "    src_path = project_root / 'src'\n"
        "    if str(project_root) not in sys.path:\n"
        "        sys.path.insert(0, str(project_root))\n"
        "    if str(src_path) not in sys.path:\n"
        "        sys.path.insert(0, str(src_path))\n"
        "    _install_pkg_resources_shim()\n"
        "    from sam3_text_prompt_tracker.pipeline import run_pipeline\n"
        "    summary = run_pipeline(\n"
        "        config_path=str(Path(args.config).resolve()),\n"
        "        video_path=str(Path(args.video).resolve()),\n"
        "        run_name=str(args.run_name),\n"
        "        show_progress_override=False,\n"
        "        on_preview_frame=_emit_progress,\n"
        "    )\n"
        "    print(SAM3_SUMMARY_PREFIX + json.dumps(summary, ensure_ascii=True), flush=True)\n"
        "    print('SAM3 remote completed', flush=True)\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


class RemoteSam3Worker(QObject):
    progress = pyqtSignal(object)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, request: RemoteSam3Request) -> None:
        super().__init__()
        self.request = request
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _raise_if_stop_requested(self) -> None:
        if self._stop_requested:
            raise WorkerStoppedError()

    def run(self) -> None:
        try:
            payload = self._run_remote()
            self.finished.emit(payload)
        except WorkerStoppedError:
            self.finished.emit({"stopped": True, "output_dir": self.request.output_dir, "preview_items": []})
        except Exception as exc:
            self.failed.emit(str(exc))

    def _split_remote_command_parts(self, command_text: str, is_remote_windows: bool) -> list[str]:
        raw = str(command_text).strip()
        if not raw:
            return []
        try:
            return shlex.split(raw, posix=(not is_remote_windows))
        except Exception:
            return [raw]


    def _remote_windows_python_env_paths(self, command_parts: Sequence[str]) -> list[str]:
        if not command_parts:
            return []

        exe = str(command_parts[0]).strip().strip('"').replace("\\", "/")
        if not re.match(r"^[A-Za-z]:/.+/python\.exe$", exe, flags=re.IGNORECASE):
            return []

        env_root = exe.rsplit("/", 1)[0]
        candidates = [
            env_root,
            f"{env_root}/Library/bin",
            f"{env_root}/Library/usr/bin",
            f"{env_root}/Library/mingw-w64/bin",
            f"{env_root}/Scripts",
        ]

        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            win_path = candidate.replace("/", "\\").rstrip("\\")
            key = win_path.lower()
            if not win_path or key in seen:
                continue
            seen.add(key)
            ordered.append(win_path)
        return ordered

    def _quote_powershell_arg(self, value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    def _build_remote_shell_cmd(self, command_parts: Sequence[str], is_remote_windows: bool) -> str:
        normalized = [str(part) for part in command_parts if str(part).strip()]
        if is_remote_windows:
            env_paths = self._remote_windows_python_env_paths(normalized)
            if not env_paths:
                return subprocess.list2cmdline(normalized)

            quoted_paths = ", ".join(self._quote_powershell_arg(path) for path in env_paths)
            quoted_args = " ".join(self._quote_powershell_arg(part) for part in normalized)
            script = (
                "$env:KMP_DUPLICATE_LIB_OK='TRUE'; "
                "$prefixes=@(" + quoted_paths + "); "
                "$filtered=@(); "
                "foreach ($item in ($env:PATH -split ';')) { "
                "$trimmed=[string]$item; "
                "if ([string]::IsNullOrWhiteSpace($trimmed)) { continue }; "
                "$trimmed=$trimmed.Trim(); "
                "$lower=$trimmed.ToLowerInvariant(); "
                "if ($lower -like '*\\anaconda3*' -or "
                "$lower -like '*\\miniconda3*' -or "
                "$lower -like '*\\condabin*' -or "
                "$lower -like '*\\conda\\envs\\*' -or "
                "$lower -like '*\\conda_envs\\*') { continue }; "
                "$filtered += $trimmed; "
                "}; "
                "$env:PATH=((@($prefixes) + @($filtered)) | Where-Object { $_ } | Select-Object -Unique) -join ';'; "
                f"& {quoted_args}"
            )
            return subprocess.list2cmdline(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script]
            )
        return " ".join(shlex.quote(part) for part in normalized)



    def _python_candidates(self, cfg: RemoteSam3Config, is_remote_windows: bool) -> list[str]:
        candidates: list[str] = []
        preferred = str(cfg.python_cmd).strip()
        if preferred:
            candidates.append(preferred)
        if is_remote_windows:
            user_home = f"C:/Users/{cfg.ssh_user}"
            fallback = [
                "G:/conda/envs/PJ_310_LLM_SAM3/python.exe",
                "G:/conda/envs/PJ_310_SAM3/python.exe",
                "G:/conda/envs/PJ_310_LLM/python.exe",
                "G:/conda_envs/PJ_310_SAM3/python.exe",
                "G:/conda_envs/PJ_310_LLM_SAM3/python.exe",
                "G:/conda_envs/PJ_310_LLM/python.exe",
                "G:/anaconda3/envs/PJ_310_SAM3/python.exe",
                "G:/miniconda3/envs/PJ_310_SAM3/python.exe",
                "python",
                "py -3",
                "py",
                f"{user_home}/anaconda3/envs/PJ_310_SAM3/python.exe",
                f"{user_home}/anaconda3/envs/PJ_310_LLM_SAM3/python.exe",
                f"{user_home}/anaconda3/envs/PJ_310_LLM/python.exe",
                f"{user_home}/anaconda3/python.exe",
                f"{user_home}/miniconda3/envs/PJ_310_SAM3/python.exe",
                f"{user_home}/miniconda3/envs/PJ_310_LLM_SAM3/python.exe",
                f"{user_home}/miniconda3/python.exe",
                "C:/conda_envs/PJ_310_SAM3/python.exe",
                "C:/conda_envs/PJ_310_LLM_SAM3/python.exe",
                "D:/conda_envs/PJ_310_SAM3/python.exe",
            ]
        else:
            fallback = ["python3", "python"]
        for candidate in fallback:
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _discover_remote_python_candidates(self, client: Any, is_remote_windows: bool) -> list[str]:
        if not is_remote_windows:
            return []
        discovered: list[str] = []
        probe_cmds = [
            "cmd /c where python",
            "cmd /c where py",
            r'cmd /c for /d %i in (G:\conda\envs\*) do @if exist "%i\python.exe" echo %i\python.exe',
            r'cmd /c for /d %i in (G:\conda_envs\*) do @if exist "%i\python.exe" echo %i\python.exe',
            r'cmd /c for /d %i in (C:\conda_envs\*) do @if exist "%i\python.exe" echo %i\python.exe',
            r'cmd /c for /d %i in (D:\conda_envs\*) do @if exist "%i\python.exe" echo %i\python.exe',
            r'cmd /c for /d %i in (G:\anaconda3\envs\*) do @if exist "%i\python.exe" echo %i\python.exe',
            r'cmd /c for /d %i in (G:\miniconda3\envs\*) do @if exist "%i\python.exe" echo %i\python.exe',
        ]
        for probe_cmd in probe_cmds:
            try:
                _stdin, stdout, _stderr = client.exec_command(probe_cmd)
                status = int(stdout.channel.recv_exit_status())
                output = _decode_remote_output(stdout.read(), is_remote_windows)
            except Exception:
                continue
            if status != 0:
                continue
            for raw_line in output.splitlines():
                candidate = _normalize_discovered_windows_python_candidate(raw_line)
                if not candidate:
                    continue
                if candidate and candidate not in discovered:
                    discovered.append(candidate)
        return discovered

    def _select_remote_python(self, client: Any, cfg: RemoteSam3Config, is_remote_windows: bool) -> tuple[str, list[str], dict[str, Any]]:
        probe_prefix = "__SAM3_PYTHON__="
        require_cuda = str(cfg.device or "").strip().lower().startswith("cuda")
        probe_code = (
            "import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); "
            "import importlib, json, sys; "
            "mods = {}; "
            "names = ['numpy', 'cv2', 'torch', 'PIL', 'tqdm', 'timm', 'ftfy', 'iopath', 'huggingface_hub', 'supervision']; "
            "[mods.setdefault(n, getattr(__import__(n), '__version__', 'ok')) for n in ['numpy', 'cv2', 'torch', 'PIL', 'tqdm', 'timm']]; [mods.setdefault(n, '') for n in ['ftfy', 'iopath', 'huggingface_hub', 'supervision']]; "
            "import torch; "
            "payload = {"
            "'executable': sys.executable, "
            "'python': sys.version.split()[0], "
            "'numpy': mods.get('numpy', ''), "
            "'cv2': mods.get('cv2', ''), "
            "'torch': mods.get('torch', ''), "
            "'supervision': mods.get('supervision', ''), "
            "'cuda_available': bool(torch.cuda.is_available()), "
            "'cuda_device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0, "
            "'cuda_device_name': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else '', "
            "'pillow': mods.get('PIL', ''), "
            "'tqdm': mods.get('tqdm', ''), "
            "'timm': mods.get('timm', ''), "
            "'ftfy': mods.get('ftfy', ''), "
            "'iopath': mods.get('iopath', ''), "
            "'huggingface_hub': mods.get('huggingface_hub', '')"
            "}; "
            "print('__SAM3_PYTHON__=' + json.dumps(payload, ensure_ascii=True))"
        )
        failures: list[str] = []
        candidate_list = self._python_candidates(cfg, is_remote_windows)
        for discovered in self._discover_remote_python_candidates(client, is_remote_windows):
            if discovered not in candidate_list:
                candidate_list.append(discovered)
        for candidate in candidate_list:
            parts = self._split_remote_command_parts(candidate, is_remote_windows)
            if not parts:
                continue
            shell_cmd = self._build_remote_shell_cmd([*parts, "-c", probe_code], is_remote_windows)
            try:
                _stdin, stdout, stderr = client.exec_command(shell_cmd)
                status = int(stdout.channel.recv_exit_status())
                stdout_text = _decode_remote_output(stdout.read(), is_remote_windows)
                stderr_text = _decode_remote_output(stderr.read(), is_remote_windows)
            except Exception:
                failures.append(f"- {candidate}: 명령 실행 확인에 실패했습니다.")
                continue
            if status != 0:
                tail = [line.strip() for line in (stderr_text or stdout_text).splitlines() if line.strip()]
                failures.append(f"- {candidate}: {(tail[-1] if tail else f'exit code {status}')}")
                continue
            for raw_line in stdout_text.splitlines():
                line = str(raw_line).strip()
                if not line.startswith(probe_prefix):
                    continue
                payload = json.loads(line[len(probe_prefix) :].strip())
                if isinstance(payload, Mapping):
                    if require_cuda and (not bool(payload.get("cuda_available", False))):
                        failures.append(f"- {candidate}: torch cuda unavailable")
                        break
                    return candidate, parts, dict(payload)
        raise RuntimeError(
            "원격 Python을 찾지 못했습니다. "
            "numpy, cv2, torch, pillow, tqdm를 import할 수 있는 Python 경로를 "
            "NEW_OBJECT_REMOTE_PYTHON_CMD로 지정하세요.\n"
            + "\n".join(failures[-12:])
        )

    def _remote_sam3_root_candidates(self, cfg: RemoteSam3Config) -> list[str]:
        root = str(cfg.remote_work_root).replace("\\", "/").rstrip("/")
        parent = root.rsplit("/", 1)[0] if "/" in root else root
        grand_parent = parent.rsplit("/", 1)[0] if "/" in parent else parent
        candidates = [
            str(cfg.remote_sam3_root).strip(),
            _DEFAULT_REMOTE_SAM3_ROOT,
            f"{root}/sam3",
            f"{parent}/sam3",
            f"{grand_parent}/sam3",
            "G:/models/sam3",
            "G:/sam3",
            "C:/sam3",
            "D:/sam3",
            "G:/KDT10_3_1team_KLIK/sam3",
            "G:/KDT10_3_1team_KLIK/0_Program_/sam3",
        ]
        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            normalized = str(candidate).replace("\\", "/").strip().rstrip("/")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    def _select_remote_sam3_root(self, sftp: Any, cfg: RemoteSam3Config) -> str:
        checked: list[str] = []
        for candidate in self._remote_sam3_root_candidates(cfg):
            checked.append(candidate)
            checkpoint_paths = [f"{candidate}/checkpoints/sam3.pt", f"{candidate}/sam3/checkpoints/sam3.pt"]
            bpe_paths = [
                f"{candidate}/assets/bpe_simple_vocab_16e6.txt.gz",
                f"{candidate}/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            ]
            has_checkpoint = any(_sftp_path_exists(sftp, path) for path in checkpoint_paths)
            has_bpe = any(_sftp_path_exists(sftp, path) for path in bpe_paths)
            if has_checkpoint and has_bpe:
                return candidate
        raise RuntimeError(
            "Remote sam3 root not found. "
            "Set NEW_OBJECT_REMOTE_SAM3_ROOT to the server path that contains checkpoints/sam3.pt and assets/bpe_simple_vocab_16e6.txt.gz.\n"
            + "\n".join(f"- {item}" for item in checked[-8:])
        )

    def _remote_work_root_candidates(self, cfg: RemoteSam3Config) -> list[str]:
        candidates = [
            str(cfg.remote_work_root).strip(),
            str(os.getenv("TRAIN_REMOTE_WORKDIR", "")).strip(),
            _DEFAULT_REMOTE_PROGRAM_ROOT,
            f"{_DEFAULT_REMOTE_PROGRAM_ROOT}/{APP_ROOT.name}",
            "G:/KDT10_3_1team_KLIK/0_Program_",
        ]
        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            normalized = str(candidate).replace("\\", "/").strip().rstrip("/")
            if (not normalized) or (not _is_absolute_remote_path(normalized)) or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    def _select_remote_work_root(self, sftp: Any, cfg: RemoteSam3Config) -> str:
        checked: list[str] = []
        for candidate in self._remote_work_root_candidates(cfg):
            checked.append(candidate)
            if _sftp_path_exists(sftp, candidate):
                return candidate
        if checked:
            return checked[0]
        raise RuntimeError(
            "Remote work root not found. "
            "Set NEW_OBJECT_REMOTE_WORKDIR to the server project root.\n"
            + "\n".join(f"- {item}" for item in checked[-8:])
        )

    def _extract_marker_payload(self, line_text: str, marker_prefix: str) -> dict[str, Any] | None:
        idx = str(line_text).find(marker_prefix)
        if idx < 0:
            return None
        payload_text = str(line_text)[idx + len(marker_prefix) :].strip()
        if not payload_text:
            return None
        try:
            payload = json.loads(payload_text)
        except Exception:
            return None
        return dict(payload) if isinstance(payload, Mapping) else None

    def _cleanup_remote_job(self, client: Any, remote_job_root: str, is_remote_windows: bool) -> None:
        try:
            if is_remote_windows:
                cleanup_path = str(remote_job_root).replace("/", "\\")
                client.exec_command(f'cmd /c rmdir /s /q "{cleanup_path}"')
            else:
                client.exec_command(f"rm -rf {shlex.quote(str(remote_job_root))}")
        except Exception:
            pass

    def _run_remote(self) -> dict[str, Any]:
        try:
            import paramiko  # type: ignore
        except Exception as exc:
            raise RuntimeError("Remote SAM3 execution requires paramiko. Install it with: pip install paramiko") from exc

        cfg = RemoteSam3Config.from_env()
        if not cfg.enabled:
            raise RuntimeError("Remote SAM3 mode is disabled. Set NEW_OBJECT_REMOTE_ENABLE=1.")
        missing = missing_remote_ssh_fields()
        if missing:
            raise RuntimeError(
                "Remote server connection is not configured for SAM3. "
                f"Missing fields: {', '.join(missing)}. "
                "Set them in config.local.json or environment variables."
            )
        if self._stop_requested:
            raise WorkerStoppedError()

        local_project_root = resolve_local_sam3_project_root()
        local_config_path = resolve_local_sam3_config_path(local_project_root)
        output_dir = Path(self.request.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        remote_cache_dir = output_dir / "_remote_run"
        if remote_cache_dir.exists():
            shutil.rmtree(remote_cache_dir, ignore_errors=True)
        prompt_variants = _build_prompt_variants(
            self.request.prompt_text,
            self.request.class_name,
            ranked_candidates=self.request.prompt_candidates,
        )

        safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.request.experiment_id.strip() or "sam3_run")
        remote_job_name = f"{safe_run_name}_{int(time.time())}_{os.getpid()}"
        remote_program_root = ""
        remote_job_root = ""
        remote_project_root = ""
        remote_input_dir = ""
        remote_runs_root = ""
        remote_run_dir = ""
        remote_video_path = ""
        remote_config_path = ""
        remote_script_path = ""
        remote_runtime_cache_root = ""
        remote_input_cache_root = ""
        should_render_local_preview = env_flag("NEW_OBJECT_RENDER_LOCAL_PREVIEW_VIDEO", False)
        stage_timings: dict[str, float] = {}

        self.log_message.emit(
            f"원격 대상: {cfg.ssh_user}@{cfg.ssh_host}:{cfg.ssh_port}, device={cfg.device or 'auto'}"
        )
        self.log_message.emit("원격 동기화: sam_3 프로젝트와 영상을 업로드합니다...")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        sftp = None
        is_remote_windows = True
        captured_tail = ""
        remote_summary: dict[str, Any] = {}
        try:
            client.connect(
                hostname=cfg.ssh_host,
                port=int(cfg.ssh_port),
                username=cfg.ssh_user,
                password=cfg.ssh_password,
                timeout=max(3.0, float(cfg.connect_timeout_sec)),
                auth_timeout=max(3.0, float(cfg.connect_timeout_sec)),
                banner_timeout=max(5.0, float(cfg.connect_timeout_sec)),
                look_for_keys=False,
                allow_agent=False,
            )
            sftp = client.open_sftp()
            try:
                _stdin, stdout, _stderr = client.exec_command("uname")
                is_remote_windows = int(stdout.channel.recv_exit_status()) != 0
            except Exception:
                is_remote_windows = True

            remote_program_root = self._select_remote_work_root(sftp, cfg)
            cfg.remote_work_root = remote_program_root
            remote_runtime_cache_root = f"{remote_program_root}/.sam3_runtime_cache"
            remote_input_cache_root = f"{remote_program_root}/.sam3_input_cache"
            remote_job_root = f"{remote_program_root}/.remote_jobs/{remote_job_name}"
            remote_input_dir = f"{remote_job_root}/input"
            remote_runs_root = f"{remote_job_root}/runs"
            remote_run_dir = f"{remote_runs_root}/{self.request.experiment_id}"
            remote_video_path = f"{remote_input_dir}/{Path(self.request.video_path).name}"
            remote_config_path = f"{remote_job_root}/runtime_config.json"
            remote_script_path = f"{remote_job_root}/remote_sam3.py"
            selected_python_cmd, python_cmd_parts, python_meta = self._select_remote_python(client, cfg, is_remote_windows)
            remote_sam3_root = self._select_remote_sam3_root(sftp, cfg)
            self.log_message.emit(f"원격 작업 루트: {remote_program_root}")
            self.log_message.emit(
                "원격 Python 선택: "
                f"{selected_python_cmd} "
                f"(python={python_meta.get('python') or 'unknown'}, "
                f"torch={python_meta.get('torch') or 'unknown'}, "
                f"cv2={python_meta.get('cv2') or 'unknown'})"
            )
            self.log_message.emit(f"원격 sam3 경로: {remote_sam3_root}")
            self.log_message.emit(f"적용 프롬프트: {', '.join(prompt_variants)}")

            self._raise_if_stop_requested()
            _sftp_mkdir_p(sftp, remote_input_dir)
            upload_started = time.perf_counter()
            remote_project_root, reused_runtime = _ensure_remote_cached_project(
                sftp,
                local_root=local_project_root,
                remote_cache_root=remote_runtime_cache_root,
                should_stop=lambda: self._stop_requested,
            )
            stage_timings["runtime_sync_sec"] = round(max(0.0, time.perf_counter() - upload_started), 4)
            self.log_message.emit(
                "[INFO] remote runtime "
                + ("cache hit" if reused_runtime else "uploaded")
                + f": {Path(remote_project_root).name} ({stage_timings['runtime_sync_sec']:.2f}s)"
            )
            self._raise_if_stop_requested()
            video_upload_started = time.perf_counter()
            remote_video_path, reused_video = _ensure_remote_cached_input_file(
                sftp,
                local_path=Path(self.request.video_path).resolve(),
                remote_cache_root=remote_input_cache_root,
                should_stop=lambda: self._stop_requested,
            )
            stage_timings["video_upload_sec"] = round(max(0.0, time.perf_counter() - video_upload_started), 4)
            self.log_message.emit(
                "[INFO] remote video "
                + ("cache hit" if reused_video else "uploaded")
                + f": {Path(remote_video_path).name} ({stage_timings['video_upload_sec']:.2f}s)"
            )
            self._raise_if_stop_requested()

            runtime_config_text = _build_runtime_config(
                local_config_path=local_config_path,
                remote_sam3_root=remote_sam3_root,
                remote_device=cfg.device,
                remote_runs_root=remote_runs_root,
                prompts=prompt_variants,
            )
            with sftp.file(remote_config_path, "wb") as fp:
                fp.write(runtime_config_text.encode("utf-8"))
            with sftp.file(remote_script_path, "wb") as fp:
                fp.write(_build_remote_script_text().encode("utf-8"))
            try:
                sftp.chmod(remote_script_path, 0o755)
            except Exception:
                pass
            self._raise_if_stop_requested()

            python_cmd = list(python_cmd_parts)
            if "-u" not in {str(part).strip().lower() for part in python_cmd}:
                python_cmd.append("-u")
            remote_shell_cmd = self._build_remote_shell_cmd(
                [
                    *python_cmd,
                    remote_script_path,
                    "--project-root",
                    remote_project_root,
                    "--config",
                    remote_config_path,
                    "--video",
                    remote_video_path,
                    "--run-name",
                    self.request.experiment_id,
                ],
                is_remote_windows,
            )

            transport = client.get_transport()
            if transport is None or (not transport.is_active()):
                raise RuntimeError("SSH transport is not active")

            channel = transport.open_session()
            remote_exec_started = time.perf_counter()
            channel.exec_command(remote_shell_cmd)
            self.log_message.emit(f"원격 SAM3 실행 시작 ({'windows' if is_remote_windows else 'posix'} shell).")

            stdout_buffer = ""
            stderr_buffer = ""
            last_progress_frame = -1

            def _handle_remote_line(raw_line: str) -> None:
                nonlocal captured_tail, remote_summary, last_progress_frame
                line = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", str(raw_line)).strip()
                if not line:
                    return
                progress_payload = self._extract_marker_payload(line, SAM3_PROGRESS_PREFIX)
                if progress_payload is not None:
                    frames_processed = int(float(progress_payload.get("frames_processed", 0)))
                    if frames_processed <= last_progress_frame:
                        return
                    last_progress_frame = frames_processed
                    self.progress.emit(
                        {
                            "framesProcessed": frames_processed,
                            "newSamples": 0,
                            "numImages": 0,
                            "numLabels": 0,
                            "message": f"원격 SAM3 처리 프레임 수: {frames_processed}",
                            "level": "INFO",
                            "frameIndex": int(float(progress_payload.get("frame_idx", 0))),
                        }
                    )
                    return
                summary_payload = self._extract_marker_payload(line, SAM3_SUMMARY_PREFIX)
                if summary_payload is not None:
                    remote_summary = dict(summary_payload)
                    return
                captured_tail = (captured_tail + line + "\n")[-16000:]
                self.log_message.emit(f"터미널: {line}")

            def _consume_chunk(buffer: str, chunk: str) -> str:
                text = (buffer + str(chunk)).replace("\r", "\n")
                parts = text.split("\n")
                next_buffer = parts.pop() if parts else ""
                for part in parts:
                    _handle_remote_line(part)
                return next_buffer

            while True:
                if self._stop_requested:
                    try:
                        channel.close()
                    except Exception:
                        pass
                    raise WorkerStoppedError()

                has_data = False
                while channel.recv_ready():
                    chunk = _decode_remote_output(channel.recv(65536), is_remote_windows)
                    if not chunk:
                        break
                    has_data = True
                    stdout_buffer = _consume_chunk(stdout_buffer, chunk)
                while channel.recv_stderr_ready():
                    chunk = _decode_remote_output(channel.recv_stderr(65536), is_remote_windows)
                    if not chunk:
                        break
                    has_data = True
                    stderr_buffer = _consume_chunk(stderr_buffer, chunk)

                if channel.exit_status_ready() and (not channel.recv_ready()) and (not channel.recv_stderr_ready()):
                    break
                if not has_data:
                    time.sleep(0.1)

            if stdout_buffer.strip():
                _handle_remote_line(stdout_buffer)
            if stderr_buffer.strip():
                _handle_remote_line(stderr_buffer)

            exit_code = int(channel.recv_exit_status())
            stage_timings["remote_exec_sec"] = round(max(0.0, time.perf_counter() - remote_exec_started), 4)
            if exit_code != 0:
                tail = "\n".join(captured_tail.strip().splitlines()[-10:])
                raise RuntimeError(f"원격 SAM3가 코드 {exit_code}로 종료되었습니다. {tail}".strip())

            if sftp is None:
                raise RuntimeError("SFTP 세션을 열지 못했습니다.")
            if not _sftp_path_exists(sftp, remote_run_dir):
                raise RuntimeError(f"원격 실행 결과 폴더를 찾지 못했습니다: {remote_run_dir}")

            self.log_message.emit("원격 동기화: 실행 결과를 다운로드합니다...")
            download_started = time.perf_counter()
            downloaded_files = _sftp_download_selected_files(
                sftp,
                remote_run_dir,
                remote_cache_dir,
                ("summary.json", "tracks.csv", "preview.mp4"),
                should_stop=lambda: self._stop_requested,
            )
            stage_timings["result_download_sec"] = round(max(0.0, time.perf_counter() - download_started), 4)
            self._raise_if_stop_requested()
            self.log_message.emit(
                f"[INFO] remote results downloaded: {', '.join(downloaded_files) if downloaded_files else 'none'} "
                f"({stage_timings['result_download_sec']:.2f}s)"
            )
            self.log_message.emit("로컬 후처리 1/3: keep/hold/drop 결과를 정리합니다.")

            def _emit_local_postprocess_progress(done_rows: int, total_rows: int) -> None:
                if total_rows <= 0:
                    return
                self.progress.emit(
                    {
                        "framesProcessed": int(done_rows),
                        "newSamples": 0,
                        "numImages": 0,
                        "numLabels": 0,
                        "message": f"로컬 후처리 1/3: {int(done_rows)}/{int(total_rows)}",
                        "level": "INFO",
                    }
                )

            postprocess_started = time.perf_counter()
            preview_items, local_summary = build_preview_items_from_remote_run(
                output_dir=output_dir,
                video_path=self.request.video_path,
                experiment_id=self.request.experiment_id,
                progress_callback=_emit_local_postprocess_progress,
                should_stop=lambda: self._stop_requested,
            )
            stage_timings["local_postprocess_sec"] = round(max(0.0, time.perf_counter() - postprocess_started), 4)
            preview_video_local = str(local_summary.get("preview_video_local", "") or "").strip()
            if ((not preview_video_local) or (not Path(preview_video_local).is_file())) and (not should_render_local_preview):
                local_summary["preview_video_local"] = ""
                self.log_message.emit("[INFO] result preview video skipped for speed.")
            elif (not preview_video_local) or (not Path(preview_video_local).is_file()):
                self.log_message.emit("로컬 후처리 2/3: 결과 미리보기 영상을 생성합니다.")
                preview_render_started = time.perf_counter()
                preview_video_local = _render_preview_video_locally(
                    output_dir=output_dir,
                    preview_items=preview_items,
                    summary=local_summary,
                    video_path=self.request.video_path,
                )
                stage_timings["local_preview_render_sec"] = round(
                    max(0.0, time.perf_counter() - preview_render_started),
                    4,
                )
                if preview_video_local:
                    local_summary["preview_video_local"] = preview_video_local
                    self.log_message.emit("로컬 결과 비디오를 생성했습니다.")
            self.log_message.emit("로컬 후처리 3/3: export용 데이터셋과 임시 캐시를 준비합니다.")
            merged_summary = dict(remote_summary)
            merged_summary.update(local_summary)
            remote_summary = merged_summary
            counts = summarize_preview_items(preview_items)
            preview_cache_root = _preview_cache_root_for_experiment(self.request.experiment_id)

            def _emit_cache_progress(done_items: int, total_items: int) -> None:
                if total_items <= 0:
                    return
                self.progress.emit(
                    {
                        "framesProcessed": int(remote_summary.get("processed_frames", 0)),
                        "newSamples": int(done_items),
                        "numImages": 0,
                        "numLabels": 0,
                        "message": f"로컬 후처리 3/3: 캐시 저장 {int(done_items)}/{int(total_items)}",
                        "level": "INFO",
                    }
                )

            cache_write_started = time.perf_counter()
            cached_count = _persist_preview_items_to_cache(
                preview_cache_root,
                preview_items,
                video_path=self.request.video_path,
                progress_callback=_emit_cache_progress,
                should_stop=lambda: self._stop_requested,
            )
            stage_timings["preview_cache_write_sec"] = round(max(0.0, time.perf_counter() - cache_write_started), 4)
            remote_summary.update(stage_timings)
            timing_parts = [
                f"{name}={value:.2f}s"
                for name, value in (
                    ("runtime_sync", stage_timings.get("runtime_sync_sec", 0.0)),
                    ("video_sync", stage_timings.get("video_upload_sec", 0.0)),
                    ("remote_exec", stage_timings.get("remote_exec_sec", 0.0)),
                    ("result_download", stage_timings.get("result_download_sec", 0.0)),
                    ("local_postprocess", stage_timings.get("local_postprocess_sec", 0.0)),
                    ("local_preview_render", stage_timings.get("local_preview_render_sec", 0.0)),
                    ("preview_cache_write", stage_timings.get("preview_cache_write_sec", 0.0)),
                )
                if value > 0
            ]
            if timing_parts:
                self.log_message.emit(f"[INFO] timing summary: {', '.join(timing_parts)}")
            keep_items = int(remote_summary.get("keep_items", 0) or 0)
            hold_items = int(remote_summary.get("hold_items", 0) or 0)
            drop_items = int(remote_summary.get("drop_items", 0) or 0)
            self.progress.emit(
                {
                    "framesProcessed": int(remote_summary.get("processed_frames", 0)),
                    "newSamples": max(0, keep_items + hold_items),
                    "numImages": int(counts.get("num_images", 0)),
                    "numLabels": int(counts.get("num_labels", 0)),
                    "message": (
                        f"SAM3 완료: 프레임={int(remote_summary.get('processed_frames', 0))}, "
                        f"keep={keep_items}, hold={hold_items}, drop={drop_items}"
                    ),
                    "level": "INFO",
                }
            )
            return {
                "stopped": False,
                "output_dir": str(output_dir),
                "preview_items": [],
                "preview_cache_root": str(preview_cache_root),
                "preview_item_count": int(cached_count),
                "summary": remote_summary,
                "counts": counts,
                "preview_video_path": str(remote_summary.get("preview_video_local", "")).strip(),
            }
        finally:
            if sftp is not None:
                try:
                    sftp.close()
                except Exception:
                    pass
            if (not cfg.keep_remote_files) and client is not None:
                self._cleanup_remote_job(client, remote_job_root, is_remote_windows)
            try:
                client.close()
            except Exception:
                pass
