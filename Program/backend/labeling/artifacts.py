from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.paths import NEW_OBJECT_OUTPUT_ROOT


def prepare_output_dir(experiment_id: str) -> Path:
    output_dir = (NEW_OBJECT_OUTPUT_ROOT / str(experiment_id).strip()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_manifest(output_dir: str | Path, *, config: Any, progress: Any) -> Path:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config) if is_dataclass(config) else dict(config or {}),
        "progress": asdict(progress) if is_dataclass(progress) else dict(progress or {}),
    }
    target = root / "manifest.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_config_snapshot(output_dir: str | Path, *, config: Any) -> Path:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = asdict(config) if is_dataclass(config) else dict(config or {})
    lines = [
        f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"class_name: {payload.get('className', '-')}",
        f"image_path: {payload.get('imagePath', '-')}",
        f"video_path: {payload.get('videoPath', '-')}",
        f"prompt_mode: {payload.get('promptMode', '-')}",
        f"prompt_text: {payload.get('promptText', '-')}",
        f"experiment_id: {payload.get('experimentId', '-')}",
    ]
    target = root / "config.yaml"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def export_output_tree(output_dir: str | Path, export_dir: str | Path) -> Path:
    source_root = Path(output_dir).resolve()
    target_root = Path(export_dir).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"outputDir not found: {source_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    destination = target_root / source_root.name
    shutil.copytree(source_root, destination, dirs_exist_ok=True)
    return destination
