from __future__ import annotations

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_ROOT = APP_ROOT.parent

APP_ASSETS_ROOT = APP_ROOT / "assets"
STORAGE_ROOT = PROGRAM_ROOT
STORAGE_ASSETS_ROOT = STORAGE_ROOT / "assets"

DATASET_SAVE_DIR = STORAGE_ASSETS_ROOT / "dataset_save_dir"
MERGED_DATASET_SAVE_DIR = DATASET_SAVE_DIR / "merged_dataset_save_dir"
CROP_SAVE_BASE_DIR = STORAGE_ASSETS_ROOT / "crop_save_dir"
TEAM_MODEL_DIR = STORAGE_ASSETS_ROOT / "models"
TRAIN_YOLO_MODELS_DIR = TEAM_MODEL_DIR / "YOLO_models"
TRAIN_RTDETR_MODELS_DIR = TEAM_MODEL_DIR / "RT-DETR_models"
TRAIN_RUNS_DIR = STORAGE_ROOT / "runs" / "train"
PREVIEW_CACHE_BASE = STORAGE_ROOT / "runtime_preview_cache"
NEW_OBJECT_OUTPUT_ROOT = PREVIEW_CACHE_BASE / "new_object_runs"


def model_storage_dir_for_name(name: str, engine_key: str | None = None) -> Path:
    text = str(name or "").strip().casefold()
    engine = str(engine_key or "").strip().casefold()
    if engine == "rtdetr" or "rtdetr" in text:
        return TRAIN_RTDETR_MODELS_DIR
    return TRAIN_YOLO_MODELS_DIR


def normalize_root_model_artifacts() -> None:
    for path in APP_ROOT.glob("*.pt"):
        name = str(path.name).strip()
        lowered = name.casefold()
        if not (lowered.startswith("yolo") or "rtdetr" in lowered):
            continue
        target_dir = model_storage_dir_for_name(name)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / name
        try:
            if target_path.exists():
                target_path.unlink()
            path.replace(target_path)
        except Exception:
            continue


def ensure_storage_directories() -> None:
    for path in (
        STORAGE_ASSETS_ROOT,
        DATASET_SAVE_DIR,
        MERGED_DATASET_SAVE_DIR,
        NEW_OBJECT_OUTPUT_ROOT,
        CROP_SAVE_BASE_DIR,
        TEAM_MODEL_DIR,
        TRAIN_YOLO_MODELS_DIR,
        TRAIN_RTDETR_MODELS_DIR,
        TRAIN_RUNS_DIR,
        PREVIEW_CACHE_BASE,
    ):
        path.mkdir(parents=True, exist_ok=True)
    normalize_root_model_artifacts()
