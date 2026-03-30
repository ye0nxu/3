from __future__ import annotations

import contextlib
import csv
import json
import logging
import math
import os
import queue
import random
import re
import shlex
import shutil
import stat
import subprocess
import sys
import textwrap
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from config import apply_remote_env_defaults
from utils.env import env_flag as _env_flag, env_int as _env_int
try:
    from core.dataset import BoxAnnotation, FrameAnnotation
except ModuleNotFoundError:
    PROJECT_DIR = Path(__file__).resolve().parents[2]
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from core.dataset import BoxAnnotation, FrameAnnotation

from backend.pipelines.preview_postprocess import postprocess_preview_items
from backend.storage.remote_storage import remote_storage_enabled, sync_local_tree_to_remote
from core.paths import (
    CROP_SAVE_BASE_DIR,
    DATASET_SAVE_DIR,
    MERGED_DATASET_SAVE_DIR,
    TEAM_MODEL_DIR,
    TRAIN_RTDETR_MODELS_DIR,
    TRAIN_RUNS_DIR,
    TRAIN_YOLO_MODELS_DIR,
    ensure_storage_directories,
    model_storage_dir_for_name,
)
from app.studio.config import (
    DEFAULT_CLASS_NAMES,
    TEAM_MODEL_PATH,
    TRAIN_DEFAULT_FREEZE,
    TRAIN_DEFAULT_LR0,
    TRAIN_REPLAY_RATIO_DEFAULT,
    TRAIN_RETRAIN_SEED_DEFAULT,
    TRAIN_STAGE1_EPOCHS_DEFAULT,
    TRAIN_STAGE2_EPOCHS_DEFAULT,
    TRAIN_STAGE2_LR_FACTOR_DEFAULT,
    TRAIN_STAGE_UNFREEZE_BACKBONE_LAST,
    TRAIN_STAGE_UNFREEZE_CHOICES,
    TRAIN_STAGE_UNFREEZE_NECK_ONLY,
)
from core.models import (
    ExportRunSummary,
    PreviewThumbnail,
    ProgressEvent,
    WorkerOutput,
    WorkerStoppedError,
)
from app.studio.runtime import (
    FilterConfig,
    SampleCandidate,
    SampleFilterEngine,
    TeamTrackObject,
    TqdmFormatter,
    ULTRALYTICS_VERSION,
    UltralyticsRTDETR,
    UltralyticsYOLO,
    cv2,
    torch,
    yaml,
)
from app.studio.utils import (
    _build_component_from_yaml,
    _build_unique_path,
    _collect_split_counts_from_yaml,
    _dedupe_preserve_order,
    _guess_label_path_for_image,
    _load_yaml_dict,
    _metric_to_unit_interval,
    _normalize_split_entries,
    _parse_version_tuple,
    _parse_names_from_yaml_payload,
    _read_image_list_file,
    _resolve_dataset_root_from_yaml_payload,
    _resolve_managed_model_source,
    _resolve_split_images,
    _scan_images_in_directory,
    _sync_ultralytics_datasets_dir,
    _temporary_working_directory,
    _try_autofix_data_yaml_path,
    _try_autofix_data_yaml_splits,
    _write_replay_dataset_readme,
    build_dataset_folder_name,
    collect_original_components_from_yaml,
    extract_training_metrics_and_losses,
    sanitize_class_name,
    write_provenance_readme,
)


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


@dataclass(slots=True)

class RemoteTrainConfig:
    enabled: bool
    required: bool
    ssh_host: str
    ssh_port: int
    ssh_user: str
    ssh_password: str
    python_cmd: str
    remote_work_root: str
    device: str
    connect_timeout_sec: float
    keep_remote_files: bool

    @classmethod
    def from_env(cls) -> "RemoteTrainConfig":
        apply_remote_env_defaults()
        return cls(
            enabled=_env_flag("TRAIN_REMOTE_ENABLE", _env_flag("LLM_REMOTE_ENABLE", True)),
            required=_env_flag("TRAIN_REMOTE_REQUIRED", True),
            ssh_host=str(os.getenv("TRAIN_REMOTE_SSH_HOST", os.getenv("LLM_REMOTE_SSH_HOST", ""))).strip(),
            ssh_port=_env_int("TRAIN_REMOTE_SSH_PORT", _env_int("LLM_REMOTE_SSH_PORT", 8875)),
            ssh_user=str(os.getenv("TRAIN_REMOTE_SSH_USER", os.getenv("LLM_REMOTE_SSH_USER", ""))).strip(),
            ssh_password=str(os.getenv("TRAIN_REMOTE_SSH_PASSWORD", os.getenv("LLM_REMOTE_SSH_PASSWORD", ""))),
            python_cmd=str(os.getenv("TRAIN_REMOTE_PYTHON_CMD", "python")).strip() or "python",
            remote_work_root=str(os.getenv("TRAIN_REMOTE_WORKDIR", "auto_label_training")).strip() or "auto_label_training",
            device=str(os.getenv("TRAIN_REMOTE_DEVICE", "0")).strip(),
            connect_timeout_sec=max(3.0, _env_float("TRAIN_REMOTE_CONNECT_TIMEOUT", 15.0)),
            keep_remote_files=_env_flag("TRAIN_REMOTE_KEEP_FILES", False),
        )


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


def _sftp_upload_tree(sftp: Any, local_root: Path, remote_root: str) -> None:
    local_root_resolved = Path(local_root).resolve()
    remote_root_norm = str(remote_root).replace("\\", "/")
    _sftp_mkdir_p(sftp, remote_root_norm)
    for root, _dirs, files in os.walk(local_root_resolved):
        root_path = Path(root)
        rel = root_path.relative_to(local_root_resolved)
        remote_dir = remote_root_norm if str(rel) in {"", "."} else f"{remote_root_norm.rstrip('/')}/{rel.as_posix()}"
        _sftp_mkdir_p(sftp, remote_dir)
        for name in files:
            local_file = root_path / name
            remote_file = f"{remote_dir.rstrip('/')}/{name}"
            sftp.put(str(local_file), remote_file)


def _sftp_download_tree(sftp: Any, remote_root: str, local_root: Path) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(str(remote_root)):
        remote_path = f"{str(remote_root).rstrip('/')}/{entry.filename}"
        local_path = local_root / str(entry.filename)
        if stat.S_ISDIR(int(entry.st_mode)):
            _sftp_download_tree(sftp, remote_path, local_path)
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_path, str(local_path))


def _normalize_split_value_for_remote(value: Any, local_dataset_root: Path) -> Any:
    if isinstance(value, str):
        text = str(value).strip()
        if not text:
            return text
        candidate = Path(text)
        is_windows_abs = bool(re.match(r"^[A-Za-z]:[\\/]", text))
        if candidate.is_absolute() or is_windows_abs:
            try:
                rel = candidate.resolve().relative_to(local_dataset_root.resolve())
                return rel.as_posix()
            except Exception:
                return text.replace("\\", "/")
        return text.replace("\\", "/")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_split_value_for_remote(item, local_dataset_root) for item in value]
    return value


def _build_remote_data_yaml_text(
    local_data_yaml: Path,
    remote_dataset_root: str,
    local_dataset_root: Path | None = None,
) -> str:
    if yaml is None:
        raise RuntimeError("PyYAML is required for remote training mode.")
    payload = dict(_load_yaml_dict(Path(local_data_yaml)))
    if not payload:
        raise RuntimeError(f"Invalid data.yaml for remote training: {local_data_yaml}")
    if local_dataset_root is None:
        local_dataset_root = _resolve_dataset_root_from_yaml_payload(Path(local_data_yaml), payload)
    else:
        local_dataset_root = Path(local_dataset_root).resolve()
    payload["path"] = str(remote_dataset_root).replace("\\", "/")
    for key in ("train", "val", "test"):
        if key in payload:
            payload[key] = _normalize_split_value_for_remote(payload.get(key), local_dataset_root)
    return str(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


REMOTE_TRAIN_METRIC_PREFIX = "__TRAIN_METRIC__="
REMOTE_TRAIN_BATCH_PREFIX = "__TRAIN_BATCH__="
REMOTE_TRAIN_SAVE_DIR_PREFIX = "REMOTE_SAVE_DIR="


def _build_remote_train_script_text() -> str:
    return textwrap.dedent(
        f"""
        from __future__ import annotations
        import argparse
        import csv
        import json
        import threading
        from pathlib import Path

        import numpy as np

        REMOTE_TRAIN_METRIC_PREFIX = "{REMOTE_TRAIN_METRIC_PREFIX}"
        REMOTE_TRAIN_BATCH_PREFIX = "{REMOTE_TRAIN_BATCH_PREFIX}"
        REMOTE_TRAIN_SAVE_DIR_PREFIX = "{REMOTE_TRAIN_SAVE_DIR_PREFIX}"

        def _parse() -> argparse.Namespace:
            p = argparse.ArgumentParser(description="Run ultralytics train on remote GPU")
            p.add_argument("--engine", choices=["yolo", "rtdetr"], required=True)
            p.add_argument("--model-source", required=True)
            p.add_argument("--data-yaml", required=True)
            p.add_argument("--task", default="detect")
            p.add_argument("--epochs", type=int, required=True)
            p.add_argument("--imgsz", type=int, required=True)
            p.add_argument("--batch", type=int, required=True)
            p.add_argument("--patience", type=int, required=True)
            p.add_argument("--project", required=True)
            p.add_argument("--name", required=True)
            p.add_argument("--lr0", type=float, default=None)
            p.add_argument("--freeze-json", default="")
            p.add_argument("--device", default="")
            p.add_argument("--save-dir-file", default="")
            p.add_argument("--extra-params-json", default="")
            return p.parse_args()

        def _safe_float(value: object, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return float(default)

        def _extract_loss_components(trainer, engine: str) -> tuple[float, float, float]:
            box_loss = 0.0
            cls_loss = 0.0
            dfl_loss = 0.0

            label_loss_items = getattr(trainer, "label_loss_items", None)
            tloss = getattr(trainer, "tloss", None)
            if callable(label_loss_items):
                try:
                    labeled = label_loss_items(tloss, prefix="train")
                    if isinstance(labeled, dict):
                        box_loss = _safe_float(
                            labeled.get(
                                "train/box_loss",
                                labeled.get("box_loss", labeled.get("train/giou_loss", labeled.get("giou_loss", box_loss))),
                            )
                        )
                        cls_loss = _safe_float(labeled.get("train/cls_loss", labeled.get("cls_loss", cls_loss)))
                        dfl_loss = _safe_float(
                            labeled.get(
                                "train/dfl_loss",
                                labeled.get("dfl_loss", labeled.get("train/l1_loss", labeled.get("l1_loss", dfl_loss))),
                            )
                        )
                except Exception:
                    pass

            metrics_dict = getattr(trainer, "metrics", None)
            if isinstance(metrics_dict, dict):
                if engine == "rtdetr":
                    key_map = (
                        ("train/giou_loss", "box"),
                        ("giou_loss", "box"),
                        ("train/box_loss", "box"),
                        ("box_loss", "box"),
                        ("train/cls_loss", "cls"),
                        ("cls_loss", "cls"),
                        ("train/l1_loss", "dfl"),
                        ("l1_loss", "dfl"),
                        ("train/dfl_loss", "dfl"),
                        ("dfl_loss", "dfl"),
                    )
                else:
                    key_map = (
                        ("train/box_loss", "box"),
                        ("box_loss", "box"),
                        ("train/cls_loss", "cls"),
                        ("cls_loss", "cls"),
                        ("train/dfl_loss", "dfl"),
                        ("dfl_loss", "dfl"),
                    )
                for key, target in key_map:
                    if key not in metrics_dict:
                        continue
                    value = max(0.0, _safe_float(metrics_dict.get(key), 0.0))
                    if not np.isfinite(value):
                        continue
                    if target == "box":
                        box_loss = value
                    elif target == "cls":
                        cls_loss = value
                    else:
                        dfl_loss = value

            if box_loss <= 0.0 and cls_loss <= 0.0 and dfl_loss <= 0.0:
                try:
                    arr = np.asarray(getattr(trainer, "loss_items", None), dtype=np.float32).reshape(-1)
                    if arr.size >= 3:
                        box_loss = max(0.0, float(arr[0]))
                        cls_loss = max(0.0, float(arr[1]))
                        dfl_loss = max(0.0, float(arr[2]))
                except Exception:
                    pass

            return float(max(0.0, box_loss)), float(max(0.0, cls_loss)), float(max(0.0, dfl_loss))

        def _extract_accuracy_values(trainer) -> tuple[float, float]:
            map50_value = 0.0
            map50_95_value = 0.0

            validator = getattr(trainer, "validator", None)
            metrics_obj = getattr(validator, "metrics", None)
            for name in ("map50", "mAP50", "box_map50"):
                value = getattr(metrics_obj, name, None)
                if value is not None:
                    map50_value = _safe_float(value, map50_value)
                    if map50_value > 0.0:
                        break
            for name in ("map", "mAP50_95", "mAP50-95", "box_map"):
                value = getattr(metrics_obj, name, None)
                if value is not None:
                    map50_95_value = _safe_float(value, map50_95_value)
                    if map50_95_value > 0.0:
                        break

            metrics_dict = getattr(trainer, "metrics", None)
            if isinstance(metrics_dict, dict):
                for key in ("metrics/mAP50(B)", "metrics/mAP50", "mAP50", "map50"):
                    if key in metrics_dict and map50_value <= 0.0:
                        map50_value = _safe_float(metrics_dict.get(key), map50_value)
                        if map50_value > 0.0:
                            break
                for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "map50_95", "metrics/mAP50_95"):
                    if key in metrics_dict and map50_95_value <= 0.0:
                        map50_95_value = _safe_float(metrics_dict.get(key), map50_95_value)
                        if map50_95_value > 0.0:
                            break

            return float(map50_value), float(map50_95_value)

        def _emit_metric_payload(payload: dict[str, float | str], metric_state: dict[str, int]) -> None:
            epoch_value = int(_safe_float(payload.get("epoch", 0.0), 0.0))
            if epoch_value <= int(metric_state.get("last_epoch", 0)):
                return
            metric_state["last_epoch"] = epoch_value
            print(REMOTE_TRAIN_METRIC_PREFIX + json.dumps(payload, ensure_ascii=True), flush=True)

        def _emit_metric(trainer, args, metric_state: dict[str, int]) -> None:
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            total_epochs = int(getattr(getattr(trainer, "args", None), "epochs", args.epochs))
            box_loss, cls_loss, dfl_loss = _extract_loss_components(trainer, str(args.engine))
            map50, map50_95 = _extract_accuracy_values(trainer)
            payload = {{
                "epoch": float(epoch),
                "total_epochs": float(max(1, total_epochs)),
                "box_loss": float(box_loss),
                "cls_loss": float(cls_loss),
                "dfl_loss": float(dfl_loss),
                "loss": float(box_loss + cls_loss + dfl_loss),
                "accuracy": float(map50),
                "map50": float(map50),
                "map50_95": float(map50_95),
                "stage_label": "",
            }}
            _emit_metric_payload(payload, metric_state)

        def _metric_from_csv_row(row: dict[str, str], args) -> dict[str, float | str]:
            raw_epoch = int(_safe_float(row.get("epoch", row.get("Epoch", "0")), 0.0))
            epoch = raw_epoch + 1 if raw_epoch < int(args.epochs) else raw_epoch
            if epoch <= 0:
                epoch = 1
            total_epochs = max(1, int(args.epochs))
            box_loss = _safe_float(
                row.get(
                    "train/box_loss",
                    row.get("box_loss", row.get("train/giou_loss", row.get("giou_loss", "0"))),
                ),
                0.0,
            )
            cls_loss = _safe_float(row.get("train/cls_loss", row.get("cls_loss", "0")), 0.0)
            dfl_loss = _safe_float(
                row.get(
                    "train/dfl_loss",
                    row.get("dfl_loss", row.get("train/l1_loss", row.get("l1_loss", "0"))),
                ),
                0.0,
            )
            map50 = _safe_float(
                row.get(
                    "metrics/mAP50(B)",
                    row.get("metrics/mAP50", row.get("mAP50", row.get("map50", "0"))),
                ),
                0.0,
            )
            map50_95 = _safe_float(
                row.get(
                    "metrics/mAP50-95(B)",
                    row.get(
                        "metrics/mAP50-95",
                        row.get("mAP50-95", row.get("map50_95", row.get("metrics/mAP50_95", "0"))),
                    ),
                ),
                0.0,
            )
            return {{
                "epoch": float(epoch),
                "total_epochs": float(total_epochs),
                "box_loss": float(max(0.0, box_loss)),
                "cls_loss": float(max(0.0, cls_loss)),
                "dfl_loss": float(max(0.0, dfl_loss)),
                "loss": float(max(0.0, box_loss + cls_loss + dfl_loss)),
                "accuracy": float(max(0.0, map50)),
                "map50": float(max(0.0, map50)),
                "map50_95": float(max(0.0, map50_95)),
                "stage_label": "",
            }}

        def _watch_results_csv(args, metric_state: dict[str, int], stop_event: threading.Event) -> None:
            project_dir = Path(str(args.project)).expanduser().resolve()
            run_dir = project_dir / str(args.name)
            results_csv = run_dir / "results.csv"
            while not stop_event.is_set():
                if results_csv.is_file():
                    try:
                        with results_csv.open("r", encoding="utf-8", newline="") as fp:
                            rows = [row for row in csv.DictReader(fp) if isinstance(row, dict) and row]
                    except Exception:
                        rows = []
                    if rows:
                        try:
                            payload = _metric_from_csv_row(rows[-1], args)
                        except Exception:
                            payload = None
                        if isinstance(payload, dict):
                            _emit_metric_payload(payload, metric_state)
                stop_event.wait(0.8)

        def _emit_batch(trainer, args, batch_state: dict[str, object]) -> None:
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            nb = int(getattr(trainer, "nb", 0))
            pbar = getattr(trainer, "pbar", None)
            if nb <= 0 and pbar is not None:
                try:
                    nb = int(getattr(pbar, "total", 0))
                except Exception:
                    nb = 0
            if nb <= 0:
                train_loader = getattr(trainer, "train_loader", None)
                try:
                    nb = int(len(train_loader))
                except Exception:
                    nb = 0
            nb = max(1, nb)

            batch_i = -1
            raw_batch_i = getattr(trainer, "batch_i", None)
            if isinstance(raw_batch_i, (int, float)):
                batch_i = int(raw_batch_i) + 1
            if batch_i <= 0:
                raw_i = getattr(trainer, "i", None)
                if isinstance(raw_i, (int, float)):
                    batch_i = int(raw_i) + 1
            if pbar is not None:
                try:
                    pbar_n = int(getattr(pbar, "n", 0))
                except Exception:
                    pbar_n = 0
                if pbar_n > batch_i:
                    batch_i = pbar_n
            batch_i = max(1, min(batch_i, nb))

            batch_key = (int(epoch), int(batch_i), int(nb))
            if batch_key == batch_state.get("last_key"):
                return
            batch_state["last_key"] = batch_key

            total_epochs = max(1, int(getattr(getattr(trainer, "args", None), "epochs", args.epochs)))
            epoch_progress = float(batch_i) / float(nb)
            total_progress = ((float(epoch - 1) + epoch_progress) / float(total_epochs))
            payload = {{
                "epoch": float(epoch),
                "total_epochs": float(total_epochs),
                "batch": float(batch_i),
                "num_batches": float(nb),
                "epoch_progress": float(epoch_progress),
                "total_progress": float(total_progress),
            }}
            print(REMOTE_TRAIN_BATCH_PREFIX + json.dumps(payload, ensure_ascii=True), flush=True)

        def main() -> None:
            args = _parse()
            if args.engine == "rtdetr":
                from ultralytics import RTDETR as _Model
            else:
                from ultralytics import YOLO as _Model

            model = _Model(str(args.model_source))
            batch_state = {{"last_key": None}}
            metric_state = {{"last_epoch": 0}}

            try:
                model.add_callback("on_train_epoch_end", lambda trainer: _emit_metric(trainer, args, metric_state))
            except Exception:
                pass
            try:
                model.add_callback("on_fit_epoch_end", lambda trainer: _emit_metric(trainer, args, metric_state))
            except Exception:
                pass
            try:
                model.add_callback("on_train_batch_end", lambda trainer: _emit_batch(trainer, args, batch_state))
            except Exception:
                try:
                    model.add_callback("on_batch_end", lambda trainer: _emit_batch(trainer, args, batch_state))
                except Exception:
                    pass

            train_kwargs = {{
                "data": str(args.data_yaml),
                "task": str(args.task),
                "epochs": max(1, int(args.epochs)),
                "imgsz": max(32, int(args.imgsz)),
                "batch": int(args.batch),
                "patience": max(1, int(args.patience)),
                "project": str(Path(str(args.project)).expanduser().resolve()),
                "name": str(args.name),
                "exist_ok": True,
                "verbose": True,
            }}
            if args.lr0 is not None:
                train_kwargs["lr0"] = float(args.lr0)
            if args.freeze_json:
                train_kwargs["freeze"] = json.loads(args.freeze_json)
            if args.device:
                train_kwargs["device"] = str(args.device)
            if args.engine == "rtdetr":
                train_kwargs["workers"] = 4
                try:
                    import torch  # type: ignore
                    train_kwargs["amp"] = bool(torch.cuda.is_available())
                except Exception:
                    train_kwargs["amp"] = False
            if args.extra_params_json:
                _SKIP_KEYS = {{"data", "task", "project", "name", "exist_ok", "verbose", "device"}}
                try:
                    extra = json.loads(args.extra_params_json)
                    if isinstance(extra, dict):
                        for _k, _v in extra.items():
                            if _k not in _SKIP_KEYS:
                                train_kwargs[_k] = _v
                except Exception:
                    pass

            watcher_stop = threading.Event()
            watcher_thread = threading.Thread(
                target=_watch_results_csv,
                args=(args, metric_state, watcher_stop),
                daemon=True,
            )
            watcher_thread.start()
            try:
                results = model.train(**train_kwargs)
            finally:
                watcher_stop.set()
                watcher_thread.join(timeout=2.0)
            save_dir = Path(str(getattr(results, "save_dir", Path(args.project) / args.name))).resolve()
            if args.save_dir_file:
                marker = Path(str(args.save_dir_file)).expanduser().resolve()
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(save_dir.as_posix(), encoding="utf-8")
            print(REMOTE_TRAIN_SAVE_DIR_PREFIX + save_dir.as_posix(), flush=True)

        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"


class _SignalTextStream:
    """stdout/stderr 텍스트를 Qt 시그널로 전달하는 파일형 스트림 어댑터입니다."""

    def __init__(self, emit_func: Callable[[str], None]) -> None:
        self._emit = emit_func

    def write(self, text: str) -> int:
        value = str(text)
        if value:
            self._emit(value)
        return len(value)

    def flush(self) -> None:
        return

class YoloTrainWorker(QObject):
    """YOLO/RT-DETR 학습을 백그라운드 스레드에서 실행하고 실시간 지표를 발행합니다."""

    log_message = pyqtSignal(str)
    metric_changed = pyqtSignal(object)
    batch_progress = pyqtSignal(object)
    terminal_output = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        data_yaml_path: Path,
        model_source: str,
        train_engine: str,
        train_mode: str,
        task_name: str,
        epochs: int,
        imgsz: int,
        batch: int,
        patience: int,
        run_project_dir: Path,
        run_name: str,
        freeze: int | None = None,
        lr0: float = TRAIN_DEFAULT_LR0,
        retrain_recipe: Mapping[str, Any] | None = None,
        force_local: bool = False,
        extra_params: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.force_local = bool(force_local)
        self.extra_params: dict[str, Any] = dict(extra_params) if isinstance(extra_params, Mapping) else {}
        self.data_yaml_path = Path(data_yaml_path)
        self.model_source = str(model_source).strip()
        engine_key = str(train_engine).strip().lower()
        self.train_engine = engine_key if engine_key in {"yolo", "rtdetr"} else "yolo"
        mode_key = str(train_mode).strip().lower()
        self.train_mode = mode_key if mode_key in {"new", "retrain"} else "new"
        task_key = str(task_name).strip().lower()
        self.task_name = task_key if task_key in {"detect", "classify", "segment", "pose", "obb"} else "detect"
        self.epochs = max(1, int(epochs))
        self.imgsz = max(32, int(imgsz))
        parsed_batch = int(batch)
        if self.train_engine == "rtdetr":
            # RT-DETR는 auto-batch(-1) 안정성이 낮아 기본 4를 사용합니다.
            self.batch = max(1, parsed_batch if parsed_batch > 0 else 4)
        else:
            self.batch = -1 if parsed_batch == -1 else max(1, parsed_batch)
        self.patience = max(1, int(patience))
        self.run_project_dir = Path(run_project_dir)
        self.run_name = str(run_name).strip() or "train_run"
        self.freeze = None if freeze is None else max(0, int(freeze))
        self.lr0 = max(0.000001, float(lr0))
        self.retrain_recipe = dict(retrain_recipe) if isinstance(retrain_recipe, Mapping) else None
        self.stage1_epochs = int(
            self.retrain_recipe.get("stage1_epochs", TRAIN_STAGE1_EPOCHS_DEFAULT)
        ) if self.retrain_recipe else 0
        self.stage2_epochs = int(
            self.retrain_recipe.get("stage2_epochs", TRAIN_STAGE2_EPOCHS_DEFAULT)
        ) if self.retrain_recipe else 0
        self.pipeline_total_epochs = (
            max(1, self.stage1_epochs + self.stage2_epochs)
            if self.retrain_recipe and self.train_engine == "yolo" and self.train_mode == "retrain"
            else self.epochs
        )
        self._stop_requested = False
        self._zero_loss_warned = False
        self._last_metric_epoch = -1
        self._last_metric_payload: dict[str, Any] = {}
        self._last_batch_progress_key: tuple[int, int, int] | None = None

    def request_stop(self) -> None:
        self._stop_requested = True

    @pyqtSlot()
    def run(self) -> None:
        remote_cfg = RemoteTrainConfig.from_env()
        try:
            if not self.data_yaml_path.is_file():
                raise RuntimeError(f"data.yaml 파일이 없습니다: {self.data_yaml_path}")
            if self.train_engine == "rtdetr" and self.task_name != "detect":
                self.task_name = "detect"

            self.run_project_dir.mkdir(parents=True, exist_ok=True)
            self.log_message.emit(
                f"train start: engine={self.train_engine}, mode={self.train_mode}, task={self.task_name}, "
                f"model={self.model_source}, data={self.data_yaml_path}, "
                f"epochs={self.pipeline_total_epochs}, lr0={self.lr0:.6f}, run={self.run_name}, "
                f"remote={'on' if remote_cfg.enabled else 'off'}, ultralytics={ULTRALYTICS_VERSION or 'unknown'}"
            )

            if remote_cfg.enabled and not self.force_local:
                try:
                    if self._use_two_stage_retrain():
                        if UltralyticsYOLO is None:
                            raise RuntimeError("ultralytics YOLO를 불러올 수 없습니다.")
                        result_payload = self._run_two_stage_retrain(model_class=UltralyticsYOLO, remote_cfg=remote_cfg)
                    else:
                        result_payload = self._run_training_stage_remote(remote_cfg)
                    self.finished.emit(result_payload)
                    return
                except WorkerStoppedError:
                    self.finished.emit(
                        {
                            "save_dir": "",
                            "stopped": True,
                            "final_metric": {"epoch": 0.0, "total_epochs": float(self.pipeline_total_epochs)},
                            "skip_rename": True,
                        }
                    )
                    return
                except Exception as exc:
                    if remote_cfg.required:
                        raise
                    self.log_message.emit(f"warning: remote training failed, fallback to local ({exc})")

            if self.train_engine == "rtdetr":
                if UltralyticsRTDETR is None:
                    raise RuntimeError("ultralytics RT-DETR를 불러올 수 없습니다.")
                model_class: Any = UltralyticsRTDETR
            else:
                if UltralyticsYOLO is None:
                    raise RuntimeError("ultralytics YOLO를 불러올 수 없습니다.")
                model_class = UltralyticsYOLO
            if self._use_two_stage_retrain():
                result_payload = self._run_two_stage_retrain(model_class=model_class)
            else:
                train_result = self._run_training_stage(
                    model_class=model_class,
                    model_source=self.model_source,
                    run_name=self.run_name,
                    epochs=self.epochs,
                    freeze=self.freeze,
                    lr0=self.lr0,
                    total_epochs_override=self.pipeline_total_epochs,
                )
                save_dir = Path(train_result.get("save_dir", self.run_project_dir / self.run_name)).resolve()
                result_payload = {
                    "save_dir": str(save_dir),
                    "stopped": bool(self._stop_requested),
                    "final_metric": dict(train_result.get("final_metric", self._last_metric_payload)),
                }
            self.finished.emit(result_payload)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _run_training_stage_remote(
        self,
        cfg: RemoteTrainConfig,
        *,
        model_source: str | None = None,
        run_name: str | None = None,
        epochs: int | None = None,
        freeze: int | Sequence[int] | None = None,
        lr0: float | None = None,
        data_yaml_path: Path | None = None,
        stage_label: str = "",
        epoch_offset: int = 0,
        total_epochs_override: int | None = None,
    ) -> dict[str, Any]:
        try:
            import paramiko  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("Remote training requires paramiko. Install it with: pip install paramiko") from exc

        if self._stop_requested:
            raise WorkerStoppedError()

        effective_model_source = str(model_source or self.model_source).strip()
        effective_run_name = str(run_name or self.run_name).strip() or self.run_name
        effective_epochs = max(1, int(self.epochs if epochs is None else epochs))
        effective_lr0 = float(self.lr0 if lr0 is None else lr0)
        effective_stage_label = str(stage_label or "")
        effective_epoch_offset = max(0, int(epoch_offset))
        pipeline_total_epochs = max(1, int(total_epochs_override or effective_epochs))
        effective_data_yaml = Path(data_yaml_path if data_yaml_path is not None else self.data_yaml_path).resolve()

        resolved_model_source, _model_workdir = _resolve_managed_model_source(effective_model_source, engine_key=self.train_engine)
        local_model_path: Path | None = None
        resolved_model_candidate = Path(str(resolved_model_source))
        if resolved_model_candidate.is_file():
            local_model_path = resolved_model_candidate.resolve()

        local_data_yaml = effective_data_yaml
        local_data_payload = _load_yaml_dict(local_data_yaml)
        if not local_data_payload:
            raise RuntimeError(f"Invalid data.yaml: {local_data_yaml}")
        local_dataset_root = _resolve_dataset_root_from_yaml_payload(local_data_yaml, local_data_payload)
        if not local_dataset_root.is_dir():
            raise RuntimeError(f"dataset root not found: {local_dataset_root}")

        safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(effective_run_name).strip() or "train_run")
        remote_job_name = f"{safe_run_name}_{int(time.time())}_{os.getpid()}"
        remote_program_root = str(cfg.remote_work_root).rstrip("/\\")
        remote_job_root = f"{remote_program_root}/.remote_jobs/{remote_job_name}"
        remote_project_dir = f"{remote_job_root}/runs/train"
        remote_script_path = f"{remote_job_root}/remote_train.py"
        remote_save_marker_path = f"{remote_job_root}/save_dir.txt"
        remote_dataset_root = f"{remote_job_root}/dataset"
        remote_data_yaml_path = f"{remote_job_root}/{local_data_yaml.name}"
        remote_model_source = str(resolved_model_source).strip()
        if local_model_path is not None:
            remote_model_source = f"{remote_job_root}/models/{local_model_path.name}"

        freeze_value: int | list[int] | None
        effective_freeze = self.freeze if freeze is None else freeze
        if effective_freeze is None:
            freeze_value = None
        elif isinstance(effective_freeze, Sequence) and not isinstance(effective_freeze, (str, bytes, bytearray)):
            freeze_value = [int(item) for item in effective_freeze]
        elif self._supports_freeze_list():
            freeze_value = list(range(int(effective_freeze)))
        else:
            freeze_value = int(effective_freeze)
        freeze_json = "" if freeze_value is None else json.dumps(freeze_value, ensure_ascii=True)

        self.log_message.emit(
            f"remote training target: {cfg.ssh_user}@{cfg.ssh_host}:{cfg.ssh_port}, "
            f"device={cfg.device or 'auto'}"
        )
        self.log_message.emit("remote sync: uploading dataset and config...")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        sftp = None
        remote_save_dir = f"{remote_project_dir}/{effective_run_name}"
        captured_tail = ""
        is_remote_windows = False
        try:
            client.connect(
                hostname=cfg.ssh_host,
                port=int(cfg.ssh_port),
                username=cfg.ssh_user,
                password=cfg.ssh_password,
                timeout=float(cfg.connect_timeout_sec),
                auth_timeout=max(5.0, float(cfg.connect_timeout_sec)),
                banner_timeout=max(5.0, float(cfg.connect_timeout_sec)),
                look_for_keys=False,
                allow_agent=False,
            )
            sftp = client.open_sftp()

            try:
                _stdin_probe, stdout_probe, _stderr_probe = client.exec_command("uname")
                probe_status = int(stdout_probe.channel.recv_exit_status())
                is_remote_windows = (probe_status != 0)
            except Exception:
                is_remote_windows = False

            def _split_remote_command_parts(command_text: str) -> list[str]:
                raw = str(command_text).strip()
                if not raw:
                    return []
                try:
                    return shlex.split(raw, posix=(not is_remote_windows))
                except Exception:
                    return [raw]

            def _remote_windows_python_env_paths(command_parts: Sequence[str]) -> list[str]:
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

            def _quote_powershell_arg(value: str) -> str:
                return "'" + str(value).replace("'", "''") + "'"

            def _build_remote_shell_cmd(command_parts: Sequence[str]) -> str:
                normalized = [str(part) for part in command_parts if str(part).strip()]
                if is_remote_windows:
                    env_paths = _remote_windows_python_env_paths(normalized)
                    if not env_paths:
                        return subprocess.list2cmdline(normalized)

                    quoted_paths = ", ".join(_quote_powershell_arg(path) for path in env_paths)
                    quoted_args = " ".join(_quote_powershell_arg(part) for part in normalized)
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

            python_cmd_candidates: list[str] = []
            preferred_python = str(cfg.python_cmd).strip()
            if preferred_python:
                python_cmd_candidates.append(preferred_python)
            if is_remote_windows:
                user_home = f"C:/Users/{cfg.ssh_user}"
                fallback_candidates = [
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
                    "G:/conda_envs/PJ_310_SAM3/python.exe",
                    "G:/conda_envs/PJ_310_LLM_SAM3/python.exe",
                    "G:/conda_envs/PJ_310_LLM/python.exe",
                    "C:/conda_envs/PJ_310_SAM3/python.exe",
                    "C:/conda_envs/PJ_310_LLM_SAM3/python.exe",
                    "D:/conda_envs/PJ_310_SAM3/python.exe",
                ]
            else:
                fallback_candidates = ["python3", "python"]
            for candidate in fallback_candidates:
                if candidate not in python_cmd_candidates:
                    python_cmd_candidates.append(candidate)

            selected_python_cmd: str | None = None
            selected_python_meta: dict[str, Any] = {}
            selected_python_parts: list[str] | None = None
            probe_prefix = "__REMOTE_PYTHON_PROBE__="
            probe_model_class = "RTDETR" if self.train_engine == "rtdetr" else "YOLO"
            probe_code = (
                "import json, sys, numpy, ultralytics; "
                f"from ultralytics import {probe_model_class}; "
                "payload = {"
                "'executable': sys.executable, "
                "'python': sys.version.split()[0], "
                "'numpy': getattr(numpy, '__version__', ''), "
                "'ultralytics': getattr(ultralytics, '__version__', '')"
                "}; "
                f"print({probe_prefix!r} + json.dumps(payload, ensure_ascii=True))"
            )
            python_probe_failures: list[str] = []

            def _extract_probe_payload(stdout_text: str) -> dict[str, Any] | None:
                for raw_line in str(stdout_text).splitlines():
                    line = str(raw_line).strip()
                    if not line.startswith(probe_prefix):
                        continue
                    payload_text = line[len(probe_prefix):].strip()
                    try:
                        payload_obj = json.loads(payload_text)
                    except Exception:
                        continue
                    if isinstance(payload_obj, Mapping):
                        return dict(payload_obj)
                return None

            def _tail_text(text: str) -> str:
                lines = [str(line).strip() for line in str(text).splitlines() if str(line).strip()]
                return lines[-1] if lines else ""

            for candidate in python_cmd_candidates:
                candidate_parts = _split_remote_command_parts(candidate)
                if not candidate_parts:
                    continue
                candidate_test = _build_remote_shell_cmd([*candidate_parts, "-c", probe_code])
                try:
                    _stdin_test, stdout_test, _stderr_test = client.exec_command(candidate_test)
                    status_test = int(stdout_test.channel.recv_exit_status())
                    if status_test == 0:
                        stdout_test_text = stdout_test.read().decode("utf-8", errors="replace")
                        selected_python_meta = _extract_probe_payload(stdout_test_text) or {
                            "executable": candidate,
                            "python": "",
                            "numpy": "",
                            "ultralytics": "",
                        }
                        selected_python_cmd = candidate
                        selected_python_parts = candidate_parts
                        break
                    stderr_test_text = _stderr_test.read().decode("utf-8", errors="replace")
                    stdout_test_text = stdout_test.read().decode("utf-8", errors="replace")
                    error_text = _tail_text(stderr_test_text) or _tail_text(stdout_test_text) or f"exit code {status_test}"
                    python_probe_failures.append(f"- {candidate}: {error_text}")
                except Exception:
                    python_probe_failures.append(f"- {candidate}: command probe failed")
                    continue
            if not selected_python_cmd or not selected_python_parts:
                detail_text = "\n".join(python_probe_failures[-5:])
                raise RuntimeError(
                    "Remote Python with required packages not found. "
                    "Set TRAIN_REMOTE_PYTHON_CMD to a Python that can import numpy and ultralytics.\n"
                    f"{detail_text}".rstrip()
                )

            python_cmd_parts = list(selected_python_parts)
            lower_cmd_parts = {str(part).strip().lower() for part in python_cmd_parts}
            if "-u" not in lower_cmd_parts:
                python_cmd_parts.append("-u")
            self.log_message.emit(
                "remote python selected: "
                f"{selected_python_cmd} "
                f"(python={selected_python_meta.get('python') or 'unknown'}, "
                f"numpy={selected_python_meta.get('numpy') or 'unknown'}, "
                f"ultralytics={selected_python_meta.get('ultralytics') or 'unknown'})"
            )

            remote_cmd_args = [
                *python_cmd_parts,
                remote_script_path,
                "--engine",
                self.train_engine,
                "--model-source",
                remote_model_source,
                "--data-yaml",
                remote_data_yaml_path,
                "--task",
                self.task_name,
                "--epochs",
                str(effective_epochs),
                "--imgsz",
                str(self.imgsz),
                "--batch",
                str(self.batch),
                "--patience",
                str(self.patience),
                "--project",
                remote_project_dir,
                "--name",
                effective_run_name,
                "--lr0",
                f"{effective_lr0:.8f}",
                "--save-dir-file",
                remote_save_marker_path,
            ]
            if freeze_json:
                remote_cmd_args.extend(["--freeze-json", freeze_json])
            if str(cfg.device).strip():
                remote_cmd_args.extend(["--device", str(cfg.device).strip()])
            if self.extra_params:
                _SKIP_REMOTE = {"data", "task", "project", "name", "exist_ok", "verbose", "device"}
                extra_for_remote = {k: v for k, v in self.extra_params.items() if k not in _SKIP_REMOTE}
                if extra_for_remote:
                    remote_cmd_args.extend(["--extra-params-json", json.dumps(extra_for_remote, ensure_ascii=True)])
            remote_shell_cmd = _build_remote_shell_cmd(remote_cmd_args)

            _sftp_mkdir_p(sftp, remote_job_root)
            _sftp_upload_tree(sftp, local_dataset_root, remote_dataset_root)

            remote_yaml_text = _build_remote_data_yaml_text(
                local_data_yaml,
                remote_dataset_root,
                local_dataset_root=local_dataset_root,
            )
            _sftp_mkdir_p(sftp, str(remote_data_yaml_path).rsplit("/", 1)[0])
            with sftp.file(remote_data_yaml_path, "wb") as fp:
                fp.write(remote_yaml_text.encode("utf-8"))

            if local_model_path is not None:
                _sftp_mkdir_p(sftp, str(remote_model_source).rsplit("/", 1)[0])
                sftp.put(str(local_model_path), remote_model_source)

            script_text = _build_remote_train_script_text()
            with sftp.file(remote_script_path, "wb") as fp:
                fp.write(script_text.encode("utf-8"))
            try:
                sftp.chmod(remote_script_path, 0o755)
            except Exception:
                pass

            transport = client.get_transport()
            if transport is None or not transport.is_active():
                raise RuntimeError("SSH transport is not active")

            channel = transport.open_session()
            channel.exec_command(remote_shell_cmd)
            self.log_message.emit(f"remote training started ({'windows' if is_remote_windows else 'posix'} shell).")

            stdout_buffer = ""
            stderr_buffer = ""

            def _extract_json_marker_payload(line_text: str, marker_prefix: str) -> dict[str, Any] | None:
                idx = line_text.find(marker_prefix)
                if idx < 0:
                    return None
                text = line_text[idx + len(marker_prefix):].strip()
                if not text:
                    return None
                start_idx = text.find("{")
                if start_idx >= 0:
                    text = text[start_idx:].strip()
                if not text:
                    return None
                try:
                    payload_obj = json.loads(text)
                except Exception:
                    try:
                        decoded_obj, _end_idx = json.JSONDecoder().raw_decode(text)
                    except Exception:
                        return None
                    payload_obj = decoded_obj
                if not isinstance(payload_obj, Mapping):
                    return None
                return dict(payload_obj)

            def _extract_text_marker_payload(line_text: str, marker_prefix: str) -> str | None:
                idx = line_text.find(marker_prefix)
                if idx < 0:
                    return None
                text = line_text[idx + len(marker_prefix):].strip()
                return text or None

            def _process_remote_line(raw_line: str) -> None:
                nonlocal captured_tail, remote_save_dir
                line = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", str(raw_line)).strip()
                if not line:
                    return
                metric_payload = _extract_json_marker_payload(line, REMOTE_TRAIN_METRIC_PREFIX)
                if metric_payload is not None:
                    stage_epoch = int(float(metric_payload.get("epoch", 0.0)))
                    display_epoch = stage_epoch + effective_epoch_offset
                    metric_payload["stage_epoch"] = float(stage_epoch)
                    metric_payload["epoch"] = float(display_epoch)
                    metric_payload["total_epochs"] = float(pipeline_total_epochs)
                    metric_payload["stage_label"] = effective_stage_label
                    if display_epoch <= self._last_metric_epoch:
                        return
                    self._last_metric_epoch = display_epoch
                    self._last_metric_payload = dict(metric_payload)
                    self.metric_changed.emit(metric_payload)
                    return
                batch_payload = _extract_json_marker_payload(line, REMOTE_TRAIN_BATCH_PREFIX)
                if batch_payload is not None:
                    raw_epoch = int(float(batch_payload.get("epoch", 0.0)))
                    display_epoch = raw_epoch + effective_epoch_offset
                    batch_payload["epoch"] = float(display_epoch)
                    batch_payload["total_epochs"] = float(pipeline_total_epochs)
                    epoch_progress = float(batch_payload.get("epoch_progress", 0.0))
                    total_progress = (
                        (float(max(1, display_epoch) - 1) + float(max(0.0, epoch_progress)))
                        / float(max(1, pipeline_total_epochs))
                    )
                    batch_payload["total_progress"] = float(total_progress)
                    self.batch_progress.emit(batch_payload)
                    return
                save_dir_payload = _extract_text_marker_payload(line, REMOTE_TRAIN_SAVE_DIR_PREFIX)
                if save_dir_payload is not None:
                    remote_save_dir = save_dir_payload
                    return
                captured_tail = (captured_tail + line + "\n")[-20000:]
                self.terminal_output.emit(line + "\n")

            def _consume_remote_chunk(buffer: str, chunk: str) -> str:
                text = (buffer + str(chunk)).replace("\r", "\n")
                parts = text.split("\n")
                next_buffer = parts.pop() if parts else ""
                for part in parts:
                    _process_remote_line(part)
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
                    chunk = channel.recv(65536).decode("utf-8", errors="replace")
                    if not chunk:
                        break
                    has_data = True
                    stdout_buffer = _consume_remote_chunk(stdout_buffer, chunk)
                while channel.recv_stderr_ready():
                    chunk = channel.recv_stderr(65536).decode("utf-8", errors="replace")
                    if not chunk:
                        break
                    has_data = True
                    stderr_buffer = _consume_remote_chunk(stderr_buffer, chunk)

                if channel.exit_status_ready() and (not channel.recv_ready()) and (not channel.recv_stderr_ready()):
                    break
                if not has_data:
                    time.sleep(0.1)

            if stdout_buffer.strip():
                _process_remote_line(stdout_buffer)
            if stderr_buffer.strip():
                _process_remote_line(stderr_buffer)

            exit_code = int(channel.recv_exit_status())
            if exit_code != 0:
                tail_line = "\n".join(captured_tail.strip().splitlines()[-8:])
                raise RuntimeError(f"Remote training exited with code {exit_code}. {tail_line}")

            try:
                if sftp is not None:
                    with sftp.file(remote_save_marker_path, "rb") as marker_fp:
                        marker_value = marker_fp.read().decode("utf-8", errors="replace").strip()
                        if marker_value:
                            remote_save_dir = marker_value
            except Exception:
                pass

            local_save_dir = Path(self.run_project_dir / effective_run_name).resolve()
            if local_save_dir.exists():
                if local_save_dir.is_dir():
                    shutil.rmtree(local_save_dir, ignore_errors=True)
                else:
                    local_save_dir.unlink(missing_ok=True)
            self.log_message.emit("remote sync: downloading training outputs...")
            try:
                if sftp is None:
                    raise RuntimeError("SFTP session is not available")
                sftp.stat(remote_save_dir)
            except Exception as exc:
                raise RuntimeError(f"Remote run directory not found: {remote_save_dir}") from exc
            _sftp_download_tree(sftp, remote_save_dir, local_save_dir)

            return {
                "save_dir": str(local_save_dir),
                "stopped": False,
                "final_metric": dict(
                    self._last_metric_payload
                    or {
                        "epoch": float(pipeline_total_epochs),
                        "total_epochs": float(pipeline_total_epochs),
                        "stage_label": effective_stage_label,
                    }
                ),
            }
        finally:
            if sftp is not None:
                if (not cfg.keep_remote_files) and (not self._stop_requested):
                    try:
                        if is_remote_windows:
                            cleanup_path = str(remote_job_root).replace("/", "\\")
                            client.exec_command(f'cmd /c rmdir /s /q "{cleanup_path}"')
                        else:
                            client.exec_command(f"rm -rf {shlex.quote(remote_job_root)}")
                    except Exception:
                        pass
                try:
                    sftp.close()
                except Exception:
                    pass
            try:
                client.close()
            except Exception:
                pass

    def _create_training_model(self, model_class: Any, model_source: str) -> Any:
        try:
            if torch is not None and bool(torch.cuda.is_available()):  # type: ignore[union-attr]
                torch.cuda.empty_cache()  # type: ignore[union-attr]
        except Exception:
            pass
        resolved_source, workdir = _resolve_managed_model_source(model_source, engine_key=self.train_engine)
        with _temporary_working_directory(workdir):
            return model_class(resolved_source)

    def _use_two_stage_retrain(self) -> bool:
        return bool(
            self.retrain_recipe
            and self.train_engine == "yolo"
            and self.train_mode == "retrain"
            and self.task_name == "detect"
        )

    def _supports_freeze_list(self) -> bool:
        return _parse_version_tuple(ULTRALYTICS_VERSION) >= (8, 0, 0)

    def _freeze_arg_for_indices(self, stop_index_exclusive: int) -> int | list[int]:
        count = max(0, int(stop_index_exclusive))
        if self._supports_freeze_list():
            return list(range(count))
        return count

    def _freeze_arg_for_stage1(self) -> int | list[int]:
        # Local ultralytics 8.4.14 YOLO11 detect yaml uses module indices:
        # backbone=0..10, neck/head=11..22, Detect=23. Stage-1 keeps only Detect trainable.
        return self._freeze_arg_for_indices(23)

    def _freeze_arg_for_stage2(self, unfreeze_mode: str) -> int | list[int]:
        # Stage-2 neck_only -> freeze 0..10, backbone_last -> freeze 0..9.
        mode = str(unfreeze_mode or TRAIN_STAGE_UNFREEZE_NECK_ONLY).strip().lower()
        if mode == TRAIN_STAGE_UNFREEZE_BACKBONE_LAST:
            return self._freeze_arg_for_indices(10)
        return self._freeze_arg_for_indices(11)

    def _freeze_arg_text(self, freeze_value: int | Sequence[int] | None) -> str:
        if freeze_value is None:
            return "none"
        if isinstance(freeze_value, Sequence) and not isinstance(freeze_value, (str, bytes, bytearray)):
            items = [int(item) for item in freeze_value]
            if not items:
                return "[]"
            return f"[{items[0]}..{items[-1]}]"
        return str(int(freeze_value))

    def _run_two_stage_retrain(
        self,
        *,
        model_class: Any,
        remote_cfg: RemoteTrainConfig | None = None,
    ) -> dict[str, Any]:
        recipe = dict(self.retrain_recipe or {})
        merged_yaml_path = Path(str(recipe.get("merged_yaml_path", self.data_yaml_path))).resolve()
        old_eval_yaml_path = Path(str(recipe.get("old_eval_yaml_path", recipe.get("old_yaml_path", "")))).resolve()
        if not merged_yaml_path.is_file():
            raise RuntimeError(f"재학습 merged data.yaml 파일이 없습니다: {merged_yaml_path}")
        if not old_eval_yaml_path.is_file():
            raise RuntimeError(f"재학습 기준 old eval data.yaml 파일이 없습니다: {old_eval_yaml_path}")

        stage1_epochs = max(1, int(recipe.get("stage1_epochs", TRAIN_STAGE1_EPOCHS_DEFAULT)))
        stage2_epochs = max(1, int(recipe.get("stage2_epochs", TRAIN_STAGE2_EPOCHS_DEFAULT)))
        stage2_lr_factor = max(0.01, float(recipe.get("stage2_lr_factor", TRAIN_STAGE2_LR_FACTOR_DEFAULT)))
        unfreeze_mode = str(recipe.get("unfreeze_mode", TRAIN_STAGE_UNFREEZE_NECK_ONLY)).strip().lower()
        if unfreeze_mode not in TRAIN_STAGE_UNFREEZE_CHOICES:
            unfreeze_mode = TRAIN_STAGE_UNFREEZE_NECK_ONLY
        stage1_freeze = self._freeze_arg_for_stage1()
        stage2_freeze = self._freeze_arg_for_stage2(unfreeze_mode)
        stage2_lr0 = max(0.000001, float(self.lr0) * float(stage2_lr_factor))

        self.log_message.emit(
            f"retrain 2-stage recipe: ultralytics={ULTRALYTICS_VERSION or 'unknown'}, "
            f"stage1_freeze={self._freeze_arg_text(stage1_freeze)}, "
            f"stage2_freeze={self._freeze_arg_text(stage2_freeze)}, "
            f"unfreeze_mode={unfreeze_mode}"
        )
        baseline_old = self._run_validation_metrics(
            model_class=model_class,
            weights_path=Path(self.model_source).resolve(),
            data_yaml_path=old_eval_yaml_path,
            label="[Baseline][Old]",
        )
        if self._stop_requested:
            return {
                "save_dir": "",
                "stopped": True,
                "final_metric": dict(self._last_metric_payload),
                "skip_rename": True,
            }

        self.log_message.emit(
            f"[Stage1] start: data={merged_yaml_path}, epochs={stage1_epochs}, "
            f"lr0={self.lr0:.6f}, freeze={self._freeze_arg_text(stage1_freeze)}"
        )
        if remote_cfg is not None:
            stage1_result = self._run_training_stage_remote(
                remote_cfg,
                model_source=self.model_source,
                run_name=f"{self.run_name}_stage1",
                epochs=stage1_epochs,
                freeze=stage1_freeze,
                lr0=self.lr0,
                data_yaml_path=merged_yaml_path,
                stage_label="[Stage1]",
                epoch_offset=0,
                total_epochs_override=self.pipeline_total_epochs,
            )
        else:
            stage1_result = self._run_training_stage(
                model_class=model_class,
                model_source=self.model_source,
                run_name=f"{self.run_name}_stage1",
                epochs=stage1_epochs,
                freeze=stage1_freeze,
                lr0=self.lr0,
                data_yaml_path=merged_yaml_path,
                stage_label="[Stage1]",
                epoch_offset=0,
                total_epochs_override=self.pipeline_total_epochs,
            )
        stage1_dir = Path(str(stage1_result.get("save_dir", self.run_project_dir / f"{self.run_name}_stage1"))).resolve()
        stage1_best = (stage1_dir / "weights" / "best.pt").resolve()
        stage1_last = (stage1_dir / "weights" / "last.pt").resolve()
        if not stage1_best.is_file():
            raise RuntimeError(f"[Stage1] best.pt 파일이 없습니다: {stage1_best}")
        if not stage1_last.is_file():
            raise RuntimeError(f"[Stage1] last.pt 파일이 없습니다: {stage1_last}")
        stage1_best_alias = (stage1_dir / "stage1_best.pt").resolve()
        shutil.copy2(str(stage1_best), str(stage1_best_alias))

        stage1_new_metrics = self._run_validation_metrics(
            model_class=model_class,
            weights_path=stage1_best_alias,
            data_yaml_path=merged_yaml_path,
            label="[Stage1][New]",
        )
        stage1_old_metrics = self._run_validation_metrics(
            model_class=model_class,
            weights_path=stage1_best_alias,
            data_yaml_path=old_eval_yaml_path,
            label="[Stage1][Old]",
        )
        self._emit_old_drop_warning(
            stage_label="[Stage1]",
            baseline_old=baseline_old,
            current_old=stage1_old_metrics,
        )
        if self._stop_requested:
            return {
                "save_dir": str(stage1_dir),
                "stopped": True,
                "final_metric": dict(stage1_result.get("final_metric", self._last_metric_payload)),
                "skip_rename": True,
            }

        self.log_message.emit(
            f"[Stage2] start: data={merged_yaml_path}, epochs={stage2_epochs}, "
            f"lr0={stage2_lr0:.6f}, freeze={self._freeze_arg_text(stage2_freeze)}, "
            f"model_in={stage1_best_alias}"
        )
        if remote_cfg is not None:
            stage2_result = self._run_training_stage_remote(
                remote_cfg,
                model_source=str(stage1_best_alias),
                run_name=f"{self.run_name}_stage2",
                epochs=stage2_epochs,
                freeze=stage2_freeze,
                lr0=stage2_lr0,
                data_yaml_path=merged_yaml_path,
                stage_label="[Stage2]",
                epoch_offset=stage1_epochs,
                total_epochs_override=self.pipeline_total_epochs,
            )
        else:
            stage2_result = self._run_training_stage(
                model_class=model_class,
                model_source=str(stage1_best_alias),
                run_name=f"{self.run_name}_stage2",
                epochs=stage2_epochs,
                freeze=stage2_freeze,
                lr0=stage2_lr0,
                data_yaml_path=merged_yaml_path,
                stage_label="[Stage2]",
                epoch_offset=stage1_epochs,
                total_epochs_override=self.pipeline_total_epochs,
            )
        stage2_dir = Path(str(stage2_result.get("save_dir", self.run_project_dir / f"{self.run_name}_stage2"))).resolve()
        stage2_best = (stage2_dir / "weights" / "best.pt").resolve()
        stage2_last = (stage2_dir / "weights" / "last.pt").resolve()
        if not stage2_best.is_file():
            raise RuntimeError(f"[Stage2] best.pt 파일이 없습니다: {stage2_best}")
        if not stage2_last.is_file():
            raise RuntimeError(f"[Stage2] last.pt 파일이 없습니다: {stage2_last}")
        stage2_best_alias = (stage2_dir / "stage2_best.pt").resolve()
        shutil.copy2(str(stage2_best), str(stage2_best_alias))

        stage2_new_metrics = self._run_validation_metrics(
            model_class=model_class,
            weights_path=stage2_best_alias,
            data_yaml_path=merged_yaml_path,
            label="[Stage2][New]",
        )
        stage2_old_metrics = self._run_validation_metrics(
            model_class=model_class,
            weights_path=stage2_best_alias,
            data_yaml_path=old_eval_yaml_path,
            label="[Stage2][Old]",
        )
        self._emit_old_drop_warning(
            stage_label="[Stage2]",
            baseline_old=baseline_old,
            current_old=stage2_old_metrics,
        )

        retrain_summary = {
            "ultralytics_version": ULTRALYTICS_VERSION,
            "new_yaml_paths": list(recipe.get("new_yaml_paths", [])) if isinstance(recipe.get("new_yaml_paths", []), Sequence) and not isinstance(recipe.get("new_yaml_paths", []), (str, bytes, bytearray)) else [],
            "old_yaml_path": str(recipe.get("old_yaml_path", "")),
            "old_eval_yaml_path": str(old_eval_yaml_path),
            "merged_yaml_path": str(merged_yaml_path),
            "merged_dataset_root": str(recipe.get("merged_dataset_root", merged_yaml_path.parent)),
            "replay_ratio_old": float(recipe.get("replay_ratio_old", TRAIN_REPLAY_RATIO_DEFAULT)),
            "seed": int(recipe.get("seed", TRAIN_RETRAIN_SEED_DEFAULT)),
            "new_train_count": int(recipe.get("new_train_count", 0) or 0),
            "old_train_available_count": int(recipe.get("old_train_available_count", 0) or 0),
            "old_replay_count": int(recipe.get("old_replay_count", 0) or 0),
            "merged_train_count": int(recipe.get("merged_train_count", 0) or 0),
            "baseline_old": baseline_old,
            "stage1": {
                "model_in": str(Path(self.model_source).resolve()),
                "epochs": stage1_epochs,
                "lr0": float(self.lr0),
                "freeze": self._freeze_arg_text(stage1_freeze),
                "run_dir": str(stage1_dir),
                "best_pt": str(stage1_best_alias),
                "last_pt": str(stage1_last),
                "metrics_new": stage1_new_metrics,
                "metrics_old": stage1_old_metrics,
            },
            "stage2": {
                "model_in": str(stage1_best_alias),
                "epochs": stage2_epochs,
                "lr0": float(stage2_lr0),
                "freeze": self._freeze_arg_text(stage2_freeze),
                "unfreeze_mode": unfreeze_mode,
                "run_dir": str(stage2_dir),
                "best_pt": str(stage2_best_alias),
                "last_pt": str(stage2_last),
                "metrics_new": stage2_new_metrics,
                "metrics_old": stage2_old_metrics,
            },
        }
        self._write_retrain_summary_files(stage2_dir=stage2_dir, summary=retrain_summary)
        return {
            "save_dir": str(stage2_dir),
            "stopped": bool(self._stop_requested),
            "final_metric": dict(stage2_result.get("final_metric", self._last_metric_payload)),
            "skip_rename": True,
            "retrain_summary": retrain_summary,
            "stage1_dir": str(stage1_dir),
            "stage2_dir": str(stage2_dir),
        }

    def _emit_old_drop_warning(
        self,
        *,
        stage_label: str,
        baseline_old: Mapping[str, Any],
        current_old: Mapping[str, Any],
    ) -> None:
        baseline_m50 = _metric_to_unit_interval(float(baseline_old.get("map50", 0.0)))
        current_m50 = _metric_to_unit_interval(float(current_old.get("map50", 0.0)))
        if (baseline_m50 - current_m50) > 0.02:
            drop_pct = (baseline_m50 - current_m50) * 100.0
            self.log_message.emit(
                f"{stage_label} warning: old validation mAP50 dropped by {drop_pct:.2f}%p. "
                "replay 비율을 높이거나 lr0를 더 낮춰 보세요."
            )

    def _run_validation_metrics(
        self,
        *,
        model_class: Any,
        weights_path: Path,
        data_yaml_path: Path,
        label: str,
    ) -> dict[str, Any]:
        if self._stop_requested:
            return {"map50": 0.0, "map50_95": 0.0}
        if not weights_path.is_file():
            raise RuntimeError(f"{label} 가중치 파일이 없습니다: {weights_path}")
        if not data_yaml_path.is_file():
            raise RuntimeError(f"{label} data.yaml 파일이 없습니다: {data_yaml_path}")
        model = self._create_training_model(model_class, str(weights_path))
        val_batch = 1 if self.batch == -1 else max(1, self.batch)
        stream = _SignalTextStream(self.terminal_output.emit)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            results = model.val(
                data=str(data_yaml_path),
                imgsz=self.imgsz,
                batch=val_batch,
                verbose=False,
            )
        summary = self._extract_validation_metrics(results)
        self.log_message.emit(
            f"{label} validation: mAP50={float(summary.get('map50', 0.0)):.4f}, "
            f"mAP50-95={float(summary.get('map50_95', 0.0)):.4f}"
        )
        return summary

    def _extract_validation_metrics(self, results: Any) -> dict[str, Any]:
        map50_value = 0.0
        map50_95_value = 0.0

        box_obj = getattr(results, "box", None)
        if box_obj is not None:
            for attr_name in ("map50", "mAP50"):
                try:
                    value = float(getattr(box_obj, attr_name))
                except Exception:
                    continue
                if value > 0.0:
                    map50_value = value
                    break
            for attr_name in ("map", "map50_95", "mAP50_95"):
                try:
                    value = float(getattr(box_obj, attr_name))
                except Exception:
                    continue
                if value > 0.0:
                    map50_95_value = value
                    break

        for key_name in ("results_dict", "metrics"):
            metrics_obj = getattr(results, key_name, None)
            if isinstance(metrics_obj, Mapping):
                for key in ("metrics/mAP50(B)", "metrics/mAP50", "mAP50", "map50"):
                    if key in metrics_obj and map50_value <= 0.0:
                        try:
                            map50_value = float(metrics_obj[key])
                            break
                        except Exception:
                            continue
                for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "map50_95", "metrics/mAP50_95"):
                    if key in metrics_obj and map50_95_value <= 0.0:
                        try:
                            map50_95_value = float(metrics_obj[key])
                            break
                        except Exception:
                            continue

        return {
            "map50": _metric_to_unit_interval(map50_value),
            "map50_95": _metric_to_unit_interval(map50_95_value),
        }

    def _write_retrain_summary_files(self, *, stage2_dir: Path, summary: Mapping[str, Any]) -> None:
        run_dir = Path(stage2_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        recipe_path = run_dir / "train_recipe.json"
        recipe_path.write_text(json.dumps(dict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        lines: list[str] = []
        lines.append("2-stage 재학습 요약")
        lines.append("")
        lines.append(f"Ultralytics 버전: {ULTRALYTICS_VERSION or 'unknown'}")
        lines.append(f"merged data.yaml: {summary.get('merged_yaml_path', '')}")
        lines.append(f"old data.yaml: {summary.get('old_yaml_path', '')}")
        lines.append(f"replay_ratio_old: {float(summary.get('replay_ratio_old', TRAIN_REPLAY_RATIO_DEFAULT)):.3f}")
        lines.append(f"seed: {int(summary.get('seed', TRAIN_RETRAIN_SEED_DEFAULT))}")
        lines.append(
            "train 구성: "
            f"new={int(summary.get('new_train_count', 0) or 0)}, "
            f"replay_old={int(summary.get('old_replay_count', 0) or 0)}, "
            f"old_available={int(summary.get('old_train_available_count', 0) or 0)}, "
            f"merged_total={int(summary.get('merged_train_count', 0) or 0)}"
        )
        lines.append("")
        baseline_old = summary.get("baseline_old", {})
        if isinstance(baseline_old, Mapping):
            lines.append(
                f"Baseline old: mAP50={float(baseline_old.get('map50', 0.0)):.4f}, "
                f"mAP50-95={float(baseline_old.get('map50_95', 0.0)):.4f}"
            )
        for stage_name in ("stage1", "stage2"):
            stage_data = summary.get(stage_name, {})
            if not isinstance(stage_data, Mapping):
                continue
            metrics_new = stage_data.get("metrics_new", {})
            metrics_old = stage_data.get("metrics_old", {})
            lines.append("")
            lines.append(stage_name.upper())
            lines.append(f"- run_dir: {stage_data.get('run_dir', '')}")
            lines.append(f"- best_pt: {stage_data.get('best_pt', '')}")
            lines.append(f"- lr0: {float(stage_data.get('lr0', 0.0)):.6f}")
            lines.append(f"- freeze: {stage_data.get('freeze', '')}")
            lines.append(
                f"- new val: mAP50={float(metrics_new.get('map50', 0.0)):.4f}, "
                f"mAP50-95={float(metrics_new.get('map50_95', 0.0)):.4f}"
            )
            lines.append(
                f"- old val: mAP50={float(metrics_old.get('map50', 0.0)):.4f}, "
                f"mAP50-95={float(metrics_old.get('map50_95', 0.0)):.4f}"
            )
        (run_dir / "RETRAIN_README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _build_train_kwargs(
        self,
        *,
        run_name: str,
        epochs: int,
        freeze: int | Sequence[int] | None = None,
        lr0: float | None = None,
        data_yaml_path: Path | None = None,
    ) -> dict[str, Any]:
        train_kwargs: dict[str, Any] = {
            "data": str((Path(data_yaml_path) if data_yaml_path is not None else self.data_yaml_path).resolve()),
            "task": self.task_name,
            "epochs": max(1, int(epochs)),
            "imgsz": self.imgsz,
            "batch": self.batch,
            "patience": self.patience,
            "project": str(self.run_project_dir),
            "name": str(run_name).strip() or self.run_name,
            "exist_ok": True,
            "verbose": False,
        }
        if freeze is not None:
            if isinstance(freeze, Sequence) and not isinstance(freeze, (str, bytes, bytearray)):
                train_kwargs["freeze"] = [int(item) for item in freeze]
            else:
                train_kwargs["freeze"] = int(freeze)
        if lr0 is not None:
            train_kwargs["lr0"] = float(lr0)
        if self.train_engine == "rtdetr":
            # Team_code의 RT-DETR 기본값(workers=4, amp=True)을 반영합니다.
            train_kwargs["workers"] = 4
            use_amp = True
            try:
                use_amp = bool(torch is not None and torch.cuda.is_available())  # type: ignore[union-attr]
            except Exception:
                use_amp = False
            train_kwargs["amp"] = use_amp
        # 고급 파라미터 다이얼로그에서 설정된 값을 병합합니다 (내부 관리 키는 덮어쓰지 않음).
        # device는 force_local 모드에서 extra_params에 주입되므로 허용합니다.
        _INTERNAL_KEYS = {"data", "task", "project", "name", "exist_ok", "verbose"}
        if self.extra_params:
            for _k, _v in self.extra_params.items():
                if _k not in _INTERNAL_KEYS:
                    train_kwargs[_k] = _v
        return train_kwargs

    def _run_training_stage(
        self,
        *,
        model_class: Any,
        model_source: str,
        run_name: str,
        epochs: int,
        freeze: int | Sequence[int] | None = None,
        lr0: float | None = None,
        data_yaml_path: Path | None = None,
        stage_label: str = "",
        epoch_offset: int = 0,
        total_epochs_override: int | None = None,
    ) -> dict[str, Any]:
        train_epochs = max(1, int(epochs))
        pipeline_total_epochs = max(1, int(total_epochs_override or train_epochs))
        train_last_metric_payload: dict[str, Any] = {}
        model = self._create_training_model(model_class, model_source)

        def _on_epoch_end(trainer: Any) -> None:
            if self._stop_requested:
                try:
                    trainer.stop = True
                except Exception:
                    pass
            metric: dict[str, Any] = dict(self._extract_metric_row(trainer))
            stage_epoch = int(metric.get("epoch", 0))
            epoch = stage_epoch + max(0, int(epoch_offset))
            metric["stage_epoch"] = float(stage_epoch)
            metric["epoch"] = float(epoch)
            metric["total_epochs"] = float(pipeline_total_epochs)
            metric["stage_label"] = str(stage_label)
            if epoch <= self._last_metric_epoch:
                return
            self._last_metric_epoch = epoch
            train_last_metric_payload.clear()
            train_last_metric_payload.update(metric)
            self._last_metric_payload = dict(metric)
            self.metric_changed.emit(metric)
            total = int(metric.get("total_epochs", pipeline_total_epochs))
            box_loss = float(metric.get("box_loss", 0.0))
            cls_loss = float(metric.get("cls_loss", 0.0))
            dfl_loss = float(metric.get("dfl_loss", 0.0))
            map50 = float(metric.get("map50", 0.0))
            map50_95 = float(metric.get("map50_95", 0.0))
            if self.train_engine == "rtdetr":
                loss1_name, loss2_name, loss3_name = "giou", "cls", "l1"
            else:
                loss1_name, loss2_name, loss3_name = "box", "cls", "dfl"
            if (not self._zero_loss_warned) and (box_loss <= 0.0 and cls_loss <= 0.0 and dfl_loss <= 0.0):
                self._zero_loss_warned = True
                self.log_message.emit(
                    f"warning: {loss1_name}/{loss2_name}/{loss3_name} loss가 0으로 보고됩니다. 학습 로그 키를 다시 확인해 주세요."
                )
            self.log_message.emit(
                f"{stage_label} epoch {epoch}/{total} done | "
                f"{loss1_name}={box_loss:.4f}, {loss2_name}={cls_loss:.4f}, {loss3_name}={dfl_loss:.4f}, "
                f"mAP50={map50:.4f}, mAP50-95={map50_95:.4f}"
            )

        def _on_batch_end(trainer: Any) -> None:
            if self._stop_requested:
                try:
                    trainer.stop = True
                except Exception:
                    pass
            epoch = int(getattr(trainer, "epoch", 0)) + 1

            nb = int(getattr(trainer, "nb", 0))
            pbar = getattr(trainer, "pbar", None)
            if nb <= 0 and pbar is not None:
                try:
                    nb = int(getattr(pbar, "total", 0))
                except Exception:
                    nb = 0
            if nb <= 0:
                train_loader = getattr(trainer, "train_loader", None)
                try:
                    nb = int(len(train_loader))
                except Exception:
                    nb = 0
            nb = max(1, nb)

            batch_i = -1
            raw_batch_i = getattr(trainer, "batch_i", None)
            if isinstance(raw_batch_i, (int, float)):
                batch_i = int(raw_batch_i) + 1
            if batch_i <= 0:
                raw_i = getattr(trainer, "i", None)
                if isinstance(raw_i, (int, float)):
                    batch_i = int(raw_i) + 1
            if pbar is not None:
                try:
                    pbar_n = int(getattr(pbar, "n", 0))
                except Exception:
                    pbar_n = 0
                if pbar_n > batch_i:
                    batch_i = pbar_n
            batch_i = max(1, min(batch_i, nb))
            display_epoch = epoch + max(0, int(epoch_offset))

            batch_key = (int(display_epoch), int(batch_i), int(nb))
            if batch_key == self._last_batch_progress_key:
                return
            self._last_batch_progress_key = batch_key

            epoch_progress = float(batch_i) / float(nb)
            total_progress = ((float(display_epoch - 1) + epoch_progress) / float(max(1, pipeline_total_epochs)))
            self.batch_progress.emit(
                {
                    "epoch": float(display_epoch),
                    "total_epochs": float(pipeline_total_epochs),
                    "batch": float(batch_i),
                    "num_batches": float(nb),
                    "epoch_progress": float(epoch_progress),
                    "total_progress": float(total_progress),
                }
            )

        try:
            model.add_callback("on_train_epoch_end", _on_epoch_end)
        except Exception:
            pass
        try:
            model.add_callback("on_fit_epoch_end", _on_epoch_end)
        except Exception:
            pass
        batch_callback_registered = False
        try:
            model.add_callback("on_train_batch_end", _on_batch_end)
            batch_callback_registered = True
        except Exception:
            pass
        if not batch_callback_registered:
            try:
                model.add_callback("on_batch_end", _on_batch_end)
            except Exception:
                pass

        stream = _SignalTextStream(self.terminal_output.emit)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            results = model.train(
                **self._build_train_kwargs(
                    run_name=run_name,
                    epochs=train_epochs,
                    freeze=freeze,
                    lr0=lr0,
                    data_yaml_path=data_yaml_path,
                )
            )
        save_dir = Path(str(getattr(results, "save_dir", self.run_project_dir / run_name))).resolve()
        return {
            "save_dir": str(save_dir),
            "final_metric": dict(train_last_metric_payload or self._last_metric_payload),
        }

    def _extract_metric_row(self, trainer: Any) -> dict[str, float]:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(getattr(trainer, "args", None), "epochs", self.epochs))
        box_loss, cls_loss, dfl_loss = self._extract_loss_components(trainer)
        loss_value = float(box_loss + cls_loss + dfl_loss)
        map50, map50_95 = self._extract_accuracy_values(trainer)
        return {
            "epoch": float(epoch),
            "total_epochs": float(total_epochs),
            "box_loss": float(box_loss),
            "cls_loss": float(cls_loss),
            "dfl_loss": float(dfl_loss),
            "loss": float(loss_value),
            "accuracy": float(map50),
            "map50": float(map50),
            "map50_95": float(map50_95),
        }

    def _extract_loss_components(self, trainer: Any) -> tuple[float, float, float]:
        box_loss = 0.0
        cls_loss = 0.0
        dfl_loss = 0.0

        label_loss_items = getattr(trainer, "label_loss_items", None)
        tloss = getattr(trainer, "tloss", None)
        if callable(label_loss_items):
            try:
                labeled = label_loss_items(tloss, prefix="train")
                if isinstance(labeled, Mapping):
                    box_loss = float(
                        labeled.get(
                            "train/box_loss",
                            labeled.get("box_loss", labeled.get("train/giou_loss", labeled.get("giou_loss", box_loss))),
                        )
                    )
                    cls_loss = float(labeled.get("train/cls_loss", labeled.get("cls_loss", cls_loss)))
                    dfl_loss = float(
                        labeled.get(
                            "train/dfl_loss",
                            labeled.get("dfl_loss", labeled.get("train/l1_loss", labeled.get("l1_loss", dfl_loss))),
                        )
                    )
            except Exception:
                pass

        def _mean_of(values: Any) -> float | None:
            if values is None:
                return None
            try:
                arr = np.asarray(values, dtype=np.float32).reshape(-1)
                if arr.size <= 0:
                    return None
                return float(arr.mean())
            except Exception:
                try:
                    return float(values)
                except Exception:
                    return None

        # metrics dict keys fallback
        metrics_dict = getattr(trainer, "metrics", None)
        if isinstance(metrics_dict, Mapping):
            if self.train_engine == "rtdetr":
                loss_key_map = (
                    ("train/giou_loss", "box"),
                    ("giou_loss", "box"),
                    ("train/box_loss", "box"),
                    ("box_loss", "box"),
                    ("train/cls_loss", "cls"),
                    ("cls_loss", "cls"),
                    ("train/l1_loss", "dfl"),
                    ("l1_loss", "dfl"),
                    ("train/dfl_loss", "dfl"),
                    ("dfl_loss", "dfl"),
                )
            else:
                loss_key_map = (
                    ("train/box_loss", "box"),
                    ("box_loss", "box"),
                    ("train/cls_loss", "cls"),
                    ("cls_loss", "cls"),
                    ("train/dfl_loss", "dfl"),
                    ("dfl_loss", "dfl"),
                )
            for key, target in loss_key_map:
                if key not in metrics_dict:
                    continue
                try:
                    value = max(0.0, float(metrics_dict[key]))
                except Exception:
                    continue
                if not np.isfinite(value):
                    continue
                if target == "box":
                    box_loss = value
                elif target == "cls":
                    cls_loss = value
                else:
                    dfl_loss = value

        # last fallback: unpack loss_items vector [box, cls, dfl]
        if box_loss <= 0.0 and cls_loss <= 0.0 and dfl_loss <= 0.0:
            loss_items = getattr(trainer, "loss_items", None)
            try:
                arr = np.asarray(loss_items, dtype=np.float32).reshape(-1)
                if arr.size >= 3:
                    box_loss = max(0.0, float(arr[0]))
                    cls_loss = max(0.0, float(arr[1]))
                    dfl_loss = max(0.0, float(arr[2]))
            except Exception:
                pass

        # keep finite
        if not np.isfinite(box_loss):
            box_loss = _mean_of(tloss) or 0.0
        if not np.isfinite(cls_loss):
            cls_loss = 0.0
        if not np.isfinite(dfl_loss):
            dfl_loss = 0.0
        return float(max(0.0, box_loss)), float(max(0.0, cls_loss)), float(max(0.0, dfl_loss))

    def _extract_accuracy_values(self, trainer: Any) -> tuple[float, float]:
        map50_value = 0.0
        map50_95_value = 0.0
        validator = getattr(trainer, "validator", None)
        metrics_obj = getattr(validator, "metrics", None)
        for name in ("map50", "mAP50", "box_map50"):
            value = getattr(metrics_obj, name, None)
            if value is not None:
                try:
                    map50_value = float(value)
                    break
                except Exception:
                    pass
        for name in ("map", "mAP50_95", "mAP50-95", "box_map"):
            value = getattr(metrics_obj, name, None)
            if value is not None:
                try:
                    map50_95_value = float(value)
                    break
                except Exception:
                    pass
        metrics_dict = getattr(trainer, "metrics", None)
        if isinstance(metrics_dict, Mapping):
            for key in ("metrics/mAP50(B)", "metrics/mAP50", "mAP50", "map50"):
                if key in metrics_dict and map50_value <= 0.0:
                    try:
                        map50_value = float(metrics_dict[key])
                        break
                    except Exception:
                        pass
            for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "map50_95", "metrics/mAP50_95"):
                if key in metrics_dict and map50_95_value <= 0.0:
                    try:
                        map50_95_value = float(metrics_dict[key])
                        break
                    except Exception:
                        pass
        return float(map50_value), float(map50_95_value)

