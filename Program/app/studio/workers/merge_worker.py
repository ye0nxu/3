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

class MultiDatasetMergeWorker(QObject):
    """여러 YOLO data.yaml을 검증/병합해 단일 merged data.yaml을 생성합니다."""

    progress = pyqtSignal(str, int, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def __init__(self, yaml_paths: Sequence[Path], output_root: Path) -> None:
        super().__init__()
        unique: list[Path] = []
        seen: set[str] = set()
        for raw in yaml_paths:
            path = Path(raw).resolve()
            key = str(path).casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        self.yaml_paths = unique
        self.output_root = Path(output_root)

    @pyqtSlot()
    def run(self) -> None:
        try:
            if yaml is None:
                raise RuntimeError("PyYAML 모듈을 찾을 수 없습니다. yaml 파싱을 진행할 수 없습니다.")
            if not self.yaml_paths:
                raise RuntimeError("병합할 data.yaml이 선택되지 않았습니다.")

            specs: list[dict[str, Any]] = []
            total_yaml = len(self.yaml_paths)
            self.progress.emit("Validating", 0, max(1, total_yaml))
            for idx, yaml_path in enumerate(self.yaml_paths, start=1):
                spec = self._load_yaml_spec(yaml_path)
                specs.append(spec)
                self.progress.emit("Validating", idx, max(1, total_yaml))

            names_ref: list[str] = []
            for spec in specs:
                names_ref = self._merge_class_names(
                    names_ref,
                    list(spec["names"]),
                    Path(spec["yaml_path"]),
                )
            if not names_ref:
                raise RuntimeError("병합할 클래스 정보(names)를 찾지 못했습니다.")

            merged_name = self._build_merged_dataset_name(names_ref)
            merged_root = _build_unique_path(self.output_root / merged_name)
            merged_root.mkdir(parents=True, exist_ok=True)

            split_dirs = self._prepare_merged_split_dirs(merged_root)
            total_items = sum(
                len(spec["pairs"].get(split_key, []))
                for spec in specs
                for split_key in ("train", "valid", "test")
            )
            done = 0
            self.progress.emit("Merging", 0, max(1, total_items))
            used_stems: dict[str, set[str]] = {k: set() for k in ("train", "valid", "test")}
            for dataset_idx, spec in enumerate(specs, start=1):
                source_name = self._dataset_base_name_from_yaml(spec["yaml_path"])
                prefix = f"d{dataset_idx:02d}_{source_name}"
                for split_key in ("train", "valid", "test"):
                    pairs = list(spec["pairs"].get(split_key, []))
                    if not pairs:
                        continue
                    image_dir = split_dirs[split_key]["images"]
                    label_dir = split_dirs[split_key]["labels"]
                    for image_path, label_path in pairs:
                        image_src = Path(image_path)
                        ext = image_src.suffix.lower() or ".jpg"
                        if ext not in self.IMAGE_EXTENSIONS:
                            ext = ".jpg"
                        candidate_stem = f"{prefix}_{image_src.stem}"
                        safe_stem = self._allocate_unique_stem(
                            candidate_stem,
                            used_stems[split_key],
                            image_dir,
                            ext,
                        )
                        image_dst = image_dir / f"{safe_stem}{ext}"
                        label_dst = label_dir / f"{safe_stem}.txt"
                        self._link_or_copy_file(image_src, image_dst)
                        if label_path is not None and Path(label_path).is_file():
                            self._link_or_copy_file(Path(label_path), label_dst)
                        else:
                            label_dst.write_text("", encoding="utf-8")
                        done += 1
                        if done % 40 == 0 or done >= total_items:
                            self.progress.emit("Merging", done, max(1, total_items))

            self.progress.emit("Writing YAML", 0, 1)
            merged_yaml_path = self._write_merged_yaml(merged_root, names_ref)
            self.progress.emit("Writing YAML", 1, 1)
            provenance_components = self._build_provenance_components(specs)
            self._write_merged_readme(
                merged_root=merged_root,
                class_names=names_ref,
                components=provenance_components,
            )
            self.progress.emit("Done", 1, 1)
            logging.getLogger(__name__).info(
                "dataset merge completed: output=%s, yaml_count=%d",
                merged_root,
                len(specs),
            )
            self.finished.emit(
                {
                    "merged_yaml_path": str(merged_yaml_path),
                    "merged_dataset_root": str(merged_root),
                    "run_base_name": str(merged_root.name),
                    "yaml_count": int(len(specs)),
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))

    def _load_yaml_spec(self, yaml_path: Path) -> dict[str, Any]:
        path = Path(yaml_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"YAML 파일이 없습니다: {path}")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as exc:
            raise RuntimeError(f"YAML 파싱 실패: {path}\n{exc}") from exc
        if not isinstance(data, Mapping):
            raise RuntimeError(f"YOLO data.yaml 형식이 아닙니다: {path}")

        names_raw = data.get("names")
        nc_raw = data.get("nc")
        names = self._parse_yaml_names(names_raw)
        if not names:
            raise RuntimeError(f"YOLO names 정보가 없습니다: {path}")
        try:
            nc = int(nc_raw if nc_raw is not None else len(names))
        except Exception:
            nc = len(names)
        if nc != len(names):
            raise RuntimeError(
                f"YAML 내 nc와 names 길이가 일치하지 않습니다: {path}\n"
                f"nc={nc}, len(names)={len(names)}"
            )

        dataset_root = self._resolve_dataset_root(path, data.get("path"))
        train_entry = data.get("train")
        valid_entry = data.get("val", data.get("valid"))
        test_entry = data.get("test")

        train_pairs = self._resolve_split_pairs(path, dataset_root, train_entry, "train", required=True)
        valid_pairs = self._resolve_split_pairs(path, dataset_root, valid_entry, "valid", required=True)
        test_pairs: list[tuple[Path, Path | None]] = []
        if test_entry is not None:
            test_pairs = self._resolve_split_pairs(path, dataset_root, test_entry, "test", required=True)

        return {
            "yaml_path": path,
            "dataset_root": dataset_root,
            "names": names,
            "nc": nc,
            "pairs": {
                "train": train_pairs,
                "valid": valid_pairs,
                "test": test_pairs,
            },
        }

    def _merge_class_names(
        self,
        existing: Sequence[str],
        incoming: Sequence[str],
        source_path: Path | None = None,
    ) -> list[str]:
        base = [str(name).strip() for name in existing if str(name).strip()]
        add = [str(name).strip() for name in incoming if str(name).strip()]
        if not add:
            return base
        if not base:
            return add

        merged = list(base)
        for idx, name in enumerate(add):
            if idx < len(merged):
                if merged[idx] != name:
                    source_text = f" ({source_path})" if source_path is not None else ""
                    raise RuntimeError(
                        f"class index mismatch at {idx}: '{merged[idx]}' != '{name}'{source_text}"
                    )
            else:
                merged.append(name)
        return merged

    def _resolve_dataset_root(self, yaml_path: Path, root_raw: object) -> Path:
        if root_raw is None:
            return yaml_path.parent.resolve()
        base = Path(str(root_raw))
        if not base.is_absolute():
            base = (yaml_path.parent / base).resolve()
        return base

    def _resolve_split_pairs(
        self,
        yaml_path: Path,
        dataset_root: Path,
        split_raw: object,
        split_name: str,
        required: bool,
    ) -> list[tuple[Path, Path | None]]:
        entries = self._normalize_split_entries(split_raw)
        if required and (not entries):
            raise RuntimeError(f"{split_name} 경로가 비어 있습니다: {yaml_path}")

        image_paths: list[Path] = []
        for entry in entries:
            source_path = Path(entry)
            if not source_path.is_absolute():
                source_path = (dataset_root / source_path).resolve()
            if source_path.is_file():
                if source_path.suffix.lower() == ".txt":
                    image_paths.extend(self._read_image_list_file(source_path))
                    continue
                if source_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                    image_paths.append(source_path)
                    continue
                raise RuntimeError(f"{split_name} 경로 형식을 지원하지 않습니다: {source_path}")
            if source_path.is_dir():
                image_paths.extend(self._scan_images_in_directory(source_path))
                continue
            raise RuntimeError(f"{split_name} 경로를 찾을 수 없습니다: {source_path}")

        pairs: list[tuple[Path, Path | None]] = []
        for image_path in image_paths:
            resolved_image = Path(image_path).resolve()
            if not resolved_image.is_file():
                continue
            label_path = self._guess_label_path(resolved_image)
            pairs.append((resolved_image, label_path))
        if required and (not pairs):
            raise RuntimeError(f"{split_name}에서 병합할 이미지를 찾지 못했습니다: {yaml_path}")
        return pairs

    def _normalize_split_entries(self, split_raw: object) -> list[str]:
        if split_raw is None:
            return []
        if isinstance(split_raw, (str, Path)):
            text = str(split_raw).strip()
            return [text] if text else []
        if isinstance(split_raw, Sequence) and not isinstance(split_raw, (bytes, bytearray)):
            values: list[str] = []
            for item in split_raw:
                text = str(item).strip()
                if text:
                    values.append(text)
            return values
        return []

    def _read_image_list_file(self, list_path: Path) -> list[Path]:
        paths: list[Path] = []
        try:
            lines = list_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as exc:
            raise RuntimeError(f"이미지 목록 파일을 읽을 수 없습니다: {list_path}\n{exc}") from exc
        for line in lines:
            text = str(line).strip()
            if not text:
                continue
            candidate = Path(text)
            if not candidate.is_absolute():
                candidate = (list_path.parent / candidate).resolve()
            paths.append(candidate)
        return paths

    def _scan_images_in_directory(self, directory: Path) -> list[Path]:
        result: list[Path] = []
        for child in directory.rglob("*"):
            if not child.is_file():
                continue
            if child.suffix.lower() in self.IMAGE_EXTENSIONS:
                result.append(child.resolve())
        result.sort()
        return result

    def _guess_label_path(self, image_path: Path) -> Path | None:
        parts_lower = [part.casefold() for part in image_path.parts]
        if "images" in parts_lower:
            idx = len(parts_lower) - 1 - parts_lower[::-1].index("images")
            label_parts = list(image_path.parts)
            label_parts[idx] = "labels"
            candidate = Path(*label_parts).with_suffix(".txt")
            if candidate.is_file():
                return candidate

        sibling_label = image_path.parent / f"{image_path.stem}.txt"
        if sibling_label.is_file():
            return sibling_label

        parent_label = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if parent_label.is_file():
            return parent_label
        return None

    def _build_merged_dataset_name(self, class_names: Sequence[str]) -> str:
        return build_dataset_folder_name(class_names, kind="merged", created_at=datetime.now())

    def _build_provenance_components(self, specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        components: list[dict[str, Any]] = []
        seen: set[str] = set()
        for spec in specs:
            yaml_path = Path(spec.get("yaml_path", "")).resolve()
            fallback_names = spec.get("names", [])
            expanded = collect_original_components_from_yaml(yaml_path, fallback_class_names=fallback_names)
            for item in expanded:
                key = (
                    f"{str(item.get('data_yaml_path', '')).casefold()}|"
                    f"{str(item.get('dataset_name', '')).casefold()}"
                )
                if key in seen:
                    continue
                seen.add(key)
                components.append(dict(item))
        return components

    def _write_merged_readme(
        self,
        *,
        merged_root: Path,
        class_names: Sequence[str],
        components: Sequence[Mapping[str, Any]],
    ) -> None:
        write_provenance_readme(
            merged_root / "README.txt",
            created_at=datetime.now(),
            output_folder_name=str(merged_root.name),
            class_names=class_names,
            components=components,
        )

    def _dataset_base_name_from_yaml(self, yaml_path: Path) -> str:
        path = Path(yaml_path)
        parent_name = path.parent.name.strip()
        stem = path.stem.strip()
        if stem.casefold() == "data" and parent_name:
            return self._sanitize_name(parent_name, fallback="dataset")
        if stem:
            return self._sanitize_name(stem, fallback="dataset")
        return self._sanitize_name(parent_name, fallback="dataset")

    def _sanitize_name(self, text: str, fallback: str = "item") -> str:
        value = str(text or "").strip().casefold()
        value = re.sub(r"[^a-z0-9]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        if not value:
            return fallback
        if value[0].isdigit():
            return f"{fallback}_{value}"
        return value

    def _prepare_merged_split_dirs(self, merged_root: Path) -> dict[str, dict[str, Path]]:
        split_dirs: dict[str, dict[str, Path]] = {}
        for split_key in ("train", "valid", "test"):
            image_dir = merged_root / split_key / "images"
            label_dir = merged_root / split_key / "labels"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            split_dirs[split_key] = {"images": image_dir, "labels": label_dir}
        return split_dirs

    def _allocate_unique_stem(
        self,
        base_stem: str,
        used_stems: set[str],
        target_image_dir: Path,
        extension: str,
    ) -> str:
        candidate = str(base_stem).strip()
        if not candidate:
            candidate = "sample"
        serial = 2
        while (
            candidate in used_stems
            or (target_image_dir / f"{candidate}{extension}").exists()
            or (target_image_dir / f"{candidate}.jpg").exists()
        ):
            candidate = f"{base_stem}_{serial:03d}"
            serial += 1
        used_stems.add(candidate)
        return candidate

    def _link_or_copy_file(self, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        try:
            os.link(str(source), str(target))
            return
        except Exception:
            pass
        shutil.copy2(str(source), str(target))

    def _write_merged_yaml(
        self,
        merged_root: Path,
        names: Sequence[str],
    ) -> Path:
        yaml_text = (
            f"path: {merged_root.as_posix()}\n"
            "train: train/images\n"
            "val: valid/images\n"
            "test: test/images\n"
            f"nc: {len(names)}\n"
            "names:\n"
            + "\n".join([f"  {idx}: {name}" for idx, name in enumerate(names)])
            + ("\n" if names else "")
        )
        merged_yaml = merged_root / "data.yaml"
        merged_yaml.write_text(yaml_text, encoding="utf-8")
        classes_text = "\n".join(f"{idx}: {name}" for idx, name in enumerate(names))
        (merged_root / "classes.txt").write_text(classes_text + ("\n" if classes_text else ""), encoding="utf-8")
        return merged_yaml

    def _parse_yaml_names(self, names_raw: object) -> list[str]:
        if isinstance(names_raw, Mapping):
            values: list[tuple[int, str]] = []
            for key, value in names_raw.items():
                try:
                    idx = int(key)
                except Exception:
                    continue
                values.append((idx, str(value)))
            values.sort(key=lambda item: item[0])
            return [name.strip() for _, name in values if str(name).strip()]
        if isinstance(names_raw, Sequence) and not isinstance(names_raw, (str, bytes, bytearray)):
            return [str(v).strip() for v in names_raw if str(v).strip()]
        return []

class ReplayDatasetMergeWorker(MultiDatasetMergeWorker):
    """재학습용으로 신규 데이터 + 구 데이터 replay 샘플을 병합합니다."""

    def __init__(
        self,
        new_yaml_paths: Sequence[Path],
        old_yaml_path: Path,
        output_root: Path,
        *,
        replay_ratio_old: float = TRAIN_REPLAY_RATIO_DEFAULT,
        seed: int = TRAIN_RETRAIN_SEED_DEFAULT,
    ) -> None:
        super().__init__(yaml_paths=new_yaml_paths, output_root=output_root)
        self.old_yaml_path = Path(old_yaml_path).resolve()
        self.replay_ratio_old = max(0.0, min(0.95, float(replay_ratio_old)))
        self.seed = int(seed)

    @pyqtSlot()
    def run(self) -> None:
        try:
            if yaml is None:
                raise RuntimeError("PyYAML 모듈을 찾을 수 없습니다. yaml 파싱을 진행할 수 없습니다.")
            if not self.yaml_paths:
                raise RuntimeError("재학습용 신규 data.yaml이 선택되지 않았습니다.")
            if not self.old_yaml_path.is_file():
                raise RuntimeError(f"기준(구) data.yaml 파일이 없습니다: {self.old_yaml_path}")

            total_yaml = len(self.yaml_paths) + 1
            self.progress.emit("Validating", 0, max(1, total_yaml))

            new_specs: list[dict[str, Any]] = []
            for idx, yaml_path in enumerate(self.yaml_paths, start=1):
                spec = self._load_yaml_spec(yaml_path)
                new_specs.append(spec)
                self.progress.emit("Validating", idx, max(1, total_yaml))

            old_spec = self._load_yaml_spec(self.old_yaml_path)
            self.progress.emit("Validating", total_yaml, max(1, total_yaml))

            names_ref: list[str] = []
            for spec in [*new_specs, old_spec]:
                names_ref = self._merge_class_names(
                    names_ref,
                    list(spec["names"]),
                    Path(spec["yaml_path"]),
                )
            if not names_ref:
                raise RuntimeError("재학습 병합용 클래스 정보(names)를 찾지 못했습니다.")

            merged_name = self._build_merged_dataset_name(names_ref)
            merged_root = _build_unique_path(self.output_root / merged_name)
            merged_root.mkdir(parents=True, exist_ok=True)
            split_dirs = self._prepare_merged_split_dirs(merged_root)

            new_train_count = sum(len(spec["pairs"].get("train", [])) for spec in new_specs)
            old_train_pairs = list(old_spec["pairs"].get("train", []))
            old_train_available_count = len(old_train_pairs)
            old_replay_count = self._calculate_replay_sample_count(
                new_train_count=new_train_count,
                old_train_available_count=old_train_available_count,
            )
            replay_pairs = self._sample_replay_pairs(old_train_pairs, old_replay_count)
            new_valid_count = sum(len(spec["pairs"].get("valid", [])) for spec in new_specs)
            new_test_count = sum(len(spec["pairs"].get("test", [])) for spec in new_specs)
            total_items = new_train_count + len(replay_pairs) + new_valid_count + new_test_count
            done = 0
            self.progress.emit("Merging", 0, max(1, total_items))

            used_stems: dict[str, set[str]] = {k: set() for k in ("train", "valid", "test")}
            for dataset_idx, spec in enumerate(new_specs, start=1):
                source_name = self._dataset_base_name_from_yaml(spec["yaml_path"])
                prefix = f"new{dataset_idx:02d}_{source_name}"
                done = self._copy_pairs_into_split(
                    pairs=spec["pairs"].get("train", []),
                    split_key="train",
                    prefix=prefix,
                    split_dirs=split_dirs,
                    used_stems=used_stems,
                    done=done,
                    total=max(1, total_items),
                )
            done = self._copy_pairs_into_split(
                pairs=replay_pairs,
                split_key="train",
                prefix=f"old_{self._dataset_base_name_from_yaml(old_spec['yaml_path'])}",
                split_dirs=split_dirs,
                used_stems=used_stems,
                done=done,
                total=max(1, total_items),
            )
            for dataset_idx, spec in enumerate(new_specs, start=1):
                source_name = self._dataset_base_name_from_yaml(spec["yaml_path"])
                prefix = f"new{dataset_idx:02d}_{source_name}"
                done = self._copy_pairs_into_split(
                    pairs=spec["pairs"].get("valid", []),
                    split_key="valid",
                    prefix=prefix,
                    split_dirs=split_dirs,
                    used_stems=used_stems,
                    done=done,
                    total=max(1, total_items),
                )
                done = self._copy_pairs_into_split(
                    pairs=spec["pairs"].get("test", []),
                    split_key="test",
                    prefix=prefix,
                    split_dirs=split_dirs,
                    used_stems=used_stems,
                    done=done,
                    total=max(1, total_items),
                )

            self.progress.emit("Writing YAML", 0, 2)
            merged_yaml_path = self._write_merged_yaml(merged_root, names_ref)
            old_eval_yaml_path = self._write_alias_yaml(
                target_path=merged_root / "old_eval.yaml",
                spec=old_spec,
                names=names_ref,
            )
            self.progress.emit("Writing YAML", 2, 2)

            new_components = self._build_provenance_components(new_specs)
            old_components = self._build_provenance_components([old_spec])
            counts = {
                "new_train_count": int(new_train_count),
                "old_train_available_count": int(old_train_available_count),
                "old_replay_count": int(len(replay_pairs)),
                "merged_train_count": int(new_train_count + len(replay_pairs)),
                "new_valid_count": int(new_valid_count),
                "new_test_count": int(new_test_count),
            }
            _write_replay_dataset_readme(
                merged_root / "README.txt",
                created_at=datetime.now(),
                output_folder_name=str(merged_root.name),
                class_names=names_ref,
                new_components=new_components,
                old_components=old_components,
                replay_ratio_old=self.replay_ratio_old,
                seed=self.seed,
                counts=counts,
            )
            self.progress.emit("Done", 1, 1)
            logging.getLogger(__name__).info(
                "replay dataset merge completed: output=%s new_yaml_count=%d replay_old=%d/%d",
                merged_root,
                len(new_specs),
                len(replay_pairs),
                old_train_available_count,
            )
            self.finished.emit(
                {
                    "merged_yaml_path": str(merged_yaml_path),
                    "merged_dataset_root": str(merged_root),
                    "run_base_name": str(merged_root.name),
                    "yaml_count": int(len(new_specs)),
                    "new_yaml_paths": [str(Path(spec["yaml_path"]).resolve()) for spec in new_specs],
                    "old_yaml_path": str(self.old_yaml_path),
                    "old_eval_yaml_path": str(old_eval_yaml_path),
                    "replay_ratio_old": float(self.replay_ratio_old),
                    "seed": int(self.seed),
                    **counts,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))

    def _calculate_replay_sample_count(self, *, new_train_count: int, old_train_available_count: int) -> int:
        if new_train_count <= 0 or old_train_available_count <= 0 or self.replay_ratio_old <= 0.0:
            return 0
        denominator = max(0.000001, 1.0 - float(self.replay_ratio_old))
        target = int(round((float(new_train_count) * float(self.replay_ratio_old)) / denominator))
        if target <= 0:
            target = 1
        return max(0, min(target, int(old_train_available_count)))

    def _sample_replay_pairs(
        self,
        pairs: Sequence[tuple[Path, Path | None]],
        sample_count: int,
    ) -> list[tuple[Path, Path | None]]:
        items = [
            (Path(image_path).resolve(), Path(label_path).resolve() if label_path is not None else None)
            for image_path, label_path in pairs
        ]
        items.sort(key=lambda item: str(item[0]).casefold())
        sample_count = max(0, min(int(sample_count), len(items)))
        if sample_count <= 0:
            return []
        if sample_count >= len(items):
            return list(items)
        rng = random.Random(self.seed)
        selected = rng.sample(items, sample_count)
        selected.sort(key=lambda item: str(item[0]).casefold())
        return selected

    def _copy_pairs_into_split(
        self,
        *,
        pairs: Sequence[tuple[Path, Path | None]],
        split_key: str,
        prefix: str,
        split_dirs: Mapping[str, Mapping[str, Path]],
        used_stems: dict[str, set[str]],
        done: int,
        total: int,
    ) -> int:
        pair_list = list(pairs)
        if not pair_list:
            return done
        image_dir = Path(split_dirs[split_key]["images"])
        label_dir = Path(split_dirs[split_key]["labels"])
        for image_path, label_path in pair_list:
            image_src = Path(image_path).resolve()
            ext = image_src.suffix.lower() or ".jpg"
            if ext not in self.IMAGE_EXTENSIONS:
                ext = ".jpg"
            safe_stem = self._allocate_unique_stem(
                f"{prefix}_{image_src.stem}",
                used_stems[split_key],
                image_dir,
                ext,
            )
            image_dst = image_dir / f"{safe_stem}{ext}"
            label_dst = label_dir / f"{safe_stem}.txt"
            self._link_or_copy_file(image_src, image_dst)
            if label_path is not None and Path(label_path).is_file():
                self._link_or_copy_file(Path(label_path), label_dst)
            else:
                label_dst.write_text("", encoding="utf-8")
            done += 1
            if done % 40 == 0 or done >= total:
                self.progress.emit("Merging", done, total)
        return done

    def _write_alias_yaml(
        self,
        *,
        target_path: Path,
        spec: Mapping[str, Any],
        names: Sequence[str],
    ) -> Path:
        yaml_path = Path(target_path)
        dataset_root = Path(spec.get("dataset_root", yaml_path.parent)).resolve()
        payload: dict[str, Any] = {
            "path": dataset_root.as_posix(),
            "train": "train/images",
            "val": "valid/images",
            "nc": int(len(names)),
            "names": {idx: name for idx, name in enumerate(names)},
        }
        source_yaml = Path(spec.get("yaml_path", ""))
        source_payload = _load_yaml_dict(source_yaml)
        train_entry = source_payload.get("train")
        valid_entry = source_payload.get("val", source_payload.get("valid"))
        test_entry = source_payload.get("test")
        if train_entry is not None:
            payload["train"] = train_entry
        if valid_entry is not None:
            payload["val"] = valid_entry
        if test_entry is not None:
            payload["test"] = test_entry
        yaml_text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
        yaml_path.write_text(yaml_text, encoding="utf-8")
        return yaml_path

