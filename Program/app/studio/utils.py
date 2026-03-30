from __future__ import annotations

import contextlib
import csv
import json
import logging
import math
import os
import re
import unicodedata
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.paths import (
    APP_ROOT as PROJECT_ROOT,
    DATASET_SAVE_DIR,
    TRAIN_RTDETR_MODELS_DIR,
    TRAIN_YOLO_MODELS_DIR,
    model_storage_dir_for_name,
)
from app.studio.config import (
    IMAGE_EXTENSIONS_LOWER,
    INVALID_WIN_PATH_CHARS_RE,
    PROVENANCE_JSON_BEGIN,
    PROVENANCE_JSON_END,
)
from app.studio.runtime import ULTRALYTICS_VERSION, yaml

STAGE_NAME_KR: dict[str, str] = {
    "Filtering": "필터링",
    "Detection": "객체 검출",
    "Tracking": "객체 추적",
    "Labeling": "라벨링",
    "Completed": "완료",
}


def _to_korean_stage(stage: str) -> str:
    """영문 단계 키를 UI 표시용 한글 단계명으로 변환해 반환합니다."""
    return STAGE_NAME_KR.get(str(stage), str(stage))


def _to_korean_class_name(name: str) -> str:
    """클래스 이름 표시값을 정리(공백 제거)해 반환합니다."""
    return str(name).strip()

@contextlib.contextmanager
def _temporary_working_directory(path: Path | None):
    if path is None:
        yield
        return
    original = Path.cwd()
    try:
        os.chdir(str(path))
        yield
    finally:
        try:
            os.chdir(str(original))
        except Exception:
            pass


def _resolve_managed_model_source(model_source: str, engine_key: str | None = None) -> tuple[str, Path | None]:
    text = str(model_source or "").strip()
    if not text:
        return "", None
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            return str(candidate.resolve()), None
        except Exception:
            return str(candidate), None
    if candidate.parent != Path("."):
        try:
            return str(candidate.resolve()), None
        except Exception:
            return text, None
    if candidate.suffix.lower() != ".pt":
        return text, None

    target_dir = model_storage_dir_for_name(candidate.name, engine_key=engine_key)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    target_path = target_dir / candidate.name
    stray_root_path = PROJECT_ROOT / candidate.name
    try:
        if stray_root_path.is_file():
            if target_path.exists():
                target_path.unlink()
            stray_root_path.replace(target_path)
    except Exception:
        pass
    if target_path.is_file():
        try:
            return str(target_path.resolve()), None
        except Exception:
            return str(target_path), None
    return candidate.name, target_dir


def sanitize_class_name(name: str, fallback: str = "class") -> str:
    """클래스명을 파일/폴더 이름에 안전한 문자열로 정규화합니다."""
    value = str(name or "").strip()
    if not value:
        return fallback
    value = value.replace(" ", "-")
    value = INVALID_WIN_PATH_CHARS_RE.sub("-", value)
    value = re.sub(r"-+", "-", value).strip("-._ ")
    return value or fallback


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def build_dataset_folder_name(
    class_names: Sequence[str],
    *,
    kind: str,
    created_at: datetime | None = None,
) -> str:
    """클래스 목록과 용도(kind)에 맞춰 데이터셋 폴더명을 생성합니다."""
    ordered = _dedupe_preserve_order([sanitize_class_name(name) for name in class_names])
    if not ordered:
        ordered = ["class"]
    ts = (created_at or datetime.now()).strftime("%Y%m%d_%H%M%S")

    kind_key = str(kind).strip().lower()
    if kind_key == "export":
        prefix = "dataset" if len(ordered) == 1 else "new"
    elif kind_key == "merged":
        prefix = "merged"
    else:
        prefix = "dataset"

    if kind_key == "export" and len(ordered) == 1:
        return f"dataset_{ordered[0]}_{ts}"
    return f"{prefix}_{'_'.join(ordered)}_{ts}"


def _build_unique_path(base_path: Path) -> Path:
    """동일 경로가 존재하면 _001, _002... 접미사를 붙여 유일한 경로를 반환합니다."""
    base = Path(base_path)
    if not base.exists():
        return base
    serial = 1
    while True:
        candidate = base.with_name(f"{base.name}_{serial:03d}")
        if not candidate.exists():
            return candidate
        serial += 1
        if serial > 999:
            return candidate


def _parse_names_from_yaml_payload(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("names")
    if isinstance(raw, Mapping):
        pairs: list[tuple[int, str]] = []
        for key, value in raw.items():
            try:
                idx = int(key)
            except Exception:
                continue
            label = str(value).strip()
            if label:
                pairs.append((idx, label))
        pairs.sort(key=lambda item: item[0])
        return [name for _, name in pairs]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _load_yaml_dict(yaml_path: Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    path = Path(yaml_path)
    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _try_autofix_data_yaml_path(data_yaml_path: Path) -> bool:
    """data.yaml의 path를 YAML 부모 폴더 절대경로로 정규화합니다."""
    if yaml is None:
        return False
    yaml_file = Path(data_yaml_path).resolve()
    if not yaml_file.is_file():
        return False
    payload = _load_yaml_dict(yaml_file)
    if not payload:
        return False

    parent = yaml_file.parent.resolve()
    current_root = _resolve_dataset_root_from_yaml_payload(yaml_file, payload)
    train_ok = (parent / "train" / "images").is_dir()
    valid_ok = (parent / "valid" / "images").is_dir() or (parent / "val" / "images").is_dir()
    if not (train_ok and valid_ok):
        return False
    if current_root == parent and payload.get("path") == parent.as_posix():
        return False

    fixed_payload = dict(payload)
    fixed_payload["path"] = parent.as_posix()
    return _write_yaml_safe(yaml_file, fixed_payload, "data.yaml path auto-fixed", "data.yaml path auto-fix failed")


def _try_autofix_data_yaml_splits(data_yaml_path: Path) -> bool:
    """train/val split이 비어 있으면 비어있지 않은 split 경로로 data.yaml을 보정합니다."""
    if yaml is None:
        return False
    yaml_file = Path(data_yaml_path).resolve()
    if not yaml_file.is_file():
        return False
    payload = _load_yaml_dict(yaml_file)
    if not payload:
        return False

    dataset_root = _resolve_dataset_root_from_yaml_payload(yaml_file, payload)
    fixed_payload = dict(payload)
    changed = False
    valid_key = "val" if ("val" in fixed_payload or "valid" not in fixed_payload) else "valid"

    def _resolve_count(entry: object) -> int:
        return int(len(_resolve_split_images(yaml_file, dataset_root, entry)))

    train_entry = fixed_payload.get("train")
    valid_entry = fixed_payload.get(valid_key)
    test_entry = fixed_payload.get("test")

    train_count = _resolve_count(train_entry)
    valid_count = _resolve_count(valid_entry)
    if train_count <= 0:
        for candidate in (valid_entry, test_entry):
            if _resolve_count(candidate) > 0:
                fixed_payload["train"] = candidate
                train_entry = candidate
                train_count = _resolve_count(candidate)
                changed = True
                break

    if valid_count <= 0:
        for candidate in (train_entry, test_entry):
            if _resolve_count(candidate) > 0:
                fixed_payload[valid_key] = candidate
                valid_count = _resolve_count(candidate)
                changed = True
                break

    if not changed:
        return False
    return _write_yaml_safe(yaml_file, fixed_payload, "data.yaml split auto-fixed", "data.yaml split auto-fix failed")


def _write_yaml_safe(yaml_file: Path, payload: dict[str, Any], ok_msg: str, fail_msg: str) -> bool:
    ## =====================================
    ## 함수 기능 : payload를 YAML 파일로 안전하게 저장하고 성공/실패 로그를 남깁니다
    ## 매개 변수 : yaml_file(Path), payload(dict), ok_msg(str), fail_msg(str)
    ## 반환 결과 : bool -> 저장 성공 여부
    ## =====================================
    try:
        yaml_text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
        yaml_file.write_text(yaml_text, encoding="utf-8")
        logging.getLogger(__name__).info("%s: %s", ok_msg, yaml_file)
        return True
    except Exception as exc:
        logging.getLogger(__name__).warning("%s (%s): %s", fail_msg, yaml_file, exc)
        return False


def _sync_ultralytics_datasets_dir() -> None:
    """Ultralytics 전역 설정의 datasets_dir를 프로젝트 데이터셋 루트로 동기화합니다."""
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        return
    settings_path = Path(appdata) / "Ultralytics" / "settings.json"
    desired = str(DATASET_SAVE_DIR.resolve())
    try:
        payload: dict[str, Any] = {}
        if settings_path.is_file():
            payload_raw = json.loads(settings_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(payload_raw, Mapping):
                payload = dict(payload_raw)
        current = str(payload.get("datasets_dir", "")).strip()
        if current == desired:
            return
        payload["datasets_dir"] = desired
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logging.getLogger(__name__).info("ultralytics settings datasets_dir synced: %s", desired)
    except Exception as exc:
        logging.getLogger(__name__).warning("ultralytics settings sync failed: %s", exc)


def _resolve_dataset_root_from_yaml_payload(yaml_path: Path, payload: Mapping[str, Any]) -> Path:
    root_raw = payload.get("path")
    if root_raw is None:
        return yaml_path.parent.resolve()
    base = Path(str(root_raw))
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    else:
        base = base.resolve()
    return base


def _normalize_split_entries(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (str, Path)):
        text = str(raw).strip()
        return [text] if text else []
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        values: list[str] = []
        for item in raw:
            text = str(item).strip()
            if text:
                values.append(text)
        return values
    return []


def _read_image_list_file(list_path: Path) -> list[Path]:
    lines = []
    try:
        lines = list_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    result: list[Path] = []
    for line in lines:
        text = str(line).strip()
        if not text:
            continue
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = (list_path.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        result.append(candidate)
    return result


def _scan_images_in_directory(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    result: list[Path] = []
    for child in directory.rglob("*"):
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS_LOWER:
            result.append(child.resolve())
    result.sort()
    return result


def _resolve_split_images(
    yaml_path: Path,
    dataset_root: Path,
    split_raw: object,
) -> list[Path]:
    resolved: list[Path] = []
    entries = _normalize_split_entries(split_raw)
    for entry in entries:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = (dataset_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.is_file():
            suffix = candidate.suffix.lower()
            if suffix == ".txt":
                resolved.extend(_read_image_list_file(candidate))
            elif suffix in IMAGE_EXTENSIONS_LOWER:
                resolved.append(candidate)
            continue
        if candidate.is_dir():
            resolved.extend(_scan_images_in_directory(candidate))
    unique: list[Path] = []
    seen: set[str] = set()
    for image_path in resolved:
        key = str(image_path).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(image_path)
    return unique


def _guess_label_path_for_image(image_path: Path) -> Path | None:
    parts_lower = [part.casefold() for part in image_path.parts]
    if "images" in parts_lower:
        idx = len(parts_lower) - 1 - parts_lower[::-1].index("images")
        label_parts = list(image_path.parts)
        label_parts[idx] = "labels"
        candidate = Path(*label_parts).with_suffix(".txt")
        if candidate.is_file():
            return candidate
    sibling = image_path.parent / f"{image_path.stem}.txt"
    if sibling.is_file():
        return sibling
    fallback = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if fallback.is_file():
        return fallback
    return None


def _collect_split_counts_from_yaml(data_yaml_path: Path) -> dict[str, dict[str, int]]:
    payload = _load_yaml_dict(data_yaml_path)
    if not payload:
        return {
            "train": {"images": 0, "labels": 0},
            "valid": {"images": 0, "labels": 0},
            "test": {"images": 0, "labels": 0},
        }
    dataset_root = _resolve_dataset_root_from_yaml_payload(data_yaml_path, payload)
    split_mapping = {
        "train": payload.get("train"),
        "valid": payload.get("val", payload.get("valid")),
        "test": payload.get("test"),
    }
    split_counts: dict[str, dict[str, int]] = {}
    for split_key, split_raw in split_mapping.items():
        images = _resolve_split_images(data_yaml_path, dataset_root, split_raw)
        label_count = 0
        for image_path in images:
            label_path = _guess_label_path_for_image(image_path)
            if label_path is not None and label_path.is_file():
                label_count += 1
        split_counts[split_key] = {"images": int(len(images)), "labels": int(label_count)}
    return split_counts


def _build_component_from_yaml(
    data_yaml_path: Path,
    fallback_class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    yaml_path = Path(data_yaml_path).resolve()
    payload = _load_yaml_dict(yaml_path)
    class_names = _parse_names_from_yaml_payload(payload)
    if not class_names and fallback_class_names:
        class_names = [str(name).strip() for name in fallback_class_names if str(name).strip()]
    class_names = _dedupe_preserve_order(class_names)
    return {
        "dataset_name": str(yaml_path.parent.name),
        "data_yaml_path": str(yaml_path),
        "class_names": class_names,
        "split_counts": _collect_split_counts_from_yaml(yaml_path),
    }


def _normalize_provenance_component(component: Mapping[str, Any]) -> dict[str, Any]:
    dataset_name = str(component.get("dataset_name", component.get("name", "dataset"))).strip() or "dataset"
    yaml_path = str(component.get("data_yaml_path", component.get("yaml_path", ""))).strip()
    class_names_raw = component.get("class_names", component.get("classes", []))
    class_names = (
        [str(item).strip() for item in class_names_raw if str(item).strip()]
        if isinstance(class_names_raw, Sequence) and not isinstance(class_names_raw, (str, bytes, bytearray))
        else []
    )
    split_counts_raw = component.get("split_counts", {})
    split_counts: dict[str, dict[str, int]] = {}
    if isinstance(split_counts_raw, Mapping):
        for split_key in ("train", "valid", "test"):
            data = split_counts_raw.get(split_key, {})
            if isinstance(data, Mapping):
                images = int(data.get("images", 0) or 0)
                labels = int(data.get("labels", 0) or 0)
            else:
                images = 0
                labels = 0
            split_counts[split_key] = {"images": max(0, images), "labels": max(0, labels)}
    else:
        split_counts = {
            "train": {"images": 0, "labels": 0},
            "valid": {"images": 0, "labels": 0},
            "test": {"images": 0, "labels": 0},
        }
    return {
        "dataset_name": dataset_name,
        "data_yaml_path": yaml_path,
        "class_names": class_names,
        "split_counts": split_counts,
    }


def read_provenance_from_readme(readme_path: Path) -> dict[str, Any] | None:
    """README.txt에서 provenance JSON 블록을 읽어 반환합니다."""
    path = Path(readme_path)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    begin_idx = text.find(PROVENANCE_JSON_BEGIN)
    end_idx = text.find(PROVENANCE_JSON_END)
    if begin_idx >= 0 and end_idx > begin_idx:
        json_text = text[begin_idx + len(PROVENANCE_JSON_BEGIN) : end_idx].strip()
        if json_text:
            try:
                payload = json.loads(json_text)
            except Exception:
                payload = None
            if isinstance(payload, Mapping):
                components_raw = payload.get("components", [])
                components: list[dict[str, Any]] = []
                if isinstance(components_raw, Sequence) and not isinstance(components_raw, (str, bytes, bytearray)):
                    for item in components_raw:
                        if isinstance(item, Mapping):
                            components.append(_normalize_provenance_component(item))
                return {"components": components, "raw": dict(payload)}

    # JSON이 없는 과거 README를 위한 best-effort 파싱
    summary_names: list[str] = []
    for line in text.splitlines():
        raw = str(line).strip()
        if raw.startswith("병합 원본:") or raw.startswith("Merged from:"):
            _, _, right = raw.partition(":")
            summary_names = [item.strip() for item in right.split("+") if item.strip()]
            break
    if not summary_names:
        return None
    components = [
        _normalize_provenance_component(
            {
                "dataset_name": name,
                "data_yaml_path": "",
                "class_names": [],
                "split_counts": {},
            }
        )
        for name in summary_names
    ]
    return {"components": components, "raw": {"merged_from_names": summary_names}}


def collect_original_components_from_yaml(
    data_yaml_path: Path,
    fallback_class_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """입력 data.yaml의 원본 구성 데이터셋 리스트(재귀 확장)를 반환합니다."""
    yaml_path = Path(data_yaml_path).resolve()
    readme_payload = read_provenance_from_readme(yaml_path.parent / "README.txt")
    components: list[dict[str, Any]] = []
    if readme_payload is not None:
        for item in readme_payload.get("components", []):
            if isinstance(item, Mapping):
                component = _normalize_provenance_component(item)
                if component["data_yaml_path"]:
                    candidate = Path(component["data_yaml_path"])
                    if candidate.is_file():
                        # 경로가 유효하면 최신 split 카운트를 다시 계산해 정확도를 높입니다.
                        component["split_counts"] = _collect_split_counts_from_yaml(candidate)
                        payload = _load_yaml_dict(candidate)
                        names = _parse_names_from_yaml_payload(payload)
                        if names:
                            component["class_names"] = _dedupe_preserve_order(names)
                components.append(component)
    if not components:
        components.append(_build_component_from_yaml(yaml_path, fallback_class_names=fallback_class_names))

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in components:
        key = (
            f"{str(item.get('data_yaml_path', '')).casefold()}|"
            f"{str(item.get('dataset_name', '')).casefold()}"
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def write_provenance_readme(
    readme_path: Path,
    *,
    created_at: datetime,
    output_folder_name: str,
    class_names: Sequence[str],
    components: Sequence[Mapping[str, Any]],
) -> None:
    """병합 데이터셋 provenance README.txt(한글 + JSON 블록)를 생성합니다."""
    path = Path(readme_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_classes = _dedupe_preserve_order([str(name).strip() for name in class_names if str(name).strip()])
    normalized_components = [_normalize_provenance_component(item) for item in components]
    merged_from_names = [str(item.get("dataset_name", "dataset")).strip() or "dataset" for item in normalized_components]

    lines: list[str] = []
    lines.append("병합 데이터셋 생성 정보")
    lines.append("")
    lines.append(f"생성 시각: {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"출력 폴더명: {output_folder_name}")
    lines.append(f"최종 클래스 목록: {', '.join(ordered_classes) if ordered_classes else '-'}")
    lines.append(f"병합 원본: {' + '.join(merged_from_names) if merged_from_names else '-'}")
    lines.append("")
    lines.append("구성 데이터셋")
    for idx, component in enumerate(normalized_components, start=1):
        split_counts = component.get("split_counts", {})
        train = split_counts.get("train", {"images": 0, "labels": 0})
        valid = split_counts.get("valid", {"images": 0, "labels": 0})
        test = split_counts.get("test", {"images": 0, "labels": 0})
        class_text = ", ".join(component.get("class_names", [])) if component.get("class_names") else "-"
        yaml_text = str(component.get("data_yaml_path", "")).strip() or "(경로 정보 없음)"
        lines.append(f"{idx}. 데이터셋 폴더명: {component.get('dataset_name', 'dataset')}")
        lines.append(f"   - data.yaml 경로: {yaml_text}")
        lines.append(f"   - 클래스 목록: {class_text}")
        lines.append(
            "   - 분할별 이미지/라벨 수: "
            f"train {int(train.get('images', 0))}/{int(train.get('labels', 0))}, "
            f"valid {int(valid.get('images', 0))}/{int(valid.get('labels', 0))}, "
            f"test {int(test.get('images', 0))}/{int(test.get('labels', 0))}"
        )
    lines.append("")
    lines.append(PROVENANCE_JSON_BEGIN)
    payload = {
        "created_at": created_at.isoformat(timespec="seconds"),
        "output_folder_name": str(output_folder_name),
        "class_names": list(ordered_classes),
        "merged_from_names": merged_from_names,
        "components": normalized_components,
    }
    lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
    lines.append(PROVENANCE_JSON_END)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_version_tuple(version_text: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", str(version_text or ""))
    if not numbers:
        return (0,)
    return tuple(int(value) for value in numbers[:4])


def _write_replay_dataset_readme(
    readme_path: Path,
    *,
    created_at: datetime,
    output_folder_name: str,
    class_names: Sequence[str],
    new_components: Sequence[Mapping[str, Any]],
    old_components: Sequence[Mapping[str, Any]],
    replay_ratio_old: float,
    seed: int,
    counts: Mapping[str, Any],
) -> None:
    path = Path(readme_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_classes = _dedupe_preserve_order([str(name).strip() for name in class_names if str(name).strip()])
    normalized_new = [_normalize_provenance_component(item) for item in new_components]
    normalized_old = [_normalize_provenance_component(item) for item in old_components]
    new_names = [str(item.get("dataset_name", "dataset")).strip() or "dataset" for item in normalized_new]
    old_names = [str(item.get("dataset_name", "dataset")).strip() or "dataset" for item in normalized_old]

    replay_count = int(counts.get("old_replay_count", 0) or 0)
    new_train_count = int(counts.get("new_train_count", 0) or 0)
    total_train_count = int(counts.get("merged_train_count", 0) or 0)
    old_available_count = int(counts.get("old_train_available_count", 0) or 0)

    lines: list[str] = []
    lines.append("재학습용 Replay 데이터셋 생성 정보")
    lines.append("")
    lines.append(f"생성 시각: {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"출력 폴더명: {output_folder_name}")
    lines.append(f"최종 클래스 목록: {', '.join(ordered_classes) if ordered_classes else '-'}")
    lines.append(f"신규 데이터셋: {' + '.join(new_names) if new_names else '-'}")
    lines.append(f"기준(구) 데이터셋: {' + '.join(old_names) if old_names else '-'}")
    lines.append(f"replay_ratio_old: {float(replay_ratio_old):.3f}")
    lines.append(f"seed: {int(seed)}")
    lines.append(
        "train 구성: "
        f"new={new_train_count}, replay_old={replay_count}, "
        f"old_available={old_available_count}, merged_total={total_train_count}"
    )
    lines.append("")
    lines.append("신규 데이터셋 구성")
    if normalized_new:
        for idx, component in enumerate(normalized_new, start=1):
            split_counts = component.get("split_counts", {})
            train = split_counts.get("train", {"images": 0, "labels": 0})
            valid = split_counts.get("valid", {"images": 0, "labels": 0})
            test = split_counts.get("test", {"images": 0, "labels": 0})
            lines.append(f"{idx}. 데이터셋 폴더명: {component.get('dataset_name', 'dataset')}")
            lines.append(f"   - data.yaml 경로: {component.get('data_yaml_path', '(경로 정보 없음)')}")
            lines.append(
                "   - 분할별 이미지/라벨 수: "
                f"train {int(train.get('images', 0))}/{int(train.get('labels', 0))}, "
                f"valid {int(valid.get('images', 0))}/{int(valid.get('labels', 0))}, "
                f"test {int(test.get('images', 0))}/{int(test.get('labels', 0))}"
            )
    else:
        lines.append("- 신규 데이터셋 정보를 찾지 못했습니다.")
    lines.append("")
    lines.append("기준(구) 데이터셋 구성")
    if normalized_old:
        for idx, component in enumerate(normalized_old, start=1):
            split_counts = component.get("split_counts", {})
            train = split_counts.get("train", {"images": 0, "labels": 0})
            valid = split_counts.get("valid", {"images": 0, "labels": 0})
            test = split_counts.get("test", {"images": 0, "labels": 0})
            lines.append(f"{idx}. 데이터셋 폴더명: {component.get('dataset_name', 'dataset')}")
            lines.append(f"   - data.yaml 경로: {component.get('data_yaml_path', '(경로 정보 없음)')}")
            lines.append(
                "   - 분할별 이미지/라벨 수: "
                f"train {int(train.get('images', 0))}/{int(train.get('labels', 0))}, "
                f"valid {int(valid.get('images', 0))}/{int(valid.get('labels', 0))}, "
                f"test {int(test.get('images', 0))}/{int(test.get('labels', 0))}"
            )
    else:
        lines.append("- 기준 데이터셋 정보를 찾지 못했습니다.")
    lines.append("")
    lines.append(PROVENANCE_JSON_BEGIN)
    payload = {
        "created_at": created_at.isoformat(timespec="seconds"),
        "output_folder_name": str(output_folder_name),
        "class_names": list(ordered_classes),
        "replay_ratio_old": float(replay_ratio_old),
        "seed": int(seed),
        "counts": {
            "new_train_count": new_train_count,
            "old_train_available_count": old_available_count,
            "old_replay_count": replay_count,
            "merged_train_count": total_train_count,
            "new_valid_count": int(counts.get("new_valid_count", 0) or 0),
            "new_test_count": int(counts.get("new_test_count", 0) or 0),
        },
        "new_components": normalized_new,
        "old_components": normalized_old,
    }
    lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
    lines.append(PROVENANCE_JSON_END)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_to_unit_interval(value: float) -> float:
    v = float(value)
    if v > 1.0 and v <= 100.0:
        v /= 100.0
    return max(0.0, min(1.0, v))


def extract_training_metrics_and_losses(save_dir: Path) -> dict[str, Any]:
    """results.csv 기준으로 best checkpoint 성능/손실 정보를 추출합니다."""
    result: dict[str, Any] = {
        "best_epoch": 0,
        "overall_map50": 0.0,
        "overall_map50_95": 0.0,
        "per_class": [],
        "losses": {},
        "extra_losses": {},
        "derivation": "results.csv에서 best 행 산출 실패",
    }
    csv_path = Path(save_dir) / "results.csv"
    if not csv_path.is_file():
        result["derivation"] = "results.csv가 없어 성능/손실을 추출할 수 없음"
        return result

    best_row: dict[str, str] | None = None
    best_score = float("-inf")
    headers: list[str] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            headers = [str(name).strip() for name in (reader.fieldnames or []) if str(name).strip()]
            header_lookup = {name.casefold(): name for name in headers}

            def _find_header(*candidates: str) -> str | None:
                for candidate in candidates:
                    key = str(candidate).strip().casefold()
                    if key in header_lookup:
                        return header_lookup[key]
                return None

            fitness_key = _find_header("fitness")
            map50_95_key = _find_header("metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50_95", "mAP50-95", "map50_95")
            map50_key = _find_header("metrics/mAP50(B)", "metrics/mAP50", "mAP50", "map50")

            if fitness_key is not None:
                best_score_key = fitness_key
                result["derivation"] = "results.csv에서 fitness가 최대인 행 기준(ultralytics best.pt 기준)"
            elif map50_95_key is not None:
                best_score_key = map50_95_key
                result["derivation"] = "results.csv에서 mAP50-95가 최대인 행 기준"
            elif map50_key is not None:
                best_score_key = map50_key
                result["derivation"] = "results.csv에서 mAP50이 최대인 행 기준"
            else:
                best_score_key = None
                result["derivation"] = "results.csv에 fitness/mAP 컬럼이 없어 마지막 epoch 행 기준"

            last_row: dict[str, str] | None = None
            for row in reader:
                row_dict = {str(key).strip(): str(value).strip() for key, value in dict(row).items() if str(key).strip()}
                if not row_dict:
                    continue
                last_row = row_dict
                if best_score_key is None:
                    continue
                try:
                    score = float(row_dict.get(best_score_key, 0.0) or 0.0)
                except Exception:
                    score = 0.0
                if best_score_key != fitness_key:
                    score = _metric_to_unit_interval(score)
                if score > best_score:
                    best_score = score
                    best_row = row_dict
            if best_row is None:
                best_row = last_row
    except Exception:
        result["derivation"] = "results.csv 파싱 실패"
        return result

    if best_row is None:
        result["derivation"] = "results.csv에 유효한 행이 없음"
        return result

    header_lookup = {name.casefold(): name for name in headers}

    def _find_header(*candidates: str) -> str | None:
        for candidate in candidates:
            key = str(candidate).strip().casefold()
            if key in header_lookup:
                return header_lookup[key]
        return None

    def _row_float(key: str | None, default: float = 0.0) -> float:
        if not key:
            return float(default)
        try:
            return float(best_row.get(key, default) or default)
        except Exception:
            return float(default)

    raw_epoch = _row_float(_find_header("epoch"), 0.0)
    epoch_num = int(round(raw_epoch))
    if epoch_num <= 0:
        epoch_num = max(1, int(round(raw_epoch + 1.0)))
    result["best_epoch"] = epoch_num

    map50 = _row_float(_find_header("metrics/mAP50(B)", "metrics/mAP50", "mAP50", "map50"), 0.0)
    map50_95 = _row_float(_find_header("metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50_95", "mAP50-95", "map50_95"), 0.0)
    result["overall_map50"] = _metric_to_unit_interval(map50)
    result["overall_map50_95"] = _metric_to_unit_interval(map50_95)

    canonical_loss_aliases: dict[str, tuple[str, ...]] = {
        "train_box_loss": ("train/box_loss", "box_loss"),
        "train_cls_loss": ("train/cls_loss", "cls_loss"),
        "train_dfl_loss": ("train/dfl_loss", "dfl_loss"),
        "val_box_loss": ("val/box_loss",),
        "val_cls_loss": ("val/cls_loss",),
        "val_dfl_loss": ("val/dfl_loss",),
    }
    losses = {
        "train_box_loss": 0.0,
        "train_cls_loss": 0.0,
        "train_dfl_loss": 0.0,
        "val_box_loss": 0.0,
        "val_cls_loss": 0.0,
        "val_dfl_loss": 0.0,
    }
    used_loss_headers: set[str] = set()
    for key, aliases in canonical_loss_aliases.items():
        header = _find_header(*aliases)
        if header is None:
            continue
        losses[key] = _row_float(header, 0.0)
        used_loss_headers.add(header)
    result["losses"] = losses

    extra_losses: dict[str, float] = {}
    for header in headers:
        name = str(header).strip()
        if (not name) or (name in used_loss_headers):
            continue
        if "loss" not in name.casefold():
            continue
        extra_losses[name] = _row_float(name, 0.0)
    result["extra_losses"] = extra_losses

    per_class_map: dict[str, dict[str, float]] = {}
    for header in headers:
        m50_match = re.match(r"(?:metrics/)?mAP50\((.+)\)", str(header), flags=re.IGNORECASE)
        if m50_match:
            class_key = str(m50_match.group(1)).strip()
            if class_key.upper() == "B":
                continue
            per_class_map.setdefault(class_key, {})
            per_class_map[class_key]["map50"] = _metric_to_unit_interval(_row_float(header, 0.0))
        m95_match = re.match(r"(?:metrics/)?mAP50-95\((.+)\)", str(header), flags=re.IGNORECASE)
        if m95_match:
            class_key = str(m95_match.group(1)).strip()
            if class_key.upper() == "B":
                continue
            per_class_map.setdefault(class_key, {})
            per_class_map[class_key]["map50_95"] = _metric_to_unit_interval(_row_float(header, 0.0))
    per_class_rows: list[dict[str, Any]] = []
    for class_name, values in per_class_map.items():
        per_class_rows.append(
            {
                "class_name": class_name,
                "map50": float(values.get("map50", 0.0)),
                "map50_95": float(values.get("map50_95", 0.0)),
            }
        )
    result["per_class"] = per_class_rows
    return result


