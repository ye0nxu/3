"""원격 서버 및 환경 설정 로더.

config.local.json을 읽어 환경변수에 기본값을 주입합니다.
하드코딩된 경로 기본값은 config.local.json이 없을 때만 사용하는 최후 수단입니다.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

APP_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_ROOT = APP_ROOT.parent

# 최후 수단 기본값 - config.local.json 으로 반드시 덮어쓰기 권장
_DEFAULT_REMOTE_STORAGE_BASE_ROOT = "G:/KDT10_3_1team_KLIK/0_Program_"
_DEFAULT_REMOTE_LLM_MODEL = "G:/models/Qwen2.5-7B-Instruct"
# _DEFAULT_REMOTE_LLM_MODEL = "G:/models/Qwen2.5-14B-Instruct"
_DEFAULT_REMOTE_SAM3_ROOT = "G:/models/sam3"
_DEFAULT_REMOTE_PYTHON = "G:/conda/envs/PJ_310_LLM_SAM3/python.exe"

_CONFIG_CACHE: dict[str, Any] | None = None


def local_config_path() -> Path:
    env_path = str(os.getenv("APP_LOCAL_CONFIG", "")).strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (APP_ROOT / "config.local.json").resolve()


def load_local_config(force_reload: bool = False) -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return dict(_CONFIG_CACHE)
    config_path = local_config_path()
    if not config_path.is_file():
        _CONFIG_CACHE = {}
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("config.local.json 파싱 오류 - 기본값 사용: %s", exc)
        payload = {}
    except OSError as exc:
        logger.warning("config.local.json 읽기 실패 - 기본값 사용: %s", exc)
        payload = {}
    _CONFIG_CACHE = payload if isinstance(payload, dict) else {}
    return dict(_CONFIG_CACHE)


def _nested_get(payload: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _env_setdefault(name: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    os.environ.setdefault(name, text)


def _json_value(*path: str, default: Any = None) -> Any:
    payload = load_local_config()
    return _nested_get(payload, path, default=default)


def apply_remote_env_defaults(program_root: Path | None = None) -> None:
    active_program_root = Path(program_root or PROGRAM_ROOT).resolve()
    program_name = active_program_root.name

    ssh_host = _json_value("remote", "ssh", "host", default="")
    ssh_port = _json_value("remote", "ssh", "port", default="")
    ssh_user = _json_value("remote", "ssh", "user", default="")
    ssh_password = _json_value("remote", "ssh", "password", default="")
    llm_python = _json_value("remote", "python", "llm", default=_DEFAULT_REMOTE_PYTHON)
    sam3_python = _json_value("remote", "python", "sam3", default=llm_python or _DEFAULT_REMOTE_PYTHON)
    train_python = _json_value("remote", "python", "train", default=llm_python or _DEFAULT_REMOTE_PYTHON)
    storage_base_root = (
        str(_json_value("remote", "storage_base_root", default=_DEFAULT_REMOTE_STORAGE_BASE_ROOT)).strip()
        or _DEFAULT_REMOTE_STORAGE_BASE_ROOT
    )
    remote_program_root = str(_json_value("remote", "program_root", default="")).strip()
    if not remote_program_root:
        remote_program_root = f"{storage_base_root.rstrip('/\\\\')}/{program_name}"
    llm_model = str(_json_value("remote", "models", "llm", default=_DEFAULT_REMOTE_LLM_MODEL)).strip() or _DEFAULT_REMOTE_LLM_MODEL
    sam3_root = str(_json_value("remote", "models", "sam3_root", default=_DEFAULT_REMOTE_SAM3_ROOT)).strip() or _DEFAULT_REMOTE_SAM3_ROOT

    for env_name in ("LLM_REMOTE_SSH_HOST", "TRAIN_REMOTE_SSH_HOST", "NEW_OBJECT_REMOTE_SSH_HOST"):
        _env_setdefault(env_name, ssh_host)
    for env_name in ("LLM_REMOTE_SSH_PORT", "TRAIN_REMOTE_SSH_PORT", "NEW_OBJECT_REMOTE_SSH_PORT"):
        _env_setdefault(env_name, ssh_port)
    for env_name in ("LLM_REMOTE_SSH_USER", "TRAIN_REMOTE_SSH_USER", "NEW_OBJECT_REMOTE_SSH_USER"):
        _env_setdefault(env_name, ssh_user)
    for env_name in ("LLM_REMOTE_SSH_PASSWORD", "TRAIN_REMOTE_SSH_PASSWORD", "NEW_OBJECT_REMOTE_SSH_PASSWORD"):
        _env_setdefault(env_name, ssh_password)

    _env_setdefault("APP_REMOTE_STORAGE_BASE_ROOT", storage_base_root)
    _env_setdefault("APP_REMOTE_STORAGE_PROGRAM_ROOT", remote_program_root)
    _env_setdefault("LLM_REMOTE_WORKDIR", remote_program_root)
    _env_setdefault("TRAIN_REMOTE_WORKDIR", remote_program_root)
    _env_setdefault("NEW_OBJECT_REMOTE_WORKDIR", remote_program_root)

    _env_setdefault("LLM_REMOTE_PYTHON_CMD", llm_python)
    _env_setdefault("TRAIN_REMOTE_PYTHON_CMD", train_python)
    _env_setdefault("NEW_OBJECT_REMOTE_PYTHON_CMD", sam3_python)
    _env_setdefault("LLM_REMOTE_MODEL_ID", llm_model)
    _env_setdefault("NEW_OBJECT_REMOTE_SAM3_ROOT", sam3_root)

    llm_enabled = _json_value("remote", "features", "llm_enabled", default=None)
    if llm_enabled is not None:
        _env_setdefault("LLM_REMOTE_ENABLE", "1" if bool(llm_enabled) else "0")
        _env_setdefault("LLM_REMOTE_DIRECT_ENABLE", "1" if bool(llm_enabled) else "0")
    sam3_enabled = _json_value("remote", "features", "sam3_enabled", default=None)
    if sam3_enabled is not None:
        _env_setdefault("NEW_OBJECT_REMOTE_ENABLE", "1" if bool(sam3_enabled) else "0")
    train_enabled = _json_value("remote", "features", "train_enabled", default=None)
    if train_enabled is not None:
        _env_setdefault("TRAIN_REMOTE_ENABLE", "1" if bool(train_enabled) else "0")

    _env_setdefault("LLM_SERVER_AUTOSTART", "0")


def missing_remote_ssh_fields() -> list[str]:
    apply_remote_env_defaults()
    missing: list[str] = []
    mapping = {
        "host": os.getenv("LLM_REMOTE_SSH_HOST", ""),
        "port": os.getenv("LLM_REMOTE_SSH_PORT", ""),
        "user": os.getenv("LLM_REMOTE_SSH_USER", ""),
        "password": os.getenv("LLM_REMOTE_SSH_PASSWORD", ""),
    }
    for key, value in mapping.items():
        if not str(value).strip():
            missing.append(key)
    return missing
