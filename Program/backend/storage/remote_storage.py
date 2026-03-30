from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from config import apply_remote_env_defaults
from core.paths import PROGRAM_ROOT
from utils.env import env_flag, env_int

# 하드코딩 폴백: config.local.json["remote"]["storage_base_root"] 로 설정 권장
_DEFAULT_REMOTE_STORAGE_BASE_ROOT = "G:/KDT10_3_1team_KLIK/0_Program_"


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _normalize_remote_path(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if text.startswith("/") and len(text) >= 3 and text[1].isalpha() and text[2] == ":":
        text = text[1:]
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/")


def _join_remote_path(base: str, *parts: str) -> str:
    current = _normalize_remote_path(base)
    for raw in parts:
        piece = _normalize_remote_path(raw).strip("/")
        if not piece:
            continue
        current = f"{current}/{piece}" if current else piece
    return _normalize_remote_path(current)


def _remote_parent_path(path_value: str) -> str:
    normalized = _normalize_remote_path(path_value)
    if not normalized:
        return ""
    if "/" not in normalized:
        return normalized
    return normalized.rsplit("/", 1)[0]


@dataclass(slots=True)
class RemoteStorageConfig:
    enabled: bool
    required: bool
    ssh_host: str
    ssh_port: int
    ssh_user: str
    ssh_password: str
    remote_base_root: str
    connect_timeout_sec: float

    @classmethod
    def from_env(cls) -> "RemoteStorageConfig":
        apply_remote_env_defaults()
        base_root = str(
            os.getenv(
                "APP_REMOTE_STORAGE_BASE_ROOT",
                os.getenv("APP_REMOTE_STORAGE_ROOT", _DEFAULT_REMOTE_STORAGE_BASE_ROOT),
            )
        ).strip() or _DEFAULT_REMOTE_STORAGE_BASE_ROOT
        return cls(
            enabled=env_flag("APP_REMOTE_STORAGE_ENABLE", False),
            required=env_flag("APP_REMOTE_STORAGE_REQUIRED", False),
            ssh_host=str(os.getenv("TRAIN_REMOTE_SSH_HOST", os.getenv("LLM_REMOTE_SSH_HOST", ""))).strip(),
            ssh_port=env_int("TRAIN_REMOTE_SSH_PORT", env_int("LLM_REMOTE_SSH_PORT", 8875)),
            ssh_user=str(os.getenv("TRAIN_REMOTE_SSH_USER", os.getenv("LLM_REMOTE_SSH_USER", ""))).strip(),
            ssh_password=str(os.getenv("TRAIN_REMOTE_SSH_PASSWORD", os.getenv("LLM_REMOTE_SSH_PASSWORD", ""))),
            remote_base_root=_normalize_remote_path(base_root),
            connect_timeout_sec=max(3.0, _env_float("APP_REMOTE_STORAGE_CONNECT_TIMEOUT", _env_float("TRAIN_REMOTE_CONNECT_TIMEOUT", 15.0))),
        )

    @property
    def remote_program_root(self) -> str:
        override = str(os.getenv("APP_REMOTE_STORAGE_PROGRAM_ROOT", "")).strip()
        if override:
            return _normalize_remote_path(override)
        return _join_remote_path(self.remote_base_root, PROGRAM_ROOT.name)

    @property
    def remote_assets_root(self) -> str:
        return _join_remote_path(self.remote_program_root, "assets")

    @property
    def remote_dataset_root(self) -> str:
        return _join_remote_path(self.remote_assets_root, "dataset_save_dir")

    @property
    def remote_merged_dataset_root(self) -> str:
        return _join_remote_path(self.remote_dataset_root, "merged_dataset_save_dir")

    @property
    def remote_team_model_root(self) -> str:
        return _join_remote_path(self.remote_assets_root, "models")

    @property
    def remote_yolo_model_root(self) -> str:
        return _join_remote_path(self.remote_team_model_root, "YOLO_models")

    @property
    def remote_rtdetr_model_root(self) -> str:
        return _join_remote_path(self.remote_team_model_root, "RT-DETR_models")

    @property
    def remote_train_runs_root(self) -> str:
        return _join_remote_path(self.remote_program_root, "runs", "train")


def remote_storage_enabled() -> bool:
    return RemoteStorageConfig.from_env().enabled


def remote_path_for_local(local_path: Path) -> str | None:
    try:
        relative = Path(local_path).resolve().relative_to(PROGRAM_ROOT.resolve())
    except Exception:
        return None
    cfg = RemoteStorageConfig.from_env()
    return _join_remote_path(cfg.remote_program_root, relative.as_posix())


def is_remote_program_path(path_value: str) -> bool:
    cfg = RemoteStorageConfig.from_env()
    if not cfg.enabled:
        return False
    normalized = _normalize_remote_path(path_value)
    prefix = _normalize_remote_path(cfg.remote_program_root).casefold()
    return bool(normalized) and normalized.casefold().startswith(prefix)


def _sftp_mkdir_p(sftp: Any, remote_dir: str) -> None:
    target = _normalize_remote_path(remote_dir)
    if not target:
        return
    drive_prefix = ""
    remainder = target
    if len(target) >= 2 and target[1] == ":":
        drive_prefix = target[:2]
        remainder = target[2:].lstrip("/")
    current = drive_prefix
    for piece in [part for part in remainder.split("/") if part]:
        current = f"{current}/{piece}" if current else piece
        try:
            sftp.stat(current)
        except Exception:
            sftp.mkdir(current)


def _sftp_upload_tree(sftp: Any, local_root: Path, remote_root: str) -> None:
    local_base = Path(local_root).resolve()
    _sftp_mkdir_p(sftp, remote_root)
    for local_path in local_base.rglob("*"):
        relative = local_path.relative_to(local_base).as_posix()
        remote_path = _join_remote_path(remote_root, relative)
        if local_path.is_dir():
            _sftp_mkdir_p(sftp, remote_path)
            continue
        _sftp_mkdir_p(sftp, _remote_parent_path(remote_path))
        sftp.put(str(local_path), remote_path)


@contextmanager
def open_remote_storage_session(cfg: RemoteStorageConfig | None = None) -> Iterator[tuple[Any, Any]]:
    active_cfg = cfg or RemoteStorageConfig.from_env()
    if not active_cfg.enabled:
        raise RuntimeError("Remote storage is disabled.")
    try:
        import paramiko  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Remote storage requires paramiko. Install it with: pip install paramiko") from exc

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp = None
    try:
        client.connect(
            hostname=active_cfg.ssh_host,
            port=int(active_cfg.ssh_port),
            username=active_cfg.ssh_user,
            password=active_cfg.ssh_password,
            timeout=float(active_cfg.connect_timeout_sec),
            auth_timeout=max(5.0, float(active_cfg.connect_timeout_sec)),
            banner_timeout=max(5.0, float(active_cfg.connect_timeout_sec)),
            look_for_keys=False,
            allow_agent=False,
        )
        sftp = client.open_sftp()
        yield client, sftp
    finally:
        if sftp is not None:
            try:
                sftp.close()
            except Exception:
                pass
        try:
            client.close()
        except Exception:
            pass


def sync_local_tree_to_remote(local_root: Path) -> str:
    local_dir = Path(local_root).resolve()
    if not local_dir.is_dir():
        raise RuntimeError(f"Local directory not found: {local_dir}")
    remote_root = remote_path_for_local(local_dir)
    if not remote_root:
        raise RuntimeError(f"Could not map local directory to remote storage: {local_dir}")
    cfg = RemoteStorageConfig.from_env()
    with open_remote_storage_session(cfg) as (_client, sftp):
        _sftp_upload_tree(sftp, local_dir, remote_root)
    return remote_root


def sync_local_file_to_remote(local_file: Path) -> str:
    source = Path(local_file).resolve()
    if not source.is_file():
        raise RuntimeError(f"Local file not found: {source}")
    remote_path = remote_path_for_local(source)
    if not remote_path:
        raise RuntimeError(f"Could not map local file to remote storage: {source}")
    cfg = RemoteStorageConfig.from_env()
    with open_remote_storage_session(cfg) as (_client, sftp):
        _sftp_mkdir_p(sftp, _remote_parent_path(remote_path))
        sftp.put(str(source), remote_path)
    return remote_path


def list_remote_training_model_sources(engine_key: str, retrain_mode: bool) -> list[tuple[str, str]]:
    cfg = RemoteStorageConfig.from_env()
    if not cfg.enabled:
        return []

    items: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _append(display: str, source: str) -> None:
        key = _normalize_remote_path(source).casefold()
        if key in seen:
            return
        seen.add(key)
        items.append((display, _normalize_remote_path(source)))

    with open_remote_storage_session(cfg) as (_client, sftp):
        if retrain_mode:
            try:
                for entry in sftp.listdir_attr(cfg.remote_train_runs_root):
                    if not stat.S_ISDIR(entry.st_mode):
                        continue
                    run_dir = _join_remote_path(cfg.remote_train_runs_root, entry.filename)
                    best_path = _join_remote_path(run_dir, "weights", "best.pt")
                    try:
                        best_stat = sftp.stat(best_path)
                    except Exception:
                        continue
                    if stat.S_ISREG(best_stat.st_mode):
                        _append(f"best.pt ({entry.filename}) [server]", best_path)
            except Exception:
                pass
            model_roots = [cfg.remote_yolo_model_root, cfg.remote_rtdetr_model_root]
        else:
            model_roots = [cfg.remote_rtdetr_model_root] if str(engine_key).strip().lower() == "rtdetr" else [cfg.remote_yolo_model_root]

        for root in model_roots:
            try:
                entries = sftp.listdir_attr(root)
            except Exception:
                continue
            for entry in entries:
                if not stat.S_ISREG(entry.st_mode):
                    continue
                if not entry.filename.lower().endswith(".pt"):
                    continue
                path = _join_remote_path(root, entry.filename)
                _append(f"{entry.filename} [server]", path)

    items.sort(key=lambda item: (Path(item[1]).name.casefold(), item[1].casefold()))
    return items
