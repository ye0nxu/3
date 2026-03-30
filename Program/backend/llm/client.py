from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import apply_remote_env_defaults
from config.settings import missing_remote_ssh_fields
from backend.llm.prompting import heuristic_english_candidates, normalize_user_text
from core.paths import PREVIEW_CACHE_BASE
from utils.env import env_flag, env_int


REMOTE_READY_PREFIX = "__LLM_REMOTE_READY__="
REMOTE_RESULT_PREFIX = "__LLM_REMOTE_RESULT__="
RUNTIME_CORE_MODULE_NAME = "runtime"  # 구 runtime_core -> runtime
# 하드코딩 폴백: config.local.json 으로 반드시 덮어쓰기 권장
_DEFAULT_REMOTE_MODEL_ID = "G:/models/Qwen2.5-7B-Instruct"
# _DEFAULT_REMOTE_MODEL_ID = "G:/models/Qwen2.5-14B-Instruct"

_DEFAULT_REMOTE_PYTHON_CMD = "G:/conda/envs/PJ_310_LLM_SAM3/python.exe"
ALLOWED_REMOTE_PYTHON_MARKERS = (
    "/pj_310_llm_sam3/python.exe",
    "/pj_310_sam3/python.exe",
    "/pj_310_llm/python.exe",
    "/sam_3/",
    "g:/anaconda3/python.exe",
    "c:/programdata/anaconda3/envs/sam_3/python.exe",
)


def _default_remote_work_root() -> str:
    apply_remote_env_defaults()
    env_root = os.getenv("APP_REMOTE_STORAGE_PROGRAM_ROOT", "")
    if env_root:
        return env_root
    program_root_name = Path(__file__).resolve().parents[2].name
    storage_base = os.getenv("APP_REMOTE_STORAGE_BASE_ROOT", "G:/KDT10_3_1team_KLIK/0_Program_")
    return f"{storage_base.rstrip('/')}/{program_root_name}"


def _is_allowed_remote_python_candidate(candidate: str) -> bool:
    normalized = str(candidate or "").strip().replace("\\", "/").lower()
    if not normalized:
        return False
    if normalized in {"python", "py", "py -3"}:
        return True
    if not normalized.endswith("python.exe"):
        return False
    return any(marker in normalized for marker in ALLOWED_REMOTE_PYTHON_MARKERS)


def _normalize_discovered_remote_python_candidate(raw_line: str) -> str:
    text = str(raw_line or "").strip().replace("\\", "/")
    if not text or text.startswith("#"):
        return ""
    lowered = text.lower()
    if "windowsapps/python.exe" in lowered:
        return ""

    python_match = re.search(r"([A-Za-z]:/[^\s]*python\.exe)\s*$", text, flags=re.IGNORECASE)
    if python_match:
        return python_match.group(1).strip()

    env_match = re.search(r"([A-Za-z]:/[^\s]+)\s*$", text, flags=re.IGNORECASE)
    if not env_match:
        return ""
    env_path = env_match.group(1).strip().rstrip("/")
    lowered_env = env_path.lower()
    if lowered_env.endswith("/python.exe"):
        return env_path
    if lowered_env in {"g:/anaconda3", "c:/programdata/anaconda3"}:
        return f"{env_path}/python.exe"
    if any(token in lowered_env for token in ("/envs/", "/conda_envs/", "/miniconda3/envs/", "/anaconda3/envs/")):
        return f"{env_path}/python.exe"
    return ""


def _default_llm_base_url() -> str:
    env_base = str(os.getenv("LLM_API_BASE_URL", "")).strip()
    if env_base:
        if "://" not in env_base:
            env_base = f"http://{env_base}"
        return env_base.rstrip("/")

    host = str(os.getenv("LLM_SERVER_HOST", "127.0.0.1")).strip() or "127.0.0.1"
    port = env_int("LLM_SERVER_PORT", 8008)
    return f"http://{host}:{port}"


def _decode_remote_output(data: bytes, is_remote_windows: bool = True) -> str:
    raw = bytes(data or b"")
    encodings = ["utf-8"]
    if is_remote_windows:
        encodings.extend(["cp949", "euc-kr"])
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _split_remote_command_parts(command_text: str, is_remote_windows: bool) -> list[str]:
    raw = str(command_text).strip()
    if not raw:
        return []
    try:
        return shlex.split(raw, posix=(not is_remote_windows))
    except Exception:
        return [raw]


def _build_remote_shell_cmd(command_parts: list[str], is_remote_windows: bool) -> str:
    normalized = [str(part) for part in command_parts if str(part).strip()]
    if is_remote_windows:
        return subprocess.list2cmdline(normalized)
    return " ".join(shlex.quote(part) for part in normalized)


@dataclass(slots=True)
class _RemoteDirectConfig:
    enabled: bool
    required: bool
    ssh_host: str
    ssh_port: int
    ssh_user: str
    ssh_password: str
    python_cmd: str
    remote_work_root: str
    model_id: str
    connect_timeout_sec: float
    startup_timeout_sec: float
    request_timeout_sec: float
    keep_remote_files: bool

    @classmethod
    def from_env(cls) -> "_RemoteDirectConfig":
        apply_remote_env_defaults()
        default_work_root = _default_remote_work_root()
        return cls(
            enabled=env_flag("LLM_REMOTE_DIRECT_ENABLE", env_flag("LLM_REMOTE_ENABLE", True)),
            required=env_flag("LLM_REMOTE_REQUIRED", False),
            ssh_host=str(os.getenv("LLM_REMOTE_SSH_HOST", "")).strip(),
            ssh_port=env_int("LLM_REMOTE_SSH_PORT", 8875),
            ssh_user=str(os.getenv("LLM_REMOTE_SSH_USER", "")).strip(),
            ssh_password=str(os.getenv("LLM_REMOTE_SSH_PASSWORD", "")),
            python_cmd=str(
                os.getenv(
                    "LLM_REMOTE_PYTHON_CMD",
                    os.getenv("TRAIN_REMOTE_PYTHON_CMD", _DEFAULT_REMOTE_PYTHON_CMD),
                )
            ).strip()
            or _DEFAULT_REMOTE_PYTHON_CMD,
            remote_work_root=str(
                os.getenv("LLM_REMOTE_WORKDIR", os.getenv("TRAIN_REMOTE_WORKDIR", default_work_root))
            ).strip()
            or default_work_root,
            model_id=str(os.getenv("LLM_REMOTE_MODEL_ID", os.getenv("LLM_MODEL_ID", _DEFAULT_REMOTE_MODEL_ID))).strip()
            or _DEFAULT_REMOTE_MODEL_ID,
            connect_timeout_sec=float(os.getenv("LLM_REMOTE_CONNECT_TIMEOUT", "10")),
            startup_timeout_sec=float(os.getenv("LLM_REMOTE_STARTUP_TIMEOUT", "60")),
            request_timeout_sec=float(os.getenv("LLM_REMOTE_REQUEST_TIMEOUT", "180")),
            keep_remote_files=env_flag("LLM_REMOTE_KEEP_FILES", False),
        )


@dataclass
class _RemoteDirectSession:
    client: Any
    channel: Any
    stdin: Any
    stdout: Any
    stderr: Any
    remote_job_root: str
    remote_python_cmd: str
    module_signature: str
    stdout_buffer: str = ""
    stderr_tail: str = ""
    next_request_id: int = 1


def _local_runtime_signature() -> str:
    core_path = Path(__file__).resolve().with_name(f"{RUNTIME_CORE_MODULE_NAME}.py")
    worker_text = _remote_worker_script_text()
    digest = hashlib.sha256()
    try:
        digest.update(core_path.read_bytes())
    except Exception:
        digest.update(str(core_path).encode("utf-8", errors="replace"))
    digest.update(worker_text.encode("utf-8"))
    return digest.hexdigest()


def _remote_worker_script_text() -> str:
    return (
        "from __future__ import annotations\n"
        "import argparse\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "import traceback\n"
        "from pathlib import Path\n\n"
        f"READY_PREFIX = {REMOTE_READY_PREFIX!r}\n"
        f"RESULT_PREFIX = {REMOTE_RESULT_PREFIX!r}\n\n"
        "def _print_payload(prefix, payload):\n"
        "    print(prefix + json.dumps(payload, ensure_ascii=False), flush=True)\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser(description='Remote LLM direct worker')\n"
        "    parser.add_argument('--module-dir', required=True)\n"
        "    parser.add_argument('--model-id', default='')\n"
        "    args = parser.parse_args()\n"
        "    os.environ.setdefault('PYTHONUTF8', '1')\n"
        "    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')\n"
        "    os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')\n"
        "    sys.path.insert(0, str(Path(args.module_dir).resolve()))\n"
        "    _print_payload(READY_PREFIX, {'status': 'ready', 'model_id': args.model_id or ''})\n"
        "    build_health_payload = None\n"
        "    build_warmup_payload = None\n"
        "    build_rank_payload = None\n"
        "    for raw in sys.stdin:\n"
        "        raw = str(raw).strip()\n"
        "        if not raw:\n"
        "            continue\n"
        "        request_id = None\n"
        "        try:\n"
        "            req = json.loads(raw)\n"
        "            request_id = req.get('id')\n"
        "            path = str(req.get('path', '')).strip()\n"
        "            payload = dict(req.get('payload') or {})\n"
        "            model_id = str(req.get('model_id') or args.model_id or '').strip() or None\n"
        "            if path == '/health':\n"
        "                if build_health_payload is None:\n"
        "                    result = {'status': 'ok', 'model_id': model_id or args.model_id or '', 'load_mode': 'lazy', 'device': 'pending', 'loaded': False}\n"
        "                else:\n"
        "                    result = build_health_payload(model_id=model_id)\n"
        "            elif path == '/warmup':\n"
        "                if build_rank_payload is None or build_health_payload is None or build_warmup_payload is None:\n"
        f"                    from {RUNTIME_CORE_MODULE_NAME} import (\n"
        "                        build_health_payload as _build_health_payload,\n"
        "                        build_rank_payload as _build_rank_payload,\n"
        "                        build_warmup_payload as _build_warmup_payload,\n"
        "                    )\n"
        "                    build_health_payload = _build_health_payload\n"
        "                    build_rank_payload = _build_rank_payload\n"
        "                    build_warmup_payload = _build_warmup_payload\n"
        "                result = build_warmup_payload(model_id=model_id)\n"
        "            elif path == '/rank-prompts':\n"
        "                if build_rank_payload is None or build_health_payload is None or build_warmup_payload is None:\n"
        f"                    from {RUNTIME_CORE_MODULE_NAME} import (\n"
        "                        build_health_payload as _build_health_payload,\n"
        "                        build_rank_payload as _build_rank_payload,\n"
        "                        build_warmup_payload as _build_warmup_payload,\n"
        "                    )\n"
        "                    build_health_payload = _build_health_payload\n"
        "                    build_warmup_payload = _build_warmup_payload\n"
        "                    build_rank_payload = _build_rank_payload\n"
        "                result = build_rank_payload(\n"
        "                    user_text=str(payload.get('user_text', '')),\n"
        "                    n=int(payload.get('n', 5)),\n"
        "                    debug=bool(payload.get('debug', False)),\n"
        "                    model_id=model_id,\n"
        "                )\n"
        "            elif path == '/shutdown':\n"
        "                _print_payload(RESULT_PREFIX, {'ok': True, 'id': request_id, 'payload': {'status': 'bye'}})\n"
        "                break\n"
        "            else:\n"
        "                raise ValueError(f'Unsupported path: {path}')\n"
        "            _print_payload(RESULT_PREFIX, {'ok': True, 'id': request_id, 'payload': result})\n"
        "        except Exception as exc:\n"
        "            _print_payload(\n"
        "                RESULT_PREFIX,\n"
        "                {'ok': False, 'id': request_id, 'error': str(exc), 'traceback': traceback.format_exc()},\n"
        "            )\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


def _importlib_metadata_shim_text() -> str:
    return (
        "from importlib.metadata import *  # noqa: F401,F403\n"
        "from importlib.metadata import Distribution, EntryPoint, EntryPoints, PackageNotFoundError\n"
        "from importlib.metadata import distribution, distributions, entry_points, files, metadata, packages_distributions, requires, version\n"
    )


@dataclass
class LLMApiClient:
    base_url: str = field(default_factory=_default_llm_base_url)
    timeout_sec: float = 120.0
    _remote_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _remote_session: _RemoteDirectSession | None = field(default=None, init=False, repr=False)
    _prompt_cache_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _prompt_cache: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _prompt_cache_loaded: bool = field(default=False, init=False, repr=False)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        with self._remote_lock:
            session = self._remote_session
            self._remote_session = None
        if session is None:
            return
        try:
            try:
                session.stdin.write(json.dumps({"id": -1, "path": "/shutdown", "payload": {}}, ensure_ascii=False) + "\n")
                session.stdin.flush()
            except Exception:
                pass
            try:
                session.channel.close()
            except Exception:
                pass
            try:
                session.client.close()
            except Exception:
                pass
        finally:
            cfg = _RemoteDirectConfig.from_env()
            if cfg.keep_remote_files:
                return
            try:
                import paramiko  # type: ignore

                cleanup_client = paramiko.SSHClient()
                cleanup_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                cleanup_client.connect(
                    hostname=cfg.ssh_host,
                    port=int(cfg.ssh_port),
                    username=cfg.ssh_user,
                    password=cfg.ssh_password,
                    timeout=max(3.0, float(cfg.connect_timeout_sec)),
                    look_for_keys=False,
                    allow_agent=False,
                )
                cleanup_path = str(session.remote_job_root).replace("/", "\\")
                cleanup_client.exec_command(f'cmd /c rmdir /s /q "{cleanup_path}"')
                cleanup_client.close()
            except Exception:
                pass

    def _prompt_cache_file(self) -> Path:
        cache_dir = (PREVIEW_CACHE_BASE / "nlp").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "prompt_cache.json"

    def _ensure_prompt_cache_loaded(self) -> None:
        with self._prompt_cache_lock:
            if self._prompt_cache_loaded:
                return
            cache_path = self._prompt_cache_file()
            payload: dict[str, dict[str, Any]] = {}
            if cache_path.is_file():
                try:
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        payload = {str(key): dict(value) for key, value in raw.items() if isinstance(value, dict)}
                except Exception:
                    payload = {}
            self._prompt_cache = payload
            self._prompt_cache_loaded = True

    def _persist_prompt_cache(self) -> None:
        with self._prompt_cache_lock:
            cache_path = self._prompt_cache_file()
            cache_path.write_text(json.dumps(self._prompt_cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _prompt_cache_key(self, *, user_text: str, n: int, debug: bool, model_id: str) -> str:
        normalized_text = re.sub(r"\s+", " ", str(user_text or "").strip())
        payload = {
            "user_text": normalized_text,
            "n": int(n),
            "debug": bool(debug),
            "model_id": str(model_id or "").strip(),
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _cached_rank_payload(self, *, user_text: str, n: int, debug: bool, model_id: str) -> dict[str, Any] | None:
        self._ensure_prompt_cache_loaded()
        cache_key = self._prompt_cache_key(user_text=user_text, n=n, debug=debug, model_id=model_id)
        with self._prompt_cache_lock:
            cached = self._prompt_cache.get(cache_key)
            if not isinstance(cached, dict):
                return None
            result = json.loads(json.dumps(cached, ensure_ascii=False))
        load_mode = str(result.get("load_mode", "")).strip().lower()
        meta = dict(result.get("_meta") or {})
        fallback_mode = str(meta.get("fallback_mode", "")).strip().lower()
        if load_mode == "local-heuristic-fallback" or fallback_mode == "local_heuristic":
            return None
        result = self._prioritize_heuristic_items(result, user_text=user_text, n=n)
        if "elapsed_ms" in meta and "remote_elapsed_ms" not in meta:
            meta["remote_elapsed_ms"] = meta.get("elapsed_ms")
        meta["cache_hit"] = True
        meta["elapsed_ms"] = 0.0
        result["_meta"] = meta
        return result

    def _store_rank_payload(self, *, user_text: str, n: int, debug: bool, model_id: str, payload: dict[str, Any]) -> None:
        self._ensure_prompt_cache_loaded()
        load_mode = str(payload.get("load_mode", "")).strip().lower()
        meta = dict(payload.get("_meta") or {})
        fallback_mode = str(meta.get("fallback_mode", "")).strip().lower()
        if load_mode == "local-heuristic-fallback" or fallback_mode == "local_heuristic":
            return
        cache_key = self._prompt_cache_key(user_text=user_text, n=n, debug=debug, model_id=model_id)
        to_store = json.loads(json.dumps(payload, ensure_ascii=False))
        meta = dict(to_store.get("_meta") or {})
        meta["cache_hit"] = False
        to_store["_meta"] = meta
        with self._prompt_cache_lock:
            self._prompt_cache[cache_key] = to_store
            if len(self._prompt_cache) > 128:
                oldest_key = next(iter(self._prompt_cache))
                self._prompt_cache.pop(oldest_key, None)
        self._persist_prompt_cache()

    def _ensure_remote_ssh_ready(self) -> None:
        apply_remote_env_defaults()
        missing = missing_remote_ssh_fields()
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                "Remote server connection is not configured. "
                f"Missing fields: {joined}. "
                "Set them in config.local.json or environment variables."
            )

    def _http_request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        body: bytes | None = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url=url, method=method.upper(), data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return {}
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot connect to LLM API at {url}: {exc.reason}") from exc

    def _http_health_available(self, timeout_sec: float = 1.5) -> bool:
        url = f"{self.base_url.rstrip('/')}/health"
        req = urllib.request.Request(url=url, method="GET", headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=max(0.2, float(timeout_sec))) as resp:
                return int(getattr(resp, "status", 0)) == 200
        except Exception:
            return False

    def _python_candidates(self, cfg: _RemoteDirectConfig) -> list[str]:
        preferred = str(cfg.python_cmd).strip()
        candidates: list[str] = [preferred] if preferred else []
        fallback = [
            _DEFAULT_REMOTE_PYTHON_CMD,
            "G:/conda/envs/PJ_310_SAM3/python.exe",
            "G:/conda/envs/PJ_310_LLM/python.exe",
            "G:/conda_envs/PJ_310_LLM_SAM3/python.exe",
            "G:/conda_envs/PJ_310_SAM3/python.exe",
            "G:/conda_envs/PJ_310_LLM/python.exe",
            "G:/anaconda3/envs/PJ_310_LLM_SAM3/python.exe",
            "G:/anaconda3/envs/PJ_310_SAM3/python.exe",
            "G:/miniconda3/envs/PJ_310_LLM_SAM3/python.exe",
            "G:/miniconda3/envs/PJ_310_SAM3/python.exe",
            "python",
            "py -3",
            "py",
            f"C:/Users/{cfg.ssh_user}/anaconda3/envs/PJ_310_LLM_SAM3/python.exe",
            f"C:/Users/{cfg.ssh_user}/anaconda3/envs/PJ_310_SAM3/python.exe",
            "G:/anaconda3/python.exe",
            "C:/ProgramData/anaconda3/envs/sam_3/python.exe",
            "C:/conda_envs/PJ_310_LLM_SAM3/python.exe",
            "C:/conda_envs/PJ_310_SAM3/python.exe",
            "D:/conda_envs/PJ_310_LLM_SAM3/python.exe",
            "D:/conda_envs/PJ_310_SAM3/python.exe",
        ]
        for item in fallback:
            if item not in candidates:
                candidates.append(item)
        return candidates

    def _discover_remote_python_candidates(self, client: Any) -> list[str]:
        discovered: list[str] = []
        probe_cmds = [
            "cmd /c where python",
            r'cmd /c for /d %i in (G:\conda_envs\*) do @if exist "%i\python.exe" echo %i\python.exe',
            r'cmd /c for /d %i in (G:\conda\envs\*) do @if exist "%i\python.exe" echo %i\python.exe',
            r'cmd /c for /d %i in (C:\conda_envs\*) do @if exist "%i\python.exe" echo %i\python.exe',
            r'cmd /c for /d %i in (D:\conda_envs\*) do @if exist "%i\python.exe" echo %i\python.exe',
            r'cmd /c for /d %i in (G:\anaconda3\envs\*) do @if exist "%i\python.exe" echo %i\python.exe',
            r'cmd /c for /d %i in (G:\miniconda3\envs\*) do @if exist "%i\python.exe" echo %i\python.exe',
        ]
        for probe_cmd in probe_cmds:
            try:
                _stdin, stdout, _stderr = client.exec_command(probe_cmd)
                status = int(stdout.channel.recv_exit_status())
                output = _decode_remote_output(stdout.read(), True)
            except Exception:
                continue
            if status != 0:
                continue
            for raw_line in output.splitlines():
                candidate = _normalize_discovered_remote_python_candidate(raw_line)
                if not candidate:
                    continue
                if not _is_allowed_remote_python_candidate(candidate):
                    continue
                if candidate not in discovered:
                    discovered.append(candidate)
        return discovered

    def _select_remote_python(self, client: Any, cfg: _RemoteDirectConfig) -> str:
        probe_prefix = "__LLM_REMOTE_PY__="
        probe_code = (
            "import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); "
            "import importlib, json, sys; "
            "mods = {}; "
            "names = ['torch', 'transformers']; "
            "[mods.setdefault(name, getattr(importlib.import_module(name), '__version__', '')) for name in names]; "
            "print("
            + repr(probe_prefix)
            + " + json.dumps({'python': sys.version.split()[0], 'torch': mods.get('torch', ''), 'transformers': mods.get('transformers', ''), 'executable': sys.executable}, ensure_ascii=True))"
        )
        failures: list[str] = []
        candidates = self._python_candidates(cfg)
        for discovered in self._discover_remote_python_candidates(client):
            if discovered not in candidates:
                candidates.append(discovered)
        for candidate in candidates:
            parts = _split_remote_command_parts(candidate, True)
            if not parts:
                continue
            shell_cmd = _build_remote_shell_cmd([*parts, "-c", probe_code], True)
            try:
                _stdin, stdout, stderr = client.exec_command(shell_cmd)
                status = int(stdout.channel.recv_exit_status())
                stdout_text = _decode_remote_output(stdout.read(), True)
                stderr_text = _decode_remote_output(stderr.read(), True)
            except Exception:
                failures.append(f"- {candidate}: remote probe execution failed")
                continue
            if status != 0:
                tail = [line.strip() for line in (stderr_text or stdout_text).splitlines() if line.strip()]
                failures.append(f"- {candidate}: {(tail[-1] if tail else f'exit code {status}')}")
                continue
            if probe_prefix in stdout_text:
                return candidate
        raise RuntimeError(
            "Remote NLP Python not found. Set LLM_REMOTE_PYTHON_CMD to a Python that can import torch and transformers.\n"
            + "\n".join(failures[-12:])
        )

    def _ensure_remote_session(self, cfg: _RemoteDirectConfig) -> _RemoteDirectSession:
        self._ensure_remote_ssh_ready()
        current_signature = _local_runtime_signature()
        session = self._remote_session
        if session is not None:
            try:
                if (
                    session.channel is not None
                    and (not session.channel.closed)
                    and session.channel.active
                    and str(getattr(session, "module_signature", "")) == current_signature
                ):
                    return session
            except Exception:
                pass
            self.close()

        try:
            import paramiko  # type: ignore
        except Exception as exc:
            raise RuntimeError("paramiko is required for remote NLP. Install it with: pip install paramiko") from exc

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
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
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(30)
        sftp = client.open_sftp()
        remote_python_cmd = self._select_remote_python(client, cfg)
        safe_name = f"llm_direct_{int(time.time())}_{os.getpid()}"
        remote_root = str(cfg.remote_work_root).replace("\\", "/").rstrip("/")
        remote_job_root = f"{remote_root}/.remote_jobs/{safe_name}"
        remote_module_dir = f"{remote_job_root}/modules"
        remote_worker_path = f"{remote_job_root}/remote_llm_worker.py"
        local_core_path = Path(__file__).resolve().with_name(f"{RUNTIME_CORE_MODULE_NAME}.py")

        for remote_dir in (remote_job_root, remote_module_dir):
            parts = str(remote_dir).replace("\\", "/").split("/")
            current = parts[0]
            try:
                sftp.stat(current)
            except Exception:
                try:
                    sftp.mkdir(current)
                except Exception:
                    pass
            for part in parts[1:]:
                current = f"{current}/{part}"
                try:
                    sftp.stat(current)
                except Exception:
                    try:
                        sftp.mkdir(current)
                    except Exception:
                        pass

        sftp.put(str(local_core_path), f"{remote_module_dir}/{RUNTIME_CORE_MODULE_NAME}.py")
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix="_importlib_metadata.py") as fp:
            fp.write(_importlib_metadata_shim_text())
            local_importlib_metadata_path = Path(fp.name)
        try:
            sftp.put(str(local_importlib_metadata_path), f"{remote_module_dir}/importlib_metadata.py")
        finally:
            try:
                local_importlib_metadata_path.unlink(missing_ok=True)
            except Exception:
                pass
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix="_remote_llm_worker.py") as fp:
            fp.write(_remote_worker_script_text())
            local_worker_path = Path(fp.name)
        try:
            sftp.put(str(local_worker_path), remote_worker_path)
        finally:
            try:
                local_worker_path.unlink(missing_ok=True)
            except Exception:
                pass

        python_parts = _split_remote_command_parts(remote_python_cmd, True)
        lower_python_parts = {str(part).strip().lower() for part in python_parts}
        if "-u" not in lower_python_parts:
            python_parts.append("-u")
        remote_shell_cmd = _build_remote_shell_cmd(
            [
                *python_parts,
                remote_worker_path,
                "--module-dir",
                remote_module_dir,
                "--model-id",
                cfg.model_id,
            ],
            True,
        )
        stdin, stdout, stderr = client.exec_command(remote_shell_cmd)
        channel = stdout.channel
        channel.settimeout(0.1)

        deadline = time.time() + max(10.0, float(cfg.startup_timeout_sec))
        ready_lines: list[str] = []
        stderr_tail = ""
        while time.time() < deadline:
            if channel.recv_ready():
                chunk = _decode_remote_output(channel.recv(65536), True)
                ready_lines.extend(line.strip() for line in chunk.replace("\r", "\n").split("\n") if line.strip())
                for line in ready_lines:
                    if line.startswith(REMOTE_READY_PREFIX):
                        session = _RemoteDirectSession(
                            client=client,
                            channel=channel,
                            stdin=stdin,
                            stdout=stdout,
                            stderr=stderr,
                            remote_job_root=remote_job_root,
                            remote_python_cmd=remote_python_cmd,
                            module_signature=current_signature,
                        )
                        self._remote_session = session
                        try:
                            sftp.close()
                        except Exception:
                            pass
                        return session
            if channel.recv_stderr_ready():
                err_text = _decode_remote_output(channel.recv_stderr(65536), True).strip()
                if err_text:
                    stderr_tail = (stderr_tail + "\n" + err_text).strip()[-8000:]
            if channel.exit_status_ready():
                status = int(channel.recv_exit_status())
                raise RuntimeError(
                    f"Remote NLP worker exited during startup (code {status})"
                    + (f": {stderr_tail.splitlines()[-1]}" if stderr_tail else "")
                )
            time.sleep(0.05)
        raise RuntimeError(
            "Remote NLP worker did not become ready in time"
            + (f": {stderr_tail.splitlines()[-1]}" if stderr_tail else "")
        )

    def _remote_direct_request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg = _RemoteDirectConfig.from_env()
        with self._remote_lock:
            session = self._ensure_remote_session(cfg)
            request_id = int(session.next_request_id)
            session.next_request_id += 1
            request_payload = {
                "id": request_id,
                "method": str(method).upper(),
                "path": str(path).strip(),
                "payload": dict(payload or {}),
                "model_id": cfg.model_id,
            }
            try:
                session.stdin.write(json.dumps(request_payload, ensure_ascii=False) + "\n")
                session.stdin.flush()
            except Exception as exc:
                self.close()
                raise RuntimeError(f"Failed to send remote NLP request: {exc}") from exc

            deadline = time.time() + max(float(cfg.request_timeout_sec), float(self.timeout_sec))
            while time.time() < deadline:
                if session.channel.recv_ready():
                    chunk = _decode_remote_output(session.channel.recv(65536), True)
                    session.stdout_buffer += chunk.replace("\r", "\n")
                    while "\n" in session.stdout_buffer:
                        line, session.stdout_buffer = session.stdout_buffer.split("\n", 1)
                        text = str(line).strip()
                        if not text:
                            continue
                        if not text.startswith(REMOTE_RESULT_PREFIX):
                            continue
                        try:
                            result = json.loads(text[len(REMOTE_RESULT_PREFIX) :].strip())
                        except Exception:
                            continue
                        if int(result.get("id", -999999)) != request_id:
                            continue
                        if not bool(result.get("ok", False)):
                            error_text = str(result.get("error", "remote NLP inference failed"))
                            tb_text = str(result.get("traceback", "")).strip()
                            self.close()
                            if tb_text:
                                raise RuntimeError(f"{error_text}\n{tb_text}")
                            raise RuntimeError(error_text)
                        payload_obj = result.get("payload", {})
                        return dict(payload_obj) if isinstance(payload_obj, dict) else {}
                if session.channel.recv_stderr_ready():
                    err_text = _decode_remote_output(session.channel.recv_stderr(65536), True).strip()
                    if err_text:
                        session.stderr_tail = (session.stderr_tail + "\n" + err_text).strip()[-8000:]
                if session.channel.exit_status_ready():
                    status = int(session.channel.recv_exit_status())
                    stderr_tail = session.stderr_tail.strip()
                    self.close()
                    raise RuntimeError(
                        f"Remote NLP worker exited (code {status})"
                        + (f": {stderr_tail.splitlines()[-1]}" if stderr_tail else "")
                    )
                time.sleep(0.05)
            self.close()
            raise RuntimeError("Remote NLP request timed out")

    def _quick_remote_health(self) -> dict[str, Any]:
        cfg = _RemoteDirectConfig.from_env()
        self._ensure_remote_ssh_ready()
        try:
            import paramiko  # type: ignore
        except Exception as exc:
            raise RuntimeError("paramiko is required for remote NLP. Install it with: pip install paramiko") from exc

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
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
            remote_python_cmd = str(cfg.python_cmd).strip() or self._select_remote_python(client, cfg)
            python_parts = _split_remote_command_parts(remote_python_cmd, True)
            probe_code = (
                "import json, sys; "
                "print(json.dumps({"
                "'status': 'ok', "
                f"'model_id': {cfg.model_id!r}, "
                "'load_mode': 'lazy', "
                "'device': 'pending', "
                "'loaded': False, "
                "'python': sys.version.split()[0], "
                "'executable': sys.executable"
                "}, ensure_ascii=True))"
            )
            shell_cmd = _build_remote_shell_cmd([*python_parts, "-c", probe_code], True)
            _stdin, stdout, stderr = client.exec_command(shell_cmd)
            status = int(stdout.channel.recv_exit_status())
            stdout_text = _decode_remote_output(stdout.read(), True).strip()
            stderr_text = _decode_remote_output(stderr.read(), True).strip()
            if status != 0:
                if str(cfg.python_cmd).strip() and remote_python_cmd == str(cfg.python_cmd).strip():
                    remote_python_cmd = self._select_remote_python(client, cfg)
                    python_parts = _split_remote_command_parts(remote_python_cmd, True)
                    shell_cmd = _build_remote_shell_cmd([*python_parts, "-c", probe_code], True)
                    _stdin, stdout, stderr = client.exec_command(shell_cmd)
                    status = int(stdout.channel.recv_exit_status())
                    stdout_text = _decode_remote_output(stdout.read(), True).strip()
                    stderr_text = _decode_remote_output(stderr.read(), True).strip()
            if status != 0:
                raise RuntimeError(stderr_text or stdout_text or f"exit code {status}")
            payload = json.loads(stdout_text) if stdout_text else {}
            if isinstance(payload, dict):
                payload["remote_python"] = remote_python_cmd
                return payload
            return {"status": "ok", "model_id": cfg.model_id, "load_mode": "lazy", "device": "pending", "loaded": False}
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        remote_cfg = _RemoteDirectConfig.from_env()
        if remote_cfg.enabled:
            try:
                return self._remote_direct_request(method, path, payload)
            except Exception as exc:
                self.close()
                if remote_cfg.required:
                    raise
                if self._http_health_available():
                    try:
                        return self._http_request(method, path, payload)
                    except Exception as http_exc:
                        raise RuntimeError(
                            f"Remote NLP direct request failed: {exc}\nHTTP fallback failed: {http_exc}"
                        ) from http_exc
                raise RuntimeError(f"Remote NLP direct request failed: {exc}") from exc
        return self._http_request(method, path, payload)

    def _prepare_remote_user_text(self, user_text: str) -> str:
        ## =====================================
        ## 함수 기능 : LLM에 전달할 사용자 텍스트 정규화
        ## 매개 변수 : user_text(str) - 원본 사용자 입력
        ## 반환 결과 : str - 정규화된 텍스트 (한국어 포함 시 원문 그대로 전달)
        ## =====================================
        return normalize_user_text(user_text)

    def _heuristic_fallback_payload(self, user_text: str, *, model_id: str, reason: str) -> dict[str, Any] | None:
        heuristic_candidates = heuristic_english_candidates(user_text)
        if not heuristic_candidates:
            normalized = normalize_user_text(user_text)
            if not normalized:
                return None
            heuristic_candidates = [normalized] if all(ord(ch) < 128 for ch in normalized) else ["object"]
        total = float(sum(range(1, len(heuristic_candidates) + 1))) or 1.0
        items: list[dict[str, Any]] = []
        weight = len(heuristic_candidates)
        for candidate in heuristic_candidates:
            items.append(
                {
                    "english_prompt": candidate,
                    "korean_gloss": "",
                    "probability": float(weight / total),
                    "loss": float(len(heuristic_candidates) - weight),
                }
            )
            weight -= 1
        return {
            "model_id": model_id,
            "load_mode": "local-heuristic-fallback",
            "device": "local",
            "items": items,
            "_meta": {
                "fallback_mode": "local_heuristic",
                "fallback_reason": str(reason).strip(),
                "cache_hit": False,
            },
        }

    def _prioritize_heuristic_items(self, payload: dict[str, Any], *, user_text: str, n: int) -> dict[str, Any]:
        heuristic_candidates = heuristic_english_candidates(user_text)
        if not heuristic_candidates:
            return payload
        raw_items = payload.get("items")
        existing_items = list(raw_items) if isinstance(raw_items, list) else []
        by_prompt: dict[str, dict[str, Any]] = {}
        ordered_remote: list[dict[str, Any]] = []
        for item in existing_items:
            if not isinstance(item, dict):
                continue
            prompt = normalize_user_text(str(item.get("english_prompt", "")))
            if not prompt:
                continue
            key = prompt.casefold()
            item_copy = dict(item)
            item_copy["english_prompt"] = prompt
            if key not in by_prompt:
                by_prompt[key] = item_copy
                ordered_remote.append(item_copy)

        merged_items: list[dict[str, Any]] = []
        seen: set[str] = set()
        heuristic_total = max(1, len(heuristic_candidates))
        for index, candidate in enumerate(heuristic_candidates, start=1):
            key = candidate.casefold()
            source_item = dict(by_prompt.get(key) or {})
            source_item["english_prompt"] = candidate
            source_item.setdefault("korean_gloss", "")
            if not source_item.get("probability"):
                source_item["probability"] = round((heuristic_total - index + 1) / float(heuristic_total), 6)
            source_item["source"] = "heuristic"
            merged_items.append(source_item)
            seen.add(key)

        for item in ordered_remote:
            prompt = normalize_user_text(str(item.get("english_prompt", "")))
            key = prompt.casefold()
            if not prompt or key in seen:
                continue
            item_copy = dict(item)
            item_copy["english_prompt"] = prompt
            item_copy.setdefault("source", "remote")
            merged_items.append(item_copy)
            seen.add(key)

        if merged_items:
            payload["items"] = merged_items[: max(1, int(n))]
        meta = dict(payload.get("_meta") or {})
        meta["heuristic_candidates"] = heuristic_candidates[: max(1, int(n))]
        meta["heuristic_promoted"] = True
        payload["_meta"] = meta
        return payload

    def health(self) -> dict[str, Any]:
        remote_cfg = _RemoteDirectConfig.from_env()
        if remote_cfg.enabled:
            with self._remote_lock:
                has_session = self._remote_session is not None
            if not has_session:
                return self._quick_remote_health()
        return self._request("GET", "/health")

    def warmup(self) -> dict[str, Any]:
        remote_cfg = _RemoteDirectConfig.from_env()
        if remote_cfg.enabled:
            return self._request("POST", "/warmup", payload={})
        try:
            return self._http_request("POST", "/warmup", payload={})
        except Exception:
            return self.health()

    def rank_prompts(self, user_text: str, n: int = 5, debug: bool = False) -> dict[str, Any]:
        remote_cfg = _RemoteDirectConfig.from_env()
        selected_model_id = str(remote_cfg.model_id).strip() or _DEFAULT_REMOTE_MODEL_ID
        cached = self._cached_rank_payload(user_text=user_text, n=n, debug=debug, model_id=selected_model_id)
        if cached is not None:
            return cached
        remote_user_text = self._prepare_remote_user_text(user_text)
        payload = {"user_text": remote_user_text, "n": int(n), "debug": bool(debug)}
        started_at = time.monotonic()
        try:
            result = self._request("POST", "/rank-prompts", payload=payload)
        except Exception as exc:
            fallback = self._heuristic_fallback_payload(
                user_text,
                model_id=selected_model_id,
                reason=str(exc),
            )
            if fallback is None:
                raise
            meta = dict(fallback.get("_meta") or {})
            meta["elapsed_ms"] = round((time.monotonic() - started_at) * 1000.0, 2)
            if remote_user_text != normalize_user_text(user_text):
                meta["remote_query_text"] = remote_user_text
            fallback["_meta"] = meta
            self._store_rank_payload(user_text=user_text, n=n, debug=debug, model_id=selected_model_id, payload=fallback)
            return fallback
        meta = dict(result.get("_meta") or {})
        meta["cache_hit"] = False
        meta["elapsed_ms"] = round((time.monotonic() - started_at) * 1000.0, 2)
        if remote_user_text != normalize_user_text(user_text):
            meta["remote_query_text"] = remote_user_text
        result["_meta"] = meta
        result = self._prioritize_heuristic_items(result, user_text=user_text, n=n)
        self._store_rank_payload(user_text=user_text, n=n, debug=debug, model_id=selected_model_id, payload=result)
        return result
