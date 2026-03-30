from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from config import apply_remote_env_defaults
from utils.env import env_flag

# 하드코딩 폴백: config.local.json["remote"]["models"]["llm"] 로 설정 권장
_DEFAULT_REMOTE_MODEL_DIR = Path(r"G:\models\Qwen2.5-7B-Instruct")
# _DEFAULT_REMOTE_MODEL_DIR = Path(r"G:\models\Qwen2.5-14B-Instruct")


def _is_loopback_host(host: str) -> bool:
    normalized = str(host).strip().lower().strip("[]")
    return normalized in {"127.0.0.1", "localhost", "::1"}


@dataclass
class LLMServerManager:
    host: str = "127.0.0.1"
    port: int = 8008
    startup_wait_sec: float = 2.0
    health_timeout_sec: float = 1.0

    def __post_init__(self) -> None:
        apply_remote_env_defaults()
        self._process: subprocess.Popen[str] | None = None
        self._owns_process = False
        self._project_root = Path(__file__).resolve().parents[2]
        self._base_url = self._resolve_base_url()
        parsed = urllib.parse.urlparse(self._base_url)
        self._target_host = str(parsed.hostname or self.host).strip() or self.host
        self._target_port = int(parsed.port or self.port)
        self._auto_start_local = env_flag("LLM_SERVER_AUTOSTART", True) and self._is_local_target(parsed)

    @property
    def base_url(self) -> str:
        return self._base_url

    def _resolve_base_url(self) -> str:
        env_base = str(os.getenv("LLM_API_BASE_URL", "")).strip()
        if env_base:
            if "://" not in env_base:
                env_base = f"http://{env_base}"
            return env_base.rstrip("/")

        env_host = str(os.getenv("LLM_SERVER_HOST", "")).strip()
        host = env_host or self.host
        env_port = str(os.getenv("LLM_SERVER_PORT", "")).strip()
        if env_port:
            try:
                port = int(env_port)
            except ValueError:
                port = self.port
        else:
            port = self.port
        return f"http://{host}:{port}"

    def _is_local_target(self, parsed: urllib.parse.ParseResult) -> bool:
        if parsed.scheme not in {"http", ""}:
            return False
        if parsed.path not in {"", "/"}:
            return False
        if parsed.query or parsed.fragment:
            return False
        return _is_loopback_host(parsed.hostname or "")

    def _health_ok(self) -> bool:
        req = urllib.request.Request(f"{self.base_url}/health", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.health_timeout_sec) as resp:
                return int(getattr(resp, "status", 0)) == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def _resolve_model_id(self) -> str:
        env_remote_model = str(os.getenv("LLM_REMOTE_MODEL_ID", "")).strip()
        if env_remote_model:
            return env_remote_model
        env_model = str(os.getenv("LLM_MODEL_ID", "")).strip()
        if env_model:
            return env_model
        if _DEFAULT_REMOTE_MODEL_DIR.is_dir():
            return str(_DEFAULT_REMOTE_MODEL_DIR)
        local_model = self._project_root / "assets" / "models" / "LLM_models" / "Qwen2.5-7B-Instruct"
        # local_model = self._project_root / "assets" / "models" / "LLM_models" / "Qwen2.5-14B-Instruct"
        
        if local_model.is_dir():
            return str(local_model)
        return "Qwen/Qwen2.5-7B-Instruct"
        # return "Qwen/Qwen2.5-14B-Instruct"
        

    def start(self) -> None:
        if self._health_ok():
            self._owns_process = False
            return

        if not self._auto_start_local:
            return

        run_script = self._project_root / "run_llm_server.py"
        if not run_script.is_file():
            return

        model_id = self._resolve_model_id()
        cmd = [
            sys.executable,
            str(run_script),
            "--host",
            self._target_host,
            "--port",
            str(self._target_port),
            "--model-id",
            model_id,
        ]

        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW"))

        try:
            self._process = subprocess.Popen(  # noqa: S603
                cmd,
                cwd=str(self._project_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                creationflags=creationflags,
            )
            self._owns_process = True
        except Exception:
            self._process = None
            self._owns_process = False
            return

        deadline = time.time() + max(0.2, float(self.startup_wait_sec))
        while time.time() < deadline:
            if self._process.poll() is not None:
                break
            if self._health_ok():
                break
            time.sleep(0.2)

    def stop(self) -> None:
        if not self._owns_process or self._process is None:
            return
        if self._process.poll() is not None:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass
