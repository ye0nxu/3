from __future__ import annotations

import os

# PyTorch(libomp)와 MKL(libiomp5md) OpenMP 런타임 충돌 방지 (Windows)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config import apply_remote_env_defaults
from utils.env import env_flag


def _configure_runtime_defaults() -> None:
    apply_remote_env_defaults()
    os.environ.setdefault("APP_REMOTE_STORAGE_ENABLE", "0")
    if env_flag("LLM_REMOTE_DIRECT_ENABLE", env_flag("LLM_REMOTE_ENABLE", True)):
        os.environ["LLM_SERVER_AUTOSTART"] = "0"


def main() -> None:
    _configure_runtime_defaults()
    from app.main import main as app_main

    app_main()


if __name__ == "__main__":
    main()