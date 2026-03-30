from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import apply_remote_env_defaults, load_local_config


class RemoteConfigTests(unittest.TestCase):
    def test_apply_remote_env_defaults_from_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.local.json"
            config_path.write_text(
                json.dumps(
                    {
                        "remote": {
                            "ssh": {
                                "host": "example-host",
                                "port": 2222,
                                "user": "tester",
                                "password": "secret",
                            },
                            "python": {"llm": "G:/envs/test/python.exe"},
                            "models": {"llm": "G:/models/qwen"},
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            previous_env = {
                "APP_LOCAL_CONFIG": os.getenv("APP_LOCAL_CONFIG"),
                "LLM_REMOTE_SSH_HOST": os.getenv("LLM_REMOTE_SSH_HOST"),
                "LLM_REMOTE_SSH_PORT": os.getenv("LLM_REMOTE_SSH_PORT"),
                "LLM_REMOTE_SSH_USER": os.getenv("LLM_REMOTE_SSH_USER"),
                "LLM_REMOTE_SSH_PASSWORD": os.getenv("LLM_REMOTE_SSH_PASSWORD"),
                "LLM_REMOTE_PYTHON_CMD": os.getenv("LLM_REMOTE_PYTHON_CMD"),
                "LLM_REMOTE_MODEL_ID": os.getenv("LLM_REMOTE_MODEL_ID"),
            }
            try:
                os.environ["APP_LOCAL_CONFIG"] = str(config_path)
                for key in (
                    "LLM_REMOTE_SSH_HOST",
                    "LLM_REMOTE_SSH_PORT",
                    "LLM_REMOTE_SSH_USER",
                    "LLM_REMOTE_SSH_PASSWORD",
                    "LLM_REMOTE_PYTHON_CMD",
                    "LLM_REMOTE_MODEL_ID",
                ):
                    os.environ.pop(key, None)
                load_local_config(force_reload=True)
                apply_remote_env_defaults()
                self.assertEqual(os.getenv("LLM_REMOTE_SSH_HOST"), "example-host")
                self.assertEqual(os.getenv("LLM_REMOTE_SSH_PORT"), "2222")
                self.assertEqual(os.getenv("LLM_REMOTE_SSH_USER"), "tester")
                self.assertEqual(os.getenv("LLM_REMOTE_SSH_PASSWORD"), "secret")
                self.assertEqual(os.getenv("LLM_REMOTE_PYTHON_CMD"), "G:/envs/test/python.exe")
                self.assertEqual(os.getenv("LLM_REMOTE_MODEL_ID"), "G:/models/qwen")
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                os.environ.pop("APP_LOCAL_CONFIG", None)
                load_local_config(force_reload=True)


if __name__ == "__main__":
    unittest.main()
