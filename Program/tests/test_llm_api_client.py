from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm.client import (
    LLMApiClient,
    _is_allowed_remote_python_candidate,
    _normalize_discovered_remote_python_candidate,
)


class LLMApiClientTests(unittest.TestCase):
    def test_rank_prompts_uses_local_cache_for_repeat_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_env = {
                "APP_LOCAL_CONFIG": os.getenv("APP_LOCAL_CONFIG"),
                "LLM_REMOTE_DIRECT_ENABLE": os.getenv("LLM_REMOTE_DIRECT_ENABLE"),
                "LLM_REMOTE_ENABLE": os.getenv("LLM_REMOTE_ENABLE"),
                "LLM_REMOTE_MODEL_ID": os.getenv("LLM_REMOTE_MODEL_ID"),
            }
            try:
                os.environ["APP_LOCAL_CONFIG"] = str(Path(tmp_dir) / "missing.json")
                os.environ["LLM_REMOTE_DIRECT_ENABLE"] = "0"
                os.environ["LLM_REMOTE_ENABLE"] = "0"
                os.environ["LLM_REMOTE_MODEL_ID"] = "G:/models/Qwen2.5-7B-Instruct"

                client = LLMApiClient()
                client._prompt_cache = {}
                client._prompt_cache_loaded = True

                with patch.object(
                    client,
                    "_request",
                    return_value={
                        "model_id": "G:/models/Qwen2.5-7B-Instruct",
                        "load_mode": "cuda-4bit",
                        "device": "cuda:0",
                        "items": [{"english_prompt": "white car", "korean_gloss": "\ud770\uc0c9 \uc2b9\uc6a9\ucc28"}],
                    },
                ) as request_mock:
                    first = client.rank_prompts("\ud770\uc0c9 \uc2b9\uc6a9\ucc28", n=3, debug=False)
                    second = client.rank_prompts("\ud770\uc0c9 \uc2b9\uc6a9\ucc28", n=3, debug=False)

                self.assertEqual(request_mock.call_count, 1)
                self.assertFalse(bool(first.get("_meta", {}).get("cache_hit", True)))
                self.assertTrue(bool(second.get("_meta", {}).get("cache_hit", False)))
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_health_fails_clearly_when_remote_config_is_missing(self) -> None:
        previous_env = {
            "APP_LOCAL_CONFIG": os.getenv("APP_LOCAL_CONFIG"),
            "LLM_REMOTE_DIRECT_ENABLE": os.getenv("LLM_REMOTE_DIRECT_ENABLE"),
            "LLM_REMOTE_ENABLE": os.getenv("LLM_REMOTE_ENABLE"),
            "LLM_REMOTE_REQUIRED": os.getenv("LLM_REMOTE_REQUIRED"),
            "LLM_REMOTE_SSH_HOST": os.getenv("LLM_REMOTE_SSH_HOST"),
            "LLM_REMOTE_SSH_PORT": os.getenv("LLM_REMOTE_SSH_PORT"),
            "LLM_REMOTE_SSH_USER": os.getenv("LLM_REMOTE_SSH_USER"),
            "LLM_REMOTE_SSH_PASSWORD": os.getenv("LLM_REMOTE_SSH_PASSWORD"),
        }
        try:
            os.environ["APP_LOCAL_CONFIG"] = str(PROJECT_ROOT / "tests" / "missing.config.local.json")
            os.environ["LLM_REMOTE_DIRECT_ENABLE"] = "1"
            os.environ["LLM_REMOTE_ENABLE"] = "1"
            os.environ["LLM_REMOTE_REQUIRED"] = "0"
            for key in (
                "LLM_REMOTE_SSH_HOST",
                "LLM_REMOTE_SSH_PORT",
                "LLM_REMOTE_SSH_USER",
                "LLM_REMOTE_SSH_PASSWORD",
            ):
                os.environ.pop(key, None)

            client = LLMApiClient()
            with self.assertRaises(RuntimeError) as ctx:
                client.health()
            self.assertIn("Remote server connection is not configured", str(ctx.exception))
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_rank_prompts_returns_explicit_heuristic_fallback_when_remote_request_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_env = {
                "APP_LOCAL_CONFIG": os.getenv("APP_LOCAL_CONFIG"),
                "LLM_REMOTE_DIRECT_ENABLE": os.getenv("LLM_REMOTE_DIRECT_ENABLE"),
                "LLM_REMOTE_ENABLE": os.getenv("LLM_REMOTE_ENABLE"),
                "LLM_REMOTE_MODEL_ID": os.getenv("LLM_REMOTE_MODEL_ID"),
            }
            try:
                os.environ["APP_LOCAL_CONFIG"] = str(Path(tmp_dir) / "missing.json")
                os.environ["LLM_REMOTE_DIRECT_ENABLE"] = "1"
                os.environ["LLM_REMOTE_ENABLE"] = "1"
                os.environ["LLM_REMOTE_MODEL_ID"] = "G:/models/Qwen2.5-7B-Instruct"

                client = LLMApiClient()
                client._prompt_cache = {}
                client._prompt_cache_loaded = True

                with patch.object(client, "_request", side_effect=RuntimeError("remote tokenizer failed")):
                    result = client.rank_prompts("\ud558\uc580 \uc2b9\uc6a9\ucc28", n=3, debug=False)

                self.assertEqual(result["load_mode"], "local-heuristic-fallback")
                self.assertEqual(result["items"][0]["english_prompt"], "white car")
                self.assertEqual(result["_meta"]["fallback_mode"], "local_heuristic")
                self.assertIn("remote tokenizer failed", result["_meta"]["fallback_reason"])
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_rank_prompts_promotes_korean_heuristics_ahead_of_remote_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_env = {
                "APP_LOCAL_CONFIG": os.getenv("APP_LOCAL_CONFIG"),
                "LLM_REMOTE_DIRECT_ENABLE": os.getenv("LLM_REMOTE_DIRECT_ENABLE"),
                "LLM_REMOTE_ENABLE": os.getenv("LLM_REMOTE_ENABLE"),
                "LLM_REMOTE_MODEL_ID": os.getenv("LLM_REMOTE_MODEL_ID"),
            }
            try:
                os.environ["APP_LOCAL_CONFIG"] = str(Path(tmp_dir) / "missing.json")
                os.environ["LLM_REMOTE_DIRECT_ENABLE"] = "0"
                os.environ["LLM_REMOTE_ENABLE"] = "0"
                os.environ["LLM_REMOTE_MODEL_ID"] = "G:/models/Qwen2.5-7B-Instruct"

                client = LLMApiClient()
                client._prompt_cache = {}
                client._prompt_cache_loaded = True

                with patch.object(
                    client,
                    "_request",
                    return_value={
                        "model_id": "G:/models/Qwen2.5-7B-Instruct",
                        "load_mode": "cuda-4bit",
                        "device": "cuda:0",
                        "items": [
                            {"english_prompt": "cat", "korean_gloss": "\uace0\uc591\uc774", "probability": 0.7},
                            {"english_prompt": "phone", "korean_gloss": "\uc804\ud654\uae30", "probability": 0.2},
                        ],
                    },
                ):
                    result = client.rank_prompts("\ud558\uc580 \uc2b9\uc6a9\ucc28", n=3, debug=False)

                prompts = [str(item.get("english_prompt", "")) for item in result.get("items", [])]
                self.assertGreaterEqual(len(prompts), 2)
                self.assertEqual(prompts[0], "white car")
                self.assertEqual(prompts[1], "car")
                self.assertTrue(bool(result.get("_meta", {}).get("heuristic_promoted")))
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_local_heuristic_fallback_payload_is_not_cached(self) -> None:
        client = LLMApiClient()
        client._prompt_cache = {}
        client._prompt_cache_loaded = True
        payload = {
            "model_id": "G:/models/Qwen2.5-7B-Instruct",
            "load_mode": "local-heuristic-fallback",
            "device": "local",
            "items": [{"english_prompt": "white car", "korean_gloss": "", "probability": 1.0, "loss": 0.0}],
            "_meta": {"fallback_mode": "local_heuristic", "fallback_reason": "remote failed"},
        }

        with patch.object(client, "_persist_prompt_cache") as persist_mock:
            client._store_rank_payload(
                user_text="하얀 승용차",
                n=3,
                debug=False,
                model_id="G:/models/Qwen2.5-7B-Instruct",
                payload=payload,
            )

        self.assertEqual(client._prompt_cache, {})
        persist_mock.assert_not_called()

    def test_cached_local_heuristic_fallback_payload_is_ignored(self) -> None:
        client = LLMApiClient()
        client._prompt_cache_loaded = True
        cache_key = client._prompt_cache_key(
            user_text="하얀 승용차",
            n=3,
            debug=False,
            model_id="G:/models/Qwen2.5-7B-Instruct",
        )
        client._prompt_cache = {
            cache_key: {
                "model_id": "G:/models/Qwen2.5-7B-Instruct",
                "load_mode": "local-heuristic-fallback",
                "device": "local",
                "items": [{"english_prompt": "white car", "korean_gloss": "", "probability": 1.0, "loss": 0.0}],
                "_meta": {"fallback_mode": "local_heuristic", "fallback_reason": "remote failed"},
            }
        }

        cached = client._cached_rank_payload(
            user_text="하얀 승용차",
            n=3,
            debug=False,
            model_id="G:/models/Qwen2.5-7B-Instruct",
        )

        self.assertIsNone(cached)

    def test_rank_prompts_uses_generic_fallback_for_non_korean_heuristic_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_env = {
                "APP_LOCAL_CONFIG": os.getenv("APP_LOCAL_CONFIG"),
                "LLM_REMOTE_DIRECT_ENABLE": os.getenv("LLM_REMOTE_DIRECT_ENABLE"),
                "LLM_REMOTE_ENABLE": os.getenv("LLM_REMOTE_ENABLE"),
                "LLM_REMOTE_MODEL_ID": os.getenv("LLM_REMOTE_MODEL_ID"),
            }
            try:
                os.environ["APP_LOCAL_CONFIG"] = str(Path(tmp_dir) / "missing.json")
                os.environ["LLM_REMOTE_DIRECT_ENABLE"] = "1"
                os.environ["LLM_REMOTE_ENABLE"] = "1"
                os.environ["LLM_REMOTE_MODEL_ID"] = "G:/models/Qwen2.5-7B-Instruct"

                client = LLMApiClient()
                client._prompt_cache = {}
                client._prompt_cache_loaded = True

                with patch.object(client, "_request", side_effect=RuntimeError("remote tokenizer failed")):
                    result = client.rank_prompts("pickup truck", n=3, debug=False)

                self.assertEqual(result["load_mode"], "local-heuristic-fallback")
                self.assertEqual(result["items"][0]["english_prompt"], "pickup truck")
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_request_uses_http_fallback_when_remote_direct_fails(self) -> None:
        previous_env = {
            "LLM_REMOTE_DIRECT_ENABLE": os.getenv("LLM_REMOTE_DIRECT_ENABLE"),
            "LLM_REMOTE_ENABLE": os.getenv("LLM_REMOTE_ENABLE"),
        }
        try:
            os.environ["LLM_REMOTE_DIRECT_ENABLE"] = "1"
            os.environ["LLM_REMOTE_ENABLE"] = "1"
            client = LLMApiClient()
            with patch.object(client, "_remote_direct_request", side_effect=RuntimeError("direct failed")):
                with patch.object(client, "_http_health_available", return_value=True):
                    with patch.object(client, "_http_request", return_value={"status": "ok"}) as http_mock:
                        result = client._request("POST", "/rank-prompts", payload={"user_text": "white car"})

            self.assertEqual(result, {"status": "ok"})
            http_mock.assert_called_once()
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_normalize_discovered_remote_python_candidate_parses_conda_env_line(self) -> None:
        candidate = _normalize_discovered_remote_python_candidate(
            "PJ_310_LLM_SAM3          G:/conda/envs/PJ_310_LLM_SAM3"
        )
        self.assertEqual(candidate, "G:/conda/envs/PJ_310_LLM_SAM3/python.exe")

    def test_qwen_coder_env_is_not_an_allowed_remote_python_candidate(self) -> None:
        candidate = _normalize_discovered_remote_python_candidate("G:/conda/envs/qwen-coder32b/python.exe")
        self.assertEqual(candidate, "G:/conda/envs/qwen-coder32b/python.exe")
        self.assertFalse(_is_allowed_remote_python_candidate(candidate))


if __name__ == "__main__":
    unittest.main()
