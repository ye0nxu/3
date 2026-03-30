from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm.runtime import _tokenize_chat_messages


class _TensorChatTemplateTokenizer:
    def apply_chat_template(self, _messages, **kwargs):
        if kwargs.get("return_dict"):
            raise TypeError("return_dict unsupported")
        if kwargs.get("tokenize") and kwargs.get("return_tensors") == "pt":
            return torch.tensor([11, 12, 13], dtype=torch.long)
        if kwargs.get("tokenize"):
            return [11, 12, 13]
        return "unused"

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("fallback tokenizer call should not run")


class _ManualPromptFallbackTokenizer:
    def apply_chat_template(self, _messages, **kwargs):
        if kwargs.get("tokenize"):
            raise TypeError("tokenize unsupported")
        return "\x00\x00"

    def __call__(self, value, **_kwargs):
        text = str(value[0] if isinstance(value, list) else value)
        if "system: system prompt" in text and "user: white car" in text:
            return {"input_ids": [[21, 22, 23]], "attention_mask": [[1, 1, 1]]}
        raise ValueError("bad prompt")

    def tokenize(self, _text):
        return []

    def convert_tokens_to_ids(self, _pieces):
        return []

    def encode(self, _text, add_special_tokens=False):
        return []


class LLMRuntimeCoreTests(unittest.TestCase):
    def test_tokenize_chat_messages_accepts_tensor_chat_template_output(self) -> None:
        runtime = SimpleNamespace(tokenizer=_TensorChatTemplateTokenizer())

        tokenized = _tokenize_chat_messages(
            runtime,
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "white car"},
            ],
        )

        self.assertEqual(tuple(tokenized["input_ids"].shape), (1, 3))
        self.assertEqual(tuple(tokenized["attention_mask"].shape), (1, 3))
        self.assertTrue(torch.equal(tokenized["attention_mask"], torch.ones((1, 3), dtype=torch.long)))

    def test_tokenize_chat_messages_falls_back_to_manual_prompt_text(self) -> None:
        runtime = SimpleNamespace(tokenizer=_ManualPromptFallbackTokenizer())

        tokenized = _tokenize_chat_messages(
            runtime,
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "white car"},
            ],
        )

        self.assertEqual(tuple(tokenized["input_ids"].shape), (1, 3))
        self.assertEqual(tokenized["input_ids"].tolist(), [[21, 22, 23]])


if __name__ == "__main__":
    unittest.main()
