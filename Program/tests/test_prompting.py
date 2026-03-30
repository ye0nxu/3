from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm.prompting import (
    build_sam_prompt_candidates,
    extract_ranked_prompt_candidates,
    normalize_user_text,
)


class PromptingTests(unittest.TestCase):
    def test_normalize_user_text_strips_control_chars_and_whitespace(self) -> None:
        raw = "\ud770\uc0c9 \n \uc2b9\uc6a9\ucc28\x00 "
        self.assertEqual(normalize_user_text(raw), "\ud770\uc0c9 \uc2b9\uc6a9\ucc28")

    def test_extract_ranked_prompt_candidates_dedupes(self) -> None:
        payload = {
            "items": [
                {"english_prompt": "white car", "korean_gloss": "\ud770\uc0c9 \uc2b9\uc6a9\ucc28"},
                {"english_prompt": "white car", "korean_gloss": "\ud770\uc0c9 \uc790\ub3d9\ucc28"},
                {"english_prompt": "license plate", "korean_gloss": "\ubc88\ud638\ud310"},
            ]
        }
        self.assertEqual(extract_ranked_prompt_candidates(payload), ["white car", "license plate"])

    def test_build_sam_prompt_candidates_uses_korean_heuristics_first(self) -> None:
        candidates = build_sam_prompt_candidates(
            prompt_text="\ud770\uc0c9 \uc2b9\uc6a9\ucc28",
            class_name="car",
            ranked_candidates=["cat", "phone"],
        )
        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0], "white car")
        self.assertIn("white car.", candidates)
        self.assertIn("car", candidates)
        self.assertNotEqual(candidates[0], "cat")

    def test_build_sam_prompt_candidates_keeps_specific_english_prompt_first(self) -> None:
        candidates = build_sam_prompt_candidates(
            prompt_text="white car",
            class_name="car",
        )
        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0], "white car")
        self.assertEqual(candidates[1], "white car.")
        self.assertIn("car", candidates)


if __name__ == "__main__":
    unittest.main()
