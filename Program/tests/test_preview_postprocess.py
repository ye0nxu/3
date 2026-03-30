from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipelines.preview_postprocess import postprocess_preview_items
from core.models import PreviewThumbnail


class PreviewPostprocessTests(unittest.TestCase):
    def test_keep_dedupe_prefers_higher_score_and_renumbers_by_track(self) -> None:
        items = [
            PreviewThumbnail(
                frame_index=10,
                category="keep",
                item_id="raw_a",
                boxes=[{"x1": 0, "y1": 0, "x2": 10, "y2": 10, "track_id": 7, "score": 0.2, "status": "keep"}],
            ),
            PreviewThumbnail(
                frame_index=10,
                category="keep",
                item_id="raw_b",
                boxes=[{"x1": 0, "y1": 0, "x2": 10, "y2": 10, "track_id": 7, "score": 0.9, "status": "keep"}],
            ),
            PreviewThumbnail(
                frame_index=11,
                category="hold",
                item_id="raw_c",
                boxes=[{"x1": 1, "y1": 1, "x2": 11, "y2": 11, "track_id": 7, "score": 0.1, "status": "hold"}],
            ),
        ]

        processed = postprocess_preview_items(items)

        self.assertEqual(len(processed), 2)
        self.assertEqual(processed[0].item_id, "id7_001")
        self.assertEqual(processed[1].item_id, "id7_002")
        self.assertEqual(float(processed[0].boxes[0]["score"]), 0.9)


if __name__ == "__main__":
    unittest.main()
