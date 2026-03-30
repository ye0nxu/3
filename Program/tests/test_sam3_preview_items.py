from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.labeling.sam3_runner import _build_single_box_preview_item


class Sam3PreviewItemTests(unittest.TestCase):
    def test_single_box_preview_item_uses_one_box_per_sample(self) -> None:
        item = _build_single_box_preview_item(
            experiment_id="exp01",
            frame_index=42,
            category="keep",
            sample_index=7,
            box_payload={
                "x1": 10,
                "y1": 20,
                "x2": 30,
                "y2": 40,
                "track_id": 3,
                "score": 0.8,
                "status": "hold",
            },
        )

        self.assertEqual(item.category, "keep")
        self.assertEqual(len(item.boxes), 1)
        self.assertEqual(item.boxes[0]["status"], "keep")
        self.assertEqual(item.boxes[0]["preview_item_id"], item.item_id)
        self.assertIn("trk0003", item.item_id)


if __name__ == "__main__":
    unittest.main()
