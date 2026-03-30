from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.labeling.sam3_runner import _build_runtime_config, _persist_preview_items_to_cache
from core.models import PreviewThumbnail


class Sam3RuntimePerfTests(unittest.TestCase):
    def test_build_runtime_config_keeps_preview_video_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "a100.json"
            config_path.write_text(
                json.dumps(
                    {
                        "sam3_root": "../../sam3",
                        "run_root": "../runs",
                        "device": "cuda",
                        "prompts": ["car."],
                        "output": {
                            "save_preview_video": False,
                            "save_masks": True,
                            "show_progress": True,
                            "live_preview": True,
                            "ui_emit_interval": 2,
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            previous = os.getenv("NEW_OBJECT_REMOTE_SAVE_PREVIEW_VIDEO")
            try:
                os.environ.pop("NEW_OBJECT_REMOTE_SAVE_PREVIEW_VIDEO", None)
                payload = json.loads(
                    _build_runtime_config(
                        local_config_path=config_path,
                        remote_sam3_root="G:/models/sam3",
                        remote_device="cuda",
                        remote_runs_root="G:/runs/sam3",
                        prompts=["bus."],
                    )
                )
            finally:
                if previous is None:
                    os.environ.pop("NEW_OBJECT_REMOTE_SAVE_PREVIEW_VIDEO", None)
                else:
                    os.environ["NEW_OBJECT_REMOTE_SAVE_PREVIEW_VIDEO"] = previous

        self.assertFalse(payload["output"]["save_preview_video"])
        self.assertFalse(payload["output"]["save_masks"])
        self.assertFalse(payload["output"]["show_progress"])
        self.assertFalse(payload["output"]["live_preview"])
        self.assertEqual(payload["output"]["ui_emit_interval"], 2)
        self.assertEqual(payload["prompts"], ["bus."])

    def test_preview_cache_reuses_frame_thumb_per_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            video_path = temp_root / "sample.avi"
            self._write_sample_video(video_path)
            cache_root = temp_root / "cache"
            items = [
                PreviewThumbnail(
                    frame_index=0,
                    category="keep",
                    item_id="keep_a",
                    boxes=[{"x1": 1, "y1": 1, "x2": 10, "y2": 10, "status": "keep"}],
                ),
                PreviewThumbnail(
                    frame_index=0,
                    category="hold",
                    item_id="hold_a",
                    boxes=[{"x1": 2, "y1": 2, "x2": 11, "y2": 11, "status": "hold"}],
                ),
                PreviewThumbnail(
                    frame_index=1,
                    category="drop",
                    item_id="drop_a",
                    boxes=[{"x1": 3, "y1": 3, "x2": 12, "y2": 12, "status": "drop"}],
                ),
            ]

            cached_count = _persist_preview_items_to_cache(
                cache_root,
                items,
                video_path=str(video_path),
            )

            keep_manifest = json.loads((cache_root / "keep" / "keep_a.json").read_text(encoding="utf-8"))
            hold_manifest = json.loads((cache_root / "hold" / "hold_a.json").read_text(encoding="utf-8"))
            drop_manifest = json.loads((cache_root / "drop" / "drop_a.json").read_text(encoding="utf-8"))
            self.assertEqual(cached_count, 3)
            self.assertEqual(keep_manifest["thumb_path"], hold_manifest["thumb_path"])
            self.assertNotEqual(keep_manifest["thumb_path"], drop_manifest["thumb_path"])
            self.assertTrue(Path(keep_manifest["thumb_path"]).is_file())
            self.assertTrue(Path(drop_manifest["thumb_path"]).is_file())
            self.assertEqual(len(list((cache_root / "_frames").glob("*.jpg"))), 2)

    def _write_sample_video(self, video_path: Path) -> None:
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
        self.assertTrue(writer.isOpened())
        try:
            writer.write(np.full((24, 32, 3), 32, dtype=np.uint8))
            writer.write(np.full((24, 32, 3), 196, dtype=np.uint8))
        finally:
            writer.release()


if __name__ == "__main__":
    unittest.main()
