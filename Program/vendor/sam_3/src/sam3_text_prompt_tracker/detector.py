from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class Detection:
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    prompt: str
    mask: Optional[np.ndarray] = None


def _resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _resolve_existing_path(base: Path, preferred: str, fallbacks: Sequence[str]) -> Path:
    candidates: List[Path] = []
    candidates.append(_resolve_path(base, preferred))
    for fb in fallbacks:
        if fb == preferred:
            continue
        candidates.append(_resolve_path(base, fb))
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _ensure_sam3_importable(sam3_root: Path) -> None:
    import sys

    root = str(sam3_root.resolve())
    # 업로드된 vendor 버전(perflib Windows 패치 포함)이 sys.path 앞에 있으면
    # insert(0)으로 덮어쓰지 않고 append로 추가하여 업로드 버전 우선 유지
    if root not in sys.path:
        current_file = Path(__file__).resolve()
        vendor_sam3_root = current_file.parent.parent.parent  # vendor/sam_3
        vendor_sam3_mod = vendor_sam3_root / "sam3"
        if vendor_sam3_mod.is_dir() and str(vendor_sam3_root) in sys.path:
            sys.path.append(root)
        else:
            sys.path.insert(0, root)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _nms(detections: Sequence[Detection], iou_threshold: float) -> List[Detection]:
    if not detections:
        return []
    order = sorted(range(len(detections)), key=lambda i: detections[i].score, reverse=True)
    keep: List[Detection] = []
    while order:
        cur = order.pop(0)
        cur_det = detections[cur]
        keep.append(cur_det)
        remaining: List[int] = []
        cur_box = np.asarray(cur_det.bbox_xyxy, dtype=np.float32)
        for idx in order:
            other_box = np.asarray(detections[idx].bbox_xyxy, dtype=np.float32)
            if _iou_xyxy(cur_box, other_box) <= iou_threshold:
                remaining.append(idx)
        order = remaining
    return keep


class Sam3TextDetector:
    def __init__(
        self,
        config_base: Path,
        sam3_root: str,
        checkpoint: str,
        bpe: str,
        device: str,
        confidence_threshold: float,
        infer_size: Optional[Tuple[int, int]],
        nms_iou_threshold: float,
        min_area: float,
        max_per_prompt: int = 0,
        max_total: int = 0,
        use_amp: bool = True,
    ) -> None:
        self.config_base = config_base.resolve()
        self.sam3_root = _resolve_path(self.config_base, sam3_root)
        self.checkpoint_path = _resolve_existing_path(
            self.sam3_root,
            checkpoint,
            ["sam3/checkpoints/sam3.pt", "checkpoints/sam3.pt"],
        )
        self.bpe_path = _resolve_existing_path(
            self.sam3_root,
            bpe,
            ["sam3/assets/bpe_simple_vocab_16e6.txt.gz", "assets/bpe_simple_vocab_16e6.txt.gz"],
        )
        self.device = device
        self.infer_size = infer_size
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.min_area = float(min_area)
        self.max_per_prompt = int(max_per_prompt)
        self.max_total = int(max_total)
        self.use_amp = bool(use_amp)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {self.checkpoint_path}")
        if not self.bpe_path.exists():
            raise FileNotFoundError(f"SAM3 BPE file not found: {self.bpe_path}")

        _ensure_sam3_importable(self.sam3_root)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = build_sam3_image_model(
            bpe_path=str(self.bpe_path),
            checkpoint_path=str(self.checkpoint_path),
            load_from_HF=False,
            device=self.device,
        )
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=confidence_threshold,
        )

    @staticmethod
    def _to_numpy(x):
        if x is None:
            return None
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def detect(
        self,
        frame_bgr: np.ndarray,
        prompts: Sequence[str],
        include_masks: bool = False,
    ) -> List[Detection]:
        if frame_bgr is None:
            return []
        if len(prompts) == 0:
            return []

        orig_h, orig_w = frame_bgr.shape[:2]
        proc = frame_bgr
        if self.infer_size is not None:
            infer_w, infer_h = self.infer_size
            if infer_w > 0 and infer_h > 0 and (infer_w != orig_w or infer_h != orig_h):
                proc = cv2.resize(frame_bgr, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
        proc_h, proc_w = proc.shape[:2]
        sx = float(orig_w) / float(proc_w)
        sy = float(orig_h) / float(proc_h)

        proc_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

        try:
            import torch

            use_cuda_amp = self.use_amp and str(self.device).lower().startswith("cuda")
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_cuda_amp
                else nullcontext()
            )
            infer_ctx = torch.inference_mode()
        except Exception:
            autocast_ctx = nullcontext()
            infer_ctx = nullcontext()

        detections: List[Detection] = []
        with infer_ctx:
            with autocast_ctx:
                state = self.processor.set_image(Image.fromarray(proc_rgb))

                for prompt in prompts:
                    p = prompt.strip()
                    if not p:
                        continue
                    out = self.processor.set_text_prompt(prompt=p, state=state)
                    boxes = self._to_numpy(out.get("boxes"))
                    scores = self._to_numpy(out.get("scores"))
                    masks = self._to_numpy(out.get("masks")) if include_masks else None
                    if boxes is None or scores is None:
                        continue
                    if boxes.size == 0:
                        continue

                    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
                    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
                    if self.max_per_prompt > 0 and len(scores) > self.max_per_prompt:
                        order = np.argsort(scores)[::-1][: self.max_per_prompt]
                        boxes = boxes[order]
                        scores = scores[order]
                        if masks is not None and len(masks) >= len(order):
                            masks = np.asarray(masks)[order]

                    for i in range(min(len(boxes), len(scores))):
                        box = boxes[i].copy()
                        box[0] *= sx
                        box[2] *= sx
                        box[1] *= sy
                        box[3] *= sy
                        x0 = float(np.clip(box[0], 0, orig_w - 1))
                        y0 = float(np.clip(box[1], 0, orig_h - 1))
                        x1 = float(np.clip(box[2], 0, orig_w - 1))
                        y1 = float(np.clip(box[3], 0, orig_h - 1))
                        if x1 <= x0 or y1 <= y0:
                            continue
                        area = (x1 - x0) * (y1 - y0)
                        if area < self.min_area:
                            continue

                        mask_arr: Optional[np.ndarray] = None
                        if include_masks and masks is not None and i < len(masks):
                            m = masks[i]
                            if m.ndim == 3:
                                m = m[0]
                            m = m.astype(np.uint8)
                            if (proc_w, proc_h) != (orig_w, orig_h):
                                m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                            mask_arr = m.astype(bool)

                        detections.append(
                            Detection(
                                bbox_xyxy=(x0, y0, x1, y1),
                                score=float(scores[i]),
                                prompt=p,
                                mask=mask_arr,
                            )
                        )

        kept = _nms(detections, self.nms_iou_threshold)
        if self.max_total > 0 and len(kept) > self.max_total:
            kept = sorted(kept, key=lambda d: d.score, reverse=True)[: self.max_total]
        return kept
