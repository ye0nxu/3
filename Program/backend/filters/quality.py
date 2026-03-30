from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import FilterConfig


@dataclass(slots=True)
class QualityMetrics:
    blur_score: float
    white_ratio: float
    black_ratio: float


def compute_blur_score(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_exposure_ratios(
    image_bgr: np.ndarray,
    white_thr: int,
    black_thr: int,
) -> tuple[float, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    white_ratio = float(np.mean(gray >= int(white_thr)))
    black_ratio = float(np.mean(gray <= int(black_thr)))
    return white_ratio, black_ratio


def evaluate_quality(image_bgr: np.ndarray, cfg: FilterConfig) -> tuple[str, QualityMetrics]:
    """Return PASS or exposure/blur drop reason."""
    blur_score = compute_blur_score(image_bgr)
    white_ratio, black_ratio = compute_exposure_ratios(
        image_bgr=image_bgr,
        white_thr=cfg.white_thr,
        black_thr=cfg.black_thr,
    )
    metrics = QualityMetrics(
        blur_score=float(blur_score),
        white_ratio=float(white_ratio),
        black_ratio=float(black_ratio),
    )
    if white_ratio > float(cfg.white_ratio_thr):
        return "DROP_OVEREXPOSED", metrics
    if black_ratio > float(cfg.black_ratio_thr):
        return "DROP_UNDEREXPOSED", metrics
    if blur_score < float(cfg.blur_threshold):
        return "DROP_BLUR", metrics
    return "PASS", metrics
