from __future__ import annotations

import cv2
import numpy as np


def dhash64(image_bgr: np.ndarray) -> int:
    """Return 64-bit dHash as int."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    bit_index = 0
    for row in diff:
        for flag in row:
            if bool(flag):
                value |= 1 << bit_index
            bit_index += 1
    return int(value)


def hamming_distance64(a: int, b: int) -> int:
    return int((int(a) ^ int(b)).bit_count())


def hash_prefix(value: int, prefix_bits: int) -> int:
    bits = max(1, min(63, int(prefix_bits)))
    return int(value >> (64 - bits))


def make_thumb_gray(image_bgr: np.ndarray, size: int = 32) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

