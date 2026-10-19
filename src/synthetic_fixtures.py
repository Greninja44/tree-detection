"""Synthetic demo fixtures for the Streamlit app and manual smoke tests.

**These images are computer-generated rectangles, not real forest photographs
and not real annotation data.**  They exist so the UI and pipeline can be
exercised end-to-end before any real dataset or trained weights are available.
"""

from __future__ import annotations

import cv2
import numpy as np


def synthetic_forest_image(
    width: int = 960,
    height: int = 640,
    n_trunks: int = 6,
    seed: int = 7,
    brightness: int = 150,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Generate a synthetic image with vertical trunk-like rectangles.

    Returns ``(bgr_image, ground_truth_boxes_xyxy)``.
    """
    rng = np.random.default_rng(seed)
    bg = int(np.clip(brightness * 0.55, 0, 255))
    image = np.full((height, width, 3), (bg, min(bg + 6, 255), bg), dtype=np.uint8)
    image = np.clip(
        image.astype(np.float32) + rng.normal(0, 6, size=image.shape), 0, 255
    ).astype(np.uint8)

    # Foliage-like blobs at the top so the scene reads as "forest-ish".
    for _ in range(max(4, n_trunks * 2)):
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height // 3))
        r = int(rng.integers(width // 20, width // 8))
        green = (
            int(np.clip(brightness * 0.3, 10, 255)),
            int(np.clip(brightness * 0.7, 20, 255)),
            int(np.clip(brightness * 0.3, 10, 255)),
        )
        cv2.circle(image, (cx, cy), r, green, -1)

    boxes: list[tuple[int, int, int, int]] = []
    margin = width // (n_trunks + 1)
    for i in range(n_trunks):
        w = int(rng.integers(width // 34, width // 22))
        x = int(np.clip(margin * (i + 1) - w // 2, 0, width - w - 1))
        trunk_h = int(rng.integers(int(height * 0.55), int(height * 0.95)))
        y0 = height - trunk_h
        brown = (
            int(np.clip(brightness * 0.35, 20, 255)),
            int(np.clip(brightness * 0.55, 30, 255)),
            int(np.clip(brightness * 0.45, 25, 255)),
        )
        cv2.rectangle(image, (x, y0), (x + w, height - 1), brown, -1)
        boxes.append((x, y0, x + w, height - 1))
    return image, boxes
