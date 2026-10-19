"""Shared fixtures.

**All images generated here are synthetic test fixtures** (simple rectangles
mimicking vertical trunks).  They exist to validate pipeline mechanics —
statistics, metrics, drawing — and must never be presented as real forestry
evaluation data.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest


def _synthetic_forest(
    width: int = 640,
    height: int = 480,
    n_trunks: int = 5,
    seed: int = 7,
    brightness: int = 140,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Build a synthetic image of vertical 'trunk' rectangles.

    Returns ``(bgr_image, boxes_xyxy)`` where *boxes* are the drawn rectangles —
    a known ground truth for pipeline validation (not real annotation data).
    """
    rng = np.random.default_rng(seed)
    bg = int(np.clip(brightness * 0.55, 0, 255))
    image = np.full((height, width, 3), (bg, bg + 6, bg), dtype=np.uint8)
    # Slight texture noise so contrast statistics are non-degenerate.
    noise = rng.normal(0, 6, size=image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    boxes: list[tuple[int, int, int, int]] = []
    margin = width // (n_trunks + 1)
    for i in range(n_trunks):
        w = int(rng.integers(width // 28, width // 18))
        x = margin * (i + 1) - w // 2
        x = int(np.clip(x, 0, width - w - 1))
        trunk_h = int(rng.integers(int(height * 0.6), int(height * 0.95)))
        y0 = height - trunk_h
        brown = (
            int(np.clip(brightness * 0.35, 20, 255)),
            int(np.clip(brightness * 0.55, 30, 255)),
            int(np.clip(brightness * 0.45, 25, 255)),
        )
        cv2.rectangle(image, (x, y0), (x + w, height - 1), brown, -1)
        boxes.append((x, y0, x + w, height - 1))
    return image, boxes


@pytest.fixture(scope="session")
def synthetic_forest() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Synthetic BGR image with known trunk-like rectangles (fixture data only)."""
    return _synthetic_forest()


@pytest.fixture()
def bright_image() -> np.ndarray:
    img, _ = _synthetic_forest(brightness=230, seed=1)
    return img


@pytest.fixture()
def dark_image() -> np.ndarray:
    img, _ = _synthetic_forest(brightness=35, seed=2)
    return img


@pytest.fixture()
def gray_image() -> np.ndarray:
    """Uniform gray image (value 100) for exact statistics checks."""
    return np.full((40, 60), 100, dtype=np.uint8)


@pytest.fixture()
def image_dir(tmp_path):
    """Folder with two valid synthetic images (fixture data only)."""
    folder = tmp_path / "images"
    folder.mkdir()
    for i in range(2):
        img, _ = _synthetic_forest(seed=10 + i)
        cv2.imwrite(str(folder / f"synth_{i:03d}.jpg"), img)
    return folder
