"""Image-based lighting analysis and real-world lighting category labels.

Two complementary notions of "lighting condition" exist in this project:

1. **Real-world categories** (``REAL_WORLD_CATEGORIES``) are labels a human
   assigns to an image: bright daylight, partial shade, deep shadow, overcast,
   low light.  They are stored alongside evaluation results.
2. **Measured statistics** (``LightingStats``) are computed from pixels: mean /
   median luminance, contrast, dark & bright pixel percentages, HSV value
   statistics.  A heuristic label derived from these numbers is provided as a
   convenience only.

**Important:** brightness thresholds are heuristics.  They do **not** prove the
weather, the time of day, or the actual illumination geometry of a scene.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .preprocessing import ensure_bgr

#: Real-world lighting categories that users may assign to images.
REAL_WORLD_CATEGORIES: tuple[str, ...] = (
    "bright_daylight",
    "partial_shade",
    "deep_shadow",
    "overcast",
    "low_light",
)

#: Category used when no real-world label has been assigned.
UNLABELED_CATEGORY = "unlabeled"

#: Synthetic test conditions are tracked separately from real-world labels.
SYNTHETIC_PREFIX = "synthetic:"

#: Convenience mapping of category keys to human-readable names.
CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "bright_daylight": "Bright daylight",
    "partial_shade": "Partial shade",
    "deep_shadow": "Deep shadow",
    "overcast": "Overcast / cloudy",
    "low_light": "Low light",
    "unlabeled": "Unlabeled",
    "unknown": "Unknown",
}


@dataclasses.dataclass(frozen=True)
class LightingStats:
    """Illumination statistics computed from a single image."""

    mean_gray: float
    median_gray: float
    contrast_std: float
    dark_pixel_pct: float
    bright_pixel_pct: float
    hsv_value_mean: float
    hsv_value_std: float
    estimated_category: str
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of the statistics."""
        return dataclasses.asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize the statistics to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def _to_gray(image: np.ndarray) -> np.ndarray:
    image = ensure_bgr(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def analyze_lighting(image: np.ndarray) -> LightingStats:
    """Compute illumination statistics for *image*.

    Parameters
    ----------
    image:
        Grayscale, BGR or BGRA image as ``uint8`` numpy array.

    Notes
    -----
    - ``dark_pixel_pct``  : percentage of pixels with gray value ``< 60``.
    - ``bright_pixel_pct``: percentage of pixels with gray value ``> 200``.
    - ``contrast_std``    : standard deviation of gray intensity (global contrast proxy).
    - ``estimated_category`` is a **heuristic** label, not a verified observation.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("analyze_lighting expects a non-empty numpy image array.")

    gray = _to_gray(image).astype(np.float32)
    hsv = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2HSV)
    value = hsv[..., 2].astype(np.float32)

    mean_gray = float(gray.mean())
    median_gray = float(np.median(gray))
    contrast_std = float(gray.std())
    dark_pct = float((gray < 60).mean() * 100.0)
    bright_pct = float((gray > 200).mean() * 100.0)

    return LightingStats(
        mean_gray=round(mean_gray, 2),
        median_gray=round(median_gray, 2),
        contrast_std=round(contrast_std, 2),
        dark_pixel_pct=round(dark_pct, 2),
        bright_pixel_pct=round(bright_pct, 2),
        hsv_value_mean=round(float(value.mean()), 2),
        hsv_value_std=round(float(value.std()), 2),
        estimated_category=estimate_category(mean_gray, contrast_std, dark_pct, bright_pct),
        width=int(image.shape[1]),
        height=int(image.shape[0]),
    )


def estimate_category(
    mean_gray: float,
    contrast_std: float,
    dark_pixel_pct: float,
    bright_pixel_pct: float,
    *,
    low_light_max: float = 60.0,
    dark_max: float = 100.0,
    bright_min: float = 140.0,
    high_contrast_min: float = 65.0,
) -> str:
    """Heuristically map measured statistics to a lighting category.

    These thresholds are a **convenience estimate**.  A dark scene might be deep
    shade at noon or a forest at night; the pixels alone cannot distinguish
    them.  Treat the result as a suggestion and prefer user-assigned labels for
    reporting.

    Returns one of ``REAL_WORLD_CATEGORIES`` or ``"unknown"``.
    """
    if mean_gray < low_light_max:
        return "low_light"
    if mean_gray < dark_max:
        return "deep_shadow" if dark_pixel_pct >= 35.0 else "partial_shade"
    if mean_gray >= bright_min:
        if contrast_std >= high_contrast_min and dark_pixel_pct > 8.0:
            return "bright_daylight"  # strong direct sunlight with hard shadows
        return "bright_daylight"
    # Mid-range brightness: very flat contrast suggests overcast light.
    if contrast_std < 40.0 and dark_pixel_pct < 25.0:
        return "overcast"
    if dark_pixel_pct >= 35.0 or bright_pixel_pct <= 1.0:
        return "deep_shadow"
    return "partial_shade"


def load_category_labels(path: str | Path) -> dict[str, str]:
    """Load a mapping of ``filename -> lighting category`` from a JSON file.

    Expected JSON shape: ``{"forest_001.jpg": "overcast", ...}``.
    Unknown categories are mapped to ``"unlabeled"`` rather than raising.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Lighting label file not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Lighting label file must be a JSON object: {p}")
    labels: dict[str, str] = {}
    for name, cat in raw.items():
        cat = str(cat).strip().lower()
        labels[str(name)] = cat if cat in REAL_WORLD_CATEGORIES else UNLABELED_CATEGORY
    return labels


def save_category_labels(labels: dict[str, str], path: str | Path) -> Path:
    """Write a ``filename -> category`` mapping to *path* as JSON."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")
    return p


def describe_category(category: str) -> str:
    """Return a human-readable name for *category*."""
    if category.startswith(SYNTHETIC_PREFIX):
        base = category[len(SYNTHETIC_PREFIX) :]
        return f"Synthetic: {base}"
    return CATEGORY_DISPLAY_NAMES.get(category, category)
