"""Synthetic lighting transformations for robustness testing.

Every transformation takes an image plus a configurable ``strength`` in
``[0, 1]`` and returns a **new** array; the input image is never modified in
place.  Transformed images are *synthetic test conditions* and are tracked with
the ``synthetic:`` prefix so they are never confused with genuine real-world
lighting labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .lighting_analysis import SYNTHETIC_PREFIX
from .preprocessing import ensure_bgr

logger = logging.getLogger(__name__)


class TransformationError(ValueError):
    """Raised for unknown transformations or invalid strength values."""


def _check_strength(strength: float) -> float:
    try:
        s = float(strength)
    except (TypeError, ValueError) as exc:
        raise TransformationError(f"strength must be a number, got {strength!r}") from exc
    if not 0.0 <= s <= 1.0:
        raise TransformationError(f"strength must be within [0, 1], got {s}")
    return s


def _identity(image: np.ndarray, strength: float) -> np.ndarray:
    """Return an unchanged copy (strength is unused)."""
    _check_strength(strength)
    return image.copy()


def adjust_brightness(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Darken the image.  ``strength=0`` → unchanged, ``1`` → strongly dark."""
    s = _check_strength(strength)
    image = ensure_bgr(image)
    # beta in [-160, 0] keeps most pixels from clipping to pure black.
    beta = -160.0 * s
    return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)


def increase_brightness(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Brighten the image.  ``strength=1`` → strongly over-exposed."""
    s = _check_strength(strength)
    image = ensure_bgr(image)
    # alpha in [1, 1.9] plus a small beta lift.
    alpha = 1.0 + 0.9 * s
    beta = 25.0 * s
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def adjust_gamma(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Gamma correction with strength mapped to gamma ∈ [0.35, 1.0] (darker).

    Use :func:`increase_brightness` for the bright direction.
    """
    s = _check_strength(strength)
    image = ensure_bgr(image)
    gamma = 1.0 - 0.65 * s
    inv = 1.0 / max(gamma, 1e-3)
    table = np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)], dtype=np.float32)
    return cv2.LUT(image, table.astype(np.uint8))


def adjust_contrast(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Reduce contrast (strength 1 → very flat, low-contrast image)."""
    s = _check_strength(strength)
    image = ensure_bgr(image).astype(np.float32)
    mean = float(image.mean())
    factor = 1.0 - 0.75 * s
    out = (image - mean) * factor + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_shadow(image: np.ndarray, strength: float = 0.5, seed: int | None = 0) -> np.ndarray:
    """Simulate a shadow-like gradient darkening one side of the frame.

    A linear ramp multiplies pixel values, darkest at one edge.  ``strength``
    controls how dark the shadowed edge becomes.  *seed* makes the shadow side
    deterministic for reproducible tests.
    """
    s = _check_strength(strength)
    image = ensure_bgr(image)
    h, w = image.shape[:2]
    rng = np.random.default_rng(seed)
    horizontal = bool(rng.integers(0, 2))
    ramp = np.linspace(1.0 - 0.8 * s, 1.0, w if horizontal else h, dtype=np.float32)
    mask = np.tile(ramp, (h, 1)) if horizontal else np.tile(ramp.reshape(-1, 1), (1, w))
    if not horizontal:
        if bool(rng.integers(0, 2)):
            mask = mask[::-1, ...]
    mask3 = mask[..., None]
    out = image.astype(np.float32) * mask3
    return np.clip(out, 0, 255).astype(np.uint8)


def add_noise(image: np.ndarray, strength: float = 0.5, seed: int | None = 0) -> np.ndarray:
    """Add mild Gaussian sensor noise.  ``strength=1`` ≈ sigma 18/255."""
    s = _check_strength(strength)
    image = ensure_bgr(image)
    sigma = 18.0 * s
    if sigma <= 0:
        return image.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    out = image.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


#: Registry of available transformations: ``name -> (function, default strength, description)``.
TRANSFORMATIONS: dict[str, tuple[Callable[..., np.ndarray], float, str]] = {
    "original": (_identity, 0.0, "Unchanged reference image"),
    "darker": (adjust_brightness, 0.6, "Brightness reduction (darker scene)"),
    "brighter": (increase_brightness, 0.6, "Brightness increase (over-exposed scene)"),
    "gamma": (adjust_gamma, 0.5, "Gamma correction (non-linear darkening)"),
    "low_contrast": (adjust_contrast, 0.6, "Reduced contrast (flat, hazy light)"),
    "shadow": (apply_shadow, 0.6, "Shadow-like gradient darkening"),
    "noise": (add_noise, 0.5, "Mild sensor noise (low-light grain)"),
}

#: Names of the transformations, in a stable display order.
TRANSFORMATION_NAMES: tuple[str, ...] = tuple(TRANSFORMATIONS)


def apply_transformation(
    image: np.ndarray,
    name: str,
    strength: float | None = None,
    seed: int | None = 0,
) -> np.ndarray:
    """Apply the transformation *name* to *image* and return a new array.

    Raises ``TransformationError`` for unknown names or out-of-range strength.
    """
    if name not in TRANSFORMATIONS:
        raise TransformationError(
            f"Unknown transformation '{name}'. Available: {', '.join(TRANSFORMATION_NAMES)}"
        )
    fn, default_strength, _ = TRANSFORMATIONS[name]
    s = default_strength if strength is None else _check_strength(strength)
    try:
        return fn(image, s, seed=seed) if name in {"shadow", "noise"} else fn(image, s)
    except TransformationError:
        raise
    except Exception as exc:  # pragma: no cover - defensive, cv2 failures
        raise TransformationError(f"Transformation '{name}' failed: {exc}") from exc


def generate_variants(
    image: np.ndarray,
    names: list[str] | tuple[str, ...] | None = None,
    strength: float | None = None,
    seed: int | None = 0,
) -> dict[str, np.ndarray]:
    """Generate several transformed copies of *image*.

    Returns a dict keyed by the original name (``"original"`` maps to an
    unchanged copy).  The source image is never modified.
    """
    selected = list(names) if names else list(TRANSFORMATION_NAMES)
    original = ensure_bgr(image)
    source = original.copy()
    variants: dict[str, np.ndarray] = {}
    for name in selected:
        variants[name] = apply_transformation(source, name, strength=strength, seed=seed)
    logger.debug("Generated %d lighting variant(s)", len(variants))
    return variants


def synthetic_category(name: str) -> str:
    """Return the tracking label for a synthetic condition (``synthetic:<name>``)."""
    if name == "original":
        return "original"
    return f"{SYNTHETIC_PREFIX}{name}"


def save_variant(image: np.ndarray, path: str | Path, name: str = "variant") -> Path:
    """Save a transformed image to *path* (creating parent directories)."""
    p = Path(path).expanduser()
    if p.is_dir() or p.suffix == "":
        p = p / f"{name}.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(p), ensure_bgr(image))
    if not ok:
        raise IOError(f"Could not write transformed image to {p}")
    return p
