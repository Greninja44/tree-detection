"""Tests for lighting statistics, category heuristics and synthetic transforms."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.lighting_analysis import (
    REAL_WORLD_CATEGORIES,
    analyze_lighting,
    describe_category,
    estimate_category,
    load_category_labels,
    save_category_labels,
)
from src.lighting_augmentation import (
    TRANSFORMATION_NAMES,
    TransformationError,
    apply_transformation,
    generate_variants,
    synthetic_category,
)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def test_uniform_gray_stats_are_exact(gray_image) -> None:
    stats = analyze_lighting(gray_image)
    assert stats.mean_gray == pytest.approx(100.0, abs=0.5)
    assert stats.median_gray == pytest.approx(100.0, abs=0.5)
    assert stats.contrast_std == pytest.approx(0.0, abs=0.5)
    assert stats.dark_pixel_pct == pytest.approx(0.0, abs=0.01)
    assert stats.bright_pixel_pct == pytest.approx(0.0, abs=0.01)
    assert stats.width == 60 and stats.height == 40
    # Mean exactly 100, flat contrast -> the 'overcast' bucket of the heuristic.
    assert stats.estimated_category == "overcast"


def test_dark_and_bright_percentages() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:5] = 255  # top half white, bottom half black
    stats = analyze_lighting(img)
    assert stats.dark_pixel_pct == pytest.approx(50.0, abs=0.5)
    assert stats.bright_pixel_pct == pytest.approx(50.0, abs=0.5)
    assert stats.mean_gray == pytest.approx(127.5, abs=1.0)


def test_stats_json_roundtrip(gray_image) -> None:
    stats = analyze_lighting(gray_image)
    payload = json.loads(stats.to_json())
    assert payload["mean_gray"] == pytest.approx(stats.mean_gray)
    assert set(payload) >= {"contrast_std", "estimated_category", "hsv_value_mean"}


def test_analyze_lighting_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        analyze_lighting(np.zeros((0,)))
    with pytest.raises(Exception):
        analyze_lighting("not an image")  # type: ignore[arg-type]


def test_bright_scene_estimated_category(bright_image) -> None:
    stats = analyze_lighting(bright_image)
    assert stats.mean_gray > 100
    assert stats.estimated_category in REAL_WORLD_CATEGORIES or stats.estimated_category == "unknown"


def test_gray_and_color_paths_agree() -> None:
    rng = np.random.default_rng(0)
    color = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    import cv2

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    s_color = analyze_lighting(color)
    s_gray = analyze_lighting(gray)
    assert s_color.mean_gray == pytest.approx(s_gray.mean_gray, abs=1.0)
    assert s_color.contrast_std == pytest.approx(s_gray.contrast_std, abs=1.0)


# --------------------------------------------------------------------------- #
# Category heuristic
# --------------------------------------------------------------------------- #

def test_estimate_category_boundaries() -> None:
    assert estimate_category(30, 20, 80, 0) == "low_light"
    assert estimate_category(80, 30, 50, 0) == "deep_shadow"
    assert estimate_category(80, 30, 10, 0) == "partial_shade"
    assert estimate_category(160, 30, 5, 10) == "bright_daylight"
    assert estimate_category(120, 30, 5, 10) == "overcast"
    assert estimate_category(190, 80, 15, 40) == "bright_daylight"


def test_category_labels_roundtrip(tmp_path) -> None:
    labels = {"a.jpg": "overcast", "b.jpg": "not_a_category", "c.jpg": "low_light"}
    path = tmp_path / "labels.json"
    save_category_labels(labels, path)
    loaded = load_category_labels(path)
    assert loaded["a.jpg"] == "overcast"
    assert loaded["c.jpg"] == "low_light"
    # Unknown category degrades to 'unlabeled' instead of raising.
    assert loaded["b.jpg"] == "unlabeled"


def test_missing_label_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_category_labels(tmp_path / "missing.json")


def test_describe_category() -> None:
    assert describe_category("bright_daylight") == "Bright daylight"
    assert describe_category("synthetic:shadow").startswith("Synthetic")
    assert describe_category("custom_value") == "custom_value"


# --------------------------------------------------------------------------- #
# Synthetic transformations
# --------------------------------------------------------------------------- #

def test_every_registered_transform_runs(synthetic_forest) -> None:
    image, _ = synthetic_forest
    for name in TRANSFORMATION_NAMES:
        out = apply_transformation(image, name, strength=0.7)
        assert out.shape == image.shape, name
        assert out.dtype == np.uint8, name


def test_original_image_never_modified(synthetic_forest) -> None:
    image, _ = synthetic_forest
    before = image.copy()
    generate_variants(image, strength=0.8)
    assert np.array_equal(image, before), "transformations must not mutate the input"


def test_darker_is_darker_and_brighter_is_brighter(synthetic_forest) -> None:
    image, _ = synthetic_forest
    base = float(image.mean())
    darker = apply_transformation(image, "darker", strength=0.8)
    brighter = apply_transformation(image, "brighter", strength=0.8)
    assert float(darker.mean()) < base
    assert float(brighter.mean()) > base


def test_low_contrast_reduces_std(synthetic_forest) -> None:
    image, _ = synthetic_forest
    base_std = float(image.std())
    flat = apply_transformation(image, "low_contrast", strength=0.9)
    assert float(flat.std()) < base_std


def test_strength_zero_is_identity(synthetic_forest) -> None:
    image, _ = synthetic_forest
    for name in TRANSFORMATION_NAMES:
        out = apply_transformation(image, name, strength=0.0)
        diff = np.abs(out.astype(np.int16) - image.astype(np.int16))
        assert diff.max() <= 1, f"{name} changed pixels at strength=0"


def test_invalid_strength_and_unknown_name(synthetic_forest) -> None:
    image, _ = synthetic_forest
    with pytest.raises(TransformationError):
        apply_transformation(image, "darker", strength=1.5)
    with pytest.raises(TransformationError):
        apply_transformation(image, "darker", strength=-0.2)
    with pytest.raises(TransformationError):
        apply_transformation(image, "no_such_transform")


def test_synthetic_category_labeling() -> None:
    assert synthetic_category("shadow") == "synthetic:shadow"
    assert synthetic_category("original") == "original"


def test_variants_subset_and_reproducibility(synthetic_forest) -> None:
    image, _ = synthetic_forest
    subset = generate_variants(image, names=["darker", "shadow"], strength=0.5, seed=3)
    assert set(subset) == {"darker", "shadow"}
    again = generate_variants(image, names=["shadow"], strength=0.5, seed=3)
    assert np.array_equal(subset["shadow"], again["shadow"]), "seeded transform must be deterministic"
