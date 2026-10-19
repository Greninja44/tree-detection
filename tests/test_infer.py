"""Tests for the model-free helpers in src/infer.py (path geometry and drawing)."""
import numpy as np

from src import infer


def _mask(h=100, w=200, x0=0, x1=0):
    m = np.zeros((h, w), bool)
    m[75:, x0:x1] = True
    return m


def test_path_result_centered_has_zero_offset():
    r = infer._path_result(_mask(x0=90, x1=110))
    assert abs(r.offset) < 0.02
    assert r.coverage > 0


def test_path_result_left_and_right_sign():
    assert infer._path_result(_mask(x0=0, x1=40)).offset < -0.5
    assert infer._path_result(_mask(x0=160, x1=200)).offset > 0.5


def test_path_result_empty_is_nan():
    r = infer._path_result(np.zeros((100, 200), bool))
    assert np.isnan(r.offset) and r.coverage == 0.0


def test_draw_path_does_not_mutate_input_and_tints_mask():
    img = np.full((100, 200, 3), 100, np.uint8)
    before = img.copy()
    out = infer.draw_path(img, infer._path_result(_mask(x0=0, x1=200)))
    assert np.array_equal(img, before)
    assert out[90, 10, 1] > out[10, 10, 1]  # green channel raised inside the mask


def test_draw_path_none_and_draw_trunks_empty_return_same_pixels():
    img = np.full((50, 50, 3), 7, np.uint8)
    assert np.array_equal(infer.draw_path(img, None), img)
    assert np.array_equal(infer.draw_trunks(img, []), img)


def test_draw_trunks_draws_box():
    img = np.zeros((100, 100, 3), np.uint8)
    out = infer.draw_trunks(img, [(10, 10, 50, 90, 0.9)])
    assert out.any() and not img.any()
