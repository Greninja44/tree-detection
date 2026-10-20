"""Regression tests for the leaf-litter / confidence-aware fusion and the structured STOP reason codes."""
import cv2
import numpy as np
import pytest

from src.navigation import (INSUFFICIENT_CORRIDOR_WIDTH, LOW_CONFIDENCE_TERRAIN, NO_TRAVERSABLE_REGION, NO_VALID_CONNECTED_CORRIDOR,
                            OBSTACLE_BLOCKS_CORRIDOR, STOP_CODES, NavConfig, plan_local_path)
from src.traversability import (OBSTACLE, TRAVERSABLE, UNKNOWN, VEGETATION, ImageError, SegOutput, TravConfig,
                                detect_traversable_region, fuse_confidence_aware)

S = 12                      # tiny synthetic scene for the fusion tests
NAMES = {0: "trail", 1: "grass", 2: "vegetation", 3: "obstacle", 4: "sky"}


def _primary(p_veg=0.0, p_obs=0.0, label=2, p_ground=0.0, klass=VEGETATION):
    shape = (S, S)
    return SegOutput(np.full(shape, klass, np.uint8), np.full(shape, p_ground, np.float32), np.ones(shape, bool), "prim", [],
                     np.full(shape, label, np.uint8), NAMES, np.full(shape, p_veg + p_obs, np.float32),
                     np.full(shape, p_obs, np.float32), np.full(shape, p_veg, np.float32))


def _classical(p=0.9):
    return SegOutput(np.full((S, S), TRAVERSABLE, np.uint8), np.full((S, S), p, np.float32), np.ones((S, S), bool), "classical", [])


def test_uncertain_litter_is_not_rejected_by_a_weak_vegetation_penalty():
    out = fuse_confidence_aware(_primary(p_veg=0.6), _classical(0.9), veg_weight=0.3)
    assert (out.p_trav >= 0.5).all() and (out.classes == TRAVERSABLE).all()


def test_strong_vegetation_penalty_makes_ground_ineligible_but_not_a_hard_obstacle():
    out = fuse_confidence_aware(_primary(p_veg=0.6), _classical(0.9), veg_weight=1.0)
    assert (out.p_trav < 0.5).all()
    assert not (out.classes == OBSTACLE).any()          # no safety-margin inflation for merely unsure / vegetation pixels


def test_confirmed_obstacle_is_never_overridden_by_a_confident_classical_route():
    out = fuse_confidence_aware(_primary(p_obs=0.9, klass=OBSTACLE, label=3), _classical(0.99), veg_weight=0.0)
    assert (out.p_trav == 0).all() and (out.classes == OBSTACLE).all()


def test_dense_vegetation_stays_blocked_when_the_veto_is_softened_but_not_removed():
    out = fuse_confidence_aware(_primary(p_veg=0.95), _classical(0.99), veg_weight=0.9)
    assert (out.p_trav < 0.2).all()


def test_unknown_pixels_follow_the_documented_policy_not_blocked_by_themselves():
    out = fuse_confidence_aware(_primary(klass=UNKNOWN, label=4), _classical(0.9), veg_weight=1.0)
    assert (out.p_trav >= 0.5).all() and (out.classes == TRAVERSABLE).all()   # unknown != obstacle: classical ground survives


def test_grass_caution_changes_route_cost_only_not_eligibility():
    out = fuse_confidence_aware(_primary(label=1, klass=VEGETATION), _classical(0.74), veg_weight=1.0)
    assert (out.p_trav >= 0.5).all()                    # eligibility unchanged by the grass multiplier
    assert (out.p_cost < out.p_trav).all()              # but routes through grass cost more
    assert fuse_confidence_aware(_primary(label=1), _classical(0.9), allow_grass=False).p_trav.max() == 0


def test_fallback_without_probability_maps_and_shape_alignment():
    plain = SegOutput(np.full((S, S), TRAVERSABLE, np.uint8), np.full((S, S), 0.9, np.float32), np.ones((S, S), bool), "plain", [],
                      np.ones((S, S), np.uint8), {1: "dirt"})
    out = fuse_confidence_aware(plain, _classical(0.9))
    assert out.p_trav.shape == out.classes.shape == out.valid.shape == (S, S)
    bad = lambda im: SegOutput(np.zeros((3, 3), np.uint8), np.zeros((3, 3), np.float32), np.ones((3, 3), bool), "bad", [])  # noqa: E731
    with pytest.raises(ImageError):
        detect_traversable_region(np.zeros((64, 64, 3), np.uint8), segmenter=bad)


def test_trunk_footprint_still_cuts_a_confident_corridor():
    H, W = 120, 160
    ground = SegOutput(np.full((H, W), TRAVERSABLE, np.uint8), np.full((H, W), 0.95, np.float32), np.ones((H, W), bool), "g", [])
    free = detect_traversable_region(np.zeros((H, W, 3), np.uint8), [], lambda _i: ground, TravConfig())
    cut = detect_traversable_region(np.zeros((H, W, 3), np.uint8), [(60, 10, 100, 118, 0.9)], lambda _i: ground, TravConfig())
    assert free.traversable[112, 80] and not cut.traversable[112, 80]   # inside the trunk's ground footprint (bottom strip of the box)
    assert cut.obstacles[116, 80] and cut.traversable[40, 20]          # elsewhere the ground is untouched


# ---- planner reason codes -------------------------------------------------------------------------------------------
HH, WW = 360, 640


def _ground(x0=200, x1=440, top=0.3):
    g = np.zeros((HH, WW), bool)
    g[int(top * HH):, x0:x1] = True
    return g


def test_reason_code_no_traversable_region_for_empty_masks():
    p = plan_local_path(np.zeros((HH, WW), bool), np.zeros((HH, WW), bool))
    assert not p.success and p.reason_code == NO_TRAVERSABLE_REGION


def test_reason_code_obstacle_blocks_an_otherwise_valid_corridor():
    wall = np.zeros((HH, WW), bool)
    wall[200:240, :] = True
    p = plan_local_path(_ground(), wall)
    assert not p.success and p.reason_code == OBSTACLE_BLOCKS_CORRIDOR


def test_reason_code_insufficient_width():
    narrow = np.zeros((HH, WW), bool)
    narrow[:, 316:324] = True
    p = plan_local_path(narrow, np.zeros((HH, WW), bool), NavConfig(min_clearance=0.03))
    assert not p.success and p.reason_code == INSUFFICIENT_CORRIDOR_WIDTH


def test_reason_code_fragmented_ground_has_no_connected_corridor():
    g = _ground()
    g[250:300, :] = False                                # a band of non-ground (not an obstacle) inside the look-ahead range cuts the corridor
    p = plan_local_path(g, np.zeros((HH, WW), bool))
    assert not p.success and p.reason_code == NO_VALID_CONNECTED_CORRIDOR


def test_reason_code_low_confidence_and_code_vocabulary():
    p = plan_local_path(_ground(), np.zeros((HH, WW), bool), NavConfig(min_confidence=0.999))
    assert not p.success and p.reason_code == LOW_CONFIDENCE_TERRAIN
    ok = plan_local_path(_ground(), np.zeros((HH, WW), bool))
    assert ok.success and ok.reason_code == ""
    assert len(set(STOP_CODES)) == 5


def test_defaults_are_unchanged():
    from src.pipeline import NAVIGATION, NavParams
    p = NavParams()
    assert p.path_method == NAVIGATION and p.conf == 0.4 and p.imgsz == 480 and p.allow_grass is True
    assert NavConfig().min_confidence == 0.35 and NavConfig().safety_margin == 0.02 and TravConfig().min_confidence == 0.5


# ---- multi-source fusion -------------------------------------------------------------------------------------------
def _rugd(label, p_trav=0.0, p_obs=0.0, p_bush=0.0, name="mulch"):
    names = {0: "void", 1: name}
    return SegOutput(np.full((S, S), UNKNOWN, np.uint8), np.full((S, S), p_trav, np.float32), np.ones((S, S), bool), "rugd", [],
                     np.full((S, S), 1, np.uint8), names, None, np.full((S, S), p_obs, np.float32), np.full((S, S), p_bush, np.float32))


def _ds(p_trav=0.0, p_obs=0.0, p_veg=0.0):
    return SegOutput(np.full((S, S), UNKNOWN, np.uint8), np.full((S, S), p_trav, np.float32), np.ones((S, S), bool), "ds", [],
                     np.full((S, S), 2, np.uint8), NAMES, None, np.full((S, S), p_obs, np.float32), np.full((S, S), p_veg, np.float32))


def test_multi_source_litter_survives_deepscene_vegetation_label():
    from src.traversability import fuse_multi_source
    out = fuse_multi_source(_rugd(1, p_trav=0.9), _ds(p_veg=0.9), _classical(0.8))
    assert (out.p_trav >= 0.5).all() and (out.classes == TRAVERSABLE).all()      # DeepScene "vegetation" is not a veto here


def test_multi_source_vetoes_come_from_any_confirmed_obstacle_or_bush():
    from src.traversability import fuse_multi_source
    for rg, ds in ((_rugd(1, p_trav=0.9, p_obs=0.8), _ds()), (_rugd(1, p_trav=0.9, p_bush=0.8, name="bush"), _ds()), (_rugd(1, p_trav=0.9), _ds(p_obs=0.8))):
        out = fuse_multi_source(rg, ds, _classical(0.95))
        assert (out.p_trav == 0).all() and (out.classes == OBSTACLE).all()


def test_multi_source_ground_is_the_union_of_evidence_and_grass_is_cost_only():
    from src.traversability import fuse_multi_source
    only_ds = fuse_multi_source(_rugd(1), _ds(p_trav=0.9), _classical(0.1))
    assert (only_ds.p_trav >= 0.5).all()                                          # any one source can supply ground
    grass = fuse_multi_source(_rugd(1, p_trav=0.6, name="grass"), _ds(), _classical(0.6))
    assert (grass.p_trav >= 0.5).all() and (grass.p_cost < grass.p_trav).all()
    assert fuse_multi_source(_rugd(1, p_trav=0.9, name="grass"), _ds(), _classical(0.9), allow_grass=False).p_trav.max() == 0
