"""Tests for src/traversability.py (model-free: fake segmenters and synthetic images)."""
import numpy as np
import pytest

from src.traversability import (OBSTACLE, TRAVERSABLE, UNKNOWN, ImageError, SegOutput, TravConfig, box_footprints,
                                detect_traversable_region, fuse_obstacle_veto, fuse_segmentations, fuse_terrain_aware,
                                normalize_lighting, plant_detections, segment_classical, start_region_mask)

H, W = 240, 320


def _img(v=120):
    return np.full((H, W, 3), v, np.uint8)


def _fake(mask: np.ndarray, p=0.95, obstacle: np.ndarray | None = None):
    """Segmenter that calls `mask` traversable with probability p."""
    def run(image):
        classes = np.where(mask, TRAVERSABLE, UNKNOWN).astype(np.uint8)
        if obstacle is not None:
            classes[obstacle] = OBSTACLE
        return SegOutput(classes, np.where(mask, p, 0.05).astype(np.float32), np.ones((H, W), bool), "fake")
    return run


def test_validate_rejects_bad_images():
    with pytest.raises(ImageError):
        detect_traversable_region(np.zeros((H, W), np.uint8))
    with pytest.raises(ImageError):
        detect_traversable_region(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ImageError):
        detect_traversable_region(_img().astype(np.float32))


def test_segmenter_shape_mismatch_is_reported():
    bad = lambda im: SegOutput(np.zeros((5, 5), np.uint8), np.zeros((5, 5), np.float32), np.ones((5, 5), bool), "bad")  # noqa: E731
    with pytest.raises(ImageError):
        detect_traversable_region(_img(), segmenter=bad)


def test_isolated_patch_is_rejected_connected_ground_kept():
    m = np.zeros((H, W), bool)
    m[150:, 100:220] = True        # connected to the start band
    m[30:70, 20:60] = True         # isolated patch far from the vehicle
    r = detect_traversable_region(_img(), segmenter=_fake(m))
    assert r.traversable[200, 160] and not r.traversable[50, 40]
    assert r.rejected[50, 40]
    assert r.classes[50, 40] == UNKNOWN  # rejected patches are never labelled safe


def test_low_probability_pixels_are_not_traversable():
    m = np.zeros((H, W), bool)
    m[150:, 100:220] = True
    r = detect_traversable_region(_img(), segmenter=_fake(m, p=0.3))
    assert not r.traversable.any()
    assert any("No traversable" in n for n in r.notes)


def test_trunk_boxes_become_obstacles_and_cut_ground():
    m = np.zeros((H, W), bool)
    m[120:, 40:280] = True
    r = detect_traversable_region(_img(), obstacle_boxes=[(150, 20, 175, 200, 0.9)], segmenter=_fake(m))
    assert r.obstacles[195, 160] and r.classes[195, 160] == OBSTACLE
    assert not r.traversable[195, 160]


def test_box_footprints_geometry_and_degenerate_boxes():
    f = box_footprints((H, W), [(10, 0, 30, 100, 0.9), (50, 50, 50, 80, 0.9), (-5, 200, 20, 400, 0.5)])
    assert f[95, 20] and not f[10, 20]       # only the bottom strip of a box
    assert not f[60, 50]                     # zero-width box ignored
    assert f[239, 5]                         # box clipped to the image


def test_start_region_is_bottom_centre():
    s = start_region_mask((H, W))
    assert s[H - 1, W // 2] and not s[0, W // 2] and not s[H - 1, 0]


def test_normalize_lighting_only_when_needed():
    ok = np.repeat(np.tile(np.linspace(40, 220, W, dtype=np.uint8), (H, 1))[..., None], 3, axis=2)  # well exposed, contrasty
    same, applied = normalize_lighting(ok)
    assert not applied and same is ok
    dark = (ok // 5).astype(np.uint8)
    out, applied = normalize_lighting(dark)
    assert applied and out.mean() > dark.mean()


def test_classical_baseline_finds_a_trail_and_flags_itself():
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (30, 90, 40)                       # green vegetation (BGR)
    img[100:, 120:200] = (60, 110, 150)         # brown trail from the bottom centre
    seg = segment_classical(img)
    assert seg.method == "Classical baseline"
    assert seg.classes[200, 160] == TRAVERSABLE
    assert seg.classes[200, 20] != TRAVERSABLE
    assert any("BASELINE" in n for n in seg.notes)


def _seg(classes, p):
    return SegOutput(classes.astype(np.uint8), p.astype(np.float32), np.ones(classes.shape, bool), "t")


def test_obstacle_veto_removes_ground_only_where_primary_says_obstacle():
    base = _seg(np.full((10, 10), TRAVERSABLE), np.full((10, 10), 0.9))
    cls = np.full((10, 10), UNKNOWN)
    cls[:, :5] = OBSTACLE
    out = fuse_obstacle_veto(_seg(cls, np.zeros((10, 10))), base)
    assert (out.p_trav[:, :5] == 0).all() and (out.p_trav[:, 5:] == 0.9).all()
    assert (out.classes[:, :5] == OBSTACLE).all() and (out.classes[:, 5:] == TRAVERSABLE).all()


def test_hybrid_is_vetoed_by_vegetation_but_veto_variant_is_not():
    veg = np.full((10, 10), 2)  # VEGETATION
    base = _seg(np.full((10, 10), TRAVERSABLE), np.full((10, 10), 0.9))
    prim = _seg(veg, np.zeros((10, 10)))
    assert (fuse_segmentations(prim, base).p_trav < 0.5).all()
    assert (fuse_obstacle_veto(prim, base).p_trav > 0.5).all()


def test_classical_is_the_default_path_method():
    from src.pipeline import CLASSICAL, PATH_METHODS, NavParams
    assert PATH_METHODS[0] == CLASSICAL and NavParams().path_method == CLASSICAL


def _terrain_seg():
    """4x4 scene: row 0 dirt, row 1 grass, row 2 bush, row 3 tree (RUGD-style labels)."""
    names = {0: "void", 1: "dirt", 2: "grass", 3: "bush", 4: "tree"}
    labels = np.repeat(np.array([1, 2, 3, 4], np.uint8)[:, None], 4, axis=1)
    classes = np.array([[TRAVERSABLE] * 4, [2] * 4, [2] * 4, [OBSTACLE] * 4], np.uint8)
    p = np.array([[0.95] * 4, [0.05] * 4, [0.05] * 4, [0.0] * 4], np.float32)
    return SegOutput(classes, p, np.ones((4, 4), bool), "t", [], labels, names)


def test_terrain_aware_blocks_bush_and_tree_but_allows_grass_at_a_cost():
    base = _seg(np.full((4, 4), TRAVERSABLE), np.full((4, 4), 0.9))
    out = fuse_terrain_aware(_terrain_seg(), base)
    assert (out.p_trav[0] > 0.8).all()                       # dirt: confident ground
    assert (out.p_trav[1] > 0.5).all() and (out.p_trav[1] < out.p_trav[0]).all()   # grass: passable, costlier
    assert (out.p_trav[2] == 0).all() and (out.p_trav[3] == 0).all()                # bush, tree: blocked
    assert (out.classes[3] == OBSTACLE).all() and (out.classes[2] == 2).all()       # bush is vegetation, not ground
    assert (fuse_terrain_aware(_terrain_seg(), base, allow_grass=False).p_trav[1] == 0).all()


def test_terrain_aware_without_labels_falls_back_to_obstacle_veto():
    base = _seg(np.full((4, 4), TRAVERSABLE), np.full((4, 4), 0.9))
    assert fuse_terrain_aware(_seg(np.full((4, 4), UNKNOWN), np.zeros((4, 4))), base).method == "Classical + obstacle veto"


def test_plant_detections_reports_labels_and_bush_instances():
    from src.traversability import TraversabilityResult
    H, W = 100, 100
    names = {0: "void", 1: "dirt", 2: "grass", 3: "bush"}
    labels = np.ones((H, W), np.uint8)
    labels[:30] = 2
    labels[60:90, 10:50] = 3
    z = np.zeros((H, W), bool)
    r = TraversabilityResult(np.zeros((H, W), np.uint8), np.zeros((H, W), np.float32), z, z, np.ones((H, W), bool), z, "t", False, [], labels, names)
    d = plant_detections(r)
    assert d["source"] == "RUGD-Segformer" and d["summary"]["grass"] == pytest.approx(30.0)
    assert len(d["bushes"]) == 1 and d["bushes"][0][:4] == (10, 60, 50, 90)
    r.labels = None   # no semantic model: only an approximate vegetation share
    assert plant_detections(r)["source"].startswith("approximate")
