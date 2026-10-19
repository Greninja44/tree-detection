"""Tests for src/navigation.py: planner, continuity, direction estimation, coverage and smoothing."""
import cv2
import numpy as np
import pytest

from src.navigation import (STOP, Direction, DirectionSmoother, NavConfig, estimate_navigation_direction, path_coverage,
                            plan_local_path)

H, W = 360, 640
NONE = np.zeros((H, W), bool)


def _corridor(centre, half=40):
    m = np.zeros((H, W), np.uint8)
    for y in range(int(0.3 * H), H):
        t = (y - 0.3 * H) / (0.7 * H)
        cx, hw = centre(t), half * (0.3 + 0.7 * t)
        cv2.line(m, (int(cx - hw), y), (int(cx + hw), y), 1, 1)
    return m.astype(bool)


def test_straight_corridor_gives_straight_route():
    p = plan_local_path(_corridor(lambda t: 320), NONE)
    d = estimate_navigation_direction(p.points, W, H)
    assert p.success and p.continuous and p.reason == ""
    assert d.label == "STRAIGHT"
    assert p.points[0, 1] > p.points[-1, 1]          # progresses up the image
    assert p.corridor.any() and p.confidence > 0.35


@pytest.mark.parametrize("sign,label", [(-1, "LEFT"), (+1, "RIGHT")])
def test_bending_corridor_turns_the_right_way(sign, label):
    p = plan_local_path(_corridor(lambda t: 320 + sign * (1 - t) ** 1.5 * 220), NONE)
    d = estimate_navigation_direction(p.points, W, H)
    assert p.success and label in d.label
    assert (d.angle_deg < 0) == (sign < 0)


def test_route_goes_around_an_obstacle_and_avoids_blocked_cells():
    ground = np.zeros((H, W), bool)
    ground[int(.3 * H):, 200:440] = True
    obst = np.zeros((H, W), bool)
    obst[200:260, 300:340] = True
    p = plan_local_path(ground, obst)
    xs, ys = p.points[:, 0].astype(int).clip(0, W - 1), p.points[:, 1].astype(int).clip(0, H - 1)
    assert p.success and not p.blocked[ys, xs].any()
    assert not (300 <= xs[(ys > 195) & (ys < 265)]).all() or not (xs[(ys > 195) & (ys < 265)] <= 340).all()


def test_fully_blocked_corridor_returns_no_safe_path_with_reason():
    ground = np.zeros((H, W), bool)
    ground[int(.3 * H):, 200:440] = True
    wall = np.zeros((H, W), bool)
    wall[200:240, :] = True
    p = plan_local_path(ground, wall)
    assert not p.success and not p.continuous and len(p.points) == 0
    assert "look-ahead" in p.reason
    assert estimate_navigation_direction(p.points, W, H).label == STOP


def test_no_ground_and_too_short_ground_are_rejected():
    assert not plan_local_path(NONE, NONE).success
    short = np.zeros((H, W), bool)
    short[int(.85 * H):, 250:390] = True
    p = plan_local_path(short, NONE)
    assert not p.success and p.reason


def test_corridor_narrower_than_min_clearance_is_rejected():
    narrow = np.zeros((H, W), bool)
    narrow[:, 318:322] = True
    assert not plan_local_path(narrow, NONE).success


def test_partial_route_is_flagged_and_less_confident():
    full = plan_local_path(_corridor(lambda t: 320), NONE)
    early = np.zeros((H, W), bool)
    early[int(.62 * H):, 250:390] = True
    p = plan_local_path(early, NONE)
    assert p.success and p.partial and 0.7 <= p.reached_fraction < 1.0
    assert p.confidence < full.confidence


def test_low_confidence_route_is_rejected():
    p = plan_local_path(_corridor(lambda t: 320), NONE, NavConfig(min_confidence=0.99))
    assert not p.success and "confidence" in p.reason and len(p.candidate_points) > 0


def test_blocked_start_shifts_sideways_instead_of_failing():
    ground = np.zeros((H, W), bool)
    ground[int(.3 * H):, 200:440] = True
    obst = np.zeros((H, W), bool)
    obst[250:350, 290:350] = True
    p = plan_local_path(ground, obst)
    assert p.success and abs(p.points[0, 0] - 320) > 20


def test_plan_validates_inputs():
    with pytest.raises(ValueError):
        plan_local_path(NONE, np.zeros((10, 10), bool))
    with pytest.raises(ValueError):
        plan_local_path(np.zeros((H, W, 3), bool), NONE)


@pytest.mark.parametrize("x_end,label", [(320, "STRAIGHT"), (350, "SLIGHT RIGHT"), (290, "SLIGHT LEFT"),
                                         (520, "TURN RIGHT"), (120, "TURN LEFT")])
def test_direction_thresholds(x_end, label):
    pts = np.array([[320, 342], [(320 + x_end) / 2, 250], [x_end, 160]], np.float32)
    assert estimate_navigation_direction(pts, W, H).label == label


def test_direction_without_path_is_stop():
    assert estimate_navigation_direction(np.zeros((0, 2)), W, H) == Direction(STOP, None)
    assert estimate_navigation_direction(None, W, H).label == STOP


def test_path_coverage_definition_and_validation():
    t = np.zeros((100, 100), bool)
    t[60:, :50] = True                       # 40 rows x 50 cols traversable, all inside the ROI (rows >= 40)
    cov, area = path_coverage(t, None, roi_top=0.4)
    assert cov == pytest.approx(100 * 2000 / 6000)
    assert area == pytest.approx(20.0)
    valid = np.ones_like(t)
    valid[:40] = False                       # sky excluded from the denominator
    cov2, area2 = path_coverage(t, valid, roi_top=0.4)
    assert cov2 == pytest.approx(cov) and area2 == pytest.approx(100 * 2000 / 6000)
    assert path_coverage(np.zeros((10, 10), bool), np.zeros((10, 10), bool)) == (0.0, 0.0)
    with pytest.raises(ValueError):
        path_coverage(np.zeros((4, 4, 3), bool))


def test_smoother_stops_immediately_resumes_after_consecutive_frames_and_damps_jitter():
    s = DirectionSmoother(alpha=0.5, resume_frames=2)
    assert s.update(Direction("STRAIGHT", 0.0)).label == STOP        # not yet trusted
    assert s.update(Direction("STRAIGHT", 0.0)).label == "STRAIGHT"
    assert s.update(Direction("SLIGHT RIGHT", 16.0)).label == "SLIGHT RIGHT"  # EMA -> 8 deg
    assert s.update(Direction(STOP, None)).label == STOP             # immediate stop
    assert s.update(Direction("TURN LEFT", -40.0)).label == STOP     # needs resume_frames
    assert s.update(Direction("TURN LEFT", -40.0)).label == "TURN LEFT"
