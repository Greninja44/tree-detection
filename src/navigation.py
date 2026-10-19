"""Local route planning on a traversability mask, direction estimation and the path-coverage metric.

This is a *visual* local route estimate on a 2D image.  Without depth, camera calibration and vehicle geometry it
does not guarantee a collision-free physical path, and the direction labels are steering *suggestions*, not
validated vehicle control commands.

Planner: Dijkstra over a down-sampled grid.  A cell is passable only if it is traversable, outside every obstacle
safety margin and at least ``min_clearance`` away from the nearest blocked cell (narrow gaps are rejected).
Cell cost = 1 + k_clear / (1 + clearance) + k_conf * (1 - P(traversable)), so routes prefer open ground and
confident pixels; sideways and backwards steps are penalised for smooth forward progress.  The raw route is
smoothed, and the smoothing is reduced automatically if it would cut a corner into blocked space.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

STOP = "STOP / NO SAFE PATH"
DIRECTIONS = ("TURN LEFT", "SLIGHT LEFT", "STRAIGHT", "SLIGHT RIGHT", "TURN RIGHT", STOP)


@dataclass
class NavConfig:
    start_x: float = 0.5          # start region centre (fraction of width)
    start_y: float = 0.95         # start row (fraction of height)
    start_half_width: float = 0.15  # preferred start region half width (fraction of width)
    max_start_shift: float = 0.35   # the route may start this far (fraction of width) from the centre, at a cost
    k_start: float = 30.0           # cost of shifting the start outside the preferred region (per image width)
    min_reach: float = 0.70         # accept a route that reaches at least this fraction of the look-ahead distance
    lookahead: float = 0.40       # route must reach this far up the image (fraction of height above the start)
    safety_margin: float = 0.02   # obstacle inflation (fraction of image width)
    min_clearance: float = 0.015  # minimum half-width of a passable corridor (fraction of image width)
    smoothing: float = 0.5        # 0 = raw grid route, 1 = strongest smoothing
    min_confidence: float = 0.35  # routes below this confidence are rejected
    grid_width: int = 96          # planner grid resolution
    k_clear: float = 8.0
    k_conf: float = 3.0
    k_goal: float = 6.0           # cost per grid row of look-ahead distance left unreached (endpoint scoring)
    k_lat: float = 2.0            # cost of lateral distance from the start x (prefers going where the vehicle points)
    heading_lookahead: float = 0.6  # steer toward the route point at this fraction of the look-ahead distance
    straight_deg: float = 7.0     # |heading| below this -> STRAIGHT
    slight_deg: float = 20.0      # |heading| below this -> SLIGHT, above -> TURN
    roi_top: float = 0.40         # coverage ROI: rows below this fraction of the height


@dataclass
class PathPlan:
    success: bool
    points: np.ndarray                     # (N, 2) float32 pixel (x, y) from start to look-ahead; empty on failure
    corridor: np.ndarray                   # (H, W) bool navigable corridor around the route
    blocked: np.ndarray                    # (H, W) bool obstacles + safety margin
    confidence: float
    clearance_px: float                    # smallest half-width of free space along the route (pixels)
    reason: str                            # "" on success, otherwise why no safe path was found
    start_xy: tuple[float, float]
    target_xy: tuple[float, float] | None
    continuous: bool                       # a continuous route start -> look-ahead exists
    partial: bool = False                  # route reaches only part of the look-ahead distance (ground ends early)
    reached_fraction: float = 0.0          # fraction of the look-ahead distance the route reaches
    candidate_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))


@dataclass
class Direction:
    label: str
    angle_deg: float | None  # + = right of straight ahead; None when there is no path


def path_coverage(traversable: np.ndarray, valid: np.ndarray | None = None, roi_top: float = 0.40) -> tuple[float, float]:
    """Return ``(path_coverage %, traversable_area %)``.

    ``path_coverage = traversable_pixels / valid_roi_pixels * 100`` where the ROI is every pixel in rows at or below
    ``roi_top * H`` and *valid* pixels exclude sky.  ``traversable_area`` uses the same valid mask over the whole image.
    Unknown/vegetation/obstacle pixels stay in the denominator (conservative).
    """
    if traversable.ndim != 2:
        raise ValueError("traversable must be a 2D mask")
    H, W = traversable.shape
    valid = np.ones_like(traversable, bool) if valid is None else valid.astype(bool)
    if valid.shape != traversable.shape:
        raise ValueError("valid mask shape mismatch")
    roi = np.zeros_like(traversable, bool)
    roi[int(roi_top * H):] = True
    denom_roi = int((roi & valid).sum())
    denom_all = int(valid.sum())
    cov = 100.0 * int((traversable & roi & valid).sum()) / denom_roi if denom_roi else 0.0
    area = 100.0 * int((traversable & valid).sum()) / denom_all if denom_all else 0.0
    return cov, area


def _fail(shape, reason, start_xy, blocked, candidate=None, continuous=False, target=None) -> PathPlan:
    return PathPlan(False, np.zeros((0, 2), np.float32), np.zeros(shape, bool), blocked, 0.0, 0.0, reason,
                    start_xy, target, continuous,
                    candidate_points=candidate if candidate is not None else np.zeros((0, 2), np.float32))


def _dijkstra(cost: np.ndarray, passable: np.ndarray, starts: list[tuple[int, int, float]]):
    """Full shortest-path tree from the start cells over passable cells (8-connected)."""
    h, w = passable.shape
    dist = np.full((h, w), np.inf)
    prev = np.full(h * w, -1, np.int64)
    heap: list[tuple[float, int]] = []
    for y, x, extra in starts:
        dist[y, x] = cost[y, x] + extra
        heapq.heappush(heap, (dist[y, x], y * w + x))
    nbrs = [(-1, 0, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (0, -1, 1.0), (0, 1, 1.0),
            (1, 0, 1.0), (1, -1, 1.414), (1, 1, 1.414)]
    while heap:
        d, i = heapq.heappop(heap)
        y, x = divmod(i, w)
        if d > dist[y, x]:
            continue
        for dy, dx, step in nbrs:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w) or not passable[ny, nx]:
                continue
            pen = 3.0 if dy > 0 else (1.6 if dy == 0 else 1.0)  # discourage backwards / pure sideways moves
            nd = d + step * pen * 0.5 * (cost[y, x] + cost[ny, nx])
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                prev[ny * w + nx] = y * w + x
                heapq.heappush(heap, (nd, ny * w + nx))
    return dist, prev


def _backtrace(prev: np.ndarray, w: int, y: int, x: int) -> list[tuple[int, int]]:
    idx, path = y * w + x, []
    while idx != -1:
        path.append(divmod(int(idx), w))
        idx = prev[idx]
    return path[::-1]


def _smooth(points: np.ndarray, strength: float, ok) -> np.ndarray:
    """Moving-average smoothing with pinned end points; back off if the smoothed route leaves free space."""
    n = len(points)
    if n < 5 or strength <= 0:
        return points
    for s in (strength, strength * 0.5, strength * 0.25):
        win = max(3, int(round(s * 0.35 * n)) | 1)
        pad = win // 2
        padded = np.vstack([np.repeat(points[:1], pad, 0), points, np.repeat(points[-1:], pad, 0)])
        k = np.ones(win) / win
        sm = np.stack([np.convolve(padded[:, c], k, mode="valid") for c in (0, 1)], 1)
        sm[0], sm[-1] = points[0], points[-1]
        if all(ok(p) for p in sm):
            return sm
    return points


def plan_local_path(traversable: np.ndarray, obstacles: np.ndarray, config: NavConfig | None = None,
                    p_trav: np.ndarray | None = None) -> PathPlan:
    """Plan a route from the bottom-centre start region to the look-ahead row.

    ``traversable`` / ``obstacles`` are (H, W) bool masks; ``p_trav`` is an optional (H, W) confidence map.
    Returns a :class:`PathPlan`; ``success`` is False with an explicit ``reason`` when no sufficiently safe,
    continuous, confident corridor exists.
    """
    cfg = config or NavConfig()
    if traversable.ndim != 2 or traversable.shape != obstacles.shape:
        raise ValueError("traversable and obstacles must be 2D masks of the same shape")
    H, W = traversable.shape
    start_xy = (cfg.start_x * W, cfg.start_y * H)
    gw = cfg.grid_width
    gh = max(8, int(round(gw * H / W)))
    sc = W / gw  # pixels per grid cell (x); y scale is H / gh

    def down(m):
        return cv2.resize(m.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)

    trav = down(traversable) > 0.5
    obst = down(obstacles) > 0.15
    margin = max(1, int(round(cfg.safety_margin * gw)))
    blocked = cv2.dilate(obst.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)) > 0
    blocked_full = cv2.resize(blocked.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    free = trav & ~blocked
    clear = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 3)
    min_clear = max(1.0, cfg.min_clearance * gw)
    passable = free & (clear >= min_clear)

    sy = min(gh - 1, int(round(cfg.start_y * gh)))
    cx = cfg.start_x * gw
    sx0 = max(0, int((cfg.start_x - cfg.max_start_shift) * gw))
    sx1 = min(gw, int(math.ceil((cfg.start_x + cfg.max_start_shift) * gw)))
    sy0 = max(0, sy - max(2, int(0.08 * gh)))
    hw = cfg.start_half_width * gw
    starts = [(y, x, cfg.k_start * max(0.0, abs(x + 0.5 - cx) - hw) / gw)
              for y in range(sy0, sy + 1) for x in range(sx0, sx1) if passable[y, x]]
    goal_row = max(0, int(round(sy - cfg.lookahead * gh)))
    target_xy = (cfg.start_x * W, (goal_row + 0.5) * H / gh)
    if not free[sy0:sy + 1, sx0:sx1].any():
        return _fail((H, W), "No traversable, unblocked ground near the start position.", start_xy, blocked_full, target=target_xy)
    if not starts:
        return _fail((H, W), "Ground near the start position is too narrow or too close to obstacles.", start_xy,
                     blocked_full, target=target_xy)

    prob = down(p_trav) if p_trav is not None else np.full((gh, gw), 0.7, np.float32)
    lateral = np.abs(np.arange(gw)[None, :] + 0.5 - cx) / gw
    cost = 1.0 + cfg.k_clear / (1.0 + clear) + cfg.k_conf * (1.0 - np.clip(prob, 0, 1)) + cfg.k_lat * lateral
    dist, prev = _dijkstra(cost, passable, starts)
    ys, xs = np.nonzero(np.isfinite(dist))
    need = max(1, sy - goal_row)
    progress = (sy - ys) / need
    eligible = progress >= cfg.min_reach
    to_px = lambda c: np.array([(c[1] + 0.5) * sc, (c[0] + 0.5) * H / gh], np.float32)  # noqa: E731
    if not eligible.any():
        i = int(ys.argmin())
        raw = np.array([to_px(c) for c in _backtrace(prev, gw, int(ys[i]), int(xs[i]))], np.float32)
        top = f"{100 * (1 - raw[-1, 1] / H):.0f}%"
        return _fail((H, W), f"No continuous safe corridor reaches the look-ahead point (route stops at {top} image height).",
                     start_xy, blocked_full, candidate=raw, continuous=False, target=target_xy)
    # Score every sufficiently-far endpoint: path cost + a penalty for the look-ahead distance left unreached, so a
    # slightly shorter route near the vehicle's heading beats a full-length one far off to the side.
    score = dist[ys, xs] + cfg.k_goal * np.maximum(0, ys - goal_row)
    score[~eligible] = np.inf
    i = int(score.argmin())
    cells = _backtrace(prev, gw, int(ys[i]), int(xs[i]))
    raw = np.array([to_px(c) for c in cells], np.float32)
    reached = int(ys[i]) <= goal_row
    got = min(1.0, float(progress[i]))

    def ok(p):
        gx, gy = int(np.clip(p[0] / sc, 0, gw - 1)), int(np.clip(p[1] * gh / H, 0, gh - 1))
        return passable[gy, gx]

    pts = _smooth(raw, cfg.smoothing, ok)
    cl = np.array([clear[int(np.clip(p[1] * gh / H, 0, gh - 1)), int(np.clip(p[0] / sc, 0, gw - 1))] for p in pts])
    clearance_px = float(cl.min() * sc)
    mean_p = float(np.mean([prob[int(np.clip(p[1] * gh / H, 0, gh - 1)), int(np.clip(p[0] / sc, 0, gw - 1))] for p in pts]))
    conf = mean_p * (0.5 + 0.5 * min(1.0, float(cl.min()) / (3.0 * min_clear)))
    partial = not reached
    reach = min(1.0, got)
    if partial:
        conf *= reach ** 0.5
    corridor = np.zeros((H, W), np.uint8)
    for p, c in zip(pts, cl):
        cv2.circle(corridor, (int(p[0]), int(p[1])), max(2, int(c * sc)), 1, -1)
    corridor = (corridor > 0) & ~blocked_full & traversable
    if conf < cfg.min_confidence:
        return _fail((H, W), f"Route confidence {conf:.2f} is below the minimum {cfg.min_confidence:.2f}.", start_xy,
                     blocked_full, candidate=pts, continuous=True, target=target_xy)
    return PathPlan(True, pts, corridor, blocked_full, conf, clearance_px, "", start_xy, tuple(map(float, pts[-1])),
                    True, partial, reach)


def estimate_navigation_direction(path_points: np.ndarray, image_width: int, image_height: int,
                                  config: NavConfig | None = None) -> Direction:
    """Heading label from the lower-centre start position to the route point at 60% of the look-ahead distance.

    Aiming at a fixed image row (not a fraction of route length) keeps the heading stable when the route
    sidesteps an obstacle first.  If the route ends before that row, its last point is used.  The angle is
    measured in the image plane from straight up (+ = right); perspective compresses lateral distance, so the
    thresholds are heuristics, not ground-plane bearings.
    """
    cfg = config or NavConfig()
    if path_points is None or len(path_points) < 2:
        return Direction(STOP, None)
    pts = np.asarray(path_points, np.float32)
    sx, sy = cfg.start_x * image_width, cfg.start_y * image_height
    tgt_y = sy - cfg.heading_lookahead * cfg.lookahead * image_height
    p = pts[-1]
    for a, b in zip(pts[:-1], pts[1:]):
        if a[1] >= tgt_y >= b[1] and a[1] > b[1]:
            p = a + (b - a) * ((a[1] - tgt_y) / (a[1] - b[1]))
            break
    dx, dy = float(p[0] - sx), float(sy - p[1])
    if dy <= 1e-6:
        return Direction(STOP, None)
    return _label_from_angle(math.degrees(math.atan2(dx, dy)), cfg)


def resample_path(points: np.ndarray, n: int = 30) -> np.ndarray:
    """Resample a polyline to ``n`` points evenly spaced by arc length (for frame-to-frame route smoothing)."""
    pts = np.asarray(points, np.float32)
    if len(pts) < 2:
        return pts
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if seg[-1] <= 0:
        return pts[:1].repeat(n, 0)
    t = np.linspace(0, seg[-1], n)
    return np.stack([np.interp(t, seg, pts[:, 0]), np.interp(t, seg, pts[:, 1])], 1).astype(np.float32)


class DirectionSmoother:
    """Temporal smoothing for video: EMA on the heading angle, label hysteresis, and an immediate safe stop.

    A missing path stops immediately (conservative); resuming needs ``resume_frames`` consecutive valid frames.
    """

    def __init__(self, alpha: float = 0.4, resume_frames: int = 2, config: NavConfig | None = None):
        self.alpha, self.resume_frames, self.cfg = alpha, resume_frames, config or NavConfig()
        self._angle: float | None = None
        self._ok = 0
        self._stopped = True

    def update(self, direction: Direction) -> Direction:
        if direction.angle_deg is None:
            self._ok, self._stopped, self._angle = 0, True, None
            return Direction(STOP, None)
        self._ok += 1
        self._angle = direction.angle_deg if self._angle is None else (
            self.alpha * direction.angle_deg + (1 - self.alpha) * self._angle)
        if self._stopped and self._ok < self.resume_frames:
            return Direction(STOP, None)
        self._stopped = False
        return _label_from_angle(self._angle, self.cfg)


def _label_from_angle(ang: float, cfg: NavConfig) -> Direction:
    a = abs(ang)
    if a <= cfg.straight_deg:
        return Direction("STRAIGHT", ang)
    if a <= cfg.slight_deg:
        return Direction("SLIGHT RIGHT" if ang > 0 else "SLIGHT LEFT", ang)
    return Direction("TURN RIGHT" if ang > 0 else "TURN LEFT", ang)
