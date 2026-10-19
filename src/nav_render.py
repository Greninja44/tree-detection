"""Drawing for the navigation panels (detections panel and navigation panel)."""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from src.navigation import STOP, Direction, PathPlan
from src.traversability import OBSTACLE, TraversabilityResult

GREEN, RED, YELLOW, CYAN, WHITE, ORANGE = (60, 200, 60), (40, 40, 230), (0, 220, 255), (255, 200, 0), (255, 255, 255), (0, 140, 255)


def _tint(img: np.ndarray, mask: np.ndarray, color, alpha: float) -> None:
    if mask.any():
        img[mask] = ((1 - alpha) * img[mask] + alpha * np.array(color, np.float32)).astype(np.uint8)


def _text(img, text, org, scale=0.55, color=WHITE, thick=1, bg=True):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    if bg:
        cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base + 1), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


TERRAIN_COLORS = {"grass": (80, 230, 120), "bush": (200, 60, 220), "tree": (40, 140, 40), "log": (0, 140, 255),
                  "rock": (0, 140, 255), "rock-bed": (0, 170, 255), "water": (230, 140, 30)}


def draw_detections(image: np.ndarray, boxes: Sequence[Sequence[float]], trav: TraversabilityResult | None = None,
                    plants: dict | None = None) -> np.ndarray:
    """Panel 2: plants/terrain (grass, bushes, trees, logs, rocks, water - tinted and listed), trunk boxes (orange),
    other obstacles (red outline) and bush instances (magenta boxes)."""
    out = image.copy()
    legend: list[tuple[tuple, str]] = []
    if trav is not None and trav.labels is not None and trav.label_names:
        for i, name in trav.label_names.items():
            if name in TERRAIN_COLORS:
                _tint(out, (trav.labels == i) & trav.valid, TERRAIN_COLORS[name], 0.30)
        if plants:
            legend = [(TERRAIN_COLORS[n], f"{n} {v:.0f}%") for n, v in plants["summary"].items() if n in TERRAIN_COLORS][:6]
    elif trav is not None and plants:  # classical only: approximate vegetation
        veg = (trav.classes == 2) & trav.valid
        _tint(out, veg, TERRAIN_COLORS["grass"], 0.30)
        legend = [(TERRAIN_COLORS["grass"], f"{n} {v:.0f}%") for n, v in plants["summary"].items()]
    if trav is not None:
        other = (trav.classes == OBSTACLE).astype(np.uint8)
        cs, _ = cv2.findContours(other, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, [c for c in cs if cv2.contourArea(c) > 0.001 * out.shape[0] * out.shape[1]], -1, RED, 1)
    for x0, y0, x1, y1, frac in (plants or {}).get("bushes", []):
        cv2.rectangle(out, (x0, y0), (x1, y1), TERRAIN_COLORS["bush"], 2)
        _text(out, "bush", (x0, max(14, y0 - 4)), 0.45, TERRAIN_COLORS["bush"])
    for x0, y0, x1, y1, c in boxes:
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), ORANGE, 2)
        _text(out, f"trunk {c:.2f}", (int(x0), max(14, int(y0) - 4)), 0.45, ORANGE)
    n_b = len((plants or {}).get("bushes", []))
    _text(out, f"{len(boxes)} trunk(s)" + (f", {n_b} bush(es)" if n_b else ""), (6, 18), 0.55)
    y = 40
    for col, label in legend:
        cv2.rectangle(out, (8, y - 9), (22, y + 3), col, -1)
        _text(out, label, (28, y + 2), 0.4)
        y += 17
    return out


def draw_navigation(image: np.ndarray, trav: TraversabilityResult | None, plan: PathPlan | None,
                    direction: Direction | None, message: str = "") -> np.ndarray:
    """Panel 3: traversable ground (green), obstacles (red), safety margin / rejected patches (yellow),
    corridor outline, route (cyan), start/target markers, heading arrow and legend over the visible image."""
    out = image.copy()
    H, W = out.shape[:2]
    if trav is not None:
        _tint(out, trav.traversable, GREEN, 0.38)
        if plan is not None:
            _tint(out, plan.blocked & ~trav.obstacles, YELLOW, 0.30)
        _tint(out, trav.rejected, YELLOW, 0.18)
        _tint(out, trav.obstacles, RED, 0.45)
    if plan is not None:
        if plan.corridor.any():
            cs, _ = cv2.findContours(plan.corridor.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cs, -1, CYAN, 1, cv2.LINE_AA)
        pts = plan.points if plan.success else plan.candidate_points
        if len(pts) >= 2:
            poly = pts.astype(np.int32).reshape(-1, 1, 2)
            if plan.success:
                cv2.polylines(out, [poly], False, WHITE, 9, cv2.LINE_AA)
                cv2.polylines(out, [poly], False, CYAN, 5, cv2.LINE_AA)
            else:  # failed candidate: thin dashed-looking red line
                for a, b in zip(pts[::2], pts[1::2]):
                    cv2.line(out, tuple(map(int, a)), tuple(map(int, b)), RED, 2, cv2.LINE_AA)
        if plan.success and len(plan.points) and np.hypot(*(plan.points[0] - np.array(plan.start_xy))) > 12:
            cv2.line(out, tuple(map(int, plan.start_xy)), tuple(map(int, plan.points[0])), WHITE, 2, cv2.LINE_AA)
        cv2.circle(out, tuple(map(int, plan.start_xy)), 9, WHITE, -1, cv2.LINE_AA)
        cv2.circle(out, tuple(map(int, plan.start_xy)), 6, (200, 80, 0), -1, cv2.LINE_AA)
        if plan.target_xy is not None and plan.success:  # target marker
            tx, ty = map(int, plan.target_xy)
            cv2.circle(out, (tx, ty), 11, WHITE, 2, cv2.LINE_AA)
            cv2.circle(out, (tx, ty), 4, CYAN, -1, cv2.LINE_AA)
    # heading banner + arrow
    label = direction.label if direction is not None else "NAVIGATION OFF"
    stop = direction is None or direction.label == STOP
    col = RED if stop else (60, 170, 60)
    cv2.rectangle(out, (0, 0), (W, 30), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col if stop else WHITE, 2, cv2.LINE_AA)
    if direction is not None and direction.angle_deg is not None:
        ang = math.radians(direction.angle_deg)
        c = (int(W - 45), 70)
        tip = (int(c[0] + 32 * math.sin(ang)), int(c[1] - 32 * math.cos(ang)))
        cv2.circle(out, c, 38, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.arrowedLine(out, c, tip, CYAN, 4, cv2.LINE_AA, tipLength=0.35)
    msg = message or (plan.reason if plan is not None and not plan.success else "")
    if msg:
        _text(out, msg[:90], (8, H - 10), 0.45, (80, 80, 255))
    # legend
    items = [(GREEN, "traversable"), (RED, "obstacle"), (YELLOW, "margin / uncertain"), (CYAN, "route")]
    y = 44
    for col_, name in items:
        cv2.rectangle(out, (8, y - 9), (22, y + 3), col_, -1)
        _text(out, name, (28, y + 2), 0.4)
        y += 17
    return out
