"""RGB-only free-space estimate: monocular depth (Depth Anything V2) + a fitted ground plane.

The depth network gives *relative* inverse depth (larger = closer, unknown scale and shift per frame).  For any flat or
gently sloped surface seen by a pinhole camera, inverse depth is an affine function of the pixel position, so the ground is a
plane ``d = a*u + b*v + c`` in (u, v, d) space.  We fit that plane robustly (RANSAC) to the area in front of the camera
and classify every pixel by how far its inverse depth is from the plane:

* on the plane             -> traversable   (office floor, farm track, grass, forest path: the method does not care)
* clearly closer than it   -> obstacle      (person, furniture, crop row, trunk, wall)
* clearly farther than it  -> unknown       (drop-off, hole, kerb edge: never promoted to traversable)

Assumptions (reported in ``notes``): the strip right in front of the camera is mostly ground; the camera has no extreme
pitch; there is one dominant ground plane.  Without calibration this is a visual route, not metric distances.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.traversability import OBSTACLE, TRAVERSABLE, UNKNOWN, SegOutput, Segmenter, validate_image

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def fit_ground_plane(depth: np.ndarray, seed: np.ndarray, rng: np.random.Generator, iters: int = 80,
                     prev: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """Robust plane ``d = [u, v, 1] @ coef`` over the seed pixels.  Returns ``(coef, inlier_fraction, tau)``.

    ``u`` and ``v`` are normalised to [-1, 1].  ``tau`` is the inlier threshold in depth units.
    """
    H, W = depth.shape
    ys, xs = np.nonzero(seed)
    if len(ys) < 50:
        raise ValueError("seed region too small")
    take = rng.choice(len(ys), size=min(len(ys), 4000), replace=False)
    ys, xs = ys[take], xs[take]
    A = np.stack([xs / W * 2 - 1, ys / H * 2 - 1, np.ones(len(xs))], 1)
    d = depth[ys, xs].astype(np.float64)
    tau = max(0.02 * float(np.ptp(depth)), 1e-4)
    best, best_n = None, -1
    cands = [prev] if prev is not None else []
    for _ in range(iters):
        idx = rng.choice(len(d), 3, replace=False)
        try:
            cands.append(np.linalg.solve(A[idx], d[idx]))
        except np.linalg.LinAlgError:
            continue
    for c in cands:
        n = int((np.abs(A @ c - d) < tau).sum())
        if n > best_n:
            best, best_n = c, n
    inl = np.abs(A @ best - d) < tau
    if inl.sum() >= 3:
        best = np.linalg.lstsq(A[inl], d[inl], rcond=None)[0]
        inl = np.abs(A @ best - d) < tau
    return best, float(inl.mean()), tau


def classify(depth: np.ndarray, coef: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(classes, p_trav, valid, residual)`` for an inverse-depth map and a ground plane."""
    H, W = depth.shape
    u = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    v = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    ground = coef[0] * u + coef[1] * v + coef[2]
    r = depth - ground                      # > 0: closer than the ground plane predicts
    classes = np.full((H, W), UNKNOWN, np.uint8)
    classes[np.abs(r) <= 1.5 * tau] = TRAVERSABLE
    classes[r > 3.0 * tau] = OBSTACLE
    p_trav = np.exp(-0.5 * (np.clip(r, 0, None) / (1.5 * tau)) ** 2).astype(np.float32)
    p_trav[r < -3.0 * tau] = 0.0            # below the ground plane: never route into a drop
    # Rows at/above the horizon (predicted ground disparity ~ 0) say nothing about the floor.
    valid = ground > 0.04 * float(np.ptp(depth) + 1e-6) + float(depth.min())
    return classes, p_trav, valid, r


class DepthModel:
    """Depth Anything V2 wrapper returning an (H, W) float32 relative inverse-depth map for a BGR image."""

    def __init__(self, path: str = MODEL_ID, device: str = "cpu", width: int = 364):
        import torch
        from transformers import AutoModelForDepthEstimation
        self.torch, self.device, self.width = torch, device, width
        self.net = AutoModelForDepthEstimation.from_pretrained(path).to(device).eval()
        if device.startswith("cuda"):
            self.net = self.net.half()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        torch = self.torch
        H, W = image.shape[:2]
        w = self.width // 14 * 14
        h = max(14, round(H * w / W / 14) * 14)
        x = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        t = torch.from_numpy(((x - self.mean) / self.std).transpose(2, 0, 1)[None]).to(self.device)
        if self.device.startswith("cuda"):
            t = t.half()
        with torch.no_grad():
            out = self.net(pixel_values=t).predicted_depth[0].float().cpu().numpy()
        return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)


def make_depth_segmenter(model: DepthModel, seed_rows: tuple[float, float] = (0.62, 0.98), seed_cols: tuple[float, float] = (0.2, 0.8),
                         min_inliers: float = 0.35, ema: float = 0.6, seed: int = 0) -> Segmenter:
    """Segmenter for the existing pipeline.  Keeps the last good ground plane and blends it over time (EMA) so a bad frame
    (obstacle filling the view, lens covered) does not teleport the plane; a frame whose fit is poor reuses the last plane
    and says so in ``notes``."""
    rng = np.random.default_rng(seed)
    state: dict = {"coef": None}

    def run(image: np.ndarray) -> SegOutput:
        validate_image(image)
        H, W = image.shape[:2]
        depth = model(image)
        lo, hi = np.percentile(depth, [1, 99])
        depth = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)   # frame-local scale, [0, 1]
        seed_mask = np.zeros((H, W), bool)
        seed_mask[int(seed_rows[0] * H):int(seed_rows[1] * H), int(seed_cols[0] * W):int(seed_cols[1] * W)] = True
        notes = ["RGB-only: monocular depth (Depth Anything V2 Small) + ground-plane fit. Depth is relative (no metric scale); "
                 "assumes the area in front of the camera is mostly ground and the camera is not extremely pitched."]
        try:
            coef, inl, tau = fit_ground_plane(depth, seed_mask, rng, prev=state["coef"])
            ok = inl >= min_inliers
        except ValueError:
            coef, inl, tau, ok = state["coef"], 0.0, 0.05, False
        if ok:
            state["coef"] = coef if state["coef"] is None else ema * coef + (1 - ema) * state["coef"]
            coef = state["coef"]
        elif state["coef"] is not None:
            coef = state["coef"]
            notes.append(f"Ground-plane fit was poor (inlier fraction {inl:.2f}); reusing the previous plane.")
        else:
            notes.append(f"Ground-plane fit was poor (inlier fraction {inl:.2f}) and there is no earlier plane: no route.")
            return SegOutput(np.full((H, W), UNKNOWN, np.uint8), np.zeros((H, W), np.float32), np.ones((H, W), bool),
                             "Depth ground plane", notes)
        classes, p_trav, valid, r = classify(depth, coef, tau)
        p_obs = np.clip((r - 2 * tau) / (3 * tau), 0, 1).astype(np.float32)
        return SegOutput(classes, p_trav, valid, "Depth ground plane", notes, None, None, p_obs, p_obs, None)

    return run
