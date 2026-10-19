"""End-to-end local navigation pipeline: trunk detection + traversability + route planning + direction."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src import infer
from src.navigation import (STOP, Direction, DirectionSmoother, NavConfig, PathPlan, estimate_navigation_direction,
                            path_coverage, plan_local_path)
from src.traversability import (Segmenter, TravConfig, TraversabilityResult, detect_traversable_region,
                                make_hybrid_segmenter, make_rugd_segmenter, make_terrain_segmenter, make_veto_segmenter,
                                make_yolo_path_segmenter, plant_detections)

MODELS = infer.PROJECT_ROOT / "models"
HYBRID = "RUGD + classical hybrid"
TERRAIN = "Terrain-aware (classical + RUGD)"
VETO = "Classical + obstacle veto"
CLASSICAL = "Classical baseline"
PATH_METHODS = (CLASSICAL, TERRAIN, VETO, HYBRID, "RUGD-Segformer", "YOLO path model", "Off")


class ModelUnavailable(RuntimeError):
    """A required model/weights file is missing or cannot be loaded; ``remedy`` says how to fix it."""

    def __init__(self, message: str, remedy: str):
        super().__init__(f"{message} {remedy}")
        self.remedy = remedy


@dataclass
class NavParams:
    conf: float = 0.4
    imgsz: int = 480
    path_method: str = CLASSICAL
    allow_grass: bool = True
    trunk_weights: str = "tree_trunk.pt"
    path_weights: str = "forest_path_seg.pt"
    device: str = "auto"
    nav: NavConfig = field(default_factory=NavConfig)
    trav: TravConfig = field(default_factory=TravConfig)


@dataclass
class NavOutput:
    boxes: list = field(default_factory=list)
    trav: TraversabilityResult | None = None
    plan: PathPlan | None = None
    direction: Direction | None = None
    metrics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    plants: dict = field(default_factory=dict)


def resolve_device(pref: str = "auto") -> tuple[str, str | None]:
    """Return ``(device, warning)``; GPU is used only if CUDA actually works on this machine."""
    if pref == "cpu":
        return "cpu", None
    gpu = infer.cuda_functional()
    if gpu:
        return "cuda:0", None
    return "cpu", ("CUDA is not usable with this PyTorch build; running on CPU." if pref == "cuda" else None)


class Models:
    """Lazy, cached model loader (one load per process)."""

    def __init__(self) -> None:
        self._cache: dict = {}

    def yolo(self, name: str):
        p = MODELS / name
        if not p.exists():
            raise ModelUnavailable(f"Weights not found: {p}.", "Copy the trained .pt file into models/ (see README: Model setup).")
        if ("yolo", name) not in self._cache:
            self._cache[("yolo", name)] = infer.load_yolo(str(p))
        return self._cache[("yolo", name)]

    def rugd(self):
        if not (infer.RUGD_DIR / "model.safetensors").exists():
            raise ModelUnavailable(f"RUGD-Segformer weights not found in {infer.RUGD_DIR}.",
                                   "Run: python -c \"from huggingface_hub import snapshot_download as s; "
                                   "s('JasonTStanley/RUGD-Segformer', local_dir='models/rugd_segformer')\" "
                                   "or choose the 'Classical baseline' path method.")
        if "rugd" not in self._cache:
            try:
                self._cache["rugd"] = infer.load_rugd()
            except Exception as exc:  # transformers missing, corrupt file ...
                raise ModelUnavailable(f"Could not load RUGD-Segformer ({exc}).", "pip install transformers safetensors")
        return self._cache["rugd"]

    def segmenter(self, method: str, imgsz: int, device: str, path_weights: str, allow_grass: bool = True) -> Segmenter | None:
        if method == TERRAIN:
            return make_terrain_segmenter(make_rugd_segmenter(self.rugd(), imgsz, device), allow_grass)
        if method == "RUGD-Segformer":
            return make_rugd_segmenter(self.rugd(), imgsz, device)
        if method == VETO:
            return make_veto_segmenter(make_rugd_segmenter(self.rugd(), imgsz, device))
        if method == HYBRID:
            return make_hybrid_segmenter(make_rugd_segmenter(self.rugd(), imgsz, device))
        if method == "YOLO path model":
            return make_yolo_path_segmenter(self.yolo(path_weights), imgsz, device)
        return None  # classical baseline


def run_navigation(image: np.ndarray, models: Models, params: NavParams | None = None,
                   smoother: DirectionSmoother | None = None) -> NavOutput:
    """Detect trunks, estimate traversable ground, plan a local route and a direction for one BGR image."""
    p = params or NavParams()
    out = NavOutput()
    t0 = time.perf_counter()
    device, warn = resolve_device(p.device)
    if warn:
        out.warnings.append(warn)
    t = time.perf_counter()
    out.boxes = infer.detect_trunks(models.yolo(p.trunk_weights), image, p.conf, p.imgsz, device)
    out.timings["trunk_s"] = time.perf_counter() - t
    if p.path_method == "Off":
        out.warnings.append("Path method is Off: navigation is disabled.")
        out.timings["total_s"] = time.perf_counter() - t0
        return out
    t = time.perf_counter()
    try:
        seg = models.segmenter(p.path_method, p.imgsz, device, p.path_weights, p.allow_grass)
    except ModelUnavailable as exc:
        if p.path_method != TERRAIN:
            raise
        seg = None  # classical fallback for the default method, clearly reported
        out.warnings.append(f"Terrain model unavailable, using the classical baseline (plants shown approximately): {exc}")
    out.trav = detect_traversable_region(image, out.boxes, seg, p.trav)
    out.timings["segment_s"] = time.perf_counter() - t
    out.plants = plant_detections(out.trav)
    t = time.perf_counter()
    out.plan = plan_local_path(out.trav.traversable, out.trav.obstacles, p.nav, out.trav.p_trav)
    out.timings["plan_s"] = time.perf_counter() - t
    H, W = image.shape[:2]
    out.direction = estimate_navigation_direction(out.plan.points if out.plan.success else None, W, H, p.nav)
    if smoother is not None:
        out.direction = smoother.update(out.direction)
    cov, area = path_coverage(out.trav.traversable, out.trav.valid, p.nav.roi_top)
    out.metrics = {
        "path_coverage_pct": cov,
        "traversable_area_pct": area,
        "path_confidence": out.plan.confidence,
        "continuous_route": out.plan.continuous and out.plan.success,
        "clearance_px": out.plan.clearance_px if out.plan.success else None,
        "stop_reason": out.plan.reason,
    }
    out.warnings.extend(out.trav.notes)
    out.timings["total_s"] = time.perf_counter() - t0
    return out
