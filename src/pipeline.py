"""End-to-end local navigation pipeline: trunk detection + traversability + route planning + direction.

Two modes only:

* ``NAVIGATION`` (default): multi-source ground estimate - classical + RUGD-Segformer + DeepScene evidence, with a hard veto from
  RUGD obstacles/bushes and DeepScene obstacles (see ``traversability.fuse_multi_source``).  ~5 s/frame on the Jetson CPU.
* ``FAST``: the classical colour/texture estimate alone (~0.4 s/frame).  No semantic model, so no obstacle veto: it can route
  through people, machines and dense vegetation.  Intended as a preview only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src import infer
from src.navigation import (Direction, DirectionSmoother, NavConfig, PathPlan, estimate_navigation_direction, path_coverage,
                            plan_local_path)
from src.traversability import (Segmenter, TravConfig, TraversabilityResult, detect_traversable_region, fuse_multi_source,
                                make_deepscene_segmenter, make_rugd_segmenter, plant_detections, segment_classical)

MODELS = infer.PROJECT_ROOT / "models"
NAVIGATION = "Navigation"
FAST = "Fast preview"
DEPTH = "RGB depth"
PATH_METHODS = (NAVIGATION, FAST, DEPTH)


def available_methods() -> list[str]:
    """Navigation needs the RUGD and DeepScene weights; the fast preview needs nothing."""
    ok = (infer.RUGD_DIR / "model.safetensors").exists() and (MODELS / "deepscene" / "fcn_resnet18.onnx").exists()
    methods = [NAVIGATION, FAST] if ok else [FAST]
    return methods + [DEPTH] if (MODELS / "depth_anything_v2_small" / "model.safetensors").exists() else methods


class ModelUnavailable(RuntimeError):
    """A required model/weights file is missing or cannot be loaded; ``remedy`` says how to fix it."""

    def __init__(self, message: str, remedy: str):
        super().__init__(f"{message} {remedy}")
        self.remedy = remedy


@dataclass
class NavParams:
    conf: float = 0.4
    imgsz: int = 480
    path_method: str = NAVIGATION
    allow_grass: bool = True
    trunk_weights: str = "tree_trunk.pt"
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
    method_used: str = NAVIGATION


def resolve_device(pref: str = "auto") -> tuple[str, str | None]:
    """Return ``(device, warning)``; GPU is used only if CUDA actually works on this machine."""
    if pref == "cpu":
        return "cpu", None
    if infer.cuda_functional():
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
                                   "s('JasonTStanley/RUGD-Segformer', local_dir='models/rugd_segformer')\" or use the Fast preview.")
        if "rugd" not in self._cache:
            try:
                self._cache["rugd"] = infer.load_rugd()
            except Exception as exc:  # transformers missing, corrupt file ...
                raise ModelUnavailable(f"Could not load RUGD-Segformer ({exc}).", "pip install transformers safetensors")
        return self._cache["rugd"]

    def deepscene_path(self) -> Path:
        p = MODELS / "deepscene" / "fcn_resnet18.onnx"
        if not p.exists():
            raise ModelUnavailable(f"DeepScene model not found: {p}.", "See README (Model setup) for the download command, or use the Fast preview.")
        return p

    def depth(self, device: str):
        p = MODELS / "depth_anything_v2_small"
        if not (p / "model.safetensors").exists():
            raise ModelUnavailable(f"Depth model not found in {p}.",
                                   "Run: python -c \"from huggingface_hub import snapshot_download as s; "
                                   "s('depth-anything/Depth-Anything-V2-Small-hf', local_dir='models/depth_anything_v2_small')\"")
        if ("depth", device) not in self._cache:
            from src.depth_ground import DepthModel
            self._cache[("depth", device)] = DepthModel(str(p), device)
        return self._cache[("depth", device)]

    def segmenter(self, method: str, imgsz: int, device: str, allow_grass: bool = True) -> Segmenter | None:
        if method == NAVIGATION:
            rg, ds = make_rugd_segmenter(self.rugd(), imgsz, device), make_deepscene_segmenter(str(self.deepscene_path()))
            return lambda image: fuse_multi_source(rg(image), ds(image), segment_classical(image), allow_grass)
        if method == DEPTH:
            from src.depth_ground import make_depth_segmenter
            key = ("depth_seg", device)   # keep one segmenter per device so the ground plane is smoothed over time
            if key not in self._cache:
                self._cache[key] = make_depth_segmenter(self.depth(device))
            return self._cache[key]
        return None  # fast preview: classical estimate


def run_navigation(image: np.ndarray, models: Models, params: NavParams | None = None,
                   smoother: DirectionSmoother | None = None) -> NavOutput:
    """Detect trunks, estimate traversable ground, plan a local route and a direction for one BGR image."""
    p = params or NavParams()
    out = NavOutput(method_used=p.path_method)
    t0 = time.perf_counter()
    device, warn = resolve_device(p.device)
    if warn:
        out.warnings.append(warn)
    t = time.perf_counter()
    out.boxes = infer.detect_trunks(models.yolo(p.trunk_weights), image, p.conf, p.imgsz, device)
    out.timings["trunk_s"] = time.perf_counter() - t
    t = time.perf_counter()
    try:
        seg = models.segmenter(p.path_method, p.imgsz, device, p.allow_grass)
    except ModelUnavailable as exc:
        seg, out.method_used = None, FAST   # fall back, clearly reported
        out.warnings.append(f"Navigation models unavailable, using the Fast preview (no obstacle veto): {exc}")
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
        "stop_code": out.plan.reason_code,
    }
    out.warnings.extend(out.trav.notes)
    out.timings["total_s"] = time.perf_counter() - t0
    return out
