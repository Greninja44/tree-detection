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
                                make_ade_segmenter, make_deepscene_segmenter, make_hybrid_segmenter, make_rugd_segmenter,
                                make_segformer_segmenter, make_terrain_segmenter, make_veto_segmenter,
                                make_yolo_path_segmenter, plant_detections)

MODELS = infer.PROJECT_ROOT / "models"
HYBRID = "RUGD + classical hybrid"
TERRAIN = "Terrain-aware (classical + RUGD)"
VETO = "Classical + obstacle veto"
CLASSICAL = "Classical baseline"
DEEPSCENE = "DeepScene FCN"
ADE = "ADE20K SegFormer"
TERRAIN_DEEPSCENE = "Terrain-aware (classical + DeepScene)"
TERRAIN_ADE = "Terrain-aware (classical + ADE20K)"
PATH_METHODS = (CLASSICAL, TERRAIN, TERRAIN_DEEPSCENE, TERRAIN_ADE, VETO, HYBRID, "RUGD-Segformer", DEEPSCENE, ADE, "YOLO path model", "Off")


def available_methods() -> list[str]:
    """Path methods whose weights are present (the classical baseline and Off always are)."""
    has = {"rugd": (infer.RUGD_DIR / "model.safetensors").exists(),
           "deepscene": (MODELS / "deepscene" / "fcn_resnet18.onnx").exists(),
           "ade": (MODELS / "ade_segformer_b2" / "pytorch_model.bin").exists(),
           "yolo": (MODELS / "forest_path_seg.pt").exists()}
    needs = {CLASSICAL: True, "Off": True, TERRAIN: has["rugd"], VETO: has["rugd"], HYBRID: has["rugd"],
             "RUGD-Segformer": has["rugd"], TERRAIN_DEEPSCENE: has["deepscene"], DEEPSCENE: has["deepscene"],
             TERRAIN_ADE: has["ade"], ADE: has["ade"], "YOLO path model": has["yolo"]}
    return [m for m in PATH_METHODS if needs[m]]


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
    hazard_detector: bool = False         # add COCO person/vehicle/animal boxes (models/yolov8n.pt) to the obstacle footprints
    veg_weight: float | None = None       # set -> confidence-aware fusion for terrain-aware methods (overrides the veto)
    obstacle_threshold: float = 0.5
    veto_threshold: float | None = 0.7  # soft veto for terrain-aware methods (None = argmax veto); 0.7 scored best on GT frames
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
    hazards: list = field(default_factory=list)


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

    def deepscene_path(self) -> Path:
        p = MODELS / "deepscene" / "fcn_resnet18.onnx"
        if not p.exists():
            raise ModelUnavailable(f"DeepScene model not found: {p}.", "Download FCN-ResNet18-DeepScene-576x320 from the jetson-inference model mirror (see README).")
        return p

    def ade(self):
        d = MODELS / "ade_segformer_b2"
        if not (d / "pytorch_model.bin").exists():
            raise ModelUnavailable(f"ADE20K SegFormer weights not found in {d}.",
                                   "Run: python -c \"from huggingface_hub import snapshot_download as s; s('nvidia/segformer-b2-finetuned-ade-512-512', local_dir='models/ade_segformer_b2')\"")
        if "ade" not in self._cache:
            from transformers import SegformerForSemanticSegmentation
            self._cache["ade"] = SegformerForSemanticSegmentation.from_pretrained(str(d)).eval()
        return self._cache["ade"]

    def primary(self, name: str, imgsz: int, device: str) -> Segmenter:
        if name == "RUGD":
            return make_rugd_segmenter(self.rugd(), imgsz, device)
        if name == "DeepScene":
            return make_deepscene_segmenter(str(self.deepscene_path()))
        if name == "ADE20K":
            return make_ade_segmenter(self.ade().to(device), max(imgsz, 512), device)
        raise ValueError(name)

    def segmenter(self, method: str, imgsz: int, device: str, path_weights: str, allow_grass: bool = True,
                  veto_threshold: float | None = None, veg_weight: float | None = None,
                  obstacle_threshold: float = 0.5) -> Segmenter | None:
        if method == TERRAIN:
            return make_terrain_segmenter(make_rugd_segmenter(self.rugd(), imgsz, device), allow_grass, veto_threshold, veg_weight, obstacle_threshold)
        if method == TERRAIN_DEEPSCENE:
            return make_terrain_segmenter(self.primary("DeepScene", imgsz, device), allow_grass, veto_threshold, veg_weight, obstacle_threshold)
        if method == TERRAIN_ADE:
            return make_terrain_segmenter(self.primary("ADE20K", imgsz, device), allow_grass, veto_threshold, veg_weight, obstacle_threshold)
        if method == DEEPSCENE:
            return self.primary("DeepScene", imgsz, device)
        if method == ADE:
            return self.primary("ADE20K", imgsz, device)
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
    if p.hazard_detector:
        out.hazards = infer.detect_hazards(models.yolo("yolov8n.pt"), image, 0.25, 640, device)
    out.timings["trunk_s"] = time.perf_counter() - t
    if p.path_method == "Off":
        out.warnings.append("Path method is Off: navigation is disabled.")
        out.timings["total_s"] = time.perf_counter() - t0
        return out
    t = time.perf_counter()
    try:
        seg = models.segmenter(p.path_method, p.imgsz, device, p.path_weights, p.allow_grass, p.veto_threshold,
                               p.veg_weight, p.obstacle_threshold)
    except ModelUnavailable as exc:
        if p.path_method != TERRAIN:
            raise
        seg = None  # classical fallback for the default method, clearly reported
        out.warnings.append(f"Terrain model unavailable, using the classical baseline (plants shown approximately): {exc}")
    out.trav = detect_traversable_region(image, list(out.boxes) + list(out.hazards), seg, p.trav)
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
