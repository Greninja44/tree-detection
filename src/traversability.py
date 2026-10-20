"""Traversability estimation: which parts of a forest image look like ground a vehicle could travel on.

Terrain classes (``classes`` map):

* ``TRAVERSABLE`` - bare/clear ground (dirt, sand, gravel, mulch, ...) connected to the vehicle's start region
* ``VEGETATION``  - grass, bushes, dense foliage (treated as *not* safe: may be impassable or hide hazards)
* ``OBSTACLE``    - tree trunks, logs, rocks, poles, water, vehicles ... plus trunk footprints from the detector
* ``UNKNOWN``     - low-confidence, sky, isolated ground patches that do not connect to the route start

Two interchangeable segmenters feed :func:`detect_traversable_region`:

* a pretrained semantic model (RUGD-Segformer, see :func:`make_rugd_segmenter`), or
* :func:`segment_classical`, a clearly-labelled *baseline* (colour similarity to the ground in front of the
  vehicle + low texture + position prior).  It has no semantic understanding and fails on cluttered or
  unusual ground; use it only when no model weights are available.

Nothing here claims physical collision-freeness: it is a 2D visual estimate without depth or calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np

UNKNOWN, TRAVERSABLE, VEGETATION, OBSTACLE = 0, 1, 2, 3
CLASS_NAMES = {UNKNOWN: "unknown", TRAVERSABLE: "traversable", VEGETATION: "vegetation", OBSTACLE: "obstacle"}

# RUGD label -> navigation class.  Anything not listed (void, sky, rock-bed, bridge) stays UNKNOWN.
RUGD_TRAVERSABLE = {"dirt", "sand", "gravel", "mulch", "asphalt", "concrete"}
RUGD_VEGETATION = {"grass", "bush"}
RUGD_OBSTACLE = {"tree", "pole", "water", "vehicle", "container/generic-object", "building", "log", "bicycle",
                 "person", "fence", "sign", "rock", "picnic-table"}
RUGD_SKY = {"sky"}

# (x_min, x_max, y_min) as fractions: the central bottom band the vehicle starts in (route retention).
START_BAND = (0.20, 0.80, 0.88)


class ImageError(ValueError):
    """Raised for images/masks with an unusable shape or dtype."""


@dataclass
class TravConfig:
    min_confidence: float = 0.5      # P(traversable) needed to call a pixel traversable
    unknown_confidence: float = 0.4  # below this top-class probability a pixel is UNKNOWN
    normalize_lighting: bool = True  # CLAHE on dark images only
    foot_fraction: float = 0.12      # bottom fraction of a trunk box treated as its ground footprint
    min_foot_px: int = 10
    min_region_fraction: float = 0.002  # drop traversable components smaller than this fraction of the image


@dataclass
class SegOutput:
    """Raw output of a segmenter, before obstacle fusion and connectivity filtering."""
    classes: np.ndarray        # (H, W) uint8 in {UNKNOWN, TRAVERSABLE, VEGETATION, OBSTACLE}
    p_trav: np.ndarray         # (H, W) float32 in [0, 1]: probability the pixel is traversable
    valid: np.ndarray          # (H, W) bool: False for sky / out-of-scene pixels (excluded from coverage)
    method: str
    notes: list[str] = field(default_factory=list)
    labels: np.ndarray | None = None          # (H, W) uint8 fine terrain id (e.g. RUGD: grass, bush, dirt ...) or None
    label_names: dict[int, str] | None = None
    p_block: np.ndarray | None = None         # (H, W) float32 P(obstacle or dense vegetation) for soft vetoes, or None
    p_obstacle: np.ndarray | None = None      # (H, W) float32 P(confirmed obstacle class) or None
    p_vegetation: np.ndarray | None = None    # (H, W) float32 P(dense-vegetation class) or None
    p_cost: np.ndarray | None = None          # (H, W) float32 route-cost confidence (decision probability x caution); planner only


@dataclass
class TraversabilityResult:
    classes: np.ndarray        # (H, W) uint8 final class map
    p_trav: np.ndarray         # (H, W) float32
    traversable: np.ndarray    # (H, W) bool: ground connected to the start region
    obstacles: np.ndarray      # (H, W) bool: segmenter obstacles + trunk footprints (no safety margin yet)
    valid: np.ndarray          # (H, W) bool
    rejected: np.ndarray       # (H, W) bool: traversable-looking pixels dropped (isolated / too small)
    method: str
    lighting_normalized: bool
    notes: list[str] = field(default_factory=list)
    labels: np.ndarray | None = None
    label_names: dict[int, str] | None = None
    p_decision: np.ndarray | None = None      # eligibility probability (what was thresholded); p_trav is the planner-cost map


Segmenter = Callable[[np.ndarray], SegOutput]


def validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ImageError("Expected an HxWx3 uint8 BGR image.")
    if image.shape[0] < 32 or image.shape[1] < 32:
        raise ImageError(f"Image too small: {image.shape[1]}x{image.shape[0]}.")


def normalize_lighting(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Apply gentle CLAHE on the L channel only if the image is dark (mean L < 70).

    Checked on synthetic variants of 4 frames (hybrid method): CLAHE raised the traversable-mask overlap with the
    original-lighting result for dark images (IoU 0.63 -> 0.76) but slightly *lowered* it for flat or bright ones
    (0.97 -> 0.92), so well-exposed, over-exposed and low-contrast images are left untouched.
    """
    validate_image(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    if float(L.mean()) >= 70:
        return image, False
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), True


def box_footprints(shape: tuple[int, int], boxes: Sequence[Sequence[float]], foot_fraction: float = 0.12,
                   min_foot_px: int = 10) -> np.ndarray:
    """Mask of the ground contact area of detected trunks: the bottom strip of each (x0, y0, x1, y1, ...) box."""
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    for b in boxes:
        x0, y0, x1, y1 = (float(v) for v in b[:4])
        x0, x1, y0, y1 = max(0.0, x0), min(float(W), x1), max(0.0, y0), min(float(H), y1)  # clip to the image
        if x1 <= x0 or y1 <= y0:
            continue
        foot = max(min_foot_px, foot_fraction * (y1 - y0))
        xa, xb = int(np.floor(x0)), int(np.ceil(x1))
        ya, yb = int(max(0, np.floor(y1 - foot))), int(np.ceil(y1))
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = 1
    return m.astype(bool)


def start_region_mask(shape: tuple[int, int], band: tuple[float, float, float] = START_BAND) -> np.ndarray:
    H, W = shape
    m = np.zeros((H, W), bool)
    m[int(band[2] * H):, int(band[0] * W):int(band[1] * W)] = True
    return m


@dataclass(frozen=True)
class ClassGroups:
    """Which label names of a semantic model map to each navigation class (matched on the name before any comma)."""
    traversable: frozenset
    vegetation: frozenset
    obstacle: frozenset
    sky: frozenset
    traversable_weights: tuple = ()  # ((name, weight), ...) partial credit, e.g. grass passable at lower confidence
    dense_vegetation: frozenset = frozenset()  # subset of `vegetation` counted in P(vegetation); empty = all of `vegetation`


# RUGD: grass is passable (with caution), bushes are the dense vegetation that blocks.
RUGD_GROUPS = ClassGroups(frozenset(RUGD_TRAVERSABLE), frozenset(RUGD_VEGETATION), frozenset(RUGD_OBSTACLE), frozenset(RUGD_SKY),
                          dense_vegetation=frozenset({"bush"}))
# ADE20K (150 classes, SegFormer): bare ground / paths are traversable, plants/grass are vegetation, "mountain"/"hill" stay unknown.
ADE_GROUPS = ClassGroups(
    frozenset({"path", "dirt track", "earth", "sand", "road", "sidewalk", "floor"}),
    frozenset({"grass", "plant", "field", "flower"}),
    frozenset({"tree", "rock", "water", "wall", "building", "fence", "person", "car", "pole", "streetlight", "boat", "bus",
               "truck", "van", "bicycle", "animal", "stairs", "hovel", "bench", "table", "chair", "sign", "signboard", "stone"}),
    frozenset({"sky"}),
)
# DeepScene (Freiburg Forest): trail is the drivable path, grass is passable at a cost.
DEEPSCENE_NAMES = {0: "trail", 1: "grass", 2: "vegetation", 3: "obstacle", 4: "sky"}


def make_segformer_segmenter(model, imgsz: int = 480, device: str = "cpu", groups: ClassGroups = RUGD_GROUPS,
                             method: str = "RUGD-Segformer", note: str = "") -> Segmenter:
    """Wrap a loaded Hugging Face ``SegformerForSemanticSegmentation`` as a :data:`Segmenter`."""
    import torch

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    short = {i: v.split(",")[0].strip() for i, v in id2label.items()}
    n = max(id2label) + 1
    cls_of = np.full(n, UNKNOWN, np.uint8)
    for i, name in short.items():
        cls_of[i] = (TRAVERSABLE if name in groups.traversable else VEGETATION if name in groups.vegetation
                     else OBSTACLE if name in groups.obstacle else UNKNOWN)
    trav_ids = [i for i, nme in short.items() if nme in groups.traversable]
    sky_ids = [i for i, nme in short.items() if nme in groups.sky]
    block_ids = [i for i, nme in short.items() if nme in groups.obstacle]
    veg_ids = [i for i, nme in short.items() if nme in (groups.dense_vegetation or groups.vegetation)]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    cfg = TravConfig()

    def run(image: np.ndarray) -> SegOutput:
        validate_image(image)
        H, W = image.shape[:2]
        x = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (imgsz, imgsz)).astype(np.float32) / 255.0
        x = torch.from_numpy(((x - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(pixel_values=x).logits, dim=1)[0].cpu().numpy()  # (C, h, w)
        top = prob.max(0)
        arg = prob.argmax(0)
        classes = cls_of[arg]
        classes[top < cfg.unknown_confidence] = UNKNOWN
        p_trav = prob[trav_ids].sum(0).astype(np.float32)
        p_block = prob[block_ids].sum(0).astype(np.float32)
        sky = np.isin(arg, sky_ids)
        sky[int(0.45 * sky.shape[0]):] = False  # "sky" low in the frame is more likely snow/bright ground: unknown, not invalid
        up = lambda a, interp: cv2.resize(a, (W, H), interpolation=interp)  # noqa: E731
        return SegOutput(
            classes=up(classes, cv2.INTER_NEAREST),
            p_trav=np.clip(up(p_trav, cv2.INTER_LINEAR), 0, 1),
            valid=~up(sky.astype(np.uint8), cv2.INTER_NEAREST).astype(bool),
            method=method,
            notes=[note or f"{method}: pretrained semantic model; not validated on this camera or forest."],
            labels=up(arg.astype(np.uint8), cv2.INTER_NEAREST),
            label_names=short,
            p_block=np.clip(up(p_block, cv2.INTER_LINEAR), 0, 1),
            p_obstacle=np.clip(up(p_block, cv2.INTER_LINEAR), 0, 1),
            p_vegetation=np.clip(up(prob[veg_ids].sum(0).astype(np.float32), cv2.INTER_LINEAR), 0, 1),
        )

    return run


def make_rugd_segmenter(model, imgsz: int = 480, device: str = "cpu") -> Segmenter:
    """RUGD-Segformer (off-road imagery, 25 classes) as a :data:`Segmenter`."""
    return make_segformer_segmenter(model, imgsz, device, RUGD_GROUPS, "RUGD-Segformer",
                                    "Pretrained on RUGD off-road imagery; not validated on this camera or forest.")


def make_ade_segmenter(model, imgsz: int = 512, device: str = "cpu") -> Segmenter:
    """SegFormer fine-tuned on ADE20K (150 classes incl. path, dirt track, earth, sand, grass, plant, tree, rock, water)."""
    return make_segformer_segmenter(model, imgsz, device, ADE_GROUPS, "ADE20K SegFormer",
                                    "ADE20K SegFormer (NVIDIA non-commercial license): generic scene model, not off-road specific.")


def make_deepscene_segmenter(onnx_path: str, grass_weight: float = 0.6) -> Segmenter:
    """NVIDIA jetson-inference FCN-ResNet18-DeepScene (576x320): trail / grass / vegetation / obstacle / sky.

    Runs with OpenCV's DNN module (no extra dependency).  The raw output is a coarse 10x18 grid, so class
    probabilities are upsampled bilinearly.  P(traversable) = P(trail) + grass_weight * P(grass).
    """
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    cls_of = np.array([TRAVERSABLE, VEGETATION, VEGETATION, OBSTACLE, UNKNOWN], np.uint8)

    def run(image: np.ndarray) -> SegOutput:
        validate_image(image)
        H, W = image.shape[:2]
        x = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (576, 320)).astype(np.float32) / 255.0
        net.setInput(((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32))
        logits = net.forward()[0]  # (5, 10, 18)
        e = np.exp(logits - logits.max(0, keepdims=True))
        prob = e / e.sum(0, keepdims=True)
        up_p = np.stack([cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR) for c in prob])  # (5, H, W)
        arg = up_p.argmax(0)
        classes = cls_of[arg]
        classes[up_p.max(0) < TravConfig().unknown_confidence] = UNKNOWN
        p_trav = np.clip(up_p[0] + grass_weight * up_p[1], 0, 1).astype(np.float32)
        sky = arg == 4
        sky[int(0.45 * H):] = False
        p_block = np.clip(up_p[2] + up_p[3], 0, 1).astype(np.float32)  # dense vegetation + obstacle
        return SegOutput(classes, p_trav, ~sky, "DeepScene FCN", ["DeepScene FCN-ResNet18 (Freiburg Forest trails): summer-forest trained, coarse output."],
                         arg.astype(np.uint8), DEEPSCENE_NAMES, p_block, np.clip(up_p[3], 0, 1).astype(np.float32),
                         np.clip(up_p[2], 0, 1).astype(np.float32))

    return run


def make_yolo_path_segmenter(model, imgsz: int = 480, device: str = "cpu", conf: float = 0.25) -> Segmenter:
    """Wrap the custom YOLOv8-seg path model.  It only knows the narrow groomed 'Track' class, so everything
    outside its masks is UNKNOWN - expect empty results on most scenes."""

    def run(image: np.ndarray) -> SegOutput:
        validate_image(image)
        H, W = image.shape[:2]
        r = model.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        p = np.zeros((H, W), np.float32)
        if r.masks is not None:
            for poly, c in zip(r.masks.xy, r.boxes.conf.cpu().numpy()):
                tmp = np.zeros((H, W), np.uint8)
                cv2.fillPoly(tmp, [poly.astype(np.int32)], 1)
                p = np.maximum(p, tmp * float(c))
        classes = np.where(p > 0, TRAVERSABLE, UNKNOWN).astype(np.uint8)
        return SegOutput(classes, p, np.ones((H, W), bool), "YOLO path model",
                         ["Trained on 175 images of one snowy forest; detects only the groomed 'Track' class."])

    return run


def segment_classical(image: np.ndarray, config: TravConfig | None = None) -> SegOutput:
    """BASELINE ground estimator (no semantics).

    Ground is assumed to look like the patch directly in front of the vehicle (bottom centre): pixels whose
    CLAHE-normalised Lab colour is close to that patch, with low local texture, below the likely horizon.
    Fails when the start patch is on a bush/trunk/shadow edge or the ground changes colour.
    """
    validate_image(image)
    H, W = image.shape[:2]
    s = 320.0 / W
    small = cv2.resize(image, (320, max(32, int(round(H * s)))), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    feat = lab.astype(np.float32)
    feat[..., 0] *= 0.5  # lightness matters less than chroma (shadows)
    y0, y1, x0, x1 = int(0.84 * h), int(0.98 * h), int(0.38 * w), int(0.62 * w)
    seed = feat[y0:y1, x0:x1].reshape(-1, 3)
    mu = seed.mean(0)
    cov = np.cov(seed.T) + np.eye(3) * 12.0
    d = feat.reshape(-1, 3) - mu
    d2 = np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d).reshape(h, w)
    sim = np.exp(-0.5 * d2 / 4.0)

    gray = lab[..., 0].astype(np.float32)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.blur(np.sqrt(gx * gx + gy * gy), (9, 9))
    seed_tex = float(mag[y0:y1, x0:x1].mean())
    texfac = 1.0 / (1.0 + (mag / (2.0 * seed_tex + 8.0)) ** 2)
    ypos = (np.arange(h, dtype=np.float32) / h)[:, None]
    prior = np.clip((ypos - 0.30) / 0.20, 0, 1)
    p = (sim * texfac * prior).astype(np.float32)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] > 60)
    sky = (hsv[..., 2] > 190) & (hsv[..., 1] < 50) & (ypos < 0.5)
    classes = np.full((h, w), UNKNOWN, np.uint8)
    classes[green & (p < 0.5)] = VEGETATION
    classes[p >= 0.5] = TRAVERSABLE
    up = lambda a, interp: cv2.resize(a, (W, H), interpolation=interp)  # noqa: E731
    notes = ["Classical BASELINE: colour/texture similarity to the ground in front of the vehicle; no semantic understanding."]
    if seed_tex > 25:
        notes.append("The patch in front of the vehicle is highly textured - baseline is likely unreliable here.")
    return SegOutput(up(classes, cv2.INTER_NEAREST), np.clip(up(p, cv2.INTER_LINEAR), 0, 1),
                     ~up(sky.astype(np.uint8), cv2.INTER_NEAREST).astype(bool), "Classical baseline", notes)


def fuse_segmentations(primary: SegOutput, baseline: SegOutput, baseline_weight: float = 0.8) -> SegOutput:
    """Let a classical ground estimate fill in ground the pretrained model does not know (e.g. snow).

    The baseline only acts where the primary model says UNKNOWN or TRAVERSABLE; the primary's obstacle and
    vegetation labels veto it, and its probability is down-weighted (it has no semantic understanding).
    """
    allowed = (primary.classes == UNKNOWN) | (primary.classes == TRAVERSABLE)
    p = np.maximum(primary.p_trav, np.where(allowed, baseline.p_trav * baseline_weight, 0.0)).astype(np.float32)
    classes = primary.classes.copy()
    classes[(p >= 0.5) & (classes == UNKNOWN)] = TRAVERSABLE
    notes = list(primary.notes) + ["Hybrid: a classical ground estimate fills in ground types the pretrained model does not "
                                   "know (e.g. snow); the pretrained model's obstacle/vegetation labels veto it."] + list(baseline.notes)
    return SegOutput(classes, p, primary.valid, f"{primary.method} + classical", notes)


def fuse_obstacle_veto(primary: SegOutput, baseline: SegOutput) -> SegOutput:
    """Classical ground estimate led, with the pretrained model vetoing only what it calls an OBSTACLE.

    Keeps the classical baseline's accuracy on trails (grass/litter the pretrained model calls vegetation stays
    usable) but stops it from marking machines, trees, rocks or people as ground.
    """
    p = np.where(primary.classes == OBSTACLE, 0.0, baseline.p_trav).astype(np.float32)
    classes = baseline.classes.copy()
    classes[primary.classes == OBSTACLE] = OBSTACLE
    notes = list(baseline.notes) + ["Obstacle veto: classes the pretrained model labels as obstacles (trees, vehicles, rocks, ...) "
                                    "are removed from the classical ground estimate."]
    return SegOutput(classes, p, baseline.valid, "Classical + obstacle veto", notes)


# Terrain-aware passability: RUGD labels that are never ground, and labels that are passable only with caution.
TERRAIN_BLOCKED = {"bush", "water", "tree", "log", "rock", "pole", "fence", "building", "vehicle", "person", "bicycle",
                   "container/generic-object", "sign", "picnic-table",          # RUGD
                   "vegetation", "obstacle", "plant", "wall", "car", "stone"}   # DeepScene / ADE20K
TERRAIN_CAUTION = {"grass": 0.75, "rock-bed": 0.55}  # score multiplier: passable but costlier / less certain
PLANT_LABELS = ("tree", "bush", "grass", "log")


def fuse_terrain_aware(primary: SegOutput, baseline: SegOutput, allow_grass: bool = True,
                       veto_threshold: float | None = None) -> SegOutput:
    """Terrain-aware ground: union of the classical estimate and the pretrained ground classes, minus what the
    pretrained model says cannot be driven over (obstacles, bushes, water, rocks, ...).

    Grass and rock-bed stay passable at reduced confidence (so routes prefer dirt/gravel/sand/mulch but can cross
    grass); with ``allow_grass=False`` grass is blocked.  The classical estimate covers ground the pretrained
    model does not know (e.g. snow).  Needs ``primary.labels``; otherwise falls back to the plain obstacle veto.

    ``veto_threshold`` makes the veto *soft*: when the primary provides ``p_block`` (P of obstacle/dense vegetation),
    a pixel is blocked only where ``p_block >= veto_threshold`` instead of wherever blocked classes win the argmax.
    A higher threshold keeps more ground that the primary is unsure about (fewer false stops, e.g. on leaf litter)
    at the price of a weaker veto.
    """
    if primary.labels is None or primary.label_names is None:
        return fuse_obstacle_veto(primary, baseline)
    names = primary.label_names
    ids = lambda group: [i for i, n in names.items() if n in group]  # noqa: E731
    if veto_threshold is not None and primary.p_block is not None:
        blocked = primary.p_block >= veto_threshold
    else:
        blocked = np.isin(primary.labels, ids(TERRAIN_BLOCKED)) | (primary.classes == OBSTACLE)
    scale = np.ones(primary.labels.shape, np.float32)
    for name, mult in TERRAIN_CAUTION.items():
        scale[np.isin(primary.labels, ids({name}))] = mult
    if not allow_grass:
        blocked |= np.isin(primary.labels, ids({"grass"}))
    p = np.maximum(baseline.p_trav * 0.9, primary.p_trav) * scale
    p = np.where(blocked, 0.0, p).astype(np.float32)
    classes = np.full(p.shape, UNKNOWN, np.uint8)
    classes[np.isin(primary.labels, ids({"grass", "bush"}))] = VEGETATION
    classes[p >= 0.5] = TRAVERSABLE
    classes[blocked & ~np.isin(primary.labels, ids({"bush"}))] = OBSTACLE
    notes = list(primary.notes) + ["Terrain-aware: ground = classical estimate U pretrained ground classes, minus obstacles, "
                                   "bushes, water and rocks; grass is passable at lower confidence."] + list(baseline.notes)
    return SegOutput(classes, p, primary.valid, "Terrain-aware (classical + RUGD)", notes, primary.labels, names)


def fuse_confidence_aware(primary: SegOutput, baseline: SegOutput, allow_grass: bool = True,
                          obstacle_threshold: float = 0.5, veg_weight: float = 1.0) -> SegOutput:
    """Classical-led ground with a confidence-weighted vegetation penalty and a *separate* hard obstacle rule.

    * ``p_decision = max(0.9 * classical, primary ground) * (1 - veg_weight * P(vegetation))`` is what gets thresholded.
      ``veg_weight = 1`` blocks ground the model calls vegetation; smaller values trade safety for fewer false stops.
    * Hard OBSTACLE (with the planner's safety margin) only where ``P(obstacle) >= obstacle_threshold`` - confirmed
      evidence.  Vegetation-vetoed or unsure pixels are merely not eligible (class VEGETATION / UNKNOWN), no margin.
    * Caution for grass / rock-bed lowers only the planner's *cost* confidence (``p_cost``), never eligibility.
    Needs ``primary.p_obstacle`` / ``p_vegetation``; otherwise falls back to :func:`fuse_terrain_aware`.
    """
    if primary.p_obstacle is None or primary.p_vegetation is None:
        return fuse_terrain_aware(primary, baseline, allow_grass)
    base = np.maximum(baseline.p_trav * 0.9, primary.p_trav)
    p_dec = (base * (1.0 - veg_weight * primary.p_vegetation)).astype(np.float32)
    hard = primary.p_obstacle >= obstacle_threshold
    scale = np.ones(p_dec.shape, np.float32)
    if primary.labels is not None and primary.label_names is not None:
        for name, mult in TERRAIN_CAUTION.items():
            ids_ = [i for i, n in primary.label_names.items() if n == name]
            if ids_:
                scale[np.isin(primary.labels, ids_)] = mult
        if not allow_grass:
            g_ = [i for i, n in primary.label_names.items() if n == "grass"]
            if g_:
                p_dec[np.isin(primary.labels, g_)] = 0.0
    p_dec[hard] = 0.0
    classes = np.full(p_dec.shape, UNKNOWN, np.uint8)
    classes[primary.classes == VEGETATION] = VEGETATION
    classes[p_dec >= 0.5] = TRAVERSABLE
    classes[hard] = OBSTACLE
    notes = list(primary.notes) + [f"Confidence-aware fusion (veg weight {veg_weight}, obstacle threshold {obstacle_threshold}): hard obstacles "
                                   "only from confirmed obstacle evidence; vegetation lowers ground confidence; grass caution affects route cost only."] + list(baseline.notes)
    return SegOutput(classes, p_dec, primary.valid, "Confidence-aware (classical + " + primary.method + ")", notes, primary.labels, primary.label_names,
                     p_cost=(p_dec * scale).astype(np.float32))


def fuse_multi_source(rugd: SegOutput, deepscene: SegOutput, baseline: SegOutput, allow_grass: bool = True,
                      obstacle_threshold: float = 0.5, bush_threshold: float = 0.5) -> SegOutput:
    """Ground = union of evidence; hard veto = any source that confirms an obstacle.

    Ground eligibility: ``max(0.9 * classical, RUGD ground, DeepScene trail)``.  Hard OBSTACLE (with margin) where RUGD's obstacle
    group (trees, people, vehicles, logs, rocks, water ...) or DeepScene's obstacle class has P >= ``obstacle_threshold``, or RUGD's
    *bush* class has P >= ``bush_threshold``.  DeepScene's "vegetation" is deliberately NOT a veto (it labels leaf litter that way).
    Grass / rock-bed stay passable at a higher route cost (RUGD labels).  Trunk and hazard-detector footprints are added by the caller.
    """
    p_ground = np.maximum.reduce([baseline.p_trav * 0.9, rugd.p_trav, deepscene.p_trav]).astype(np.float32)
    hard = (rugd.p_obstacle >= obstacle_threshold) | (deepscene.p_obstacle >= obstacle_threshold) | (rugd.p_vegetation >= bush_threshold)
    scale = np.ones(p_ground.shape, np.float32)
    if rugd.labels is not None and rugd.label_names is not None:
        for name, mult in TERRAIN_CAUTION.items():
            ids_ = [i for i, n in rugd.label_names.items() if n == name]
            if ids_:
                scale[np.isin(rugd.labels, ids_)] = mult
        if not allow_grass:
            g_ = [i for i, n in rugd.label_names.items() if n == "grass"]
            if g_:
                p_ground[np.isin(rugd.labels, g_)] = 0.0
    p_ground[hard] = 0.0
    classes = np.full(p_ground.shape, UNKNOWN, np.uint8)
    classes[p_ground >= 0.5] = TRAVERSABLE
    classes[hard] = OBSTACLE
    notes = ["Multi-source: classical + RUGD + DeepScene ground evidence; hard veto from RUGD obstacles/bushes and DeepScene obstacles."] + list(rugd.notes)
    return SegOutput(classes, p_ground, rugd.valid & deepscene.valid, "Multi-source (RUGD + DeepScene + classical)", notes, rugd.labels, rugd.label_names,
                     p_cost=(p_ground * scale).astype(np.float32))


def make_terrain_segmenter(primary: Segmenter, allow_grass: bool = True, veto_threshold: float | None = None,
                           veg_weight: float | None = None, obstacle_threshold: float = 0.5) -> Segmenter:
    """Terrain-aware fusion.  ``veg_weight`` set -> confidence-aware fusion; else the hard / soft veto fusion."""
    if veg_weight is not None:
        return lambda image: fuse_confidence_aware(primary(image), segment_classical(image), allow_grass, obstacle_threshold, veg_weight)
    return lambda image: fuse_terrain_aware(primary(image), segment_classical(image), allow_grass, veto_threshold)


def plant_detections(result: TraversabilityResult, min_area_fraction: float = 0.004) -> dict:
    """Plants and terrain found in the image.

    Returns ``{"summary": {label: % of valid pixels}, "bushes": [(x0, y0, x1, y1, area_fraction)], "source": str}``.
    With a pretrained terrain model the labels are its own (e.g. grass, bush, tree, log or DeepScene's trail/grass/vegetation/obstacle); without one only an
    *approximate* ``vegetation`` share is available (green, textured pixels from the classical estimator).
    """
    valid = result.valid
    n = max(1, int(valid.sum()))
    if result.labels is None or result.label_names is None:
        veg = (result.classes == VEGETATION) & valid
        return {"summary": {"vegetation (approx.)": 100.0 * int(veg.sum()) / n}, "bushes": [], "source": "approximate (colour)"}
    summary = {}
    for i, name in result.label_names.items():
        pct = 100.0 * int(((result.labels == i) & valid).sum()) / n
        if pct >= 0.5 and name not in ("sky", "void"):
            summary[name] = pct
    bushes = []
    bid = [i for i, nme in result.label_names.items() if nme == "bush"]
    if bid:
        m = (np.isin(result.labels, bid)).astype(np.uint8)
        k = _odd(0.01 * m.shape[1])
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cnt, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for j in range(1, cnt):
            frac = st[j, cv2.CC_STAT_AREA] / m.size
            if frac >= min_area_fraction:
                x, y, w, h = (int(st[j, c]) for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
                bushes.append((x, y, x + w, y + h, float(frac)))
    return {"summary": dict(sorted(summary.items(), key=lambda kv: -kv[1])), "bushes": bushes, "source": result.method}


def make_veto_segmenter(primary: Segmenter) -> Segmenter:
    return lambda image: fuse_obstacle_veto(primary(image), segment_classical(image))


def make_hybrid_segmenter(primary: Segmenter) -> Segmenter:
    """Primary (pretrained) segmenter fused with the classical baseline via :func:`fuse_segmentations`."""
    return lambda image: fuse_segmentations(primary(image), segment_classical(image))


def _odd(n: float, lo: int = 3) -> int:
    n = max(lo, int(round(n)))
    return n if n % 2 else n + 1


def detect_traversable_region(
    image: np.ndarray,
    obstacle_boxes: Sequence[Sequence[float]] = (),
    segmenter: Segmenter | None = None,
    config: TravConfig | None = None,
) -> TraversabilityResult:
    """Estimate traversable ground, vegetation, obstacles and unknown areas for one BGR image.

    Steps: dark-image contrast normalisation -> segmenter (or the classical baseline) -> fuse trunk footprints into
    the obstacle mask -> threshold P(traversable) -> morphological open/close -> keep only ground connected to
    the start band at the bottom centre (isolated patches are rejected, shown as UNKNOWN) -> final class map.
    Uncertain pixels are never promoted to traversable.
    """
    cfg = config or TravConfig()
    validate_image(image)
    H, W = image.shape[:2]
    img, normalized = normalize_lighting(image) if cfg.normalize_lighting else (image, False)
    seg = segmenter(img) if segmenter is not None else segment_classical(img, cfg)
    for name, a in (("classes", seg.classes), ("p_trav", seg.p_trav), ("valid", seg.valid)):
        if a.shape != (H, W):
            raise ImageError(f"Segmenter output '{name}' has shape {a.shape}, expected {(H, W)}.")

    obstacles = (seg.classes == OBSTACLE) | box_footprints((H, W), obstacle_boxes, cfg.foot_fraction, cfg.min_foot_px)
    cand = ((seg.p_trav >= cfg.min_confidence) & ~obstacles & seg.valid).astype(np.uint8)
    k = _odd(0.012 * W)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cand &= (~obstacles).astype(np.uint8)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    start = start_region_mask((H, W))
    keep = np.zeros(n, bool)
    min_area = cfg.min_region_fraction * H * W
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area and start[lbl == i].any():
            keep[i] = True
    traversable = keep[lbl]
    rejected = (cand > 0) & ~traversable

    classes = seg.classes.copy()
    classes[(classes == TRAVERSABLE)] = UNKNOWN
    classes[traversable] = TRAVERSABLE
    classes[obstacles] = OBSTACLE
    notes = list(seg.notes)
    if normalized:
        notes.append("Contrast normalisation (CLAHE) was applied because the image is dark.")
    if not traversable.any():
        notes.append("No traversable ground connected to the start region was found.")
    return TraversabilityResult(classes, seg.p_cost if seg.p_cost is not None else seg.p_trav, traversable, obstacles, seg.valid,
                                rejected, seg.method, normalized, notes, seg.labels, seg.label_names, seg.p_trav)
