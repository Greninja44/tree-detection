"""YOLO-based tree-trunk detection pipeline.

Design notes
------------
- The model is loaded **once** and cached at module level, so frame-by-frame
  video processing never reloads weights.
- Ultralytics / PyTorch are imported lazily: the rest of the project (lighting
  analysis, metrics, tests) keeps working when they are unavailable.
- A generic pretrained detector (e.g. COCO ``yolov8n.pt``) does **not** contain
  a ``tree_trunk`` class.  Such weights are accepted for pipeline smoke tests
  but are explicitly flagged via :attr:`TreeDetector.is_tree_model` and never
  silently reported as a tree detector.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .preprocessing import ensure_bgr, validate_image_path

logger = logging.getLogger(__name__)

#: Canonical class name used for tree trunks in custom datasets.
TREE_TRUNK_CLASS_NAME = "tree_trunk"

#: Class-name tokens accepted as evidence that a checkpoint is a tree model.
_TREE_CLASS_TOKENS: tuple[str, ...] = ("tree", "trunk", "stem", "baum", "arbre", "bosque")

#: Default location searched for model weights.
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

#: Environment variable that can override the default models directory.
MODELS_DIR_ENV = "TREE_DETECTION_MODELS_DIR"


class DetectorError(RuntimeError):
    """Base class for detector failures."""


class DetectorUnavailableError(DetectorError):
    """Raised when the inference backend or the weights are not available."""


class WeightsNotFoundError(DetectorError):
    """Raised when a checkpoint file cannot be located."""


class InvalidWeightsError(DetectorError):
    """Raised when a checkpoint exists but cannot be loaded or is incompatible."""


@dataclasses.dataclass(frozen=True)
class Detection:
    """A single detected object."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    frame_index: int = -1
    source_name: str = ""
    inference_time_ms: float = 0.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "x1": round(float(self.x1), 2),
            "y1": round(float(self.y1), 2),
            "x2": round(float(self.x2), 2),
            "y2": round(float(self.y2), 2),
            "confidence": round(float(self.confidence), 4),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "frame_index": int(self.frame_index),
            "source_name": self.source_name,
            "inference_time_ms": round(float(self.inference_time_ms), 3),
        }


@dataclasses.dataclass
class DetectionResult:
    """All detections for one image or video frame."""

    source_name: str
    detections: list[Detection]
    inference_time_ms: float
    image_width: int
    image_height: int
    frame_index: int = -1
    model_name: str = ""
    is_tree_model: bool = False

    @property
    def count(self) -> int:
        """Number of detected objects (after confidence filtering)."""
        return len(self.detections)

    @property
    def boxes(self) -> np.ndarray:
        """Detections as an ``(N, 4)`` float32 array of ``xyxy`` boxes (possibly empty)."""
        if not self.detections:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray([d.xyxy for d in self.detections], dtype=np.float32)

    @property
    def confidences(self) -> np.ndarray:
        """Confidence scores as a 1-D float32 array (possibly empty)."""
        if not self.detections:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray([d.confidence for d in self.detections], dtype=np.float32)

    @property
    def class_names(self) -> list[str]:
        return [d.class_name for d in self.detections]

    @property
    def fps(self) -> float:
        """Instantaneous processing rate for this result (0 if instantaneous time is 0)."""
        if self.inference_time_ms <= 0:
            return 0.0
        return 1000.0 / self.inference_time_ms

    def filtered(self, confidence_threshold: float) -> "DetectionResult":
        """Return a copy keeping only detections with ``confidence >= threshold``."""
        keep = [d for d in self.detections if d.confidence >= confidence_threshold]
        return dataclasses.replace(self, detections=keep)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "frame_index": self.frame_index,
            "count": self.count,
            "inference_time_ms": round(self.inference_time_ms, 3),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "model_name": self.model_name,
            "is_tree_model": self.is_tree_model,
            "detections": [d.to_dict() for d in self.detections],
        }


# --------------------------------------------------------------------------- #
# Pure-numpy helpers (unit-testable without a model)
# --------------------------------------------------------------------------- #

def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union of two ``xyxy`` boxes."""
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[int]:
    """Greedy non-maximum suppression.

    Parameters
    ----------
    boxes:
        ``(N, 4)`` array of ``xyxy`` boxes.
    scores:
        ``(N,)`` array of confidences (higher = better).
    iou_threshold:
        Overlap above which a lower-scoring box is suppressed.

    Returns
    -------
    list of int
        Indices of kept boxes in descending score order.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.shape[0] == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.asarray([box_iou(boxes[i], boxes[j]) for j in rest], dtype=np.float32)
        order = rest[ious <= float(iou_threshold)]
    return keep


def filter_detections(
    detections: Sequence[Detection],
    confidence_threshold: float,
    iou_threshold: float = 0.5,
    apply_nms: bool = True,
) -> list[Detection]:
    """Filter by confidence and (optionally) apply class-agnostic NMS."""
    kept = [d for d in detections if d.confidence >= confidence_threshold]
    if not apply_nms or len(kept) <= 1:
        return kept
    boxes = np.asarray([d.xyxy for d in kept], dtype=np.float32)
    scores = np.asarray([d.confidence for d in kept], dtype=np.float32)
    idx = nms(boxes, scores, iou_threshold=iou_threshold)
    return [kept[i] for i in idx]


# --------------------------------------------------------------------------- #
# Weight resolution & capability validation
# --------------------------------------------------------------------------- #

def models_dir() -> Path:
    """Directory searched for checkpoints (env var ``TREE_DETECTION_MODELS_DIR`` overrides)."""
    env = os.environ.get(MODELS_DIR_ENV)
    return Path(env).expanduser() if env else DEFAULT_MODELS_DIR


def list_available_weights() -> list[Path]:
    """Return ``.pt`` / ``.onnx`` / ``.engine`` files inside the models directory."""
    d = models_dir()
    if not d.is_dir():
        return []
    found = [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in {".pt", ".onnx", ".engine"}]
    return found


def resolve_weights(weights: str | Path | None) -> Path:
    """Locate a checkpoint file.

    Resolution order: exact path → ``models/<weights>`` → error listing the
    files that *are* present (so users know what to supply).
    """
    if weights is None or str(weights).strip() == "":
        available = list_available_weights()
        hint = (
            f" Files currently in '{models_dir()}': {', '.join(p.name for p in available)}"
            if available
            else f" No checkpoint files found in '{models_dir()}'."
        )
        raise WeightsNotFoundError(
            "No model weights specified. Supply a trained tree-trunk checkpoint "
            f"(e.g. runs/tree_trunk/weights/best.pt).{hint}"
        )
    p = Path(str(weights)).expanduser()
    if p.is_file():
        return p
    candidate = models_dir() / p.name if p.suffix else models_dir() / f"{p.name}.pt"
    if candidate.is_file():
        return candidate
    available = list_available_weights()
    hint = (
        f" Available in '{models_dir()}': {', '.join(q.name for q in available)}"
        if available
        else f" The directory '{models_dir()}' is empty."
    )
    raise WeightsNotFoundError(f"Model weights not found: {weights}.{hint}")


def is_tree_class_name(name: str) -> bool:
    """Heuristic check whether a class label refers to a tree/trunk."""
    lowered = str(name).strip().lower()
    return any(tok in lowered for tok in _TREE_CLASS_TOKENS)


def validate_class_names(names: dict[int, str] | list[str] | None) -> tuple[bool, list[str]]:
    """Check whether a checkpoint exposes a tree-trunk-like class.

    Returns ``(is_tree_model, class_names)``.  COCO-style models return
    ``(False, [...])`` — the caller must then treat outputs as a generic
    baseline, not as tree detections.
    """
    if names is None:
        return False, []
    if isinstance(names, dict):
        name_list = [str(v) for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
    else:
        name_list = [str(v) for v in names]
    return any(is_tree_class_name(n) for n in name_list), name_list


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #

class TreeDetector:
    """Runs YOLO inference for tree-trunk detection.

    Parameters
    ----------
    weights:
        Path (or models-dir filename) of a ``.pt`` checkpoint.  ``None`` raises
        :class:`WeightsNotFoundError` with setup instructions.
    confidence_threshold:
        Detections below this score are dropped.
    iou_threshold:
        IoU threshold for non-maximum suppression.
    device:
        ``"cpu"``, ``"cuda"`` or ``None`` for automatic selection.
    imgsz:
        Inference size in pixels.
    class_names_filter:
        Optional class ids to keep (defaults to every tree-trunk-like class, or
        every class when the checkpoint is a verified tree model).
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | None = None,
        imgsz: int = 640,
        class_names_filter: Sequence[int] | None = None,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.weights_path = resolve_weights(weights)
        self.device = device
        self.class_names_filter = list(class_names_filter) if class_names_filter else None

        self._model = _load_model(str(self.weights_path))
        self.names = _extract_names(self._model)
        self.is_tree_model, self.class_names = validate_class_names(self.names)
        self.backend_device = _detect_device(self._model, self.device)
        self.model_label = self.weights_path.name

        if not self.is_tree_model:
            logger.warning(
                "Checkpoint '%s' does not expose a '%s' class (classes: %s). "
                "Outputs are a generic baseline and must not be reported as tree detections.",
                self.weights_path.name,
                TREE_TRUNK_CLASS_NAME,
                ", ".join(self.class_names[:12]) + ("..." if len(self.class_names) > 12 else ""),
            )

    # -- introspection ----------------------------------------------------- #

    @property
    def supports_tree_class(self) -> bool:
        """``True`` only when the checkpoint has a tree-trunk-like class."""
        return self.is_tree_model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def describe(self) -> dict[str, Any]:
        """Human-readable summary used by the UI and reports."""
        return {
            "weights": str(self.weights_path),
            "model_label": self.model_label,
            "is_tree_model": self.is_tree_model,
            "classes": self.class_names,
            "device": self.backend_device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "imgsz": self.imgsz,
        }

    # -- inference --------------------------------------------------------- #

    def detect(
        self,
        image: np.ndarray,
        source_name: str = "",
        frame_index: int = -1,
    ) -> DetectionResult:
        """Run detection on a BGR/gray numpy image and return a :class:`DetectionResult`.

        Empty detections are valid and returned as an empty list (never ``None``).
        """
        bgr = ensure_bgr(image)
        h, w = bgr.shape[:2]
        start = time.perf_counter()
        try:
            results = self._model.predict(
                source=bgr,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.backend_device,
                verbose=False,
            )
        except Exception as exc:  # backend failure must not crash batch jobs
            raise DetectorError(
                f"Inference failed for '{source_name or 'image'}': {exc}"
            ) from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        detections = self._to_detections(
            results,
            source_name=source_name,
            frame_index=frame_index,
            inference_time_ms=elapsed_ms,
            width=w,
            height=h,
        )
        detections = filter_detections(
            detections,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            apply_nms=True,
        )
        result = DetectionResult(
            source_name=source_name,
            detections=detections,
            inference_time_ms=elapsed_ms,
            image_width=w,
            image_height=h,
            frame_index=frame_index,
            model_name=self.model_label,
            is_tree_model=self.is_tree_model,
        )
        logger.debug(
            "%s: %d detection(s) in %.1f ms", source_name or "image", result.count, elapsed_ms
        )
        return result

    def detect_image_file(
        self, path: str | Path, frame_index: int = -1
    ) -> tuple[np.ndarray, DetectionResult]:
        """Load an image file, detect, and return ``(image, result)``."""
        import cv2  # local import keeps module import light

        p = validate_image_path(path)
        image = cv2.imread(str(p))
        if image is None:
            from .preprocessing import InputValidationError

            raise InputValidationError(f"Failed to decode image '{p.name}'.")
        result = self.detect(image, source_name=p.name, frame_index=frame_index)
        return image, result

    # -- internals --------------------------------------------------------- #

    def _to_detections(
        self,
        results: Any,
        source_name: str,
        frame_index: int,
        inference_time_ms: float,
        width: int,
        height: int,
    ) -> list[Detection]:
        detections: list[Detection] = []
        if not results:
            return detections
        first = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(first, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
        except AttributeError:  # pragma: no cover - alt backend shapes
            xyxy = np.asarray(boxes.xyxy, dtype=np.float32)
            confs = np.asarray(boxes.conf, dtype=np.float32)
            clss = np.asarray(boxes.cls, dtype=int)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            cls_id = int(cls_id)
            if self.class_names_filter is not None and cls_id not in self.class_names_filter:
                continue
            name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else f"class_{cls_id}"
            detections.append(
                Detection(
                    x1=float(np.clip(x1, 0, width)),
                    y1=float(np.clip(y1, 0, height)),
                    x2=float(np.clip(x2, 0, width)),
                    y2=float(np.clip(y2, 0, height)),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=name,
                    frame_index=frame_index,
                    source_name=source_name,
                    inference_time_ms=inference_time_ms,
                )
            )
        return detections


# --------------------------------------------------------------------------- #
# Backend loading helpers (cached)
# --------------------------------------------------------------------------- #

_MODEL_CACHE: dict[str, Any] = {}


def _import_ultralytics() -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise DetectorUnavailableError(
            "The 'ultralytics' package is not installed (or its dependency 'torch' is missing). "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return YOLO


def _load_model(weights_path: str) -> Any:
    """Load (or reuse) a YOLO model for *weights_path*.

    Models are cached by absolute path + mtime so repeated detector
    construction never reloads weights from disk.
    """
    key = f"{os.path.abspath(weights_path)}|{os.path.getmtime(weights_path):.0f}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    YOLO = _import_ultralytics()
    try:
        model = YOLO(weights_path)
    except Exception as exc:
        raise InvalidWeightsError(
            f"Could not load checkpoint '{weights_path}': {exc}. "
            "Make sure the file is a valid Ultralytics YOLO checkpoint."
        ) from exc
    _MODEL_CACHE[key] = model
    logger.info("Loaded model weights from %s", weights_path)
    return model


def _extract_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


_CUDA_WORKING: dict[str, bool] = {}


def cuda_functional() -> bool:
    """Return ``True`` only if CUDA tensors actually *run* on this machine.

    ``torch.cuda.is_available()`` can be ``True`` while the installed build has
    no kernels for the local GPU (common on Jetson with generic PyPI wheels).
    The result is cached after the first probe.
    """
    if "working" in _CUDA_WORKING:
        return _CUDA_WORKING["working"]
    try:
        import torch

        if not torch.cuda.is_available():
            _CUDA_WORKING["working"] = False
            return False
        a = torch.ones(8, 8, device="cuda")
        b = (a @ a).sum().item()
        torch.cuda.synchronize()
        working = bool(abs(b - 64.0) < 1e-3)
    except Exception as exc:
        logger.warning("CUDA probe failed (%s); using CPU.", exc)
        working = False
    if not working:
        logger.warning(
            "torch.cuda.is_available() is True but a CUDA kernel test failed. "
            "This usually means the installed PyTorch build does not support this GPU's "
            "compute capability (e.g. generic PyPI wheels on Jetson). Falling back to CPU."
        )
    _CUDA_WORKING["working"] = working
    return working


def _detect_device(model: Any, requested: str | None) -> str:
    """Resolve the inference device.

    - ``requested`` set to ``cpu`` → always CPU.
    - ``requested`` set to ``cuda…`` → used only if :func:`cuda_functional`,
      otherwise falls back to CPU with a warning.
    - ``requested=None`` → ``cuda:0`` when functional, else ``cpu``.
    """
    if requested:
        requested = requested.strip().lower()
        if requested == "cpu":
            return "cpu"
        if requested.startswith("cuda"):
            if cuda_functional():
                return requested
            logger.warning("CUDA requested but not functional here; falling back to CPU.")
            return "cpu"
        return requested
    return "cuda:0" if cuda_functional() else "cpu"


def available_devices() -> list[str]:
    """Return device strings usable for inference (``["cpu"]`` plus CUDA if present)."""
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.insert(0, "cuda:0")
    except ImportError:
        pass
    return devices
