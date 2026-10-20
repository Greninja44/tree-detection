"""Single-image inference helpers used by app.py: trunk detection and drivable-path isolation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detector import cuda_functional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUGD_DIR = PROJECT_ROOT / "models" / "rugd_segformer"
RUGD_DRIVABLE = {"dirt", "sand", "gravel", "mulch", "asphalt"}
GREEN = np.array([0, 255, 0], np.float32)


def device() -> str:
    return "cuda:0" if cuda_functional() else "cpu"


@dataclass
class PathResult:
    mask: np.ndarray  # (H, W) bool
    coverage: float
    offset: float  # lateral centre of the path in the lower quarter; -1 = far left, +1 = far right (nan if none)


def load_yolo(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def load_rugd():
    from transformers import SegformerForSemanticSegmentation

    return SegformerForSemanticSegmentation.from_pretrained(str(RUGD_DIR)).to(device()).eval()


def detect_trunks(model, img_bgr: np.ndarray, conf: float, imgsz: int,
                  dev: str | None = None) -> list[tuple[float, float, float, float, float]]:
    r = model.predict(img_bgr, conf=conf, imgsz=imgsz, device=dev or device(), verbose=False)[0]
    return [(*map(float, b), float(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]


COCO_HAZARD_IDS = [0, 1, 2, 3, 5, 6, 7] + list(range(14, 24))   # person, bicycle, car, motorcycle, bus, train, truck, animals


def detect_hazards(model, img_bgr: np.ndarray, conf: float = 0.25, imgsz: int = 640,
                   dev: str | None = None) -> list[tuple[float, float, float, float, float]]:
    """People / vehicles / animals from a COCO detector (e.g. yolov8n.pt), as (x0, y0, x1, y1, conf) obstacle boxes.

    Partial coverage only: it finds people reasonably often but not reliably machines or unusual vehicles."""
    r = model.predict(img_bgr, conf=conf, imgsz=imgsz, device=dev or device(), verbose=False)[0]
    keep = np.isin(r.boxes.cls.cpu().numpy().astype(int), COCO_HAZARD_IDS)
    return [(*map(float, b), float(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy()[keep], r.boxes.conf.cpu().numpy()[keep])]


def _path_result(mask: np.ndarray) -> PathResult:
    H, W = mask.shape
    xs = np.nonzero(mask[int(H * 0.75):])[1]
    offset = float((xs.mean() - W / 2) / (W / 2)) if xs.size else float("nan")
    return PathResult(mask, float(mask.mean()), offset)


def segment_path_yolo(model, img_bgr: np.ndarray, conf: float, imgsz: int) -> PathResult:
    H, W = img_bgr.shape[:2]
    r = model.predict(img_bgr, conf=conf, imgsz=imgsz, device=device(), verbose=False)[0]
    mask = np.zeros((H, W), np.uint8)
    if r.masks is not None:
        for poly in r.masks.xy:
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return _path_result(mask.astype(bool))


def segment_path_rugd(model, img_bgr: np.ndarray) -> PathResult:
    import torch

    H, W = img_bgr.shape[:2]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (640, 640)).astype(np.float32) / 255.0
    x = torch.from_numpy(((x - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device())
    with torch.no_grad():
        logits = model(pixel_values=x).logits
    seg = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
    ids = [int(i) for i, n in model.config.id2label.items() if n in RUGD_DRIVABLE]
    return _path_result(np.isin(seg, ids))


def draw_path(img_bgr: np.ndarray, path: PathResult | None) -> np.ndarray:
    out = img_bgr.astype(np.float32)
    if path is not None:
        out[path.mask] = 0.5 * out[path.mask] + 0.5 * GREEN
    return out.astype(np.uint8)


def draw_trunks(img_bgr: np.ndarray, boxes) -> np.ndarray:
    out = img_bgr.copy()
    for x0, y0, x1, y1, c in boxes:
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 140, 255), 2)
        cv2.putText(out, f"{c:.2f}", (int(x0), max(12, int(y0) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)
    return out
