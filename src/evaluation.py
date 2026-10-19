"""Detection evaluation metrics for tree-trunk detectors.

Implements precision / recall / F1, average precision (all-point interpolation),
mAP@0.5, mAP@0.5:0.95, false positives / negatives, inference timing, FPS and
tree-count error — computed per lighting category as well as overall.

All metrics are **pure functions over detections + ground truth**.  When ground
truth is missing, :func:`evaluate_predictions` returns a report whose
``has_ground_truth`` flag is ``False`` and whose metric fields are ``None`` so
that callers can display "unavailable" instead of fabricated numbers.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .detector import Detection, box_iou

#: IoU thresholds swept for mAP@0.5:0.95.
MAP_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))

#: Threshold for counting a match as a true positive at a single IoU level.
IOU_MATCH_THRESHOLD = 0.5


class EvaluationError(ValueError):
    """Raised for malformed ground truth or metric inputs."""


# --------------------------------------------------------------------------- #
# Ground truth containers
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class GroundTruthBox:
    """A labelled ground-truth box (absolute ``xyxy`` pixels)."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def parse_yolo_label_lines(
    lines: Iterable[str],
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> list[GroundTruthBox]:
    """Convert YOLO label lines (``cls cx cy w h`` normalized) to absolute boxes.

    Malformed lines are skipped with a warning-free filter; a
    :class:`EvaluationError` is raised when the image size is invalid.
    """
    if image_width <= 0 or image_height <= 0:
        raise EvaluationError(f"Invalid image size {image_width}x{image_height}")
    boxes: list[GroundTruthBox] = []
    for raw in lines:
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            continue
        half_w, half_h = w * image_width / 2.0, h * image_height / 2.0
        x_c, y_c = cx * image_width, cy * image_height
        boxes.append(
            GroundTruthBox(
                x1=max(0.0, x_c - half_w),
                y1=max(0.0, y_c - half_h),
                x2=min(float(image_width), x_c + half_w),
                y2=min(float(image_height), y_c + half_h),
                class_id=cls if class_id == 0 else class_id,
            )
        )
    return boxes


def load_yolo_label_file(
    path: str | Path, image_width: int, image_height: int
) -> list[GroundTruthBox]:
    """Load a YOLO-format label file (``labels/<stem>.txt``)."""
    p = Path(path).expanduser()
    if not p.exists():
        return []
    return parse_yolo_label_lines(p.read_text(encoding="utf-8").splitlines(), image_width, image_height)


# --------------------------------------------------------------------------- #
# Matching primitives
# --------------------------------------------------------------------------- #

def match_predictions(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = IOU_MATCH_THRESHOLD,
) -> tuple[int, int, int, list[tuple[int, int, float]]]:
    """Greedy score-ordered matching of predictions to ground truth.

    Returns ``(true_positives, false_positives, false_negatives, matches)``
    where ``matches`` is a list of ``(pred_idx, gt_idx, iou)``.
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)

    n_pred, n_gt = pred_boxes.shape[0], gt_boxes.shape[0]
    if n_pred == 0:
        return 0, 0, n_gt, []
    if n_gt == 0:
        return 0, n_pred, 0, []

    order = np.argsort(-pred_scores)
    gt_taken = np.zeros(n_gt, dtype=bool)
    matches: list[tuple[int, int, float]] = []
    tp = fp = 0
    for pi in order:
        best_iou, best_gi = 0.0, -1
        for gi in range(n_gt):
            if gt_taken[gi]:
                continue
            iou = box_iou(pred_boxes[pi], gt_boxes[gi])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0 and best_iou >= iou_threshold:
            gt_taken[best_gi] = True
            tp += 1
            matches.append((int(pi), int(best_gi), float(best_iou)))
        else:
            fp += 1
    fn = int((~gt_taken).sum())
    return tp, fp, fn, matches


def _detection_to_arrays(
    detections: Sequence[Detection],
) -> tuple[np.ndarray, np.ndarray]:
    if not detections:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    boxes = np.asarray([d.xyxy for d in detections], dtype=np.float32)
    scores = np.asarray([d.confidence for d in detections], dtype=np.float32)
    return boxes, scores


# --------------------------------------------------------------------------- #
# Average precision
# --------------------------------------------------------------------------- #

def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """All-point interpolated AP for a precision-recall curve."""
    recalls = np.asarray(recalls, dtype=np.float64)
    precisions = np.asarray(precisions, dtype=np.float64)
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    # Monotone-decreasing envelope from the right.
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _ap_at_iou(
    per_image: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    iou_threshold: float,
) -> float:
    """Compute AP for one IoU threshold over ``[(pred_boxes, pred_scores, gt_boxes), ...]``."""
    all_scores: list[float] = []
    all_tp: list[int] = []
    total_gt = 0
    for pred_boxes, pred_scores, gt_boxes in per_image:
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
        pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
        total_gt += gt_boxes.shape[0]
        if pred_boxes.shape[0] == 0:
            continue
        n_pred = pred_boxes.shape[0]
        if gt_boxes.shape[0] == 0:
            all_scores.extend(float(s) for s in pred_scores)
            all_tp.extend([0] * n_pred)
            continue
        ious = np.zeros((n_pred, gt_boxes.shape[0]), dtype=np.float32)
        for gi in range(gt_boxes.shape[0]):
            for pi in range(n_pred):
                ious[pi, gi] = box_iou(pred_boxes[pi], gt_boxes[gi])
        gt_taken = np.zeros(gt_boxes.shape[0], dtype=bool)
        order = np.argsort(-pred_scores)
        tp_flags = np.zeros(n_pred, dtype=np.int32)
        for pi in order:
            gi = int(np.argmax(ious[pi]))
            best = float(ious[pi, gi])
            if best >= iou_threshold and not gt_taken[gi]:
                gt_taken[gi] = True
                tp_flags[pi] = 1
        all_scores.extend(float(s) for s in pred_scores)
        all_tp.extend(int(v) for v in tp_flags)

    if total_gt == 0 or not all_scores:
        return 0.0
    scores = np.asarray(all_scores, dtype=np.float64)
    tp = np.asarray(all_tp, dtype=np.int64)
    order = np.argsort(-scores)
    tp_cum = np.cumsum(tp[order])
    fp_cum = np.cumsum(1 - tp[order])
    recalls = tp_cum / float(total_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return average_precision(recalls, precisions)


# --------------------------------------------------------------------------- #
# Report container
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class EvaluationReport:
    """Metrics for a set of images (optionally grouped by lighting category)."""

    has_ground_truth: bool
    num_images: int = 0
    num_predictions: int = 0
    num_ground_truth: int = 0
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    false_positives: int | None = None
    false_negatives: int | None = None
    true_positives: int | None = None
    mean_inference_ms: float | None = None
    fps: float | None = None
    count_mae: float | None = None
    count_errors: list[int] = dataclasses.field(default_factory=list)
    per_image: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    by_category: dict[str, "EvaluationReport"] = dataclasses.field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (nested categories included)."""
        d = dataclasses.asdict(self)
        d["by_category"] = {k: v.to_dict() for k, v in self.by_category.items()}
        return d

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_flat_rows(self) -> list[dict[str, Any]]:
        """Flatten to one row per category (plus an ``overall`` row) for CSV export."""
        rows: list[dict[str, Any]] = [_flat_row("overall", self)]
        for cat, rep in sorted(self.by_category.items()):
            rows.append(_flat_row(cat, rep))
        return rows

    def metrics_table(self) -> list[dict[str, Any]]:
        """Compact list of dicts with the headline metrics per category."""
        rows = []
        for name, rep in [("overall", self), *sorted(self.by_category.items())]:
            rows.append(
                {
                    "lighting_category": name,
                    "images": rep.num_images,
                    "precision": rep.precision,
                    "recall": rep.recall,
                    "f1": rep.f1,
                    "mAP@0.5": rep.map50,
                    "mAP@0.5:0.95": rep.map50_95,
                    "count_MAE": rep.count_mae,
                    "mean_inference_ms": rep.mean_inference_ms,
                }
            )
        return rows

    @property
    def unavailable_reason(self) -> str | None:
        """Message explaining why accuracy metrics are unavailable, else ``None``."""
        if self.has_ground_truth:
            return None
        return (
            "Detection accuracy (precision, recall, mAP) is unavailable: no labeled "
            "ground-truth bounding boxes were provided for this run."
        )


def _flat_row(category: str, rep: EvaluationReport) -> dict[str, Any]:
    return {
        "lighting_category": category,
        "has_ground_truth": rep.has_ground_truth,
        "images": rep.num_images,
        "predictions": rep.num_predictions,
        "ground_truth": rep.num_ground_truth,
        "precision": rep.precision,
        "recall": rep.recall,
        "f1": rep.f1,
        "map50": rep.map50,
        "map50_95": rep.map50_95,
        "true_positives": rep.true_positives,
        "false_positives": rep.false_positives,
        "false_negatives": rep.false_negatives,
        "mean_inference_ms": rep.mean_inference_ms,
        "fps": rep.fps,
        "count_mae": rep.count_mae,
        "note": rep.note,
    }


# --------------------------------------------------------------------------- #
# Main evaluation entry point
# --------------------------------------------------------------------------- #

def evaluate_predictions(
    predictions: dict[str, Sequence[Detection]],
    ground_truth: dict[str, Sequence[GroundTruthBox]] | None = None,
    *,
    categories: dict[str, str] | None = None,
    inference_times_ms: dict[str, float] | None = None,
    _group_recursively: bool = True,
) -> EvaluationReport:
    """Evaluate detections against ground truth, globally and per lighting category.

    Parameters
    ----------
    predictions:
        ``filename -> detections`` produced by the detector.
    ground_truth:
        ``filename -> ground-truth boxes``.  Pass ``None`` (or omit entries) to
        run a statistics-only pass: the report is returned with
        ``has_ground_truth=False`` and accuracy metrics set to ``None``.
    categories:
        ``filename -> real-world lighting category`` for per-category breakdown.
    inference_times_ms:
        Optional ``filename -> milliseconds`` override; otherwise the mean
        per-detection inference time recorded by the detector is used.

    Notes
    -----
    - ``count_mae`` is the mean absolute difference between predicted and
      ground-truth object counts (needs ground truth).
    - The *test* split must never be used for training or threshold tuning.
    """
    if not predictions:
        raise EvaluationError("No predictions supplied — run the detector first.")

    ground_truth = ground_truth or {}
    categories = categories or {}
    inference_times_ms = inference_times_ms or {}

    names = sorted(predictions.keys())
    has_gt = bool(ground_truth) and all(n in ground_truth for n in names)

    per_image_rows: list[dict[str, Any]] = []
    total_pred = 0
    total_gt = 0
    times: list[float] = []
    count_errors: list[int] = []

    # Per-IoU-threshold accumulators for mAP.
    per_image_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    tp_sum = fp_sum = fn_sum = 0

    for name in names:
        dets = list(predictions[name])
        gts = list(ground_truth.get(name, []))
        total_pred += len(dets)
        total_gt += len(gts)

        p_boxes, p_scores = _detection_to_arrays(dets)
        g_boxes = (
            np.asarray([g.xyxy for g in gts], dtype=np.float32).reshape(-1, 4)
            if gts
            else np.zeros((0, 4), dtype=np.float32)
        )
        per_image_arrays.append((p_boxes, p_scores, g_boxes))

        if name in inference_times_ms:
            t = float(inference_times_ms[name])
        elif dets:
            t = float(np.mean([d.inference_time_ms for d in dets]))
        else:
            t = 0.0
        if t > 0:
            times.append(t)

        tp, fp, fn, _ = match_predictions(p_boxes, p_scores, g_boxes, IOU_MATCH_THRESHOLD)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        err = abs(len(dets) - len(gts)) if has_gt else 0
        if has_gt:
            count_errors.append(err)

        per_image_rows.append(
            {
                "source_name": name,
                "lighting_category": categories.get(name, "unlabeled"),
                "predicted_count": len(dets),
                "ground_truth_count": len(gts) if has_gt else None,
                "count_error": err if has_gt else None,
                "mean_confidence": round(float(np.mean(p_scores)), 4) if p_scores.size else None,
                "inference_time_ms": round(t, 3),
                "true_positives": tp if has_gt else None,
                "false_positives": fp if has_gt else None,
                "false_negatives": fn if has_gt else None,
            }
        )

    mean_ms = float(np.mean(times)) if times else None
    fps = (1000.0 / mean_ms) if mean_ms and mean_ms > 0 else None

    if not has_gt:
        note = (
            "Accuracy metrics unavailable: ground-truth boxes are missing for at least "
            "one input image. Only inference statistics are reported."
        )
        report = EvaluationReport(
            has_ground_truth=False,
            num_images=len(names),
            num_predictions=total_pred,
            num_ground_truth=0,
            mean_inference_ms=round(mean_ms, 3) if mean_ms else None,
            fps=round(fps, 2) if fps else None,
            per_image=per_image_rows,
            note=note,
        )
    else:
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
        ap50 = _ap_at_iou(per_image_arrays, 0.5)
        aps = [_ap_at_iou(per_image_arrays, t) for t in MAP_IOU_THRESHOLDS]
        map50_95 = float(np.mean(aps)) if aps else 0.0
        report = EvaluationReport(
            has_ground_truth=True,
            num_images=len(names),
            num_predictions=total_pred,
            num_ground_truth=total_gt,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            map50=round(ap50, 4),
            map50_95=round(map50_95, 4),
            false_positives=fp_sum,
            false_negatives=fn_sum,
            true_positives=tp_sum,
            mean_inference_ms=round(mean_ms, 3) if mean_ms else None,
            fps=round(fps, 2) if fps else None,
            count_mae=round(float(np.mean(count_errors)), 3) if count_errors else 0.0,
            count_errors=count_errors,
            per_image=per_image_rows,
            note="",
        )

    # Per-category breakdown (categories default to 'unlabeled').  Skipped for
    # the recursive per-category calls themselves to avoid infinite recursion.
    if _group_recursively:
        grouped: dict[str, dict[str, Detection]] = {}
        grouped_gt: dict[str, dict[str, GroundTruthBox]] = {}
        grouped_times: dict[str, dict[str, float]] = {}
        for name in names:
            cat = categories.get(name, "unlabeled")
            grouped.setdefault(cat, {})[name] = predictions[name]
            if name in ground_truth:
                grouped_gt.setdefault(cat, {})[name] = ground_truth[name]
            if name in inference_times_ms:
                grouped_times.setdefault(cat, {})[name] = inference_times_ms[name]

        for cat, preds in grouped.items():
            sub = evaluate_predictions(
                preds,
                grouped_gt.get(cat) if has_gt else None,
                categories={n: cat for n in preds},
                inference_times_ms=grouped_times.get(cat),
                _group_recursively=False,
            )
            report.by_category[cat] = sub

    return report


def summarize_count_error(
    predicted_counts: Sequence[int], ground_truth_counts: Sequence[int]
) -> dict[str, float | int]:
    """Return MAE, RMSE, mean bias and exact-match rate between two count vectors."""
    if len(predicted_counts) != len(ground_truth_counts):
        raise EvaluationError(
            f"Count vectors differ in length: {len(predicted_counts)} vs {len(ground_truth_counts)}"
        )
    if not predicted_counts:
        return {"n_images": 0, "mae": 0.0, "rmse": 0.0, "mean_bias": 0.0, "exact_match_rate": 0.0}
    p = np.asarray(predicted_counts, dtype=np.float64)
    g = np.asarray(ground_truth_counts, dtype=np.float64)
    diff = p - g
    abs_diff = np.abs(diff)
    return {
        "n_images": int(p.size),
        "mae": round(float(abs_diff.mean()), 4),
        "rmse": round(float(np.sqrt((diff**2).mean())), 4),
        "mean_bias": round(float(diff.mean()), 4),
        "exact_match_rate": round(float((abs_diff < 0.5).mean()), 4),
        "total_absolute_error": int(abs_diff.sum()),
    }


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #

def export_report_csv(report: EvaluationReport, path: str | Path) -> Path:
    """Write the per-category metric table to a CSV file."""
    import pandas as pd

    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(report.to_flat_rows())
    df.to_csv(p, index=False)
    return p


def export_report_json(report: EvaluationReport, path: str | Path) -> Path:
    """Write the full report (including per-image rows) to JSON."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report.to_json(indent=2), encoding="utf-8")
    return p


def export_per_image_csv(report: EvaluationReport, path: str | Path) -> Path:
    """Write per-image statistics (counts, confidence, timing) to CSV."""
    import pandas as pd

    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report.per_image).to_csv(p, index=False)
    return p
