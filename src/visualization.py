"""Drawing helpers and evaluation charts.

Charts are generated with the non-interactive ``Agg`` backend so they work in
headless batch jobs; the Streamlit app renders the returned figures directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from .lighting_analysis import describe_category  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from .detector import DetectionResult
    from .evaluation import EvaluationReport

logger = logging.getLogger(__name__)

#: Box colors (BGR) keyed by class name, with a default for unknown classes.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "tree_trunk": (60, 170, 60),
    "tree": (60, 170, 60),
    "default": (40, 190, 240),
}


def _color_for(class_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(class_name.lower(), CLASS_COLORS["default"])


def draw_detections(
    image: np.ndarray,
    result: "DetectionResult | Sequence[Any]",
    *,
    draw_count: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Draw labelled bounding boxes on *image* and return a new annotated copy.

    Accepts a :class:`~src.detector.DetectionResult` or a plain sequence of
    :class:`~src.detector.Detection`.  An empty detection list returns an
    untouched copy (never ``None``).
    """
    from .detector import Detection, DetectionResult
    from .preprocessing import ensure_bgr

    canvas = ensure_bgr(image).copy()
    detections: list[Detection]
    if isinstance(result, DetectionResult):
        detections = list(result.detections)
        source = result
    elif isinstance(result, (list, tuple)):
        detections = [d for d in result if isinstance(d, Detection)]
        source = None
    else:  # pragma: no cover - defensive
        raise TypeError(f"Unsupported detection container: {type(result)!r}")

    label_size_px = max(12, int(round(canvas.shape[0] / 45.0)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for det in detections:
        p1 = (int(round(det.x1)), int(round(det.y1)))
        p2 = (int(round(det.x2)), int(round(det.y2)))
        color = _color_for(det.class_name)
        cv2.rectangle(canvas, p1, p2, color, thickness)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, font, 0.55, 1)
        ty = p1[1] - 4 if p1[1] - th - 8 > 0 else p1[1] + th + 8
        cv2.rectangle(
            canvas,
            (p1[0], ty - th - 4),
            (p1[0] + tw + 4, ty + baseline),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (p1[0] + 2, ty - 1),
            font,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if draw_count:
        count = len(detections)
        text = f"trees: {count}" if source is None else f"trees: {count}"
        if source is not None and not source.is_tree_model:
            text = f"objects: {count} (baseline model)"
        (tw, th), baseline = cv2.getTextSize(text, font, 0.8, 2)
        cv2.rectangle(canvas, (8, 8), (8 + tw + 12, 8 + th + baseline + 10), (30, 30, 30), -1)
        cv2.putText(
            canvas,
            text,
            (14, 8 + th + 4),
            font,
            0.8,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
    return canvas


def side_by_side(original: np.ndarray, annotated: np.ndarray) -> np.ndarray:
    """Stack *original* and *annotated* images horizontally (resized to equal height)."""
    from .preprocessing import ensure_bgr

    a = ensure_bgr(original)
    b = ensure_bgr(annotated)
    if a.shape[0] != b.shape[0]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return np.hstack([a, b])


def save_annotated(image: np.ndarray, path: str | Path) -> Path:
    """Write an annotated image to disk, creating parent directories."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(p), image):
        raise IOError(f"Could not write annotated image to {p}")
    return p


def _clean_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def _category_labels(report: "EvaluationReport") -> list[str]:
    return [describe_category(k) for k in report.by_category]


def _metric_series(report: "EvaluationReport", attr: str) -> list[float | None]:
    return [getattr(v, attr) for v in report.by_category.values()]


def plot_precision_recall_by_category(report: "EvaluationReport") -> plt.Figure:
    """Bar chart of precision and recall per lighting category."""
    cats = _category_labels(report)
    precision = [v if v is not None else 0.0 for v in _metric_series(report, "precision")]
    recall = [v if v is not None else 0.0 for v in _metric_series(report, "recall")]
    x = np.arange(len(cats))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, precision, width, label="Precision", color="#2a9d8f")
    ax.bar(x + width / 2, recall, width, label="Recall", color="#e76f51")
    ax.set_xticks(x, cats, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Precision and recall by lighting condition")
    ax.legend()
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_map_by_category(report: "EvaluationReport") -> plt.Figure:
    """Bar chart of mAP@0.5 and mAP@0.5:0.95 per lighting category."""
    cats = _category_labels(report)
    m50 = [v if v is not None else 0.0 for v in _metric_series(report, "map50")]
    m95 = [v if v is not None else 0.0 for v in _metric_series(report, "map50_95")]
    x = np.arange(len(cats))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, m50, width, label="mAP@0.5", color="#264653")
    ax.bar(x + width / 2, m95, width, label="mAP@0.5:0.95", color="#e9c46a")
    ax.set_xticks(x, cats, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mAP")
    ax.set_title("Average precision by lighting condition")
    ax.legend()
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_inference_time_by_category(report: "EvaluationReport") -> plt.Figure:
    """Bar chart of mean inference time (ms) per lighting category."""
    cats = _category_labels(report)
    times = [v if v is not None else 0.0 for v in _metric_series(report, "mean_inference_ms")]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(cats, times, color="#457b9d")
    ax.set_ylabel("Mean inference time (ms)")
    ax.set_title("Mean inference time by lighting condition")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_count_error_by_category(report: "EvaluationReport") -> plt.Figure:
    """Bar chart of tree-count mean absolute error per lighting category."""
    cats = _category_labels(report)
    mae = [v if v is not None else 0.0 for v in _metric_series(report, "count_mae")]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(cats, mae, color="#bc4749")
    ax.set_ylabel("Count MAE (trees)")
    ax.set_title("Tree-count error by lighting condition")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_confidence_distribution(
    confidences: Sequence[float],
    title: str = "Confidence-score distribution",
) -> plt.Figure:
    """Histogram of detection confidences; empty input yields an explanatory figure."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = [float(c) for c in confidences]
    if values:
        ax.hist(values, bins=20, range=(0.0, 1.0), color="#2a9d8f", edgecolor="white")
        ax.axvline(np.mean(values), color="#e76f51", linestyle="--", label=f"mean = {np.mean(values):.3f}")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No detections above threshold\n(no confidence scores to show)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#555555",
        )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(title)
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_confidence_by_category(per_image_rows: Sequence[dict[str, Any]]) -> plt.Figure:
    """Box plot of per-image mean confidence grouped by lighting category."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    grouped: dict[str, list[float]] = {}
    for row in per_image_rows:
        mc = row.get("mean_confidence")
        if mc is None:
            continue
        grouped.setdefault(describe_category(str(row.get("lighting_category", "unlabeled"))), []).append(
            float(mc)
        )
    if not grouped:
        ax.text(
            0.5,
            0.5,
            "No confidence statistics available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#555555",
        )
    else:
        labels = list(grouped)
        ax.boxplot([grouped[k] for k in labels], tick_labels=labels)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_ylabel("Mean confidence per image")
    ax.set_title("Confidence scores by lighting condition")
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 120) -> Path:
    """Save a matplotlib figure to *path* and close it."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart %s", p)
    return p


def generate_all_charts(report: "EvaluationReport", output_dir: str | Path) -> list[Path]:
    """Generate the standard comparison chart set for *report*.

    Charts that need ground truth are skipped (with a log message) when
    metrics are unavailable.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    if not report.by_category:
        logger.info("No lighting categories in report — skipping category charts.")
        return saved

    if report.has_ground_truth:
        saved.append(save_figure(plot_precision_recall_by_category(report), out / "precision_recall_by_lighting.png"))
        saved.append(save_figure(plot_map_by_category(report), out / "map_by_lighting.png"))
        saved.append(save_figure(plot_count_error_by_category(report), out / "count_error_by_lighting.png"))
    else:
        logger.info("Ground truth unavailable — skipping precision/recall/mAP/count charts.")

    saved.append(save_figure(plot_inference_time_by_category(report), out / "inference_time_by_lighting.png"))
    all_conf = [
        c
        for row in report.per_image
        for c in ([row["mean_confidence"]] if row.get("mean_confidence") is not None else [])
    ]
    saved.append(save_figure(plot_confidence_distribution(all_conf), out / "confidence_distribution.png"))
    saved.append(save_figure(plot_confidence_by_category(report.per_image), out / "confidence_by_lighting.png"))
    return saved
