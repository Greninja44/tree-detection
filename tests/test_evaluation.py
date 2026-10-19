"""Tests for metric calculations (precision, recall, AP, mAP, count error)."""

from __future__ import annotations

import numpy as np
import pytest

from src.detector import Detection
from src.evaluation import (
    EvaluationError,
    EvaluationReport,
    average_precision,
    evaluate_predictions,
    export_report_csv,
    export_report_json,
    match_predictions,
    parse_yolo_label_lines,
    summarize_count_error,
)


def _det(x1, y1, x2, y2, conf=0.9, name="img.jpg") -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf,
                     class_id=0, class_name="tree_trunk", source_name=name)


def _gt(x1, y1, x2, y2):
    from src.evaluation import GroundTruthBox

    return GroundTruthBox(x1=x1, y1=y1, x2=x2, y2=y2)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

def test_match_perfect_predictions() -> None:
    pred = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    tp, fp, fn, matches = match_predictions(pred, scores, gt, 0.5)
    assert (tp, fp, fn) == (2, 0, 0)
    assert len(matches) == 2


def test_match_no_predictions() -> None:
    tp, fp, fn, matches = match_predictions(
        np.zeros((0, 4)), np.zeros((0,)), np.array([[0, 0, 5, 5]], dtype=np.float32)
    )
    assert (tp, fp, fn) == (0, 0, 1)
    assert matches == []


def test_match_no_ground_truth() -> None:
    tp, fp, fn, _ = match_predictions(
        np.array([[0, 0, 5, 5]], dtype=np.float32), np.array([0.9]), np.zeros((0, 4))
    )
    assert (tp, fp, fn) == (0, 1, 0)


def test_match_duplicates_are_false_positives() -> None:
    pred = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
    tp, fp, fn, _ = match_predictions(pred, scores, gt, 0.5)
    assert (tp, fp, fn) == (1, 1, 0)


def test_match_low_iou_counts_as_miss() -> None:
    pred = np.array([[0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.95], dtype=np.float32)
    gt = np.array([[50, 50, 60, 60]], dtype=np.float32)
    tp, fp, fn, _ = match_predictions(pred, scores, gt, 0.5)
    assert (tp, fp, fn) == (0, 1, 1)


# --------------------------------------------------------------------------- #
# Average precision
# --------------------------------------------------------------------------- #

def test_average_precision_perfect_curve() -> None:
    r = np.array([0.5, 1.0])
    p = np.array([1.0, 1.0])
    assert average_precision(r, p) == pytest.approx(1.0)


def test_average_precision_empty() -> None:
    assert average_precision(np.array([]), np.array([])) == 0.0


def test_average_precision_partial() -> None:
    # 1 TP out of 2 gt at the top rank, then 1 FP -> PR drops.
    r = np.array([0.5, 0.5])
    p = np.array([1.0, 0.5])
    assert 0.4 < average_precision(r, p) <= 0.5 + 1e-9


# --------------------------------------------------------------------------- #
# End-to-end report
# --------------------------------------------------------------------------- #

def test_perfect_evaluation() -> None:
    preds = {"a.jpg": [_det(0, 0, 10, 10, 0.95)], "b.jpg": [_det(5, 5, 15, 15, 0.85)]}
    gts = {"a.jpg": [_gt(0, 0, 10, 10)], "b.jpg": [_gt(5, 5, 15, 15)]}
    rep = evaluate_predictions(preds, gts)
    assert rep.has_ground_truth
    assert rep.precision == pytest.approx(1.0)
    assert rep.recall == pytest.approx(1.0)
    assert rep.f1 == pytest.approx(1.0)
    assert rep.map50 == pytest.approx(1.0)
    assert rep.map50_95 == pytest.approx(1.0)
    assert rep.false_positives == 0 and rep.false_negatives == 0
    assert rep.count_mae == pytest.approx(0.0)
    assert rep.count_errors == [0, 0]


def test_wrong_counts_and_misses() -> None:
    preds = {"a.jpg": [_det(0, 0, 10, 10, 0.9), _det(100, 100, 120, 120, 0.8)]}
    gts = {"a.jpg": [_gt(0, 0, 10, 10), _gt(200, 200, 220, 220)]}
    rep = evaluate_predictions(preds, gts)
    assert rep.precision == pytest.approx(0.5)
    assert rep.recall == pytest.approx(0.5)
    assert rep.f1 == pytest.approx(0.5)
    assert rep.false_positives == 1
    assert rep.false_negatives == 1
    # counts match (2 pred, 2 gt) -> zero count error despite bad localization
    assert rep.count_mae == pytest.approx(0.0)


def test_count_mae_across_images() -> None:
    preds = {"a.jpg": [_det(0, 0, 10, 10)], "b.jpg": [_det(0, 0, 10, 10), _det(20, 20, 30, 30)]}
    gts = {"a.jpg": [_gt(0, 0, 10, 10)], "b.jpg": [_gt(0, 0, 10, 10)]}
    rep = evaluate_predictions(preds, gts)
    assert rep.count_mae == pytest.approx(0.5)  # |1-1| + |2-1| / 2


def test_report_without_ground_truth_marks_metrics_unavailable() -> None:
    preds = {"a.jpg": [_det(0, 0, 10, 10, 0.9)]}
    rep = evaluate_predictions(preds, None, inference_times_ms={"a.jpg": 12.5})
    assert not rep.has_ground_truth
    for field in ("precision", "recall", "f1", "map50", "map50_95",
                  "false_positives", "false_negatives", "count_mae"):
        assert getattr(rep, field) is None, field
    assert rep.unavailable_reason is not None
    assert "unavailable" in rep.unavailable_reason.lower()
    # Statistics that do not need labels are still reported.
    assert rep.num_images == 1
    assert rep.num_predictions == 1
    assert rep.mean_inference_ms == pytest.approx(12.5)


def test_inference_time_taken_from_detections_when_not_overridden() -> None:
    det = Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.9, class_id=0,
                    class_name="tree_trunk", source_name="a.jpg", inference_time_ms=8.0)
    rep = evaluate_predictions({"a.jpg": [det]}, None)
    assert rep.mean_inference_ms == pytest.approx(8.0)
    assert rep.fps == pytest.approx(125.0)


def test_report_with_missing_gt_entry_degrades_safely() -> None:
    preds = {"a.jpg": [_det(0, 0, 10, 10)], "b.jpg": [_det(0, 0, 10, 10)]}
    gts = {"a.jpg": [_gt(0, 0, 10, 10)]}  # 'b.jpg' label missing
    rep = evaluate_predictions(preds, gts)
    assert not rep.has_ground_truth
    assert rep.precision is None


def test_grouping_by_lighting_category() -> None:
    preds = {
        "sun.jpg": [_det(0, 0, 10, 10, 0.9)],
        "shade.jpg": [_det(0, 0, 10, 10, 0.9), _det(30, 30, 40, 40, 0.7)],
    }
    gts = {
        "sun.jpg": [_gt(0, 0, 10, 10)],
        "shade.jpg": [_gt(0, 0, 10, 10)],
    }
    cats = {"sun.jpg": "bright_daylight", "shade.jpg": "deep_shadow"}
    rep = evaluate_predictions(preds, gts, categories=cats)
    assert set(rep.by_category) == {"bright_daylight", "deep_shadow"}
    assert rep.by_category["bright_daylight"].precision == pytest.approx(1.0)
    assert rep.by_category["deep_shadow"].recall == pytest.approx(1.0)
    assert rep.by_category["deep_shadow"].false_positives == 1
    assert rep.by_category["bright_daylight"].num_images == 1
    # Per-image rows carry the assigned category.
    rows = {r["source_name"]: r for r in rep.per_image}
    assert rows["shade.jpg"]["lighting_category"] == "deep_shadow"


def test_grouping_without_labels_uses_unlabeled() -> None:
    preds = {"a.jpg": [_det(0, 0, 10, 10)]}
    gts = {"a.jpg": [_gt(0, 0, 10, 10)]}
    rep = evaluate_predictions(preds, gts, categories={"a.jpg": "overcast"})
    assert "overcast" in rep.by_category


def test_empty_predictions_raise() -> None:
    with pytest.raises(EvaluationError):
        evaluate_predictions({})


def test_summarize_count_error() -> None:
    s = summarize_count_error([3, 1, 2], [2, 1, 4])
    assert s["n_images"] == 3
    assert s["mae"] == pytest.approx((1 + 0 + 2) / 3)
    assert s["total_absolute_error"] == 3
    assert s["exact_match_rate"] == pytest.approx(1 / 3, abs=1e-3)
    with pytest.raises(EvaluationError):
        summarize_count_error([1], [1, 2])


def test_report_export(tmp_path) -> None:
    preds = {"a.jpg": [_det(0, 0, 10, 10, 0.9)]}
    gts = {"a.jpg": [_gt(0, 0, 10, 10)]}
    rep = evaluate_predictions(preds, gts, categories={"a.jpg": "overcast"})
    csv_path = export_report_csv(rep, tmp_path / "m.csv")
    json_path = export_report_json(rep, tmp_path / "m.json")
    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert json_path.exists()
    import json

    payload = json.loads(json_path.read_text())
    assert payload["has_ground_truth"] is True
    assert "by_category" in payload
    rows = rep.to_flat_rows()
    assert rows[0]["lighting_category"] == "overall"
    assert any(r["lighting_category"] == "overcast" for r in rows)


# --------------------------------------------------------------------------- #
# YOLO label parsing
# --------------------------------------------------------------------------- #

def test_parse_yolo_labels_normalizes_coordinates() -> None:
    lines = ["0 0.5 0.5 0.2 0.4"]
    boxes = parse_yolo_label_lines(lines, image_width=1000, image_height=500)
    assert len(boxes) == 1
    b = boxes[0]
    assert b.x1 == pytest.approx(400.0)
    assert b.x2 == pytest.approx(600.0)
    assert b.y1 == pytest.approx(150.0)
    assert b.y2 == pytest.approx(350.0)


def test_parse_yolo_labels_skips_malformed() -> None:
    lines = ["not a label", "0 0.5 0.5", "0 0.5 0.5 0.1 0.1"]
    boxes = parse_yolo_label_lines(lines, 100, 100)
    assert len(boxes) == 1


def test_parse_yolo_labels_invalid_size() -> None:
    with pytest.raises(EvaluationError):
        parse_yolo_label_lines(["0 0.5 0.5 0.1 0.1"], 0, 100)


def test_inference_timing_aggregates() -> None:
    preds = {"a.jpg": [_det(0, 0, 10, 10, 0.9)]}
    rep = evaluate_predictions(preds, None, inference_times_ms={"a.jpg": 20.0})
    assert rep.mean_inference_ms == pytest.approx(20.0)
    assert rep.fps == pytest.approx(50.0)


def test_report_json_serializable() -> None:
    import json

    preds = {"a.jpg": [_det(0, 0, 10, 10)]}
    gts = {"a.jpg": [_gt(0, 0, 10, 10)]}
    rep = evaluate_predictions(preds, gts, categories={"a.jpg": "overcast"})
    payload = json.loads(rep.to_json())
    assert payload["by_category"]["overcast"]["precision"] == pytest.approx(1.0)
    assert isinstance(rep.metrics_table(), list)
