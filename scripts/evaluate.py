#!/usr/bin/env python3
"""Evaluate a tree-trunk detector on an image split with optional ground truth.

Examples
--------
Statistics-only pass (no labels — accuracy is reported as *unavailable*)::

    python scripts/evaluate.py --weights models/tree_trunk.pt --images data/images

Full evaluation against YOLO label files, grouped by assigned lighting labels::

    python scripts/evaluate.py --weights runs/train/tree_trunk_exp/weights/best.pt \
        --images data/images_test --labels data/labels_test \
        --categories data/lighting_labels.json --output reports/test

Notes
-----
- YOLO labels are ``labels/<stem>.txt`` with lines ``class cx cy w h`` (normalized).
- The *test* split must be used only for final evaluation, never for training
  or threshold tuning.
- Metrics are never invented: without ground truth the accuracy fields are
  written as ``null`` with an explanatory ``note``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import (  # noqa: E402
    DetectorError,
    DetectorUnavailableError,
    TreeDetector,
    WeightsNotFoundError,
)
from src.evaluation import (  # noqa: E402
    EvaluationReport,
    export_per_image_csv,
    export_report_csv,
    export_report_json,
    evaluate_predictions,
    load_yolo_label_file,
)
from src.lighting_analysis import (  # noqa: E402
    analyze_lighting,
    load_category_labels,
)
from src.preprocessing import (  # noqa: E402
    InputValidationError,
    list_images_in_folder,
    load_image,
    validate_image_path,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a detector with/without ground-truth labels.")
    p.add_argument("--weights", required=True, help="Checkpoint to evaluate (e.g. runs/train/exp/weights/best.pt).")
    p.add_argument("--images", required=True, help="Image file or folder (the test split).")
    p.add_argument("--labels", default=None, help="Folder with YOLO .txt labels (optional but required for accuracy).")
    p.add_argument("--categories", default=None, help="JSON {filename: real_world_lighting_category}.")
    p.add_argument("--group-by", choices=("assigned", "estimated"), default=None,
                   help="Group metrics by assigned labels (default when --categories given) or image-based estimates.")
    p.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "eval"), help="Report output directory.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    p.add_argument("--device", default=None, help="cpu / cuda / cuda:0 (default: auto).")
    p.add_argument("--no-charts", action="store_true", help="Skip PNG chart generation.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def collect_inputs(images_arg: str) -> list[Path]:
    p = Path(images_arg).expanduser()
    if p.is_dir():
        return list_images_in_folder(p)
    return [validate_image_path(p)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("evaluate")

    try:
        image_paths = collect_inputs(args.images)
    except InputValidationError as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        detector = TreeDetector(
            weights=args.weights,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            imgsz=args.imgsz,
        )
    except WeightsNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2
    except (DetectorUnavailableError, DetectorError) as exc:
        print(f"ERROR: {exc}")
        return 2

    group_by = args.group_by
    if group_by is None:
        group_by = "assigned" if args.categories else "estimated"

    assigned: dict[str, str] = {}
    if args.categories:
        try:
            assigned = load_category_labels(args.categories)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1

    predictions: dict[str, list] = {}
    ground_truth: dict[str, list] = {}
    categories: dict[str, str] = {}
    lighting_stats: dict[str, dict] = {}
    inference_times: dict[str, float] = {}
    warnings: list[str] = []

    labels_dir = Path(args.labels).expanduser() if args.labels else None
    if labels_dir and not labels_dir.exists():
        print(f"ERROR: labels folder not found: {labels_dir}")
        return 1

    start = time.perf_counter()
    for path in image_paths:
        try:
            image = load_image(path)
        except InputValidationError as exc:
            log.warning("Skipping %s: %s", path.name, exc)
            continue
        result = detector.detect(image, source_name=path.name)
        predictions[path.name] = list(result.detections)
        inference_times[path.name] = result.inference_time_ms
        stats = analyze_lighting(image)
        lighting_stats[path.name] = stats.to_dict()

        if group_by == "assigned":
            categories[path.name] = assigned.get(path.name, "unlabeled")
        else:
            categories[path.name] = stats.estimated_category

        if labels_dir is not None:
            label_file = (labels_dir / path.stem).with_suffix(".txt")
            ground_truth[path.name] = load_yolo_label_file(
                label_file, image.shape[1], image.shape[0]
            )
            if not label_file.exists():
                log.warning("No label file for %s (treated as 0 ground-truth boxes).", path.name)
    elapsed = time.perf_counter() - start

    if not predictions:
        print("ERROR: no images could be processed.")
        return 1

    has_any_gt = bool(ground_truth) and len(ground_truth) == len(predictions)
    report: EvaluationReport = evaluate_predictions(
        predictions,
        ground_truth if has_any_gt else None,
        categories=categories,
        inference_times_ms=inference_times,
    )
    if group_by == "estimated":
        note = (
            "Lighting groups are IMAGE-BASED ESTIMATES derived from brightness/contrast "
            "statistics, not verified real-world conditions. Supply --categories with "
            "human-assigned labels for reporting by real-world lighting."
        )
        report.note = (report.note + " " + note).strip()

    out = Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    export_report_json(report, out / "metrics.json")
    export_report_csv(report, out / "metrics.csv")
    export_per_image_csv(report, out / "per_image.csv")
    (out / "detections.json").write_text(
        json.dumps(
            {name: [d.to_dict() for d in dets] for name, dets in predictions.items()}, indent=2
        ),
        encoding="utf-8",
    )
    (out / "lighting_stats.json").write_text(json.dumps(lighting_stats, indent=2), encoding="utf-8")

    chart_paths: list[Path] = []
    if not args.no_charts:
        try:
            from src.visualization import generate_all_charts

            chart_paths = generate_all_charts(report, out / "charts")
        except Exception as exc:  # charts must never fail the evaluation
            log.warning("Chart generation failed: %s", exc)

    # ---- console summary ------------------------------------------------ #
    print(f"\nEvaluated {report.num_images} image(s) with '{detector.model_label}' "
          f"in {elapsed:.1f}s (device={detector.backend_device}).")
    if not detector.is_tree_model:
        print("WARNING: checkpoint has no tree-trunk class — this is a generic baseline run.")
    if report.has_ground_truth:
        print(f"  precision      : {report.precision}")
        print(f"  recall         : {report.recall}")
        print(f"  F1             : {report.f1}")
        print(f"  mAP@0.5        : {report.map50}")
        print(f"  mAP@0.5:0.95   : {report.map50_95}")
        print(f"  false positives: {report.false_positives}  false negatives: {report.false_negatives}")
        print(f"  count MAE      : {report.count_mae}")
    else:
        print("  Accuracy metrics: UNAVAILABLE (no ground-truth boxes supplied).")
        print(f"  {report.unavailable_reason}")
    print(f"  mean inference : {report.mean_inference_ms} ms  ({report.fps} FPS)")
    for row in report.metrics_table():
        ms = row["mean_inference_ms"]
        ms_txt = f"{ms:.3f}" if isinstance(ms, (int, float)) else "n/a"
        print(f"  [{row['lighting_category']}] images={row['images']} "
              f"P={row['precision']} R={row['recall']} mAP50={row['mAP@0.5']} "
              f"count_MAE={row['count_MAE']} t={ms_txt} ms")
    print(f"\nReports written to {out}")
    for p in chart_paths:
        print(f"  chart: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
