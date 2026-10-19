#!/usr/bin/env python3
"""Batch tree-trunk detection over images, folders or videos.

Examples
--------
    python scripts/batch_detect.py --source data/images --output outputs/batch
    python scripts/batch_detect.py --source forest.mp4 --output outputs/video \
        --weights models/tree_trunk.pt --conf 0.4
    python scripts/batch_detect.py --source data/images --output outputs/base \
        --weights yolov8n.pt --categories lighting_labels.json

Outputs written to ``--output``:

- ``annotated/``            annotated images / video
- ``detections.json``       full machine-readable records (per image & per detection)
- ``detections.csv``        flat per-detection rows
- ``summary.json``          run summary (counts, timings, warnings)

Exit code 0 = success, 1 = no processable input, 2 = weights/backend problem.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import (  # noqa: E402
    DetectionResult,
    DetectorError,
    DetectorUnavailableError,
    TreeDetector,
    WeightsNotFoundError,
)
from src.lighting_analysis import analyze_lighting, load_category_labels  # noqa: E402
from src.preprocessing import (  # noqa: E402
    InputValidationError,
    is_video_path,
    list_images_in_folder,
    load_image,
)
from src.video_processor import process_folder, process_video  # noqa: E402
from src.visualization import draw_detections  # noqa: E402

logger = logging.getLogger("batch_detect")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run tree-trunk detection over images, a folder or a video.")
    p.add_argument("--source", required=True, help="Image file, folder, or video file.")
    p.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "batch"), help="Output directory.")
    p.add_argument("--weights", default=None, help="Checkpoint path (default: first .pt in models/).")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold [0, 1].")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold [0, 1].")
    p.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    p.add_argument("--device", default=None, help="cpu / cuda / cuda:0 (default: auto).")
    p.add_argument("--max-frames", type=int, default=None, help="Limit video frames (debug).")
    p.add_argument("--categories", default=None, help="JSON file: {filename: lighting_category}.")
    p.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    p.add_argument("--save-annotated", action="store_true", default=True, help="Save annotated copies (default on).")
    p.add_argument("--no-save-annotated", dest="save_annotated", action="store_false")
    p.add_argument("--log-level", default="INFO", help="Logging level.")
    return p.parse_args(argv)


def _write_outputs(
    out_dir: Path,
    results: list[DetectionResult],
    categories: dict[str, str],
    lighting: dict[str, Any],
    extra_summary: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    rows: list[dict[str, Any]] = []
    for res in results:
        stats = lighting.get(res.source_name)
        rec = res.to_dict()
        rec["lighting_category"] = categories.get(res.source_name, "unlabeled")
        rec["lighting_stats"] = stats.to_dict() if stats is not None else None
        rec["estimated_lighting"] = stats.estimated_category if stats is not None else None
        records.append(rec)
        for det in res.detections:
            row = det.to_dict()
            row["lighting_category"] = rec["lighting_category"]
            row["image_count"] = res.count
            rows.append(row)

    (out_dir / "detections.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if rows:
        fieldnames = list(rows[0].keys())
        with (out_dir / "detections.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        (out_dir / "detections.csv").write_text("", encoding="utf-8")

    summary = {
        "images_processed": len(results),
        "total_detections": sum(r.count for r in results),
        "mean_detections_per_image": (
            round(sum(r.count for r in results) / len(results), 3) if results else 0.0
        ),
        "mean_inference_ms": (
            round(sum(r.inference_time_ms for r in results) / len(results), 3) if results else None
        ),
        "confidence_threshold": extra_summary.get("confidence_threshold"),
        "weights": extra_summary.get("weights"),
        "is_tree_model": extra_summary.get("is_tree_model"),
        "warnings": extra_summary.get("warnings", []),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s")

    source = Path(args.source).expanduser()
    warnings: list[str] = []

    # --- detector -------------------------------------------------------- #
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
        print("Place a trained checkpoint in models/ or pass --weights /path/to/best.pt")
        return 2
    except (DetectorUnavailableError, DetectorError) as exc:
        print(f"ERROR: {exc}")
        return 2

    if not detector.is_tree_model:
        msg = (
            f"Weights '{detector.model_label}' expose no tree-trunk class "
            f"(classes: {detector.class_names[:8]}). Running a generic baseline — results are NOT "
            "tree-trunk detections. Train a custom model for real results."
        )
        warnings.append(msg)
        logger.warning(msg)

    categories: dict[str, str] = {}
    if args.categories:
        try:
            categories = load_category_labels(args.categories)
        except (OSError, ValueError) as exc:
            logger.warning("Could not load lighting categories: %s", exc)
            warnings.append(str(exc))

    out_dir = Path(args.output).expanduser()
    annotated_dir = out_dir / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)
    lighting_stats: dict[str, Any] = {}

    def _record_lighting(path: Path, image: "np.ndarray") -> None:
        lighting_stats[path.name] = analyze_lighting(image)

    if not source.exists():
        print(f"ERROR: source not found: {source}")
        return 1

    try:
        if source.is_dir():
            results, _ = process_folder(
                detector, source, annotated_dir, recursive=args.recursive,
                image_hook=_record_lighting,
            )
            if not results:
                print(f"ERROR: no readable images in {source}")
                return 1
        elif is_video_path(source):
            video_out = annotated_dir / f"{source.stem}_annotated.mp4"
            summary = process_video(
                detector, source, video_out, max_frames=args.max_frames
            )
            (out_dir / "video_summary.json").write_text(
                json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
            )
            print(json.dumps(summary.to_dict(), indent=2))
            return 0
        else:
            image = load_image(source)
            lighting_stats[source.name] = analyze_lighting(image)
            result = detector.detect(image, source_name=source.name)
            results = [result]
            if args.save_annotated:
                annotated_dir.mkdir(parents=True, exist_ok=True)
                from cv2 import imwrite  # local alias

                imwrite(str(annotated_dir / source.name), draw_detections(image, result))
    except InputValidationError as exc:
        print(f"ERROR: {exc}")
        return 1
    except DetectorError as exc:
        print(f"ERROR: {exc}")
        return 2

    _write_outputs(
        out_dir,
        results,
        categories,
        lighting_stats,
        {
            "confidence_threshold": args.conf,
            "weights": str(detector.weights_path),
            "is_tree_model": detector.is_tree_model,
            "warnings": warnings,
        },
    )
    print(
        f"Processed {len(results)} image(s), {sum(r.count for r in results)} detection(s) "
        f"-> {out_dir}"
    )
    if not detector.is_tree_model:
        print("NOTE: baseline model without a tree class — see warnings above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
