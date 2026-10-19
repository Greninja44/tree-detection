#!/usr/bin/env python3
"""Train a custom YOLO tree-trunk detector.

Examples
--------
Validate the dataset only (no training)::

    python scripts/train.py --validate-only

Train from a pretrained lightweight backbone::

    python scripts/train.py --model yolov8n.pt --data configs/dataset.yaml \
        --epochs 80 --imgsz 640 --batch 16

Resume an interrupted run::

    python scripts/train.py --resume runs/train/tree_trunk_exp/weights/last.pt

This script only reports metrics that Ultralytics actually computes.  It never
writes results that were not produced by a real training run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import SUPPORTED_IMAGE_EXTENSIONS  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a YOLO tree-trunk detector.")
    p.add_argument("--model", default="yolov8n.pt", help="Base weights (.pt path or model name, e.g. yolov8n.pt).")
    p.add_argument("--data", default=str(PROJECT_ROOT / "configs" / "dataset.yaml"), help="YOLO dataset yaml.")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference/training image size.")
    p.add_argument("--batch", default=16, help="Batch size (int or 'auto').")
    p.add_argument("--device", default=None, help="cpu, 0, 0,1 ... (default: auto).")
    p.add_argument("--workers", type=int, default=4, help="Data-loader workers.")
    p.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "train"), help="Output project dir.")
    p.add_argument("--name", default="tree_trunk_exp", help="Run name.")
    p.add_argument("--resume", nargs="?", const=True, default=None, help="Resume from a checkpoint path (or last run).")
    p.add_argument("--patience", type=int, default=30, help="Early-stopping patience (epochs).")
    p.add_argument("--optimizer", default="auto", help="Optimizer name or 'auto'.")
    p.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--validate-only", action="store_true", help="Only validate the dataset structure, do not train.")
    return p.parse_args(argv)


def validate_dataset_yaml(data_yaml: Path) -> list[str]:
    """Return a list of human-readable problems with the dataset config.

    An empty list means the config is structurally valid.  Checked items:
    file exists, ``train``/``val`` keys, ``names`` present, split folders exist,
    label files well-formed and paired with images.
    """
    problems: list[str] = []
    if not data_yaml.exists():
        return [f"Dataset config not found: {data_yaml}"]
    try:
        import yaml
    except ImportError:
        return ["PyYAML is not installed — cannot read the dataset config."]

    try:
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Could not parse {data_yaml}: {exc}"]
    if not isinstance(cfg, dict):
        return [f"{data_yaml} must contain a YAML mapping."]

    for key in ("train", "val"):
        if key not in cfg:
            problems.append(f"Missing required key '{key}' in {data_yaml.name}.")
    names = cfg.get("names")
    if not names:
        problems.append("Missing 'names' (class list) in the dataset config.")
    elif isinstance(names, dict):
        class_names = [str(v) for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
    else:
        class_names = [str(v) for v in names]
    if not any("tree" in n.lower() or "trunk" in n.lower() for n in class_names):
        problems.append(
            f"Class list {class_names} contains no tree/trunk class — this config will not train a tree detector."
        )

    # Validate split folders + label pairing.  Ultralytics resolves `path:`
    # relative to this yaml, and split entries relative to `path`.
    base = data_yaml.parent
    raw_root = cfg.get("path")
    if raw_root:
        root = Path(str(raw_root))
        root = root.resolve() if root.is_absolute() else (base / root).resolve()
    else:
        root = base
    for split in ("train", "val"):
        raw = cfg.get(split)
        if raw is None:
            continue
        split_path = Path(str(raw))
        if not split_path.is_absolute():
            split_path = (root / split_path).resolve()
        if not split_path.exists():
            problems.append(f"Split '{split}' path does not exist: {split_path}")
            continue
        image_dir = split_path / "images" if (split_path / "images").is_dir() else split_path
        label_dir = split_path / "labels" if (split_path / "labels").is_dir() else image_dir
        images = [f for f in sorted(image_dir.rglob("*")) if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
        if not images:
            problems.append(f"Split '{split}' contains no images: {image_dir}")
            continue
        missing_labels = 0
        malformed = 0
        for img in images:
            label = (label_dir / img.stem).with_suffix(".txt")
            if not label.exists():
                missing_labels += 1
                continue
            for i, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    malformed += 1
                    problems.append(f"{label.name} line {i}: expected 'class cx cy w h', got {len(parts)} fields.")
                    continue
                try:
                    cls = int(float(parts[0]))
                    vals = [float(v) for v in parts[1:5]]
                except ValueError:
                    malformed += 1
                    problems.append(f"{label.name} line {i}: non-numeric values.")
                    continue
                if cls < 0 or cls >= len(class_names):
                    problems.append(f"{label.name} line {i}: class id {cls} out of range for {len(class_names)} classes.")
                if any(v < -0.001 or v > 1.001 for v in vals):
                    problems.append(f"{label.name} line {i}: YOLO values must be normalized to [0, 1].")
        if missing_labels:
            problems.append(
                f"Split '{split}': {missing_labels}/{len(images)} images have no label file "
                "(unlabeled images are ignored by the trainer)."
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_yaml = Path(args.data).expanduser()
    if not data_yaml.is_absolute():
        data_yaml = (Path.cwd() / data_yaml).resolve()

    print(f"Validating dataset config: {data_yaml}")
    problems = validate_dataset_yaml(data_yaml)
    if problems:
        print(f"\nDataset validation found {len(problems)} problem(s):")
        for msg in problems:
            print(f"  - {msg}")
    else:
        print("Dataset config looks structurally valid.")
    if args.validate_only:
        return 1 if problems else 0

    if problems:
        print("\nAborting training: fix the dataset problems above first (or use --validate-only).")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(
            "ERROR: ultralytics is not installed. Install dependencies first:\n"
            "  pip install -r requirements.txt"
        )
        print(f"({exc})")
        return 2

    batch: int | str
    try:
        batch = int(args.batch)
    except ValueError:
        batch = str(args.batch)

    resume_arg: str | bool | None
    if args.resume is None:
        resume_arg = False
    elif isinstance(args.resume, bool):
        resume_arg = True
    else:
        resume_arg = str(args.resume)

    model = YOLO(args.model)
    print(
        f"Starting training: model={args.model} data={data_yaml.name} epochs={args.epochs} "
        f"imgsz={args.imgsz} batch={batch} device={args.device or 'auto'}"
    )
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            resume=resume_arg,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.lr0,
            seed=args.seed,
            exist_ok=True,
        )
    except Exception as exc:
        print(f"ERROR: training failed: {exc}")
        return 3

    save_dir = Path(str(getattr(results, "save_dir", Path(args.project) / args.name)))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    metrics_path = save_dir / "results.csv"
    print("\nTraining finished.")
    print(f"  save_dir : {save_dir}")
    print(f"  best.pt  : {best if best.exists() else 'NOT FOUND'}")
    print(f"  last.pt  : {last if last.exists() else 'NOT FOUND'}")
    print(f"  metrics  : {metrics_path if metrics_path.exists() else 'results.csv not found'}")
    if best.exists():
        print("\nNext steps:")
        print(f"  python scripts/evaluate.py --weights {best} --images data/images_test --labels data/labels_test")
        print(f"  .venv/bin/streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
