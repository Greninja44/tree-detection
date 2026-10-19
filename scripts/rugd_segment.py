"""Pretrained RUGD-Segformer (JasonTStanley/RUGD-Segformer): isolate drivable ground and trees.

    python scripts/rugd_segment.py --source data/images/test --output outputs/rugd --max-images 8
Green = drivable ground (dirt, sand, gravel, mulch, asphalt), red = tree/log, blue-grey = everything else
is left untouched.  path.csv gives drivable coverage and the lateral offset of the drivable region in the
lower quarter of the frame (negative = left of centre, positive = right).
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

EXTS = {".jpg", ".jpeg", ".png"}
DRIVABLE = {"dirt", "sand", "gravel", "mulch", "asphalt"}
TREES = {"tree", "log"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Image or folder.")
    ap.add_argument("--weights", default="models/rugd_segformer", help="Folder with config.json + model.safetensors.")
    ap.add_argument("--output", default="outputs/rugd")
    ap.add_argument("--imgsz", type=int, default=640, help="Square network input (model was trained at 640).")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    src = Path(a.source)
    files = [src] if src.is_file() else sorted(f for f in src.iterdir() if f.suffix.lower() in EXTS)
    if a.max_images:
        step = max(1, len(files) // a.max_images)
        files = files[::step][: a.max_images]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    model = SegformerForSemanticSegmentation.from_pretrained(a.weights).to(a.device).eval()
    names = {int(k): v for k, v in model.config.id2label.items()}
    drive_ids = [i for i, n in names.items() if n in DRIVABLE]
    tree_ids = [i for i, n in names.items() if n in TREES]
    rows = []
    for f in files:
        img = cv2.imread(str(f))
        H, W = img.shape[:2]
        x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (a.imgsz, a.imgsz)).astype(np.float32) / 255.0
        x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(a.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(pixel_values=x).logits
        logits = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        seg = logits.argmax(1)[0].cpu().numpy()
        dt = time.perf_counter() - t0
        drive = np.isin(seg, drive_ids)
        trees = np.isin(seg, tree_ids)
        ov = img.astype(np.float32)
        ov[drive] = 0.5 * ov[drive] + 0.5 * np.array([0, 255, 0], np.float32)
        ov[trees] = 0.6 * ov[trees] + 0.4 * np.array([0, 0, 255], np.float32)
        cv2.imwrite(str(out / f"rugd_{f.stem}.jpg"), ov.astype(np.uint8))
        xs = np.nonzero(drive[int(H * 0.75):])[1]
        off = float((xs.mean() - W / 2) / (W / 2)) if xs.size else float("nan")
        rows.append({"image": f.name, "drivable_coverage": round(float(drive.mean()), 4),
                     "tree_coverage": round(float(trees.mean()), 4), "offset": round(off, 3), "seconds": round(dt, 2)})
        print(rows[-1])
    with open(out / "path.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
