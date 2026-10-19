"""Isolate the drivable forest path with a YOLO-seg model and report a steering hint.

    python scripts/detect_path.py --source data_path/images/test --weights models/forest_path_seg.pt
Writes overlays to --output and path.csv (coverage, lateral offset of the path centre at the lower image
band; negative = path is left of image centre, positive = right).
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Image or folder.")
    ap.add_argument("--weights", default="models/forest_path_seg.pt")
    ap.add_argument("--output", default="outputs/path")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    src = Path(a.source)
    files = [src] if src.is_file() else sorted(f for f in src.iterdir() if f.suffix.lower() in EXTS)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    rows = []
    for f in files:
        img = cv2.imread(str(f))
        H, W = img.shape[:2]
        r = model.predict(img, conf=a.conf, imgsz=a.imgsz, device=a.device, verbose=False)[0]
        mask = np.zeros((H, W), np.uint8)
        if r.masks is not None:
            for poly in r.masks.xy:
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        band = mask[int(H * 0.75):]
        xs = np.nonzero(band)[1]
        offset = float((xs.mean() - W / 2) / (W / 2)) if xs.size else float("nan")
        overlay = img.copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.imwrite(str(out / f"path_{f.stem}.jpg"), overlay)
        rows.append({"image": f.name, "path_coverage": round(float(mask.mean()), 4), "offset": round(offset, 3)})
    with open(out / "path.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "path_coverage", "offset"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} image(s) -> {out}")


if __name__ == "__main__":
    main()
