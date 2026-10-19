"""Build a YOLO-seg dataset of drivable forest paths (FinnWoodlands `Track` class) in data_path/.

Masks come from the Voxel51 FiftyOne export (samples.json; per-detection numpy masks cropped to the
detection's bounding box).  Contiguous blocks are used for train/val/test (sequential video frames).
"""
import base64
import io
import json
import shutil
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))


def main(raw: Path, out: Path) -> None:
    samples = sorted(json.loads((raw / "samples.json").read_text())["samples"], key=lambda s: s["filepath"])
    n, start = len(samples), 0
    for si, (split, frac) in enumerate(SPLITS):
        end = n if si == len(SPLITS) - 1 else start + round(n * frac)
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        polys = 0
        for s in samples[start:end]:
            W, H = s["metadata"]["width"], s["metadata"]["height"]
            full = np.zeros((H, W), np.uint8)
            for d in s["segmentations"]["detections"]:
                if d["label"] != "Track":
                    continue
                m = np.load(io.BytesIO(zlib.decompress(base64.b64decode(d["mask"]["$binary"]["base64"]))))
                bx, by, bw, bh = d["bounding_box"]
                x0, y0 = round(bx * W), round(by * H)
                h, w = m.shape
                full[y0:y0 + h, x0:x0 + w][: H - y0, : W - x0] |= m[: H - y0, : W - x0].astype(np.uint8)
            stem = Path(s["filepath"]).stem
            lines = []
            cs, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                if cv2.contourArea(c) < 0.002 * W * H:
                    continue
                c = cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True).reshape(-1, 2)
                if len(c) >= 3:
                    lines.append("0 " + " ".join(f"{x / W:.5f} {y / H:.5f}" for x, y in c))
            polys += len(lines)
            shutil.copy(raw / "img" / f"{stem}.jpg", out / "images" / split / f"{stem}.jpg")
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"{split}: {end - start} images, {polys} path polygons")
        start = end


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
