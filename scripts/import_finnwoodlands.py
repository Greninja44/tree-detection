"""Convert the Voxel51/FinnWoodlands FiftyOne export into the YOLO layout in data/.

Trunk classes (Birch, Spruce, Pine, Tree) are merged into class 0 `tree_trunk`.
Frames are sequential, so splits are contiguous blocks to avoid train/test leakage.
"""
import json
import shutil
import sys
from pathlib import Path

TRUNKS = {"Birch", "Spruce", "Pine", "Tree"}
SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))


def main(raw: Path, out: Path) -> None:
    samples = sorted(json.loads((raw / "samples.json").read_text())["samples"],
                     key=lambda s: s["filepath"])
    n, start = len(samples), 0
    for si, (split, frac) in enumerate(SPLITS):
        end = n if si == len(SPLITS) - 1 else start + round(n * frac)
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        boxes = 0
        for s in samples[start:end]:
            stem = Path(s["filepath"]).stem
            lines = []
            for d in s["detections"]["detections"]:
                if d["label"] in TRUNKS:
                    x, y, w, h = d["bounding_box"]
                    lines.append(f"0 {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
            shutil.copy(raw / "img" / f"{stem}.jpg", out / "images" / split / f"{stem}.jpg")
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            boxes += len(lines)
        print(f"{split}: {end - start} images, {boxes} trunk boxes")
        start = end


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
