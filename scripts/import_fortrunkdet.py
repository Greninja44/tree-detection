"""Add ForTrunkDet (Zenodo 5213825, CC BY 4.0) visible-light images to the YOLO layout in data/.

Thermal frames (640x512 FLIR) are skipped.  Images are grouped by (site folder, width) -- one group per
camera/site -- and each group is split into contiguous train/val/test blocks so near-identical
neighbouring video frames never straddle a split.  Files are prefixed `ft_`.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
MAX_SIDE = 1280


def parse(xml: str):
    folder = re.search(r"<folder>(.*?)</folder>", xml)
    w = int(re.search(r"<width>(\d+)", xml).group(1))
    h = int(re.search(r"<height>(\d+)", xml).group(1))
    boxes = []
    for obj in re.findall(r"<object>.*?</object>", xml, re.S):
        if "<name>trunk</name>" not in obj:
            continue
        b = [float(re.search(rf"<{k}>([\d.]+)</{k}>", obj).group(1)) for k in ("xmin", "ymin", "xmax", "ymax")]
        boxes.append(b)
    return (folder.group(1) if folder else "?"), w, h, boxes


def main(raw: Path, out: Path) -> None:
    groups = defaultdict(list)
    for xml_path in sorted((raw / "Annotations").glob("*.xml")):
        folder, w, h, boxes = parse(xml_path.read_text())
        if w <= 640 or not (raw / "JPEGImages" / f"{xml_path.stem}.jpg").exists():
            continue
        groups[(folder, w)].append((xml_path.stem, w, h, boxes))
    totals = defaultdict(lambda: [0, 0])
    for key, items in sorted(groups.items()):
        items.sort(key=lambda t: int(re.sub(r"\D", "", t[0]) or 0))
        n, start = len(items), 0
        for si, (split, frac) in enumerate(SPLITS):
            end = n if si == len(SPLITS) - 1 else start + round(n * frac)
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            for stem, w, h, boxes in items[start:end]:
                name = f"ft_{stem}"
                img = Image.open(raw / "JPEGImages" / f"{stem}.jpg").convert("RGB")
                if max(img.size) > MAX_SIDE:
                    s = MAX_SIDE / max(img.size)
                    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
                img.save(out / "images" / split / f"{name}.jpg", quality=92)
                lines = []
                for x0, y0, x1, y1 in boxes:
                    x0, x1 = max(0.0, x0), min(float(w), x1)
                    y0, y1 = max(0.0, y0), min(float(h), y1)
                    if x1 - x0 < 2 or y1 - y0 < 2:
                        continue
                    lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
                (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                totals[split][0] += 1
                totals[split][1] += len(lines)
            start = end
        print(f"group {key}: {n} images")
    for split, (ni, nb) in totals.items():
        print(f"{split}: +{ni} images, +{nb} boxes")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
