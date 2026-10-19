"""Run the navigation pipeline on a fixed set of real-image scenarios and write a visual report.

    python scripts/nav_scenarios.py                       # hybrid method, default frames
    python scripts/nav_scenarios.py --method "Classical baseline"

Frames were chosen by eye from candidate sheets (not by pipeline output).  Bend frames were picked from the
FinnWoodlands ground-truth track direction.  Results are a PRELIMINARY visual check - there are no ground-truth
routes for these scenes, so nothing here is a quantitative accuracy claim.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import nav_render  # noqa: E402
from src.lighting_augmentation import apply_transformation  # noqa: E402
from src.pipeline import PATH_METHODS, Models, NavParams, run_navigation  # noqa: E402

# (scenario, frame stem, optional synthetic lighting variant)
SCENARIOS = [
    ("1 visible trail into forest", "ft_img2374", None),
    ("2 trail partly blocked by trunks", "ft_img2851", None),
    ("3 clear route bends left", "00051", None),
    ("4 clear route bends right", "00163", None),
    ("5 dense vegetation, no obvious corridor", "ft_img0707", None),
    ("6a heavy shadow (natural)", "ft_img2319", None),
    ("6b low light (synthetic: darker 0.8)", "00263", "darker"),
    ("7a no valid path (dense forest)", "00166", None),
    ("7b no valid path (narrow gap)", "00178", None),
    ("8 near miss: narrow snow track (false stop?)", "00003", None),
    ("9 machine blocks centre, open grass beside", "ft_img0627", None),
]


def find(stem: str) -> Path:
    for root in ("data/images", "data_path/images"):
        for split in ("test", "val", "train"):
            p = PROJECT_ROOT / root / split / f"{stem}.jpg"
            if p.exists():
                return p
    raise SystemExit(f"frame {stem} not found under data/images or data_path/images")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=PATH_METHODS[0], choices=PATH_METHODS)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--output", default="reports/navigation_scenarios")
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    models, params = Models(), NavParams(path_method=a.method, imgsz=a.imgsz)
    tiles, lines = [], [f"# Navigation scenarios ({a.method}, imgsz {a.imgsz})\n", "| scenario | frame | trunks | direction | route | coverage | confidence | note |", "|---|---|---|---|---|---|---|---|"]
    for name, stem, variant in SCENARIOS:
        img = cv2.imread(str(find(stem)))
        if variant:
            img = apply_transformation(img, variant, 0.8)
        o = run_navigation(img, models, params)
        m = o.metrics
        ok = o.plan.success
        note = ("partial route" if ok and o.plan.partial else "lighting normalised" if o.trav.lighting_normalized else "") if ok else o.plan.reason
        lines.append(f"| {name} | {stem} | {len(o.boxes)} | {o.direction.label} | {'yes' if ok else 'NO'} | {m['path_coverage_pct']:.0f}% | {m['path_confidence']:.2f} | {note} |")
        print(lines[-1], flush=True)
        vis = cv2.resize(nav_render.draw_navigation(img, o.trav, o.plan, o.direction), (640, 360))
        cv2.rectangle(vis, (0, 330), (640, 360), (0, 0, 0), -1)
        cv2.putText(vis, name, (6, 351), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(vis)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out / "scenarios.jpg"), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
