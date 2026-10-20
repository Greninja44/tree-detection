"""Side-by-side visual comparison of fusion variants on chosen frames (outcome + STOP reason code on every tile).

    python scripts/compare_nav_variants.py ft_img2374 ft_img0707 --out reports/leaf_litter
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import eval_leaf_litter as el  # noqa: E402
from src import nav_render  # noqa: E402
from src.navigation import estimate_navigation_direction, plan_local_path  # noqa: E402
from src.pipeline import Models, NavParams  # noqa: E402
from src.traversability import detect_traversable_region, make_deepscene_segmenter  # noqa: E402

VARIANTS = ["classical", "current: soft veto 0.7", "conf-aware w=1.0", "conf-aware w=0.0 +coco"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--out", default="reports/leaf_litter")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models, params = Models(), NavParams(path_method="Terrain-aware (classical + DeepScene)")
    ds_fn = make_deepscene_segmenter(str(models.deepscene_path()))
    frames = {f["image"]: f for f in el.load_frames()}
    rows = []
    for stem in a.frames:
        f = frames.get(stem) or {"image": stem, "hazards": [], "role": "adhoc", "sequence": "x", "terrain": "?", "expected": "?", "gt_label": None}
        p = el.prepare_one(f, models, params, ds_fn)
        tiles = []
        for name in VARIANTS:
            seg = el.METHODS[name](p["ds"], p["cl"])
            boxes = list(p["boxes"]) + (list(p["coco"]) if name.endswith("+coco") else [])
            trav = detect_traversable_region(p["img"], boxes, lambda _i, _s=seg: _s, params.trav)
            plan = plan_local_path(trav.traversable, trav.obstacles, params.nav, trav.p_trav)
            d = estimate_navigation_direction(plan.points if plan.success else None, p["W"], p["H"], params.nav)
            vis = cv2.resize(nav_render.draw_navigation(p["img"], trav, plan, d), (480, 270))
            for h in f.get("hazards", []):
                cv2.rectangle(vis, (int(h[0] * 480), int(h[1] * 270)), (int(h[2] * 480), int(h[3] * 270)), (255, 0, 255), 2)
            cv2.rectangle(vis, (0, 240), (480, 270), (0, 0, 0), -1)
            cv2.putText(vis, f"{name}: {'ROUTE' if plan.success else 'STOP ' + plan.reason_code}", (4, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(vis)
        label = np.zeros((24, 480 * len(VARIANTS), 3), np.uint8)
        cv2.putText(label, f"{stem}  expected={f.get('expected')}  terrain={f.get('terrain')}  role={f.get('role')}   (magenta = labelled hazard box)", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        rows.append(np.vstack([label, np.hstack(tiles)]))
    cv2.imwrite(str(out / "variant_comparison.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out / "variant_comparison.jpg")


if __name__ == "__main__":
    main()
