"""Compare the RGB-depth ground-plane method with the other navigation modes on the pilot frames (same planner, same labels).

    python scripts/eval_depth.py            # writes reports/depth_vs_navigation.json

Exploratory: the pilot labels come from one annotator and were revised after earlier results (see data/leaf_litter_pilot).
A fresh depth segmenter is built per frame so the ground-plane smoothing does not leak between unrelated images.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import eval_leaf_litter as el  # noqa: E402
from src.depth_ground import make_depth_segmenter  # noqa: E402
from src.pipeline import Models, NavParams  # noqa: E402
from src.traversability import make_deepscene_segmenter  # noqa: E402

NAMES = ["classical", "multi-source", "RGB depth"]


def main() -> None:
    models, params = Models(), NavParams()
    frames = el.load_frames()
    ds_fn = make_deepscene_segmenter(str(models.deepscene_path()))
    depth = models.depth("cpu")
    rows = {n: [] for n in NAMES}
    for k, f in enumerate(frames):
        p = el.prepare_one(f, models, params, ds_fn, True, lambda im: make_depth_segmenter(depth)(im))
        for n in NAMES:
            rows[n].append(el.run_method(n, el.METHODS.get("classical"), f, p, params))
        if k % 20 == 0:
            print(f"  {k}/{len(frames)}", flush=True)
    res = {n: {role: el.summarize([r for r in rs if r["role"] == role]) for role in ("dev", "final_test")} for n, rs in rows.items()}
    res["all"] = {n: el.summarize(rs) for n, rs in rows.items()}
    Path("reports/depth_vs_navigation.json").write_text(json.dumps(res, indent=1, default=float))
    fmt = lambda v: "n/a" if v != v else f"{100 * v:.0f}%"  # noqa: E731
    print(f"{'method':14s} {'avail':>6s} {'false-stop':>10s} {'FS 95% CI':>12s} {'hazard hit':>10s} {'side':>5s} {'err deg':>7s} {'iou?':>5s}")
    for n in NAMES:
        d = res["all"][n]
        ci = d["false_stop_ci"]
        print(f"{n:14s} {fmt(d['availability']):>6s} {fmt(d['false_stop']):>10s} {f'[{100*ci[0]:.0f},{100*ci[1]:.0f}]':>12s} {fmt(d['false_safe_hazard']):>10s} {fmt(d['side_ok']):>5s} {d['abs_err_deg']:7.1f} {fmt(d['gt_recall']):>5s}")
    d0 = res["all"]["classical"]
    print(f"n: labelled={d0['n_labeled']} GO={d0['n_GO']} STOP={d0['n_STOP']} hazard-frames={d0['n_hazard_frames']} GT-frames={d0['n_gt_frames']}")


if __name__ == "__main__":
    main()
