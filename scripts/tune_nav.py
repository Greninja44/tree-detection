"""Tune safety-neutral planner parameters on FinnWoodlands GT-track frames (train split) and report held-out (val+test).

    python scripts/tune_nav.py --method "Terrain-aware (classical + DeepScene)" --veto-threshold 0.7

Frozen on purpose: look-ahead rows, safety margin, minimum clearance, minimum confidence and minimum reach, so the GT
target row does not move and the planner cannot "improve" by simply being less careful.
Score per frame = 0 if no route, else 1 - min(|heading error|, 45 deg) / 45.  GT is only the narrow groomed track.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import eval_navigation as ev  # noqa: E402
from src import infer  # noqa: E402
from src.navigation import NavConfig, estimate_navigation_direction, plan_local_path  # noqa: E402
from src.pipeline import Models, NavParams, resolve_device  # noqa: E402
from src.traversability import detect_traversable_region  # noqa: E402

GRID = {"k_lat": [0.0, 1.0, 2.0, 4.0, 6.0], "k_goal": [3.0, 6.0, 10.0], "k_clear": [4.0, 8.0, 12.0],
        "smoothing": [0.2, 0.5, 0.8], "k_start": [10.0, 30.0, 60.0], "max_start_shift": [0.2, 0.35, 0.5]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="Classical baseline")
    ap.add_argument("--veto-threshold", type=float, default=None)
    ap.add_argument("--output", default="reports/nav_tuning")
    a = ap.parse_args()
    out = Path(a.output)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    models = Models()
    base = NavParams(path_method=a.method, veto_threshold=a.veto_threshold)
    device, _ = resolve_device("cpu")
    frames = {"train": [], "heldout": []}
    for split in ("train", "val", "test"):
        for lab in sorted((PROJECT_ROOT / "data_path/labels" / split).glob("*.txt")):
            if not lab.read_text().strip():
                continue
            img = cv2.imread(str(PROJECT_ROOT / "data_path/images" / split / f"{lab.stem}.jpg"))
            H, W = img.shape[:2]
            gt = ev.gt_mask(lab, (H, W))
            nav = base.nav
            tgt = nav.start_y - nav.heading_lookahead * nav.lookahead
            ys, xs = np.nonzero(gt[int((tgt - .05) * H):int((tgt + .05) * H)])
            if not xs.size:
                continue
            gt_ang = float(np.degrees(np.arctan2(xs.mean() - .5 * W, (nav.start_y - tgt) * H)))
            bf = out / "cache" / f"boxes_{lab.stem}.npy"
            boxes = None
            if bf.exists():
                try:
                    boxes = [tuple(b) for b in np.load(bf)]
                except (EOFError, ValueError, OSError):  # truncated by an interrupted run: recompute
                    boxes = None
            if boxes is None:
                boxes = infer.detect_trunks(models.yolo(base.trunk_weights), img, base.conf, base.imgsz, device)
                tmp = bf.with_suffix(".tmp.npy")
                np.save(tmp, np.array(boxes, np.float32).reshape(-1, 5))
                tmp.replace(bf)
            seg = ev.cached_seg(models, base, device, img, lab.stem, PROJECT_ROOT / "reports/navigation_eval/cache")
            trav = detect_traversable_region(img, boxes, lambda _i, _s=seg: _s, base.trav)
            frames["train" if split == "train" else "heldout"].append((W, H, trav, gt_ang))
    print(f"frames: train={len(frames['train'])} held-out={len(frames['heldout'])}", flush=True)

    def score(cfg: NavConfig, fr) -> tuple[float, float, float]:
        sc, found, errs = [], 0, []
        for W, H, trav, gt_ang in fr:
            plan = plan_local_path(trav.traversable, trav.obstacles, cfg, trav.p_trav)
            if not plan.success:
                sc.append(0.0)
                continue
            found += 1
            err = abs(gt_ang - estimate_navigation_direction(plan.points, W, H, cfg).angle_deg)
            errs.append(err)
            sc.append(1 - min(err, 45.0) / 45.0)
        return float(np.mean(sc)), found / len(fr), float(np.mean(errs)) if errs else float("nan")

    cfg = base.nav
    s0 = score(cfg, frames["train"])
    print(f"start  train score={s0[0]:.3f} found={s0[1]:.2f} err={s0[2]:.1f}", flush=True)
    for sweep in range(2):
        for name, values in GRID.items():
            res = [(score(dataclasses.replace(cfg, **{name: v}), frames["train"])[0], v) for v in values]
            best = max(res)
            if best[0] > score(cfg, frames["train"])[0] + 1e-4:
                cfg = dataclasses.replace(cfg, **{name: best[1]})
                print(f"  sweep {sweep}: {name} -> {best[1]} (train score {best[0]:.3f})", flush=True)
    for label, c in (("default", base.nav), ("tuned", cfg)):
        tr, ho = score(c, frames["train"]), score(c, frames["heldout"])
        print(f"{label:8s} train: score={tr[0]:.3f} found={tr[1]:.2f} err={tr[2]:.1f} | HELD-OUT: score={ho[0]:.3f} found={ho[1]:.2f} err={ho[2]:.1f}", flush=True)
    changed = {k: v for k, v in dataclasses.asdict(cfg).items() if v != getattr(base.nav, k)}
    (out / f"tuned_{a.method.split()[0]}.json").write_text(json.dumps({"method": a.method, "veto_threshold": a.veto_threshold, "changed": changed}, indent=2))
    print("CHANGED:", changed)


if __name__ == "__main__":
    main()
