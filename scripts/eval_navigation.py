"""Preliminary check of the navigation pipeline against the FinnWoodlands 'Track' masks (data_path/labels).

    python scripts/eval_navigation.py --stride 3

The only ground truth is the narrow groomed track, so wide drivable ground outside it counts against the route
(`route_on_track`).  Treat results as a sanity check, not as accuracy for arbitrary forests.
GT heading = angle from the nominal start to the centroid of the GT track within +-5% of the image height around the
row the pipeline steers toward (start - 0.6 * look-ahead).  Segmentation/trunk outputs are cached in <output>/cache so
planner changes can be re-scored without re-running the models.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import infer  # noqa: E402
from src.navigation import estimate_navigation_direction, path_coverage, plan_local_path  # noqa: E402
from src.pipeline import HYBRID, TERRAIN, VETO, Models, NavParams, resolve_device  # noqa: E402
from src.traversability import SegOutput, detect_traversable_region  # noqa: E402

SIDE = lambda a, dead: "S" if abs(a) <= dead else ("R" if a > 0 else "L")  # noqa: E731


def gt_mask(label_file: Path, shape) -> np.ndarray:
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    for line in label_file.read_text().splitlines():
        v = np.array(line.split()[1:], np.float32).reshape(-1, 2) * [W, H]
        cv2.fillPoly(m, [v.astype(np.int32)], 1)
    return m.astype(bool)


def cached_seg(models, params, device, img, key, cache: Path):
    """SegOutput for `img`, loaded from / saved to the cache.  The hybrid reuses the cached RUGD output."""
    from src.traversability import fuse_obstacle_veto, fuse_segmentations, fuse_terrain_aware, normalize_lighting, segment_classical

    norm, _ = normalize_lighting(img) if params.trav.normalize_lighting else (img, False)
    if params.path_method == HYBRID:
        base = cached_seg(models, NavParams(path_method="RUGD-Segformer", imgsz=params.imgsz, device=params.device), device, img, key, cache)
        return fuse_segmentations(base, segment_classical(norm, params.trav))
    if params.path_method == TERRAIN:
        base = cached_seg(models, NavParams(path_method="RUGD-Segformer", imgsz=params.imgsz, device=params.device), device, img, key, cache)
        return fuse_terrain_aware(base, segment_classical(norm, params.trav), params.allow_grass)
    if params.path_method == VETO:
        base = cached_seg(models, NavParams(path_method="RUGD-Segformer", imgsz=params.imgsz, device=params.device), device, img, key, cache)
        return fuse_obstacle_veto(base, segment_classical(norm, params.trav))
    f = cache / f"seg_{params.path_method[:4]}_{params.imgsz}_{key}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        names = {int(k): v for k, v in z["names"].item().items()} if "names" in z.files else None
        return SegOutput(z["classes"], z["p_trav"].astype(np.float32), z["valid"], str(z["method"]), list(z["notes"]),
                         z["labels"] if "labels" in z.files else None, names)
    seg_fn = models.segmenter(params.path_method, params.imgsz, device, params.path_weights)
    seg = seg_fn(norm) if seg_fn else segment_classical(norm, params.trav)
    extra = {} if seg.labels is None else {"labels": seg.labels, "names": np.array(seg.label_names, dtype=object)}
    np.savez_compressed(f, classes=seg.classes, p_trav=seg.p_trav.astype(np.float16), valid=seg.valid, method=seg.method,
                        notes=np.array(seg.notes, dtype=object), **extra)
    return seg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="Classical baseline,Terrain-aware (classical + RUGD),Classical + obstacle veto,RUGD + classical hybrid")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--output", default="reports/navigation_eval")
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ("train", "val", "test"):
        for lab in sorted((PROJECT_ROOT / "data_path/labels" / split).glob("*.txt")):
            if lab.read_text().strip():
                frames.append((PROJECT_ROOT / "data_path/images" / split / f"{lab.stem}.jpg", lab))
    frames = frames[::a.stride]
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    models, rows = Models(), []
    for method in a.methods.split(","):
        params = NavParams(path_method=method, imgsz=a.imgsz)
        dead = params.nav.straight_deg
        for img_p, lab in frames:
            img = cv2.imread(str(img_p))
            H, W = img.shape[:2]
            gt = gt_mask(lab, (H, W))
            tgt = params.nav.start_y - params.nav.heading_lookahead * params.nav.lookahead
            band = gt[int((tgt - .05) * H):int((tgt + .05) * H)]
            ys, xs = np.nonzero(band)
            gt_ang = float(np.degrees(np.arctan2(xs.mean() - .5 * W, (params.nav.start_y - tgt) * H))) if xs.size else None
            device, _ = resolve_device(params.device)
            bf = cache / f"boxes_{params.trunk_weights}_{params.conf}_{params.imgsz}_{img_p.stem}.npy"
            if bf.exists():
                boxes = [tuple(b) for b in np.load(bf)]
            else:
                boxes = infer.detect_trunks(models.yolo(params.trunk_weights), img, params.conf, params.imgsz, device)
                np.save(bf, np.array(boxes, np.float32).reshape(-1, 5))
            seg = cached_seg(models, params, device, img, img_p.stem, cache)
            trav = detect_traversable_region(img, boxes, lambda _im, _s=seg: _s, params.trav)
            plan = plan_local_path(trav.traversable, trav.obstacles, params.nav, trav.p_trav)
            direction = estimate_navigation_direction(plan.points if plan.success else None, W, H, params.nav)
            found = plan.success
            row = {"method": method, "image": img_p.name, "route_found": int(found),
                   "gt_angle": None if gt_ang is None else round(gt_ang, 1),
                   "pred_angle": None if not found else round(direction.angle_deg, 1),
                   "gt_recall_in_traversable": round(float((trav.traversable & gt).sum() / max(1, gt.sum())), 3)}
            if found:
                near = cv2.dilate(gt.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
                pts = plan.points.astype(int).clip([0, 0], [W - 1, H - 1])
                row["route_on_track"] = round(float(near[pts[:, 1], pts[:, 0]].mean()), 3)
                if gt_ang is not None:
                    row["side_agree"] = int(SIDE(gt_ang, dead) == SIDE(direction.angle_deg, dead))
                    row["abs_angle_err"] = round(abs(gt_ang - direction.angle_deg), 1)
            rows.append(row)
            print(row, flush=True)
    keys = ["method", "image", "route_found", "gt_angle", "pred_angle", "abs_angle_err", "side_agree", "route_on_track", "gt_recall_in_traversable"]
    with open(out / "per_frame.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out / "summary.txt", "w") as fh:
        for method in a.methods.split(","):
            r = [x for x in rows if x["method"] == method]
            f = [x for x in r if x["route_found"]]
            g = [x for x in f if "side_agree" in x]
            line = (f"{method}: frames={len(r)} route_found={len(f)}/{len(r)} "
                    f"side_agree={np.mean([x['side_agree'] for x in g]) if g else float('nan'):.2f} (n={len(g)}) "
                    f"mean_abs_angle_err={np.mean([x['abs_angle_err'] for x in g]) if g else float('nan'):.1f}deg "
                    f"route_on_track={np.mean([x['route_on_track'] for x in f]) if f else float('nan'):.2f} "
                    f"gt_recall={np.mean([x['gt_recall_in_traversable'] for x in r]):.2f}")
            print(line)
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
