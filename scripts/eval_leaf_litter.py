"""Leaf-litter / safety evaluation: route availability vs false stops vs false-safe routes, on identical frames.

    python scripts/eval_leaf_litter.py --select           # tune on dev, report final_test once
    python scripts/eval_leaf_litter.py --methods all

Frames: data/leaf_litter_pilot/annotations.json (single-annotator pilot labels, see its `protocol`) plus every FinnWoodlands
frame with a ground-truth groomed track (labelled GO automatically).  AMBIGUOUS frames are excluded from all rates.
`role == dev` frames may be used to choose parameters; `final_test` frames are only scored, never tuned on.
CIs are cluster bootstraps over `sequence` ids (adjacent frames are correlated).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src import infer  # noqa: E402
from src.navigation import estimate_navigation_direction, plan_local_path  # noqa: E402
from src.pipeline import Models, NavParams  # noqa: E402
from src.traversability import (SegOutput, detect_traversable_region, fuse_confidence_aware, fuse_obstacle_veto, fuse_segmentations,  # noqa: E402
                                fuse_multi_source, fuse_terrain_aware, make_deepscene_segmenter, make_rugd_segmenter, normalize_lighting, segment_classical)

ANN = PROJECT_ROOT / "data/leaf_litter_pilot/annotations.json"
CACHE = PROJECT_ROOT / "reports/leaf_litter/cache"
W_GRID = [1.0, 0.6, 0.4, 0.3, 0.2, 0.0]
COCO_HAZARD_IDS = [0, 1, 2, 3, 5, 6, 7] + list(range(14, 24))   # person, bicycle, car, motorcycle, bus, train, truck, animals


def gt_mask(label_file: Path, shape) -> np.ndarray:
    """Rasterise a YOLO-seg polygon label file (FinnWoodlands groomed track) to a boolean mask."""
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    for line in label_file.read_text().splitlines():
        v = np.array(line.split()[1:], np.float32).reshape(-1, 2) * [W, H]
        cv2.fillPoly(m, [v.astype(np.int32)], 1)
    return m.astype(bool)


def find_image(stem: str) -> Path:
    for root in ("data/images", "data_path/images"):
        for s in ("train", "val", "test"):
            p = PROJECT_ROOT / root / s / f"{stem}.jpg"
            if p.exists():
                return p
    raise FileNotFoundError(stem)


def load_frames() -> list[dict]:
    doc = json.loads(ANN.read_text())
    frames = [dict(f, gt_mask=None) for f in doc["frames"]]
    have = {f["image"] for f in frames}
    for split in ("train", "val", "test"):
        for lab in sorted((PROJECT_ROOT / "data_path/labels" / split).glob("*.txt")):
            if lab.read_text().strip() and lab.stem not in have:
                frames.append({"image": lab.stem, "source": "FinnWoodlands", "split": split, "role": "dev" if split == "train" else "final_test",
                               "sequence": f"finn_{int(lab.stem) // 20}", "expected": "GO", "terrain": "snow_track", "lighting": "mixed",
                               "hazards": [], "note": "GT groomed track", "gt_label": lab})
    return frames


METHODS = {
    "classical": lambda ds, cl: cl,
    "argmax_veto (v1)": lambda ds, cl: fuse_terrain_aware(ds, cl, True, None),
    "current: soft veto 0.7": lambda ds, cl: fuse_terrain_aware(ds, cl, True, 0.7),
    "obstacle_veto_only (classical+DS obstacle)": lambda ds, cl: fuse_obstacle_veto(ds, cl),
    **{f"conf-aware w={w}": (lambda w: lambda ds, cl: fuse_confidence_aware(ds, cl, True, 0.5, w))(w) for w in W_GRID},
}
# same fusions, with COCO person/vehicle/animal detections added to the obstacle footprints (name suffix " +coco")
COCO_METHODS = ["classical", "current: soft veto 0.7"] + [f"conf-aware w={w}" for w in (1.0, 0.4, 0.2, 0.0)]
for _n in COCO_METHODS:
    METHODS[_n + " +coco"] = METHODS[_n]


# RUGD-Segformer primary (needs models/rugd_segformer): evaluated on the same frames with the same planner and rules.
RUGD_METHODS = {
    "RUGD alone": lambda rg, cl: rg,
    "RUGD+classical hybrid": lambda rg, cl: fuse_segmentations(rg, cl),
    "RUGD classical-led + obstacle veto": lambda rg, cl: fuse_obstacle_veto(rg, cl),
    "RUGD terrain-aware (label veto)": lambda rg, cl: fuse_terrain_aware(rg, cl, True, None),
    **{f"RUGD conf-aware w={w}": (lambda w: lambda rg, cl: fuse_confidence_aware(rg, cl, True, 0.5, w))(w) for w in (1.0, 0.5, 0.0)},
}


# Multi-source (needs both primaries); designed AFTER seeing final_test results of the RUGD variants -> exploratory, see report.
MULTI_METHODS = ["multi-source", "multi-source +coco", "rugd+classical multi (no DeepScene)", "rugd+classical multi (no DeepScene) +coco"]


def rugd_cached(f, models, params, norm, cache):
    """RUGD SegOutput for a frame, cached on disk (it costs ~5 s/frame on the Jetson CPU)."""
    fp = cache / f"rugd_{f['image']}.npz"
    if fp.exists():
        try:
            z = np.load(fp, allow_pickle=True)
            return SegOutput(z["classes"], z["p_trav"].astype(np.float32), z["valid"], "RUGD-Segformer", [], z["labels"],
                             {int(k): v for k, v in z["names"].item().items()}, z["p_block"].astype(np.float32),
                             z["p_obstacle"].astype(np.float32), z["p_vegetation"].astype(np.float32))
        except Exception:
            pass
    seg = make_rugd_segmenter(models.rugd(), params.imgsz, "cpu")(norm)
    tmp = fp.with_name(fp.stem + ".tmp.npz")
    np.savez_compressed(tmp, classes=seg.classes, p_trav=seg.p_trav.astype(np.float16), valid=seg.valid, labels=seg.labels,
                        names=np.array(seg.label_names, dtype=object), p_block=seg.p_block.astype(np.float16),
                        p_obstacle=seg.p_obstacle.astype(np.float16), p_vegetation=seg.p_vegetation.astype(np.float16))
    tmp.replace(fp)
    return seg


def hazard_hit(points: np.ndarray, hazards, W: int, H: int) -> bool:
    pad = 0.01 * W
    for x0, y0, x1, y1 in hazards:
        bx0, by0, bx1, by1 = x0 * W - pad, y0 * H - pad, x1 * W + pad, y1 * H + pad
        if ((points[:, 0] >= bx0) & (points[:, 0] <= bx1) & (points[:, 1] >= by0) & (points[:, 1] <= by1)).any():
            return True
    return False


def prepare_one(f, models, params, ds_fn, with_rugd=False):
    """Everything needed for one frame (kept only while that frame is being scored, to bound memory)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(find_image(f["image"])))
    H, W = img.shape[:2]
    norm, _ = normalize_lighting(img)
    bf = CACHE / f"boxes_{f['image']}.npy"
    boxes = None
    if bf.exists():
        try:
            boxes = [tuple(b) for b in np.load(bf)]
        except Exception:
            boxes = None
    if boxes is None:
        boxes = infer.detect_trunks(models.yolo(params.trunk_weights), img, params.conf, params.imgsz, "cpu")
        np.save(bf.with_suffix(".tmp.npy"), np.array(boxes, np.float32).reshape(-1, 5)); bf.with_suffix(".tmp.npy").replace(bf)
    cf = CACHE / f"coco_{f['image']}.npy"
    coco = None
    if cf.exists():
        try:
            coco = [tuple(b) for b in np.load(cf)]
        except Exception:
            coco = None
    if coco is None:
        r = models.yolo("yolov8n.pt").predict(img, conf=0.25, imgsz=640, device="cpu", verbose=False)[0]
        keep = np.isin(r.boxes.cls.cpu().numpy().astype(int), COCO_HAZARD_IDS)
        coco = [tuple(map(float, b)) + (float(c),) for b, c in zip(r.boxes.xyxy.cpu().numpy()[keep], r.boxes.conf.cpu().numpy()[keep])]
        np.save(cf.with_suffix(".tmp.npy"), np.array(coco, np.float32).reshape(-1, 5)); cf.with_suffix(".tmp.npy").replace(cf)
    t0 = time.perf_counter(); ds = ds_fn(norm); cl = segment_classical(norm, params.trav); seg_t = time.perf_counter() - t0
    gt, gt_ang = None, None
    if f.get("gt_label") is not None:
        gt = gt_mask(f["gt_label"], (H, W))
        nav = params.nav
        tgt = nav.start_y - nav.heading_lookahead * nav.lookahead
        ys, xs = np.nonzero(gt[int((tgt - .05) * H):int((tgt + .05) * H)])
        gt_ang = float(np.degrees(np.arctan2(xs.mean() - .5 * W, (nav.start_y - tgt) * H))) if xs.size else None
    rg = rugd_cached(f, models, params, norm, CACHE) if with_rugd else None
    return dict(img=img, boxes=boxes, coco=coco, rg=rg, ds=ds, cl=cl, W=W, H=H, gt=gt, gt_ang=gt_ang, seg_s=seg_t)


def run_method(name, fuse, f, p, params):
    t0 = time.perf_counter()
    if name.startswith("rugd+classical multi"):   # ablation: DeepScene removed (a source that never says ground or obstacle)
        z = np.zeros_like(p["ds"].p_trav)
        null_ds = SegOutput(p["ds"].classes, z, p["ds"].valid | True, "none", [], None, None, None, z, z)
        seg = fuse_multi_source(p["rg"], null_ds, p["cl"])
    elif name.startswith("multi-source"):
        seg = fuse_multi_source(p["rg"], p["ds"], p["cl"])
    else:
        seg = fuse(p["rg"] if name in RUGD_METHODS else p["ds"], p["cl"])
    obstacle_boxes = list(p["boxes"]) + (list(p["coco"]) if name.endswith(" +coco") else [])
    trav = detect_traversable_region(p["img"], obstacle_boxes, lambda _i, _s=seg: _s, params.trav)
    plan = plan_local_path(trav.traversable, trav.obstacles, params.nav, trav.p_trav)
    dt = time.perf_counter() - t0
    r = {"method": name, "image": f["image"], "role": f["role"], "sequence": f["sequence"], "terrain": f["terrain"], "expected": f["expected"],
         "route": bool(plan.success), "code": plan.reason_code, "latency_fuse_plan_s": dt, "seg_s": p["seg_s"]}
    if f["hazards"]:    # every frame with a hazard box counts, whatever its corridor label
        r["hazard_hit"] = bool(plan.success and hazard_hit(plan.points, f["hazards"], p["W"], p["H"]))
    if p["gt"] is not None:
        r["gt_recall"] = float((trav.traversable & p["gt"]).sum() / max(1, p["gt"].sum()))
        r["iou"] = float((trav.traversable & p["gt"]).sum() / max(1, (trav.traversable | p["gt"]).sum()))
        if plan.success and p["gt_ang"] is not None:
            ang = estimate_navigation_direction(plan.points, p["W"], p["H"], params.nav).angle_deg
            r["abs_err"] = abs(p["gt_ang"] - ang)
            r["side_ok"] = float((abs(p["gt_ang"]) <= params.nav.straight_deg and abs(ang) <= params.nav.straight_deg)
                                 or (abs(p["gt_ang"]) > params.nav.straight_deg and abs(ang) > params.nav.straight_deg and np.sign(ang) == np.sign(p["gt_ang"])))
    return r


def cluster_boot(rows, fn, n=2000, seed=0):
    seqs = sorted({r["sequence"] for r in rows})
    if len(seqs) < 2:
        return (float("nan"), float("nan"))
    by = {s: [r for r in rows if r["sequence"] == s] for s in seqs}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        samp = [r for s in rng.choice(seqs, len(seqs)) for r in by[s]]
        v = fn(samp)
        if v == v:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def summarize(rows):
    lab = [r for r in rows if r["expected"] in ("GO", "STOP")]
    go, st = [r for r in lab if r["expected"] == "GO"], [r for r in lab if r["expected"] == "STOP"]
    hz = [r for r in rows if "hazard_hit" in r]
    gt = [r for r in rows if "gt_recall" in r]
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
    out = {"n_labeled": len(lab), "n_GO": len(go), "n_STOP": len(st), "n_hazard_frames": len(hz), "n_gt_frames": len(gt),
           "availability": mean([r["route"] for r in lab]),
           "false_stop": mean([not r["route"] for r in go]), "false_safe_label": mean([r["route"] for r in st]),
           "false_safe_hazard": mean([r["hazard_hit"] for r in hz]),
           "gt_recall": mean([r["gt_recall"] for r in gt]), "iou": mean([r["iou"] for r in gt]),
           "side_ok": mean([r["side_ok"] for r in rows if "side_ok" in r]), "abs_err_deg": mean([r["abs_err"] for r in rows if "abs_err" in r]),
           "latency_fuse_plan_s": mean([r["latency_fuse_plan_s"] for r in rows]), "seg_s": mean([r["seg_s"] for r in rows])}
    out["false_stop_ci"] = cluster_boot(go, lambda s: mean([not r["route"] for r in s if r["expected"] == "GO"]))
    if st:
        out["false_safe_label_ci"] = cluster_boot(st, lambda s: mean([r["route"] for r in s if r["expected"] == "STOP"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true", help="choose the conf-aware weight on dev, then score final_test once")
    ap.add_argument("--output", default="reports/leaf_litter_metrics.json")
    ap.add_argument("--with-rugd", action="store_true", help="also score the RUGD-Segformer variants (cached on disk; ~5 s/frame first time)")
    a = ap.parse_args()
    models, params = Models(), NavParams()
    frames = load_frames()
    print(f"frames: {len(frames)} (pilot labelled + Finn GT)", flush=True)
    ds_fn = make_deepscene_segmenter(str(models.deepscene_path()))
    if a.with_rugd:
        METHODS.update(RUGD_METHODS)
        METHODS.update({n: None for n in MULTI_METHODS})
    rows = {name: [] for name in METHODS}
    for k, f in enumerate(frames):
        p = prepare_one(f, models, params, ds_fn, a.with_rugd)
        for name, fn in METHODS.items():
            rows[name].append(run_method(name, fn, f, p, params))
        del p
        if k % 40 == 0:
            print(f"  scored {k}/{len(frames)}", flush=True)
    res = {"protocol": "dev = may tune; final_test = scored only. AMBIGUOUS excluded.", "methods": {}, "by_terrain_final_test": {}}
    for name, rs in rows.items():
        res["methods"][name] = {role: summarize([r for r in rs if r["role"] == role]) for role in ("dev", "final_test")}
        res["by_terrain_final_test"][name] = {t: summarize([r for r in rs if r["role"] == "final_test" and r["terrain"] == t])
                                              for t in sorted({r["terrain"] for r in rs if r["role"] == "final_test"})}
    # stop-code histogram for current method on dev+final GO frames that stopped
    res["stop_codes_on_GO_frames"] = {name: {c: sum(1 for r in rs if r["expected"] == "GO" and not r["route"] and r["code"] == c)
                                             for c in sorted({r["code"] for r in rs if not r["route"]})} for name, rs in rows.items()}
    if a.select:
        cur_dev = res["methods"]["current: soft veto 0.7"]["dev"]
        ok = []
        for name in METHODS:
            if not (name.startswith("conf-aware") or name.startswith("RUGD")):   # multi-source is exploratory: never auto-selected
                continue
            d = res["methods"][name]["dev"]
            safe = (d["false_safe_label"] <= cur_dev["false_safe_label"] + 1e-9 or d["n_STOP"] == 0) and d["false_safe_hazard"] <= cur_dev["false_safe_hazard"] + 1e-9
            ok.append((name, d["false_stop"], safe))
        cand = [x for x in ok if x[2]]
        chosen = min(cand, key=lambda x: (x[1], x[0]))[0] if cand else None
        res["selection"] = {"rule": "min dev false_stop subject to dev false_safe (label and hazard) <= current soft veto; ties -> larger w", "candidates": ok, "chosen": chosen}
        print("selection:", res["selection"], flush=True)
    Path(a.output).write_text(json.dumps(res, indent=1, default=float))
    rows_flat = [r for rs in rows.values() for r in rs]
    Path("reports/leaf_litter/per_frame" + ("_rugd" if a.with_rugd else "") + ".json").write_text(json.dumps(rows_flat, default=float))
    fmt = lambda v: "n/a" if v != v else f"{100 * v:.0f}%"  # noqa: E731
    for role in ("dev", "final_test"):
        print(f"\n=== {role} ===")
        print(f"{'method':44s} {'avail':>6s} {'false-stop':>10s} {'FS 95% CI':>14s} {'false-safe(label)':>18s} {'false-safe(hazard)':>19s} {'side':>5s} {'err':>5s} {'recall':>6s}")
        for name in METHODS:
            d = res["methods"][name][role]
            ci = d["false_stop_ci"]
            print(f"{name:44s} {fmt(d['availability']):>6s} {fmt(d['false_stop']):>10s} {f'[{100*ci[0]:.0f},{100*ci[1]:.0f}]':>14s} {fmt(d['false_safe_label']):>18s} {fmt(d['false_safe_hazard']):>19s} {fmt(d['side_ok']):>5s} {d['abs_err_deg']:5.1f} {fmt(d['gt_recall']):>6s}")
        d0 = res["methods"]["classical"][role]
        print(f"n: labelled={d0['n_labeled']} GO={d0['n_GO']} STOP={d0['n_STOP']} hazard-frames={d0['n_hazard_frames']} GT-frames={d0['n_gt_frames']}")


if __name__ == "__main__":
    main()
