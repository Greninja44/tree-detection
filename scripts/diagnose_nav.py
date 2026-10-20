"""Trace one frame through every navigation stage and test counterfactuals (which component causes a STOP?).

    python scripts/diagnose_nav.py data/images/test/ft_img2374.jpg --out outputs/leaf_litter

Writes <out>/diag_<stem>.jpg (8 panels) and prints/returns a JSON-able dict with per-stage statistics and counterfactual
planner outcomes: full pipeline, DeepScene veto off, trunk footprints off, classical only.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import infer  # noqa: E402
from src.navigation import NavConfig, plan_local_path  # noqa: E402
from src.pipeline import Models, NavParams  # noqa: E402
from src.traversability import (TravConfig, detect_traversable_region, fuse_terrain_aware, make_deepscene_segmenter,  # noqa: E402
                                normalize_lighting, segment_classical)

DS_COLORS = np.array([[75, 155, 200], [100, 210, 85], [20, 100, 15], [0, 185, 255], [255, 120, 0]], np.uint8)  # BGR: trail grass veg obstacle sky


def deepscene_probs(onnx: str, img: np.ndarray) -> np.ndarray:
    net = cv2.dnn.readNetFromONNX(onnx)
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (576, 320)).astype(np.float32) / 255.0
    net.setInput(((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32))
    lg = net.forward()[0]
    e = np.exp(lg - lg.max(0, keepdims=True))
    p = e / e.sum(0, keepdims=True)
    return np.stack([cv2.resize(c, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR) for c in p])


def heat(a: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap((np.clip(a, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


def tint(img, mask, color, a=0.5):
    o = img.copy()
    o[mask] = ((1 - a) * o[mask] + a * np.array(color, np.float32)).astype(np.uint8)
    return o


def outcome(plan) -> str:
    return "ROUTE" if plan.success else "STOP"


def diagnose(img: np.ndarray, models: Models, params: NavParams, out_path: Path | None = None) -> dict:
    H, W = img.shape[:2]
    device = "cpu"
    norm, _ = normalize_lighting(img)
    boxes = infer.detect_trunks(models.yolo(params.trunk_weights), img, params.conf, params.imgsz, device)
    onnx = str(models.deepscene_path())
    ds_primary = make_deepscene_segmenter(onnx)(norm)
    classical = segment_classical(norm, params.trav)
    probs = deepscene_probs(onnx, norm)
    soft = fuse_terrain_aware(ds_primary, classical, True, params.veto_threshold)
    nav, tc = params.nav, params.trav

    def run(seg, boxes_=boxes):
        t = detect_traversable_region(img, boxes_, lambda _i, _s=seg: _s, tc)
        return t, plan_local_path(t.traversable, t.obstacles, nav, t.p_trav)

    t_full, p_full = run(soft)
    t_nov, p_nov = run(fuse_terrain_aware(ds_primary, classical, True, 1.01))       # DeepScene veto disabled (threshold > 1)
    t_nof, p_nof = run(soft, [])                                                      # trunk footprints off
    t_cl, p_cl = run(classical)                                                       # classical only
    roi = np.zeros((H, W), bool); roi[int(0.4 * H):] = True
    cl_ground = t_cl.traversable                                                      # the floor the classical method found
    n_ground = max(1, int(cl_ground.sum()))
    argmax = probs.argmax(0)
    stats = {
        "classical_ground_px_pct_of_roi": 100 * cl_ground.sum() / max(1, roi.sum()),
        "on_classical_ground": {
            "mean_p_trail": float(probs[0][cl_ground].mean()), "mean_p_grass": float(probs[1][cl_ground].mean()),
            "mean_p_vegetation": float(probs[2][cl_ground].mean()), "mean_p_obstacle": float(probs[3][cl_ground].mean()),
            "argmax_share_pct": {n: float(100 * (argmax[cl_ground] == i).mean()) for i, n in enumerate(["trail", "grass", "vegetation", "obstacle", "sky"])},
            "p_block_ge_0.5_pct": float(100 * (ds_primary.p_block[cl_ground] >= 0.5).mean()),
            "p_block_ge_0.7_pct": float(100 * (ds_primary.p_block[cl_ground] >= 0.7).mean()),
            "p_block_ge_0.85_pct": float(100 * (ds_primary.p_block[cl_ground] >= 0.85).mean()),
            "vetoed_pct_at_threshold": float(100 * (soft.p_trav[cl_ground] == 0).mean()),
            "mean_classical_p": float(classical.p_trav[cl_ground].mean()),
        },
        "trunks": len(boxes), "trunk_footprint_px_pct_of_classical_ground": float(100 * (t_full.obstacles & cl_ground).sum() / n_ground),
        "outcomes": {"full": outcome(p_full), "no_deepscene_veto": outcome(p_nov), "no_trunk_footprints": outcome(p_nof), "classical_only": outcome(p_cl)},
        "reasons": {"full": p_full.reason, "no_deepscene_veto": p_nov.reason, "no_trunk_footprints": p_nof.reason, "classical_only": p_cl.reason},
        "traversable_pct_of_roi": {"full": float(100 * (t_full.traversable & roi).sum() / roi.sum()), "no_deepscene_veto": float(100 * (t_nov.traversable & roi).sum() / roi.sum()),
                                   "classical_only": float(100 * (t_cl.traversable & roi).sum() / roi.sum())},
    }
    if out_path is not None:
        def ov(t, p):
            o = tint(img, t.traversable, (60, 200, 60), 0.4); o = tint(o, t.obstacles, (40, 40, 230), 0.5)
            pts = (p.points if p.success else p.candidate_points)
            if len(pts) > 1:
                cv2.polylines(o, [pts.astype(np.int32).reshape(-1, 1, 2)], False, (255, 200, 0) if p.success else (0, 0, 255), 4, cv2.LINE_AA)
            return o
        panels = [("1 original", img), ("2 classical P(ground)", heat(classical.p_trav)), ("3 DeepScene argmax", DS_COLORS[argmax]),
                  ("4 DeepScene P(obstacle+veg)", heat(ds_primary.p_block)), ("5 trunks + obstacles", tint(img, t_full.obstacles, (40, 40, 230), 0.6)),
                  ("6 fused P(ground) [veto 0.7]", heat(soft.p_trav)), (f"7 FULL -> {outcome(p_full)}", ov(t_full, p_full)),
                  (f"8 veto off -> {outcome(p_nov)}", ov(t_nov, p_nov))]
        tiles = []
        for name, im in panels:
            t = cv2.resize(im, (480, 270)); cv2.rectangle(t, (0, 0), (480, 20), (0, 0, 0), -1)
            cv2.putText(t, name, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA); tiles.append(t)
        cv2.imwrite(str(out_path), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 8, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="outputs/leaf_litter")
    ap.add_argument("--veto-threshold", type=float, default=0.7)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    models, params = Models(), NavParams(path_method="Terrain-aware (classical + DeepScene)", veto_threshold=a.veto_threshold)
    res = {}
    for f in a.images:
        img = cv2.imread(f)
        res[Path(f).stem] = diagnose(img, models, params, out / f"diag_{Path(f).stem}.jpg")
        print(Path(f).stem, json.dumps(res[Path(f).stem]["outcomes"]), flush=True)
    (out / "diagnosis.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
