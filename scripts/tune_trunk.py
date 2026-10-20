"""Tune the trunk detector's inference settings (input size, confidence) on the VAL split, report once on TEST.

    python scripts/tune_trunk.py --weights models/tree_trunk.pt --sizes 640 800 960

Detections are cached in <output>/cache so the sweep can be re-run cheaply.  Matching: greedy by confidence,
IoU >= 0.5.  The test split is only touched for the baseline (640 px, conf 0.4) and the single setting chosen on
validation, so the reported gain is not tuned on test data.
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

EXTS = {".jpg", ".jpeg", ".png"}


def load_gt(label_file: Path, w: int, h: int) -> np.ndarray:
    rows = []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                _, cx, cy, bw, bh = map(float, p)
                rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, np.float32).reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(det: np.ndarray, conf: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """True-positive flag per detection (sorted by descending confidence)."""
    order = np.argsort(-conf)
    tp = np.zeros(len(det), bool)
    used = set()
    iou = iou_matrix(det, gt)
    for i in order:
        if iou.shape[1] == 0:
            continue
        j = int(iou[i].argmax())
        if iou[i, j] >= thr and j not in used:
            tp[i] = True
            used.add(j)
    return tp


def metrics(images: list[dict], thr: float) -> dict:
    tp = fp = fn = 0
    cnt_err = []
    for im in images:
        keep = im["conf"] >= thr
        det, conf, gt = im["det"][keep], im["conf"][keep], im["gt"]
        m = match(det, conf, gt)
        tp += int(m.sum()); fp += int((~m).sum()); fn += len(gt) - int(m.sum())
        cnt_err.append(abs(len(det) - len(gt)))
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"conf": thr, "P": p, "R": r, "F1": 2 * p * r / max(1e-9, p + r), "count_MAE": float(np.mean(cnt_err)), "FP": fp, "FN": fn}


def ap50(images: list[dict]) -> float:
    flags, confs, n_gt = [], [], 0
    for im in images:
        m = match(im["det"], im["conf"], im["gt"])
        flags.append(m); confs.append(im["conf"]); n_gt += len(im["gt"])
    if not flags or n_gt == 0:
        return 0.0
    f, c = np.concatenate(flags), np.concatenate(confs)
    o = np.argsort(-c)
    tpc, fpc = np.cumsum(f[o]), np.cumsum(~f[o])
    rec, prec = tpc / n_gt, tpc / np.maximum(1, tpc + fpc)
    mrec, mpre = np.r_[0, rec, 1], np.r_[0, prec, 0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def predict_split(model, split: str, imgsz: int, cache: Path, iou: float = 0.45, device: str = "cpu") -> list[dict]:
    f = cache / f"{split}_{imgsz}_iou{iou}.npz"
    if f.exists():
        try:
            return list(np.load(f, allow_pickle=True)["images"])
        except Exception:  # truncated by an interrupted run: recompute
            f.unlink()
    out = []
    files = sorted(p for p in (PROJECT_ROOT / "data/images" / split).iterdir() if p.suffix.lower() in EXTS)
    for k, p in enumerate(files):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        r = model.predict(img, conf=0.05, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
        out.append({"name": p.name, "det": r.boxes.xyxy.cpu().numpy().astype(np.float32), "conf": r.boxes.conf.cpu().numpy().astype(np.float32),
                    "gt": load_gt(PROJECT_ROOT / "data/labels" / split / f"{p.stem}.txt", w, h)})
        if k % 50 == 0:
            print(f"  {split} {imgsz}px iou{iou}: {k}/{len(files)}", flush=True)
    tmp = f.with_name(f.stem + ".tmp.npz")
    np.savez_compressed(tmp, images=np.array(out, dtype=object))
    tmp.replace(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/tree_trunk.pt")
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 800, 960])
    ap.add_argument("--ious", type=float, nargs="+", default=[0.45])
    ap.add_argument("--output", default="reports/trunk_tuning")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = Path(a.output)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO(a.weights)
    thr_grid = [round(t, 2) for t in np.arange(0.10, 0.80, 0.05)]
    best, table = None, []
    for sz in a.sizes:
        for iou in a.ious:
            val = predict_split(model, "val", sz, out / "cache", iou, a.device)
            ap_v = ap50(val)
            rows = [metrics(val, t) for t in thr_grid]
            top = max(rows, key=lambda r: r["F1"])
            table.append({"imgsz": sz, "iou": iou, "val_AP50": ap_v, **{f"val_{k}": v for k, v in top.items()}})
            print(f"VAL imgsz={sz} iou={iou}: AP50={ap_v:.3f} best F1={top['F1']:.3f} @conf {top['conf']} (P {top['P']:.2f} R {top['R']:.2f} countMAE {top['count_MAE']:.2f})", flush=True)
            if best is None or top["F1"] > best[0]["F1"]:
                best = (top, sz, iou)
    top, sz, iou = best
    print(f"CHOSEN on val: imgsz={sz} iou={iou} conf={top['conf']}")
    res = {"chosen": {"imgsz": sz, "iou": iou, "conf": top["conf"]}, "val_table": table}
    for name, (s, i, c) in {"baseline (640 px, iou 0.45, conf 0.4)": (640, 0.45, 0.4), "chosen": (sz, iou, top["conf"])}.items():
        test = predict_split(model, "test", s, out / "cache", i, a.device)
        m = metrics(test, c)
        res[f"test_{name.split()[0]}"] = {**m, "AP50": ap50(test), "imgsz": s, "iou": i}
        print(f"TEST {name}: P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} countMAE={m['count_MAE']:.2f} FP={m['FP']} FN={m['FN']} AP50={ap50(test):.3f}", flush=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
