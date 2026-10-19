"""Run the navigation pipeline on an image, a folder of images, or a video, with temporal smoothing.

    python scripts/navigate.py --source forest1.jpg
    python scripts/navigate.py --source clip.mp4 --method "Classical baseline" --stride 2

Writes annotated frames/video + navigation.csv to --output and prints *measured* latency and FPS.
For video, the route is smoothed across frames (EMA on arc-length-resampled points) and the direction is
smoothed with hysteresis; a missing route stops immediately and needs two valid frames to resume.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import nav_render  # noqa: E402
from src.navigation import DirectionSmoother, NavConfig, resample_path  # noqa: E402
from src.pipeline import PATH_METHODS, Models, ModelUnavailable, NavParams, run_navigation  # noqa: E402
from src.preprocessing import SUPPORTED_IMAGE_EXTENSIONS  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Image, folder of images, or video file.")
    ap.add_argument("--output", default="outputs/navigation")
    ap.add_argument("--method", default=PATH_METHODS[0], choices=PATH_METHODS)
    ap.add_argument("--weights", default="tree_trunk.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--lookahead", type=float, default=0.40)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--smoothing", type=float, default=0.5)
    ap.add_argument("--min-confidence", type=float, default=0.35)
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth video frame.")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    src, out = Path(a.source), Path(a.output)
    if not src.exists():
        print(f"ERROR: source not found: {src}")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    nav = NavConfig(lookahead=a.lookahead, safety_margin=a.margin, smoothing=a.smoothing, min_confidence=a.min_confidence)
    params = NavParams(conf=a.conf, imgsz=a.imgsz, path_method=a.method, trunk_weights=a.weights, device=a.device, nav=nav)
    models, smoother = Models(), DirectionSmoother(config=nav, start_stopped=False)
    is_video = src.suffix.lower() in VIDEO_EXT
    rows, lat, prev_pts, writer = [], [], None, None

    def frames():
        if is_video:
            cap = cv2.VideoCapture(str(src))
            if not cap.isOpened():
                raise SystemExit(f"ERROR: could not open video {src}")
            i = 0
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                if i % a.stride == 0:
                    yield f"frame_{i:06d}", f
                i += 1
            cap.release()
        else:
            files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
            for p in files:
                f = cv2.imread(str(p))
                if f is None:
                    print(f"WARN: unreadable image skipped: {p}")
                    continue
                yield p.stem, f

    try:
        for n, (name, img) in enumerate(frames()):
            if a.max_frames and n >= a.max_frames:
                break
            o = run_navigation(img, models, params, smoother if is_video else None)
            if is_video and o.plan is not None and o.plan.success:  # temporal route smoothing
                cur = resample_path(o.plan.points)
                prev_pts = cur if prev_pts is None else 0.5 * cur + 0.5 * prev_pts
                o.plan.points = prev_pts
            elif is_video:
                prev_pts = None
            vis = nav_render.draw_navigation(img, o.trav, o.plan, o.direction)
            lat.append(o.timings["total_s"])
            if is_video:
                if writer is None:
                    writer = cv2.VideoWriter(str(out / "navigation.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            else:
                cv2.imwrite(str(out / f"nav_{name}.jpg"), vis)
            m = o.metrics
            rows.append({"frame": name, "trunks": len(o.boxes), "direction": o.direction.label if o.direction else "OFF",
                         "path_coverage_pct": round(m.get("path_coverage_pct", 0), 1), "route_confidence": round(m.get("path_confidence", 0), 2),
                         "continuous": int(bool(m.get("continuous_route"))), "latency_s": round(o.timings["total_s"], 3),
                         "stop_reason": m.get("stop_reason", "")})
            print(rows[-1], flush=True)
    except ModelUnavailable as exc:
        print(f"ERROR: {exc}")
        return 3
    finally:
        if writer is not None:
            writer.release()
    if not rows:
        print("ERROR: nothing processed.")
        return 4
    with open(out / "navigation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lat_a = np.array(lat)
    print(f"{len(rows)} frame(s): mean latency {lat_a.mean():.2f}s (median {np.median(lat_a):.2f}s) = {1 / lat_a.mean():.2f} FPS measured -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
