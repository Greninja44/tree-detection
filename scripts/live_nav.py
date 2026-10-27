"""Live webcam navigation: camera -> trunk detection + traversable ground + local route + direction.

    python scripts/live_nav.py                       # camera 0, Fast preview, view at http://<jetson-ip>:8518
    python scripts/live_nav.py --method Navigation   # adds RUGD/DeepScene obstacle veto (~5 s/frame on this Jetson)
    python scripts/live_nav.py --show                # local window instead (needs a display)

The camera is read on its own thread so the model always sees the *newest* frame (slow models drop frames instead of
building up lag).  The annotated frame is served as MJPEG so it works headless over the network.  Route: the
traversable-ground estimate is generic (colour/texture, plus semantic models in Navigation mode); it is not validated
for offices or farms, and without depth or calibration it is a visual route on the image, not a metric path.
"""
import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import nav_render  # noqa: E402
from src.navigation import DirectionSmoother, NavConfig, resample_path  # noqa: E402
from src.pipeline import DEPTH, PATH_METHODS, Models, ModelUnavailable, NavParams, run_navigation  # noqa: E402


class LatestFrame:
    """Background reader that keeps only the most recent camera frame."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap, self.frame, self.ok, self.lock = cap, None, True, threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while self.ok:
            ok, f = self.cap.read()
            if not ok:
                self.ok = False
                return
            with self.lock:
                self.frame = f

    def get(self):
        with self.lock:
            f, self.frame = self.frame, None
        return f


class Stream:
    jpeg: bytes | None = None


def serve(port: int) -> None:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/frame.jpg":
                if Stream.jpeg is None:
                    self.send_error(503)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(Stream.jpeg)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                try:
                    last = None
                    while True:
                        j = Stream.jpeg
                        if j is not None and j is not last:
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j) + j + b"\r\n")
                            last = j
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                body = b"<html><body style='margin:0;background:#111'><img src='/stream' style='width:100%;max-width:1280px'></body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body)

    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--method", default=DEPTH, choices=PATH_METHODS)
    ap.add_argument("--weights", default="tree_trunk.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--port", type=int, default=8518)
    ap.add_argument("--show", action="store_true", help="Open a local window (needs a display); q quits.")
    ap.add_argument("--no-trunks", action="store_true", help="Skip the forest trunk detector (use outside forests).")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {a.camera}. Check /dev/video* and that nothing else is using it.")
        return 2
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    nav = NavConfig()
    params = NavParams(conf=a.conf, imgsz=a.imgsz, path_method=a.method, trunk_weights=a.weights, device=a.device, nav=nav)
    models, smoother = Models(), DirectionSmoother(config=nav, start_stopped=False)
    if a.no_trunks:
        import src.infer as infer
        infer.detect_trunks = lambda *args, **kw: []
    if not a.show:
        threading.Thread(target=serve, args=(a.port,), daemon=True).start()
        print(f"Live view: http://<jetson-ip>:{a.port}   (Ctrl+C to stop)", flush=True)
    reader, prev_pts, ema, last = LatestFrame(cap), None, None, time.perf_counter()
    try:
        while reader.ok:
            img = reader.get()
            if img is None:
                time.sleep(0.005)
                continue
            try:
                o = run_navigation(img, models, params, smoother)
            except ModelUnavailable as exc:
                print(f"ERROR: {exc}")
                return 3
            if o.plan is not None and o.plan.success:
                cur = resample_path(o.plan.points)
                prev_pts = cur if prev_pts is None else 0.5 * cur + 0.5 * prev_pts
                o.plan.points = prev_pts
            else:
                prev_pts = None
            vis = nav_render.draw_navigation(img, o.trav, o.plan, o.direction)
            now = time.perf_counter()
            fps = 1.0 / max(now - last, 1e-6)
            last, ema = now, fps if ema is None else 0.9 * ema + 0.1 * fps
            label = f"{o.method_used} | {ema:.1f} FPS | {o.direction.label if o.direction else 'OFF'}"
            cv2.putText(vis, label, (8, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(vis, label, (8, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if a.show:
                cv2.imshow("live navigation", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    Stream.jpeg = buf.tobytes()
            print(f"\r{label}  warnings={len(o.warnings)}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        reader.ok = False
        cap.release()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
