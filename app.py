"""GUI: upload a forest image or video -> tree/obstacle detections + traversable ground + planned local route + direction.

    .venv/bin/streamlit run app.py --server.port 8517
"""
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import nav_render
from src.navigation import NavConfig, STOP
from src.pipeline import MODELS, Models, ModelUnavailable, NavParams, available_methods, run_navigation
from src.traversability import TravConfig
from src.video_nav import VIDEO_EXT, iter_navigation, video_info, write_webm

st.set_page_config(page_title="Forest Navigation", layout="wide")
st.title("Forest off-road navigation: upload an image or video")


@st.cache_resource
def models() -> Models:
    return Models()


trunk_files = sorted(p.name for p in MODELS.glob("tree_trunk*.pt"))

with st.sidebar:
    st.header("Settings")
    trunk_w = st.selectbox("Tree detection model", trunk_files,
                           index=trunk_files.index("tree_trunk.pt") if "tree_trunk.pt" in trunk_files else 0) if trunk_files else None
    method = st.radio("Path segmentation method", available_methods(),
                      format_func=lambda m: {
                          "Terrain-aware (classical + RUGD)": "Terrain-aware (plants + terrain types; not yet evaluated)",
                          "Classical baseline": "Classical baseline (fast, best on trails; no safety veto)",
                          "Terrain-aware (classical + DeepScene)": "Terrain-aware + DeepScene (safer, fast; plants & terrain)",
                          "DeepScene FCN": "DeepScene FCN only (trail/grass/vegetation/obstacle)",
                          "Terrain-aware (classical + ADE20K)": "Terrain-aware + ADE20K (misses snow)",
                          "ADE20K SegFormer": "ADE20K SegFormer only",
                          "Classical + obstacle veto": "Classical + obstacle veto (safer, needs RUGD, ~5 s)",
                          "RUGD + classical hybrid": "RUGD + classical hybrid (cautious)",
                          "RUGD-Segformer": "RUGD-Segformer only (pretrained)",
                          "YOLO path model": "YOLO path model (groomed track only)",
                          "Off": "Off"}.get(m, m))
    conf = st.slider("Tree confidence threshold", 0.05, 0.9, 0.4, 0.05)
    imgsz = st.select_slider("Inference resolution", [320, 480, 640, 800, 960], value=480)
    margin = st.slider("Obstacle safety margin (% of image width)", 0.0, 8.0, 2.0, 0.5)
    lookahead = st.slider("Look-ahead distance (% of image height)", 15, 70, 40, 5)
    smoothing = st.slider("Path smoothing strength", 0.0, 1.0, 0.5, 0.1)
    min_conf = st.slider("Minimum path confidence", 0.0, 0.9, 0.35, 0.05)
    device = st.selectbox("Device", ["auto", "cpu", "cuda"])
    with st.expander("Advanced"):
        p_trav = st.slider("Ground confidence needed (P traversable)", 0.2, 0.9, 0.5, 0.05)
        clear = st.slider("Minimum corridor half-width (% of width)", 0.5, 5.0, 1.5, 0.5)
        allow_grass = st.checkbox("Allow driving over grass (at a cost)", True, help="Terrain-aware method only; bushes, water, rocks and trees are always blocked.")
        soft = st.checkbox("Soft obstacle/vegetation veto (terrain-aware)", True,
                           help="Block a pixel only where the terrain model's obstacle+vegetation probability is high, instead of wherever those classes win. "
                                "More routes found on the ground-truth check; weaker veto. Does not fix leaf-litter false stops.")
        veto_t = st.slider("Veto threshold", 0.5, 0.95, 0.7, 0.05) if soft else None
        normalize = st.checkbox("Contrast normalisation (CLAHE) for dark images", True)

up = st.file_uploader("Upload an image or a video", type=["jpg", "jpeg", "png"] + [e[1:] for e in VIDEO_EXT])
if up is None:
    st.info("Upload a forest image or video to see tree detections, traversable ground, the planned route and obstacle avoidance.")
    st.stop()
if trunk_w is None:
    st.error("No tree model found. Copy a tree_trunk*.pt file into models/ (see README: Model setup).")
    st.stop()

params = NavParams(
    conf=conf, imgsz=imgsz, path_method=method, trunk_weights=trunk_w, device=device, allow_grass=allow_grass, veto_threshold=veto_t,
    nav=NavConfig(lookahead=lookahead / 100, safety_margin=margin / 100, smoothing=smoothing,
                  min_confidence=min_conf, min_clearance=clear / 100),
    trav=TravConfig(min_confidence=p_trav, normalize_lighting=normalize),
)
rgb = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB)  # noqa: E731


def small(img: np.ndarray, w: int = 640) -> np.ndarray:
    return cv2.resize(img, (w, max(1, int(img.shape[0] * w / img.shape[1]))), interpolation=cv2.INTER_AREA)


def render_video(upload) -> None:
    """Process a video: progress bar, annotated clip, frame explorer, charts, downloads."""
    st.subheader("Video navigation")
    c1, c2 = st.columns(2)
    stride = c1.slider("Process every Nth frame", 1, 15, 3, help="Higher = faster; the route is smoothed across processed frames.")
    max_frames = c2.slider("Maximum frames to process", 10, 300, 60, 10)
    suffix = Path(upload.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(upload.getvalue())
        path = tmp.name
    try:
        try:
            info = video_info(path)
        except ValueError as exc:
            st.error(str(exc))
            return
        n_proc = min(max_frames, max(1, -(-info.frames // stride))) if info.frames > 0 else max_frames
        st.caption(f"{info.width}x{info.height}, {info.fps:.1f} fps, {info.frames} frames -> about {n_proc} frames to process "
                   f"(method: {method}). Estimated time depends on the method: ~0.3 s/frame classical, ~5 s/frame with RUGD on the Jetson CPU.")
        key = (upload.name, upload.size, repr(params), stride, max_frames)
        if st.session_state.get("video_key") != key:
            st.session_state.pop("video_result", None)
        if "video_result" not in st.session_state:
            if not st.button("Process video", type="primary"):
                return
            bar, txt = st.progress(0.0), st.empty()
            rows, panels, navs, t0 = [], [], [], time.perf_counter()
            try:
                for n, (idx, frame, out) in enumerate(iter_navigation(path, models(), params, stride, max_frames), 1):
                    nav_img = nav_render.draw_navigation(frame, out.trav, out.plan, out.direction)
                    det_img = nav_render.draw_detections(frame, out.boxes, out.trav, out.plants)
                    panels.append((small(frame), small(det_img), small(nav_img)))
                    navs.append(small(nav_img, 960))
                    m = out.metrics
                    rows.append({"frame": idx, "time_s": round(idx / info.fps, 2), "trunks": len(out.boxes),
                                 "direction": out.direction.label if out.direction else "OFF",
                                 "angle_deg": None if not out.direction or out.direction.angle_deg is None else round(out.direction.angle_deg, 1),
                                 "path_coverage_pct": round(m.get("path_coverage_pct", 0.0), 1), "route_confidence": round(m.get("path_confidence", 0.0), 2),
                                 "route": bool(m.get("continuous_route")), "latency_s": round(out.timings["total_s"], 2),
                                 "stop_reason": m.get("stop_reason", "")})
                    bar.progress(min(1.0, n / max(1, n_proc)))
                    txt.caption(f"frame {n}/{n_proc} · {rows[-1]['direction']} · {n / (time.perf_counter() - t0):.2f} fps so far")
            except ModelUnavailable as exc:
                st.error(f"Model problem: {exc}")
                return
            except Exception as exc:
                st.error(f"Processing failed: {type(exc).__name__}: {exc}")
                return
            bar.empty()
            txt.empty()
            if not rows:
                st.error("No frames could be read from this video.")
                return
            out_fps = max(1.0, info.fps / stride)
            webm = None
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as wf:
                wpath = wf.name
            if write_webm(navs, wpath, out_fps):
                webm = Path(wpath).read_bytes()
            os.unlink(wpath)
            st.session_state.update(video_key=key, video_result={"rows": rows, "panels": panels, "webm": webm, "fps": out_fps,
                                                                 "elapsed": time.perf_counter() - t0})
        res = st.session_state["video_result"]
    finally:
        os.unlink(path)

    df = pd.DataFrame(res["rows"])
    ok = df["route"].mean()
    k = st.columns(4)
    k[0].metric("Frames processed", len(df))
    k[1].metric("Frames with a safe route", f"{ok:.0%}")
    k[2].metric("STOP frames", int((df["direction"] == STOP).sum()))
    k[3].metric("Measured speed", f"{len(df) / res['elapsed']:.2f} fps")
    if res["webm"]:
        st.video(res["webm"], format="video/webm")
        st.download_button("Download annotated video (WebM)", res["webm"], "navigation.webm", "video/webm")
    else:
        st.warning("This machine's OpenCV cannot encode WebM; use the frame viewer below and the CSV.")
    i = st.slider("Frame viewer", 1, len(df), 1)
    row = df.iloc[i - 1]
    st.caption(f"Source frame {int(row['frame'])} ({row['time_s']} s) · {row['direction']} · coverage {row['path_coverage_pct']:.0f}% · "
               f"confidence {row['route_confidence']:.2f}" + (f" · {row['stop_reason']}" if row["stop_reason"] else ""))
    a, b, c = st.columns(3)
    a.image(rgb(res["panels"][i - 1][0]), caption="Original", width="stretch")
    b.image(rgb(res["panels"][i - 1][1]), caption="Plants, trees & obstacles", width="stretch")
    c.image(rgb(res["panels"][i - 1][2]), caption="Navigation", width="stretch")
    st.line_chart(df.set_index("frame")[["angle_deg", "path_coverage_pct"]])
    st.caption("Heading angle (+ = right, gaps = STOP) and path coverage over the processed frames.")
    with st.expander("Per-frame table"):
        st.dataframe(df, width="stretch")
    st.download_button("Download per-frame CSV", df.to_csv(index=False).encode(), "navigation.csv", "text/csv")


if Path(up.name).suffix.lower() in VIDEO_EXT:
    render_video(up)
    st.stop()

img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if img is None:
    st.error("Could not read that image (corrupt or unsupported format).")
    st.stop()
try:
    with st.spinner("Running detection and navigation..."):
        out = run_navigation(img, models(), params)
except ModelUnavailable as exc:
    st.error(f"Model problem: {exc}")
    st.stop()
except Exception as exc:  # inference failure: report, never fake a path
    st.error(f"Inference failed: {type(exc).__name__}: {exc}")
    st.stop()

if method == "Off":
    st.warning("Path method is Off: navigation is disabled. Only tree detection is shown.")

m = out.metrics
cols = st.columns(6)
cols[0].metric("Trees detected", len(out.boxes))
if m:
    cols[1].metric("Path coverage", f"{m['path_coverage_pct']:.0f}%", help="traversable pixels / valid (non-sky) pixels in the lower 60% of the image")
    cols[2].metric("Traversable area", f"{m['traversable_area_pct']:.0f}%", help="traversable pixels / all valid pixels")
    cols[3].metric("Route confidence", f"{m['path_confidence']:.2f}")
    cols[4].metric("Continuous route", "yes" if m["continuous_route"] else "no")
    cols[5].metric("Direction", out.direction.label)
    if out.plan.success:
        note = f"Clearance: ~{m['clearance_px']:.0f}px half-width"
        note += f" · partial route ({out.plan.reached_fraction:.0%} of look-ahead)" if out.plan.partial else ""
        st.caption(note)
    else:
        st.error(f"{STOP}: {out.plan.reason}")

if out.plants:
    src = out.plants.get("source", "")
    txt = ", ".join(f"{n} {v:.0f}%" for n, v in list(out.plants["summary"].items())[:7])
    st.caption(f"🌿 Terrain / plants ({src}): {txt}" + (f" · {len(out.plants['bushes'])} bush patch(es)" if out.plants.get("bushes") else ""))

a, b, c = st.columns(3)
a.image(rgb(img), caption="Original", width="stretch")
b.image(rgb(nav_render.draw_detections(img, out.boxes, out.trav, out.plants)), caption="Plants, trees & obstacles", width="stretch")
if out.trav is not None:
    c.image(rgb(nav_render.draw_navigation(img, out.trav, out.plan, out.direction)), caption="Navigation (visual estimate)", width="stretch")
else:
    c.info("Navigation disabled (path method Off).")

t = out.timings
st.caption(" · ".join(f"{k[:-2]} {v:.2f}s" for k, v in t.items()) + f"  ·  {1 / t['total_s']:.2f} FPS (this image, incl. all stages)")
for w in dict.fromkeys(out.warnings):
    st.caption(f"ℹ️ {w}")
with st.expander("How to read this / limitations"):
    st.markdown(
        "- **Visual estimate only.** A single 2D image without depth, camera calibration or vehicle geometry cannot guarantee "
        "a collision-free route. Direction labels are steering *suggestions*, not validated control commands.\n"
        "- **Path coverage** = traversable pixels ÷ valid (non-sky) pixels in the lower 60% of the image.\n"
        "- A route is reported only if a continuous, confident corridor with enough clearance connects the start (bottom centre, "
        "shifting sideways at a cost) to the look-ahead row; otherwise you get STOP / NO SAFE PATH.\n"
        "- Grass/bushes are never treated as safe. The classical baseline has no semantic understanding."
    )
