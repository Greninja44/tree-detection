"""Forest Navigator: upload a forest image or video -> safe route, obstacles and heading.

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
from src.navigation import STOP, NavConfig
from src.pipeline import FAST, MODELS, NAVIGATION, Models, ModelUnavailable, NavParams, available_methods, run_navigation
from src.traversability import TravConfig
from src.video_nav import VIDEO_EXT, iter_navigation, video_info, write_webm

st.set_page_config(page_title="Forest Navigator", page_icon="🌲", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.status {border-radius: 14px; padding: 18px 20px; margin-bottom: 14px; color: #fff;}
.status.go {background: linear-gradient(135deg, #1f8a4c, #2fb866);}
.status.stop {background: linear-gradient(135deg, #b3261e, #e0463c);}
.status .big {font-size: 1.7rem; font-weight: 700; line-height: 1.15;}
.status .sub {opacity: .92; margin-top: 6px; font-size: .95rem;}
.status .code {opacity: .75; margin-top: 6px; font-size: .75rem; letter-spacing: .04em;}
.chip {display:inline-block; padding: 2px 10px; margin: 2px 4px 2px 0; border-radius: 999px; background: rgba(120,120,120,.18); font-size: .82rem;}
</style>
""",
    unsafe_allow_html=True,
)

REASONS = {
    "NO_TRAVERSABLE_REGION": "No drivable ground was found in front of the vehicle.",
    "OBSTACLE_BLOCKS_CORRIDOR": "An obstacle (tree, person, machine or bush) blocks the way ahead.",
    "LOW_CONFIDENCE_TERRAIN": "A route exists, but the terrain is too uncertain to trust.",
    "INSUFFICIENT_CORRIDOR_WIDTH": "The gap ahead is too narrow to pass.",
    "NO_VALID_CONNECTED_CORRIDOR": "The ground ahead is broken up - no continuous path reaches far enough.",
}


@st.cache_resource
def models() -> Models:
    return Models()


def rgb(a: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def small(img: np.ndarray, w: int = 640) -> np.ndarray:
    return cv2.resize(img, (w, max(1, int(img.shape[0] * w / img.shape[1]))), interpolation=cv2.INTER_AREA)


def status_card(out) -> None:
    plan = out.plan
    if plan is not None and plan.success:
        label = out.direction.label.title() if out.direction else "Go"
        sub = f"Route confidence {plan.confidence:.2f} · clearance ≈ {out.metrics['clearance_px']:.0f}px"
        if plan.partial:
            sub += f" · ground visible for {plan.reached_fraction:.0%} of the look-ahead distance"
        st.markdown(f'<div class="status go"><div class="big">GO · {label}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)
    else:
        why = REASONS.get(plan.reason_code, plan.reason) if plan is not None else "Navigation did not run."
        code = plan.reason_code if plan is not None else ""
        st.markdown(f'<div class="status stop"><div class="big">STOP</div><div class="sub">{why}</div><div class="code">{code}</div></div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------------------------- sidebar
available = available_methods()
with st.sidebar:
    st.title("🌲 Forest Navigator")
    st.caption("Upload an image or video. The route, obstacles and heading are estimated from the picture alone.")
    fast = st.toggle("Fast preview", value=(NAVIGATION not in available), disabled=(NAVIGATION not in available),
                     help="Colour/texture ground estimate only (~0.4 s/frame). It has no obstacle checks and can route through people, "
                          "machines and dense vegetation. The default Navigation mode takes ~5 s/frame on the Jetson CPU.")
    if NAVIGATION not in available:
        st.warning("Navigation models are missing (models/rugd_segformer, models/deepscene), so only the Fast preview is available. See the README.")
    with st.expander("Advanced"):
        conf = st.slider("Tree detection confidence", 0.05, 0.9, 0.4, 0.05)
        imgsz = st.select_slider("Image size", [320, 480, 640, 800, 960], value=480)
        margin = st.slider("Obstacle safety margin (% of width)", 0.0, 8.0, 2.0, 0.5)
        lookahead = st.slider("Look-ahead distance (% of height)", 15, 70, 40, 5)
        smoothing = st.slider("Route smoothing", 0.0, 1.0, 0.5, 0.1)
        min_conf = st.slider("Minimum route confidence", 0.0, 0.9, 0.35, 0.05)
        device = st.selectbox("Device", ["auto", "cpu", "cuda"])
        allow_grass = st.checkbox("Allow driving over grass (at a cost)", True)
    st.caption("A visual estimate only: no depth or calibration, so it cannot guarantee a collision-free path.")

trunk_files = sorted(p.name for p in MODELS.glob("tree_trunk*.pt"))
if not trunk_files:
    st.error("No tree model found. Copy a tree_trunk*.pt file into models/ (see README: Model setup).")
    st.stop()

params = NavParams(
    conf=conf, imgsz=imgsz, path_method=FAST if fast else NAVIGATION, allow_grass=allow_grass,
    trunk_weights="tree_trunk.pt" if "tree_trunk.pt" in trunk_files else trunk_files[0], device=device,
    nav=NavConfig(lookahead=lookahead / 100, safety_margin=margin / 100, smoothing=smoothing, min_confidence=min_conf),
    trav=TravConfig(),
)

st.markdown("### Upload an image or a video")
up = st.file_uploader("Drag and drop, or browse", type=["jpg", "jpeg", "png"] + [e[1:] for e in VIDEO_EXT], label_visibility="collapsed")
if up is None:
    st.info("Supported: JPG, PNG, MP4, AVI, MOV, MKV, WEBM. You'll get a GO / STOP decision, the planned route, detected obstacles, trees and terrain.")
    st.stop()


# ---------------------------------------------------------------------------------------------------------------- video
def render_video(upload) -> None:
    st.markdown("#### Video")
    c1, c2 = st.columns(2)
    stride = c1.slider("Process every Nth frame", 1, 15, 3, help="Higher = faster. The route is smoothed across processed frames.")
    max_frames = c2.slider("Maximum frames", 10, 300, 60, 10)
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
        per = 0.45 if params.path_method == FAST else 5.5
        st.caption(f"{info.width}×{info.height} · {info.fps:.1f} fps · {info.frames} frames → about {n_proc} frames to process "
                   f"(≈ {n_proc * per / 60:.1f} min on the Jetson CPU in {'Fast preview' if params.path_method == FAST else 'Navigation'} mode).")
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
                    rows.append({"frame": idx, "time_s": round(idx / info.fps, 2), "trees": len(out.boxes),
                                 "decision": out.direction.label if out.direction else "-",
                                 "angle_deg": None if not out.direction or out.direction.angle_deg is None else round(out.direction.angle_deg, 1),
                                 "path_coverage_pct": round(m.get("path_coverage_pct", 0.0), 1), "confidence": round(m.get("path_confidence", 0.0), 2),
                                 "route": bool(m.get("continuous_route")), "latency_s": round(out.timings["total_s"], 2),
                                 "stop_code": m.get("stop_code", ""), "stop_reason": m.get("stop_reason", "")})
                    bar.progress(min(1.0, n / max(1, n_proc)))
                    txt.caption(f"frame {n}/{n_proc} · {rows[-1]['decision']} · {n / (time.perf_counter() - t0):.2f} fps so far")
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
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as wf:
                wpath = wf.name
            webm = Path(wpath).read_bytes() if write_webm(navs, wpath, out_fps) else None
            os.unlink(wpath)
            st.session_state.update(video_key=key, video_result={"rows": rows, "panels": panels, "webm": webm, "elapsed": time.perf_counter() - t0})
        res = st.session_state["video_result"]
    finally:
        os.unlink(path)

    df = pd.DataFrame(res["rows"])
    k = st.columns(4)
    k[0].metric("Frames processed", len(df))
    k[1].metric("Frames with a safe route", f"{df['route'].mean():.0%}")
    k[2].metric("STOP frames", int((df["decision"] == STOP).sum()))
    k[3].metric("Speed", f"{len(df) / res['elapsed']:.2f} fps")
    t_play, t_frames, t_chart, t_table = st.tabs(["▶ Annotated video", "🎞 Frame viewer", "📈 Charts", "📄 Table & downloads"])
    with t_play:
        if res["webm"]:
            st.video(res["webm"], format="video/webm")
            st.download_button("Download video (WebM)", res["webm"], "navigation.webm", "video/webm")
        else:
            st.warning("This machine's OpenCV cannot encode WebM - use the frame viewer and the CSV.")
    with t_frames:
        i = st.slider("Frame", 1, len(df), 1)
        row = df.iloc[i - 1]
        st.caption(f"Source frame {int(row['frame'])} ({row['time_s']} s) · {row['decision']} · coverage {row['path_coverage_pct']:.0f}% · confidence {row['confidence']:.2f}"
                   + (f" · {REASONS.get(row['stop_code'], row['stop_reason'])}" if row["stop_code"] else ""))
        a, b, c = st.columns(3)
        a.image(rgb(res["panels"][i - 1][0]), caption="Original", width="stretch")
        b.image(rgb(res["panels"][i - 1][1]), caption="Detections", width="stretch")
        c.image(rgb(res["panels"][i - 1][2]), caption="Navigation", width="stretch")
    with t_chart:
        st.line_chart(df.set_index("frame")[["angle_deg", "path_coverage_pct"]])
        st.caption("Heading angle (+ = right, gaps = STOP) and path coverage over the processed frames.")
    with t_table:
        st.dataframe(df, width="stretch")
        st.download_button("Download per-frame CSV", df.to_csv(index=False).encode(), "navigation.csv", "text/csv")


if Path(up.name).suffix.lower() in VIDEO_EXT:
    render_video(up)
    st.stop()

# ---------------------------------------------------------------------------------------------------------------- image
img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if img is None:
    st.error("Could not read that image (corrupt or unsupported format).")
    st.stop()
try:
    with st.spinner("Analysing the scene…" if fast else "Analysing the scene (about 5 s)…"):
        out = run_navigation(img, models(), params)
except ModelUnavailable as exc:
    st.error(f"Model problem: {exc}")
    st.stop()
except Exception as exc:  # inference failure: report, never fake a path
    st.error(f"Inference failed: {type(exc).__name__}: {exc}")
    st.stop()

left, right = st.columns([2.2, 1], gap="large")
with right:
    status_card(out)
    m = out.metrics
    c1, c2 = st.columns(2)
    c1.metric("Trees detected", len(out.boxes))
    c2.metric("Path coverage", f"{m['path_coverage_pct']:.0f}%", help="Traversable pixels ÷ valid (non-sky) pixels in the lower 60% of the image.")
    c3, c4 = st.columns(2)
    c3.metric("Route confidence", f"{m['path_confidence']:.2f}")
    c4.metric("Traversable area", f"{m['traversable_area_pct']:.0f}%")
    if out.plants:
        chips = "".join(f'<span class="chip">{n} {v:.0f}%</span>' for n, v in list(out.plants["summary"].items())[:6])
        extra = f'<span class="chip">{len(out.plants["bushes"])} bush patch(es)</span>' if out.plants.get("bushes") else ""
        st.markdown("**Terrain & plants**<br>" + chips + extra, unsafe_allow_html=True)
    if out.method_used == FAST:
        st.warning("Fast preview: no obstacle checks - it can route through people, machines and dense vegetation.")
    with st.expander("Details"):
        t = out.timings
        st.caption(" · ".join(f"{k[:-2]} {v:.2f}s" for k, v in t.items()) + f" · {1 / t['total_s']:.2f} FPS (this image)")
        for w in dict.fromkeys(out.warnings):
            st.caption(f"ℹ️ {w}")
        st.markdown(
            "- **Visual estimate only.** One 2D image without depth, calibration or vehicle geometry cannot guarantee a collision-free route; "
            "the heading is a steering *suggestion*, not a validated command.\n"
            "- **Path coverage** = traversable pixels ÷ valid pixels in the lower 60% of the image.\n"
            "- A route is shown only if a continuous, confident corridor with enough clearance connects the start (bottom centre) to the look-ahead row; otherwise STOP.\n"
            "- Grass is passable at a higher cost; bushes, trees, people, vehicles and water are blocked."
        )
with left:
    t_nav, t_det, t_orig = st.tabs(["🧭 Navigation", "🌿 Detections", "🖼 Original"])
    t_nav.image(rgb(nav_render.draw_navigation(img, out.trav, out.plan, out.direction)), width="stretch")
    t_nav.caption("Green = drivable ground · red = obstacles · yellow = safety margin / uncertain · cyan = planned route · arrow = heading.")
    t_det.image(rgb(nav_render.draw_detections(img, out.boxes, out.trav, out.plants)), width="stretch")
    t_det.caption("Orange boxes = tree trunks · terrain tints = grass, bushes, trees, people, vehicles … (Navigation mode).")
    t_orig.image(rgb(img), width="stretch")
