"""Forest Navigator: live webcam -> safe route, obstacles and heading.

    .venv/bin/streamlit run app.py --server.port 8517
"""

import cv2
import numpy as np
import streamlit as st

from src import nav_render
from src.camera import Camera, CameraError, list_cameras
from src.navigation import DirectionSmoother, NavConfig
from src.pipeline import DEPTH, FAST, MODELS, NAVIGATION, Models, ModelUnavailable, NavParams, available_methods, run_navigation
from src.traversability import TravConfig

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
available = [m for m in (DEPTH, NAVIGATION, FAST) if m in available_methods()]
with st.sidebar:
    st.title("🌲 Forest Navigator")
    st.caption("Live camera input only. The route, obstacles and heading are estimated from the picture alone.")
    labels = {DEPTH: "RGB depth (~1 s/frame, recommended)", NAVIGATION: "Navigation (~5 s/frame, slower)", FAST: "Fast preview (no obstacle checks)"}
    method = st.radio("Method", available, format_func=labels.get,
                      help="RGB depth: monocular depth + fitted ground plane; anything standing proud of the ground is an obstacle. "
                           "Navigation: segmentation models; slower, so the route lags the camera. "
                           "Fast preview: colour/texture only; it can route through people, machines and dense vegetation.")
    if len(available) < 3:
        st.warning("Some models are missing (models/depth_anything_v2_small, models/rugd_segformer, models/deepscene); see the README.")
    cams = list_cameras() if "cams" not in st.session_state else st.session_state["cams"]
    st.session_state["cams"] = cams
    cam_index = st.selectbox("Camera", cams or [0], format_func=lambda i: f"/dev/video{i}")
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
    conf=conf, imgsz=imgsz, path_method=method, allow_grass=allow_grass,
    trunk_weights="tree_trunk.pt" if "tree_trunk.pt" in trunk_files else trunk_files[0], device=device,
    nav=NavConfig(lookahead=lookahead / 100, safety_margin=margin / 100, smoothing=smoothing, min_confidence=min_conf),
    trav=TravConfig(),
)


st.markdown("### Live camera")
live = st.toggle("Start camera", key="live", help="Opens the webcam on the Jetson. Turn it off to release the camera for other programs.")
left, right = st.columns([2.2, 1], gap="large")
view, side = left.empty(), right.empty()

if not live:
    st.session_state.pop("cam", None)
    view.info("Camera is off. Switch on **Start camera** to get a live GO / STOP decision, route, obstacles and trees.")
    st.stop()

try:
    cam = Camera(cam_index)
except CameraError as exc:
    st.error(str(exc))
    st.stop()

for k in [k for k in models()._cache if k[0] == "depth_seg"]:   # new session: forget the previous ground plane
    models()._cache.pop(k)
smoother = DirectionSmoother(config=params.nav, start_stopped=True)
try:
    while True:   # a widget change reruns the script, which ends this loop and releases the camera
        frame = cam.read()
        out = run_navigation(frame, models(), params, smoother)
        view.image(rgb(nav_render.draw_navigation(frame, out.trav, out.plan, out.direction)), channels="RGB", width="stretch",
                   caption="Green = drivable ground · red = obstacles · yellow = margin/uncertain · cyan = route · arrow = heading.")
        with side.container():
            status_card(out)
            m = out.metrics
            c1, c2 = st.columns(2)
            c1.metric("Trees detected", len(out.boxes))
            c2.metric("Path coverage", f"{m['path_coverage_pct']:.0f}%")
            c3, c4 = st.columns(2)
            c3.metric("Route confidence", f"{m['path_confidence']:.2f}")
            c4.metric("Speed", f"{1 / out.timings['total_s']:.2f} FPS")
            if out.plants:
                chips = "".join(f'<span class="chip">{n} {v:.0f}%</span>' for n, v in list(out.plants["summary"].items())[:6])
                st.markdown("**Terrain & plants**<br>" + chips, unsafe_allow_html=True)
            if out.method_used == FAST:
                st.warning("Fast preview: no obstacle checks - it can route through people, machines and dense vegetation.")
            for w in dict.fromkeys(out.warnings):
                st.caption(f"ℹ️ {w}")
except CameraError as exc:
    st.error(str(exc))
except ModelUnavailable as exc:
    st.error(f"Model problem: {exc}")
except Exception as exc:  # inference failure: report, never fake a path
    st.error(f"Inference failed: {type(exc).__name__}: {exc}")
finally:
    cam.release()
