"""GUI: upload a forest image -> tree/obstacle detections + traversable ground + planned local route + direction.

    .venv/bin/streamlit run app.py --server.port 8517
"""
import cv2
import numpy as np
import streamlit as st

from src import nav_render
from src.navigation import NavConfig, STOP
from src.pipeline import MODELS, PATH_METHODS, Models, ModelUnavailable, NavParams, run_navigation
from src.traversability import TravConfig

st.set_page_config(page_title="Forest Navigation", layout="wide")
st.title("Forest off-road navigation: trees, traversable ground and local route")


@st.cache_resource
def models() -> Models:
    return Models()


trunk_files = sorted(p.name for p in MODELS.glob("tree_trunk*.pt"))

with st.sidebar:
    st.header("Settings")
    trunk_w = st.selectbox("Tree detection model", trunk_files,
                           index=trunk_files.index("tree_trunk.pt") if "tree_trunk.pt" in trunk_files else 0) if trunk_files else None
    method = st.radio("Path segmentation method", PATH_METHODS,
                      format_func=lambda m: {
                          "Terrain-aware (classical + RUGD)": "Terrain-aware (plants + terrain types; not yet evaluated)",
                          "Classical baseline": "Classical baseline (fast, best on trails; no safety veto)",
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
        normalize = st.checkbox("Contrast normalisation (CLAHE) for dark images", True)

up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if up is None:
    st.info("Upload a forest image to see tree detections, traversable ground and the planned route.")
    st.stop()
if trunk_w is None:
    st.error("No tree model found. Copy a tree_trunk*.pt file into models/ (see README: Model setup).")
    st.stop()
img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if img is None:
    st.error("Could not read that image (corrupt or unsupported format).")
    st.stop()

params = NavParams(
    conf=conf, imgsz=imgsz, path_method=method, trunk_weights=trunk_w, device=device, allow_grass=allow_grass,
    nav=NavConfig(lookahead=lookahead / 100, safety_margin=margin / 100, smoothing=smoothing,
                  min_confidence=min_conf, min_clearance=clear / 100),
    trav=TravConfig(min_confidence=p_trav, normalize_lighting=normalize),
)
try:
    with st.spinner("Running detection and navigation..."):
        out = run_navigation(img, models(), params)
except ModelUnavailable as exc:
    st.error(f"Model problem: {exc}")
    st.stop()
except Exception as exc:  # inference failure: report, never fake a path
    st.error(f"Inference failed: {type(exc).__name__}: {exc}")
    st.stop()

rgb = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB)  # noqa: E731
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
