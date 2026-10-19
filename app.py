"""Simple GUI: upload a forest image -> tree-trunk detections + drivable path.

    .venv/bin/streamlit run app.py --server.port 8517
"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src import infer

MODELS = infer.PROJECT_ROOT / "models"

st.set_page_config(page_title="Forest Path & Tree Detection", layout="wide")
st.title("Forest path & tree detection")


@st.cache_resource
def yolo(path: str):
    return infer.load_yolo(path)


@st.cache_resource
def rugd():
    return infer.load_rugd()


trunk_files = sorted(p.name for p in MODELS.glob("tree_trunk*.pt"))
path_files = sorted(p.name for p in MODELS.glob("forest_path_seg*.pt"))
has_rugd = (infer.RUGD_DIR / "model.safetensors").exists()

with st.sidebar:
    st.header("Settings")
    trunk_w = st.selectbox("Trunk model", trunk_files, index=trunk_files.index("tree_trunk.pt") if "tree_trunk.pt" in trunk_files else 0) if trunk_files else None
    path_opts = (["YOLO path model"] if path_files else []) + (["RUGD-Segformer (slow)"] if has_rugd else []) + ["Off"]
    path_mode = st.radio("Path method", path_opts)
    path_w = st.selectbox("Path model", path_files) if path_files and path_mode == "YOLO path model" else None
    conf = st.slider("Trunk confidence", 0.05, 0.9, 0.4, 0.05)
    imgsz = st.select_slider("Inference size", [480, 640, 800, 960], value=640)
    st.caption(f"Device: {infer.device()}")

up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if up is None:
    st.info("Upload a forest image to see tree detections and the drivable path.")
    st.stop()
if trunk_w is None:
    st.error("No trunk model found. Put tree_trunk*.pt in models/.")
    st.stop()

img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if img is None:
    st.error("Could not read that image.")
    st.stop()

with st.spinner("Running models..."):
    boxes = infer.detect_trunks(yolo(str(MODELS / trunk_w)), img, conf, imgsz)
    path = None
    if path_mode == "YOLO path model" and path_w:
        path = infer.segment_path_yolo(yolo(str(MODELS / path_w)), img, 0.25, imgsz)
    elif path_mode.startswith("RUGD"):
        path = infer.segment_path_rugd(rugd(), img)

rgb = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB)  # noqa: E731
c1, c2, c3 = st.columns(3)
c1.metric("Trees detected", len(boxes))
if path is not None:
    c2.metric("Path coverage", f"{path.coverage:.0%}")
    if np.isnan(path.offset):
        c3.metric("Path direction", "no path found")
    else:
        side = "ahead" if abs(path.offset) < 0.1 else ("right" if path.offset > 0 else "left")
        c3.metric("Path direction", side, f"{path.offset:+.2f}", delta_color="off")

a, b, c = st.columns(3)
a.image(rgb(img), caption="Input", use_container_width=True)
b.image(rgb(infer.draw_trunks(img, boxes)), caption=f"Tree detections ({len(boxes)})", use_container_width=True)
c.image(rgb(infer.draw_path(img, path)), caption="Drivable path" if path is not None else "Path off", use_container_width=True)

st.image(rgb(infer.draw_trunks(infer.draw_path(img, path), boxes)), caption="Combined", use_container_width=True)
