"""Interactive Streamlit application for tree-trunk detection under varying lighting.

Sections
--------
1. **Detection**          – image / video / folder detection with annotated output.
2. **Lighting Analysis**  – measured statistics, real-world labels, synthetic variants.
3. **Evaluation Dashboard** – batch evaluation on a labeled split + charts + export.
4. **Model Settings**     – checkpoint selection, inference size, device info.

Launch::

    .venv/bin/streamlit run app.py

The app stays usable with **no custom weights**: it shows a setup message
instead of silently running a generic model that cannot detect tree trunks.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import (  # noqa: E402
    DetectorError,
    DetectorUnavailableError,
    TreeDetector,
    WeightsNotFoundError,
    available_devices,
    cuda_functional,
    list_available_weights,
    models_dir,
)
from src.evaluation import (  # noqa: E402
    EvaluationReport,
    evaluate_predictions,
    export_report_csv,
    export_report_json,
    load_yolo_label_file,
)
from src.lighting_analysis import (  # noqa: E402
    REAL_WORLD_CATEGORIES,
    analyze_lighting,
    describe_category,
    load_category_labels,
)
from src.lighting_augmentation import (  # noqa: E402
    TRANSFORMATION_NAMES,
    TransformationError,
    generate_variants,
    synthetic_category,
)
from src.preprocessing import (  # noqa: E402
    InputValidationError,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    list_images_in_folder,
)
from src.synthetic_fixtures import synthetic_forest_image  # noqa: E402
from src.video_processor import process_video  # noqa: E402
from src.visualization import (  # noqa: E402
    draw_detections,
    generate_all_charts,
    plot_confidence_distribution,
    plot_precision_recall_by_category,
    plot_map_by_category,
    plot_inference_time_by_category,
    plot_count_error_by_category,
    side_by_side,
)

st.set_page_config(
    page_title="Tree Trunk Detection – Lighting Robustness",
    page_icon="🌲",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Caching helpers
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_detector(
    weights: str, confidence: float, iou: float, device: str, imgsz: int
) -> tuple[TreeDetector | None, str | None]:
    """Build (and cache) a detector.

    Returns ``(detector, error_message)`` — never raises, so the UI can show a
    setup message when weights/backend are missing.  The underlying model
    object itself is cached module-side, so threshold changes do not reload it.
    """
    try:
        det = TreeDetector(
            weights=weights,
            confidence_threshold=confidence,
            iou_threshold=iou,
            device=device or None,
            imgsz=imgsz,
        )
        return det, None
    except (WeightsNotFoundError, DetectorUnavailableError, DetectorError) as exc:
        return None, str(exc)
    except Exception as exc:  # pragma: no cover - unexpected backend errors
        return None, f"Unexpected model loading error: {exc}"


def decode_uploaded_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise InputValidationError("Could not decode the uploaded image (corrupted file?).")
    return image


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise IOError("Failed to encode PNG output.")
    return buf.tobytes()


def setup_message(problem: str) -> None:
    """Uniform, explicit setup message when no usable model is available."""
    st.warning("### Model not ready — detection disabled")
    st.code(problem, language="text")
    st.markdown(
        """
**How to fix this**

1. Obtain or train a *tree-trunk* checkpoint (see README → *Training the detector*):
   ```bash
   python scripts/train.py --model yolov8n.pt --data configs/dataset.yaml --epochs 80
   ```
2. Copy the resulting weights into `models/` (or select them in **Model Settings**):
   ```bash
   cp runs/train/tree_trunk_exp/weights/best.pt models/tree_trunk.pt
   ```
3. Rerun this section.

A generic pretrained model (COCO) has no `tree_trunk` class, so this app will
**not** silently use one to fake tree detections.
"""
    )


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

DEFAULTS: dict[str, Any] = {
    "labels": {},                     # filename -> assigned real-world category
    "last_image": None,               # np.ndarray BGR of the active image
    "last_image_name": "",
    "last_result": None,              # DetectionResult
    "last_annotated": None,           # np.ndarray
    "eval_report": None,              # EvaluationReport
    "eval_dir": None,                 # Path of last evaluation export
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


# --------------------------------------------------------------------------- #
# Sidebar: model configuration (shared by all tabs)
# --------------------------------------------------------------------------- #

st.sidebar.title("🌲 Tree Detection")
st.sidebar.caption("Robustness of tree-trunk detection under different lighting conditions.")

available = list_available_weights()
weight_options = ["(none)"] + [p.name for p in available]
selected = st.sidebar.selectbox(
    "Checkpoint in models/",
    weight_options,
    help=f"Scanned from {models_dir()}",
)
selected_path = "" if selected == "(none)" else str(models_dir() / selected)
weights_input = st.sidebar.text_input(
    "…or checkpoint path",
    value=selected_path,
    help="Explicit path wins over the dropdown selection. Clear it to follow the dropdown.",
)
device_options = ["auto", *available_devices()]
device_choice = st.sidebar.selectbox("Device", device_options, index=0)
imgsz = st.sidebar.number_input("Inference size (px)", 320, 1920, 640, step=32)
conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou_threshold = st.sidebar.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05)

# An explicit path always wins; otherwise fall back to the dropdown choice
# (the text input keeps its own widget state across reruns).
effective_weights = weights_input.strip() or selected_path
detector, det_error = (None, "No checkpoint selected.")
if effective_weights:
    detector, det_error = get_detector(
        effective_weights,
        float(conf_threshold),
        float(iou_threshold),
        "" if device_choice == "auto" else device_choice,
        int(imgsz),
    )

if detector is not None:
    st.sidebar.success(
        f"Model: `{detector.model_label}`\n\n"
        f"Device: **{detector.backend_device}**\n\n"
        f"Tree class: **{'yes' if detector.is_tree_model else 'NO (generic baseline)'}**"
    )
else:
    st.sidebar.error("Model not loaded — see setup message in the tabs.")

with st.sidebar.expander("Hardware / environment"):
    cuda_ok = cuda_functional()
    st.write(
        {
            "requested devices": available_devices(),
            "CUDA functional": "yes" if cuda_ok else "no",
            "inference backend": "GPU" if cuda_ok else "CPU",
        }
    )
    if not cuda_ok:
        st.caption(
            "torch.cuda.is_available() may still report True while kernels are "
            "unsupported for this GPU; the app verifies CUDA with a functional "
            "probe and falls back to CPU."
        )

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

tab_detect, tab_light, tab_eval, tab_model = st.tabs(
    ["🔎 Detection", "🌗 Lighting Analysis", "📊 Evaluation Dashboard", "⚙️ Model Settings"]
)


# --------------------------------------------------------------------------- #
# Tab 1: Detection
# --------------------------------------------------------------------------- #
with tab_detect:
    st.header("Detection")
    uploaded = st.file_uploader(
        "Upload an image or a video",
        type=[e.lstrip(".") for e in SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS],
        key="detect_uploader",
    )
    col_a, col_b = st.columns(2)

    if uploaded is None:
        st.info(
            "Upload a forest image/video to run detection. "
            "You can also load a synthetic demo fixture below to test the pipeline UI."
        )
        if st.button("Load synthetic demo image (fixture, not real forestry data)"):
            image, _boxes = synthetic_forest_image()
            st.session_state.last_image = image
            st.session_state.last_image_name = "synthetic_demo.jpg"
            st.session_state.last_result = None
            st.session_state.last_annotated = None

    if st.session_state.last_image is None and uploaded is not None:
        data = uploaded.getvalue()
        try:
            if uploaded.name.lower().endswith(tuple(e.lstrip(".") for e in SUPPORTED_VIDEO_EXTENSIONS)):
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(uploaded.name).suffix or ".mp4"
                )
                tmp.write(data)
                tmp.close()
                st.session_state.video_path = tmp.name
                st.session_state.last_image = None
            else:
                st.session_state.last_image = decode_uploaded_image(data)
                st.session_state.last_image_name = uploaded.name
                st.session_state.last_result = None
                st.session_state.last_annotated = None
        except InputValidationError as exc:
            st.error(str(exc))

    # --- image path ---------------------------------------------------- #
    if st.session_state.get("video_path") and st.session_state.last_image is None:
        video_path = st.session_state.video_path
        max_frames = st.number_input("Max frames to process", 1, 10_000, 100, 10)
        if st.button("Run video detection", type="primary"):
            if detector is None:
                setup_message(det_error or "Model unavailable.")
            else:
                out_path = PROJECT_ROOT / "outputs" / f"{Path(video_path).stem}_annotated.mp4"
                with st.spinner("Processing video…"):
                    try:
                        summary = process_video(
                            detector, video_path, out_path, max_frames=int(max_frames)
                        )
                    except (InputValidationError, DetectorError) as exc:
                        st.error(str(exc))
                        summary = None
                if summary is not None:
                    st.json(summary.to_dict())
                    if summary.output_path and Path(summary.output_path).exists():
                        with open(summary.output_path, "rb") as fh:
                            st.download_button(
                                "⬇️ Download annotated video",
                                fh.read(),
                                file_name=Path(summary.output_path).name,
                                mime="video/mp4",
                            )
                    if summary.errors:
                        st.warning("Warnings during processing:\n" + "\n".join(summary.errors[:10]))

    elif st.session_state.last_image is not None:
        image = st.session_state.last_image
        if st.button("Run detection", type="primary"):
            if detector is None:
                setup_message(det_error or "Model unavailable.")
            else:
                t0 = time.perf_counter()
                try:
                    result = detector.detect(image, source_name=st.session_state.last_image_name)
                except DetectorError as exc:
                    st.error(str(exc))
                    result = None
                wall_ms = (time.perf_counter() - t0) * 1000.0
                if result is not None:
                    st.session_state.last_result = result
                    st.session_state.last_annotated = draw_detections(image, result)

        result = st.session_state.get("last_result")
        annotated = st.session_state.get("last_annotated")
        if result is not None and annotated is not None:
            with col_a:
                st.subheader("Original")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            with col_b:
                st.subheader("Annotated")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Detected trees", result.count)
            m2.metric("Inference time", f"{result.inference_time_ms:.1f} ms")
            m3.metric("Instantaneous FPS", f"{result.fps:.1f}")
            m4.metric("Confidence ≥ thr.", f"{conf_threshold:.2f}")
            if detector is not None and not detector.is_tree_model:
                st.error(
                    f"⚠️ `{detector.model_label}` has no `tree_trunk` class — the boxes above are a "
                    "**generic model baseline**, not validated tree detections."
                )
            if result.count == 0:
                st.info("No detections above the confidence threshold.")

            with st.expander("Detection records"):
                st.json(result.to_dict())

            st.download_button(
                "⬇️ Download annotated image",
                encode_png(annotated),
                file_name=f"annotated_{st.session_state.last_image_name or 'image.png'}",
                mime="image/png",
            )

            # Lighting stats shortcut
            stats = analyze_lighting(image)
            st.caption(
                f"Lighting stats — mean {stats.mean_gray} · median {stats.median_gray} · "
                f"contrast σ {stats.contrast_std} · dark {stats.dark_pixel_pct}% · "
                f"bright {stats.bright_pixel_pct}% · estimated: "
                f"**{describe_category(stats.estimated_category)}**"
            )
    else:
        st.caption("No image loaded.")


# --------------------------------------------------------------------------- #
# Tab 2: Lighting analysis
# --------------------------------------------------------------------------- #
with tab_light:
    st.header("Lighting Analysis")
    image = st.session_state.get("last_image")
    if image is None:
        st.info(
            "Upload an image in the **Detection** tab (or load the synthetic demo) "
            "to analyze its lighting."
        )
    else:
        name = st.session_state.last_image_name or "image"
        stats = analyze_lighting(image)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Measured statistics")
            st.dataframe(
                pd.DataFrame([stats.to_dict()]),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Brightness/contrast thresholds are heuristics — they describe pixels, "
                "**not** verified weather or time of day."
            )
        with c2:
            st.subheader("Real-world lighting category (assigned)")
            current = st.session_state.labels.get(name, "unlabeled")
            options = ["unlabeled", *REAL_WORLD_CATEGORIES]
            chosen = st.selectbox(
                "Assign a category to this image",
                options,
                index=options.index(current) if current in options else 0,
                key="cat_selector",
            )
            st.session_state.labels[name] = chosen
            est = describe_category(stats.estimated_category)
            st.write(f"Image-based estimate: **{est}**")
            if chosen != "unlabeled" and chosen != stats.estimated_category:
                st.caption(
                    "Note: the assigned label differs from the image-based estimate. "
                    "The assigned (human) label takes precedence in reports."
                )
            if chosen == "unlabeled":
                st.caption("No real-world label assigned — this image counts as 'unlabeled' in reports.")

        st.divider()
        st.subheader("Synthetic lighting variants")
        st.caption(
            "Synthetic conditions are tracked as `synthetic:<name>` and are **never** "
            "mixed up with genuine real-world categories."
        )
        sel = st.multiselect(
            "Transformations",
            list(TRANSFORMATION_NAMES),
            default=["darker", "brighter", "low_contrast", "shadow", "noise"],
        )
        strength = st.slider("Transformation strength", 0.0, 1.0, 0.6, 0.05)

        if st.button("Generate variants and compare detection", type="primary",
                     disabled=detector is None):
            if detector is None:
                setup_message(det_error or "Model unavailable.")
            else:
                with st.spinner("Generating variants and running detection…"):
                    variants = generate_variants(image, names=sel or None, strength=float(strength))
                    rows = []
                    annotated: dict[str, np.ndarray] = {}
                    for vname, var in variants.items():
                        try:
                            res = detector.detect(var, source_name=f"{name}#{vname}")
                        except DetectorError as exc:
                            st.error(str(exc))
                            continue
                        annotated[vname] = draw_detections(var, res)
                        vstats = analyze_lighting(var)
                        rows.append(
                            {
                                "variant": vname,
                                "condition_label": synthetic_category(vname),
                                "mean_gray": vstats.mean_gray,
                                "contrast_std": vstats.contrast_std,
                                "dark_%": vstats.dark_pixel_pct,
                                "detections": res.count,
                                "mean_confidence": round(float(res.confidences.mean()), 4)
                                if res.confidences.size
                                else None,
                                "inference_ms": round(res.inference_time_ms, 2),
                            }
                        )
                    st.session_state.variant_rows = rows
                    st.session_state.variant_images = annotated

                rows = st.session_state.get("variant_rows", [])
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

                variants = st.session_state.get("variant_images", {})
                if variants:
                    cols = st.columns(min(3, len(variants)))
                    for i, (vname, var) in enumerate(variants.items()):
                        with cols[i % len(cols)]:
                            st.caption(f"{describe_category(synthetic_category(vname))}")
                            st.image(cv2.cvtColor(var, cv2.COLOR_BGR2RGB))

                if rows:
                    base = next((r["detections"] for r in rows if r["variant"] == "original"), None)
                    others = [r for r in rows if r["variant"] != "original"]
                    if base is not None and others:
                        deltas = {r["variant"]: r["detections"] - base for r in others}
                        st.write("Detection-count change vs. original:", deltas)


# --------------------------------------------------------------------------- #
# Tab 3: Evaluation dashboard
# --------------------------------------------------------------------------- #
with tab_eval:
    st.header("Evaluation Dashboard")
    st.caption(
        "Batch evaluation on a labeled test split. Accuracy metrics (precision, recall, "
        "mAP) **require ground-truth bounding boxes**; without them they are reported as "
        "unavailable rather than estimated."
    )

    ec1, ec2, ec3 = st.columns(3)
    images_dir = ec1.text_input("Test images folder", value=str(PROJECT_ROOT / "data" / "images" / "test"))
    labels_dir = ec2.text_input("YOLO labels folder (optional)", value=str(PROJECT_ROOT / "data" / "labels" / "test"))
    categories_file = ec3.text_input(
        "Lighting labels JSON (optional)", value=str(PROJECT_ROOT / "data" / "lighting_labels.json")
    )

    if st.button("Run batch evaluation", type="primary", disabled=detector is None):
        if detector is None:
            setup_message(det_error or "Model unavailable.")
        else:
            try:
                image_paths = list_images_in_folder(images_dir)
            except InputValidationError as exc:
                st.error(str(exc))
                image_paths = []

            assigned: dict[str, str] = {}
            cat_path = Path(categories_file)
            if image_paths and cat_path.exists():
                try:
                    assigned = load_category_labels(cat_path)
                except (OSError, ValueError) as exc:
                    st.warning(f"Could not load lighting labels: {exc}")

            preds: dict[str, list] = {}
            gts: dict[str, list] = {}
            cats: dict[str, str] = {}
            times: dict[str, float] = {}
            stats_rows: list[dict[str, Any]] = []

            progress = st.progress(0.0, text="Evaluating…")
            for i, path in enumerate(image_paths):
                try:
                    img = cv2.imread(str(path))
                    if img is None:
                        raise InputValidationError("decode failed")
                except Exception as exc:
                    st.warning(f"Skipping {path.name}: {exc}")
                    continue
                res = detector.detect(img, source_name=path.name)
                preds[path.name] = list(res.detections)
                times[path.name] = res.inference_time_ms
                st_stats = analyze_lighting(img)
                stats_rows.append(st_stats.to_dict())
                if assigned:
                    cats[path.name] = assigned.get(path.name, "unlabeled")
                else:
                    cats[path.name] = st_stats.estimated_category
                label_file = (Path(labels_dir) / path.stem).with_suffix(".txt")
                if label_file.exists():
                    gts[path.name] = load_yolo_label_file(label_file, img.shape[1], img.shape[0])
                progress.progress((i + 1) / max(1, len(image_paths)))
            progress.empty()

            if not preds:
                st.error("No images could be processed — check the folder paths.")
            else:
                report = evaluate_predictions(
                    preds,
                    gts if len(gts) == len(preds) else None,
                    categories=cats,
                    inference_times_ms=times,
                )
                if not assigned:
                    report.note = (
                        "Lighting groups are image-based estimates (no lighting-label JSON found). "
                        + report.note
                    ).strip()
                st.session_state.eval_report = report

                if report.has_ground_truth:
                    st.success(f"Evaluated {report.num_images} images with ground truth.")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Precision", f"{report.precision:.3f}")
                    c2.metric("Recall", f"{report.recall:.3f}")
                    c3.metric("F1", f"{report.f1:.3f}")
                    c4.metric("mAP@0.5", f"{report.map50:.3f}")
                    c5.metric("Count MAE", f"{report.count_mae:.3f}")
                    c6, c7, c8 = st.columns(3)
                    c6.metric("False positives", report.false_positives)
                    c7.metric("False negatives", report.false_negatives)
                    c8.metric("mAP@0.5:0.95", f"{report.map50_95:.3f}")
                else:
                    st.warning(report.unavailable_reason)
                    st.info(
                        "Inference statistics are still reported below. To enable accuracy "
                        "metrics, provide YOLO label files for every evaluated image."
                    )

                st.subheader("Metrics by lighting condition")
                st.dataframe(pd.DataFrame(report.metrics_table()), width="stretch",
                             hide_index=True)
                if report.note:
                    st.caption(report.note)

                st.subheader("Per-image statistics")
                st.dataframe(pd.DataFrame(report.per_image), width="stretch",
                             hide_index=True)

    report: EvaluationReport | None = st.session_state.get("eval_report")
    if report is not None:
        st.divider()
        st.subheader("Comparison charts")
        try:
            figs = []
            if report.has_ground_truth and report.by_category:
                figs.append(("Precision / recall by lighting",
                             plot_precision_recall_by_category(report)))
                figs.append(("mAP by lighting", plot_map_by_category(report)))
                figs.append(("Tree-count error by lighting", plot_count_error_by_category(report)))
            if report.by_category:
                figs.append(("Inference time by lighting",
                             plot_inference_time_by_category(report)))
            confs = [
                r["mean_confidence"]
                for r in report.per_image
                if r.get("mean_confidence") is not None
            ]
            figs.append(("Confidence distribution", plot_confidence_distribution(confs)))
            for title, fig in figs:
                st.write(f"**{title}**")
                st.pyplot(fig, clear_figure=True)
        except Exception as exc:
            st.warning(f"Chart generation failed: {exc}")

        st.subheader("Export")
        export_dir = PROJECT_ROOT / "reports" / "latest"
        try:
            export_report_json(report, export_dir / "metrics.json")
            export_report_csv(report, export_dir / "metrics.csv")
            st.session_state.eval_dir = str(export_dir)
        except Exception as exc:
            st.warning(f"Could not write report files: {exc}")
        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇️ metrics.json",
            report.to_json(indent=2),
            file_name="metrics.json",
            mime="application/json",
        )
        buf = export_report_csv(report, Path(tempfile.mkdtemp()) / "metrics.csv")
        c2.download_button(
            "⬇️ metrics.csv",
            buf.read_bytes(),
            file_name="metrics.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------- #
# Tab 4: Model settings
# --------------------------------------------------------------------------- #
with tab_model:
    st.header("Model Settings")
    st.markdown(
        f"""
**Checkpoint directory:** `{models_dir()}`
Files found: {", ".join(f"`{p.name}`" for p in available) if available else "_none yet_"}

**Device:** requested `{device_choice}` → active **{detector.backend_device if detector else "n/a"}**
CUDA functional probe: **{"yes" if cuda_functional() else "no"}** (CPU inference is used when CUDA kernels are unsupported for this GPU).
"""
    )

    st.subheader("When is a custom tree-trunk model required?")
    st.markdown(
        """
| Checkpoint type | Has `tree_trunk` class? | Usable for tree counting? |
|---|---|---|
| **Custom trained** (`models/tree_trunk.pt`, `runs/.../best.pt`) | yes | ✅ yes |
| **Generic pretrained** (`yolov8n.pt`, COCO, 80 classes) | no | ❌ only as a pipeline baseline |

Generic COCO weights contain no tree/trunk class. This app will run them **only**
if you explicitly select them, and it labels the output as a *generic baseline*
instead of pretending the boxes are trees.

To train a proper model:
```bash
python scripts/train.py --validate-only               # check dataset layout
python scripts/train.py --model yolov8n.pt \\
    --data configs/dataset.yaml --epochs 80 --imgsz 640
cp runs/train/tree_trunk_exp/weights/best.pt models/tree_trunk.pt
```
"""
    )

    st.subheader("Loaded model detail")
    if detector is not None:
        st.json(detector.describe())
    else:
        st.info("No model loaded." + (f" Reason: {det_error}" if det_error else ""))

    if detector is not None and not detector.is_tree_model:
        st.warning(
            f"Checkpoint `{detector.model_label}` exposes classes "
            f"{detector.class_names[:12]} — no tree-trunk class. "
            "Train or supply a custom checkpoint for real tree detections."
        )

st.sidebar.divider()
st.sidebar.caption(
    "Synthetic fixtures and generic baselines are clearly labelled throughout; "
    "no accuracy figures are shown without ground-truth labels."
)
