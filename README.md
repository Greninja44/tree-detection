# Robust Tree Trunk Detection and Forest-Path Isolation

A computer-vision pipeline for **detecting tree trunks and isolating the drivable path in forest
imagery** under varying lighting (bright sunlight, partial shade, deep shadow, overcast, low
light), with an emphasis on *measuring* how lighting affects the models.

The project provides:

- a **YOLOv8 tree-trunk detector** trained on public forest datasets (bounding boxes, confidence, tree counts)
- **drivable-path isolation** two ways: a small custom YOLOv8-seg model and the pretrained
  [RUGD-Segformer](https://huggingface.co/JasonTStanley/RUGD-Segformer) (25 off-road classes)
- a **simple GUI** (Streamlit): upload an image -> tree detections + path overlay + steering offset
- batch inference over images, folders and videos; **lighting analysis** and **synthetic lighting
  transformations** (brightness, gamma, contrast, shadow, noise)
- an **evaluation module**: precision, recall, F1, AP, mAP, FP/FN, timing and tree-count MAE,
  **per lighting category**, with charts and CSV/JSON exports
- dataset import scripts, a validated training pipeline, and a **Windows-GPU training bundle**
- **pytest** unit tests (68)

---

## Architecture

Training happens on a GPU PC; the Jetson (CPU-only PyTorch build) is used for inference.

```
 PUBLIC DATA                         WINDOWS PC (RTX GPU)                    JETSON (inference)
 ┌──────────────────────┐   scripts  ┌──────────────────────┐   scp .pt     ┌────────────────────────────┐
 │ ForTrunkDet (Zenodo) │──import_──▶│ pc_bundle/           │──────────────▶│ models/tree_trunk*.pt      │
 │ FinnWoodlands (HF)   │  *.py      │  train_on_pc.py      │               │ models/forest_path_seg.pt  │
 └──────────────────────┘            │  YOLOv8s  (trunks)   │               │ models/rugd_segformer/     │
        │ split: contiguous          │  YOLOv8s-seg (path)  │               └─────────────┬──────────────┘
        ▼ blocks (no leakage)        └──────────────────────┘                             │
 data/{images,labels}/{train,val,test}                                                    ▼
 data_path/...  (YOLO-seg polygons)                         src/infer.py  ◀── app.py (Streamlit GUI)
                                                              │  detect_trunks()   YOLOv8 boxes
                                                              │  segment_path_yolo()  YOLOv8-seg mask
                                                              └  segment_path_rugd()  SegFormer-B5 mask
                                                      scripts/{batch_detect,evaluate,detect_path,rugd_segment}.py
```

| Component | Model | Input -> output |
|---|---|---|
| Trunk detector | YOLOv8s (Ultralytics), single class `tree_trunk` | image -> boxes + confidence + count |
| Path (custom) | YOLOv8s-seg, single class `path` (FinnWoodlands "Track" masks) | image -> path mask, coverage, left/right offset |
| Path (pretrained) | RUGD-Segformer (SegFormer-B5, 339 MB) | image -> 25-class map; `dirt/sand/gravel/mulch/asphalt` = drivable, `tree/log` = trees |

The "offset" is the lateral centre of the path in the lower quarter of the frame
(-1 far left, 0 centred, +1 far right) - a steering hint, not a planner.

### Data used

| Dataset | Used for | Size after import | License |
|---|---|---|---|
| [ForTrunkDet](https://zenodo.org/records/5213825) (Portuguese forests, 4 cameras) | trunk boxes | 2,029 visible-light images (thermal skipped) | CC BY 4.0 |
| [FinnWoodlands](https://huggingface.co/datasets/Voxel51/FinnWoodlands) (Finnish forests) | trunk boxes + path masks | 250 images | (c) Springer Nature, research use - check before commercial use |

Combined trunk dataset: **1,595 train / 343 val / 341 test** images. Splits are contiguous blocks
per camera/site, because the images are sequential video frames (random splits leak).
Images are not committed; recreate them with `scripts/import_fortrunkdet.py` and
`scripts/import_finnwoodlands.py` (see Section 5).

### Results so far (held-out test split, confidence 0.4)

Trunk detector (YOLOv8s, 640 px, trained 100 epochs on an RTX 5080), 341 test images:

| | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Count MAE | CPU time/frame (Jetson) |
|---|---|---|---|---|---|---|
| First CPU-trained baseline (YOLOv8n, 175 imgs) | 0.52 | 0.49 | 0.36 | 0.16 | 1.98 | 0.34 s |
| **Current (YOLOv8s, 1,595 imgs)** | **0.73** | **0.71** | **0.64** | **0.38** | **1.29** | 0.99 s |

By lighting category (categories are **image-statistics estimates**, not human labels):

| Category | Images | mAP@0.5 | Recall |
|---|---|---|---|
| deep_shadow | 193 | 0.65 | 0.73 |
| partial_shade | 113 | 0.65 | 0.71 |
| bright_daylight | 28 | **0.24** | **0.25** |
| low_light | 7 | 0.77 | 0.81 (too few images to trust) |

Path model (YOLOv8s-seg): test mAP@0.5 0.995, mAP@0.5:0.95 about 0.72 - **on only 37 images /
17 path instances from one Finnish forest, so treat as unproven.** It learns the narrow groomed
"Track" class, not every drivable surface. RUGD-Segformer marks all drivable-looking ground
(more general, but also marks any forest floor, and runs about 8 s/frame on the Jetson CPU).

---

## Remaining work

- [ ] **Higher-accuracy trunk run** (planned, not run): `python train_on_pc.py --only trunk --trunk-model yolov8m.pt --imgsz 960 --batch 8 --epochs 150 --patience 50 --tag _m960`
- [ ] **Bright-daylight / overcast data**: bright-sun recall is only 0.25 and no dataset here covers all five lighting conditions. Collect or find more, then retrain.
- [ ] **Real lighting labels**: create `data/lighting_labels.json`; current categories are estimated from pixels.
- [ ] **Path models compared on real frames**: custom YOLOv8-seg vs RUGD-Segformer side by side, with a quantitative metric, on images from the target robot's camera. Gather path labels beyond one Finnish forest.
- [ ] **Real-time on Jetson**: install a JetPack-matched PyTorch (or export to TensorRT) so inference uses the GPU; today it is about 1 s/frame (trunks) and about 8 s/frame (RUGD) on CPU.
- [ ] **Tree-count accuracy**: count MAE is 1.29 per image; tune the confidence threshold on the val split (never the test split) and check the dataset's unlabelled-trunk effect on false positives.
- [ ] **Lighting robustness A/B** with the new weights (synthetic darker/brighter/low-contrast/shadow variants; scripts exist, results not yet generated). The old Streamlit Lighting Analysis and Evaluation tabs were removed in the GUI rewrite - the CLI (`scripts/evaluate.py`) still does this; re-add to the GUI if wanted.
- [ ] **Model weights hosting**: `*.pt` is git-ignored; publish trained weights (release / Hugging Face) so others need not retrain.
- [ ] **Licenses**: RUGD-Segformer weights list no license; FinnWoodlands is research-use. Confirm before any commercial deployment.
- [ ] **Temporal-leakage audit**: splits are contiguous blocks, but frames at block borders are still near-neighbours.

---

## 1. System requirements

- **Python 3.10+** (developed and tested on Python 3.12)
- ~2 GB free disk for dependencies (PyTorch wheels are large)
- **CPU inference works fine** for `yolov8n`-class models.
- GPU is optional. CUDA is auto-detected *and functionally verified*: if the installed
  PyTorch build has no kernels for the local GPU (common with generic PyPI wheels on
  Jetson/Orin), the app falls back to CPU instead of crashing.

Libraries: OpenCV, Ultralytics YOLO, PyTorch, NumPy, Pandas, Matplotlib, Seaborn,
Streamlit, Transformers (for RUGD-Segformer), pytest, PyYAML.

---

## 2. Setup

```bash
cd tree-detection

# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run the automated tests
python -m pytest
```

---

## 3. Launch the application

```bash
.venv/bin/streamlit run app.py --server.port 8517
```

Open `http://<host>:8517`, upload a `.jpg`/`.png`, and the page shows the **tree detections**
(count + boxes), the **drivable path** (green overlay, coverage, left/ahead/right offset) and a
combined view. The sidebar picks the trunk model (any `models/tree_trunk*.pt`), the path method
(YOLO path model / RUGD-Segformer / off), trunk confidence and inference size. The inference
logic lives in `src/infer.py`, so it can be reused without the GUI.

Command-line equivalents:

```bash
python scripts/batch_detect.py --source data/images/test --output outputs/test --weights models/tree_trunk.pt --conf 0.4
python scripts/detect_path.py  --source data/images/test --weights models/forest_path_seg.pt
python scripts/rugd_segment.py --source data/images/test --max-images 8
```

---

## 4. Supplying compatible model weights

The detector accepts any **Ultralytics YOLO checkpoint** whose class list contains a
tree/trunk-like class (e.g. `tree_trunk`).

```bash
# recommended: put checkpoints in models/
cp /path/to/your/best.pt models/tree_trunk.pt
```

or pass an explicit path with the `--weights` CLI flag, or pick a `models/tree_trunk*.pt` file in the GUI sidebar.

Resolution rules (`src/detector.py`):

1. exact path given → must exist, otherwise a `WeightsNotFoundError` explains what was searched;
2. otherwise `<models-dir>/<name>.pt` (models dir = `models/`, override with env var
   `TREE_DETECTION_MODELS_DIR`);
3. if the checkpoint's classes contain **no** tree/trunk name (COCO models such as
   `yolov8n.pt`), the run proceeds **only** as an explicitly-labelled *generic baseline*
   and every output warns that these are not validated tree detections.

**Optional baseline download (not a tree detector):**

```bash
.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

This fetches the COCO-pretrained `yolov8n.pt` so you can smoke-test the pipeline
end-to-end. It cannot detect tree trunks — use it to verify mechanics, not accuracy.

---

## 5. Dataset acquisition and annotation

No dataset ships with this repository. To train, you need a **YOLO-format** dataset of
forest images with `tree_trunk` bounding boxes.

### 5.1 Required layout

```
data/
├── images/
│   ├── train/   forest_001.jpg ...
│   ├── val/     forest_101.jpg ...
│   └── test/    forest_201.jpg ...   # FINAL evaluation only
└── labels/
    ├── train/   forest_001.txt ...
    ├── val/     forest_101.txt ...
    └── test/    forest_201.txt ...
```

Label format — one line per box, normalized to `[0, 1]`:

```
0 0.512 0.640 0.084 0.512     # class cx cy w h
```

### 5.2 Obtaining a dataset (verify before use)

We deliberately **do not** point to an unverified download. Before using any public
forestry/tree dataset, verify:

1. **download source** — a stable URL you can actually fetch;
2. **license** — permits your intended use (training + publication);
3. **annotations** — real bounding boxes or masks (not just image-level tags);
4. **suitability** — images show *trunks* at a scale your application needs;
5. **class mapping** — rename/map classes to `tree_trunk` in `configs/dataset.yaml`.

Datasets are commonly found on Roboflow Universe, Kaggle and academic project pages —
each must pass the checklist above. Alternatively, annotate your own images with
[CVAT](https://cvat.ai), [Label Studio](https://labelstud.io) or `labelImg`, then export
in YOLO format.

### 5.3 Validate before training

```bash
.venv/bin/python scripts/train.py --validate-only --data configs/dataset.yaml
```

Checks: config keys, split folders exist, images present, label/image pairing,
5-field YOLO lines, class-id range and normalized coordinates.

### 5.4 Lighting labels (for per-category reporting)

Optional JSON mapping filenames to real-world categories:

```json
{
  "forest_001.jpg": "bright_daylight",
  "forest_002.jpg": "partial_shade",
  "forest_003.jpg": "deep_shadow",
  "forest_004.jpg": "overcast",
  "forest_005.jpg": "low_light"
}
```

Valid keys: `bright_daylight`, `partial_shade`, `deep_shadow`, `overcast`, `low_light`
(anything else is treated as `unlabeled`). Without this file, reports fall back to
**image-based estimates** and say so — brightness thresholds are *heuristics*, not
observations of weather or time of day.

---

## 6. Training the tree-trunk detector

```bash
# from a lightweight pretrained backbone (recommended starting point)
.venv/bin/python scripts/train.py \
    --model yolov8n.pt \
    --data configs/dataset.yaml \
    --epochs 80 --imgsz 640 --batch 16

# resume an interrupted run
.venv/bin/python scripts/train.py \
    --resume runs/train/tree_trunk_exp/weights/last.pt \
    --data configs/dataset.yaml

# install the trained model for the app / scripts
cp runs/train/tree_trunk_exp/weights/best.pt models/tree_trunk.pt
```

- Outputs land in `runs/train/<name>/` — `results.csv` (per-epoch train/val metrics),
  `weights/best.pt`, `weights/last.pt`.
- **Early stopping** via `--patience` (default 30).
- Tune `--device` (`cpu`, `0`) — leave unset for auto.
- The **test split must never be used for training or threshold tuning**: use `val/`
  for model/threshold selection and reserve `test/` for the final report.

---

## 7. Running inference

### Single image / folder / video

```bash
# one image -> annotated copy + detections.json/csv + summary.json
.venv/bin/python scripts/batch_detect.py \
    --source path/to/forest.jpg --output outputs/run1

# a whole folder (recursive)
.venv/bin/python scripts/batch_detect.py \
    --source data/images/test --output outputs/batch --conf 0.4

# a video, limited to 300 frames, with lighting labels attached
.venv/bin/python scripts/batch_detect.py \
    --source forest_drive.mp4 --output outputs/video \
    --categories data/lighting_labels.json --max-frames 300
```

Outputs: `outputs/<run>/annotated/`, `detections.json`, `detections.csv`, `summary.json`
(plus `video_summary.json` for videos). Every detection record carries bbox, confidence,
class, source filename/frame index and inference time.

### Exit codes

`0` success · `1` unusable input (missing/corrupt/empty) · `2` weights/backend problem.

---

## 8. Evaluating lighting robustness

### 8.1 Two complementary lighting methods

**A. Real-world categories** — assign labels (JSON, section 5.4); stored with results.

**B. Image-based statistics** — computed by `src/lighting_analysis.py`:
mean/median gray intensity, contrast (std), `%` dark pixels (`<60`), `%` bright pixels
(`>200`), HSV value mean/std, plus a heuristic category suggestion.

**C. Synthetic transformations** — `src/lighting_augmentation.py` provides
`darker`, `brighter`, `gamma`, `low_contrast`, `shadow`, `noise`, each with a strength
in `[0, 1]`. Originals are never modified; variants are saved separately and tracked as
`synthetic:<name>` so they are never confused with genuine real-world labels. Generate
them in code (`src/lighting_augmentation.py`):

```python
from src.lighting_augmentation import generate_variants, save_variant
variants = generate_variants(image, strength=0.6)   # dict[name -> ndarray]
save_variant(variants["shadow"], "outputs/variants/shadow.jpg")
```

### 8.2 Batch evaluation (CLI)

```bash
# with ground truth -> full metrics per lighting category
.venv/bin/python scripts/evaluate.py \
    --weights models/tree_trunk.pt \
    --images data/images/test \
    --labels data/labels/test \
    --categories data/lighting_labels.json \
    --output reports/test

# without labels -> inference statistics only, accuracy reported as UNAVAILABLE
.venv/bin/python scripts/evaluate.py \
    --weights models/tree_trunk.pt \
    --images data/images/test \
    --output reports/test_nolabels
```

Reported files: `metrics.json`, `metrics.csv` (per lighting category),
`per_image.csv`, `detections.json`, `lighting_stats.json`, and `charts/`:

- `precision_recall_by_lighting.png`
- `map_by_lighting.png`
- `count_error_by_lighting.png`
- `inference_time_by_lighting.png`
- `confidence_distribution.png`
- `confidence_by_lighting.png`

(The first three are only generated when ground truth exists.)

---

## 9. Interpreting the metrics

| Metric | Meaning |
|---|---|
| **Precision** | Of the boxes predicted, how many matched a ground-truth box (IoU ≥ 0.5). Low ⇒ many false positives (e.g. shadow edges mistaken for trunks). |
| **Recall** | Of the ground-truth boxes, how many were detected. Low ⇒ missed trunks (e.g. in deep shadow). |
| **F1** | Harmonic mean of precision and recall. |
| **AP@0.5 / mAP@0.5** | Average precision at IoU 0.5, averaged over classes (all-point interpolation). |
| **mAP@0.5:0.95** | Mean over IoU 0.50…0.95 — penalizes sloppy localization. |
| **FP / FN** | Detections without a matching label / labels without a matching detection. |
| **Count MAE** | Mean absolute difference between predicted and ground-truth tree counts per image — the metric that matters for *counting* applications. |
| **Mean inference ms / FPS** | Timing per image/frame on the active device. |

**Precision, recall, mAP and count MAE require labeled ground-truth boxes.** Without them
these fields are written as `null`/shown as *unavailable* — never estimated. Confidence
distributions, lighting statistics and timing are still reported, because they do not
need labels.

### Comparing robustness

Run `scripts/evaluate.py` once on the original test set, then compare against synthetic
conditions (or evaluate separately-grouped real categories). Typical questions the charts
answer:

- Where does recall collapse (deep shadow / low light)?
- Where do false positives rise (high-contrast direct sunlight)?
- Does inference time change with lighting? (It shouldn't — same model, same size.)
- Do confidence scores shift darker vs. brighter?

**Do not assume enhancement helps.** Gamma/brightness correction can raise recall in dark
frames but can also wash out bark texture and *increase* false positives. Measure both
directions before adopting any preprocessing.

---

## 10. Project structure

```
tree-detection/
├── app.py                        # Streamlit GUI: image in -> trunks + path out
├── requirements.txt
├── pytest.ini
├── README.md
├── configs/
│   ├── dataset.yaml              # YOLO trunk dataset (1 class: tree_trunk)
│   └── path_dataset.yaml         # YOLO-seg path dataset (1 class: path)
├── src/
│   ├── infer.py                  # single-image trunk detection + path isolation (used by app.py)
│   ├── detector.py               # YOLO loading, capability checks, NMS, caching
│   ├── preprocessing.py          # input validation, loaders, folder/video helpers
│   ├── lighting_analysis.py      # statistics + category labels
│   ├── lighting_augmentation.py  # synthetic transformations
│   ├── video_processor.py        # frame-by-frame video/webcam processing
│   ├── evaluation.py             # precision/recall/AP/mAP/count-error + exports
│   ├── visualization.py          # drawing + charts
│   └── synthetic_fixtures.py     # clearly-marked synthetic demo images
├── scripts/
│   ├── import_fortrunkdet.py     # ForTrunkDet (Pascal VOC) -> YOLO, contiguous splits
│   ├── import_finnwoodlands.py   # FinnWoodlands boxes -> YOLO
│   ├── import_finn_paths.py      # FinnWoodlands "Track" masks -> YOLO-seg polygons
│   ├── train.py                  # training + dataset validation
│   ├── evaluate.py               # batch evaluation with/without labels
│   ├── batch_detect.py           # images/folder/video batch inference
│   ├── detect_path.py            # YOLO-seg path isolation + steering offset
│   └── rugd_segment.py           # pretrained RUGD-Segformer path/tree segmentation
├── pc_bundle/
│   ├── train_on_pc.py            # one-command GPU training (Windows or Linux)
│   ├── requirements.txt          # pins the same Ultralytics as the Jetson
│   └── README.txt                # step-by-step PC instructions
├── tests/
│   ├── conftest.py               # synthetic fixtures (fixture data only)
│   ├── test_detector.py
│   ├── test_infer.py
│   ├── test_lighting_analysis.py
│   └── test_evaluation.py
├── data/, data_path/             # datasets (not committed)
├── models/                       # trained weights / RUGD-Segformer (not committed)
├── outputs/, reports/, runs/     # generated outputs (not committed)
```

---
---

## 11. Testing

```bash
.venv/bin/python -m pytest           # 68 tests
.venv/bin/python -m pytest -v        # verbose
```

Coverage: input validation (missing/corrupt/empty inputs), confidence filtering, NMS,
class-capability checks, model caching, drawing on empty detections, lighting statistics
(exact values on controlled images), category heuristics, all synthetic transforms
(identity at strength 0, input never mutated, seeded determinism), precision/recall/AP/
mAP/FP/FN/count-MAE calculations, YOLO label parsing, report export — plus a synthetic
video round-trip (skipped automatically if video codecs are unavailable).

The tests use **synthetic fixture images** (computer-generated rectangles). They validate
pipeline mechanics only and are **not** forestry evaluation data.

---

## 12. Known limitations & troubleshooting

**Limitations**

- Trained weights and datasets are **not committed** (git-ignored); see Architecture and
  Section 5 for how to recreate them. Detection accuracy is moderate (see Results) and weak in bright daylight.
- COCO-pretrained models have no tree class; they run strictly as a labelled baseline.
- Synthetic lighting variants approximate illumination changes; they do not reproduce
  real sensor response, fog, or directional shadows perfectly.
- On CPU, expect ~0.2–1.0 s/frame for `yolov8n` at 640 px depending on the machine.

**Troubleshooting**

| Symptom | Fix |
|---|---|
| `No model weights specified` / `Model weights not found` | Put a checkpoint in `models/` or pass `--weights path/to/best.pt` (`--weights` flag). |
| `The 'ultralytics' package is not installed` | `pip install -r requirements.txt` inside the venv. |
| GPU listed but inference uses CPU | The app/function probes CUDA with a real kernel test. Generic PyPI torch builds often lack kernels for the local GPU (e.g. Jetson compute capability 8.7) → CPU fallback is intentional. For GPU: install the vendor-matched torch build (e.g. L4T/JetPack wheels), then `python -c "from src.detector import cuda_functional; print(cuda_functional())"` should print `True`. |
| `Could not open a video writer` | Install ffmpeg or use `--max-frames`; codecs tried: `mp4v, avc1, XVID, MJPG`. |
| Video opens but 0 frames | Corrupted file or unsupported codec — try re-encoding (`ffmpeg -i in.mp4 -c:v mjpeg out.avi`). |
| `No supported images found` | Check extensions (jpg/jpeg/png/bmp/webp/tif/tiff) and that the folder isn't empty. |
| `Failed to decode image` | File is corrupted or not really an image. |
| Training aborts with dataset problems | Run `python scripts/train.py --validate-only` and fix each reported issue. |
| Metrics show *unavailable* | Ground-truth labels missing/mismatched — provide `labels/` with one `.txt` per test image. |
| Streamlit port busy | `streamlit run app.py --server.port 8502`. |
