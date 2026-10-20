# Robust Tree Trunk Detection and Forest-Path Isolation

A computer-vision pipeline for **detecting tree trunks and isolating the drivable path in forest
imagery** under varying lighting (bright sunlight, partial shade, deep shadow, overcast, low
light), with an emphasis on *measuring* how lighting affects the models.

The project provides:

- a **YOLOv8 tree-trunk detector** trained on public forest datasets (bounding boxes, confidence, tree counts)
- a **local navigation pipeline**: traversable-ground segmentation (pretrained
  [RUGD-Segformer](https://huggingface.co/JasonTStanley/RUGD-Segformer), a classical baseline, or a hybrid of both),
  obstacle fusion with the trunk detector, a clearance-aware **route planner**, a heading estimate
  (LEFT / STRAIGHT / RIGHT / STOP) and a documented **path-coverage** metric
- a **GUI** (Streamlit): upload an image -> original | tree & obstacle detections | navigation overlay (route, corridor, margins, arrow)
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
 PUBLIC DATA                    WINDOWS PC (RTX GPU)                 JETSON (inference)
 ForTrunkDet, FinnWoodlands ──▶ pc_bundle/train_on_pc.py ──.pt──▶ models/tree_trunk*.pt, forest_path_seg.pt
   (scripts/import_*.py)         YOLOv8s trunks, YOLOv8s-seg path     models/rugd_segformer/ (pretrained)

 image ─▶ src/infer.detect_trunks ───────────────┐ trunk boxes (YOLOv8)
      └─▶ src/traversability.py                   ▼
            normalize (dark only) ─▶ segmenter ─▶ fuse trunk footprints ─▶ threshold ─▶ morphology ─▶
            keep ground connected to start ─▶ classes: TRAVERSABLE / VEGETATION / OBSTACLE / UNKNOWN
      └─▶ src/navigation.py
            safety margin ─▶ clearance map ─▶ Dijkstra cost-map route ─▶ smoothing ─▶ heading label / STOP
      └─▶ src/nav_render.py ─▶ app.py (3 panels)   |   scripts/navigate.py (images, folders, video)
```

| Component | Model / method | Output |
|---|---|---|
| Trunk detector | YOLOv8s, single class `tree_trunk` | boxes, confidence, count |
| Ground segmentation (default) | **Classical baseline**: ground = pixels whose CLAHE-normalised Lab colour is close to the patch in front of the vehicle, low texture, below the horizon prior; no semantics, no downloads, ~0.3 s/frame | class map + P(traversable) |
| Ground segmentation (safer) | **Classical + obstacle veto**: the same, but anything RUGD-Segformer (SegFormer-B5, 25 classes) labels as an obstacle (trees, vehicles, rocks, people ...) is removed from the ground | same |
| Ground segmentation (**safer option**) | **Terrain-aware + DeepScene**: union of the classical estimate and the DeepScene FCN-ResNet18 trail/grass classes (NVIDIA jetson-inference, 43 MB, ~0.2 s/frame, runs through OpenCV), minus its obstacle/vegetation classes; grass passable at lower confidence. Also drives the plants/terrain panel (grass, vegetation, obstacle shares) | class map + P(traversable) + terrain labels |
| Other methods (optional, weights not shipped) | DeepScene alone; ADE20K SegFormer (alone / terrain-aware); RUGD-Segformer (alone / hybrid / veto / terrain-aware). Methods appear in the GUI only if their weights are present. | same |
| Route planner | Dijkstra on a 96-column grid; cost = 1 + k/(1+clearance) + confidence + lateral terms | route points, corridor, confidence, clearance, or an explicit reason for no route |
| Direction | angle from lower-centre start to the route point at 60% of the look-ahead row | TURN LEFT / SLIGHT LEFT / STRAIGHT / SLIGHT RIGHT / TURN RIGHT / STOP |

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

### Navigation results (preliminary - read the caveats)

**Why the old app said "Path coverage 0% / no path found":** the path method was the custom YOLOv8-seg model, which
learned only the narrow groomed snow "Track" class from 175 images of one forest. It returned an empty mask on most
scenes (a path was found in 20/30 frames of its own forest and 4/30 elsewhere), the coverage was just that mask's share
of the *whole* image, and there was no planner - an empty mask meant "no path". The navigation pipeline replaces this.

Check against the FinnWoodlands ground-truth "Track" masks (`scripts/eval_navigation.py`, 68 frames, never used to
train the RUGD or classical methods; the GT is only the narrow groomed strip, so wide drivable ground outside it counts
against the route):

| Ground segmentation | Route found | Heading on correct side (±7° dead zone) | Mean heading error | Route points near GT track | GT track covered by traversable mask |
|---|---|---|---|---|---|
| RUGD-Segformer | 57/68 | 58% | 19.3° | 62% | 78% |
| RUGD + classical hybrid | **62/68** | 63% | 17.2° | 65% | **87%** |
| **Classical baseline (default)** | 56/68 | 77% | 12.0° | **80%** | 85% |
| Classical + obstacle veto | 49/68 | **80%** | **11.6°** | 78% | 83% |

The classical baseline is the most accurate on trails (heading agreement and error above), so it is the default. This set
is also ideal for it (a pale strip right in front of the camera on snow). The veto variant is about as accurate where it
finds a route but stops more often. RUGD alone has no snow class, which is why it does worst here.

**Plants / terrain.** With a semantic method (DeepScene, or the optional RUGD/ADE20K ones) the detections panel tints grass, bushes, trees, logs, rocks and water, lists their share of the image and boxes bush patches; with the classical baseline only an approximate "vegetation" share (colour-based) is shown. The DeepScene terrain-aware method is scored above; it is not the default because of its false STOPs on litter-covered summer floors.

**The classical baseline has no safety veto.** In the scenarios below it reports routes where none exists (dense
vegetation, dense forest, a narrow gap between trunks) and paints a parked machine as ground. Use *Classical + obstacle
veto* when a false GO matters more than speed.

Real-image scenarios (`scripts/nav_scenarios.py --method ...`, frames picked by eye, **visual checks only**):

| Scenario | Classical (default) | Classical + obstacle veto | RUGD + classical hybrid |
|---|---|---|---|
| 1 visible trail into forest | route | route (partial) | route |
| 2 trail partly blocked by trunks | route | route (partial) | route (partial) |
| 3 route bends left | TURN LEFT | TURN LEFT | TURN LEFT |
| 4 route bends right | TURN RIGHT | TURN RIGHT | TURN RIGHT |
| 5 dense vegetation, no corridor | **route (false GO)** | **route (false GO)** | STOP (correct) |
| 6a heavy natural shade | route | route | route |
| 6b synthetic darkening (0.8) | route over a garbled image (unreliable) | STOP | STOP |
| 7a no valid path: dense forest | **route (false GO)** | STOP (correct) | STOP (correct) |
| 7b no valid path: narrow gap | **route (false GO)** | STOP (correct) | STOP (correct) |
| 8 narrow snow track into forest | false STOP | false STOP | false STOP |
| 9 machine blocks the centre | STOP (machine painted as ground) | STOP | route around the machine |

Takeaway: the default is accurate on trails but not safe in clutter; the veto variant removes the false GOs on trees, trunks and
machines but not on dense bushes (only the RUGD-vegetation veto in the hybrid catches those, at the cost of missing routes on grass).

### Accuracy tuning: what helped and what did not

Everything below was tuned on training/validation data and scored on held-out data (`scripts/tune_trunk.py`, `scripts/tune_nav.py`).

| Change | Result | Adopted |
|---|---|---|
| **Soft veto (0.7)** for the DeepScene terrain-aware method (block only where P(obstacle or dense vegetation) >= 0.7, instead of wherever those classes win the argmax) | GT-track frames: route found 58 -> 63 of 68, heading error 13.2 -> 12.0 deg; every scenario's STOP/GO outcome unchanged | **yes** (terrain-aware methods) |
| Trunk confidence threshold chosen on validation (0.30 vs 0.40) | Test F1 0.720 vs 0.718, recall 0.75 vs 0.71 but tree-count error worse (1.43 vs 1.29) | no - kept 0.40 |
| Trunk inference size 800 / 960 px (model trained at 640) | Validation best F1 0.785 / 0.761 vs 0.793 at 640 | no |
| Planner parameters (lateral preference, goal penalty, clearance weight, smoothing, start shift) with look-ahead rows and all safety margins frozen | Classical: +1.8 pts held-out score (k_lat 2 -> 6); DeepScene: +1.1 pts with five changes that disagree with the classical run | no - within noise for a groomed-track proxy |
| Soft veto on **leaf-litter forest floors** | Does not help: DeepScene is >0.85 confident those floors are obstacle/vegetation, so the false STOPs remain at every threshold | n/a - needs labelled data |

Takeaway: the trunk model is limited by its data and size (more ForTrunkDet-style data or a larger model trained on the PC is the next step, e.g. `--trunk-model yolov8m.pt --imgsz 640 --epochs 150`), and the remaining false STOPs need labelled litter-floor images, not parameter tuning.

### Leaf-litter false stops: diagnosis, options and what is still unsolved

Full write-ups: `docs/leaf_litter/` (baseline, diagnosis, evaluation, metrics). **All numbers below are exploratory**: the pilot labels are single-annotator and were corrected after results were seen (a visual audit found wrong hazard boxes), and there are 0 held-out STOP frames and only 7 hazard frames.

- **Cause (measured):** DeepScene has no leaf-litter class and labels litter "vegetation" (mean P(veg) 0.26-0.76 on litter floors vs 0.53 on the one dense-bush frame - no threshold separates them); the veto also turned vetoed pixels into hard obstacles, and the grass caution removed ground from eligibility. RUGD-Segformer, which has `mulch`/`dirt` and `bush`, does separate them.
- **Code (defaults unchanged):** confidence-aware fusion, **multi-source fusion** (new opt-in method: ground = classical + RUGD + DeepScene evidence, hard veto from RUGD obstacles/bushes and DeepScene obstacles), structured STOP reason codes (`PathPlan.reason_code`), opt-in COCO person/vehicle footprints, diagnosis / evaluation / comparison tools.
- **Held-out comparison (final test: 69 GO frames = 55 snow-track + 14 litter/trail; 7 frames with hazard boxes):**

| Setting | false-stop (all / litter+trail 14 / snow 55) | hazard frames routed through (of 7) | snow heading error | CPU time / frame |
|---|---|---|---|---|
| classical (default) | 1% / 1 / 0% | 2 | 5.1 deg | 0.43 s |
| terrain-aware + soft veto 0.7 (shipped option) | 35% / 14 / 18% | 0 | 7.5 deg | 1.4 s |
| confidence-aware w=1.0 | 25% / 14 / 5% | 1 | 8.5 deg | 1.4 s |
| confidence-aware w=0 + COCO | 6% / 4 / 0% | 1 | 9.4 deg | 1.7 s |
| RUGD + classical hybrid | 1% / 0 / 2% | 1 | 19.0 deg | ~5 s |
| **multi-source (opt-in)** | **6% / 0 / 7%** | **0** | 16.7 deg | **5.1 s**, ~2 GB RAM |

  Multi-source is the only method here with no litter false stops *and* no routes through a labelled hazard, but it is slow and its snow-track heading is worse. The pre-registered selection rule (dev only) picked `conf-aware w=0`, which routed through 2/7 held-out hazards, so no default was changed. In the GUI pick *Multi-source* (needs `models/rugd_segformer` and `models/deepscene`) or use *Advanced* -> *Confidence-aware fusion* / *Detect people / vehicles*.
- **To validate:** label a fresh set from your own camera (open litter floors, dense bushes, people/machines, true dead-ends), two reviewers, frozen protocol, score once. Format: `data/leaf_litter_pilot/annotations.json` (v2, with changelog).
- **Reproduce:** `python scripts/diagnose_nav.py <frames>`; `python scripts/eval_leaf_litter.py --select --with-rugd`; `python scripts/compare_nav_variants.py ft_img2374 ft_img0707 ft_img0691 00166 00254`.

### Models evaluated and rejected

Same 68 FinnWoodlands frames and the same planner (`scripts/eval_navigation.py`; groomed-track proxy, see caveats above):

| Ground segmentation | Route found | Heading on correct side | Mean heading error | Notes |
|---|---|---|---|---|
| Classical baseline (default) | 56/68 | 77% | 12.0° | no downloads; false routes in clutter |
| **Terrain-aware, classical + DeepScene** | **58/68** | **83%** | 13.2° | stops correctly in dense vegetation/forest and at a machine; **false STOPs on 3 of 11 leaf-litter forest-floor scenarios** |
| DeepScene alone | 55/68 | 76% | 14.9° | |
| Terrain-aware, classical + RUGD | 56/68 | 59% | 20.9° | worst on snow (no snow class); 339 MB, ~5 s/frame. **Weights were deleted, then re-added** after the leaf-litter evaluation showed RUGD is the only primary that separates litter from bushes |
| ADE20K SegFormer-B2 alone | 32/68 | 81% | 13.3° | misses snow; NVIDIA non-commercial license -> **deleted** |
| Terrain-aware, classical + ADE20K | 45/68 | 78% | 15.9° | **deleted** |

Trunk detection, 341 test images at confidence 0.05: **our YOLOv8s mAP@0.5 0.746 (recall 0.87)** vs. YOLOE-11s zero-shot with the prompt "tree trunk" **0.032 (recall 0.04)** -> YOLOE **rejected and deleted**. No pretrained, openly licensed under-canopy trunk detector with released weights was found.
Also deleted as bad/superseded: the first CPU-trained trunk model (mAP 0.36) and the YOLOv8-seg path model (found a path in 4/30 frames outside its own forest). All of these can be re-created; see the model-setup commands.

---

## Remaining work

- [ ] **Representative path data**: no ground-truth *routes* exist for these forests; the check above uses only the groomed-track mask. Label drivable corridors on images from the target robot's camera and measure route/heading accuracy properly.
- [ ] **Fine-tune DeepScene (or another small segmenter) on litter-covered forest floors and dense bushes** to remove its false STOPs without losing the vegetation veto; the leaf-litter evaluation shows no threshold can (see above). Review/extend the pilot labels first.
- [ ] **Fine-tune a forest segmentation model** (traversable / vegetation / obstacle) on that data; RUGD has no snow class and was not validated on your camera. The hybrid is a stop-gap.
- [ ] **Depth and calibration** for real navigation: a single 2D image cannot give metric clearance or guarantee collision-free motion. Add stereo/depth (FinnWoodlands has depth), camera intrinsics and the vehicle footprint, and convert the image-plane corridor to a ground-plane cost map.
- [ ] **Closed-loop validation**: the direction labels are steering *suggestions*. Test on the vehicle in a safe area with an independent safety stop before using them as commands.
- [ ] **Real-time on Jetson**: install a JetPack-matched PyTorch (or export to TensorRT). Today a frame takes ~1 s for trunks plus ~0.2 s for DeepScene (or ~0.3 s classical) at 480 px on CPU.
- [ ] **Low light**: heavily darkened images produce noisy segmentation and STOP; evaluate on real night/dusk data.
- [ ] **Higher-accuracy trunk run** on the PC (planned, not run). Inference-time tuning gave no reliable gain (see above); try `python train_on_pc.py --only trunk --trunk-model yolov8m.pt --imgsz 640 --batch 8 --epochs 150 --patience 50 --tag _m640` (960 px training is unproven: 960 px inference was worse for the 640-trained model).
- [ ] **Bright-daylight / overcast data**: trunk recall in bright sun is only 0.25 and the datasets do not cover all five lighting conditions; add real lighting labels (`data/lighting_labels.json`).
- [ ] **Tune the planner thresholds** (look-ahead, minimum reach, margins) on labelled routes; the narrow-snow-track false stop is an example.
- [ ] **Model weights hosting** (`*.pt` is git-ignored) and **license checks** (FinnWoodlands is research-use; RUGD-Segformer lists none).

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

Upload a **`.jpg`/`.png` image or a video** (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`). For a video, choose every-Nth-frame and a frame limit, press *Process video*, and you get a progress bar, an annotated clip that plays in the page (VP8 WebM, downloadable), a frame-by-frame viewer with all three panels, heading and coverage charts, a per-frame table and a CSV download; the route is smoothed across frames, the heading is damped, and a lost route stops immediately. For an image the page shows three panels - **Original**, **Tree & obstacle detections**, **Navigation** (traversable
ground in green, obstacles red, safety margin/uncertain yellow, route cyan with start/target markers, corridor outline,
heading arrow and legend) - plus metrics: trees detected, path coverage, traversable area, route confidence, whether a
continuous route exists, clearance and the direction.

Settings (sidebar): tree detection model, path segmentation method (classical baseline [default] / classical + obstacle veto /
RUGD hybrid / RUGD / YOLO path model / Off), tree confidence, inference resolution (default 480), obstacle safety margin, look-ahead distance, path smoothing,
minimum path confidence, device (auto/cpu/cuda; GPU is used only if CUDA actually works), plus advanced ground-confidence,
minimum corridor width and dark-image normalisation. Missing weights or inference failures are shown as errors with a
suggested fix; with the path method Off the page says navigation is disabled.

Command line (images, folders, **video** with temporal smoothing and measured latency/FPS):

```bash
python scripts/navigate.py --source forest1.jpg                       # one image
python scripts/navigate.py --source clip.mp4 --stride 2               # video -> navigation.mp4 + navigation.csv
python scripts/nav_scenarios.py                                       # the seven scenario checks
python scripts/eval_navigation.py --stride 3                          # GT-track comparison (FinnWoodlands)
python scripts/batch_detect.py --source data/images/test --output outputs/test --weights models/tree_trunk.pt --conf 0.4
```

---

## 3b. Navigation pipeline: definitions, algorithms and limits

**Terrain classes** - `TRAVERSABLE` (dirt, sand, gravel, mulch, asphalt, concrete, and ground the classical estimate adds),
`VEGETATION` (grass, bushes: *never* treated as safe), `OBSTACLE` (trees, logs, rocks, poles, water, vehicles, people ... plus
the ground footprint - bottom 12% - of every detected trunk box), `UNKNOWN` (low confidence, sky, ground patches that
do not connect to the start region).

**`detect_traversable_region(image, obstacle_boxes, segmenter)`** (`src/traversability.py`): dark-image CLAHE (mean L < 70 only)
-> segmenter -> fuse trunk footprints -> keep pixels with P(traversable) >= 0.5 -> morphological open/close -> keep only
ground connected to the start band (bottom centre) -> reject isolated patches (shown as UNKNOWN). Uncertain pixels are never
promoted to traversable.

**`plan_local_path(traversable, obstacles, config)`** (`src/navigation.py`): obstacles are inflated by the safety margin;
cells must be traversable, unblocked and at least `min_clearance` from blocked space (narrow gaps rejected). Dijkstra over a
down-sampled grid with cost = 1 + 8/(1+clearance) + 3(1 - P(traversable)) + 2*lateral offset; backwards and sideways steps are
penalised. Every reachable endpoint is scored (cost + 6 per grid row of look-ahead left unreached) so a slightly shorter route
near the vehicle's heading beats a full-length one far to the side; a route must reach at least 70% of the look-ahead distance
(otherwise "No continuous safe corridor reaches the look-ahead point"). The route may start up to 35% of the image width off-centre
at a cost (it is **not** forced to the middle when the centre is blocked). The route is smoothed, with automatic back-off if smoothing
would cut into blocked space. Confidence = mean P(traversable) along the route x (0.5 + 0.5 x clearance factor), reduced for partial
routes; routes below the minimum confidence are rejected. Returns points, corridor, blocked mask, confidence, clearance, `partial`,
`reached_fraction` and an explicit failure `reason`.

**`estimate_navigation_direction(points, W, H)`**: angle (image plane, + = right) from the lower-centre start to the route
point at 60% of the look-ahead row; |angle| <= 7 deg STRAIGHT, <= 20 deg SLIGHT, otherwise TURN; no route -> `STOP / NO SAFE PATH`.
`DirectionSmoother` (video) smooths the angle (EMA), stops *immediately* when the route disappears and needs two valid frames to resume.

**Path coverage** = `traversable_pixels / valid_roi_pixels x 100`, where the ROI is the lower 60% of the image (rows >= 40% of the height)
and *valid* pixels exclude sky. Unknown, vegetation and obstacle pixels stay in the denominator (conservative). *Traversable area* is the
same count over the whole valid image. A route is reported only if it satisfies continuity, clearance and confidence - ground pixels alone
never imply a valid route.

**Model setup.** Required: `models/tree_trunk.pt` (trained on the PC, see `pc_bundle/README.txt`). For the DeepScene-based terrain-aware method:

```bash
mkdir -p models/deepscene && cd models/deepscene
curl -L -o ds.tar.gz https://github.com/dusty-nv/jetson-inference/releases/download/model-mirror-190618/FCN-ResNet18-DeepScene-576x320.tar.gz
tar -xzf ds.tar.gz --strip-components=1 && rm ds.tar.gz        # leaves fcn_resnet18.onnx, classes.txt
```

(Check the DeepScene dataset's license before commercial use; I did not verify it.) Optional, only if you want those methods back:

```bash
pip install transformers safetensors huggingface_hub
python -c "from huggingface_hub import snapshot_download as s; s('JasonTStanley/RUGD-Segformer', local_dir='models/rugd_segformer')"
python -c "from huggingface_hub import snapshot_download as s; s('nvidia/segformer-b2-finetuned-ade-512-512', local_dir='models/ade_segformer_b2', allow_patterns=['*.json','pytorch_model.bin'])"
```

The default *Classical baseline* needs no downloads; the other methods need their weights above and appear in the GUI only when present.

**Limits - please read.** This is a *visual* local route estimate from one 2D image. Without depth, camera calibration and vehicle geometry
it cannot guarantee a collision-free path, and the labels are steering suggestions, not validated control commands. RUGD was not trained on
snow or on your camera; the classical baseline fails when the patch in front of the vehicle is on a bush, trunk or shadow edge; very dark images
degrade segmentation; reliable operation in all lighting conditions has **not** been shown. Quantitative checks exist only for the narrow
groomed track in FinnWoodlands.

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
│   ├── traversability.py         # ground / vegetation / obstacle / unknown segmentation + fusion (RUGD, classical, hybrid)
│   ├── navigation.py             # route planner, heading estimate, path coverage, temporal smoothing
│   ├── nav_render.py             # navigation + detection panel drawing
│   ├── pipeline.py               # run_navigation(): trunks -> traversability -> plan -> direction; model loading/errors
│   ├── infer.py                  # trunk detection helpers, model loaders
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
│   ├── navigate.py               # navigation on images / folders / video (smoothing, latency, FPS)
│   ├── nav_scenarios.py          # seven real-image scenario checks -> reports/navigation_scenarios/
│   ├── eval_navigation.py        # heading/route check against FinnWoodlands GT track (cached model outputs)
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
│   ├── test_navigation.py        # planner continuity, direction, coverage, smoothing, no-path cases
│   ├── test_traversability.py    # mask processing, isolated-patch rejection, obstacle fusion, validation
│   ├── test_lighting_analysis.py
│   └── test_evaluation.py
├── data/, data_path/             # datasets (not committed)
├── models/                       # trained weights / RUGD-Segformer (not committed)
├── outputs/, reports/, runs/     # generated outputs (not committed)
```

---

## 11. Testing

```bash
.venv/bin/python -m pytest           # 123 tests
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
| "Path coverage 0%" / "no path found" | Use the classical baseline (default) or classical + obstacle veto (the YOLO path model only knows a groomed snow track). If it still stops, the message says why: no ground near the start, ground too narrow, route stops short of the look-ahead, or confidence too low. Lower the look-ahead or safety margin, or check the image really contains a clear corridor. |
| "RUGD-Segformer weights not found" | Run the snapshot_download command in Section 3b, or use the Classical baseline. |
| Navigation is slow | RUGD takes ~4 s/frame at 480 px on the Jetson CPU. The Classical baseline (default) takes ~0.3 s, or get a GPU-enabled PyTorch build. |
| Streamlit port busy | `streamlit run app.py --server.port 8502`. |
