# Forest Navigator

Upload a **forest image or video** and get a GO / STOP decision, a planned route over the picture, the detected obstacles and trees,
and a steering heading - estimated from the camera image alone (no LiDAR or depth yet).

- **Navigation** (default): combines a colour/texture ground estimate with two pretrained segmentation models, vetoes anything a model
  recognises as an obstacle (trees, people, vehicles, bushes ...), and plans a clearance-aware route. About 5 s/frame on the Jetson CPU.
- **Fast preview** (a switch in the sidebar): the colour/texture estimate alone, about 0.4 s/frame. It has **no obstacle checks** and can route
  through people, machines and dense vegetation - use it as a preview only.
- Works on **images and videos** (annotated clip, frame viewer, charts, CSV). Trunk detection is a YOLOv8s model trained on public forest datasets.

> **A visual estimate only.** One 2D image without depth, calibration or vehicle geometry cannot guarantee a collision-free path; the heading is
> a steering *suggestion*, not a validated control command.

## Quick start

```bash
cd tree-detection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/streamlit run app.py --server.port 8517          # then open http://<host>:8517
python scripts/navigate.py --source photo.jpg              # command line: image, folder or video (add --method "Fast preview")
python -m pytest                                           # 123 tests
```

The UI is one page: drop a file, read the status card (GO + heading, or STOP + the reason in plain language), then switch between the
*Navigation*, *Detections* and *Original* tabs. *Advanced* holds tree confidence, image size, safety margin, look-ahead, smoothing, minimum
route confidence, device and grass handling.

## Models

| Model | Used for | Where | Notes |
|---|---|---|---|
| `tree_trunk.pt` (YOLOv8s) | tree trunks (boxes become ground obstacles) | `models/` | trained on a GPU PC on ForTrunkDet + FinnWoodlands; test mAP@0.5 0.64 (see *Training*) |
| RUGD-Segformer (SegFormer-B5) | terrain classes: mulch/dirt/grass/bush/tree/person/vehicle ... | `models/rugd_segformer/` | no license listed; research use; ~340 MB |
| DeepScene FCN-ResNet18 | trail/grass/vegetation/obstacle | `models/deepscene/` | NVIDIA jetson-inference; check the dataset license before commercial use; runs through OpenCV |
| `yolov8n.pt` (COCO) | only the starting weights for training | `models/` | not a tree detector |

```bash
python -c "from huggingface_hub import snapshot_download as s; s('JasonTStanley/RUGD-Segformer', local_dir='models/rugd_segformer')"
mkdir -p models/deepscene && cd models/deepscene
curl -L -o ds.tar.gz https://github.com/dusty-nv/jetson-inference/releases/download/model-mirror-190618/FCN-ResNet18-DeepScene-576x320.tar.gz
tar -xzf ds.tar.gz --strip-components=1 && rm ds.tar.gz              # leaves fcn_resnet18.onnx
```

Without those two downloads the app offers only the Fast preview. Weights are git-ignored; copy `tree_trunk.pt` from your training run into `models/`.
Run `pip install transformers safetensors` for RUGD (already in `requirements.txt`).

## How it works

```
image -> tree detector (YOLOv8s) ----------------------------------------------┐ trunk ground footprints = obstacles
      -> ground evidence: colour/texture (classical) + RUGD + DeepScene          ▼
      -> hard veto: RUGD obstacles / bushes, DeepScene obstacles, trunks -> traversable mask (connected to the start)
      -> planner: safety margin, clearance map, Dijkstra route, smoothing -> route + heading, or STOP with a reason code
```

- **Ground** = union of the three sources' ground evidence. DeepScene's "vegetation" is deliberately *not* a veto (it labels leaf litter that way).
  Grass is passable at a higher route cost; bushes, trees, people, vehicles, logs, rocks and water are blocked.
- **Route** must be continuous, confident, and wide enough, and reach at least 70% of the look-ahead distance; otherwise STOP with one of
  `NO_TRAVERSABLE_REGION`, `OBSTACLE_BLOCKS_CORRIDOR`, `LOW_CONFIDENCE_TERRAIN`, `INSUFFICIENT_CORRIDOR_WIDTH`, `NO_VALID_CONNECTED_CORRIDOR`.
- **Heading**: angle from the lower-centre start to the route point at 60% of the look-ahead (STRAIGHT / SLIGHT / TURN LEFT / RIGHT / STOP).
  For video the heading is smoothed, a lost route stops immediately, and two valid frames are needed to resume.
- **Path coverage** = traversable pixels / valid (non-sky) pixels in the lower 60% of the image.

## How well does it work? (preliminary)

Held-out comparison on a small hand-labelled pilot set (`data/leaf_litter_pilot/`: 69 GO frames = 55 snow-track + 14 leaf-litter/trail, and 7 frames with a
person/robot hazard box). **Exploratory only**: single annotator, labels corrected after results were seen, no held-out STOP frames. Details in `docs/leaf_litter/`.

| | false stops (all / litter+trail) | hazard frames routed through (of 7) | snow-track heading error | time/frame (Jetson CPU) |
|---|---|---|---|---|
| **Navigation** (default) | 6% / 0 of 14 | 0 | 16.7° | ~5 s |
| Fast preview (classical only) | 1% / 1 of 14 | 2 | 5.1° | ~0.4 s |

- Navigation fixes leaf-litter false stops (an earlier DeepScene-only design stopped on every litter frame) and never routed through a labelled
  person/robot, but its heading on snow tracks is worse than the fast preview's, and it is slow.
- The trunk detector: test mAP@0.5 0.64, precision 0.73, recall 0.71 (341 images); weak in bright daylight (recall 0.25 on 28 images).
- Not validated: real-vehicle behaviour, other forests/cameras, night, dense undergrowth. Treat every number as preliminary.

## Training the trunk detector (on a GPU PC, not the Jetson)

```bash
# 1. build the dataset (needs the downloaded raw datasets; see the import scripts)
python scripts/import_fortrunkdet.py data/raw/fortrunk/x data
python scripts/import_finnwoodlands.py data/raw/finnwoodlands data
python scripts/train.py --validate-only --data configs/dataset.yaml
# 2. train on the PC: copy pc_bundle/ + the dataset zip, then follow pc_bundle/README.txt
python train_on_pc.py            # writes results/tree_trunk.pt -> scp it to models/ on the Jetson
```

Datasets: [ForTrunkDet](https://zenodo.org/records/5213825) (CC BY 4.0) and [FinnWoodlands](https://huggingface.co/datasets/Voxel51/FinnWoodlands)
(Springer licence, research use). Splits are contiguous blocks per camera/site because the images are sequential frames.

## Tools

| Script | Purpose |
|---|---|
| `scripts/navigate.py` | navigation on an image, folder or video; writes overlays/CSV, reports measured latency |
| `scripts/evaluate.py`, `scripts/batch_detect.py` | trunk-detector evaluation (per lighting category) and batch inference |
| `scripts/eval_leaf_litter.py` | compares fusion variants on the pilot set + FinnWoodlands ground-truth tracks (cluster-bootstrap CIs) |
| `scripts/diagnose_nav.py`, `scripts/compare_nav_variants.py` | trace one frame through every stage; side-by-side variants |
| `scripts/tune_trunk.py` | tune trunk confidence / input size on validation, report once on test |

## Project structure

```
app.py                      Streamlit UI (image + video)
src/pipeline.py             run_navigation(): the two modes, model loading, fallbacks
src/traversability.py       terrain classes, segmenters (classical / RUGD / DeepScene), fusion (library also holds research variants)
src/navigation.py           planner, STOP reason codes, heading, coverage, temporal smoothing
src/nav_render.py, src/video_nav.py, src/infer.py, src/detector.py, src/evaluation.py, ...
scripts/                    tools above + dataset import + training validation
pc_bundle/                  one-command GPU training for the trunk model
data/leaf_litter_pilot/     hand-labelled pilot frames (v2; v1 kept) with protocol and changelog
docs/leaf_litter/           diagnosis and evaluation write-ups
tests/                      123 tests
```

## Limitations and troubleshooting

- 2D RGB only: no depth, calibration or vehicle geometry. LiDAR/depth is the next step for real navigation.
- The first Navigation request after starting the app takes ~50 s (loading ~400 MB of models); after that ~5 s per image.
- Fast preview routes through obstacles it cannot recognise; Navigation does not recognise every machine (the pilot robot was often missed by RUGD).
- Snow: RUGD has no snow class; the colour/texture estimate carries those scenes, so heading is less accurate there.
- GPU: generic PyPI PyTorch has no kernels for the Orin GPU, so inference runs on CPU. A JetPack-matched PyTorch or a TensorRT export is needed for real time.

| Symptom | Fix |
|---|---|
| Only "Fast preview" is offered | download RUGD-Segformer and DeepScene (see *Models*) |
| "Weights not found: tree_trunk.pt" | copy your trained weights to `models/tree_trunk.pt` |
| A video will not open | re-encode: `ffmpeg -i in.mov -c:v libx264 out.mp4`; supported: mp4, avi, mov, mkv, webm |
| Port busy | `streamlit run app.py --server.port 8518` |
| STOP everywhere | read the reason code on the red card; lower the safety margin or look-ahead in *Advanced* if the corridor is genuinely clear |

## What is still open

Labelled leaf-litter / dense-bush / dead-end images from your own camera (two reviewers) to validate and improve the ground model; a larger trunk
model trained on the PC (e.g. `yolov8m`); bright-daylight data; GPU/TensorRT inference; LiDAR or depth fusion; closed-loop testing with an independent safety stop.
