# Leaf-litter false stops - evaluation

Reproduce: `python scripts/eval_leaf_litter.py --select` (about 8 min on the Jetson CPU; writes `reports/leaf_litter_metrics.json` and `reports/leaf_litter/per_frame.json`).
Visuals: `python scripts/compare_nav_variants.py ft_img2374 ft_img0707 ft_img0691 00166 00254`.

## Data, labels and limits - read first
- **Frames.** Pilot set `data/leaf_litter_pilot/annotations.json` (ForTrunkDet frames + 2 FinnWoodlands dense-forest frames) plus every FinnWoodlands frame with a GT groomed track (labelled GO automatically).
- **Labels are a single-annotator pilot (the assistant, by eye, UNREVIEWED).** Protocol is in the file: GO = open ground at the bottom centre continues unobstructed to ~mid-frame at
  vehicle-like width; STOP = hazard or dense vegetation occupies the area directly ahead; **AMBIGUOUS frames (16) are excluded from every rate**. Hazards are coarse normalised boxes.
  Please review/correct them; `data/leaf_litter_pilot/annotations.json` is the single source of truth.
- **Counts.** dev: 169 labelled (GO 166: 147 Finn snow-track + 19 pilot; **STOP 3**; hazard frames 4). final_test: 70 labelled
  (GO 69: 55 Finn snow-track + 14 pilot litter/trail frames; **STOP 1**; hazard frames 5). Sequence clusters: 16 dev, 11 final_test.
- **Splits.** `dev` = ForTrunkDet val frames, FinnWoodlands train-split GT frames, and every frame used while diagnosing. `final_test` = ForTrunkDet test frames and Finn val/test GT frames not used for diagnosis.
  Parameters were chosen on dev only. CIs are cluster bootstraps over sequences (adjacent frames are correlated; 2,000 resamples) and are wide.
- **Not independent of earlier work.** The soft-veto threshold 0.7 was picked earlier on a 68-frame set that mixed train/val/test Finn frames, so the "current" row is not a fresh held-out choice.
- **Insufficient negatives.** Clear STOP frames are rare in this data (dev 3, final test 1). **False-safe rates have almost no statistical power**; treat them as flags, not estimates. Heading/GT-track metrics exist only for the 147 + 55 snow-track frames.
- **Image-space plausibility only.** "ROUTE" means the planner proposed an image-space route; it is not robot safety (no depth, calibration or vehicle geometry).

## Methods compared (identical frames, identical planner and rules)
classical baseline; shipped terrain-aware + DeepScene soft veto 0.7 ("current"); the old argmax veto; **new confidence-aware fusion** (`fuse_confidence_aware`, hard obstacle only
from P(obstacle) >= 0.5; vegetation lowers ground confidence by weight w; grass caution affects cost only) for w in {1.0 (conservative), 0.6, 0.4, 0.3, 0.2, 0.0 (no vegetation penalty)}; and
variants with COCO person/vehicle/animal footprints (`+coco`).

## Selection rule (fixed before looking at final_test)
Pick, on dev, the confidence-aware variant with the lowest false-stop rate **subject to** dev false-safe (STOP-labelled and hazard) no worse than the current soft veto.
**Result: None - no variant qualified**, so the default behaviour is unchanged. (Every variant either routes through one more of the 3 dev STOP frames or through a dev hazard box.)

## Results - dev (may be tuned on)
| method | route availability | false-stop [95% CI] (k/n) | false-safe: STOP-labelled routed | false-safe: route through hazard box | heading side OK | mean heading err | GT-track recall |
|---|---|---|---|---|---|---|---|
| classical | 79% | 21% [4-45] (35/166) | 3/3 | 2/4 | 78% | 14.9° | 82% |
| classical +coco | 79% | 22% [5-46] (36/166) | 3/3 | 2/4 | 77% | 14.9° | 82% |
| argmax_veto (v1) | 71% | 28% [16-48] (47/166) | 1/3 | 0/4 | 82% | 13.7° | 76% |
| current: soft veto 0.7 | 81% | 18% [7-37] (30/166) | 1/3 | 0/4 | 82% | 13.4° | 77% |
| conf-aware w=1.0 | 85% | 14% [4-33] (24/166) | 1/3 | 1/4 | 85% | 12.0° | 88% |
| conf-aware w=1.0 +coco | 85% | 14% [4-33] (24/166) | 1/3 | 1/4 | 85% | 12.1° | 88% |
| conf-aware w=0.4 | 86% | 13% [4-29] (21/166) | 1/3 | 1/4 | 86% | 12.1° | 92% |
| conf-aware w=0.2 | 90% | 10% [3-22] (16/166) | 2/3 | 1/4 | 83% | 11.7° | 93% |
| conf-aware w=0.0 | 95% | 5% [0-14] (8/166) | 2/3 | 0/4 | 86% | 11.3° | 94% |
| conf-aware w=0.0 +coco | 95% | 5% [0-14] (8/166) | 2/3 | 0/4 | 86% | 11.4° | 94% |

## Results - final test (scored once, never tuned on)
| method | route availability | false-stop [95% CI] (k/n) | false-safe: STOP-labelled routed | false-safe: route through hazard box | heading side OK | mean heading err | GT-track recall |
|---|---|---|---|---|---|---|---|
| classical | 99% | 1% [0-8] (1/69) | 1/1 | 1/5 | 89% | 5.1° | 95% |
| classical +coco | 97% | 1% [0-8] (1/69) | 0/1 | 1/5 | 89% | 5.3° | 95% |
| argmax_veto (v1) | 53% | 46% [27-100] (32/69) | 0/1 | 0/5 | 84% | 8.7° | 78% |
| current: soft veto 0.7 | 64% | 35% [19-100] (24/69) | 0/1 | 0/5 | 84% | 7.5° | 82% |
| conf-aware w=1.0 | 74% | 25% [11-100] (17/69) | 0/1 | 0/5 | 85% | 8.5° | 89% |
| conf-aware w=1.0 +coco | 74% | 25% [11-100] (17/69) | 0/1 | 0/5 | 85% | 8.5° | 88% |
| conf-aware w=0.4 | 77% | 22% [9-100] (15/69) | 0/1 | 0/5 | 83% | 8.9° | 93% |
| conf-aware w=0.2 | 84% | 16% [5-70] (11/69) | 1/1 | 1/5 | 84% | 8.6° | 94% |
| conf-aware w=0.0 | 94% | 6% [0-22] (4/69) | 1/1 | 1/5 | 84% | 9.4° | 95% |
| conf-aware w=0.0 +coco | 93% | 6% [0-22] (4/69) | 0/1 | 1/5 | 84% | 9.4° | 95% |

False-stop rate on held-out **GO** frames by terrain (final test):

| method | dry_litter (n=6) | grass_litter (n=2) | leaf_litter (n=2) | snow_track (n=55) | trail (n=4) |
|---|---|---|---|---|---|
| classical | 17% | 0% | 0% | 0% | 0% |
| current: soft veto 0.7 | 100% | 100% | 100% | 18% | 100% |
| conf-aware w=1.0 | 100% | 100% | 100% | 5% | 100% |
| conf-aware w=0.4 | 100% | 100% | 100% | 2% | 100% |
| conf-aware w=0.0 | 17% | 100% | 0% | 0% | 25% |
| conf-aware w=0.0 +coco | 17% | 100% | 0% | 0% | 25% |

## Trade-off analysis
- **Snow tracks (55 held-out frames):** current soft veto false-stop 18% (CI 0-35); confidence-aware w=1.0 5.5% (0-15); w<=0.4 <=2%; classical 0%. The mechanism fixes (no hard-obstacle inflation, cost-only grass caution) help here, at equal or lower hazard false-safe (0/5 vs 0/5).
- **Portuguese litter/trails (14 held-out GO frames):** the soft veto and **every** vegetation penalty w>=0.4 stop on **14/14**; w=0.0 stops on 4/14; classical 1/14. Confidence weighting alone does **not** fix leaf litter.
- **What w=0 (+COCO hazards) costs** (the only setting that fixes litter): dev STOP frames routed 2/3 (00166 dense forest, 00178 narrow gap; current routes 1/3 - 00178) and one held-out hazard frame routed (ft_img0643, robot far ahead; classical does too).
  What it buys: held-out false-stop 35% -> 6%, snow 0%, the person-ahead frame (ft_img0691) stopped when COCO footprints are on (classical routes straight through the person - n=1, suggestive only).
- **Route availability alone is not the criterion**: the table reports the safety columns beside it. No setting dominates; choosing w is a risk decision, so it is exposed as an opt-in control (GUI Advanced: *Confidence-aware fusion*, *Vegetation penalty weight*, *Detect people/vehicles*) instead of changing the default.
- Dense bushes + person (ft_img0707, label STOP) are stopped by w=0 +coco (LOW_CONFIDENCE_TERRAIN) but routed by classical.

## Latency / memory (Jetson Orin Nano Super, CPU)
| configuration (CPU, 8 frames, after warm-up) | total / frame | notes |
|---|---|---|
| classical (default) | 0.43 s | trunk 0.32 s, segment+fuse 0.07 s, plan 0.03 s; peak RSS ~1.06 GB |
| terrain-aware DeepScene, soft veto 0.7 | 1.44 s | DeepScene + fusion 1.12 s; peak RSS ~1.21 GB |
| terrain-aware DeepScene, confidence-aware w=0 | 1.45 s | same cost as the veto |
| confidence-aware w=0 + COCO hazards | 1.70 s | COCO yolov8n adds ~0.25 s |

## Conclusion
- **Software changes (done, tested):** structured STOP reason codes; hard-obstacle vs vegetation separation; cost-only grass caution; confidence-aware fusion with a configurable weight; opt-in COCO hazard footprints; diagnosis, evaluation and comparison tools; 14 new regression tests (120 total).
- **Demonstrated accuracy improvement:** only on snow tracks (false stops 18% -> 2-5% at equal measured safety, n=55) and, suggestively, person-ahead handling (n=1). **No improvement for leaf litter at the safety level of the current default could be demonstrated**; the data cannot separate litter from dense vegetation.
- **What would change that:** labelled litter-floor and dense-bush images (several hundred, per site/sequence) to fine-tune DeepScene's vegetation/ground classes, plus more genuine STOP frames to measure false-safe rates. Annotation format: `data/leaf_litter_pilot/annotations.json`; review sheet: `python scripts/compare_nav_variants.py <frames>`.
