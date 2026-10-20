# Leaf-litter false stops - evaluation (v2 labels, EXPLORATORY)

Reproduce: `python scripts/eval_leaf_litter.py --select --with-rugd --output reports/leaf_litter_metrics_rugd.json` (RUGD costs ~5 s/frame the first time, cached on disk afterwards; ~20 min cold).
Visuals: `python scripts/compare_nav_variants.py ft_img2374 ft_img0707 ft_img0691 00166 00254`; label audit sheet: every hazard box drawn on its frame.

## Status: exploratory - read this first
1. **Labels were revised after results were seen.** A visual audit (hazard boxes drawn on the frames) found my v1 boxes wrong or shifted in several frames (e.g. in `ft_img0691` the person is on the *left*, the box was on the right),
   and showed that three STOP labels (`ft_img0691`, `00166`, `00178`) are debatable. v2 corrects the boxes from the image geometry and sets those three to AMBIGUOUS (changelog inside `data/leaf_litter_pilot/annotations.json`; v1 kept as `annotations_v1.json`).
   **Every earlier "false-safe" figure in my reports and chat was computed on the v1 boxes and is invalid.**
2. **No clean held-out set remains.** All pilot frames have now been viewed; the multi-source method was designed after seeing the RUGD variants' test results. Treat all numbers as exploratory, not as a validated test.
3. **Labels are a single-annotator pilot (the assistant, by eye, unreviewed).** Protocol in the annotations file. 167 dev / 69 final-test labelled frames; 19 AMBIGUOUS frames excluded from corridor rates.
4. **STOP frames: dev 1, final test 0.** False-safe rates on corridor labels are **unmeasurable**. The only safety signal is the *hazard-box* check (does the route cross a labelled person/robot box; counted on every frame with a box: 7 dev, 7 final test).
5. Splits: dev = ForTrunkDet val + FinnWoodlands train GT frames + frames used while diagnosing; final_test = ForTrunkDet test + Finn val/test GT frames. Counts: dev GO 166 (147 snow-track + 19 pilot), final-test GO 69 (55 snow-track + 14 pilot litter/trail).
   CIs are cluster bootstraps over sequences (16 dev, 11 final-test) and are wide. The soft-veto threshold 0.7 was chosen earlier on a set mixing train/val/test Finn frames.
6. "ROUTE" is an image-space plan, not robot safety (no depth, calibration or vehicle geometry).

## Methods (identical frames, planner and rules)
classical baseline; shipped terrain-aware + DeepScene soft veto 0.7 ("current"); confidence-aware fusion (w = vegetation penalty weight; hard obstacle only from P(obstacle) >= 0.5; grass caution = cost only); `+coco` = COCO person/vehicle/animal footprints added;
RUGD-Segformer variants (re-downloaded): alone, + classical hybrid, classical-led + obstacle veto, label-veto, confidence-aware; **multi-source** = ground is the union of classical, RUGD and DeepScene evidence; hard veto where RUGD obstacles/bushes or DeepScene obstacles have P >= 0.5 (DeepScene "vegetation" is deliberately not a veto).

## Pre-registered selection rule and its outcome
Lowest dev false-stop among {confidence-aware, RUGD} variants with dev safety (corridor-label and hazard) no worse than the current veto. **Chosen on dev: `conf-aware w=0.0`.**
On the final test it routes through 2/7 hazard frames (current: 0/7; 1/7 with COCO), so it is **not adopted as a default**. Multi-source is excluded from automatic selection (post-hoc).

## Results - dev
| method | route availability | false-stop [95% CI] (k/n) | hazard frames routed through | snow heading side OK | snow mean heading err | GT-track recall |
|---|---|---|---|---|---|---|
| classical | 79% | 21% [4-45] (35/166) | 1/7 | 78% | 14.9° | 82% |
| classical +coco | 78% | 22% [5-46] (36/166) | 1/7 | 77% | 14.9° | 82% |
| current: soft veto 0.7 | 81% | 18% [7-37] (30/166) | 0/7 | 82% | 13.4° | 77% |
| conf-aware w=1.0 | 85% | 14% [4-33] (24/166) | 0/7 | 85% | 12.0° | 88% |
| conf-aware w=0.0 | 95% | 5% [0-14] (8/166) | 0/7 | 86% | 11.3° | 94% |
| conf-aware w=0.0 +coco | 95% | 5% [0-14] (8/166) | 0/7 | 86% | 11.4° | 94% |
| RUGD alone | 86% | 14% [3-29] (23/166) | 0/7 | 63% | 18.8° | 83% |
| RUGD+classical hybrid | 90% | 10% [2-19] (16/166) | 0/7 | 65% | 18.5° | 88% |
| RUGD classical-led + obstacle veto | 75% | 25% [7-49] (42/166) | 0/7 | 78% | 13.4° | 80% |
| multi-source | 95% | 4% [0-12] (7/166) | 0/7 | 65% | 17.7° | 94% |

## Results - final test (no longer a clean test, see Status)
| method | route availability | false-stop [95% CI] (k/n) | hazard frames routed through | snow heading side OK | snow mean heading err | GT-track recall |
|---|---|---|---|---|---|---|
| classical | 99% | 1% [0-8] (1/69) | 2/7 | 89% | 5.1° | 95% |
| classical +coco | 99% | 1% [0-8] (1/69) | 1/7 | 89% | 5.3° | 95% |
| current: soft veto 0.7 | 65% | 35% [19-100] (24/69) | 0/7 | 84% | 7.5° | 82% |
| conf-aware w=1.0 | 75% | 25% [11-100] (17/69) | 1/7 | 85% | 8.5° | 89% |
| conf-aware w=0.0 | 94% | 6% [0-22] (4/69) | 2/7 | 84% | 9.4° | 95% |
| conf-aware w=0.0 +coco | 94% | 6% [0-22] (4/69) | 1/7 | 84% | 9.4° | 95% |
| RUGD alone | 93% | 7% [0-12] (5/69) | 1/7 | 46% | 32.4° | 68% |
| RUGD+classical hybrid | 99% | 1% [0-4] (1/69) | 1/7 | 63% | 19.0° | 88% |
| RUGD classical-led + obstacle veto | 84% | 16% [5-22] (11/69) | 2/7 | 82% | 8.7° | 86% |
| multi-source | 94% | 6% [0-15] (4/69) | 0/7 | 75% | 16.7° | 88% |

False-stop rate on held-out **GO** frames by terrain (final test):

| method | dry_litter (n=6) | grass_litter (n=2) | leaf_litter (n=2) | snow_track (n=55) | trail (n=4) |
|---|---|---|---|---|---|
| classical | 17% | 0% | 0% | 0% | 0% |
| current: soft veto 0.7 | 100% | 100% | 100% | 18% | 100% |
| conf-aware w=1.0 | 100% | 100% | 100% | 5% | 100% |
| conf-aware w=0.0 | 17% | 100% | 0% | 0% | 25% |
| RUGD+classical hybrid | 0% | 0% | 0% | 2% | 0% |
| multi-source | 0% | 0% | 0% | 7% | 0% |

Hazard frames the routes crossed (final test): classical `ft_img0631`, `ft_img0691`; conf-aware w=0 same two (w=0 + COCO: `ft_img0631`); RUGD hybrid `ft_img2855`; current soft veto none; **multi-source none**. Dev: classical `ft_img0500` only.

## Latency / memory (Jetson Orin Nano Super, CPU, 480 px)
| configuration | total / frame | notes |
|---|---|---|
| classical (default) | 0.43 s | peak RSS ~1.1 GB |
| terrain-aware DeepScene (veto or confidence-aware) | 1.4 s | peak RSS ~1.2 GB |
| + COCO hazards | +0.25 s | |
| **multi-source (RUGD + DeepScene + classical)** | **5.1 s** | segment+fuse 4.75 s; peak RSS ~2.0 GB - not real-time on this CPU |

## Trade-off analysis
- **Leaf litter, trails and grass (14 held-out GO frames):** the shipped veto and every DeepScene vegetation penalty >= 0.4 stop on 14/14. RUGD-based methods and multi-source stop on **0/14**: RUGD labels litter `mulch`/`dirt` and bushes `bush`, which DeepScene's "vegetation" class cannot do.
- **Hazards:** multi-source routes through 0/7 labelled hazard frames (dev 0/7), equal to the shipped veto (0/7) and better than classical (2/7) and weight-0 confidence-aware (2/7; 1/7 with COCO). RUGD knows persons/vehicles/trees; COCO adds little on top of it here.
- **Costs:** snow-track heading is worse for RUGD-based methods (side OK 63-75% vs 84-89%, mean error 16.7-19.0° vs 5.1-7.5°); multi-source falsely stops on 4/55 snow frames (7%, classical 0%); latency 5.1 s/frame.
- **No setting dominates.** Fastest/best on snow: classical (but routes through persons/robots). Safest on litter+hazards in this sample: multi-source (slow, worse snow heading). A scene-adaptive choice (e.g. DeepScene-based for snow, multi-source otherwise) was not built or tested.

## Conclusion
- **Software (done, 123 tests):** structured STOP reason codes; hard-obstacle vs vegetation separation; cost-only grass caution; confidence-aware fusion; multi-source fusion (opt-in method); opt-in COCO hazard footprints; diagnosis, evaluation and comparison tools; versioned pilot annotations with an audit trail.
- **Exploratory accuracy evidence:** the leaf-litter false-stop problem *can* be removed - held-out false stops 35% -> 6% overall (0/14 on litter/trail) with 0/7 hazard routes - by using RUGD semantics, at the cost of speed and snow-track heading. This is **not** a validated improvement (revised labels, no clean test, 7 hazard frames, 0 STOP frames).
- **To validate it:** label a fresh set from your own camera (several hundred frames: open litter floors, dense bushes, people/machines, true dead-ends) with at least two reviewers, freeze the protocol, then score once. Annotation format: `data/leaf_litter_pilot/annotations.json`.
