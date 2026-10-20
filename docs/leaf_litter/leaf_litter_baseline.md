# Leaf-litter false stops - baseline

Branch `feature/leaf-litter-fix` (from `feature/accuracy-tuning`, commit `1e1ab71`). Nothing pushed or merged.

## Repository state at the start
- `git status`: clean; branch `feature/accuracy-tuning`, HEAD `1e1ab71` (soft veto + tuning scripts).
- Existing tests: **106 passed**, 0 failed (`python -m pytest -q`).
- Relevant code: `src/traversability.py` (classes, `fuse_terrain_aware(veto_threshold)`, `detect_traversable_region`), `src/navigation.py` (planner),
  `src/pipeline.py`, `scripts/eval_navigation.py` (GT-track benchmark), `scripts/nav_scenarios.py`.

## Historical numbers, re-verified on the current code
`python scripts/eval_navigation.py --stride 3 --methods "Classical baseline,Terrain-aware (classical + DeepScene)" --veto-threshold 0.7` (68 FinnWoodlands frames with a GT groomed track; frames mix train/val/test of the same recordings):

| Method | Routes found | Heading on correct side | Mean heading error | GT-track recall |
|---|---|---|---|---|
| Classical baseline | 56/68 | 77% | 12.0° | 85% |
| Classical + DeepScene soft veto 0.7 | 63/68 | 81% | 12.0° | 80% |

These reproduce the earlier figures (the earlier "about 83%" side-agreement was the unthresholded variant). The RUGD (59%) and ADE20K (32/68 routes) figures
were not re-run: those weights were deleted as bad/superseded.

## What this benchmark cannot show
Every GT frame is a snow groomed track, so it says nothing about leaf litter. A new labelled set was built for this task (see `leaf_litter_evaluation.md`);
on it the **shipped soft veto has a 35% false-stop rate on held-out frames versus 1% for the classical baseline.**
