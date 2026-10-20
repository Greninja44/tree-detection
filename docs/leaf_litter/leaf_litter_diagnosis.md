# Leaf-litter false stops - diagnosis

Tool: `python scripts/diagnose_nav.py <frames> --out outputs/leaf_litter` (8-panel trace per frame + counterfactual planner runs; figures not committed - they contain
FinnWoodlands/ForTrunkDet frames). Variant comparison: `python scripts/compare_nav_variants.py ft_img2374 ft_img0707 ft_img0691 00166 00254`.

## Decision trace for three frames the shipped method wrongly stops on
Counterfactuals re-run the planner with one component switched off (terrain-aware DeepScene, soft veto 0.7):

| frame (label GO) | full pipeline | DeepScene veto off | trunk footprints off | classical only |
|---|---|---|---|---|
| ft_img2374 (litter trail) | STOP | **ROUTE** | STOP | ROUTE |
| ft_img2319 (shaded litter) | STOP | **ROUTE** | STOP | ROUTE |
| ft_img2851 (grass/litter) | STOP | STOP | STOP | ROUTE |

## Root causes (each demonstrated, not assumed)
1. **DeepScene calls leaf litter "vegetation".** On the classical floor of ft_img2374 / 2319 the DeepScene argmax is *vegetation* for 73% / 77% of pixels (mean P(veg) 0.65 / 0.73,
   P(obstacle) 0.03 / 0.01). Soft-veto blocking probability is >=0.7 on 63% / 71% of that floor. Switching the veto off turns both into routes. **The obstacle class is not involved**:
   P(obstacle) is ~0 on every Portuguese floor measured.
2. **The veto produced hard obstacles.** `fuse_terrain_aware` set vetoed pixels to class OBSTACLE, which `detect_traversable_region` folds into the obstacle mask and the planner then
   inflates with the safety margin - a much harsher outcome than "not eligible". (Fixed in `fuse_confidence_aware`: hard OBSTACLE only from P(obstacle) >= 0.5 or trunk/hazard footprints.)
3. **The grass caution removed eligibility.** ft_img2851 stops even with the veto off: DeepScene calls the floor "grass" (67% of pixels) and the x0.75 caution multiplier pushed
   classical confidence (0.68 mean) below the 0.5 eligibility threshold. Caution should raise route *cost*, not remove ground. (Fixed: `p_cost`.)

## What does NOT separate litter from real vegetation (negative results)
On the dev frames (19 GO litter/trail/grass frames plus the dense-bush frame ft_img0707; `00166`/`00178` were STOP in the v1 labels and are AMBIGUOUS in v2):
- DeepScene mean P(vegetation) on the classical floor: litter/trail GO frames **0.26-0.76**; the dense-bush STOP frame (ft_img0707) **0.53** - inside the GO range. No threshold on it, per pixel
  or region-averaged, can separate them (the soft veto at 0.5-0.85 never fixed the litter frames).
- "Vegetation AND green" (living plants are green, litter is brown/grey): share of the floor satisfying it is ~0% for every frame including the dense-bush one (the bushes lie outside the
  classical floor region), so colour does not help either.
- Shadows, texture and the planner were not the cause: with the veto off the same planner and masks route all three frames.
- Trunk-footprint fusion is not the cause (footprints off: still STOP).
- No viewpoint/crop assumption is involved (start band and ROI are unchanged for the frames that route).

## Consequence
The vegetation signal cannot be tuned into a correct litter/bush decision with the data available. Any rule either keeps the dense-vegetation STOP (and loses litter floors) or keeps litter
(and loses that STOP). What can be fixed is the *mechanism* (causes 2 and 3) and the *hazard handling that should not depend on vegetation guesses* (people/vehicles, see COCO footprints).
See `leaf_litter_evaluation.md` for the measured trade-offs.

## Structured STOP reasons (new)
`PathPlan.reason_code` in {`NO_TRAVERSABLE_REGION`, `OBSTACLE_BLOCKS_CORRIDOR`, `LOW_CONFIDENCE_TERRAIN`, `INSUFFICIENT_CORRIDOR_WIDTH`, `NO_VALID_CONNECTED_CORRIDOR`}, shown in the GUI error and in
`metrics["stop_code"]`. `OBSTACLE_BLOCKS_CORRIDOR` is assigned only when the same ground would have had a valid corridor ignoring obstacles and margins. Stop codes on all GO frames (dev+final):
{
 "classical": {
  "NO_VALID_CONNECTED_CORRIDOR": 33,
  "OBSTACLE_BLOCKS_CORRIDOR": 3
 },
 "argmax_veto (v1)": {
  "INSUFFICIENT_CORRIDOR_WIDTH": 8,
  "NO_TRAVERSABLE_REGION": 20,
  "NO_VALID_CONNECTED_CORRIDOR": 27,
  "OBSTACLE_BLOCKS_CORRIDOR": 24
 },
 "current: soft veto 0.7": {
  "INSUFFICIENT_CORRIDOR_WIDTH": 9,
  "NO_TRAVERSABLE_REGION": 3,
  "NO_VALID_CONNECTED_CORRIDOR": 28,
  "OBSTACLE_BLOCKS_CORRIDOR": 14
 },
 "obstacle_veto_only (classical+DS obstacle)": {
  "NO_VALID_CONNECTED_CORRIDOR": 33,
  "OBSTACLE_BLOCKS_CORRIDOR": 3
 },
 "conf-aware w=1.0": {
  "INSUFFICIENT_CORRIDOR_WIDTH": 2,
  "LOW_CONFIDENCE_TERRAIN": 15,
  "NO_TRAVERSABLE_REGION": 5,
  "NO_VALID_CONNECTED_CORRIDOR": 19
 },
 "conf-aware w=0.6": {
  "INSUFFICIENT_CORRIDOR_WIDTH": 4,
  "LOW_CONFIDENCE_TERRAIN": 16,
  "NO_TRAVERSABLE_REGION": 1,
  "NO_VALID_CONNECTED_CORRIDOR": 16
 },
 "conf-aware w=0.4": {
  "INSUFFICIENT_CORRIDOR_WIDTH": 4,
  "LOW_CONFIDENCE_TERRAIN": 15,
  "NO_TRAVERSABLE_REGION": 0,
  "NO_VALID_CONNECTED_CORRIDOR": 17
 },
 "conf-aware w=0.3": {
  "INSUFFICIENT_CORRIDOR_WIDTH": 3,
  "LOW_CONFIDENCE_TERRAIN": 17,
  "NO_VALID_CONNECTED_CORRIDOR": 16
 },
 "conf-aware w=0.2": {
  "LOW_CONFIDENCE_TERRAIN": 13,
  "NO_VALID_CONNECTED_CORRIDOR": 13,
  "OBSTACLE_BLOCKS_CORRIDOR": 1
 },
 "conf-aware w=0.0": {
  "LOW_CONFIDENCE_TERRAIN": 8,
  "NO_VALID_CONNECTED_CORRIDOR": 3,
  "OBSTACLE_BLOCKS_CORRIDOR": 1
 },
 "classical +coco": {
  "NO_VALID_CONNECTED_CORRIDOR": 33,
  "OBSTACLE_BLOCKS_CORRIDOR": 4
 },
 "current: soft veto 0.7 +coco": {
  "INSUFFICIENT_CORRIDOR_WIDTH": 8,
  "NO_TRAVERSABLE_REGION": 3,
  "NO_VALID_CONNECTED_CORRIDOR": 28,
  "OBSTACLE_BLOCKS_CORRIDOR": 15
 },
 "conf-aware w=1.0 +coco": {
  "INSUFFICIENT_CORRIDOR_WIDTH": 2,
  "LOW_CONFIDENCE_TERRAIN": 15,
  "NO_TRAVERSABLE_REGION": 5,
  "NO_VALID_CONNECTED_CORRIDOR": 19
 },
 "conf-aware w=0.4 +coco": {
  "INSUFFICIENT_CORRIDOR_WIDTH": 4,
  "LOW_CONFIDENCE_TERRAIN": 15,
  "NO_TRAVERSABLE_REGION": 0,
  "NO_VALID_CONNECTED_CORRIDOR": 17,
  "OBSTACLE_BLOCKS_CORRIDOR": 0
 },
 "conf-aware w=0.2 +coco": {
  "LOW_CONFIDENCE_TERRAIN": 13,
  "NO_VALID_CONNECTED_CORRIDOR": 13,
  "OBSTACLE_BLOCKS_CORRIDOR": 1
 },
 "conf-aware w=0.0 +coco": {
  "LOW_CONFIDENCE_TERRAIN": 8,
  "NO_VALID_CONNECTED_CORRIDOR": 3,
  "OBSTACLE_BLOCKS_CORRIDOR": 1
 }
}

## Update: why RUGD works where DeepScene does not (exploratory)
RUGD-Segformer has classes for the things that matter here: `mulch`/`dirt`/`gravel` (ground, i.e. leaf litter), `grass`, `bush` (dense vegetation), `tree`, `person`, `vehicle`, `log`, `rock`. On the held-out litter, trail and grass frames
RUGD-based methods never stop (0/14) where the DeepScene veto stops on 14/14, and the dense-bush frame is still stopped. So the failure is the *label set* of DeepScene (no litter class), not the planner or the fusion
plumbing. RUGD's weaknesses are the opposite: no snow class, and unfamiliar machines (the robot) are often not recognised. The multi-source fusion combines them (see `leaf_litter_evaluation.md`; exploratory).

## Label audit (important)
Drawing every hazard box on its frame showed that several v1 boxes were wrong or shifted and that three STOP labels were debatable. The v2 corrections (geometry only) are logged in the annotations file. All v1-based false-safe figures are invalid.
