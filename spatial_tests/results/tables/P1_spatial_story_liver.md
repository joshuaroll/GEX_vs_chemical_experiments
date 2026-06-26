# Spatial story EDA -- liver

Region definition: published anatomical zones (yu2022 L5 category).
Spots x genes: **3381 x 16569**.  Region spot counts: Zone1=829, Zone2_3=680, Zone3=555, Zone2_1=491, Zone2_2=345, portal_area=302, central_area=179.

The predicted region-resolved arm needs each gate to hold. Verdict per gate:

## GATE 1 -- are regions distinguishable above the within-region noise floor?  [WEAK]
- within-region split-half Pearson (noise ceiling): **0.9954** +/- 0.0027
- between-region Pearson: **0.9875** +/- 0.0093
- gap (within - between): **0.0079**;  between-region distance = **2.7x** the within-region noise
- read: two halves of the SAME region differ by 0.0046 (distance); two DIFFERENT regions differ by 0.0125. Signal is the ratio of these.

## GATE 2 -- is region a real axis of variance, or swamped by depth?  [PASS]
- region explains **13.8%** of top-PC expression variance
- sequencing depth explains 2.8%; residual 83.4%

## GATE 3 -- does the region signal survive into the model gene space, as biology?  [WEAK]
- top-50 region-discriminating genes retained in 10,716 space: **66%**
- technical (mito/ribo/MALAT1/Hb) fraction of top-50: **22%**
- top genes: CYP3A4, CYP2E1, SERPINA1, MT-CO1, MT-ND4, IGKC, CYP1A2, C7, ALDOB, APOA1, MT-ATP6, MT-ND2, HP, MT-ND3, MT-ND1

## BRIDGE -- region-basal similarity in the model gene space
- liver Visium -> 10,716 coverage: **84.9%**
- off-diagonal Pearson: min 0.972, mean 0.989, max 0.998

## GATE 4 (cross-species) / GATE 5 (OOD) -- scaffolded
- needs region-matched rodent labels (liver mouse = whole-sample only, r=0.927 prior; kidney mouse = .rds not yet extracted). OOD Mahalanobis from the cancer-line manifold to follow.

## Caveats
- Regions are the published L5 zone labels; cluster definition is external to the separability test. Single section (yu2022 L5).
- Pearson on log1p pseudobulks; depth regressed only as a single covariate (a fuller PVCA would add donor/section). Thresholds for PASS/WEAK/FAIL are interpretive anchors.

## Supporting (demoted niche battery)
- cell-type niche architecture (nhood/co-occurrence/Ripley) is in the per-organ `*_spatial_eda` figures; it describes tissue architecture but the model conditions on region pseudobulks, not niches, so it is context not evidence.

## Figures
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/liver_spatial_story/00_region_map.png` -- regions in space (orientation)
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/liver_spatial_story/01_gate1_separability.png` -- GATE 1 headline (within vs between)
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/liver_spatial_story/02_gate2_variance.png` -- GATE 2 variance partition
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/liver_spatial_story/03_gate3_discriminating_genes.png` -- GATE 3 top region genes + retention
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/liver_spatial_story/04_bridge_similarity_modelspace.png` -- region basals in model space
