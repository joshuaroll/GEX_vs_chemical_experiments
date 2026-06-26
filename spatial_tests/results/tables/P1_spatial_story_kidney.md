# Spatial story EDA -- kidney

Region definition: cortex vs medulla from canonical markers (external, non-circular).
Spots x genes: **3007 x 19209**.  Region spot counts: medulla=2409, cortex=598.

The predicted region-resolved arm needs each gate to hold. Verdict per gate:

## GATE 1 -- are regions distinguishable above the within-region noise floor?  [PASS]
- within-region split-half Pearson (noise ceiling): **0.9984** +/- 0.0010
- between-region Pearson: **0.9749** +/- 0.0000
- gap (within - between): **0.0235**;  between-region distance = **15.6x** the within-region noise
- read: two halves of the SAME region differ by 0.0016 (distance); two DIFFERENT regions differ by 0.0251. Signal is the ratio of these.

## GATE 2 -- is region a real axis of variance, or swamped by depth?  [PASS]
- region explains **10.6%** of top-PC expression variance
- sequencing depth explains 2.5%; residual 87.0%

## GATE 3 -- does the region signal survive into the model gene space, as biology?  [PASS]
- top-50 region-discriminating genes retained in 10,716 space: **80%**
- technical (mito/ribo/MALAT1/Hb) fraction of top-50: **0%**
- top genes: PODXL, NPHS2, PTGDS, IL1RL1, C1QL1, MME, HTRA1, ATP1A1, FGF1, CLIC5, CDKN1C, CD24, CRHBP, SPOCK2, PCOLCE2

## BRIDGE -- region-basal similarity in the model gene space
- kidney Visium -> 10,716 coverage: **90.2%**
- off-diagonal Pearson: min 0.972, mean 0.972, max 0.972

## GATE 4 (cross-species) / GATE 5 (OOD) -- scaffolded
- needs region-matched rodent labels (liver mouse = whole-sample only, r=0.927 prior; kidney mouse = .rds not yet extracted). OOD Mahalanobis from the cancer-line manifold to follow.

## Caveats
- Region was MARKER-DEFINED (cortex/medulla), so the seed markers re-appear among the top discriminators by construction; the non-marker genes in the list and the whole-transcriptome separability are the independent signal. Single section (GSM6047774); 23 sections + donor axis available for the variance gate.
- Pearson on log1p pseudobulks; depth regressed only as a single covariate (a fuller PVCA would add donor/section). Thresholds for PASS/WEAK/FAIL are interpretive anchors.

## Supporting (demoted niche battery)
- cell-type niche architecture (nhood/co-occurrence/Ripley) is in the per-organ `*_spatial_eda` figures; it describes tissue architecture but the model conditions on region pseudobulks, not niches, so it is context not evidence.

## Figures
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/kidney_spatial_story/00_region_map.png` -- regions in space (orientation)
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/kidney_spatial_story/01_gate1_separability.png` -- GATE 1 headline (within vs between)
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/kidney_spatial_story/02_gate2_variance.png` -- GATE 2 variance partition
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/kidney_spatial_story/03_gate3_discriminating_genes.png` -- GATE 3 top region genes + retention
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/kidney_spatial_story/04_bridge_similarity_modelspace.png` -- region basals in model space
