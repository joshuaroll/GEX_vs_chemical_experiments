# Tox-finetune of the pretrained MultiDCP on the Li/Tong DILI benchmark

**Question:** does starting the expression arm from a *pretrained* MultiDCP backbone, finetuned on the
benchmark's own data, flip this project's long-standing verdict that chemical structure ≥ gene
expression for DILI?

**Answer: no, and the null is strong.** On the honest per-drug metric and leakage-corrected splits,
every real predictor (ECFP4, ChemBERTa frozen/finetuned, MultiDCP) clusters at ~0.57–0.60 AUROC and is
statistically indistinguishable; measured expression and a cell/dose-only baseline are at chance. A
*fair* finetuned structure encoder (ChemBERTa) ties MultiDCP exactly, and giving that structure encoder
the same cell+dose inputs MultiDCP gets (`chemberta_meta`) still ties it — so the predicted-gene-
expression bottleneck adds nothing over the same inputs used directly. The dramatic "structure
collapses, expression wins" seen early was a profile-level pseudoreplication artifact.

## Results — drug-level AUROC (the honest per-drug metric)

v3, paper-consistent protocol: resampled `GroupShuffleSplit` (10 resamples) for the disjoint splits,
Wang/Li's fixed Training/Test (3 seeds) for the profile split. Source: `P_tox_finetune_multidcp_v3.csv`
(+ `_v2.csv` for tuned_F/both; `_distshift.csv`; `_permute.csv`).

| feature arm | profile Test (leaked) | drug-disjoint | scaffold-disjoint |
|---|---|---|---|
| structure (ECFP4) | 0.988 | 0.574 ± 0.045 | 0.583 ± 0.065 |
| chemberta_frozen (pretrained struct, frozen) | 0.986 | 0.570 ± 0.058 | 0.568 ± 0.085 |
| chemberta_ft (pretrained struct, finetuned) | 0.991 | 0.597 ± 0.037 | 0.582 ± 0.055 |
| **chemberta_meta (struct + cell/dose, finetuned)** | 0.992 | **0.603 ± 0.046** | 0.575 ± 0.040 |
| **tuned MultiDCP (SMILES→predGEX, finetuned)** | 0.934 | **0.600 ± 0.050** | 0.599 ± 0.046 |
| frozen MultiDCP | 0.740 | 0.521 ± 0.075 | 0.523 ± 0.044 |
| measured GEX (mean-shift) | 0.598 | 0.464 ± 0.037 | 0.501 ± 0.037 |
| dist_shift (distributional/density-ratio probe) | 0.471 | 0.467 ± 0.060 | 0.502 ± 0.046 |
| cell_dose_only (assay-metadata floor) | 0.547 | 0.508 ± 0.060 | 0.503 ± 0.054 |

## The decisive ablation

`chemberta_meta` (SMILES + cell + dose, direct) vs `tuned MultiDCP` (SMILES + cell + dose routed
through predicted GEX) come out **identical** (0.603 vs 0.600 drug-disjoint; 0.575 vs 0.599 scaffold,
overlapping). Routing the same inputs through a predicted-gene-expression bottleneck adds nothing. This
retires even the narrow claim "a GEX-pretrained SMILES bottleneck beats fingerprints" — it doesn't beat
a plain finetuned chemical LM.

## Controls

- **Label permutation (negative control):** with the per-drug label map shuffled, every arm returns
  chance on drug-disjoint (structure 0.48, measured 0.53, MultiDCP 0.48, chemberta_meta 0.43). No
  leakage through the shared inputs or model selection.
- **cell_dose_only:** 0.51 on both disjoint splits — the dose confound (top >16 µM bin is 94.5%
  positive) does not produce a generalizable drug-disjoint signal.
- **dist_shift:** a distributional/density-ratio summary of MODZ (scRatio-spirit probe) is at chance —
  the distributional view adds nothing over the mean on bulk data.

## Reading

- **Structure did not collapse.** ECFP4's profile-level 0.37–0.49 on the disjoint splits was a
  pseudoreplication/anti-generalization artifact (median 4 / mean 8.3 profiles per drug; Vorinostat =
  14% of profiles). At drug level, structure is a modest, real ~0.58, and a fair finetuned structure
  encoder is not sandbagged (0.58–0.60, matching MultiDCP).
- **No expression advantage exists** at drug level on novel chemistry; measured expression is at chance.
- The profile Test split stays badly leaked for structure (0.99), so the benchmark's 0.798 headline is
  only meaningful for the measured/expression arms.

## Why the profile-level view misled

DILI is a per-drug label but there are median 4 / mean 8.3 profiles per drug. Profile-level AUROC on a
drug-disjoint split counts each of a drug's profiles as an independent call; varies-per-drug features
(expression) harvest that pseudoreplication credit, constant-per-drug features (structure) do not.
Drug-level scoring removes it and the expression lead disappears.

## Caveats

- Backbone is not naive to the benchmark (L5 MultiDCP trained to predict these LINCS profiles).
- ~58% of profiles are 6H exposures; the pretrained model trained on 24H only. 859/6000 profiles
  dropped (unresolved SMILES + cells missing from CCLE basal, incl. primary hepatocytes PHH).
- ChemBERTa-77M and the MultiDCP backbone differ in size/pretraining; the tie is at the low ceiling
  (~0.60), so the comparison is between two weak-but-equal predictors, not two strong ones.

## Single-cell follow-up (scoped, then parked)

`scRatio` (flow-matching density ratio, theislab, arXiv 2602.24201) was considered as a distributional
expression feature. Gate 0 (DILIst overlap): **sci-Plex 3 gives only 30 labeled drugs (23+/7−)** —
underpowered. Tahoe-100M is the only viable single-cell source but a heavy lift with a low prior after
the bulk `dist_shift` null. Parked. See
`0_project_documents/scRatio_dili_single_cell_scoping_2026-07-02.md`.

## Reproduction

- Input: `scripts/build_wangli_multidcp_inputs.py` → `data/processed/wangli_multidcp_finetune.npz`
  (5,141 profiles). Harness: `scripts/tox_finetune_multidcp.py`. Backbone:
  `MultiDCP/trained_models/L5_split1_05032023.pt`. Env `dili_v04_env`.
- v3: `--splits test,drug,scaffold --variants
  structure,measured,frozen_pred,chemberta_frozen,cell_dose_only,tuned_E,chemberta_ft,chemberta_meta
  --seeds 3 --n-resamples 10 --max-epochs 20 --patience 6 --tag v3`.
- Two independent reviews (code + methodology) passed: no train/test leakage; the profile→drug metric
  change and the fair-structure baseline were the decisive corrections.
