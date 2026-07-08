# The Li/Tong DILI 8-layer DNN under honest evaluation

**Question:** Li/Tong 2020 (Front. Bioeng. Biotechnol., 10.3389/fbioe.2020.562677) report AUROC ~0.798
for a fully-connected 8-layer DNN on 978 LINCS L1000 landmark genes (measured differential expression).
Their own code inflates that number three ways. How much of the 0.798 is DILI signal, and how much is
inflation? And once the inflation is removed, does measured gene expression beat chemical structure?

**Answer: almost all of the 0.798 is inflation, and no expression advantage survives.** Reproducing their
exact 8-layer DNN and removing the three mechanisms one at a time drops the same network from 0.781 to
0.508 (drug-disjoint, per-drug scoring), i.e. to chance. The single biggest inflator is their leaky
profile-level split (-0.185); test-set model selection adds -0.058; profile-level pseudoreplication
scoring adds -0.030. Under honest evaluation the measured-GEX DNN ties our small-head measured baseline
at chance and sits *below* the ~0.58-0.60 cluster where structure (ECFP4), ChemBERTa, and MultiDCP land.
Extra classifier capacity (8 layers vs a linear/MLP head) does not rescue measured expression.

## The inflation decomposition (C0..C4)

Same 8-layer DNN, measured MODZ input, one fix turned on at a time. Each consecutive drop is that fix's
contribution to the collapse from the ~0.798 headline. `--arch bn` (BatchNorm+Dropout(0.2) variant),
Adam lr 1e-3, balanced `pos_weight`, batch 256, early-stop patience 10. Source:
`P_wangli_dnn_honest_full.csv` (3 seeds on the fixed profile split, 10 GroupShuffleSplit resamples on
the disjoint splits).

| Config | Split | Score | Selection | AUROC (mean ± sd) | Fix isolated (drop from previous) |
|---|---|---|---|---|---|
| C0 | profile (leaky) | profile | test | 0.781 ± 0.004 | none, reproduces their ~0.798 headline |
| C1 | drug-disjoint | profile | test | 0.596 ± 0.071 | remove leaky split (-0.185) |
| C2 | drug-disjoint | profile | honest | 0.538 ± 0.080 | honest group-disjoint val selection (-0.058) |
| C3 | drug-disjoint | drug | honest | 0.508 ± 0.033 | per-drug scoring = fully honest (-0.030) |
| C4 | scaffold-disjoint | drug | honest | 0.488 ± 0.035 | scaffold OOD (-0.019) |

C0 reproduces the published band: 0.781 sits inside [0.78, 0.82]. From there the drops are monotone and
add to the full collapse. The leaky split alone accounts for two thirds of the gap; test-set selection
and profile scoring account for the rest. By C3 the network is at chance, and holding whole Murcko
scaffolds out (C4) pushes it just below chance. Figure:
`results/figures/P_wangli_dnn_honest_decomposition.png`.

## Merged comparison of honest per-drug AUROC across feature arms

The honest Li/Tong 8-layer DNN (measured GEX) placed alongside this project's models, all on the same
split + inner-validation + drug-level scoring machinery (`scripts/tox_finetune_multidcp.py`, reused
verbatim), so the numbers merge directly. Every cell is **drug-level AUROC** (one mean-probability
prediction per compound). Disjoint columns are mean ± std over 10 GroupShuffleSplit resamples; the leaky
profile column is a 3-seed point estimate kept only as a contrast. Sources:
`P_wangli_dnn_honest_full.csv` (the DNN row) and `P_tox_finetune_multidcp_v3.csv` (all other rows).

| feature arm | leaky profile (contrast) | drug-disjoint | scaffold-disjoint |
|---|---|---|---|
| chemberta_meta (struct + cell/dose, finetuned) | 0.992 | **0.603 ± 0.046** | 0.575 ± 0.040 |
| tuned MultiDCP (SMILES→predGEX, finetuned) | 0.934 | **0.600 ± 0.050** | 0.599 ± 0.046 |
| chemberta_ft (pretrained struct, finetuned) | 0.991 | 0.597 ± 0.037 | 0.582 ± 0.055 |
| structure (ECFP4) | 0.988 | 0.575 ± 0.045 | 0.583 ± 0.065 |
| chemberta_frozen (pretrained struct, frozen) | 0.986 | 0.570 ± 0.058 | 0.568 ± 0.085 |
| frozen MultiDCP | 0.740 | 0.521 ± 0.075 | 0.523 ± 0.044 |
| cell_dose_only (assay-metadata floor) | 0.547 | 0.508 ± 0.060 | 0.503 ± 0.054 |
| **Li/Tong 8-layer DNN, measured GEX (honest)** | 0.585 | **0.508 ± 0.033** | **0.488 ± 0.035** |
| measured GEX, small head (mean-shift) | 0.598 | 0.464 ± 0.037 | 0.501 ± 0.037 |

The two measured-expression classifiers (8-layer DNN 0.508, small head 0.464 drug-disjoint) sit at the
bottom with the assay-metadata floor. The real predictors (structure, ChemBERTa, MultiDCP) cluster at
~0.57-0.60 and overlap each other within one standard deviation. No arm clears ~0.60 on novel chemistry.

## Narrative

Li/Tong's headline number and their "drug-based split" number come from the same network; the difference
is only which inflation mechanisms are left on.

Their partition was drug-disjoint by design, and their own Figure 8B drug-based split reports 0.769. But
0.769 still rode profile-level scoring (each of a held-out drug's many correlated LINCS profiles counted
as an independent test example) and test-set model selection (they fit with
`validation_data=(X_test, y_test)` and a `ModelCheckpoint(monitor='val_monitor_f', save_best_only=True)`,
so they keep the epoch that scores best on the test set). Reproducing that recipe on our strict
compound-disjoint 5,141-profile subset gives 0.596 (C1). Removing test-set selection (honest
group-disjoint validation carved from the training compounds, disjoint from both train and test) and
switching to one prediction per compound drops the same network to 0.508 (C3), which is chance. Holding
whole Murcko scaffolds out (C4) leaves it at 0.488.

Fixing the two remaining inflations therefore takes the measured-GEX DNN below even the modest honest
ceiling. That ceiling is ~0.58-0.60, and it is set by chemical structure, not expression: ECFP4, frozen
and finetuned ChemBERTa, and finetuned MultiDCP all tie there. Measured expression, given either a
linear head or a full 8-layer DNN, does not reach it. So under honest evaluation the measured-GEX DNN
ties every feature at the floor and beats none; there is no expression advantage, and the modest signal
that does exist is carried by structure.

## Controls

- **Label permutation (negative control):** with the per-drug label map shuffled, the fully-honest
  configs return chance. Drug-disjoint drug-level 0.498 ± 0.047; scaffold-disjoint 0.530 ± 0.054. No
  label leaks through the model or the shared inputs. Source:
  `P_wangli_dnn_honest_full_permute.csv`.
- **Reproduction gate:** C0 = 0.781 ∈ [0.78, 0.82], so the honest harness reproduces their published
  band before any fix is applied. The collapse is caused by the fixes, not by a broken reimplementation.
- **Consistency with the small head:** the honest 8-layer DNN on measured GEX (0.508 drug-disjoint,
  0.488 scaffold) lands in the same at-chance region as our small-head measured arm (0.464 / 0.501),
  confirming the null is a property of the measured feature, not of classifier capacity.

## Reading

- The 0.798 headline is an artifact stack: a leaky split (-0.185), test-set selection (-0.058), and
  profile pseudoreplication (-0.030). None of the three is DILI signal on novel compounds.
- Profile-level scoring on a drug-disjoint split is the subtle one. DILI is a per-drug label but there
  are many profiles per drug; scoring per profile lets a feature that varies within a drug harvest
  pseudoreplication credit that per-drug scoring removes.
- Measured gene expression carries no per-drug DILI signal on held-out chemistry (0.508 drug-disjoint,
  0.488 scaffold). The honest ceiling for any feature here is ~0.60 and it is a structure ceiling.

## Caveats

- Our C1 (0.596) is below Li/Tong's published 0.769 drug-based number because our split is strictly
  compound-disjoint and scored on the 5,141-profile MODZ subset, a tighter partition than theirs.
- C3 (0.508) is below the ~0.59 rough prior in the task framing. Honest group-disjoint validation
  selection drives the fully-honest measured-GEX number to chance rather than 0.59; the ~0.59 cluster in
  the merged table is where structure-based features land, not measured expression. The qualitative
  finding (0.798 is inflation; no expression advantage survives) is unchanged and slightly stronger.
- The 8-layer DNN here uses the BatchNorm+Dropout(0.2) variant (`--arch bn`). The published weights that
  hit ~0.798 are pure Dense/ELU; `--arch plain` builds that variant. C0 reproduces the band either way,
  so the decomposition does not hinge on the architectural detail.

## Reproduction

- Data: `data/processed/wangli_multidcp_finetune.npz` (5,141 profiles, `measured_modz` = 978 landmark
  MODZ, z-scored DE; 621 compounds, 433 Murcko scaffolds). Same file and subset as
  `P_tox_finetune_multidcp_v3.csv`.
- Harness: `scripts/wangli_dnn_honest.py`. Reuses (verbatim, with attribution) `inner_split`,
  `auroc_profile_and_drug`, the `GroupShuffleSplit(n_splits=N, test_size=0.2, random_state=0)`
  drug/scaffold resamples, the `usage=='Training'/'Test'` profile split, and the drug-level
  `--permute-labels` control from `scripts/tox_finetune_multidcp.py`. Only the classifier changed (small
  head → 8-layer ELU DNN). Env `dili_v04_env`. CUDA hygiene: `--gpu` sets `CUDA_VISIBLE_DEVICES` before
  `import torch`, auto-picks the freest GPU, leaves the rest free.
- Full grid: `python scripts/wangli_dnn_honest.py --tag full --seeds 3 --n-resamples 10`
  → `P_wangli_dnn_honest_full.csv` (+ `_per_resample.csv`).
- Permutation control:
  `python scripts/wangli_dnn_honest.py --tag full --permute-labels --split drug,scaffold --score drug --selection honest`
  → `P_wangli_dnn_honest_full_permute.csv`.
- Figure: `python scripts/plot_wangli_dnn_honest.py`
  → `results/figures/P_wangli_dnn_honest_decomposition.png` (+ `.pdf`).
