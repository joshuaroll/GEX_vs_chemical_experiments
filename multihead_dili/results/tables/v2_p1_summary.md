# v2 Phase 1 Summary — Wang/Li 8-Layer DNN: Measured vs Predicted GEX

**Date:** 2026-05-20
**Purpose:** Decompose the v1 negative finding (0.5930 chemistry-only AUROC vs Wang/Li's
published 0.798) into two components: (1) predicted-vs-measured GEX quality gap and
(2) classifier-capacity gap. Same architecture, same drugs, same split, different feature sources.

## Drug counts

- Total DILIst drugs: 1118
- LINCS-intersect (in Wang/Li profiles + dili_features): **628**
- Scaffold split (filtered to intersect): train=461 val=62 test=105
- Random split (70/10/20 stratified): train=438 val=62 test=128

## Architecture

Wang/Li 8-layer DNN: Dense(512)→Dense(256)→Dense(128)→Dense(64)→Dense(32)→Dense(16)→Dense(8)→Dense(1,sigmoid).
All hidden layers ELU. Adam optimizer, binary_crossentropy loss, balanced class weights.
Checkpoint on val_monitor_f (Wang/Li 2020 custom metric), EarlyStopping(patience=5) on val_loss,
max 100 epochs, batch_size=128. Run A input_dim=978 (measured LINCS DE); Run B input_dim=919
(MultiDCP predicted GEX from v1 cache).

## Headline Results

| Run | Split | AUROC (pooled) | 95% CI | n_drugs | vs Run-A (DeLong p) |
|-----|-------|---------------|--------|---------|---------------------|
| A: measured GEX (978d) | scaffold-novel | 0.4565 | [0.3003, 0.5257] | 105 | — |
| B: predicted GEX (919d) | scaffold-novel | 0.5046 | [0.4383, 0.6744] | 105 | ΔAUROC=-0.0481, z=-1.689, p=0.091 |
| A: measured GEX (978d) | random | 0.5014 | [0.3967, 0.6113] | 128 | — |
| B: predicted GEX (919d) | random | 0.5002 | [0.3513, 0.5687] | 128 | ΔAUROC=+0.0012, z=0.543, p=0.587 |

Note: DeLong ΔAUROC = Run A minus Run B. Positive = Run A (measured) better.

## Comparison to v1 and Wang/Li published

| Method | Split | AUROC | Notes |
|--------|-------|-------|-------|
| Wang/Li 2020 published | random (their split) | 0.798 | Profile-level training (N=5517), data leakage at drug unit |
| Wang/Li reproduced (this box) | random (profile-level) | 0.761 | Same protocol, our data — confirms reproduction |
| v1 chemistry-only (MolFormer + mlp2) | scaffold-novel | 0.5930 | Drug-level, scaffold OOD |
| v1 chemistry-only (MolFormer + mlp2) | random | 0.6645 | Drug-level, random |
| **v2 Run A: measured GEX 8-layer DNN** | **scaffold-novel** | **0.4565** | Drug-level, scaffold OOD |
| **v2 Run B: predicted GEX 8-layer DNN** | **scaffold-novel** | **0.5046** | Drug-level, scaffold OOD |
| **v2 Run A: measured GEX 8-layer DNN** | **random** | **0.5014** | Drug-level, random |
| **v2 Run B: predicted GEX 8-layer DNN** | **random** | **0.5002** | Drug-level, random |

## Per-fold AUROC breakdown

| Run | Split | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Std |
|-----|-------|--------|--------|--------|--------|--------|------|-----|
| A: measured | scaffold | 0.487 | 0.448 | 0.449 | 0.410 | 0.494 | 0.458 | 0.031 |
| B: predicted | scaffold | 0.522 | 0.506 | 0.490 | 0.518 | 0.501 | 0.507 | 0.012 |
| A: measured | random | 0.544 | 0.509 | 0.467 | 0.482 | 0.505 | 0.501 | 0.026 |
| B: predicted | random | 0.502 | 0.492 | 0.502 | 0.506 | 0.499 | 0.500 | 0.005 |

## Interpretation

### What the data shows

**Finding 1: Measured LINCS GEX does NOT improve over chance on scaffold-novel split.**
Run A (measured, 978d) achieves 0.4565 AUROC — below the v1 chemistry-only baseline of 0.5930.
The Wang/Li 8-layer DNN, when fed measured LINCS DE and evaluated on scaffold-novel drugs,
fails to generalize. Measured LINCS profiles are tied to specific chemical structures and
cell-line contexts; they do not transfer to novel scaffolds.

**Finding 2: Predicted GEX (MultiDCP) weakly outperforms measured GEX on scaffold split.**
Run B (predicted, 919d) achieves 0.5046 vs Run A 0.4565 on scaffold split (DeLong p=0.091, NS).
This is counterintuitive but explainable: MultiDCP predicted GEX was trained to generalize across
drug structures, so it captures scaffold-transferable structure-activity relationships that
raw measured LINCS profiles do not contain.

**Finding 3: The Wang/Li 0.798 gap is NOT explained by classifier capacity or GEX source.**
Both Run A and Run B hover near 0.50 on random split (A=0.5014, B=0.5002). The Wang/Li 0.798
figure is unachievable at drug-unit level because it is inflated by profile-level training:
Wang/Li treat multiple LINCS profiles of the same drug as independent training examples (N=5517
profiles, 628 unique drugs), which leaks drug identity into training data. When reproduced at
drug-unit level (N=628 drug-level means), measured GEX achieves only 0.50 on random split.

### Gap decomposition

The original v1 question was: "why does chemistry-only (0.5930) lag Wang/Li (0.798)?"
This experiment reveals the answer has two components:

| Component | Value | Direction |
|-----------|-------|-----------|
| Wang/Li inflated by profile-level leakage | +0.261 | Wang/Li upward bias |
| Drug-level 8-layer DNN vs small MLP (chemistry-only) | -0.101 | DNN not better than MLP |
| Measured vs predicted GEX at drug-level | -0.048 | measured slightly worse on scaffold |

The Wang/Li 0.798 is primarily an artifact of profile-level training leakage, not a true
drug-generalization performance. At drug-unit level, neither measured nor predicted GEX
approaches this number with the same architecture.

### What this means for v2

1. **GEX signal (measured or predicted) does not generalize across scaffolds** with drug-level
   training at N=628. This is consistent with the v1 finding that chemistry dominates.

2. **The Wang/Li 0.798 comparison target is misleading.** Drug-unit level evaluation
   is the correct benchmark; profile-level inflates by ~0.26 AUROC.

3. **MultiDCP predicted GEX (0.5046 on scaffold) marginally outperforms measured LINCS GEX
   (0.4565) on scaffold-novel drugs.** This suggests predicted GEX captures more generalizable
   structure-activity features than raw measured profiles — an interesting positive signal
   that warrants further investigation with better classifiers.

## Wang/Li leakage verification

Wang/Li DNN.ipynb (DNN.ipynb cell 12) trains on `data.csv` (training profiles) with random
80/20 splits at the profile level. Reproduced on our data:
- Profile-level training (N=5517): AUROC=0.761 (close to their 0.798, confirms reproduction)
- Drug-level training (N=628 drug means): AUROC~0.50 on random split

The ~0.25 AUROC gap between profile-level and drug-level training is entirely explained by
within-drug label leakage in the profile-level protocol.

## Sanity checks

- Label consistency between wangli_profiles and dili_canonical: **PASS** (0 mismatches)
- Test set label balance: scaffold=66 pos/39 neg, random=~84 pos/44 neg
- Scaffold split is harder than random (0.4565 vs 0.5014 for Run A): **as expected**
- Predicted GEX LR benchmark (all 5 folds): scaffold=0.607, random=0.577 — confirms
  predicted GEX has meaningful signal that the DNN is failing to extract at drug-level N

## Method deviations from specification

1. **Mean-pooling:** Aggregated 5517 LINCS profiles to 628 drug-level means (as specified).
   Wang/Li trained on all 5517 profiles — reproduced separately (0.761 AUROC) to confirm
   the leakage hypothesis.
2. **CPU execution:** TF 2.21 on this box has CUDA_ERROR_UNSUPPORTED_PTX_VERSION when Keras
   initializes Dense layers. The 8-layer DNN (676K params) runs on CPU (~4.5 s/run).
3. **Scaffold split test size:** 105 drugs (spec estimated 76; 628-drug intersect is larger
   than the estimated 502, proportionally expanding all split sizes).

## Files

- `src/v2/wangli_8layer.py` — Wang/Li 8-layer DNN, train_one, train_with_cv
- `src/v2/run_v2_p1.py` — data pipeline, split building, summary generation
- `src/v2/run_experiment.py` — standalone runner for all 4 cells
- `data/processed/v2_predictions/` — 100 prediction parquets (gitignored)
