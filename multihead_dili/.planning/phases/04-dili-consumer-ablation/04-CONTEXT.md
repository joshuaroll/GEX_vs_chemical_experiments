# Phase 4 Context — DILI Consumer 7-Way Pathway Ablation

**Phase:** 04-dili-consumer-ablation
**Generated:** 2026-05-20 (auto-generated, discuss skipped per workflow.skip_discuss=true)
**Source of truth:** ROADMAP.md Phase 4 section + Phase 3 deliverables

---

## Goal

Train 630 DILI classifiers (7 ablation variants × 3 head depths × 2 splits × 5 folds × 3 seeds) on
cached Stage-2 features. Cache test-fold predictions for Phase 5 evaluation. Check halt gate 3.

## Input Data

**Primary input:** `data/processed/dili_features.parquet`

| Property | Value |
|----------|-------|
| Rows | 1118 |
| Total columns | 1694 |
| Feature columns | 1688 |
| NaN in feature cols | 0 (EMBED-04 PASS) |
| Class balance | DILI=1: 685, DILI=0: 433 |

**Feature dimension breakdown (EXACT — deviations from design doc documented here):**
- `feat_dose`: 1 dim — MODEL_DOSE mean E-Hill prediction over 10 LINCS cells
- `feat_gex_0..918`: **919 dims** (NOT 918, NOT 978 as stated in design doc — 919 landmark genes overlap between gene_vector.csv and lincs_train_safe.parquet with header=None)
- `feat_embed_0..767`: 768 dims — frozen MolFormer (ibm-research/MoLFormer-XL-both-10pct)
- **Total feature vector: 1 + 919 + 768 = 1688 dims**

**Metadata columns:** `drug_name`, `pert_id`, `dili_binary`, `dili_severity`, `in_lincs`, `in_pdg`

**Label:** `dili_binary` (0/1 binary DILI classification)

**Split artifact:** `data/processed/dili_split.json`
- `train`: 838 pert_ids
- `val`: 112 pert_ids
- `test`: 168 pert_ids
- `scaffolds_in_test`: 30 Murcko scaffolds (leakage exclusion set)

---

## 7 Ablation Variants

Feature-column slices of the 1688-dim vector (NO separate parquets — slice in-memory):

| Var | Name | Feature columns | Input dim |
|-----|------|-----------------|-----------|
| 1 | embed-only | `feat_embed_*` (768 cols) | 768 |
| 2 | gex-only | `feat_gex_*` (919 cols) | 919 |
| 3 | dose-only | `feat_dose` (1 col) | 1 |
| 4 | embed+gex | `feat_embed_*` + `feat_gex_*` | 1687 |
| 5 | embed+dose | `feat_embed_*` + `feat_dose` | 769 |
| 6 | gex+dose | `feat_gex_*` + `feat_dose` | 920 |
| 7 | all-three | all 1688 feature cols | 1688 |

Note: Phase 4 spec uses embed = MolFormer, gex = MODEL_GEX predictions, dose = MODEL_DOSE E-Hill.
Variant numbering: 1=embed-only (MolFormer/chemistry shortcut), 2=gex-only, 3=dose-only.
HG3 check is on **embed-only (var1)** = MolFormer-only chemistry baseline.

---

## 3 Head Depths

| Name | Architecture | Hyperparameters |
|------|-------------|-----------------|
| linear | input_dim → 1 (sigmoid) | No hidden layers |
| mlp1 | input_dim → 128 → 1 | ReLU activation, dropout 0.3 |
| mlp2 | input_dim → 256 → 64 → 1 | ReLU, BatchNorm, dropout 0.3 |

Hyperparameters LOCKED across all variants (no per-variant tuning):
- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- Epochs: 100 (early stopping patience=10 on val loss)
- Batch size: 64
- Loss: BCEWithLogitsLoss (class-weighted for imbalance)

---

## 2 Splits

### Split 1: scaffold-novel (primary)
- Use pre-built `data/processed/dili_split.json` train/val/test assignment (838/112/168)
- 5 folds: 5-fold CV within the train+val portion; test set is always the same held-out set
- No drug from test leaks into train at any fold

### Split 2: random (comparability baseline)
- Generate 80/10/10 random split (5 folds × 3 seeds)
- Seeded with (split_seed=42 + fold_idx) for reproducibility
- No scaffold stratification

---

## Grid Summary

- 7 variants × 3 heads × 2 splits × 5 folds × 3 seeds = **630 total runs**
- Each run is CPU-feasible (tiny MLP on cached tabular features)
- Runs are embarrassingly parallel — parallelize across CPU cores
- Always leave one GPU free (shared box constraint); GPU not needed for these tiny classifiers

**Deterministic seeding:** Each run seeded with `hash(variant, head, split, fold, seed) % 2**31`.
Both `torch.manual_seed` and `numpy.random.seed` and Python `random.seed` set before each run.

---

## Output Schema

**Per-run prediction parquet** path:
```
data/processed/predictions/var{1..7}_head{linear|mlp1|mlp2}_split{scaffold|random}_fold{0..4}_seed{0..2}.parquet
```

**Per-run parquet columns:**
```
pert_id, drug_name, true_label, predicted_proba, predicted_label,
fold, seed, variant, head_depth, split_type
```

**Aggregate run metrics parquet:**
```
data/processed/P4_runs.parquet
```
One row per run: `variant, head_depth, split_type, fold, seed, auroc, auprc, mcc, balanced_acc`

---

## Halt Gate 3 (HG3)

**Trigger:** embed-only (var1, MolFormer-only) random-split AUROC ≤ 0.55 averaged over seeds/heads

**Check:** After all 630 runs complete (not mid-grid). Compute mean AUROC across all `var1_*_splitrandom_*` prediction files.

**Action if fires:** Write `phases/04-dili-consumer-ablation/HALT_REASON_3.md` and stop. Do NOT proceed to Phase 5 until user reviews. Chemical embedding not capturing DILI signal — whole experimental framing is questionable.

**Threshold:** 0.55 (>= 0.55 = PASS; < 0.55 = HALT)

---

## Phase 3 Deviations to Carry Forward

These deviations from the design doc are load-bearing for Phase 4:

| Deviation | Impact on Phase 4 |
|-----------|-------------------|
| feat_gex = 919-dim (not 978) | var2 input_dim=919, var6=920, var4=1687, var7=1688 |
| 10 cells used (not 9) | No impact on Phase 4 (features already cached) |
| nitroprusside has zeros for feat_dose + feat_gex | One drug in dataset has embed-only signal; should not affect aggregate results |
| multidcp_balanceloss used for MODEL_GEX (sparse_moe) | No impact on Phase 4 |

---

## Required Deliverables

1. `src/stage2/classifiers.py` — linear, mlp1, mlp2 head implementations
2. `src/stage2/train_dili_classifier.py` — grid driver (variant × head × split × fold × seed)
3. `data/processed/predictions/*.parquet` — 630 prediction files
4. `data/processed/P4_runs.parquet` — aggregate metrics (one row per run)
5. `results/tables/P4_ablation_summary.md` — per-variant per-head mean ± std AUROC + HG3 verdict
6. HG3 verdict documented in summary (PASS or HALT with HALT_REASON_3.md)

---

## Implementation Notes

- **No separate parquets per variant** — slice feature columns in-memory per run
- **Variant 1 = embed-only** = MolFormer 768-dim (chemistry shortcut baseline / HG3 subject)
- **Variant 7 = all-three** = full 1688-dim (headline condition)
- **v0.5 classifiers dir is empty** — implement fresh in `src/stage2/classifiers.py`
- **WandB logging:** group `multihead_dili`, project `joshroll/MultiDCP_multihead_dili`
- Log AUROC + AUPRC + MCC + balanced_accuracy per run to WandB and to P4_runs.parquet
