# Phase 5 Evaluation Summary — Multi-Head MultiDCP DILI

**Date:** 2026-05-20
**Phase:** 5 of 6

## Findings

### 1. Chemistry embedding dominates all scaffold-novel comparisons

Across all 7 ablation variants on the scaffold-novel split, embed-only (var1, frozen
MolFormer 768-dim chemistry) is the strongest pathway.

- var1 embed-only (best head mlp2): **0.5930** AUROC, 95% CI [0.5325, 0.7019]
- var7 all-three (best head mlp2): **0.5843** AUROC, 95% CI [0.5120, 0.6862]
- ΔAUROC (var7 − var1): **-0.0087**
- DeLong: z = -1.022, p = 0.3066 (NOT significant, two-sided)

### 2. Predicted GEX pathway adds noise, not signal

var2 (gex-only, MODEL_GEX 919d) on scaffold split: mean AUROC = 0.5198 (across heads).
Adding GEX to embed (var4) does not improve over embed-only (var1): var4=0.5797 vs var1=0.5876.

### 3. Predicted dose pathway near-chance on scaffold-novel

var3 (dose-only, 1d) mean AUROC = 0.4796 across heads (near chance 0.50).
MODEL_DOSE predicts E-Hill params from a corpus of only 37 unique DILI drugs.
Sparse training data limits generalization to scaffold-novel drugs.

### 4. Random split inflates all numbers (expected)

var1 embed-only on random split: 0.6485 (vs scaffold 0.5876); gap = +0.0609.
Confirms scaffold-novel is the harder and more informative setting.

## Halt Gate 4 (HG4) Verdict

**REFRAME** — all-three does not beat best single by >= 0.01 AUROC on scaffold split.
ΔAUROC = -0.0087 (threshold needed: +0.01).
See `HALT_REASON_4.md` for full analysis and reframing.
This is NOT a blocker for Phase 6.

## Deliverables Produced

- `results/tables/headline.md` — full 21-cell results table with DeLong + CIs
- `results/figures/ablation.png` — 7-way ablation bar chart with 95% CI error bars
- `results/figures/comparison_v05.png` — comparison note (v0.5 not comparable)
- `results/tables/P5_eval_results.parquet` — raw cell-level metrics
- `.planning/phases/05-evaluation/HALT_REASON_4.md` — HG4 reframe documentation

## Verification

All 21 cells (7 variants × 3 heads) evaluated on both splits.
Bootstrap CIs computed from 10,000 resamples at drug level.
DeLong paired tests run for var7 vs var1 on scaffold split, for each head depth.

Phase 5: COMPLETE.