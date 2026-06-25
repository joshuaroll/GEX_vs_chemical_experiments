# P2: Wang/Li 2020 DNN reproduction + drug-disjoint contrast

Reproduce the Wang/Li 2020 DILI benchmark (published AUROC ~0.798) with their exact
DNN architecture, then run the same model under a drug-disjoint split to expose how
much of the headline number is drug-identity leakage.

## Headline result

| Split | AUROC | vs published 0.798 |
|---|---|---|
| **Published (leaky) profile split — 10-seed ensemble** | **0.7820** | -0.0160 (within [0.78, 0.82]) PASS |
| Published (leaky) — single-seed mean +/- std | 0.7688 +/- 0.0067 | -0.029 |
| **Drug-disjoint (StratifiedGroupKFold, grouped by compound)** | **0.5927 +/- 0.0354** | -0.205 |

Same architecture, same training recipe, same 10-seed treatment for both splits.

**Contrast:** moving from the published profile-level split to a compound-disjoint
split drops AUROC by **~0.19** (0.782 ensemble -> 0.593 mean), to barely above chance.
About 0.19 of the 0.798 headline is attributable to drug-identity leakage: in the
published split a compound's profiles appear in both Training and Test, so the model
recognizes drugs it has already seen rather than generalizing DILI signal to new
chemistry.

## 1. Published (leaky) split — like-for-like reproduction

Train on `usage == Training` (4,426 profiles), evaluate on `usage == Test`
(1,091 profiles). This is Wang/Li's published profile-level split. It is leaky
because the split is by profile, not by drug: the same compound can contribute
profiles to both Training and Test (e.g. across cell lines, doses, time points).

Per-seed AUROC (10 independent seeds, seeds 1000-1009):

| seed | AUROC |
|---|---|
| 0 | 0.7609 |
| 1 | 0.7784 |
| 2 | 0.7722 |
| 3 | 0.7607 |
| 4 | 0.7640 |
| 5 | 0.7601 |
| 6 | 0.7758 |
| 7 | 0.7710 |
| 8 | 0.7684 |
| 9 | 0.7770 |

- Single-seed mean +/- std: **0.7688 +/- 0.0067**
- 10-seed probability-ensemble AUROC (mean of per-seed test probabilities): **0.7820**
- Published target: 0.798. Delta (ensemble): **-0.0160**. Within the [0.78, 0.82]
  acceptance band -> PASS.

## 2. Drug-disjoint split — leakage isolated

Same architecture and recipe, applied to the full 5,517 profiles with
`StratifiedGroupKFold(n_splits=5, shuffle=True)` grouped by `compound_name`
(lowercased, 628 unique compounds). No compound appears in both train and test
within a fold. Out-of-fold predictions are collected across the 5 folds and scored
at the profile level. Repeated for 10 seeds (2000-2009).

Per-seed out-of-fold AUROC: 0.6401, 0.6015, 0.5404, 0.6271, 0.5963, 0.5912,
0.5719, 0.5250, 0.6095, 0.6243.

- Mean +/- std: **0.5927 +/- 0.0354**
- This is the same DNN, the same 978-d measured DE features, the same training
  recipe. The only change is the split criterion (drug-disjoint instead of
  profile-level). AUROC collapses from ~0.78 to ~0.59.

## Exact configuration used

**Architecture (matches DNN.ipynb / dili_downstream P2_wangli_reproduction.md):**
- Input: 978 landmark-gene Bayesian COMPZ differential-expression z-scores, used
  as-is (no additional scaling; the reference used raw input).
- Layers: 978 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 1.
- ELU on all hidden layers; sigmoid on output (implemented as a linear output head
  with `BCEWithLogitsLoss`, i.e. sigmoid folded into the loss).

**Optimizer / loss / class balance:**
- Adam (default LR), loss = binary cross-entropy.
- Balanced class weights via `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`
  computed on each training pool.
- Batch size 256.

**Training schedule / model selection:**
- Up to 100 epochs, EarlyStopping with patience 5 on an internal validation split
  (10% of the training pool, seed-shuffled), restoring the best-validation epoch's
  weights.
- Best epoch / early-stop monitor = **validation AUROC** (`--monitor auroc`, the
  default). Wang/Li's DNN.ipynb selects on a custom `val_monitor_f` metric, not on
  plain val_loss. Monitoring val AUROC mirrors that intent and reproduces 0.798 to
  within the band. A literal `EarlyStopping(val_loss)` recipe (`--monitor loss`)
  gives a lower ensemble (~0.769, single-seed mean 0.750 +/- 0.014) because the
  model overfits within 1-2 epochs and val_loss selects an earlier, weaker epoch;
  it does not change the qualitative leaky-vs-disjoint contrast.

**Splits:**
- Leaky: Wang/Li's published `usage` column (Training=4,426, Test=1,091).
- Drug-disjoint: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2000+s)`,
  groups = `compound_name.str.lower()` (628 groups), out-of-fold profile-level scoring.

**Seeds:** 10 per split (mean +/- std reported; leaky also reports the probability
ensemble). **GPU:** auto-picked least-used among {2, 3} (GPU 1 avoided per the
shared-box policy), `CUDA_VISIBLE_DEVICES` set before importing torch.

## Data provenance

- `dili_downstream/data/processed/wangli_measured_de.npy` — measured LINCS L1000
  differential expression, shape (5517, 978) float32. Row i aligns to row i of the
  profiles CSV.
- `dili_downstream/data/processed/wangli_profiles.csv` — 5,517 rows; `dili_binary`
  is the label, `usage` is Wang/Li's published Training/Test split, `compound_name`
  the grouping key (628 unique compounds). Class balance: 3,346 positive (DILI=1),
  2,171 negative.
- Both files were built by `dili_downstream/scripts/build_wangli_random_only.py`
  from Wang/Li's `6000_transcriptomic_profiles_id.xlsx`
  (github.com/TingLi2016/L1000_DILI) joined to the local Bayesian COMPZ L1000 h5
  (`Bayesian_GSE92742_Level5_COMPZ_n361481x978.h5`). 442 of 6,000 profiles were
  absent from the local h5 (7.37% miss) and 4 dropped on SMILES resolution, leaving
  5,517. Real data only; no mocking or synthetic labels.

## Caveats

1. **Reproduction is by retraining, not checkpoint inference.** Direct inference with
   Wang/Li's published `optimized_model.h5` yields AUROC ~0.51 (near chance). Cause:
   the published checkpoint was trained on standard COMPZ **MODZ** z-scores
   (`GSE92742_..._Level5_COMPZ.MODZ`), but our local features are **Bayesian-shrinkage**
   COMPZ z-scores. Bayesian shrinkage attenuates drug-response signal non-uniformly
   per gene (no single rescaling recovers MODZ values; StandardScaler/MinMaxScaler
   both stay ~0.51), so the trained weights do not transfer. See
   `dili_downstream/scripts/reproduce_wangli_auroc.py` and
   `dili_downstream/results/tables/P2_wangli_reproduction.md` for the diagnosis.
   Reproduction therefore retrains Wang/Li's exact architecture on the Bayesian DE,
   which recovers the headline AUROC.

2. **Delta -0.016 from 0.798.** The ensemble lands at 0.782, inside the [0.78, 0.82]
   band but below the published 0.798. The residual gap is consistent with the
   Bayesian-vs-MODZ feature difference (caveat 1), the 7.4% h5 miss vs Wang/Li's full
   6,000 profiles, and our reconstructed val-monitor / early-stopping recipe rather
   than their exact `val_monitor_f` and checkpoint-selection code.

3. **Drug-disjoint AUROC is profile-level out-of-fold**, not a held-out single test
   set, so it pools 5 folds before scoring. The std (0.035) is over the 10 seeds
   (each a full re-fold + retrain). The qualitative finding (collapse toward chance
   under compound disjointness) is stable across seeds.

## Files

- Script: `spatial_tests/scripts/reproduce_wangli_dnn_retrain.py`
- Raw results: `spatial_tests/results/tables/P2_wangli_check_results.json`
- This report: `spatial_tests/results/tables/P2_wangli_reproduction_check.md`
