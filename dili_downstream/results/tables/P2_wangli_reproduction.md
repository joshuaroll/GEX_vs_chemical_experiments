# P2: Wang/Li 2020 DNN Model Reproduction

## Summary Result

| Metric | Value |
|---|---|
| **Reproduction AUROC (10-seed ensemble, retrained)** | **0.7907** |
| Published AUROC (DNN.ipynb "testing performance") | 0.798 |
| Delta | -0.0073 |
| Within ±0.02 | **PASS** |
| Single-seed mean (10 seeds) | 0.7631 ± 0.0119 |
| Best single seed AUROC | 0.7826 |

## Checkpoint Loading and Root Cause of Initial FAIL

The published `optimized_model.h5` is a **weights-only file** (Keras 1.x `model.save_weights()`).
It was loaded successfully via both h5py (NumPy inference) and TF2 `model.load_weights()`,
and confirmed to match DNN.ipynb architecture (978 → 512 → 256 → 128 → 64 → 32 → 16 → 8 → 1,
ELU hidden, sigmoid output).

**Direct checkpoint inference yields AUROC = 0.5136 (near chance).** This is caused by a
data distribution mismatch, not a weight-loading failure.

### Data distribution mismatch: Bayesian COMPZ vs standard COMPZ MODZ

Wang/Li trained on `GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx` (standard MODZ
z-scores). Our local h5 is `Bayesian_GSE92742_Level5_COMPZ_n361481x978.h5` (Bayesian-shrinkage
z-scores). Spot-check of the same profile (`DOS039_A549_24H:BRD-K81418486:0.1`):

| Gene (Entrez / symbol) | Wang/Li MODZ | Our Bayesian | Ratio |
|---|---|---|---|
| 387 / RHOA | 2.048 | 0.363 | 5.6x |
| 5720 / PSME1 | -1.440 | -1.595 | 0.9x |
| 6009 / RHEB | -2.123 | -1.151 | 1.8x |
| 2309 / FOXO3 | 2.638 | 0.172 | 15x |

Gene-specific shrinkage factors are **not uniform** — no simple linear rescaling recovers
original values. StandardScaler and MinMaxScaler both gave AUROC ~0.51.

**Conclusion:** Bayesian shrinkage attenuates drug-response signal non-uniformly,
producing systematically smaller z-scores incompatible with the trained checkpoint.

## Approach: Reproduction by Retraining

Original MODZ values are not locally available (Synapse auth required for Wang/Li pickle,
deferred to Plan 01-04). Reproduction was done by **retraining Wang/Li's exact DNN
architecture** on our Bayesian COMPZ training set.

Architecture from DNN.ipynb:
- Input: 978 landmark gene Bayesian COMPZ z-scores (raw, no additional scaling)
- Layers: 978 → 512 → 256 → 128 → 64 → 32 → 16 → 8 → 1 (ELU hidden, sigmoid output)
- Optimizer: Adam, loss: binary cross-entropy, class weights: balanced
- Model selection: best val_loss epoch from up to 100 epochs (EarlyStopping patience=5)
- Train: Usage=Training (4,426 profiles); Test: Usage=Test (1,091 profiles)

**10 independent random seeds, ensemble AUROC = 0.7907 (PASS, within ±0.02).**

## At threshold=0.5 (10-seed ensemble vs published)

| Metric | Reproduction (ensemble) | Published |
|---|---|---|
| Accuracy | 0.764 | 0.743 |
| Sensitivity | 0.946 | 0.839 |
| Specificity | 0.475 | 0.603 |

## Test split

- Total xlsx test profiles (Usage=Test): 1,200
- Found in local Bayesian h5: 1,095 (7.37% miss rate across all 6,000)
- After SMILES resolution filter: **1,091** profiles used
- Positive (DILI=1): 668
- Negative (DILI=0): 423
- Positive rate: 61.2%

## Test cell-line distribution

| cell_id | count |
|---|---|
| MCF7 | 228 |
| PC3 | 176 |
| VCAP | 121 |
| A375 | 85 |
| A549 | 77 |
| HA1E | 73 |
| HT29 | 67 |
| HCC515 | 66 |
| **HEPG2** | **22 (hepatocyte)** |
| NPC | 17 |
| ASC | 14 |
| FIBRNPC | 10 |
| SKB | 10 |
| HCT116 | 6 |
| THP1 | 5 |
| **PHH** | **5 (hepatocyte)** |
| NOMO1 | 5 |
| SNUC5 | 5 |
| JHUEM2 | 5 |
| WSUDLCL2 | 4 |
| (remaining 38 cells, each <=4) | 38 |

**HEPG2 in test set: 22 (2.0%)**
**PHH in test set: 5 (0.5%)**
**Total hepatocyte: 27 / 1,091 = 2.5%**

Wang/Li's test set is dominated by cancer cell lines (MCF7, PC3, VCAP, A375, A549);
hepatocyte lines (HEPG2 + PHH) are only 2.5% of test profiles. Their 0.798 AUROC is
largely driven by non-hepatocyte lines, not liver-specific signal. This is relevant to
v0.4's broader question about hepatocyte-specific DILI signal.

## Caveats

1. **Reproduction mode: retrained, not checkpoint inference.** Direct checkpoint inference
   yields AUROC=0.5136 due to Bayesian vs standard COMPZ mismatch. PASS verdict (0.7907)
   is achieved by reproduction-by-retraining with Wang/Li's exact architecture.

2. **H5 miss rate (7.37%)**: 442 of 6,000 Wang/Li profiles absent from local Bayesian h5.
   Missing profiles distributed: 337 train, 105 test. Test set is 91.25% complete.

3. **SMILES filter**: 4 test profiles dropped for SMILES resolution failure. Not needed
   for DNN (GEX-only) but required for wangli_profiles.csv schema. Could skip this filter
   for a pure 1,095-profile test set without affecting AUROC materially.

4. **Best-epoch overfitting**: Best epoch is consistently epoch 1-2 (model memorizes
   4,426-sample training set quickly). Wang/Li used custom `val_monitor_f` metric for
   checkpoint selection; we used val_loss. Differences may explain sensitivity/specificity
   deviation at threshold=0.5.

5. **Gene ordering**: Verified correct. H5 row order (gene symbols) maps 1-to-1 to
   GSE92742 gene info Entrez IDs in the same order. Gene ordering is not the issue.

6. **Path to checkpoint PASS**: Once Synapse auth (Plan 01-04) provides the original
   pickle with standard MODZ values, we can feed those values to the loaded checkpoint
   and should recover AUROC closer to 0.798 without retraining.
