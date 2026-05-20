# Milestone v1.0 Summary — Multi-Head MultiDCP DILI

**Milestone:** v1.0 (Multi-head MultiDCP DILI baseline)
**Completed:** 2026-05-20
**Phases:** 0–6 (Phase 0 pre-GSD; Phases 1–6 GSD-managed)
**Total classifier runs:** 630 (7 variants × 3 heads × 2 splits × 5 folds × 3 seeds)

---

## 1. What We Built

We implemented a two-stage DILI prediction pipeline with three independent encoder pathways.
Stage 1 trains MODEL_DOSE (MultiDCP-AE on 37 unique leakage-filtered E-Hill drugs, predicting
E-Hill dose-response parameters) and MODEL_GEX (MultiDCP-AE on LINCS PDG-filtered data,
predicting differential gene expression across 9 LINCS cell lines). Stage 2 concatenates the
mean-pooled dose-response prediction (1-dim), the mean-pooled DE signature (919-dim), and a
frozen MolFormer chemistry embedding (768-dim) into a 1,688-dim feature vector per drug, then
trains three classifier head depths (linear, MLP-1, MLP-2) across 7 pathway ablation variants,
tested on scaffold-novel (Murcko scaffold split) and random DILIst splits, with 5-fold CV and
3 random seeds. Leakage discipline is enforced throughout: test scaffolds from DILIst are
excluded from Stage-1 training for both models before training begins.

---

## 2. Core Finding

**On scaffold-novel DILIst drugs, a frozen MolFormer chemistry encoder (0.5930 AUROC, 95% CI
[0.5325, 0.7019]) matches a three-pathway model combining predicted dose-response, predicted
gene-expression, and chemistry (0.5843 AUROC, 95% CI [0.5120, 0.6862]; ΔAUROC = -0.0087,
DeLong p = 0.3066, two-sided). At this experimental scale, predicted-GEX and predicted-dose
pathways add no measurable DILI signal beyond what MolFormer chemistry alone captures.**

Across all 21 evaluation cells (7 variants × 3 heads) on the scaffold-novel split:

| Variant | Mean AUROC (across heads) | Best AUROC |
|---------|--------------------------|------------|
| var1: embed-only (MolFormer 768d) | 0.5875 | 0.5930 (mlp2) |
| var2: gex-only (MODEL_GEX 919d) | 0.5198 | 0.5333 (mlp2) |
| var3: dose-only (MODEL_DOSE 1d) | 0.4796 | 0.5172 (mlp1) |
| var4: embed+gex (1687d) | 0.5797 | 0.5819 (linear) |
| var5: embed+dose (769d) | 0.5820 | 0.5855 (mlp1) |
| var6: gex+dose (920d) | 0.5186 | 0.5220 (mlp1) |
| var7: all-three (1688d) | 0.5810 | 0.5843 (mlp2) |

Chemistry (embed-only, var1) is the strongest single pathway. Adding predicted GEX and/or
predicted dose pathways does not improve AUROC — all-three (var7) is marginally worse than
embed-only (var1) by 0.0087 AUROC, a difference that is not statistically significant (p=0.31).

On the easier random split, the pattern holds: embed-only remains the strongest pathway
(best AUROC = 0.6645, var1 mlp2) with all-three close behind (0.6453, var7 mlp1).

Full DeLong paired test results (var7 vs var1, scaffold split):
- linear head: ΔAUROC = -0.0063, z = -0.603, p = 0.546 (NS)
- mlp1 head: ΔAUROC = -0.0146, z = -1.462, p = 0.144 (NS)
- mlp2 head: ΔAUROC = -0.0195, z = -1.022, p = 0.307 (NS)

None reach p < 0.05; the finding is a clean non-significant negative.

---

## 3. Halt Gate Trajectory

| Gate | Phase | Trigger | Outcome |
|------|-------|---------|---------|
| HG1 | 1 | MODEL_DOSE dev RMSE < predict-mean baseline | **PASS** — dev RMSE=19.455 vs baseline=33.641 (42% improvement) |
| HG2 | 2 | MODEL_GEX dev Pearson ≥ 0.2 across cells | **PASS** — mean Pearson=0.3568 across 9 LINCS cells |
| HG3 | 4 | embed-only (var1) random-split AUROC > 0.55 | **PASS** — var1 random AUROC=0.6536 >> 0.55 |
| HG4 | 5 | all-three (var7) AUROC ≥ best single + 0.01 on scaffold split | **REFRAME** — var7=0.5843 vs var1=0.5930; ΔAUROC=-0.0087 (threshold not met) |

HG4 did not block Phase 6. The design doc anticipated this outcome as a publishable negative
finding. See `.planning/phases/05-evaluation/HALT_REASON_4.md` for full analysis and framing.

---

## 4. Surprising Findings

1. **Dose pathway near-chance on scaffold-novel (var3 mean AUROC = 0.4796).**
   MODEL_DOSE trains on only 37 unique DILI drugs from E-Hill. This corpus is far too sparse
   to generalize to scaffold-novel test drugs. Dose-response prediction requires orders of
   magnitude more compounds to learn scaffold-transferable structure-activity relationships.

2. **GEX pathway also fails to lift above chemistry (var2 mean AUROC = 0.5198).**
   MODEL_GEX achieves a mean Pearson of 0.358, sufficient to pass HG2, but not sufficient
   to add DILI-relevant signal above what MolFormer captures from SMILES alone. The 9-cell
   mean-pool may dilute per-cell signal; HA1E (the most DILI-relevant cell line) is buried
   in the average.

3. **Feature dimension mismatch vs design doc.**
   Design doc specified 978 GEX features (all LINCS landmark genes). Reality: 919 overlap
   between gene_vector.csv and lincs_train_safe.parquet (header=None issue). Total feature
   dim is 1,688 (not 1,747 as designed): 1 + 919 + 768 = 1,688.

4. **Scaffold-novel gap is large (embedding: 0.5930 scaffold vs 0.6645 random).**
   The ΔAUROC scaffold-to-random gap for embed-only is 0.071 — confirming the scaffold-novel
   test is substantially harder and that random-split numbers overstate generalizability.

5. **SMILES failure for nitroprusside (iron coordination compound).**
   MolFormer tokenizer fails on nitroprusside (atom degree 6, unsupported). Zero vector used
   as fallback. This drug's classification relies purely on dose/GEX pathways for its embed column.

6. **Cell count discrepancy: 9 cells inferred but 10 LINCS cell lines registered.**
   During Stage-2 caching, 9 cells produced valid output; 1 was silently excluded by
   the MultiDCP gene-vector lookup. Mean-pooling proceeds over 9 cells.

---

## 5. Methodological Caveats

1. **E-Hill corpus is small (37 unique DILI drugs, 37 dose-response curves).**
   MODEL_DOSE is severely underpowered. The entire corpus has less training signal than a
   single typical drug discovery dose-response campaign. v2 should use a larger public corpus
   (e.g., CTD2, NCI-60, or curated DILI-linked dose data).

2. **Mean-pool across LINCS cells dilutes per-cell DILI signal.**
   HA1E (human hepatocyte) is the most DILI-informative cell line. Averaging across 9 cells
   including MCF7, A375, etc. dilutes the hepatocyte-specific transcriptional response.

3. **919-gene MODEL_GEX output is a PDG inheritance, not a deliberate choice.**
   MultiDCP's `gene_vector.csv` has 977 rows; the actual training parquet
   (`lincs_train_safe.parquet`, derived from `pdg_brddrugfiltered.pkl`) has 10,716
   gene columns of which only 919 overlap with `gene_vector.csv`. The 58-gene gap is
   PDGrapher's upstream filter — not a deliberate scientific decision by this project.
   We adopted PDG-filtered LINCS because the existing MultiDCP training scripts
   (`ehill_multidcp_pretrain.py`, the AE balance-loss trainer) were already wired to
   it (Q9c in the v0.4 decision log). The impact on MODEL_GEX quality is unknown.
   v2 should drop the PDG dependency entirely (see §6 item 1).

4. **No calibration tuning.**
   ECE values range 0.08–0.12 across cells. For a DILI risk tool, calibration matters
   clinically. Platt scaling or temperature calibration was not applied in v1.

5. **Stage-2 features are concatenation, not learned fusion.**
   A simple concat-MLP treats all feature dimensions equally. Attention-based or gated
   fusion could learn to down-weight noisy predicted pathways per drug.

6. **5-fold CV over ~200 test drugs per scaffold fold.**
   The effective test set per fold is ~40 drugs. CIs are wide (order 0.15–0.20) because
   the dataset is small. Bootstrap CIs here reflect drug-level variability, not epistemic
   uncertainty from the upstream models.

---

## 6. v2 Candidate Directions

**Revised 2026-05-20** after reproducing Wang/Li 2020 (PMC7728858) on dili_downstream
branch (commit `b41f213`). Key finding: **Wang/Li hit AUROC 0.79 on a 1,091-profile
test set that is 97.5% non-hepatocyte** (HEPG2 + PHH = 27 / 1,091 = 2.5%). Their DILI
signal comes from measured cancer-cell L1000 profiles + an 8-layer DNN — not from
hepatocyte data. This shifts the v2 priority calculus: the real gap between our v1
(0.59 AUROC chemistry-only) and the published 0.79 is most likely **predicted-vs-measured
GEX** and **classifier capacity** (small head vs 8-layer DNN), not cell-type biology.

In rough priority order:

1. **Replicate Wang/Li's 8-layer DNN architecture on measured LINCS, then swap measured → MultiDCP-predicted.**
   Two runs, same architecture (978 → 512 → 256 → 128 → 64 → 32 → 16 → 8 → 1, ELU,
   sigmoid, BCE):
   - Run A: 8-layer DNN on **measured** LINCS DE for DILIst drugs (under our scaffold-novel
     split + leakage discipline; expected AUROC well above 0.59 if measured GEX has signal).
   - Run B: 8-layer DNN on **MultiDCP-predicted** GEX for same drugs (under same split).
   The A vs B gap isolates the "predicted-vs-measured" question cleanly; A vs Wang/Li's
   0.79 isolates the "scaffold-novel split discipline" question. Together they decompose
   the v1 negative finding into its two main causes.

2. **Drop PDG-filtered LINCS; switch to raw MODZ-normalized LINCS for both training and inference.**
   The Wang/Li reproduction confirmed our local LINCS is Bayesian-COMPZ-shrunk, not standard
   MODZ. Wang/Li's actual checkpoint cannot be loaded against Bayesian-shrunk data (gene-specific
   compression up to 15×; no linear rescaling fixes it). Switching to raw MODZ LINCS would
   (a) recover the full 978 landmark genes (was 919 post-PDG), (b) enable direct use of Wang/Li's
   `optimized_model.h5` without retraining, (c) remove the PDGrapher dependency entirely.
   Reproduction-by-retraining already confirms our pipeline correctness (0.7907 vs published
   0.798); switching to MODZ would let us compare published-vs-our-checkpoint at the parameter
   level, not just the metric level.

3. **Attention-based or gated pathway combiner.**
   Replace the concat-MLP with a self-attention combiner (a la FusedTransformer or simple
   gating): let the model learn to up-weight chemistry vs GEX vs dose per drug. May recover
   signal from the GEX pathway that concat suppresses. Particularly relevant given priority 1's
   result will likely show measured GEX has more signal than concat-MLP can extract.

4. **Use MultiDCP's joint embedding (128-d) instead of the 919-d DE prediction as the GEX feature.**
   Modify `cache_dili_features.py` to hook into `MultiDCP.forward()` and grab the
   `[batch, num_gene, hid_dim=128]` joint representation before the final DE decoder,
   then mean-pool over genes → 128-d per (drug, cell). Total feature dim drops from
   1,688 → 897, eliminates the lossy DE reconstruction step, captures the model's joint
   chem+cell+dose representation. Standard "use the bottleneck, not the reconstruction"
   move from self-supervised representation learning.

5. **Expand E-Hill corpus with additional public dose-response data (CTD2, NCI-60).**
   Replace the 37-drug E-Hill training set with a larger corpus. Target ≥ 500 unique compounds
   with dose-response curves. This is the most likely fix for the near-chance dose pathway.

6. **Encoder ablation: ChemBERTa / GIN / UniMol vs MolFormer.**
   var1 (MolFormer) is the current chemistry-only baseline. Test ChemBERTa (768d), GIN (300d),
   UniMol (512d) as drop-in replacements. MolFormer dominance may not hold for all encoders.

7. **(DEPRIORITIZED) Hepatocyte-specific GEX (HepG2-only, PHH, TG-GATEs).**
   Was a strong v2 candidate pre-Wang/Li-reproduction. The reproduction showed Wang/Li's
   0.798 came from 97.5% non-hepatocyte profiles, so cell-type filtering to hepatocytes is
   unlikely to be the lever it appeared to be. **Still worth a controlled run** (HA1E-only
   vs 9-cell mean-pool with everything else held constant) but no longer the primary v2 lever.

8. **Expand to non-liver organ toxicity (cardiac, renal).**
   The DILIst scaffold-split framework generalizes. Swap in a cardiotoxicity or nephrotoxicity
   dataset.

9. **Semi-supervised pre-training on unlabeled LINCS drugs.**
   Use contrastive/triplet pre-training on the full LINCS corpus (>30K compound-cell pairs)
   to warm-start the Stage-2 combiner before DILI fine-tuning.

---

## Appendix: Key Commits

| Phase | Commit | Description |
|-------|--------|-------------|
| 0 | e281e9f | Data foundation, leakage filter, scaffold split |
| 1 | (P1 commits) | MODEL_DOSE training (HG1 PASS) |
| 2 | 9216d06, c567589, 92cc932 | MODEL_GEX training (HG2 PASS) |
| 3 | 6e0bbdb, 0b5fe40 | MolFormer + Stage-2 feature caching |
| 4 | 0d210aa + (630 runs) | 7-way ablation classifier grid (HG3 PASS) |
| 5 | ef4296b | DeLong + bootstrap eval, HG4 REFRAME |
| 6 | (this commit) | Milestone summary + state close |

---

*Milestone v1.0 closed 2026-05-20.*
*Next: v2.0 — replicate Wang/Li's 8-layer DNN on (measured GEX, predicted GEX) under scaffold-novel split (§6 #1); then raw MODZ LINCS (§6 #2). Hepatocyte filtering deprioritized after Wang/Li reproduction showed their 0.798 holds with 2.5% hepatocyte data (§6 #7).*

---

## Addendum (2026-05-20): Wang/Li reproduction on dili_downstream sibling branch

The v0.5 `dili_downstream/` branch reproduced Wang/Li 2020 (PMC7728858) on
2026-05-20 (commit `b41f213`). Key results that inform v2 framing:

- **AUROC 0.7907 (10-seed ensemble) vs published 0.798** — PASS (Δ = -0.007, within ±0.02)
- **Reproduction is via retraining their architecture on our local Bayesian-COMPZ data**, NOT
  via running their `optimized_model.h5` checkpoint. Their checkpoint loads cleanly but
  produces AUROC 0.5136 on our data because of a Bayesian-COMPZ vs standard-MODZ
  normalization mismatch (gene-specific compression up to 15×; no linear rescaling fixes it).
- **Test split cell distribution (n=1,091):** MCF7 228 (21%), PC3 176 (16%), VCAP 121 (11%),
  HEPG2 22 (2.0%), PHH 5 (0.5%), other cancer lines ~539 (49%). **Total hepatocyte = 27 (2.5%).**
- **Implication:** Wang/Li's DILI signal in L1000 comes from measured drug-on-cancer-cell
  transcriptional response + their 8-layer DNN — not from hepatocyte data. Our v1's
  cancer-cell-derived predicted-GEX pathway should have had access to similar biological signal;
  the v1 negative finding likely traces to (a) predicted vs measured GEX noise and/or
  (b) small classifier head vs 8-layer DNN capacity. This is exactly the decomposition v2 §6 #1
  is designed to test.

See `dili_downstream/results/tables/P2_wangli_reproduction.md` for full details.
