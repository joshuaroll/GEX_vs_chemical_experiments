# GSD: DILI Downstream — Liver Toxicity Prediction from Predicted Signatures

**Author:** Joshua Rollins (CUNY Graduate Center, Lei Xie lab)
**Status:** Draft v0.1 — pre-implementation
**Last updated:** 2026-05-05
**Audience:** Claude Code (primary), Joshua, collaborators
**Parent project:** MultiDCP-CheMoE (see `MULTIDCP_CHEMOE_PLAN.md`)
**Scope of this document:** Liver (DILI) arm only. Kidney, heart, and brain arms are out of scope and will be specified in their own GSD documents once liver is green.

---

## 0. Conventions for Claude Code

Read this entire document before scaffolding anything. Sections are dependency-ordered. Each milestone is atomic, has explicit deliverables, and has a verification step that must pass before the next milestone starts.

**Hard rules (non-negotiable):**

1. **Real data only.** No mocking, no stubbing, no synthetic labels, no placeholder DataLoaders. If a data path is unavailable, stop and ask Joshua.
2. **Differential expression rule.** Any GEX-derived feature is computed on differential expression (`treated − diseased`), with top-k DEGs selected by `|true_treated − diseased|`. The canonical reference is `train_bl_pdg_de.py`. Raw expression as a feature or a metric is forbidden.
3. **Compound-aware splits.** No drug appears in both train and test. Random splits are forbidden. Use scaffold split (RDKit Murcko scaffolds) as the default; report cluster-based splits as a robustness check.
4. **CUDA hygiene.** Set `CUDA_VISIBLE_DEVICES` via argparse before `import torch`.
5. **Seeds.** Minimum three seeds per experimental cell, aggregated as mean ± std.
6. **wandb.** All runs log to project `MultiDCP_AE_DE` under entity `joshroll`. Use run group `dili_downstream`.
7. **Atomic git commits.** One milestone per commit. Commit message format: `dili_downstream: M{n} — {short description}`.
8. **Context isolation.** Each milestone's prompt to Claude Code spawns a fresh context. This document is the source of truth that survives across those contexts.

**File layout:**

```
experiments/dili_downstream/
├── CLAUDE_DILI_DOWNSTREAM.md          # this file (committed alongside code)
├── data/
│   ├── raw/                            # untouched downloads (DILIst, DILIrank, SMILES)
│   ├── processed/                      # harmonized splits, signature caches
│   └── splits/                         # scaffold splits, cluster splits, manifest JSON
├── src/
│   ├── data_loaders.py
│   ├── signatures/
│   │   ├── multidcp_signature.py       # call frozen MultiDCP, cache predicted GEX
│   │   ├── chemoe_signature.py         # call frozen MultiDCP-CheMoE, cache predicted GEX
│   │   └── lincs_signature.py          # measured L1000 (upper bound)
│   ├── encoders/
│   │   └── chemical_encoder.py         # Yoojean's encoder wrapper
│   ├── classifiers/
│   │   ├── mlp_classifier.py
│   │   └── concat_classifier.py        # signature ⊕ structure → DILI
│   ├── train.py
│   ├── eval.py
│   └── ablations/
│       ├── viability_mask.py           # Szalai et al. confound control
│       └── permutation_diagnostic.py
├── configs/
│   ├── baseline_structure_only.yaml
│   ├── multidcp_signature.yaml
│   ├── chemoe_signature.yaml
│   ├── lincs_upper_bound.yaml
│   └── sweep_classifier_hp.yaml
├── results/
│   ├── tables/
│   └── figures/
└── tests/
    └── (sanity tests on data loaders, never on model outputs)
```

---

## 1. Big picture (one paragraph)

We are testing whether MultiDCP-predicted gene expression signatures, combined with Yoojean's chemical representation, predict drug-induced liver injury (DILI) at parity with or better than (a) chemical structure alone and (b) measured LINCS L1000 signatures. The headline claim we are aiming for is that predicted signatures from MultiDCP transfer the L1000-DILI signal to compounds that were never measured in CMap, and that adding the CheMoE mixture-of-experts fusion improves OOD generalization on scaffold-novel drugs. This document covers the liver arm only. Closest published precedent is Wang et al. 2020 (LINCS L1000 + DILIst, AUC 0.802). Closest competitor is ChemBioHepatox (multimodal structure + biology, AUC 0.92). Our differentiator is the predicted-signature angle, which extends coverage beyond CMap and lets us evaluate cell-line generalization. The cell viability confound (Szalai et al. 2018) is the primary methodological risk; we control for it explicitly via a viability-correlated gene mask and a viability-only baseline.

---

## 2. The four signature conditions we compare

Every classifier in this study uses exactly one of the following input regimes for the biological half of the model. The chemical half (Yoojean's encoder) is held constant across all conditions. This is the central experimental axis.

| Condition | Signature source | Purpose |
|---|---|---|
| **A. Structure-only** | none | Lower bound. Tests whether biology adds anything. |
| **B. MultiDCP (baseline)** | predicted GEX from frozen MultiDCP | Baseline biology condition. |
| **C. MultiDCP-CheMoE** | predicted GEX from frozen MultiDCP-CheMoE (StructuredMoE variant) | Test condition. Does MoE fusion help downstream? |
| **D. LINCS L1000 (upper bound)** | measured GEX from L1000 | Upper bound on what biology contributes when measurements exist. |

Conditions A, B, C are evaluable on the full DILIst drug list. Condition D is only evaluable on the DILIst ∩ L1000 intersection. The key comparison is B vs. C on the DILIst \ L1000 set, which is exactly the regime where predicted signatures matter.

🟡 DECISION NEEDED: Confirm with Joshua which CheMoE variant is the headline (StructuredMoE is the leading candidate per the AI4Chemistry summit results). All milestones below assume StructuredMoE.

---

## 3. Milestones

### M0 — Repository scaffolding and data inventory

**Goal:** Stand up the directory structure and confirm all data sources are accessible. No model code yet.

**Inputs:** None.

**Tasks:**
1. Create `experiments/dili_downstream/` with the layout in §0.
2. Write `data/MANIFEST.md` enumerating each dataset, its source, version, license, and local path. Required entries:
   - DILIst (FDA NCTR, latest release)
   - DILIrank (FDA NCTR 2016)
   - LINCS L1000 Phase II level 5 signatures (the version MultiDCP was trained on)
   - DILIst SMILES (resolve via DrugBank or PubChem; document the resolution method)
   - Yoojean's chemical encoder weights and inference interface (path TBD from Joshua)
   - Frozen MultiDCP checkpoint (path: confirm with Joshua)
   - Frozen MultiDCP-CheMoE StructuredMoE checkpoint (path: confirm with Joshua)
3. Write `tests/test_data_paths.py` that asserts every path in the manifest exists and is readable. This is the only sanity test in the entire pipeline; it runs before every other milestone.

**Deliverable:** Directory structure committed. `MANIFEST.md` with all paths. `test_data_paths.py` passing.

**Verification:** `pytest tests/test_data_paths.py` returns green.

**Blocked on:** Confirmation from Joshua of MultiDCP and CheMoE checkpoint paths, Yoojean's encoder interface.

---

### M1 — DILIst label harmonization and SMILES resolution

**Goal:** Produce a single canonical CSV mapping `pert_id → SMILES → DILI binary label → DILIrank severity` for every drug in DILIst.

**Inputs:** Raw DILIst download, DILIrank download, LINCS pert_id → SMILES mapping.

**Tasks:**
1. Load DILIst raw file. Standardize binary labels: `vMost-DILI-Concern + Less-DILI-Concern → 1`, `No-DILI-Concern → 0`, drop `Ambiguous-DILI-Concern`. Document the count in each bucket.
2. Resolve SMILES via DrugBank/PubChem name match. Canonicalize via RDKit (`Chem.MolToSmiles(Chem.MolFromSmiles(s))`). Drop entries that fail to parse and log them to `data/processed/dili_smiles_resolution_failures.csv`.
3. Compute Murcko scaffolds for every resolved drug.
4. Cross-reference DILIrank: where a drug is in both, attach the four-class severity label as an additional column. This will be used for stratified analysis in M7.
5. Write `data/processed/dili_canonical.csv` with columns: `pert_id`, `drug_name`, `smiles`, `canonical_smiles`, `scaffold`, `dili_binary`, `dili_severity` (nullable), `in_lincs` (bool).

**Deliverable:** `data/processed/dili_canonical.csv`. Resolution failure log. Brief markdown summary in `results/tables/M1_data_summary.md` reporting class balance, SMILES resolution rate, and intersection size with LINCS.

**Verification:** Class balance is reported. SMILES resolution rate is ≥ 90%. Every row has a valid canonical SMILES that round-trips through RDKit. Intersection size with LINCS is reported as an absolute count and a fraction.

**Blocked on:** M0 complete.

---

### M2 — Compound-aware splits

**Goal:** Generate train/val/test splits that no compound crosses, and that respect scaffold structure.

**Inputs:** `data/processed/dili_canonical.csv`.

**Tasks:**
1. Generate scaffold split (Murcko-based, 80/10/10) using the standard scaffold-split implementation. Stratify on `dili_binary` to preserve class balance across splits.
2. Generate a second split: cluster-based (Tanimoto threshold 0.4 on Morgan fingerprints, single linkage). Same 80/10/10. This is the harder OOD test.
3. Generate a third stratification: `lincs_held_out` — drugs in the test set that are NOT in LINCS. This is the regime where MultiDCP predicted signatures must do work the measured signatures cannot. Report its size; it is the most important slice in the final results table.
4. Save splits as `data/splits/scaffold_split.json`, `data/splits/cluster_split.json` with structure `{"train": [pert_ids], "val": [...], "test": [...], "test_lincs_held_out": [...]}`.

**Deliverable:** Two split files. A summary in `results/tables/M2_split_summary.md` with class balance per split, Tanimoto-similarity histogram for train-vs-test in each split, and the size of `test_lincs_held_out`.

**Verification:** No drug appears in more than one of train/val/test in either split. Class balance is within 5 percentage points of the global rate in every split partition. Tanimoto train-test max similarity histogram is reported.

**Blocked on:** M1 complete.

---

### M3 — Signature caching for all four conditions

**Goal:** Pre-compute and cache the biological signature for every drug under each condition. Caching is mandatory because re-running MultiDCP inference per epoch is wasteful.

**Inputs:** Frozen MultiDCP, frozen MultiDCP-CheMoE, LINCS L1000 level 5 signatures, `dili_canonical.csv`.

**Tasks:**
1. Implement `signatures/multidcp_signature.py`: load frozen MultiDCP, run inference for each (drug, cell_line) pair across all nine MultiDCP cell lines, save as a tensor of shape `(n_drugs, 9_cell_lines, n_genes)` in `data/processed/sig_multidcp.pt`. Compute the differential expression form (`predicted_treated − diseased`) and save separately as `sig_multidcp_de.pt`. The DE form is what downstream classifiers consume; this is enforced by the DE rule.
2. Implement `signatures/chemoe_signature.py`: same as above with the CheMoE checkpoint, saving `sig_chemoe_de.pt`.
3. Implement `signatures/lincs_signature.py`: pull measured level 5 signatures from LINCS for the DILIst ∩ LINCS intersection, average across replicates within (drug, cell_line) per the L1000 distil_ss convention, compute DE form, save `sig_lincs_de.pt`. For drugs not in LINCS, store NaN — the loader masks these in condition D.
4. Each signature tensor is accompanied by a `_manifest.json` file listing the pert_id ordering, cell line ordering, and gene ordering. The orderings must be identical across all three signature files.

**Deliverable:** Three `.pt` files plus three manifests. Brief sanity check in `results/tables/M3_signature_summary.md` reporting tensor shapes, missingness rates, and the Pearson correlation between MultiDCP-DE and LINCS-DE on the intersection (this is a validity check on the predicted signatures themselves).

**Verification:** All three files load cleanly. Manifests align. The MultiDCP-vs-LINCS Pearson on the intersection is reported and documented; if it is below 0.3 averaged across cell lines, stop and flag — that means the predicted signatures are not capturing the measured signal and downstream results are unlikely to be meaningful.

**Blocked on:** M2 complete, MultiDCP and CheMoE checkpoints confirmed in M0.

---

### M4 — Yoojean's chemical encoder integration

**Goal:** Wrap Yoojean's encoder as a frozen feature extractor that takes SMILES and returns a fixed-dimensional embedding for every drug.

**Inputs:** Yoojean's encoder weights and code, `dili_canonical.csv`.

**Tasks:**
1. Implement `encoders/chemical_encoder.py` exposing `encode(smiles_list: list[str]) -> torch.Tensor` of shape `(n_drugs, d_chem)`. Internally calls Yoojean's encoder in eval mode with frozen weights.
2. Pre-compute embeddings for all DILIst drugs and cache as `data/processed/chem_embeddings.pt` with manifest.
3. Document `d_chem` (the embedding dimension) in `results/tables/M4_encoder_summary.md`.

**Deliverable:** Encoder wrapper, cached embeddings, manifest, summary doc.

**Verification:** Embeddings tensor shape matches `(n_drugs, d_chem)`. No NaNs. Spot check three known drug pairs: Tanimoto-similar pairs should have higher embedding cosine similarity than Tanimoto-distant pairs (qualitative sanity, not a hard test).

**Blocked on:** M1 complete, Yoojean's encoder interface from Joshua.

---

### M5 — Classifier architecture and training loop

**Goal:** Build a single classifier architecture that accepts (chem_embedding, signature) pairs and produces a DILI probability. The same architecture is used across all four conditions, with the signature input replaced or zeroed depending on condition.

**Inputs:** All cached signatures, chem embeddings, splits.

**Tasks:**
1. Implement `classifiers/concat_classifier.py`. Input: concatenation of `chem_embedding` (dim `d_chem`) and flattened signature (dim `9 × n_genes` for predicted, or a learned cell-line-aware aggregation — see decision below). Output: scalar logit. Architecture: 3-layer MLP with dropout, batchnorm, GELU. For condition A (structure-only), the signature input is replaced with a zero tensor of the same shape, so model capacity is held constant across conditions.
2. Implement `train.py`: standard binary cross-entropy training, AdamW, cosine schedule, early stopping on val AUROC with patience 10. Loads config from YAML. Logs everything to wandb.
3. Implement `eval.py`: computes AUROC, AUPRC, MCC, balanced accuracy on val and test sets. Also computes metrics on `test_lincs_held_out` slice separately.

🟡 DECISION NEEDED: Cell-line aggregation. Three options for collapsing the 9-cell-line signature into a vector for the classifier: (a) flatten and let the MLP learn cell-line weighting, (b) average across cell lines, (c) attention pool with a learned query. Default: option (a) because it is the most expressive, and we are not committed to a specific cell-line-organ mapping for liver. Revisit if the M6 results show structure-only beats predicted-signature, in which case (c) with HepG2-like prior weighting may be worth trying as future work.

**Deliverable:** Classifier code, training script, eval script. Smoke test: train condition A on the scaffold split for 5 epochs and confirm wandb is logging.

**Verification:** Smoke test completes without error. wandb run shows training and validation curves. No NaN losses.

**Blocked on:** M3 and M4 complete.

---

### M6 — Headline experiment: four-condition comparison on scaffold split

**Goal:** Run all four conditions, three seeds each, on the scaffold split. Report the headline results table.

**Inputs:** All artifacts from M0–M5.

**Tasks:**
1. Run condition A (structure-only) × 3 seeds.
2. Run condition B (MultiDCP) × 3 seeds.
3. Run condition C (MultiDCP-CheMoE) × 3 seeds.
4. Run condition D (LINCS upper bound) × 3 seeds, on the DILIst ∩ LINCS subset only. Train and test sets are restricted to this subset for D so the upper bound is comparable.
5. Aggregate results into `results/tables/M6_headline.md` with the table:

| Condition | Test AUROC | Test AUPRC | Test MCC | Test_LINCS_held_out AUROC |
|---|---|---|---|---|
| A. Structure-only | μ ± σ | μ ± σ | μ ± σ | μ ± σ |
| B. MultiDCP (baseline) | μ ± σ | μ ± σ | μ ± σ | μ ± σ |
| C. MultiDCP-CheMoE | μ ± σ | μ ± σ | μ ± σ | μ ± σ |
| D. LINCS (upper bound) | μ ± σ | μ ± σ | μ ± σ | n/a |

**Deliverable:** Headline table. wandb run group `dili_downstream_M6` with all 12 runs (4 conditions × 3 seeds).

**Verification:** All 12 runs complete. The table is populated. The key comparison `C vs B` and `B vs A` is computed with paired bootstrap CIs (10,000 resamples) over the test set.

**Decision gate:** If B does not beat A by at least 0.02 AUROC (mean), stop and consult Joshua before proceeding to M7+. The thesis claim depends on biology contributing real signal.

**Blocked on:** M5 complete.

---

### M7 — Cell viability confound ablation (Szalai et al. control)

**Goal:** Demonstrate that the M6 results are not driven solely by a generic cell-death signature.

**Inputs:** Headline results from M6.

**Tasks:**
1. Implement `ablations/viability_mask.py`. Compute Pearson correlation between every gene's MultiDCP-predicted DE and a viability label per the Szalai protocol (cross-reference with CTRP viability data on the LINCS-CTRP intersection). Identify the top 10% most viability-correlated genes.
2. Re-run conditions B and C with the viability-correlated genes masked from the signature. Three seeds each.
3. Implement a viability-only baseline: a classifier that takes only a single scalar predicted-viability score (computed by averaging the predicted DE across the viability-correlated gene set) plus the chem embedding. Three seeds.
4. Report all three results in `results/tables/M7_viability_ablation.md` alongside M6 numbers.

**Deliverable:** Ablation table showing (B with mask), (C with mask), (viability-only). Brief interpretation paragraph.

**Verification:** If `(B with mask) ≈ B` and `(C with mask) ≈ C`, the result is robust to the viability confound. If there is a large drop, the original numbers are partially confounded and need to be reframed in the writeup.

**Decision gate:** Independent of the verification, this ablation MUST appear in the final paper. Reviewers will ask for it. Do not skip.

**Blocked on:** M6 complete.

---

### M8 — Cluster split robustness check

**Goal:** Confirm that the M6 ranking holds under a harder OOD test.

**Inputs:** Cluster split from M2, classifiers from M5.

**Tasks:**
1. Repeat the four-condition comparison on the cluster split. Three seeds each.
2. Report results in `results/tables/M8_cluster_split.md`.
3. Compare the gap between scaffold split and cluster split per condition. Larger gaps indicate worse OOD generalization.

**Deliverable:** Cluster split results table. A short paragraph on OOD generalization, especially on whether C (CheMoE) maintains its lead over B (baseline) more robustly than B over A.

**Verification:** Table is populated. The OOD-gap analysis is reported.

**Blocked on:** M6 complete.

---

### M9 — Routing permutation diagnostic for CheMoE

**Goal:** Mechanistically validate that CheMoE's MoE routing is doing real work, not just adding parameters.

**Inputs:** CheMoE classifier from condition C in M6.

**Tasks:**
1. Implement `ablations/permutation_diagnostic.py` adapting the Li et al. 2026 permutation protocol: at inference time, randomly permute the gating weights of the StructuredMoE fusion layer across the test set. Measure ΔAUROC.
2. Run with 100 random permutations, report mean and 95% CI of ΔAUROC.
3. Compare against a control: permute a non-gating layer (e.g., the first hidden layer of the classifier head) by the same protocol.

**Deliverable:** `results/tables/M9_permutation.md` reporting ΔAUROC under gate-weight permutation vs. control-layer permutation. A more negative ΔAUROC under gate permutation indicates the gates carry real, non-interchangeable specialization.

**Verification:** If gate-permutation ΔAUROC is not more negative than control-layer permutation, the CheMoE advantage in M6 is unlikely to come from the MoE structure itself, and the writeup needs to reflect that honestly.

**Blocked on:** M6 complete.

---

### M10 — DILIrank severity stratification

**Goal:** Confirm that classifier confidence tracks DILI severity (Most > Less > No), not just the binary label. This is an interpretability check, not a headline result.

**Inputs:** Trained classifiers from M6 condition C, DILIrank severity labels from M1.

**Tasks:**
1. For test-set drugs that have DILIrank severity labels, compute the mean predicted DILI probability per severity bucket.
2. Report a table and a violin plot in `results/figures/M10_severity_calibration.png`.

**Deliverable:** Table and figure. Short interpretation paragraph.

**Verification:** Mean predicted probability is monotone increasing across (No → Less → Most). If not, document and discuss; this is a meaningful negative result for the interpretability claim.

**Blocked on:** M6 complete.

---

### M11 — GSEA on top features

**Goal:** Identify the gene programs that the classifier relies on, for the interpretability section of the paper.

**Inputs:** Trained classifier C from M6.

**Tasks:**
1. Compute integrated gradient attributions over the gene-input dimension on the test set.
2. Rank genes by mean absolute attribution. Take the top 200.
3. Run GSEA (Hallmark + Reactome) on the ranked list using `gseapy` or equivalent.
4. Save enrichment results table and dotplot to `results/figures/M11_gsea.{md,png}`.

**Deliverable:** GSEA table and dotplot. A short paragraph identifying the top three enriched pathways and whether they are mechanistically plausible for hepatotoxicity (e.g., bile acid metabolism, oxidative stress, mitochondrial dysfunction).

**Verification:** Top pathways are reported. Mechanistic plausibility is discussed.

**Blocked on:** M6 complete.

---

### M12 — Hyperparameter sweep on the headline classifier

**Goal:** Confirm the M6 results are not dependent on a fragile hyperparameter choice.

**Inputs:** Classifier from M5, condition C config.

**Tasks:**
1. Define a wandb sweep over: hidden dim {256, 512, 1024}, dropout {0.1, 0.2, 0.4}, learning rate {1e-4, 5e-4, 1e-3}, weight decay {1e-5, 1e-4}.
2. Run the sweep on the val set of the scaffold split.
3. Report top 5 configurations and their corresponding test AUROCs.

**Deliverable:** Sweep config in `configs/sweep_classifier_hp.yaml`. Results in `results/tables/M12_sweep.md`.

**Verification:** Sweep completes. Top 5 configurations reported. Test AUROCs across the top 5 are within 0.02 of each other (indicating stability).

**Blocked on:** M6 complete.

---

## 4. Out of scope (explicit, do not build)

- Cardiotoxicity (DICTrank). Separate GSD doc.
- Nephrotoxicity (DIRIL). Separate GSD doc.
- Neurotoxicity (Zhao et al., SIDER). Separate GSD doc.
- TG-GATEs auxiliary pretraining. Possible future extension; not part of the v1 paper.
- ChemBioHepatox head-to-head. Worth doing once M6 numbers are in hand, but only if their code or test set is publicly available; otherwise we cite their numbers and contextualize.
- Drug-drug interaction prediction.
- IC-50 / GDSC sanity checks.
- Any non-DE evaluation metric.

---

## 5. Success criteria for the liver arm

The liver arm is "complete" and ready to write up when ALL of the following are true:

1. M6 headline table populated, with B beating A on AUROC by at least 0.02 (mean), and C beating B on `test_lincs_held_out` AUROC by at least 0.01 (mean).
2. M7 viability ablation showing the result is not driven entirely by a viability confound.
3. M8 cluster-split robustness showing the ranking holds.
4. M9 routing permutation diagnostic showing CheMoE's gates carry real specialization.
5. At least one of M10, M11 producing a mechanistically plausible interpretation.
6. M12 hyperparameter sweep showing stability.

If any of these fail, document honestly in the writeup. A negative result on M9 (CheMoE not mechanistically validated) is publishable as a methodological finding, but the framing of the paper changes.

---

## 6. Open decisions for Joshua

🟡 **Cell line aggregation in the classifier (M5).** Default flatten-and-learn, alternatives are average pool or attention pool. Revisit only if M6 condition B underperforms condition A.

🟡 **CheMoE variant (M3).** Assumed StructuredMoE based on prior poster results. Confirm.

🟡 **DILIst version.** Multiple FDA NCTR releases exist. Specify the exact release that aligns with what MultiDCP was trained against in time, to avoid leakage.

🟡 **Threshold for "predicted signatures are valid" sanity check (M3).** Currently set at Pearson 0.3 averaged across cell lines for MultiDCP-vs-LINCS DE on the intersection. Confirm this threshold or set a different one.

🟡 **Bootstrapping protocol for paired CIs (M6).** Default is 10,000 resamples. Confirm.

---

## 7. Things we are deliberately NOT doing yet

- Training MultiDCP on hepatocyte cell lines. The cell-line-organ mismatch is a known limitation; we acknowledge it in the writeup but do not extend MultiDCP for this paper.
- Building a custom architecture for the downstream classifier. The MLP is intentionally simple so the headline comparison is about the *signature* condition, not the classifier.
- Multi-task learning across all four organs. This is the eventual full-thesis story, not the v1 liver paper.

---

## 8. References for context

- Wang et al. 2020. "Deep Learning on High-Throughput Transcriptomics to Predict Drug-Induced Liver Injury." PMC7728858. The direct precedent.
- Chen et al. 2016. "DILIrank." *Drug Discovery Today*.
- Thakkar et al. 2020. "DILIst." *Drug Discovery Today*.
- Szalai et al. 2018. "Signatures of cell death and proliferation in perturbation transcriptomics data." bioRxiv. The viability confound.
- Li et al. 2026. "Expert Divergence Learning." ICLR 2026. The permutation diagnostic for M9.
- Van Nieuwerburgh group, Nov 2025. "Mechanistically Interpretable Toxicity Prediction." bioRxiv 2025.11.14.686754. Closest contemporary work.
- ChemBioHepatox. The closest competitor (find primary citation; webserver at exposomex.cn:58080).

End of document.
