# Roadmap v1.0 — Multi-Head MultiDCP DILI

**Project:** Multi-Head MultiDCP DILI (parallel branch to v0.5)
**Milestone:** v1.0 (this branch)
**Core Value:** Three independent encoder→prediction pathways (predicted dose-response, predicted GEX, frozen MolFormer chemical embedding) concatenated as input to a 7-way pathway ablation DILI classifier, tested on scaffold-novel DILIst drugs.
**Created:** 2026-05-19
**Depth:** Coarse (6 GSD-managed phases) — `granularity: coarse` in `.planning/config.json`
**Code home:** `/raid/home/joshua/projects/GEX_vs_chemical_experiments/multihead_dili/`

---

## Phase Progress

- [x] Phase 0: Data foundation (pre-GSD prerequisite — complete 2026-05-20; commit e281e9f)
- [x] Phase 1: MODEL_DOSE training (HG1 PASS — dev RMSE=19.455 vs baseline=33.641 — 2026-05-20)
- [x] Phase 2: MODEL_GEX training (HG2 PASS — mean Pearson=0.3568 — 2026-05-20)
- [x] Phase 3: MolFormer download + Stage-2 feature caching (EMBED-04 PASS — 1118 drugs, 1688-dim, 0 NaN — 2026-05-20)
- [x] Phase 4: DILI consumer (7-way pathway ablation) (HG3 PASS — embed-only random AUROC=0.6536 — 2026-05-20)
- [x] Phase 5: Evaluation (HG4 REFRAME — embed-only matches all-three, DeLong NS — 2026-05-20; commit ef4296b)
- [x] Phase 6: Milestone summary (COMPLETE 2026-05-20)

---

## Overview

This branch implements a two-stage training design as a sibling subdir to `dili_downstream/` (v0.5). Stage 1 trains MODEL_DOSE (MultiDCP-AE fork on E-Hill) and MODEL_GEX (MultiDCP-AE fork on LINCS) independently — they share no parameters and can run on different GPUs in parallel. Stage 2 caches DILIst features by querying both models at all 9 MultiDCP LINCS cells and mean-pooling, then concatenates with a frozen MolFormer embedding (1,747-dim feature vector per drug). The DILI consumer is a linear/mlp1/mlp2 head on each of 7 pathway-ablation variants.

**Source-of-truth docs:**
- `/raid/home/joshua/projects/0_project_documents/multihead_multidcp_dili_three_pathway_05192026.md` (design doc, committed f921223)
- `/raid/home/joshua/projects/0_project_documents/multihead_multidcp_dili_implementation_plan_05192026.md` (this branch's implementation plan)

**WandB:** `joshroll/MultiDCP_multihead_dili`, run groups: `model_dose`, `model_gex`, `multihead_dili`.

**Key principles:**
- **Real data only.** No mocking, stubbing, or synthetic labels.
- **DE rule.** All GEX-derived features computed on differential expression (`treated − diseased`), top-k by `|true_DE|`.
- **Leakage discipline.** Murcko-scaffold + drug_name filter applied to Stage-1 training data BEFORE training. Test scaffolds from DILIst never seen by MODEL_DOSE or MODEL_GEX.
- **CUDA hygiene.** `--gpu` argparse before `import torch`. Always leave one GPU free.
- **Atomic commits.** One phase per commit. Format: `multihead_dili: P{n} — {short description}`.

---

## Phase 0: Data foundation (pre-GSD prerequisite — COMPLETE)

Phase 0 was completed via the `writing-plans` + `executing-plans` workflow before GSD took over. It is a pre-GSD prerequisite; GSD manages Phases 1–6 below.

**Deliverables (committed e281e9f):**
- `multihead_dili/` subdir skeleton (src/, configs/, scripts/, tests/, data/, results/)
- `CLAUDE.md` + `MANIFEST.md` skeleton
- `.planning/` GSD scaffolding (PROJECT.md, ROADMAP.md, MILESTONES.md, STATE.md)
- `src/data/scaffold_split.py` + `src/data/upstream_filter.py` (copied from v0.5)
- `tests/test_data_paths.py` (E-Hill + DILIst sanity)
- `scripts/build_dili_scaffold_split.py` → emits `data/processed/dili_split.json`
- `scripts/run_leakage_filter.py` → emits `data/processed/ehill_train_safe.parquet`, `data/processed/lincs_train_safe.parquet`, `data/processed/leakage_report.md`

**Status: COMPLETE** — Phases 1–6 below are GSD-managed.

---

## Phase 1: MODEL_DOSE training

**Goal:** Train MultiDCP-AE-based MODEL_DOSE on leakage-filtered E-Hill and confirm dev RMSE beats the predict-mean baseline.

**Dependencies:** Phase 0 (pre-GSD, complete). Inputs: `data/processed/ehill_train_safe.parquet`, pinned MultiDCP SHA.

**Requirements:**

- DOSE-01: Fork `MultiDCP/ehill_multidcp_pretrain.py` → `src/train/train_model_dose.py` with `--gpu` argparse and `--safe-parquet` flag to accept the leakage-filtered input
- DOSE-02: Filter dev set by same scaffold/name exclusion (emit `data/processed/ehill_dev_safe.parquet`)
- DOSE-03: Train MODEL_DOSE to convergence; log run to WandB run group `model_dose`; save `results/checkpoints/chkpt_dose.pt`
- DOSE-04: Record dev RMSE and predict-mean baseline RMSE; compare — halt gate 1 check
- DOSE-05: Write `results/tables/P1_model_dose_summary.md` (architecture, HP, dev RMSE, halt gate verdict, MultiDCP SHA)

**Plans:** TBD (≥ 1).

Plans:
- [ ] 01-01-PLAN.md — train_model_dose.py fork + leakage-safe dev filter [Wave 1]
- [ ] 01-02-PLAN.md — training run + WandB logging + checkpoint save [Wave 2]
- [ ] 01-03-PLAN.md — halt gate 1 check + P1 summary [Wave 3]

**Success criteria:**

1. `results/checkpoints/chkpt_dose.pt` exists and loads without error
2. Dev RMSE < predict-mean baseline RMSE (halt gate 1 passes)
3. `P1_model_dose_summary.md` records RMSE values, MultiDCP SHA, and halt gate verdict
4. WandB run logged under `joshroll/MultiDCP_multihead_dili`, group `model_dose`

**Verification:** Load `chkpt_dose.pt` and run `python -c "import torch; ckpt = torch.load('results/checkpoints/chkpt_dose.pt', map_location='cpu'); print(ckpt.keys())"`. Confirm `P1_model_dose_summary.md` exists. Halt gate 1 verdict must be PASS.

**Halt gate 1:** MODEL_DOSE dev RMSE ≥ predict-mean baseline RMSE → write `phases/01-model-dose/HALT_REASON.md` and stop. Do not proceed to Phase 3 until reviewed.

---

## Phase 2: MODEL_GEX training

**Goal:** Train MultiDCP-AE on leakage-filtered LINCS PDG-filtered data using the canonical DE-rule evaluator; confirm dev predicted-vs-measured Pearson ≥ 0.2 averaged over cells.

**Dependencies:** Phase 0 (pre-GSD, complete). Inputs: `data/processed/lincs_train_safe.parquet`, pinned MultiDCP SHA. Runs in parallel with Phase 1 (different GPU).

**Requirements:**

- GEX-01: Fork MultiDCP AE training script → `src/train/train_model_gex.py` with `--gpu` argparse and `--safe-parquet` flag
- GEX-02: Apply DE rule: evaluate on differential expression (`treated − diseased`), top-k by `|true_DE|`; use `train_bl_pdg_de.py` as canonical reference
- GEX-03: Train MODEL_GEX to convergence; log run to WandB run group `model_gex`; save `results/checkpoints/chkpt_gex.pt`
- GEX-04: Compute dev predicted-vs-measured Pearson per cell and averaged — halt gate 2 check
- GEX-05: Write `results/tables/P2_model_gex_summary.md` (architecture, HP, per-cell Pearson, mean Pearson, halt gate verdict, MultiDCP SHA)

**Plans:** TBD (≥ 1).

Plans:
- [x] 02-01-PLAN.md — train_model_gex.py fork + DE-rule evaluator [Wave 1] — COMPLETE (commit 9216d06)
- [x] 02-02-PLAN.md — training run + WandB logging + checkpoint save [Wave 2] — COMPLETE (commit c567589)
- [x] 02-03-PLAN.md — halt gate 2 check + P2 summary [Wave 3] — COMPLETE (commit 92cc932)

**Success criteria:**

1. `results/checkpoints/chkpt_gex.pt` exists and loads without error
2. Dev predicted-vs-measured Pearson mean ≥ 0.2 (halt gate 2 passes)
3. `P2_model_gex_summary.md` records per-cell and mean Pearson, MultiDCP SHA, halt gate verdict
4. WandB run logged under `joshroll/MultiDCP_multihead_dili`, group `model_gex`

**Verification:** Confirm `chkpt_gex.pt` loadable. Check `P2_model_gex_summary.md` for halt gate 2 PASS verdict.

**Halt gate 2:** MODEL_GEX dev predicted-vs-measured Pearson < 0.2 averaged over cells → write `phases/02-model-gex/HALT_REASON.md` and stop. Do not proceed to Phase 3 until reviewed.

---

## Phase 3: MolFormer download + Stage-2 feature caching

**Goal:** Download frozen MolFormer (HF `ibm/MoLFormer-XL-both-10pct`); run the 9-cell mean-pool inference for MODEL_DOSE and MODEL_GEX on all 1,118 DILIst drugs; concatenate with MolFormer embedding → 1,747-dim feature vector per drug; save `data/processed/dili_features.parquet`.

**Dependencies:** Phase 1 (chkpt_dose.pt), Phase 2 (chkpt_gex.pt).

**Requirements:**

- EMBED-01: Write `src/embed/molformer_wrapper.py` — frozen HF inference, SMILES → 768-dim embedding
- EMBED-02: Write `src/stage2/cache_dili_features.py` — for each DILIst drug: query MODEL_DOSE at 9 LINCS cells, mean-pool (scalar); query MODEL_GEX at 9 cells, mean-pool (978-dim DE); concat with MolFormer → 1,747-dim
- EMBED-03: Run `cache_dili_features.py` on `dili_canonical.csv`; emit `data/processed/dili_features.parquet` (1,118 rows × [1 + 978 + 768] feature cols + label cols)
- EMBED-04: Sanity-check: no NaNs in any feature column; print mean/std per pathway block
- EMBED-05: Pin MolFormer HF model SHA in `MANIFEST.md`

**Plans:** TBD (≥ 1).

Plans:
- [ ] 03-01-PLAN.md — molformer_wrapper.py + cache_dili_features.py [Wave 1]
- [ ] 03-02-PLAN.md — feature caching run + NaN check + MANIFEST pin [Wave 2]

**Success criteria:**

1. `data/processed/dili_features.parquet` exists with exactly 1,118 rows and 0 NaN values in any feature column
2. Shape is (1118, ≥ 1748) including label and ID columns
3. Mean/std sanity printed per pathway (DOSE scalar, GEX 978-dim, MolFormer 768-dim)
4. MolFormer HF model SHA recorded in `MANIFEST.md`

**Verification:** `python -c "import pandas as pd; df = pd.read_parquet('data/processed/dili_features.parquet'); assert len(df) == 1118; assert df.isnull().sum().sum() == 0; print('OK', df.shape)"`.

---

## Phase 4: DILI consumer (7-way pathway ablation)

**Goal:** Train 630 DILI classifiers (3 head depths × 7 ablation variants × 2 splits × 5 folds × 3 seeds) on cached features; cache test-fold predictions; check halt gate 3.

**Dependencies:** Phase 3 (dili_features.parquet, dili_split.json).

**Requirements:**

- ABLATE-01: Write `src/stage2/classifiers.py` — linear (1,747→1), mlp1 (1,747→128→1, dropout 0.3, ReLU), mlp2 (1,747→256→64→1, dropout 0.3, batchnorm, ReLU); locked HPs across all variants
- ABLATE-02: Define the 7 ablation variants (input feature subsets): var1=DOSE-only, var2=GEX-only, var3=MolFormer-only, var4=DOSE+GEX, var5=DOSE+MolFormer, var6=GEX+MolFormer, var7=all-three
- ABLATE-03: Write `src/stage2/train_dili_classifier.py` — grid driver over (variant × head × split × fold × seed); log to WandB group `multihead_dili`; write per-run prediction parquets
- ABLATE-04: Run all 630 classifiers; emit `data/processed/predictions/var{1..7}_head{linear,mlp1,mlp2}_split{scaffold,random}_fold{0..4}_seed{0..2}.parquet`; aggregate run-level AUROC to `P4_runs.parquet`
- ABLATE-05: Halt gate 3 check: embed-only (var3=MolFormer-only) random-split AUROC > 0.55 across seeds/heads
- ABLATE-06: Write `results/tables/P4_ablation_summary.md` (per-variant per-head mean ± std AUROC, halt gate 3 verdict)

**Plans:** TBD (≥ 1).

Plans:
- [x] 04-01-PLAN.md — classifiers.py + train_dili_classifier.py [Wave 1] — COMPLETE (commit 0d210aa)
- [x] 04-02-PLAN.md — 630-run grid execution (embarrassingly parallel) [Wave 2] — COMPLETE (630 parquets)
- [x] 04-03-PLAN.md — halt gate 3 + P4 summary [Wave 3] — COMPLETE (HG3 PASS)

**Success criteria:**

1. 630 prediction parquets exist (or equivalent consolidated file)
2. `P4_runs.parquet` has one row per run with AUROC, AUPRC, MCC, balanced accuracy
3. Halt gate 3 passes (embed-only AUROC > 0.55)
4. `P4_ablation_summary.md` reports all 21 cells (7 variants × 3 heads) mean ± std AUROC

**Verification:** `find data/processed/predictions/ -name "*.parquet" | wc -l` ≥ 630. Halt gate 3 verdict PASS in summary.

**Halt gate 3:** MolFormer-only (var3) random-split AUROC ≤ 0.55 averaged over seeds/heads → write `phases/04-dili-consumer/HALT_REASON.md` and stop. Chemical embedding not capturing DILI signal — consult before Phase 5.

---

## Phase 5: Evaluation

**Goal:** Compute full metrics (AUROC + AUPRC + MCC + balanced accuracy + ECE) with DeLong paired tests and bootstrap CIs; produce headline table, 7-way ablation figure, and comparison-to-v0.5 figure; check halt gate 4.

**Dependencies:** Phase 4 (630 prediction parquets).

**Requirements:**

- EVAL-01: Write `src/stage2/evaluate_dili.py` — DeLong paired test (all-three vs best single), bootstrap CIs (10K resamples), ECE calibration, AUROC + AUPRC + MCC + balanced accuracy per cell
- EVAL-02: Run evaluation over all 21 cells (7 variants × 3 heads) for both splits; emit `results/tables/headline.md` (LaTeX-ready)
- EVAL-03: Produce `results/figures/ablation.png` — 7-way ablation bar chart with CIs
- EVAL-04: Produce `results/figures/comparison_v05.png` — comparison of multihead DILI best cell vs v0.5 best cell (if v0.5 results are available)
- EVAL-05: Halt gate 4 check: all-three (var7) AUROC ≥ best single-pathway AUROC + 0.01 on scaffold-novel split; if fails, write `HALT_REASON_4.md` and reframe paper claim

**Plans:** TBD (≥ 1).

Plans:
- [ ] 05-01-PLAN.md — evaluate_dili.py (DeLong + bootstrap + calibration) [Wave 1]
- [ ] 05-02-PLAN.md — headline table + figures + halt gate 4 check [Wave 2]

**Success criteria:**

1. `results/tables/headline.md` publishable as-is (all 21 cells + DeLong p-values + 95% bootstrap CIs)
2. `results/figures/ablation.png` shows 7-way ablation with error bars
3. `results/figures/comparison_v05.png` exists (even if comparison is qualitative)
4. Halt gate 4: var7 AUROC ≥ best single + 0.01 on scaffold split — if fails, `HALT_REASON_4.md` documents the negative finding and reframing

**Verification:** All figures and tables exist. Halt gate 4 verdict documented (PASS or HALT with reframe).

**Halt gate 4:** All-three (var7) AUROC < best single-pathway AUROC + 0.01 on scaffold-novel split → write `phases/05-evaluation/HALT_REASON_4.md` documenting the negative finding, reframe paper claim as "no synergy between pathways on scaffold-novel drugs." Publishable as a methodological negative. Stop until user reviews reframing.

---

## Phase 6: Milestone summary

**Goal:** Write the milestone summary interpreting the headline result, listing v2.0 candidate directions (PK head, encoder-axis ablation, multi-organ DILI, etc.).

**Dependencies:** Phase 5.

**Requirements:**

- SUMMARY-01: Write `results/tables/v1_milestone_summary.md` — headline finding, per-pathway AUROC comparison, halt gate outcomes, key lessons, v2.0 candidate directions
- SUMMARY-02: Update `.planning/STATE.md` and `.planning/MILESTONES.md` to mark v1.0 complete

**Plans:** TBD (≥ 1).

Plans:
- [ ] 06-01-PLAN.md — v1_milestone_summary.md + state/milestone updates [Wave 1]

**Success criteria:**

1. `results/tables/v1_milestone_summary.md` exists with headline claim, per-pathway comparison, halt gate outcomes, and ≥ 3 v2.0 candidate directions
2. User reviews and signs off on milestone close

**Verification:** User review and sign-off.

---

## Halt Gates Summary

| # | Phase | Trigger | Action |
|---|-------|---------|--------|
| 1 | 1 | MODEL_DOSE dev RMSE ≥ predict-mean baseline RMSE | Write `phases/01-model-dose/HALT_REASON.md`; stop before Phase 3 |
| 2 | 2 | MODEL_GEX dev predicted-vs-measured Pearson < 0.2 averaged | Write `phases/02-model-gex/HALT_REASON.md`; stop before Phase 3 |
| 3 | 4 | MolFormer-only (var3) random-split AUROC ≤ 0.55 | Write `phases/04-dili-consumer/HALT_REASON.md`; stop before Phase 5 |
| 4 | 5 | All-three (var7) AUROC < best single + 0.01 on scaffold split | Write `phases/05-evaluation/HALT_REASON_4.md`; reframe paper claim; stop until user reviews |

---

## Execution Sequence

Phase 0 (pre-GSD, COMPLETE) → Phases 1 and 2 (parallel, different GPUs) → Phase 3 → Phase 4 (embarrassingly parallel, ~630 tiny classifier runs) → Phase 5 → Phase 6.

- **Phases 1 and 2** can run simultaneously on different GPUs (independent training jobs, no shared state).
- **Phase 3** requires both Phase 1 and Phase 2 complete (needs both checkpoints).
- **Phase 4** runs are embarrassingly parallel (630 independent tiny classifier trainings on cached features; parallelize across CPUs).
- **Phase 5** is CPU-only (~30 min for bootstrap + DeLong).

---

*Last updated: 2026-05-20 — GSD scaffolding restructured for /gsd-autonomous; ROADMAP converted from table to h2-header phase format*
