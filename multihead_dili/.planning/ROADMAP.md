# Multi-Head MultiDCP DILI — ROADMAP

**Milestone:** v1.0 (this branch)
**Phases:** 7 (P0 done in implementation-plan TDD; P1–P6 deferred to /gsd-plan-phase)

| Phase | Goal | Deliverable | Halt gate |
|---|---|---|---|
| P0 | Subdir scaffold + data foundation (leakage-filtered train sets) | ehill_train_safe.parquet, lincs_train_safe.parquet, leakage_report.md | None |
| P1 | MODEL_DOSE training | chkpt_dose.pt | HG1: dev RMSE > predict-mean |
| P2 | MODEL_GEX training (parallelizable with P1) | chkpt_gex.pt | HG2: dev Pearson < 0.2 |
| P3 | MolFormer download + Stage-2 feature caching | dili_features.parquet (1118 × 1747) | None |
| P4 | DILI consumer 7-way ablation | predictions_test_fold{k}_seed{s}_var{v}_head{h}.parquet (630 files) | HG3: embed-only AUROC < 0.55 |
| P5 | Evaluation (DeLong + bootstrap + calibration + headline table) | results/tables/headline.md, results/figures/ablation.png | HG4: all-three not >> best single → reframe |
| P6 | Milestone summary + v2.0 candidate directions | results/tables/milestone_summary.md | None |

P1 and P2 are parallel (different GPUs). P4 runs are embarrassingly parallel.
