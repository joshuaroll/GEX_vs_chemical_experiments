---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: multihead_dili_v1
status: executing
stopped_at: Phase 4 complete (HG3 PASS); Phase 5 (evaluation + DeLong + figures) next
last_updated: "2026-05-20T07:50:00Z"
last_activity: 2026-05-20 -- Phase 4 complete; 630 prediction parquets written; HG3 PASS (embed-only random AUROC=0.6536 >> 0.55)
progress:
  total_phases: 6
  completed_phases: 4
  total_plans: 12
  completed_plans: 8
  percent: 67
---

# Project State

## Project Reference

See: `multihead_dili/.planning/PROJECT.md`
See: `/raid/home/joshua/projects/0_project_documents/multihead_multidcp_dili_three_pathway_05192026.md`
See: `/raid/home/joshua/projects/0_project_documents/multihead_multidcp_dili_implementation_plan_05192026.md`

## Current Position

Phase 1: COMPLETE — MODEL_DOSE training (HG1 PASS, dev RMSE=19.455 vs baseline=33.641)
Phase 2: COMPLETE — MODEL_GEX training (HG2 PASS, mean Pearson=0.3568)
Phase 3: COMPLETE — MolFormer + Stage-2 feature caching (EMBED-04 PASS, 0 NaN)
Phase 4: COMPLETE — 7-way pathway ablation (HG3 PASS, embed-only random AUROC=0.6536)
Next: Phase 5 (evaluation — DeLong paired tests, bootstrap CIs, figures, HG4)

## Phase 3 Deliverables (commits 6e0bbdb + 0b5fe40)

- `src/embed/molformer_wrapper.py` — frozen MolFormer encoder (768-dim SMILES embedding)
- `src/stage2/cache_dili_features.py` — Stage-2 caching driver
- `data/processed/dili_features.parquet` — 1118 rows × 1694 cols (1688 features)
- `tests/test_stage2_features.py` — 9 CI regression tests (all pass)
- `results/tables/P3_stage2_summary.md` — phase summary with deviations documented

Key findings from Phase 3:
- SMILES failure: nitroprusside (iron coordination compound, atom degree 6 unsupported) → zeros used
- MODEL_DOSE uses `multidcp.py` (original concat), not `multidcp_balanceloss.py` (MoE)
- Gene tensor: pass [num_gene, 128] directly — model adds batch dim internally
- MolFormer: transformers 5.x rotary embedding NaN bug fixed by re-init before .to(device)

Note: Phase 0 was completed via the `writing-plans` + `executing-plans` workflow before GSD took over. It is a pre-GSD prerequisite; GSD manages Phases 1–6 only.

## Phase 0 Deliverables (pre-GSD, commit e281e9f)

Phase 0 landed the leakage-filtered data foundation required by Phases 1 and 2:

- `multihead_dili/` subdir skeleton inside the umbrella repo
- `CLAUDE.md` + `MANIFEST.md` skeleton
- `.planning/` GSD files (PROJECT.md, ROADMAP.md, MILESTONES.md, STATE.md)
- `src/data/scaffold_split.py` + `src/data/upstream_filter.py` (copied from v0.5, same Murcko-scaffold logic)
- `tests/test_data_paths.py` — E-Hill + DILIst path sanity tests
- `scripts/build_dili_scaffold_split.py` → produced `data/processed/dili_split.json` (train/val/test pert_ids + scaffolds_in_test list; test ≈ 150–170 drugs)
- `scripts/run_leakage_filter.py` → produced `data/processed/ehill_train_safe.parquet`, `data/processed/lincs_train_safe.parquet`, `data/processed/leakage_report.md` (Murcko-scaffold + drug_name exclusion of DILIst test drugs from Stage-1 training data)
- `data/processed/dili_split.json` — canonical scaffold split artifact; `scaffolds_in_test` is the load-bearing exclusion set for leakage discipline

**Leakage discipline verified:** Test scaffolds from DILIst are excluded from E-Hill train and LINCS PDG-filtered train sets. MODEL_DOSE and MODEL_GEX will never see DILIst test-set scaffolds during Stage-1 training.

## Phase 4 Deliverables (2026-05-20)

- `src/stage2/classifiers.py` — LinearHead, MLP1Head, MLP2Head, build_head factory
- `src/stage2/train_dili_classifier.py` — 630-run grid driver (7 variants × 3 heads × 2 splits × 5 folds × 3 seeds)
- `src/stage2/generate_p4_summary.py` — HG3 check + summary generator
- `data/processed/predictions/*.parquet` — 630 prediction files (90 per variant)
- `data/processed/P4_runs.parquet` — 630 rows, one per run, with AUROC/AUPRC/MCC/bal_acc
- `results/tables/P4_ablation_summary.md` — per-variant per-head mean ± std AUROC + HG3 verdict
- `.planning/phases/04-dili-consumer-ablation/04-CONTEXT.md` — auto-generated context
- `.planning/phases/04-dili-consumer-ablation/04-01/02/03-PLAN.md` — 3 plans

Key findings from Phase 4:
- Wall-clock: ~75 minutes (630 runs, 6 workers, CPU-only)
- HG3 PASS: embed-only (var1, MolFormer) random AUROC = 0.6536 ± 0.0250 >> 0.55 threshold
- embed-only is clearly the strongest single pathway (scaffold: 0.5955 >> gex: 0.5187, dose: 0.4779)
- all-three (var7, headline): scaffold AUROC = 0.5911 — NOT substantially above embed-only (0.5955)
- Multi-pathway story: minimal synergy on scaffold-novel split; Phase 5 will test with DeLong + CIs
- pert_id format mismatch fixed: parquet uses int IDs, dili_split.json uses DILIST_XXXX strings

## Blockers/Concerns

None. Phase 5 can proceed immediately.

Note: HG4 (var7 AUROC >= best single + 0.01 on scaffold split) is borderline — var7=0.5911 vs embed-only=0.5955 (var7 is LOWER). This could fire HG4 in Phase 5. Phase 5 will compute DeLong p-values and bootstrap CIs to determine if the difference is statistically significant.

Remember: always leave one GPU free (shared box constraint; see `CLAUDE.md` hard rules).
