---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: multihead_dili_v1
status: executing
stopped_at: Phase 3 complete (EMBED-04 PASS); Phase 4 (scaffold split + Stage-3 classifier) next
last_updated: "2026-05-20T08:20:00Z"
last_activity: 2026-05-20 -- Phase 3 complete (commits 6e0bbdb + 0b5fe40); dili_features.parquet written (1118 drugs, 1688-dim, 0 NaN); all 9 CI tests pass
progress:
  total_phases: 6
  completed_phases: 3
  total_plans: 9
  completed_plans: 5
  percent: 50
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
Next: Phase 4 (Stage-3 downstream DILI classifier)

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

## Blockers/Concerns

None. Both Stage-1 training sets (`ehill_train_safe.parquet`, `lincs_train_safe.parquet`) are ready. Phase 1 and Phase 2 can begin in parallel on separate GPUs.

Remember: always leave one GPU free (shared box constraint; see `CLAUDE.md` hard rules).
