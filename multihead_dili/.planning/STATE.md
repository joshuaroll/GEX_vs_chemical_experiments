---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: multihead_dili_v1
status: executing
stopped_at: Phase 2 complete (HG2 PASS); Phase 3 unblocked (pending Phase 1 completion)
last_updated: "2026-05-20T06:30:00Z"
last_activity: 2026-05-20 -- Phase 2 complete (commit 92cc932); MODEL_GEX HG2 PASS (mean Pearson=0.3568); Phase 1 still in progress concurrently
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 9
  completed_plans: 3
  percent: 17
---

# Project State

## Project Reference

See: `multihead_dili/.planning/PROJECT.md`
See: `/raid/home/joshua/projects/0_project_documents/multihead_multidcp_dili_three_pathway_05192026.md`
See: `/raid/home/joshua/projects/0_project_documents/multihead_multidcp_dili_implementation_plan_05192026.md`

## Current Position

Phase 2: COMPLETE — MODEL_GEX training (HG2 PASS, mean Pearson=0.3568)
Phase 1: IN PROGRESS concurrently (MODEL_DOSE on GPU 0; separate subagent)
Next: Phase 3 (MolFormer + Stage-2 feature caching) — BLOCKED until Phase 1 also complete (needs chkpt_dose.pt)

Note: Phase 0 was completed via the `writing-plans` + `executing-plans` workflow before GSD took over. It is a pre-GSD prerequisite; GSD manages Phases 1–6 only. `completed_phases: 0` reflects GSD's accounting (0 of 6 GSD-managed phases done).

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
