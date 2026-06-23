---
phase: 01-eda-the-bracket
plan: 06
subsystem: eda
tags: [dili, lincs, ecfp4, logistic-regression, group-cv, leakage, bootstrap, halt-gate, spatial]

requires:
  - phase: 01-eda-the-bracket (plans 01-05)
    provides: EDA libraries (labels, smiles_join, fingerprints, floor, ceiling, bootstrap, region_diagnostics) + Nyquist tests
provides:
  - "scripts/run_p1_eda.py: end-to-end P1 EDA driver (floor/ceiling/diagnostics/all) on real data with Halt Gate 2"
  - "results/tables/P1_eda.md: Phase 1 deliverable (floor/ceiling/gap+CI, leakage decomposition, power, region + cross-species diagnostics)"
  - "HALT_REASON.md: Gate 2 fired -> stop-and-REFRAME (D-02)"
  - "Reproducible leakage decomposition: Wang/Li-style benchmark is ~+0.31 AUROC drug-leakage-inflated"
affects: [phase-2-reframe, ceiling-methodology, benchmark-leakage]

tech-stack:
  added: [sklearn.StratifiedGroupKFold, sklearn.StandardScaler pipeline, Hanley-McNeil AUROC SE]
  patterns: ["drug-grouped (leakage-free) CV for any per-drug-multi-profile evaluation", "reproducible leakage decomposition reported in deliverable"]

key-files:
  created:
    - scripts/run_p1_eda.py
    - results/tables/P1_eda.md
  modified:
    - src/spatial/eda/ceiling.py
    - tests/spatial/test_eda_ceiling.py
    - .planning/phases/01-eda-the-bracket/HALT_REASON.md

key-decisions:
  - "Ceiling CV must be drug-grouped (StratifiedGroupKFold) + per-fold StandardScaler; profile-level CV leaked drug identity (one drug = up to 784 profiles)"
  - "Halt Gate 2 fired (gap -0.177, 95% CI [-0.316,-0.033]); user decision = stop-and-REFRAME (D-02)"
  - "Reframe headline = the benchmark is drug-leakage-inflated (+0.31 AUROC), NOT 'structure beats biology'"

patterns-established:
  - "Leakage-free drug-grouped evaluation: never split a drug's profiles across train/test"
  - "Report leakage decomposition + Hanley-McNeil power alongside any small-n AUROC gap"

requirements-completed: [EDA-01, EDA-02, EDA-03]

duration: ~55min (incl. checkpoint investigation)
completed: 2026-06-23
---

# Phase 1 Plan 06: P1 EDA Driver + Halt Gate 2 Summary

**End-to-end P1 EDA on real data: structure floor 0.61, leakage-free measured ceiling 0.43 (drug-level), gap -0.177 -> Halt Gate 2 FIRES; the robust finding is a +0.31 AUROC drug-leakage inflation of the Wang/Li-style benchmark, and the decision is stop-and-REFRAME (D-02).**

## Performance

- **Tasks:** 3 (2 auto + 1 human-verify checkpoint)
- **Files modified:** 5 (driver, report, ceiling lib, ceiling test, HALT_REASON)
- **Completed:** 2026-06-23

## Accomplishments
- `scripts/run_p1_eda.py` wires all six EDA libraries and runs end-to-end on the real on-disk data (DILIrank, Wang/Li measured DE, yu2022 liver Visium, GSE272564 mouse APAP0h control, PDG manifold) under `dili_v04_env`, with MANIFEST provenance assertion (D-05).
- `results/tables/P1_eda.md` produced with: structure floor (EDA-01), measured ceiling (EDA-02), floor-ceiling gap + 95% paired-bootstrap CI, **leakage decomposition**, **Hanley-McNeil power**, region distinguishability + human-rodent basal concordance (EDA-03).
- Halt Gate 2 evaluated reproducibly from the BootstrapResult; HALT_REASON.md written + exit 1 (D-02).
- **Checkpoint investigation corrected a leakage artifact** in the ceiling and surfaced the real headline finding.

## Task Commits

1. **Task 1 (auto): floor+ceiling+gap, P1_eda.md, Halt Gate 2** - `e5b56d9` (feat)
2. **Task 2 (auto): region + cross-species diagnostics** - `2557702` (feat)
3. **Task 3 (checkpoint:human-verify): Halt Gate 2 interpretation** - resolved via investigation + user decision (below)

**Post-checkpoint fixes (from the human-verify investigation):**
- `e1deeff` (fix): leakage-free drug-grouped ceiling CV; corrected Halt Gate 2 (+ RED->GREEN leakage regression test)
- `3e52a6d` (feat): reproducible leakage decomposition + power analysis; reframe-focused HALT_REASON

## Decisions Made
- **Ceiling was leakage-contaminated.** The original ceiling (0.56) used profile-level `StratifiedKFold(shuffle=True)`, but one drug carries up to 784 profiles, so a drug's profiles straddled train/test and the model memorized drug identity. Fixed to `StratifiedGroupKFold` over compound + per-fold `StandardScaler` (unit-consistent with the drug-level floor). Honest ceiling = 0.43.
- **Gate fires, but framed honestly.** Corrected gap -0.177, 95% CI [-0.316, -0.033]. Underpowered (38 negative drugs; conservative MDE 0.198 > 0.177); significance rests on the paired bootstrap. The honest profile-level measured ceiling (0.605) is comparable to the structure floor (0.611).
- **User decision (D-02): stop-and-REFRAME.** Headline = the benchmark is drug-leakage-inflated (+0.31 AUROC); measured DE does not generalize to held-out drugs above structure on this set. Do NOT proceed to Phase 2 model training.

## Deviations from Plan

The plan's Task 3 human-verify checkpoint worked exactly as designed: it caught that the measured ceiling (0.56) was implausibly far below Wang/Li's ~0.798 benchmark. Investigation found the leakage bug, which was fixed; the ceiling and gate were recomputed honestly. This is the checkpoint doing its job, not a plan defect.

### Auto-fixed issues (during Tasks 1-2, by the executor)
1. **[Rule 1] `load_dilist` column name** (`CompoundName` vs `Compound`) - `e5b56d9`
2. **[Rule 1] `build_one2one_orthologs` wrong input file** (raw BioMart TSV, not pre-filtered) - `e5b56d9`
3. **[Rule 1] AnnData duplicate gene names** (`var_names_make_unique()`) - `e5b56d9`
4. **[Rule 1] `svg_retention` denominator** (computed inline with correct denominator) - `2557702`

### Post-checkpoint correction (leakage)
5. **[Investigation] Drug leakage in ceiling CV** - replaced profile-level StratifiedKFold with drug-grouped StratifiedGroupKFold + StandardScaler; added `test_ceiling_no_drug_leakage` (RED->GREEN). `e1deeff`, `3e52a6d`

**Impact:** corrections were necessary for a trustworthy gate decision. No scope creep.

## Issues Encountered
- Measured ceiling far below benchmark (0.56 vs ~0.798) -> traced to drug-identity leakage in profile-level CV; resolved (see decisions). Profile-level reproduction is 0.912 leaky / 0.605 drug-disjoint.

## Verification
- `pytest tests/spatial/` -> 138 passed (incl. new leakage regression).
- `run_p1_eda.py all` runs end-to-end on real data; gate decision reproducible (seed=42, exit 1).

## Next Phase Readiness
- **HALT GATE 2 FIRED -> Phase 2 is BLOCKED pending reframe (D-02).** Do not execute Phase 2 model wiring.
- Reframe should center on: (1) the drug-leakage inflation of the benchmark, (2) a properly powered drug-disjoint comparison (expand the negative-drug set), (3) whether the unit of analysis should be profile-level-with-drug-disjoint-splits.

---
*Phase: 01-eda-the-bracket*
*Completed: 2026-06-23 (Halt Gate 2 fired -> reframe)*
