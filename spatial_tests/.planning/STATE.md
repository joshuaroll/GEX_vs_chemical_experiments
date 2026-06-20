# STATE: Spatial Cross-Species Toxicity Prediction (MultiDCP-CheMoE)

> Living memory across sessions. Read this first.

## Project Reference

- **Project:** Spatial Cross-Species Toxicity Prediction (MultiDCP-CheMoE)
- **Root:** `/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests/` (subdir of umbrella repo `GEX_vs_chemical_experiments`; no own `.git`).
- **Core value:** Does a predicted, region-resolved molecular response signature predict organ-specific drug toxicity better than chemical structure alone, and does that signal translate across species (rodent → human)?
- **Current focus:** Phase 0 — Dataset acquisition & MANIFEST.
- **Isolation note:** This is a standalone GSD project. NEVER read/write `/raid/home/joshua/.planning` (separate, halted "liver" v0.5 project).

## Current Position

- **Phase:** 0 — Dataset acquisition & MANIFEST (P0)
- **Plan:** none yet
- **Status:** Phase 0 ready to plan
- **Progress:** `[                    ] 0/8 phases complete`

**Next action:** `/gsd-plan-phase 0`

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases total | 8 |
| Phases complete | 0 |
| Requirements total | 19 |
| Halt gates | 6 (phases 0,1,2,3,4,6) |

## Accumulated Context

### Decisions (locked / design)
- Frozen MultiDCP/CheMoE baseline; only the downstream tox head trains.
- Spatial is an additive comparison arm (S-B/S-C/S-F vs cell-line B/C), not a replacement.
- Visium whole-transcriptome only as basal input; targeted panels are annotation-only.
- Published region annotations default; per-organ only; fixed 3-layer concat-MLP; dose-response is a first-class third channel and confound control.
- One-to-one orthologs only; report dropped fraction.

### Proposed decisions (PENDING professor sign-off — confirm at phase planning)
1. **Spatial DE-rule divergence** `predicted_treated(drug, region_basal) - region_basal` (vs parent `treated - diseased`). Confirm at P2/P4.
2. **APAP single-drug Visium anchor** (GSE280652/GSE272564) as primary direct validity stand-in. Confirm at P2.
3. **Negative result is publishable** (stop-and-reframe vs stop-and-abandon). Confirm before P1/P4 gates are acted on.

(Operational sign-offs handled in P0: approve adding `squidpy`/Tangram to env; sign off dataset-accession corrections before download scripts.)

### Todos / watch items
- Apply corrected accessions before writing download scripts: Yu 2022 liver = Figshare 10.6084/m9.figshare.17058105 (NOT GSE189994); Lake/KPMP kidney = GSE183456 + GSE183279 (GSE211785 = Abedini 2024 substitute); Maynard DLPFC = spatialLIBD (NOT GSE144239); no public human-brain MERFISH exists.
- `squidpy` not yet installed in `dili_v04_env`; required for Moran's I spatial-QC.
- `src/spatial/region_signature.py` holds the `NotImplementedError` seam — must be replaced with the real frozen-checkpoint call in Phase 2, not before.
- Extend `src/spatial/` (124 passing fixture tests); do not rebuild. New files: orthology.py [P0], tox_head.py [P2], splits.py [P3], train.py/eval.py [P4].

### Blockers
- None currently.

## Session Continuity

- **Last activity:** Project initialized from ingest synthesis (PROJECT.md, REQUIREMENTS.md, ROADMAP.md, STATE.md written). 2026-06-20.
- **Resume with:** `/gsd-plan-phase 0`
