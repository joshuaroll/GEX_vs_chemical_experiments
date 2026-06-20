---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-06-20T23:45:00Z"
last_activity: 2026-06-20
progress:
  total_phases: 8
  completed_phases: 0
  total_plans: 4
  completed_plans: 2
  percent: 50
---

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
- **Plan:** 03 COMPLETE; on to Plan 04
- **Status:** Executing (Plans 01 + 03 done; Plan 02 pending; Plan 04 pending)
- **Progress:** `[####                ] 0/8 phases complete (P0 plan 2/4 done)`

**Next action:** Execute Plan 04 (MANIFEST.md + P0_orthologs.md + P0_coverage.md + squidpy)

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

### Plan 01 decisions (locked)

- **slug field is single source of truth** for `data/raw/spatial/<slug>/` directory. Download driver and test suite must read `entry.slug`; no independent slugification.
- **test_raw_datasets_present** skips when RAW dir has no subdirectories (`.gitkeep` alone is not a trigger).
- **Stale accessions corrected** in `src/spatial/datasets.py`: Yu→figshare 22321447, Maynard→spatialLIBD, Lake/KPMP→GSE183456+GSE183279, Siletti→snRNA-seq.
- **Rodent basal-context candidates registered**: mouse liver (GSE272564 control arm), kidney (GSE252772), brain (GSE233983). Final healthy-spot confirmation at download (Pitfall 5).

### Plan 03 decisions (locked)

- **Ensembl release 116, query date 2026-06-20** — baked into raw TSV filename `orthologs_raw_116_20260620.tsv` (XC-10).
- **92.75% dropped fraction is expected** — BioMart returns all human genes including those with no homolog; strict mutual one2one yields ~15k of ~220k rows.
- **build_one2one_orthologs() accepts DataFrame, Path, or str** — offline testable against fixture without network.
- **Release probe regex** fixed to match `ensembl_mart_116` format (original matched `_gene_ensembl_NN`).

### Todos / watch items

- `squidpy` not yet installed in `dili_v04_env`; required for Moran's I spatial-QC. Install in Plan 02 or 04.
- Download driver `scripts/download_spatial.py` not yet created (Plan 02).
- MANIFEST.md not yet created (Plan 04).
- P0_orthologs.md: paste `ortholog_report` output (15956 one2one, 92.75% dropped). Plan 04 deliverable.
- P0_coverage.md: per-dataset coverage vs 10716 genes. Plan 04 deliverable.
- `src/spatial/region_signature.py` holds the `NotImplementedError` seam — must be replaced with the real frozen-checkpoint call in Phase 2, not before.
- Extend `src/spatial/` (127 passing fixture tests now); do not rebuild. New files: tox_head.py [P2], splits.py [P3], train.py/eval.py [P4].

### Blockers

- None currently.

## Session Continuity

- **Last activity:** 2026-06-20
- **Stopped at:** Plan 03 COMPLETE (00-03-SUMMARY.md written)
- **Resume with:** Execute Plan 04 (MANIFEST.md, P0_orthologs.md, P0_coverage.md, squidpy install)
