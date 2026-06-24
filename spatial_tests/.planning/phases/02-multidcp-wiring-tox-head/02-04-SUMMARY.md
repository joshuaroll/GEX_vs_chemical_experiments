---
phase: 02-multidcp-wiring-tox-head
plan: 04
subsystem: validation
tags: [apap, halt-gate-3, per-zone-pearson, ortholog-align, stop-and-reframe, wire-03]
provides:
  - "src/spatial/apap_validation.py: assign_zones (D-07), measured_zone_de (APAP-ctrl, ortholog-aligned, present_mask, D-08), zone_pearson (<2 guard), halt_gate_3_fires (pericentral<0.3, D-08/D-09)"
  - "scripts/run_apap_validation.py: GSE272564 arms (APAP0h ctrl vs APAP24h injury, Pitfall 6) vs WIRE-01 mouse cache -> per-zone Pearson -> Halt Gate 3"
  - "results/tables/P2_apap_validation.md: per-zone Pearson table, intersection sizes, GSE280652 partial-replication note, rodent-anchor caveat, gate verdict"
  - "HALT_REASON.md: Halt Gate 3 FIRED (pericentral r=0.0166<0.3) -> stop-and-REFRAME (D-09)"
affects: [P3-splits, P4-per-organ-train, P5-translatability]
tech-stack:
  added: []
  patterns: [ortholog-mapped intersection-flagged Pearson (region_diagnostics analog), present_mask flag-not-zero, run_p1_eda driver+halt-writer mirror, GSM-arm split (Pitfall 6)]
key-files:
  created: [src/spatial/apap_validation.py, scripts/run_apap_validation.py, results/tables/P2_apap_validation.md, .planning/phases/02-multidcp-wiring-tox-head/HALT_REASON.md]
  modified: []
key-decisions:
  - "GSE272564 matched arms = APAP0h (pre-injury baseline / control) vs APAP24h (classical pericentral-necrosis injury timepoint); measured DE = APAP24h_zone - APAP0h_zone (D-06)"
  - "Halt Gate 3 FIRES and that is the correct, designed outcome (D-09 stop-and-REFRAME) — frozen backbone near-zonal-invariant per 02-02; gate NOT altered to force a pass"
metrics:
  duration: ~35min
  completed: 2026-06-24
requirements: [WIRE-03]
---

# Phase 2 Plan 04: WIRE-03 APAP per-zone Pearson + Halt Gate 3 Summary

**The predicted region-resolved DE for acetaminophen does not recover the measured pericentral APAP injury pattern (pericentral Pearson 0.0166 < 0.3 over a 9,272-gene ortholog intersection), so Halt Gate 3 FIRES — the project's designed stop-and-REFRAME outcome (D-09), driven by the frozen backbone's near-zonal-invariance flagged in 02-02, not a bug.**

## Halt Gate 3 verdict

| Zone | predicted-vs-measured Pearson r | p-value | n genes compared |
|------|---------------------------------|---------|------------------|
| pericentral (gating zone) | +0.0166 | 1.10e-01 | 9,272 |
| periportal | +0.0038 | 7.14e-01 | 9,272 |

**Pericentral Pearson 0.0166 < 0.3 -> Halt Gate 3 FIRES.** Driver exited 1 (execution success: a firing gate is the correct outcome here). HALT_REASON.md written with trigger, per-zone table, decision = stop-and-REFRAME, dated.

## Performance
- **Duration:** ~35 min
- **Tasks:** 2 / 2 complete
- **Files created:** 4 (source module, driver, report, HALT_REASON); **modified:** 0

## Accomplishments
- `src/spatial/apap_validation.py`: `assign_zones` (published annotation else canonical markers Glul/Cyp2e1 pericentral, Sds/Cyp2f2 periportal, D-07); `measured_zone_de` (APAP_zone - ctrl_zone in mouse symbols, mouse->human via the one2one ortholog table, reindexed to the 10,716 PDG order with `align_to_gene_space(missing="nan")` -> `present_mask`, flag-not-zero D-08); `zone_pearson` (present-only Pearson, `<2` ValueError mirroring region_diagnostics); `halt_gate_3_fires` (pericentral-keyed `<0.3`, raises `KeyError` on a mis-keyed gate per T-02-12).
- Wave-0 `test_apap_validation.py` turned RED -> GREEN by source only (3 passed); full pure spatial suite 150 passed, 0 regressions.
- `scripts/run_apap_validation.py`: splits GSE272564 by GSM (APAP0h control vs APAP24h injury — Pitfall 6, arms never mixed), zones each arm by markers, pseudobulks per zone, reads acetaminophen predicted DE from the WIRE-01 rule-B mouse cache (`region_de_cache/mouse`), computes per-zone Pearson over the 9,272-gene intersection, fires Halt Gate 3, writes the report + HALT_REASON.md, mirrors the run_p1_eda halt-writer + `sys.exit(1)` pattern. MANIFEST provenance asserted for both APAP tars + the gene-order + ortholog files.
- `results/tables/P2_apap_validation.md`: per-zone Pearson table (all zones), per-zone spot counts, the GSE280652 partial-replication note (no matched control, D-06 amendment), the rodent-anchor-for-a-human-headline caveat (D-09), the pattern-not-magnitude caveat (D-05), and the near-zonal-invariance mechanism note.

## Scientific reading
The zonation is biologically coherent: the control arm (APAP0h) is pericentral-dominant (1146 pericentral vs 332 periportal spots) and the injury arm (APAP24h) shifts periportal-dominant (1374 vs 278), consistent with pericentral hepatocyte loss after APAP. The measured side therefore carries a real pericentral injury gradient. The predicted side does not: the cached per-zone predicted DE vectors are near-identical across zones (~float32 epsilon, 02-02 flag) because the frozen CheMoE cell-context encoder contributes little for healthy-liver basals, so the prediction is dominated by drug + dose and has essentially no pericentral-vs-periportal contrast to match. Hence Pearson ~0 and the gate fires. This is the instrumented OOD hypothesis surfacing as the result.

## Task Commits
1. **Task 1: apap_validation.py source (RED->GREEN)** — `87fe9d0`
2. **Task 2: APAP driver + report + HALT_REASON (gate FIRES)** — `a42ae95`

## Deviations from Plan

**1. [Rule 3 - blocking interpretation] GSE272564 matched-arm definition.**
- **Found during:** Task 2 (loading the tar).
- **Issue:** The plan/D-06 say "GSE272564 carries matched control + APAP arms"; the tar is actually an APAP time course (APAP0h / APAP3h / APAP6h / APAP24h), not a literal "control" + "APAP" pair.
- **Resolution:** APAP0h is the pre-injury baseline (the matched control arm — same tissue, time 0); APAP24h is the classical pericentral-necrosis injury timepoint. Measured DE = APAP24h_zone - APAP0h_zone. This honors D-06's intent (matched treated-minus-control on one series) with the timepoints the data actually provides. Documented in the driver, report, and HALT_REASON.
- **Files:** `scripts/run_apap_validation.py`. **Commit:** `a42ae95`.

No bugs auto-fixed; no architectural changes; no auth gates. The gate firing is the expected outcome, NOT a deviation — the gate was not altered, no zonal variation was fabricated, exit 1 is execution success per the plan.

## Known Stubs
None. The predicted side is real frozen-inference cache; the measured side is real GSE272564 Visium. No hardcoded/placeholder values flow into the Pearson; absent genes are flagged, not zero-filled.

## Self-Check: PASSED
- Files FOUND: `src/spatial/apap_validation.py`, `scripts/run_apap_validation.py`, `results/tables/P2_apap_validation.md`, `.planning/phases/02-multidcp-wiring-tox-head/HALT_REASON.md`.
- Commits FOUND in git log: `87fe9d0`, `a42ae95`.
- `test_apap_validation.py` 3 passed; pure spatial suite 150 passed; driver exit 1 (gate fires, expected).

## Next Phase Readiness
- Halt Gate 3 has fired. Per D-09 this is a **stop-and-REFRAME**, mirroring P1's precedent — surface to the gate-review/orchestrator before P3. The spatial claim reframes around the frozen backbone's near-zonal-invariance: predicted region-resolved DE does not recover the pericentral APAP pattern under the frozen model, which is itself the reportable finding. A region-sensitive signature (e.g. unfreezing or re-conditioning the cell-context encoder) is the open question this gate raises for later phases.
