---
phase: 01-eda-the-bracket
verified: 2026-06-23T00:00:00Z
status: passed
score: 9/9 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: none
  note: "Initial verification of Phase 1, focused on the 01-08 gap-closure (un-halt) plan."
operational_caveats:
  - "run_p1_eda.py all rewrites HALT_REASON.md via _write_halt_reason on every run, so the appended RESOLUTION note (D-10) is volatile. The durable un-halt record is the project decision in STATE.md / 01-CONTEXT.md (D-10). This is expected, not a failure."
---

# Phase 1: EDA (the bracket) Verification Report

**Phase Goal:** Bound the achievable result before any model runs — establish the structure-only floor, the measured-biology ceiling, and the gap between them (plus region distinguishability and human↔rodent basal concordance). Deliverable: results/tables/P1_eda.md. Triggers Halt Gate 2 when the floor-ceiling gap is not significantly positive.

**Verified:** 2026-06-23
**Status:** passed
**Re-verification:** No — initial verification (focus: 01-08 gap-closure / un-halt)

## Framing held during verification

The Halt Gate 2 firing is INTENDED and CORRECT. The PRIMARY drug-level gate fires (gap −0.1770, CI [−0.3161, −0.0326]) and `run_p1_eda.py all` exits 1 BY DESIGN. The 2026-06-23 gate review (D-08/D-09/D-10) ruled the halt LIFTS by completing the honest 2×2 bracket documentation regardless of the gap sign. This verification confirms the gap-closure (01-08) delivered the bracket documentation and recorded the un-halt — it does not re-litigate the gate.

## Goal Achievement

### Observable Truths (01-08 must_haves)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | D-08: halt lifts by documentation; PRIMARY drug-level gate still fires, driver still exits 1 | ✓ VERIFIED | `gate_fires`/`sys.exit(1)` intact (driver L1399-1415); P1_eda.md "Halt Gate 2 \| **FIRES**" gap −0.1770; HALT_REASON RESOLUTION states gate still fires |
| 2 | D-09: floor-leaky cell added (random StratifiedKFold, NO groups), completes 2×2; no negative-set expansion | ✓ VERIFIED | `floor_profile_leaky_auroc` body contains `StratifiedKFold`, no `groups=`; 2×2 table in P1_eda.md (Floor leaky 0.9989); no DILIrank-union/relax-Ambiguous/scaffold code added |
| 3 | D-09: 95% paired-bootstrap CI on profile-disjoint gap from aligned floor/ceiling OOF on same rows + folds | ✓ VERIFIED | Driver L723-741: `floor_oof`, `de_kept`=`de_aligned[kept_idx]`, `ceil_oof` via StratifiedGroupKFold seed=42, `pd_boot=paired_bootstrap_auroc_gap(...)`; P1_eda.md gap +0.0590, CI [0.0259, 0.0921], 10,000/10,000 |
| 4 | D-10: dated RESOLUTION note appended to HALT_REASON.md; original halt + gate_fires:** True preserved | ✓ VERIFIED | HALT_REASON.md has "## RESOLUTION (2026-06-23)" AND original "gate_fires:** True" (2 occurrences) AND "Decision per D-02" block |
| 5 | Regression test locks floor-leaky as NON-grouped (leaky AUROC > disjoint AUROC, leaky > 0.75) | ✓ VERIFIED | `test_floor_profile_leaky_exploits_drug_identity` asserts `leaky > disjoint` and `leaky > 0.75`; passes |
| 6 | All four 2×2 cells on consistent profile set; EDA-02 +0.31, 01-07 head-to-head, D-07 note, PRIMARY gate, Power, Interpretation caveat, _write_halt_reason/gate_fires/sys.exit(1) preserved | ✓ VERIFIED | P1_eda.md retains "Leakage decomposition (Phase 1 headline)", "Profile-level drug-disjoint head-to-head", "Leakage-discipline guidance (D-07)", "FIRES", Power, Interpretation caveat; driver tokens unchanged |
| 7 | floor_profile_disjoint_oof returns OOF vector matching the disjoint scalar | ✓ VERIFIED | `floor_profile_disjoint_auroc` delegates to `floor_profile_disjoint_oof`; `test_floor_profile_disjoint_oof_matches_scalar` passes (within 1e-9) |
| 8 | Both new functions in __all__ | ✓ VERIFIED | `__all__` includes `floor_profile_disjoint_oof`, `floor_profile_leaky_auroc` |
| 9 | Scope boundary held (no model/training code, no SMILES re-resolution, no other organs) | ✓ VERIFIED | files_modified limited to floor.py, run_p1_eda.py, test_eda_floor.py, P1_eda.md, HALT_REASON.md; SUMMARY "Deviations: None" |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/spatial/eda/floor.py` | floor_profile_leaky_auroc (no groups, seed=42) + floor_profile_disjoint_oof, both in __all__ | ✓ VERIFIED | Both defs present (1 each), in __all__, leaky body has StratifiedKFold and no `groups=` |
| `scripts/run_p1_eda.py` | floor-leaky wiring, 2×2 table, profile-disjoint CI, un-halt writer | ✓ VERIFIED | All tokens present; parses; data flow kept_idx→de_kept→ceil_oof→pd_boot intact; PRIMARY gate/exit unchanged |
| `tests/spatial/test_eda_floor.py` | regression test locking leaky as non-grouped | ✓ VERIFIED | `floor_profile_leaky` test present and GREEN |
| `results/tables/P1_eda.md` | completed 2×2 + diagnostic + profile-disjoint CI, preserving prior sections | ✓ VERIFIED | All required + preserved sections present; 2×2 fully populated with real numbers |
| `.planning/.../HALT_REASON.md` | appended dated RESOLUTION note, original preserved | ✓ VERIFIED | RESOLUTION + original gate_fires:** True both present (volatile per caveat) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| driver head-to-head (fps_prof/y_kept) | floor.py floor_profile_leaky_auroc | random StratifiedKFold on kept set | ✓ WIRED | L685 call on fps_prof, y_kept |
| floor_profile_disjoint_oof + ceil_oof (kept subset, same folds) | bootstrap.py paired_bootstrap_auroc_gap | aligned per-profile OOF → 95% CI | ✓ WIRED | L723-741; de_kept=de_aligned[kept_idx]; ceil_oof StratifiedGroupKFold seed=42 |
| driver 2×2/un-halt writer | HALT_REASON.md | appended dated RESOLUTION note | ✓ WIRED (volatile) | RESOLUTION present; rewritten by _write_halt_reason on each run |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| P1_eda.md 2×2 table | floor_auroc_profile_leaky (0.9989) | floor_profile_leaky_auroc(fps_prof, y_kept) | Yes (real ECFP4 on 2648 profiles) | ✓ FLOWING |
| P1_eda.md CI table | pd_boot.gap_observed/ci (+0.0590, [0.0259,0.0921]) | paired_bootstrap_auroc_gap(y_kept, floor_oof, ceil_oof) | Yes (10,000/10,000 valid resamples) | ✓ FLOWING |
| P1_eda.md PRIMARY gate | bootstrap_result (−0.1770) | unchanged drug-level paired bootstrap | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| floor.py exposes leaky fn (random KFold, no groups) | python AST-slice check | StratifiedKFold=True, groups=False | ✓ PASS |
| floor tests GREEN (leaky>disjoint, oof matches, no regression) | pytest test_eda_floor.py | 4 passed | ✓ PASS |
| driver parses | ast.parse(run_p1_eda.py) | parses ok | ✓ PASS |
| full spatial suite (regression) | pytest tests/spatial/ | 141 passed | ✓ PASS |
| P1_eda.md carries 2×2 + diagnostic + CI + preserved sections | grep checks | all present | ✓ PASS |
| HALT_REASON RESOLUTION + original preserved | grep checks | both present | ✓ PASS |

Note: the live `run_p1_eda.py all` run (exit 1 by design) was already performed by the executor during 01-08 Task 3; the regenerated P1_eda.md (mtime 23:36) and HALT_REASON.md (23:37) are the fresh driver outputs verified here. The exit-1 logic was confirmed by static inspection (L1399-1415) rather than re-run to avoid overwriting the appended RESOLUTION note (volatile per caveat).

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| EDA-01 | 01-01,02,03,06,07,08 | Label + structure brackets, structure-only floor | ✓ SATISFIED | floor (LR+RF ECFP4), profile-disjoint + leaky floor in P1_eda.md |
| EDA-02 | 01-01,04,06,08 | Measured-biology ceiling | ✓ SATISFIED | ceiling AUROC, +0.31 leakage decomposition, 2×2 ceiling cells |
| EDA-03 | 01-01,05,06,07 | Region + cross-species diagnostics | ✓ SATISFIED | region/cross-species diagnostics + D-07 guidance in P1_eda.md (per REQUIREMENTS traceability) |

No orphaned requirements: every EDA-01/02/03 ID mapped to this phase appears in at least one 01-* plan's `requirements` field.

### Anti-Patterns Found

None blocking. No TODO/FIXME/placeholder/stub patterns in the modified code paths. The 2×2 table cells and CI table render real computed values (not hardcoded empties). The `return null`/empty-data patterns are absent from the new functions.

### Human Verification Required

None. All must-haves are programmatically verifiable (pure-library AUROC functions, deterministic seed=42 outputs, markdown report content, test assertions). No visual/real-time/external-service behavior is in scope for this documentation-completion phase.

### Gaps Summary

No gaps. The 01-08 gap-closure delivered exactly what D-08/D-09/D-10 specified:
- The missing floor-leaky cell completes the floor×ceiling × leaky×disjoint 2×2 (Floor leaky 0.9989 ≈ Ceiling leaky 0.9120 — the cell that "explains the paper").
- The profile-disjoint gap (+0.0590) carries a 95% paired-bootstrap CI [0.0259, 0.0921] from aligned OOF vectors on identical rows + drug folds, documented as honesty not a gate.
- The un-halt is recorded (RESOLUTION note in HALT_REASON.md; canonical durable record in STATE.md/01-CONTEXT.md D-10).
- The PRIMARY drug-level gate still FIRES (−0.1770) and the driver still exits 1 — preserved unchanged, as intended.
- Scope boundary held; 141 spatial tests pass with no regression.

One operational caveat (not a failure): `run_p1_eda.py all` rewrites HALT_REASON.md every run, so the appended RESOLUTION note is volatile. The durable un-halt record is the D-10 project decision in STATE.md / 01-CONTEXT.md.

---

_Verified: 2026-06-23_
_Verifier: Claude (gsd-verifier)_
