# Deferred items — Phase 02 / plan 02-02

## Out-of-scope test collection errors (other plans own these)
- `tests/spatial/test_tox_head.py` imports `src.spatial.tox_head` (WIRE-02, plan 02-03) — module not yet created. Collection error is expected until 02-03.
- `tests/spatial/test_apap_validation.py` imports `src.spatial.apap_validation` (WIRE-03, plan 02-04) — module not yet created. Collection error is expected until 02-04.

These are NOT introduced by 02-02; they are the RED contract files for later plans in this phase. 02-02 runs its pure suite with `--ignore` on both.
