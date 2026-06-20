---
phase: 0
slug: dataset-acquisition-manifest
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-20
---

# Phase 0 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `00-RESEARCH.md` § Validation Architecture (accessions verified live 2026-06-20).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 (installed in `dili_v04_env`) |
| **Config file** | none — markers registered in `spatial_tests/conftest.py` (`network` marker exists) |
| **Quick run command** | `conda run -n dili_v04_env python -m pytest tests/test_data_paths.py -x -q` |
| **Full suite command** | `conda run -n dili_v04_env python -m pytest tests/ -q` (preserves the 124 existing `tests/spatial/` tests) |
| **Estimated runtime** | ~30 seconds quick; full suite under ~2 min (network tests skipped offline) |

---

## Sampling Rate

- **After every task commit:** Run `conda run -n dili_v04_env python -m pytest tests/test_data_paths.py -x -q`
- **After every plan wave:** Run `conda run -n dili_v04_env python -m pytest tests/ -q` (full suite incl. 124 existing)
- **Before `/gsd-verify-work`:** Full suite green AND `results/tables/P0_coverage.md` + `P0_orthologs.md` populated AND MANIFEST complete.
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

> Task IDs are assigned by the planner; rows below are requirement-level and map to the Wave 0 test stubs. Network-gated rows use the existing `@pytest.mark.network` marker so offline/CI runs skip downloads.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | 1 | DATA-01 | — | N/A | unit (fs) | `pytest tests/test_data_paths.py::test_raw_datasets_present -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | DATA-01 | — | N/A | unit | `pytest tests/test_data_paths.py::test_accessions_corrected -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | DATA-01 (Gate 1) | — | N/A | unit | `pytest tests/test_data_paths.py::test_whole_transcriptome_gate -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | DATA-02 | — | N/A | unit (parse) | `pytest tests/test_data_paths.py::test_manifest_complete -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | DATA-02 | — | N/A | smoke | `pytest tests/test_data_paths.py::test_squidpy_available -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | DATA-03 | — | N/A | unit | `pytest tests/test_data_paths.py::test_ortholog_one2one -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | DATA-03 | — | N/A | unit | `pytest tests/test_data_paths.py::test_coverage_report_exists -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | DATA-03 (regression) | — | N/A | unit | `pytest tests/spatial/test_gene_alignment.py -x` | ✅ exists | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_data_paths.py` — sanity stubs covering DATA-01/02/03 (filesystem + MANIFEST parse + ortholog + coverage). Sanity-only per hard rule "tests never on model outputs".
- [ ] Network-gated tests use the existing `@pytest.mark.network` marker so offline/CI runs skip downloads.
- [ ] Fixture: a small cached BioMart TSV (or the verified chr21 sample) so `test_ortholog_one2one` runs offline and deterministically.
- [ ] No framework install needed — pytest 9.0.2 already present in `dili_v04_env`.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Dataset license terms reviewed before redistribution (TG-GATEs / DrugMatrix / KPMP) | DATA-02 | License acceptability is a human/legal judgment, not a code check | Read each source's license; record SPDX-style tag + URL in MANIFEST.md `license` column |
| Halt Gate 1 decision when a planned dataset is unavailable or not whole-transcriptome | DATA-01 | Gate firing requires human re-plan of the source | If a dataset fails the coverage gate or 404s, write `HALT_REASON.md` into the phase dir and stop |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
