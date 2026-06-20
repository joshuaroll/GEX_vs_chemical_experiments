---
phase: 0
slug: dataset-acquisition-manifest
status: draft
nyquist_compliant: true
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

> Mapped to the produced plans (00-01 … 00-04). Each Wave-0 stub is CREATED by Plan 01 (Wave 1) but only turns green once the producing task runs; the "Plan/Wave" columns below name the plan/task that makes each row green (not where the stub is first written). Network-gated rows use the `@needs_net` / skip-when-absent guard so offline runs skip.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 00-01 · T2 (stub) → 00-01 · T1 (green) | 01 | 1 | DATA-01 | T-00-01 | accession integrity guard | unit | `pytest tests/test_data_paths.py::test_accessions_corrected -x` | ✅ Plan 01 | ⬜ pending |
| 00-01 · T1 | 01 | 1 | DATA-01 | T-00-01 | registry schema (new fields) | unit | `pytest tests/test_data_paths.py::test_registry_schema -x` | ✅ Plan 01 | ⬜ pending |
| 00-02 · T2 | 02 | 2 | DATA-01 | T-00-05 | files present per `entry.slug`/`expected_files` | unit (fs) | `pytest tests/test_data_paths.py::test_raw_datasets_present -x` | ✅ Plan 02 | ⬜ pending |
| 00-04 · T1 (Gate 1) | 04 | 3 | DATA-01 (Gate 1) | T-00-05 | whole-transcriptome coverage > 0.80 | unit | `pytest tests/test_data_paths.py::test_whole_transcriptome_gate -x` | ✅ Plan 04 | ⬜ pending |
| 00-04 · T2 | 04 | 3 | DATA-02 | T-00-10 | MANIFEST path+SHA256+license+WT rows | unit (parse) | `pytest tests/test_data_paths.py::test_manifest_complete -x` | ✅ Plan 04 | ⬜ pending |
| 00-04 · T1 | 04 | 3 | DATA-02 | T-00-12 | squidpy importable + env recorded | smoke | `pytest tests/test_data_paths.py::test_squidpy_available -x` | ✅ Plan 04 | ⬜ pending |
| 00-03 · T1 | 03 | 2 | DATA-03 | T-00-07 | one2one ortholog filter (offline fixture) | unit | `pytest tests/test_data_paths.py::test_ortholog_one2one -x` | ✅ Plan 03 | ⬜ pending |
| 00-04 · T1 | 04 | 3 | DATA-03 | T-00-07 | coverage + ortholog report tables exist | unit | `pytest tests/test_data_paths.py::test_coverage_report_exists -x` | ✅ Plan 04 | ⬜ pending |
| (regression) | — | — | DATA-03 (regression) | — | gene_alignment unchanged | unit | `pytest tests/spatial/test_gene_alignment.py -x` | ✅ exists | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

> Wave note: `test_ortholog_one2one` is satisfied by Wave 2 (Plan 03); `test_coverage_report_exists` is satisfied by Wave 3 (Plan 04) — corrected from the prior "Wave 1" placeholders. All eight `tests/test_data_paths.py` stubs are FIRST WRITTEN in Plan 01 Task 2 (Wave 1) and run green offline (artifact-dependent rows skip cleanly until their producing plan runs).

---

## Wave 0 Requirements

- [ ] `tests/test_data_paths.py` — sanity stubs covering DATA-01/02/03 (filesystem + MANIFEST parse + ortholog + coverage). Sanity-only per hard rule "tests never on model outputs". Created in Plan 01 Task 2.
- [ ] Network-gated tests use the `@needs_net` skipif (`TDC_NETWORK_TESTS=1`) so offline/CI runs skip downloads.
- [ ] Fixture: a small cached BioMart TSV (the verified chr21 sample, `tests/fixtures/biomart_chr21_sample.tsv`) so `test_ortholog_one2one` runs offline and deterministically. Created in Plan 01 Task 2.
- [ ] No framework install needed — pytest 9.0.2 already present in `dili_v04_env`.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Dataset license terms reviewed before redistribution (TG-GATEs / DrugMatrix / KPMP) | DATA-02 | License acceptability is a human/legal judgment, not a code check | Read each source's license; record SPDX-style tag + URL in MANIFEST.md `license` column |
| KPMP atlas ToS click-through (if GEO supplementary fails) | DATA-01 | atlas.kpmp.org may require a click-through Terms-of-Service acceptance with no API | Accept ToS at https://atlas.kpmp.org and place Visium objects into `data/raw/spatial/lake_kpmp_kidney/` (Plan 02 `user_setup`) |
| Halt Gate 1 decision when a planned dataset is unavailable or not whole-transcriptome | DATA-01 | Gate firing requires human re-plan of the source | If a dataset fails the coverage gate or 404s, write `HALT_REASON.md` into the phase dir and stop — a legitimately-halted Phase 0 is an ACCEPTED terminal state, not a plan failure |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-06-20
