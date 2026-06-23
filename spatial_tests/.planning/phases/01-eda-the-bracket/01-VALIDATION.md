---
phase: 1
slug: eda-the-bracket
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-23
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from 01-RESEARCH.md "## Validation Architecture". Task IDs are assigned by the planner; rows below map requirements → tests and the wave they land in.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (in conda env `dili_v04_env`) |
| **Config file** | none — implicit discovery; Wave 0 creates `tests/spatial/__init__.py` |
| **Quick run command** | `conda run -n dili_v04_env pytest tests/spatial/ -x -q` |
| **Full suite command** | `conda run -n dili_v04_env pytest tests/ -q` |
| **Estimated runtime** | ~60–120 seconds (quick), full suite longer with data-touching smokes |

---

## Sampling Rate

- **After every task commit:** Run `conda run -n dili_v04_env pytest tests/spatial/ -x -q`
- **After every plan wave:** Run `conda run -n dili_v04_env pytest tests/ -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task (TBD at plan) | Wave | Requirement | Behavior | Test Type | Automated Command | File Exists |
|--------------------|------|-------------|----------|-----------|-------------------|-------------|
| W0 stubs | 0 | EDA-01/02/03 | Test files + `tests/spatial/__init__.py` exist | scaffold | `conda run -n dili_v04_env pytest tests/spatial/ -q` | ❌ W0 |
| floor | — | EDA-01 | DILIrank binary encoding excludes Ambiguous (lowercase-normalized `vDILI-Concern`) | unit | `pytest tests/spatial/test_eda_labels.py::test_dilirank_binary_encoding -x` | ❌ W0 |
| floor | — | EDA-01 | SMILES join cascade: dili_canonical → drugbank fallback → TDC | unit | `pytest tests/spatial/test_eda_labels.py::test_smiles_join_cascade -x` | ❌ W0 |
| floor | — | EDA-01 | ECFP4 fingerprint: 2048-bit radius-2 on known SMILES | unit | `pytest tests/spatial/test_eda_fingerprints.py::test_ecfp4_shape -x` | ❌ W0 |
| floor | — | EDA-01 | Floor AUROC > 0.5 (better than chance) on DILIrank | smoke | `pytest tests/spatial/test_eda_floor.py::test_floor_auroc_above_chance -x` | ❌ W0 |
| ceiling | — | EDA-02 | Participation ratio formula `(Σλ)²/Σλ²` on synthetic matrix | unit | `pytest tests/spatial/test_eda_ceiling.py::test_participation_ratio -x` | ❌ W0 |
| ceiling | — | EDA-02 | Per-gene MI returns correct shape over the 978 genes | unit | `pytest tests/spatial/test_eda_ceiling.py::test_per_gene_mi_shape -x` | ❌ W0 |
| ceiling | — | EDA-02 | Ceiling AUROC ≥ floor AUROC on shared drug set | smoke | `pytest tests/spatial/test_eda_ceiling.py::test_ceiling_ge_floor -x` | ❌ W0 |
| gap | — | EDA-02 | Paired bootstrap CI width > 0 (10,000 resamples, shared drug set) | unit | `pytest tests/spatial/test_eda_bootstrap.py::test_bootstrap_ci_nonzero_width -x` | ❌ W0 |
| region | — | EDA-03 | Moran's I returns DataFrame indexed by gene (squidpy `spatial_autocorr`) | unit | `pytest tests/spatial/test_eda_region.py::test_moran_returns_dataframe -x` | ❌ W0 |
| xspecies | — | EDA-03 | Human-mouse correlation uses only one-to-one orthologs | unit | `pytest tests/spatial/test_eda_region.py::test_cross_species_ortholog_filter -x` | ❌ W0 |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky — all currently ❌ W0 (no test files exist yet).*

---

## Wave 0 Requirements

- [ ] `tests/spatial/__init__.py` — package init
- [ ] `tests/spatial/test_eda_labels.py` — EDA-01 (label loading, SMILES join cascade)
- [ ] `tests/spatial/test_eda_fingerprints.py` — EDA-01 (ECFP4 2048-bit radius-2)
- [ ] `tests/spatial/test_eda_floor.py` — EDA-01 (floor AUROC smoke)
- [ ] `tests/spatial/test_eda_ceiling.py` — EDA-02 (participation ratio, per-gene MI, ceiling AUROC)
- [ ] `tests/spatial/test_eda_bootstrap.py` — EDA-02 (paired bootstrap CI of the AUROC gap)
- [ ] `tests/spatial/test_eda_region.py` — EDA-03 (Moran's I, cross-species ortholog-filtered correlation)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Floor-ceiling gap interpretation vs Halt Gate 2 | EDA-02 | The gate decision (CI includes 0 → reframe) is a judgment surfaced from the table, acted on by the user | Read `results/tables/P1_eda.md`; confirm the reported 95% paired-bootstrap CI of (ceiling − floor) AUROC and whether it includes 0 |
| SMILES coverage adequacy | EDA-01 | "Floor underpowered" is a coverage call (<40% non-Ambiguous → flag) | Read coverage line in `P1_eda.md`; confirm covered fraction and flag if below threshold |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (7 test files above)
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
