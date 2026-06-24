---
phase: 2
slug: multidcp-wiring-tox-head
status: planned
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-24
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from 02-RESEARCH.md "## Validation Architecture". Task IDs are assigned by the planner; rows below map requirements → tests and the wave they land in.
> Real-data-only (Hard Rule 1): the frozen-model inference path CANNOT be mocked. Pure logic is fixture-tested; real inference is a `@pytest.mark.gpu` smoke run once per wave merge, not per commit.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (in conda env `dili_v04_env`; 168 tests currently green) |
| **Config file** | none — default discovery under `tests/` |
| **Quick run command** | `conda run -n dili_v04_env pytest tests/spatial/ -x -q -m "not gpu"` |
| **Full suite command** | `conda run -n dili_v04_env pytest tests/ -q` (then the gpu smoke once: `pytest tests/spatial/test_model_load.py -m gpu -x`) |
| **Estimated runtime** | ~60–150 s pure suite; real-inference smoke adds checkpoint load + 1–2 forward passes |

---

## Sampling Rate

- **After every task commit:** quick run on the touched test file (`conda run -n dili_v04_env pytest tests/spatial/test_<file>.py -x -q`)
- **After every plan wave:** full pure suite (`pytest tests/ -q -m "not gpu"`) then the gpu real-inference smoke once
- **Before `/gsd-verify-work`:** full suite green AND the APAP per-zone Pearson reported (WIRE-03 acceptance)
- **Max feedback latency:** ~150 seconds (pure suite)

---

## Per-Task Verification Map

| Task (TBD at plan) | Wave | Requirement | Behavior | Test Type | Automated Command | File Exists |
|--------------------|------|-------------|----------|-----------|-------------------|-------------|
| TBD | 0 | WIRE-01 | rule-B DE math (treated − control), 10,716 shape, no NaN | unit (fixture) | `pytest tests/spatial/test_region_signature.py -k rule_b -x` | ❌ W0 (extend) |
| TBD | 0 | WIRE-01 | cache stores DE + treated + control (D-03); manifest de_convention updated | unit | `pytest tests/spatial/test_region_signature.py -k cache_three_vector -x` | ❌ W0 |
| TBD | 1 | WIRE-01 | checkpoint loads strict 0/0 into `MultiDCP_CheMoE_AE` (row-17 ckpt) | integration (real ckpt) — CANNOT mock | `pytest tests/spatial/test_model_load.py -x -m gpu` | ❌ W0 (new) |
| TBD | 1 | WIRE-01 | one real forward → finite [10716] in normalized range (0-1 min-max sanity) | integration smoke (real model) | `pytest tests/spatial/test_model_load.py -k forward_sane -x -m gpu` | ❌ W0 |
| TBD | 0 | WIRE-02 | concat-MLP forward: zero GEX channel → finite logit; per-channel mask shapes | unit (fixture) | `pytest tests/spatial/test_tox_head.py -x` | ❌ W0 (new) |
| TBD | 0 | WIRE-02 | attention combiner feeds head; [B,n_regions,10716]→[B,10716]→logit | unit | `pytest tests/spatial/test_tox_head.py -k combiner_feed -x` | ❌ W0 |
| TBD | 1 | WIRE-02 | smoke-train condition A ≥1 epoch, loss moves, wandb logged | integration smoke (head only, synthetic batch OK) | `python scripts/smoke_train_condA.py` | ❌ W0 |
| TBD | 0 | WIRE-03 | zone assignment from markers; pseudobulk per zone | unit (fixture AnnData) | `pytest tests/spatial/test_apap_validation.py -k zone -x` | ❌ W0 (new) |
| TBD | 0 | WIRE-03 | measured DE = APAP_zone − ctrl_zone; mouse→human ortholog align; intersection size reported | unit | `pytest tests/spatial/test_apap_validation.py -k measured_de -x` | ❌ W0 |
| TBD | 0 | WIRE-03 | per-zone Pearson math + Halt-Gate-3 trigger (<0.3 pericentral) | unit | `pytest tests/spatial/test_apap_validation.py -k pearson_gate -x` | ❌ W0 |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Fixture-testable PURE logic (no real model):** rule-B subtraction, 3-vector cache assembly, manifest convention string, zone assignment, measured-DE subtraction, ortholog alignment, per-zone Pearson, Halt-Gate-3 threshold — covers the bulk of WIRE-01/02/03 acceptance deterministically.

**Real-inference smoke (CANNOT be mocked, Hard Rule 1):** checkpoint strict-load, one forward → finite in-range 10,716 vector, normalization sanity. `@pytest.mark.gpu`; once per wave merge.

---

## Wave 0 Requirements

- [ ] `tests/spatial/test_region_signature.py` — EXTEND for rule-B + 3-vector cache (file exists)
- [ ] `tests/spatial/test_model_load.py` — NEW: real-ckpt strict-load + forward sanity (gpu-marked)
- [ ] `tests/spatial/test_tox_head.py` — NEW: WIRE-02 (concat-MLP, masking, combiner feed)
- [ ] `tests/spatial/test_apap_validation.py` — NEW: WIRE-03 + Halt Gate 3 (zones, measured DE, Pearson)
- [ ] `tests/spatial/conftest.py` — shared fixtures (tiny AnnData with zone markers; synthetic treated/control vectors; fixed gene-order slice)
- [ ] register the `gpu` pytest marker (avoid unknown-marker warnings)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| APAP per-zone Pearson reported + Halt-Gate-3 read | WIRE-03 | Real Visium + real frozen inference; the gate decision is read from the produced table | Run the APAP validation driver on GSE272564 (control+APAP arms); confirm per-zone Pearson table written and the pericentral value vs the 0.3 gate is stated |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 150s (pure suite)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
