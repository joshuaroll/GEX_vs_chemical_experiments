"""Unit tests for `src/data/summarize_phase2.py`.

Tests `render_markdown(diagnostics)` against an in-memory fixture mirroring the
diagnostics dict the Wave 3 CLI driver (`scripts/build_phase2_splits.py`) will
emit. No real-data dependency.

The 7 required section headers are LOCKED — see 02-CONTEXT.md §specifics.
"""

from __future__ import annotations

import pytest

from src.data.summarize_phase2 import render_markdown


@pytest.fixture
def diagnostics_fixture() -> dict:
    """Hand-built diagnostics dict matching the structure the CLI driver passes.

    All counts are synthetic but mathematically consistent (e.g. test slice
    counts sum correctly).
    """
    return {
        "scaffold_split": {
            "size_train": 894,
            "size_val": 112,
            "size_test": 112,
            "class_balance_train": (0.615, 550, 894),  # (pos_rate, n_pos, n_total)
            "class_balance_val": (0.598, 67, 112),
            "class_balance_test": (0.607, 68, 112),
            "stratified": True,
            "tanimoto_histogram_buckets": [
                (0.0, 0.1, 23),
                (0.1, 0.2, 18),
                (0.2, 0.3, 31),
                (0.3, 0.4, 17),
                (0.4, 0.5, 12),
                (0.5, 0.6, 6),
                (0.6, 0.7, 3),
                (0.7, 0.8, 1),
                (0.8, 0.9, 1),
                (0.9, 1.0, 0),
            ],
        },
        "cluster_split": {
            "size_train": 894,
            "size_val": 112,
            "size_test": 112,
            "class_balance_train": None,
            "class_balance_val": None,
            "class_balance_test": None,
            "stratified": False,
            "tanimoto_histogram_buckets": [
                (0.0, 0.1, 30),
                (0.1, 0.2, 25),
                (0.2, 0.3, 20),
                (0.3, 0.4, 15),
                (0.4, 0.5, 10),
                (0.5, 0.6, 5),
                (0.6, 0.7, 4),
                (0.7, 0.8, 2),
                (0.8, 0.9, 1),
                (0.9, 1.0, 0),
            ],
        },
        "tdc_split": {
            "tdc_version": "0.4.17",
            "dataset_size": 475,
            "size_train": 332,
            "size_val": 47,
            "size_test": 96,
        },
        "transfer_slices": {
            "test_in_pdg": 86,
            "test_drug_novel": 26,
            "test_drug_and_scaffold_novel": 22,
        },
        "upstream_filter": {
            "d_pdg_total": 892,
            "excluded_by_scaffold": 41,
            "excluded_by_pert_id": 89,
            "excluded_intersection": 18,
            "d_pdg_train_after_exclusion": 780,
        },
        "halt_gate": {
            "value": 226,
            "passed": True,
            "threshold": 30,
        },
        "metadata": {
            "seed": 42,
            "dili_canonical_rows": 1118,
            "generated_at": "2026-05-06T12:00:00Z",
        },
    }


def test_render_includes_all_seven_required_section_headers(diagnostics_fixture):
    """Test 1: render_markdown produces every required section header (one assert each)."""
    md = render_markdown(diagnostics_fixture)
    # Each header is grep-checked by the plan's verify block — keep them stable.
    assert "# Phase 2 Split Summary" in md
    assert "## Scaffold split" in md
    assert "## Cluster split" in md
    assert "## TDC-DILI scaffold split" in md
    assert "## Three transfer slices" in md
    assert "## Upstream-train filter diagnostics" in md
    assert "## Halt gate evaluation" in md


def test_halt_gate_pass_emits_locked_format(diagnostics_fixture):
    """Test 2: halt_gate.passed=True emits the literal PASS line with the value + threshold."""
    diagnostics_fixture["halt_gate"] = {"value": 226, "passed": True, "threshold": 30}
    md = render_markdown(diagnostics_fixture)
    # Locked format from 02-CONTEXT.md / 02-03-PLAN.md <interfaces>:
    assert r"HALT-GATE PASS: |D_DILI_test \ D_PDG| = 226 (threshold 30)" in md


def test_halt_gate_fail_emits_stop_directive(diagnostics_fixture):
    """Test 3: halt_gate.passed=False emits FAIL line + 'STOP and re-discuss' directive."""
    diagnostics_fixture["halt_gate"] = {"value": 27, "passed": False, "threshold": 30}
    md = render_markdown(diagnostics_fixture)
    assert "HALT-GATE FAIL" in md
    assert "STOP and re-discuss" in md
    # Value should be reflected
    assert "27" in md


def test_class_balance_per_slice_reported_for_stratified_scaffold(diagnostics_fixture):
    """Test 4: scaffold split section reports per-slice positive rate when stratified."""
    md = render_markdown(diagnostics_fixture)
    # Class balance numbers from the fixture (61.5%, 59.8%, 60.7%) — at least one slice
    # rendered with explicit "X/Y positive" form.
    assert "550/894" in md  # train count "n_pos/n_total"
    # Positive-rate value rendered (one decimal — at minimum some flavor of "61")
    assert "61" in md


def test_cluster_split_non_stratified_emits_literal_message(diagnostics_fixture):
    """Test 5: cluster split section emits the locked non-stratified literal."""
    md = render_markdown(diagnostics_fixture)
    # Locked literal per 02-CONTEXT.md §specifics
    assert "non-stratified" in md
    assert "clusters too small" in md


def test_tanimoto_histogram_section_present(diagnostics_fixture):
    """Test 6: histogram is described as a markdown table with at least one `|` row."""
    md = render_markdown(diagnostics_fixture)
    assert "Tanimoto train-test max-similarity" in md
    # Table row marker — at minimum the table header pipe and at least one bucket row
    assert "|" in md  # markdown table pipes
    # First bucket from the fixture (count=23, scaffold)
    assert "0.0" in md
    assert "0.1" in md


def test_tdc_version_and_dataset_size_reported(diagnostics_fixture):
    """Test 7: TDC section includes pytdc version and dataset_size."""
    md = render_markdown(diagnostics_fixture)
    assert "0.4.17" in md  # tdc_version from fixture
    # dataset_size = 475 from fixture
    assert "475" in md
    # Section-localized terms
    assert "pytdc" in md.lower() or "tdc" in md.lower()


def test_three_transfer_slices_counts_reported(diagnostics_fixture):
    """Test 8: all 3 transfer-slice counts emitted with their labels."""
    md = render_markdown(diagnostics_fixture)
    # Locked label format — grep-style:
    assert "|test_in_pdg|" in md
    assert "|test_drug_novel|" in md
    assert "|test_drug_and_scaffold_novel|" in md
    # Counts from fixture
    assert "86" in md
    assert "26" in md
    assert "22" in md


def test_upstream_filter_diagnostics_section_has_five_count_labels(diagnostics_fixture):
    """Test 9: upstream-train filter section reports all 5 counts with locked key labels."""
    md = render_markdown(diagnostics_fixture)
    # Locked key labels per 02-CONTEXT.md §specifics
    assert "|D_PDG_total|" in md
    assert "|excluded_by_scaffold|" in md
    assert "|excluded_by_pert_id|" in md
    assert "|intersection|" in md
    assert "|D_PDG_train_after_exclusion|" in md
    # Counts from fixture
    assert "892" in md
    assert "780" in md
