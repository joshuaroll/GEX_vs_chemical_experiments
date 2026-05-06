"""Unit tests for `src/data/summarize_phase1.py`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.summarize_phase1 import render_markdown, summary_stats


@pytest.fixture
def synthetic_canonical() -> pd.DataFrame:
    """10 rows: 4 positives, 6 negatives, 3 with severity, 5 in LINCS, 7 in PDG."""
    return pd.DataFrame({
        "pert_id": list(range(1, 11)),
        "drug_name": [f"drug_{i}" for i in range(1, 11)],
        "smiles": ["CC"] * 10,
        "canonical_smiles": ["CC"] * 10,
        "scaffold": ["c1ccccc1"] * 10,
        "dili_binary": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "dili_severity": [
            "vMOST-DILI-concern", "vLess-DILI-concern", "vNo-DILI-concern",
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        ],
        "in_lincs": [True, True, True, True, True, False, False, False, False, False],
        "in_pdg":   [True, True, True, True, True, True, True, False, False, False],
    })


def test_class_balance_correct(synthetic_canonical):
    """Test 1: class_balance is computed correctly."""
    stats = summary_stats(synthetic_canonical, dilist_total=12)
    assert stats["class_balance"] == {
        "positive": 4,
        "negative": 6,
        "positive_frac": 0.4,
    }


def test_render_includes_all_required_headers(synthetic_canonical):
    """Test 2: render_markdown produces every required section header (one assert each)."""
    stats = summary_stats(synthetic_canonical, dilist_total=12)
    md = render_markdown(stats)
    # Each header is grep-checked by the plan's verify block — keep them stable.
    assert "# Phase 1 Data Summary" in md
    assert "## Class balance" in md
    assert "## SMILES resolution rate" in md
    assert "## DILIrank severity populated" in md
    assert "## D_DILI ∩ LINCS" in md
    assert "## D_DILI ∩ PDG" in md


def test_smiles_resolution_rate_uses_dilist_total(synthetic_canonical):
    """Test 3: rate is n_canonical / dilist_total, NOT n_canonical / n_canonical (=1.0)."""
    stats = summary_stats(synthetic_canonical, dilist_total=12)
    # 10/12 ≈ 0.8333
    assert abs(stats["smiles_resolution_rate"] - (10 / 12)) < 1e-9
    # And NOT 1.0
    assert stats["smiles_resolution_rate"] != 1.0


def test_severity_count_and_frac(synthetic_canonical):
    """severity_populated count and fraction match the synthetic fixture."""
    stats = summary_stats(synthetic_canonical, dilist_total=12)
    assert stats["severity_populated"]["count"] == 3
    assert abs(stats["severity_populated"]["frac"] - 0.3) < 1e-9


def test_intersect_counts(synthetic_canonical):
    """intersect_lincs and intersect_pdg counts match the boolean column sums."""
    stats = summary_stats(synthetic_canonical, dilist_total=12)
    assert stats["intersect_lincs"]["count"] == 5
    assert stats["intersect_pdg"]["count"] == 7
    assert abs(stats["intersect_lincs"]["frac"] - 0.5) < 1e-9
    assert abs(stats["intersect_pdg"]["frac"] - 0.7) < 1e-9


def test_render_markdown_contains_actual_numbers(synthetic_canonical):
    """The rendered markdown should embed the actual stat values (smoke check)."""
    stats = summary_stats(synthetic_canonical, dilist_total=12)
    md = render_markdown(stats)
    # 10 rows total
    assert "10" in md
    # 4 positives
    assert "4" in md
    # ratio appears
    assert "83." in md or "83%" in md or "83.333" in md  # 10/12 in some form
