"""In-memory fixture tests for `src.data.wangli_smiles_resolver`.

Pure-library unit tests — no real-data CSV reads. All canonical_df fixtures are
fabricated as `pd.DataFrame(...)` directly. The real `dili_canonical.csv` is
only exercised by the Wave-2 CLI driver (Plan 01-04).

Match-policy invariant under test (single, locked):
    "always salt-strip both sides" — the canonical-side lookup key is built as
    `_strip_salt_suffix(canonical_drug_name)` and the query side is matched
    against the same `_strip_salt_suffix(query_compound_name)`. There is no
    separate exact-lowercased-only first pass (the salt-strip helper is a
    no-op when no recognized salt token is present, so the exact-lowercased
    match is naturally subsumed).

The test `test_resolve_smiles_salt_strip_both_sides_disambiguates` proves
this is the active policy by querying with a salted query against a salted
canonical row that share the same parent — exact-lowercased-only would miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.wangli_smiles_resolver import (
    SALT_SUFFIXES,
    ResolvedSmiles,
    _strip_salt_suffix,
    resolve_smiles,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _df(rows: list[dict]) -> pd.DataFrame:
    """Build a canonical_df fixture with the 3 columns resolve_smiles needs."""
    return pd.DataFrame(rows, columns=["drug_name", "canonical_smiles", "dili_severity"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resolve_smiles_exact_lowercased_match():
    """Casefold-equivalent query (`'ABACAVIR'`) hits canonical row `'Abacavir'`."""
    df = _df([
        {"drug_name": "Abacavir", "canonical_smiles": "Cc1ncnc2c1ncn2[C@H]1CC[C@H](CO)O1", "dili_severity": "Less"},
    ])
    result = resolve_smiles(["ABACAVIR"], df)

    assert isinstance(result, ResolvedSmiles)
    assert result.smiles == ["Cc1ncnc2c1ncn2[C@H]1CC[C@H](CO)O1"]
    assert result.severity == ["Less"]
    assert result.drop_indices == []
    assert result.drop_names == []


def test_resolve_smiles_salt_strip_match():
    """`'Abacavir Sulfate'` resolves to canonical `'abacavir'` via salt-strip."""
    df = _df([
        {"drug_name": "abacavir", "canonical_smiles": "Cc1...", "dili_severity": "Less"},
    ])
    result = resolve_smiles(["Abacavir Sulfate"], df)

    assert result.smiles == ["Cc1..."]
    assert result.severity == ["Less"]
    assert result.drop_indices == []


def test_resolve_smiles_multi_token_salt_strip():
    """Multi-token suffix `'Pentamidine Isethionate'` resolves to `'pentamidine'`."""
    df = _df([
        {"drug_name": "pentamidine", "canonical_smiles": "NC(=N)c1ccc(OCCCCCOc2ccc(C(N)=N)cc2)cc1", "dili_severity": "Most"},
    ])
    result = resolve_smiles(["Pentamidine Isethionate"], df)

    assert result.smiles == ["NC(=N)c1ccc(OCCCCCOc2ccc(C(N)=N)cc2)cc1"]
    assert result.severity == ["Most"]
    assert result.drop_indices == []


def test_resolve_smiles_unresolved_returns_none():
    """Unknown compound → smiles=None, recorded in drop lists."""
    df = _df([
        {"drug_name": "aspirin", "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O", "dili_severity": np.nan},
    ])
    result = resolve_smiles(["UnknownCompound123"], df)

    assert result.smiles == [None]
    assert result.severity == [None]
    assert result.drop_indices == [0]
    assert result.drop_names == ["UnknownCompound123"]


def test_resolve_smiles_severity_optional():
    """NaN severity in canonical_df is normalized to None (not float NaN)."""
    df = _df([
        {"drug_name": "aspirin", "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O", "dili_severity": np.nan},
    ])
    result = resolve_smiles(["aspirin"], df)

    assert result.smiles == ["CC(=O)Oc1ccccc1C(=O)O"]
    assert result.severity == [None]
    # Sanity: not a float NaN that would slip through equality checks
    assert result.severity[0] is None


def test_resolve_smiles_input_order_preserved():
    """Output lists align to input compound_names order, not canonical_df order."""
    df = _df([
        {"drug_name": "drug1", "canonical_smiles": "S1", "dili_severity": "A"},
        {"drug_name": "drug2", "canonical_smiles": "S2", "dili_severity": "B"},
        {"drug_name": "drug3", "canonical_smiles": "S3", "dili_severity": "C"},
    ])
    result = resolve_smiles(["drug3", "drug1", "drug2"], df)

    assert result.smiles == ["S3", "S1", "S2"]
    assert result.severity == ["C", "A", "B"]
    assert result.drop_indices == []


def test_resolve_smiles_duplicate_canonical_drug_name_uses_first():
    """When canonical_df has duplicate drug_names, lookup picks first occurrence."""
    df = _df([
        {"drug_name": "aspirin", "canonical_smiles": "FIRST", "dili_severity": "A"},
        {"drug_name": "aspirin", "canonical_smiles": "SECOND", "dili_severity": "B"},
    ])
    result = resolve_smiles(["aspirin"], df)

    assert result.smiles == ["FIRST"]
    assert result.severity == ["A"]


def test_resolve_smiles_drop_lists_aligned():
    """drop_indices and drop_names align to input positions for unresolved entries."""
    df = _df([
        {"drug_name": "drugA", "canonical_smiles": "Sa", "dili_severity": "X"},
        {"drug_name": "drugC", "canonical_smiles": "Sc", "dili_severity": "Y"},
        {"drug_name": "drugE", "canonical_smiles": "Se", "dili_severity": "Z"},
    ])
    queries = ["drugA", "MISS_B", "drugC", "MISS_D", "drugE"]
    result = resolve_smiles(queries, df)

    assert result.smiles == ["Sa", None, "Sc", None, "Se"]
    assert result.severity == ["X", None, "Y", None, "Z"]
    assert result.drop_indices == [1, 3]
    assert result.drop_names == ["MISS_B", "MISS_D"]


def test_strip_salt_suffix_unit():
    """_strip_salt_suffix lowercases + iteratively strips trailing salt tokens."""
    # Single-token strip
    assert _strip_salt_suffix("Abacavir Sulfate") == "abacavir"
    # No-op for plain names (subsumes the exact-lowercased match path)
    assert _strip_salt_suffix("plain_name") == "plain_name"
    # Multi-token strip ("X isethionate complex" → "X")
    assert _strip_salt_suffix("Pentamidine Isethionate Complex") == "pentamidine"
    # Already-lowercased input untouched in the head
    assert _strip_salt_suffix("aspirin") == "aspirin"
    # Whitespace normalized
    assert _strip_salt_suffix("  Abacavir  Sulfate  ") == "abacavir"


def test_resolve_smiles_empty_compound_name():
    """Empty-string query → unresolved, in drop lists."""
    df = _df([
        {"drug_name": "aspirin", "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O", "dili_severity": "X"},
    ])
    result = resolve_smiles([""], df)

    assert result.smiles == [None]
    assert result.severity == [None]
    assert result.drop_indices == [0]
    assert result.drop_names == [""]


def test_resolve_smiles_canonical_df_missing_columns_raises():
    """canonical_df missing required column → ValueError naming the column."""
    bad_df = pd.DataFrame({
        "drug_name": ["aspirin"],
        # missing 'canonical_smiles'
        "dili_severity": ["X"],
    })
    with pytest.raises(ValueError, match="canonical_smiles"):
        resolve_smiles(["aspirin"], bad_df)


def test_resolve_smiles_salt_strip_both_sides_disambiguates():
    """Locked policy gate: 'always salt-strip both sides'.

    Constructs a query and a canonical row where BOTH carry (different) salt
    suffixes and only the parent name matches. Under "exact-lowercased-only"
    this would miss; under "always salt-strip both sides" it resolves.

    Query        : 'aspirin sulfate'   -> _strip_salt_suffix -> 'aspirin'
    Canonical    : 'Aspirin Hydrochloride' -> _strip_salt_suffix -> 'aspirin'
    Therefore lookup hits and SMILES is returned.
    """
    df = _df([
        {"drug_name": "Aspirin Hydrochloride", "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O", "dili_severity": "X"},
    ])
    result = resolve_smiles(["aspirin sulfate"], df)

    # The disambiguating assertion: salt-strip-both is the only policy that
    # yields a hit here. If the implementation were "exact-lowercased-only"
    # (i.e., did not strip the canonical-side suffix), this would fail with
    # smiles == [None] and drop_indices == [0].
    assert result.smiles == ["CC(=O)Oc1ccccc1C(=O)O"]
    assert result.severity == ["X"]
    assert result.drop_indices == []
    assert result.drop_names == []


def test_salt_suffixes_contains_v04_extended_set():
    """Lockdown: v0.4-SUMMARY-extended salt suffixes are in the frozenset."""
    # The plan's extended set per v0.4 P1's deviation block.
    required = {
        "mesylate", "hydrochloride", "sulfate", "citrate", "acetate",
        "fumarate", "phosphate", "tartrate", "maleate", "succinate",
        "sodium", "potassium", "calcium", "magnesium",
        "isethionate", "estolate", "glycinate", "complex",
    }
    missing = required - SALT_SUFFIXES
    assert missing == set(), f"Missing suffixes from SALT_SUFFIXES: {missing}"
