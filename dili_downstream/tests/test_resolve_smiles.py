"""Unit tests for ``src.data.resolve_smiles``.

Tests 1-5 use synthetic in-memory DataFrames so the unit-test loop stays fast.
Test 6 (the real-data ≥ 90% gate) is exercised by the full pipeline run in the
plan's ``<verify>`` block — pytest does not invoke the 1.5 GB XML parse here.
"""
from __future__ import annotations

import pandas as pd
import pytest
from rdkit import Chem

from src.data.resolve_smiles import (
    DILIST_ID_COL,
    DILIST_NAME_COL,
    FAILURE_COLUMNS,
    REASON_NOT_IN_INDEX,
    REASON_RDKIT_PARSE_FAILURE,
    RESOLVED_COLUMNS,
    canonicalize,
    resolve_dilist,
)


def _index(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a tiny DrugBank-shaped index from (name_lower, smiles) tuples."""
    return pd.DataFrame(rows, columns=["name_lower", "smiles"])


def _dilist(rows: list[tuple[int, str]]) -> pd.DataFrame:
    """Build a tiny DILIst-shaped frame from (DILIST_ID, CompoundName) tuples."""
    return pd.DataFrame(rows, columns=[DILIST_ID_COL, DILIST_NAME_COL])


# ──────────────────────────────────────────────────────────────────
# canonicalize() — narrow contract tests
# ──────────────────────────────────────────────────────────────────


def test_canonicalize_returns_canonical_and_no_error_for_valid_smiles() -> None:
    canon, err = canonicalize("CC(=O)NC1=CC=C(O)C=C1")  # acetaminophen
    assert err is None
    assert canon is not None
    # Round-trip explicitly — the function promises the canonical is round-trip-safe.
    mol = Chem.MolFromSmiles(canon)
    assert mol is not None
    assert Chem.MolToSmiles(mol) == canon


def test_canonicalize_returns_none_and_reason_for_garbage_smiles() -> None:
    canon, err = canonicalize("BAD_SMILES_NOT_PARSEABLE")
    assert canon is None
    assert err == REASON_RDKIT_PARSE_FAILURE


def test_canonicalize_handles_empty_string() -> None:
    canon, err = canonicalize("")
    assert canon is None
    assert err == REASON_RDKIT_PARSE_FAILURE


# ──────────────────────────────────────────────────────────────────
# resolve_dilist() — the spec'd 5 behaviors
# ──────────────────────────────────────────────────────────────────


def test_resolve_dilist_returns_tuple_of_frames_with_required_columns() -> None:
    """Test 1: shape + return-tuple contract."""
    dilist_df = _dilist([(1, "Aspirin"), (2, "UnknownDrug")])
    index_df = _index([("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O")])

    resolved, failures = resolve_dilist(dilist_df, index_df)

    assert isinstance(resolved, pd.DataFrame)
    assert isinstance(failures, pd.DataFrame)
    assert list(resolved.columns) == RESOLVED_COLUMNS
    assert list(failures.columns) == FAILURE_COLUMNS


def test_resolve_dilist_hits_index_and_marks_source_drugbank() -> None:
    """Test 2: name match → resolved row with source='drugbank'."""
    dilist_df = _dilist([(1, "Aspirin")])
    index_df = _index([("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O")])

    resolved, failures = resolve_dilist(dilist_df, index_df)

    assert len(resolved) == 1
    assert len(failures) == 0
    row = resolved.iloc[0]
    assert row["DILIST_ID"] == 1
    assert row["drug_name"] == "Aspirin"
    assert row["name_lower"] == "aspirin"
    assert row["source"] == "drugbank"
    assert row["smiles"] == "CC(=O)OC1=CC=CC=C1C(=O)O"
    # canonical_smiles must round-trip
    canon = row["canonical_smiles"]
    assert canon
    assert Chem.MolToSmiles(Chem.MolFromSmiles(canon)) == canon


def test_resolve_dilist_misses_produce_name_not_in_index_failure() -> None:
    """Test 3: no name match → failures row with reason=name_not_in_drugbank_index."""
    dilist_df = _dilist([(7, "TotallyMadeUpDrug")])
    index_df = _index([("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O")])

    resolved, failures = resolve_dilist(dilist_df, index_df)

    assert len(resolved) == 0
    assert len(failures) == 1
    assert failures.iloc[0]["DILIST_ID"] == 7
    assert failures.iloc[0]["drug_name"] == "TotallyMadeUpDrug"
    assert failures.iloc[0]["reason"] == REASON_NOT_IN_INDEX


def test_resolve_dilist_rdkit_failure_routes_to_failures_log() -> None:
    """Test 4: index hit but unparseable SMILES → reason=rdkit_parse_failure."""
    dilist_df = _dilist([(42, "Brokenium")])
    index_df = _index([("brokenium", "BAD_SMILES")])

    resolved, failures = resolve_dilist(dilist_df, index_df)

    assert len(resolved) == 0
    assert len(failures) == 1
    assert failures.iloc[0]["reason"] == REASON_RDKIT_PARSE_FAILURE


def test_resolve_dilist_round_trip_holds_for_every_resolved_row() -> None:
    """Test 5: every canonical_smiles in the resolved frame round-trips through RDKit."""
    dilist_df = _dilist(
        [
            (1, "Aspirin"),
            (2, "Caffeine"),
            (3, "Acetaminophen"),
            (4, "Ibuprofen"),
            (5, "Lactate"),
        ]
    )
    index_df = _index(
        [
            ("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
            ("acetaminophen", "CC(=O)NC1=CC=C(O)C=C1"),
            ("ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
            ("lactate", "CC(O)C(=O)O"),
        ]
    )

    resolved, failures = resolve_dilist(dilist_df, index_df)

    assert len(resolved) == 5
    assert len(failures) == 0
    for canon in resolved["canonical_smiles"]:
        mol = Chem.MolFromSmiles(canon)
        assert mol is not None, f"canonical did not parse: {canon!r}"
        assert Chem.MolToSmiles(mol) == canon, f"round-trip broke for: {canon!r}"


def test_resolve_dilist_normalizes_compound_name_for_lookup() -> None:
    """Whitespace + case differences in DILIst names must not block lookups."""
    dilist_df = _dilist(
        [
            (1, "  ASPIRIN  "),
            (2, "Caffeine"),
        ]
    )
    index_df = _index(
        [
            ("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ]
    )

    resolved, failures = resolve_dilist(dilist_df, index_df)
    assert len(resolved) == 2
    assert len(failures) == 0
    assert set(resolved["name_lower"]) == {"aspirin", "caffeine"}


def test_resolve_dilist_salt_suffix_recovers_parent_compound() -> None:
    """DILIst frequently lists 'X hydrochloride' / 'X sodium' / 'X mesylate' while
    DrugBank indexes only the parent. Stripping the trailing salt token and
    re-looking-up should resolve these without fabricating SMILES."""
    dilist_df = _dilist(
        [
            (1, "Levocetirizine dihydrochloride"),
            (2, "Prasugrel hydrochloride"),
            (3, "Methylnaltrexone bromide"),
            (4, "Truly Made-Up Drug Mesylate"),  # salt-stripped is also unknown → still fails
        ]
    )
    index_df = _index(
        [
            ("levocetirizine", "OC(=O)COCCN1CCN(CC1)C(c1ccccc1)c1ccc(Cl)cc1"),
            ("prasugrel", "CC(=O)OC1=C(c2ccccc2F)N(C2CCC2C(=O)c2ccccc2)CC2CCSC12"),
            ("methylnaltrexone", "C[N+]1(CC2CC2)CCC23c4c5ccc(O)c4OC2C(=O)CCC3(O)C1C5"),
        ]
    )

    resolved, failures = resolve_dilist(dilist_df, index_df)

    # Three salt-stripped recoveries
    resolved_ids = set(resolved["DILIST_ID"])
    assert {1, 2, 3}.issubset(resolved_ids)
    # Verify source still 'drugbank' (we did not fabricate; we resolved via parent compound)
    for _, row in resolved.iterrows():
        assert row["source"] == "drugbank"
        # canonical round-trips
        assert Chem.MolToSmiles(Chem.MolFromSmiles(row["canonical_smiles"])) == row["canonical_smiles"]

    # The truly-unknown drug (even with salt stripped) goes to failures
    fail_ids = set(failures["DILIST_ID"])
    assert 4 in fail_ids


def test_resolve_dilist_does_not_strip_single_word_salt_lookalikes() -> None:
    """Drug names that are JUST a single token matching a salt-suffix must not be
    incorrectly stripped (would reduce to empty)."""
    # 'Sodium' alone has no parent — make sure we don't crash and it goes to failures.
    dilist_df = _dilist([(1, "Sodium")])
    index_df = _index([("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O")])

    resolved, failures = resolve_dilist(dilist_df, index_df)
    assert len(resolved) == 0
    assert len(failures) == 1
    assert failures.iloc[0]["reason"] == REASON_NOT_IN_INDEX


def test_resolve_dilist_missing_required_columns_raises() -> None:
    """Defensive: missing DILIST_ID or CompoundName surfaces as KeyError."""
    bad = pd.DataFrame({"foo": [1], "bar": ["x"]})
    idx = _index([("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O")])
    with pytest.raises(KeyError):
        resolve_dilist(bad, idx)
