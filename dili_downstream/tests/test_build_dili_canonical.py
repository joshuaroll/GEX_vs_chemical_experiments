"""Unit tests for `src/data/build_dili_canonical.py`.

Synthetic-only fixtures — no real DrugBank XML, GCTX, or PDG pickle parsing here.
Real-data invariants are checked by the inline asserts in `scripts/build_dili_canonical.py`.

Tests cover behaviors 1-8 in `01-02-PLAN.md`:
  1. Output has exactly the 9 locked columns in canonical order.
  2. `pert_id` taken from DILIst's `DILIST_ID` column and is unique.
  3. `dili_binary` is integer 0/1 with no NaN.
  4. `dili_severity` populated only for the DILIst ∩ DILIrank intersection.
  5. `scaffold` computed via Murcko; acyclic SMILES return "" (not crash).
  6. `in_lincs` and `in_pdg` are bool dtype reflecting drug-name membership.
  7. Drugs without a resolved SMILES are dropped from the canonical table.
  8. Rows with non-binary `DILIst Classification` (NaN or e.g. 2) are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.build_dili_canonical import (
    CANONICAL_COLUMNS,
    build_canonical,
    murcko_scaffold,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dilist_df() -> pd.DataFrame:
    """5 DILIst rows: 4 valid (3 pos, 1 neg) + 1 with non-binary classification."""
    return pd.DataFrame(
        {
            "DILIST_ID": [1, 2, 3, 4, 5],
            "CompoundName": ["Acetaminophen", "Aspirin", "Caffeine",
                             "Mercaptopurine", "Junk"],
            # Trailing space matches real DILIst quirk; tests should be robust to either.
            "DILIst Classification": [1, 0, 1, 1, 2],  # row 5 has a non-binary value
            "Routs of Administration": ["Oral"] * 5,
        }
    )


@pytest.fixture
def dilirank_df() -> pd.DataFrame:
    """3 DILIrank rows; 2 overlap with DILIst by lowercased CompoundName."""
    return pd.DataFrame(
        {
            "LTKBID": ["LT0001", "LT0002", "LT0003"],
            "CompoundName": ["acetaminophen", "Aspirin", "Lonidamine"],  # mixed case
            "SeverityClass": [3, 1, 2],
            "vDILI-Concern": ["vMOST-DILI-concern", "vNo-DILI-concern",
                              "vLess-DILI-concern"],
        }
    )


@pytest.fixture
def resolved_df() -> pd.DataFrame:
    """4 of the 5 DILIst rows have resolved SMILES (DILIST_ID 4 is unresolved)."""
    return pd.DataFrame(
        {
            "DILIST_ID": [1, 2, 3, 5],
            "drug_name": ["acetaminophen", "aspirin", "caffeine", "junk"],
            "name_lower": ["acetaminophen", "aspirin", "caffeine", "junk"],
            "smiles": [
                "CC(=O)NC1=CC=C(O)C=C1",       # acetaminophen
                "CC(=O)Oc1ccccc1C(=O)O",       # aspirin
                "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
                "CCC",                          # acyclic — exercises Test 5
            ],
            "canonical_smiles": [
                "CC(=O)Nc1ccc(O)cc1",
                "CC(=O)Oc1ccccc1C(=O)O",
                "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                "CCC",
            ],
            "source": ["drugbank"] * 4,
        }
    )


@pytest.fixture
def lincs_inames_lower() -> set[str]:
    return {"acetaminophen", "caffeine", "lonidamine"}


@pytest.fixture
def pdg_inames_lower() -> set[str]:
    return {"aspirin", "caffeine"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_1_canonical_column_order(dilist_df, dilirank_df, resolved_df,
                                   lincs_inames_lower, pdg_inames_lower):
    """Test 1: returned DataFrame has exactly the 9 locked columns in order."""
    df = build_canonical(dilist_df, dilirank_df, resolved_df,
                         lincs_inames_lower, pdg_inames_lower)
    assert df.columns.tolist() == CANONICAL_COLUMNS
    assert CANONICAL_COLUMNS == [
        "pert_id", "drug_name", "smiles", "canonical_smiles", "scaffold",
        "dili_binary", "dili_severity", "in_lincs", "in_pdg",
    ]


def test_2_pert_id_from_dilist_id_and_unique(dilist_df, dilirank_df, resolved_df,
                                              lincs_inames_lower, pdg_inames_lower):
    """Test 2: `pert_id` comes from DILIST_ID and is unique."""
    df = build_canonical(dilist_df, dilirank_df, resolved_df,
                         lincs_inames_lower, pdg_inames_lower)
    assert df["pert_id"].is_unique
    # DILIST_IDs 1, 2, 3 are valid binary AND have resolved SMILES.
    # Row 4 is dropped (no resolved SMILES). Row 5 is dropped (non-binary class).
    assert sorted(df["pert_id"].tolist()) == [1, 2, 3]


def test_3_dili_binary_int_no_nan(dilist_df, dilirank_df, resolved_df,
                                   lincs_inames_lower, pdg_inames_lower):
    """Test 3: `dili_binary` is integer 0/1 with no NaN."""
    df = build_canonical(dilist_df, dilirank_df, resolved_df,
                         lincs_inames_lower, pdg_inames_lower)
    assert df["dili_binary"].notna().all()
    assert df["dili_binary"].isin([0, 1]).all()
    assert pd.api.types.is_integer_dtype(df["dili_binary"])


def test_4_severity_only_in_intersection(dilist_df, dilirank_df, resolved_df,
                                          lincs_inames_lower, pdg_inames_lower):
    """Test 4: severity populated only for DILIst ∩ DILIrank (lowercased name match)."""
    df = build_canonical(dilist_df, dilirank_df, resolved_df,
                         lincs_inames_lower, pdg_inames_lower)
    # acetaminophen and aspirin (rows 1, 2) overlap with DILIrank.
    # caffeine (row 3) does NOT overlap with DILIrank.
    sev = df.set_index("pert_id")["dili_severity"]
    assert sev.loc[1] == "vMOST-DILI-concern"
    assert sev.loc[2] == "vNo-DILI-concern"
    assert pd.isna(sev.loc[3])


def test_5_scaffold_acyclic_returns_empty(dilist_df, dilirank_df, resolved_df,
                                           lincs_inames_lower, pdg_inames_lower):
    """Test 5: Murcko scaffold returns "" (empty string) for acyclic SMILES, no crash."""
    df = build_canonical(dilist_df, dilirank_df, resolved_df,
                         lincs_inames_lower, pdg_inames_lower)
    # caffeine (id=3) has a ring → non-empty scaffold; CCC would be acyclic.
    # In our resolved_df, only "junk"/CCC is acyclic and that row is dropped (non-binary).
    # So we test the helper directly:
    assert murcko_scaffold("CCC") == ""
    assert murcko_scaffold("c1ccccc1") == "c1ccccc1"
    # And no scaffold column NaN in the produced df
    assert df["scaffold"].notna().all()


def test_6_in_flags_are_bool(dilist_df, dilirank_df, resolved_df,
                              lincs_inames_lower, pdg_inames_lower):
    """Test 6: `in_lincs` and `in_pdg` are bool dtype, reflect drug-name membership."""
    df = build_canonical(dilist_df, dilirank_df, resolved_df,
                         lincs_inames_lower, pdg_inames_lower)
    assert df["in_lincs"].dtype == bool
    assert df["in_pdg"].dtype == bool
    flags = df.set_index("pert_id")[["in_lincs", "in_pdg"]]
    # acetaminophen (id=1): in lincs only
    assert flags.loc[1, "in_lincs"] and not flags.loc[1, "in_pdg"]
    # aspirin (id=2): in pdg only
    assert not flags.loc[2, "in_lincs"] and flags.loc[2, "in_pdg"]
    # caffeine (id=3): in both
    assert flags.loc[3, "in_lincs"] and flags.loc[3, "in_pdg"]


def test_7_unresolved_smiles_dropped(dilist_df, dilirank_df, resolved_df,
                                      lincs_inames_lower, pdg_inames_lower):
    """Test 7: rows with no resolved SMILES are dropped (canonical_smiles non-nullable)."""
    df = build_canonical(dilist_df, dilirank_df, resolved_df,
                         lincs_inames_lower, pdg_inames_lower)
    # DILIst row 4 (Mercaptopurine) is NOT in resolved_df → must be absent.
    assert 4 not in df["pert_id"].tolist()
    # And canonical_smiles column has no NaN in the survivors.
    assert df["canonical_smiles"].notna().all()


def test_8_non_binary_classification_dropped(dilist_df, dilirank_df, resolved_df,
                                              lincs_inames_lower, pdg_inames_lower):
    """Test 8: rows with `DILIst Classification` not in {0, 1} are dropped."""
    df = build_canonical(dilist_df, dilirank_df, resolved_df,
                         lincs_inames_lower, pdg_inames_lower)
    # DILIst row 5 (junk, Classification=2) → dropped even though resolved SMILES exists.
    assert 5 not in df["pert_id"].tolist()


def test_handles_trailing_space_in_classification_column():
    """Real DILIst column is 'DILIst Classification ' (trailing space). Build must cope."""
    dilist = pd.DataFrame(
        {
            "DILIST_ID": [1],
            "CompoundName": ["Acetaminophen"],
            "DILIst Classification ": [1],  # trailing space — real-world quirk
            "Routs of Administration ": ["Oral"],
        }
    )
    dilirank = pd.DataFrame(
        {"LTKBID": ["LT01"], "CompoundName": ["acetaminophen"],
         "vDILI-Concern": ["vMOST-DILI-concern"]}
    )
    resolved = pd.DataFrame(
        {"DILIST_ID": [1], "drug_name": ["acetaminophen"],
         "smiles": ["CC(=O)NC1=CC=C(O)C=C1"],
         "canonical_smiles": ["CC(=O)Nc1ccc(O)cc1"]}
    )
    df = build_canonical(dilist, dilirank, resolved, set(), set())
    assert df["dili_binary"].tolist() == [1]


def test_severity_match_via_salt_suffix_strip():
    """DILIst 'abacavir' should match DILIrank 'abacavir sulfate' via salt-strip fallback.

    Rule 2 fix: without this, severity coverage on real data was 433 instead of the
    ~1,037 projected by 01-CONTEXT.md (mostly because DILIrank lists salted forms
    and DILIst uses parent names). Mirrors Plan 01's resolver salt-strip tactic.
    """
    dilist = pd.DataFrame({
        "DILIST_ID": [1, 2],
        "CompoundName": ["Abacavir", "Lonidamine sodium"],
        "DILIst Classification": [1, 1],
    })
    dilirank = pd.DataFrame({
        "LTKBID": ["LT01", "LT02"],
        "CompoundName": ["Abacavir sulfate", "Lonidamine"],  # opposite salt-state
        "vDILI-Concern": ["vMOST-DILI-concern", "vNo-DILI-concern"],
    })
    resolved = pd.DataFrame({
        "DILIST_ID": [1, 2],
        "drug_name": ["abacavir", "lonidamine sodium"],
        "smiles": ["CC", "CO"],
        "canonical_smiles": ["CC", "CO"],
    })
    df = build_canonical(dilist, dilirank, resolved, set(), set())
    sev = df.set_index("pert_id")["dili_severity"]
    # DILIst→DILIrank: 'abacavir' should hit 'abacavir sulfate' → strip → 'abacavir'
    assert sev.loc[1] == "vMOST-DILI-concern"
    # DILIst→DILIrank: 'lonidamine sodium' → strip → 'lonidamine' (exact in DILIrank)
    assert sev.loc[2] == "vNo-DILI-concern"


def test_dilirank_missing_severity_column_raises():
    """If DILIrank doesn't have 'vDILI-Concern' column, raise KeyError explicitly."""
    dilist = pd.DataFrame(
        {"DILIST_ID": [1], "CompoundName": ["A"], "DILIst Classification": [1]}
    )
    bad_dilirank = pd.DataFrame({"LTKBID": ["LT01"], "CompoundName": ["a"]})
    resolved = pd.DataFrame(
        {"DILIST_ID": [1], "drug_name": ["a"], "smiles": ["CCO"], "canonical_smiles": ["CCO"]}
    )
    with pytest.raises(KeyError):
        build_canonical(dilist, bad_dilirank, resolved, set(), set())
