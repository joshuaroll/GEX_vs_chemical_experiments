"""Unit tests for `src/data/upstream_filter.py` (SPLIT-02).

In-memory pandas fixtures — no real-data dependency.

Behaviors covered (per 02-02-PLAN.md):
  1. `filter_upstream_train(...)` returns `(list[str], dict)` tuple.
  2. `train_upstream` contains drug_name strings (NOT DILIST_xxxx — PDG keys
     remain drug_name per planner_prelim_findings #2).
  3. PDG drug whose Murcko scaffold matches a S_test scaffold is excluded.
  4. Defense in depth — PDG drug whose drug_name matches a DILIst-test row's
     drug_name (case-insensitive) is excluded even if its scaffold doesn't
     match S_test.
  5. Diagnostics dict has exactly the locked 8 keys, all integer values.
  6. Determinism — same inputs twice → identical outputs.
  7. Acyclic ("" scaffold) PDG drugs are NOT auto-excluded.
  8. Returned `train_upstream` list is sorted (deterministic).
  9. `_compute_pdg_scaffolds` builds drug_name → scaffold map; invalid SMILES
     produce "" with a warning, drug NOT excluded by scaffold rule.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.data.upstream_filter import (
    _compute_pdg_scaffolds,
    filter_upstream_train,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Real RDKit-canonicalizable SMILES + their Murcko scaffolds (precomputed for
# fixture transparency). All scaffolds here are the *real* MurckoScaffoldSmiles
# output for the given SMILES, so equality testing reflects the production code
# path rather than a mock.
#
#   "c1ccccc1"     (benzene)             → scaffold "c1ccccc1"
#   "c1ccncc1"     (pyridine)            → scaffold "c1ccncc1"
#   "c1ccc(O)cc1"  (phenol)              → scaffold "c1ccccc1"   (same as benzene)
#   "c1ccc2ccccc2c1" (naphthalene)       → scaffold "c1ccc2ccccc2c1"
#   "CCO"          (ethanol — acyclic)   → scaffold "" (acyclic)
#   "CCC"          (propane — acyclic)   → scaffold "" (acyclic)
#
# These are what the *fixture rows* are expected to compute via
# MurckoScaffoldSmiles. Tests exercise:
#   - exact-string scaffold match
#   - acyclic non-exclusion
#   - drug_name-based defense-in-depth match


@pytest.fixture
def pdg_drugs_df() -> pd.DataFrame:
    """5-row PDG drugs fixture with mixed scaffolds.

    Layout:
      - alpha:    benzene  → scaffold "c1ccccc1"          (matches phenol scaffold below)
      - beta:     pyridine → scaffold "c1ccncc1"          (NO scaffold match)
      - gamma:    naphthalene → scaffold "c1ccc2ccccc2c1" (matches naphthalene test row)
      - delta:    ethanol  → scaffold ""                  (acyclic — never matched by scaffold rule)
      - aspirin:  benzene-acetic edge case (drug-name defense match — see below)
                  scaffold "c1ccccc1" (also matches phenol scaffold, but we'll
                  isolate via fixture #5 the "drug_name match without scaffold
                  match" path).
    """
    return pd.DataFrame([
        {"drug_name": "alpha",   "cpd_smiles": "c1ccccc1"},
        {"drug_name": "beta",    "cpd_smiles": "c1ccncc1"},
        {"drug_name": "gamma",   "cpd_smiles": "c1ccc2ccccc2c1"},
        {"drug_name": "delta",   "cpd_smiles": "CCO"},
        {"drug_name": "aspirin", "cpd_smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    ])


@pytest.fixture
def dili_canonical_df() -> pd.DataFrame:
    """10-row DILIst canonical fixture with mixed scaffolds + drug_names.

    pert_ids 1..10. drug_names lowercased to match Phase 1 convention.
    """
    return pd.DataFrame([
        # In our scaffold split, 1-7 will be "train", 8-10 will be "test".
        # pert_id=1: phenol — scaffold "c1ccccc1" (will be in TRAIN, irrelevant for exclusion)
        {"pert_id": 1, "drug_name": "compound_train_1", "scaffold": "c1ccccc1"},
        {"pert_id": 2, "drug_name": "compound_train_2", "scaffold": "c1ccncc1"},
        {"pert_id": 3, "drug_name": "compound_train_3", "scaffold": "c1ccc2ccccc2c1"},
        {"pert_id": 4, "drug_name": "compound_train_4", "scaffold": "c1ccccc1"},
        {"pert_id": 5, "drug_name": "compound_train_5", "scaffold": ""},
        {"pert_id": 6, "drug_name": "compound_train_6", "scaffold": "c1ccncc1"},
        {"pert_id": 7, "drug_name": "compound_train_7", "scaffold": "c1ccc2ccccc2c1"},
        # TEST rows — these define S_test (and DILIst-test drug_names).
        # pert_id=8: a phenol-derived test drug — scaffold "c1ccccc1" → matches alpha + aspirin (benzene scaffold)
        {"pert_id": 8, "drug_name": "compound_test_phenol", "scaffold": "c1ccccc1"},
        # pert_id=9: a naphthalene test drug — scaffold "c1ccc2ccccc2c1" → matches gamma
        {"pert_id": 9, "drug_name": "compound_test_napht", "scaffold": "c1ccc2ccccc2c1"},
        # pert_id=10: an acyclic test drug whose drug_name happens to be "beta"
        # (case-insensitive) — exercises the defense-in-depth drug_name match
        # without scaffold match. Note: scaffold here is "" (acyclic), so it's
        # NOT contributed to S_test (acyclic test rows are skipped per algorithm).
        {"pert_id": 10, "drug_name": "Beta",            "scaffold": ""},
    ])


@pytest.fixture
def scaffold_split_dict() -> dict[str, list[str]]:
    """Synthetic scaffold split. Test rows are pert_ids 8, 9, 10 → DILIST_0008..0010."""
    return {
        "train": [f"DILIST_{i:04d}" for i in range(1, 8)],
        "val": [],
        "test": ["DILIST_0008", "DILIST_0009", "DILIST_0010"],
    }


# ---------------------------------------------------------------------------
# Tests for _compute_pdg_scaffolds
# ---------------------------------------------------------------------------


def test_1_compute_pdg_scaffolds_returns_drug_name_keyed_map(pdg_drugs_df):
    """Behavior 9: helper returns drug_name → scaffold map for valid SMILES."""
    result = _compute_pdg_scaffolds(pdg_drugs_df)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"alpha", "beta", "gamma", "delta", "aspirin"}
    # Spot-check: benzene scaffold is "c1ccccc1"
    assert result["alpha"] == "c1ccccc1"
    # Acyclic ethanol → ""
    assert result["delta"] == ""
    # Naphthalene → "c1ccc2ccccc2c1"
    assert result["gamma"] == "c1ccc2ccccc2c1"


def test_2_compute_pdg_scaffolds_invalid_smiles_logs_and_empty(caplog):
    """Behavior 9: invalid SMILES → scaffold "" with logged warning."""
    df = pd.DataFrame([
        {"drug_name": "good",   "cpd_smiles": "c1ccccc1"},
        {"drug_name": "broken", "cpd_smiles": "not_valid_smiles_xyz"},
    ])
    with caplog.at_level(logging.WARNING):
        result = _compute_pdg_scaffolds(df)
    assert result["good"] == "c1ccccc1"
    assert result["broken"] == ""
    # Warning should mention the bad drug or its SMILES
    assert any("broken" in rec.message or "not_valid_smiles_xyz" in rec.message
               for rec in caplog.records), \
        f"Expected warning mentioning 'broken' or its SMILES, got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Tests for filter_upstream_train
# ---------------------------------------------------------------------------


def test_3_returns_tuple_list_dict(
    pdg_drugs_df, scaffold_split_dict, dili_canonical_df
):
    """Behavior 1: returns (list[str], dict)."""
    test_pids = scaffold_split_dict["test"]
    result = filter_upstream_train(
        pdg_drugs_df, scaffold_split_dict, test_pids, dili_canonical_df
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    train_upstream, diagnostics = result
    assert isinstance(train_upstream, list)
    assert all(isinstance(x, str) for x in train_upstream)
    assert isinstance(diagnostics, dict)


def test_4_train_upstream_contains_drug_names_not_dilist_ids(
    pdg_drugs_df, scaffold_split_dict, dili_canonical_df
):
    """Behavior 2: train_upstream contains drug_name strings, NOT DILIST_xxxx."""
    test_pids = scaffold_split_dict["test"]
    train_upstream, _ = filter_upstream_train(
        pdg_drugs_df, scaffold_split_dict, test_pids, dili_canonical_df
    )
    # No element should match DILIST_NNNN pattern.
    assert not any(x.startswith("DILIST_") for x in train_upstream), \
        f"PDG keys must remain drug_name strings, got: {train_upstream}"
    # Every element must be one of the original PDG drug_names.
    pdg_names = set(pdg_drugs_df["drug_name"])
    assert set(train_upstream).issubset(pdg_names)


def test_5_excludes_pdg_drug_with_scaffold_in_s_test(
    pdg_drugs_df, scaffold_split_dict, dili_canonical_df
):
    """Behavior 3: PDG drug 'gamma' (naphthalene) matches S_test naphthalene
    scaffold (from pert_id=9), so gamma is excluded."""
    test_pids = scaffold_split_dict["test"]
    train_upstream, diag = filter_upstream_train(
        pdg_drugs_df, scaffold_split_dict, test_pids, dili_canonical_df
    )
    assert "gamma" not in train_upstream, \
        "gamma should be excluded — its naphthalene scaffold matches a S_test scaffold"
    assert diag["scaffold_match_count"] >= 1


def test_6_drug_name_defense_in_depth(
    pdg_drugs_df, scaffold_split_dict, dili_canonical_df
):
    """Behavior 4: PDG 'beta' (pyridine — scaffold NOT in S_test directly,
    since S_test acyclic row pert_id=10 is skipped) is STILL excluded because
    its drug_name (lowercased) matches DILIst-test row pert_id=10's drug_name
    'Beta' (lowercased to 'beta')."""
    test_pids = scaffold_split_dict["test"]
    train_upstream, diag = filter_upstream_train(
        pdg_drugs_df, scaffold_split_dict, test_pids, dili_canonical_df
    )
    # pyridine "c1ccncc1" is NOT in S_test (S_test = {benzene, naphthalene} after
    # acyclic-skip). So 'beta' is only excluded via drug_name match.
    assert "beta" not in train_upstream, \
        "beta should be excluded by drug_name defense in depth, not scaffold match"
    assert diag["drug_name_match_count"] >= 1


def test_7_diagnostics_keys_and_int_types(
    pdg_drugs_df, scaffold_split_dict, dili_canonical_df
):
    """Behavior 5: diagnostics has exactly the locked 8 keys, all int values."""
    test_pids = scaffold_split_dict["test"]
    _, diag = filter_upstream_train(
        pdg_drugs_df, scaffold_split_dict, test_pids, dili_canonical_df
    )
    expected_keys = {
        "d_pdg_total",
        "excluded_by_scaffold",
        "excluded_by_pert_id",
        "excluded_intersection",
        "d_pdg_train_after_exclusion",
        "s_test_size",
        "scaffold_match_count",
        "drug_name_match_count",
    }
    assert set(diag.keys()) == expected_keys, \
        f"Diagnostic keys mismatch: got {set(diag.keys())}, expected {expected_keys}"
    for k, v in diag.items():
        assert isinstance(v, int), f"diag[{k!r}] must be int, got {type(v).__name__} = {v!r}"


def test_8_determinism(pdg_drugs_df, scaffold_split_dict, dili_canonical_df):
    """Behavior 6: same inputs → identical outputs."""
    test_pids = scaffold_split_dict["test"]
    a_train, a_diag = filter_upstream_train(
        pdg_drugs_df, scaffold_split_dict, test_pids, dili_canonical_df
    )
    b_train, b_diag = filter_upstream_train(
        pdg_drugs_df, scaffold_split_dict, test_pids, dili_canonical_df
    )
    assert a_train == b_train
    assert a_diag == b_diag


def test_9_acyclic_pdg_drug_not_auto_excluded(
    scaffold_split_dict, dili_canonical_df
):
    """Behavior 7: a PDG drug with scaffold "" (acyclic) is NOT auto-excluded
    just for being acyclic. Only excluded if drug_name happens to be in
    DILIst-test (here it's not)."""
    pdg = pd.DataFrame([
        # An acyclic drug with a name NOT in DILIst-test → must survive.
        {"drug_name": "harmless_acyclic", "cpd_smiles": "CCC"},
        # A control row with a non-matching ring scaffold.
        {"drug_name": "control",          "cpd_smiles": "c1ccoc1"},  # furan, not in S_test
    ])
    test_pids = scaffold_split_dict["test"]
    train_upstream, diag = filter_upstream_train(
        pdg, scaffold_split_dict, test_pids, dili_canonical_df
    )
    assert "harmless_acyclic" in train_upstream, \
        "Acyclic PDG drug not in DILIst-test must survive the filter"


def test_10_train_upstream_is_sorted(
    pdg_drugs_df, scaffold_split_dict, dili_canonical_df
):
    """Behavior 8: returned list is sorted (deterministic ordering)."""
    test_pids = scaffold_split_dict["test"]
    train_upstream, _ = filter_upstream_train(
        pdg_drugs_df, scaffold_split_dict, test_pids, dili_canonical_df
    )
    assert train_upstream == sorted(train_upstream), \
        f"train_upstream must be sorted, got {train_upstream}"


def test_11_intersection_count_consistent(
    pdg_drugs_df, scaffold_split_dict, dili_canonical_df
):
    """Sanity invariant: |intersection| ≤ min(|excluded_by_scaffold|, |excluded_by_pert_id|).

    Also: |excluded_by_scaffold| + |excluded_by_pert_id| - |intersection|
          = total unique excluded drugs.
    """
    test_pids = scaffold_split_dict["test"]
    _, diag = filter_upstream_train(
        pdg_drugs_df, scaffold_split_dict, test_pids, dili_canonical_df
    )
    assert diag["excluded_intersection"] <= diag["excluded_by_scaffold"]
    assert diag["excluded_intersection"] <= diag["excluded_by_pert_id"]
    # d_pdg_train_after_exclusion + (union of two exclusion sets) == d_pdg_total
    union_excluded = (diag["excluded_by_scaffold"] + diag["excluded_by_pert_id"]
                      - diag["excluded_intersection"])
    assert diag["d_pdg_train_after_exclusion"] + union_excluded == diag["d_pdg_total"]
