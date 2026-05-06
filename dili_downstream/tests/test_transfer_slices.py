"""Unit tests for `src/data/transfer_slices.py` (SPLIT-05).

In-memory pandas fixtures — no real-data dependency.

Behaviors covered (per 02-02-PLAN.md):
  1. Output dict has exact keys {test_in_pdg, test_drug_novel,
     test_drug_and_scaffold_novel}.
  2. All values are list[str] of DILIST_NNNN form.
  3. Partition: set(test_in_pdg) ∪ set(test_drug_novel) == set(test_pert_ids).
  4. Disjointness: set(test_in_pdg) ∩ set(test_drug_novel) == ∅.
  5. Subset: set(test_drug_and_scaffold_novel) ⊆ set(test_drug_novel).
  6. Acyclic-scaffold handling — empty-scaffold test drugs whose drug_name is
     not in PDG belong to test_drug_novel; their membership in
     test_drug_and_scaffold_novel depends on whether train_scaffolds contains
     "" (empty-scaffold treated as a single bucket at slice time).
  7. Determinism — sorted output, same input → identical output.
  8. Halt-gate metric helper — len(test_drug_novel) is the gate input.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from src.data.transfer_slices import compute_transfer_slices

DILIST_RE = re.compile(r"^DILIST_\d{4}$")


# ---------------------------------------------------------------------------
# Fixture: 20-row canonical CSV
#
# Layout (pert_id 1..20):
#   1..10  → train rows (drug_name not in test split below)
#   11..20 → test rows (10 drugs in scaffold_split test partition)
#
# in_pdg flag set on rows whose drug_name (lowercased) appears in
# `pdg_drug_names_set`. We control this explicitly per row.
#
# Scaffolds (string-equality only — no chem):
#   - "Sa", "Sb", "Sc"      → ring scaffolds shared with at least one train row
#   - "Sx", "Sy"            → ring scaffolds NOT in any train row (test-only)
#   - ""                    → acyclic
#
# Test rows (pert_id 11..20) cover:
#   - 11: in_pdg=True, scaffold "Sa" (in train) — test_in_pdg
#   - 12: in_pdg=True, scaffold "Sb" (in train) — test_in_pdg
#   - 13: in_pdg=False, scaffold "Sa" (in train) — drug_novel, NOT scaffold_novel
#   - 14: in_pdg=False, scaffold "Sx" (NOT in train) — drug_novel + scaffold_novel
#   - 15: in_pdg=False, scaffold "Sy" (NOT in train) — drug_novel + scaffold_novel
#   - 16: in_pdg=False, scaffold "Sc" (in train) — drug_novel, NOT scaffold_novel
#   - 17: in_pdg=False, scaffold "" (acyclic) — drug_novel; if "" in train_scaffolds, NOT scaffold_novel
#   - 18: in_pdg=False, scaffold "" (acyclic) — drug_novel; same as 17
#   - 19: in_pdg=True, scaffold "Sa" — test_in_pdg
#   - 20: in_pdg=False, scaffold "Sx" — drug_novel + scaffold_novel
# ---------------------------------------------------------------------------


@pytest.fixture
def dili_canonical_df() -> pd.DataFrame:
    rows = []
    # Train rows 1..10 — varied scaffolds, all unique drug_names not in test
    train_scaffolds = ["Sa", "Sb", "Sc", "Sa", "Sb", "Sc", "", "Sa", "Sb", "Sc"]
    for i in range(1, 11):
        rows.append({
            "pert_id": i,
            "drug_name": f"train_drug_{i}",
            "scaffold": train_scaffolds[i - 1],
            "in_pdg": (i % 2 == 0),  # arbitrary; not exercised in slicing
        })
    # Test rows 11..20
    test_layout = [
        # (pert_id, drug_name,    scaffold, in_pdg)
        (11, "drug_in_pdg_a",     "Sa",     True),
        (12, "drug_in_pdg_b",     "Sb",     True),
        (13, "drug_novel_a",      "Sa",     False),
        (14, "drug_novel_x",      "Sx",     False),
        (15, "drug_novel_y",      "Sy",     False),
        (16, "drug_novel_c",      "Sc",     False),
        (17, "drug_novel_acy_a",  "",       False),
        (18, "drug_novel_acy_b",  "",       False),
        (19, "drug_in_pdg_c",     "Sa",     True),
        (20, "drug_novel_x2",     "Sx",     False),
    ]
    for pid, name, scaf, in_pdg in test_layout:
        rows.append({
            "pert_id": pid,
            "drug_name": name,
            "scaffold": scaf,
            "in_pdg": in_pdg,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def test_pert_ids() -> list[str]:
    return [f"DILIST_{i:04d}" for i in range(11, 21)]


@pytest.fixture
def pdg_drug_names() -> set[str]:
    """Lowercased drug_names known to PDG. Matches in_pdg=True rows above."""
    return {"drug_in_pdg_a", "drug_in_pdg_b", "drug_in_pdg_c"}


@pytest.fixture
def train_scaffolds_with_acyclic() -> set[str]:
    """Train scaffolds set INCLUDING empty string (since pert_id=7 is acyclic)."""
    return {"Sa", "Sb", "Sc", ""}


@pytest.fixture
def train_scaffolds_no_acyclic() -> set[str]:
    """Train scaffolds set WITHOUT empty string — exercises the alternate
    branch where acyclic test drugs DO count as scaffold-novel."""
    return {"Sa", "Sb", "Sc"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_1_output_has_exact_keys(
    test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
):
    """Behavior 1: returned dict has exactly the 3 locked keys."""
    result = compute_transfer_slices(
        test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
    )
    assert set(result.keys()) == {
        "test_in_pdg", "test_drug_novel", "test_drug_and_scaffold_novel"
    }


def test_2_values_are_lists_of_dilist_ids(
    test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
):
    """Behavior 2: all three values are list[str] of DILIST_NNNN form."""
    result = compute_transfer_slices(
        test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
    )
    for k, v in result.items():
        assert isinstance(v, list), f"{k} must be list, got {type(v)}"
        for x in v:
            assert isinstance(x, str), f"{k} element {x!r} must be str"
            assert DILIST_RE.match(x), f"{k} element {x!r} is not DILIST_NNNN"


def test_3_partition_invariant(
    test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
):
    """Behavior 3: in_pdg ∪ drug_novel == all test pert_ids."""
    result = compute_transfer_slices(
        test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
    )
    union = set(result["test_in_pdg"]) | set(result["test_drug_novel"])
    assert union == set(test_pert_ids), \
        f"Union {sorted(union)} != all test_pert_ids {sorted(test_pert_ids)}"


def test_4_disjointness(
    test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
):
    """Behavior 4: in_pdg and drug_novel are disjoint."""
    result = compute_transfer_slices(
        test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
    )
    assert set(result["test_in_pdg"]).isdisjoint(set(result["test_drug_novel"]))


def test_5_subset_invariant(
    test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
):
    """Behavior 5: test_drug_and_scaffold_novel ⊆ test_drug_novel."""
    result = compute_transfer_slices(
        test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
    )
    assert set(result["test_drug_and_scaffold_novel"]).issubset(
        set(result["test_drug_novel"])
    ), \
        f"strongest slice not subset of drug_novel: " \
        f"{result['test_drug_and_scaffold_novel']} vs {result['test_drug_novel']}"


def test_6a_acyclic_with_train_acyclic_not_scaffold_novel(
    test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
):
    """Behavior 6a: when train_scaffolds includes "", acyclic test drugs are
    drug_novel but NOT scaffold_novel."""
    result = compute_transfer_slices(
        test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
    )
    # pert_ids 17, 18 are acyclic and drug-novel. With train scaffolds containing
    # "", they should NOT be in the strongest slice.
    assert "DILIST_0017" in result["test_drug_novel"]
    assert "DILIST_0018" in result["test_drug_novel"]
    assert "DILIST_0017" not in result["test_drug_and_scaffold_novel"], \
        "Acyclic test drug with '' in train_scaffolds must NOT be scaffold-novel"
    assert "DILIST_0018" not in result["test_drug_and_scaffold_novel"]


def test_6b_acyclic_without_train_acyclic_is_scaffold_novel(
    test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_no_acyclic
):
    """Behavior 6b: when train_scaffolds does NOT contain "", acyclic test drugs
    ARE scaffold-novel."""
    result = compute_transfer_slices(
        test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_no_acyclic
    )
    assert "DILIST_0017" in result["test_drug_and_scaffold_novel"]
    assert "DILIST_0018" in result["test_drug_and_scaffold_novel"]


def test_7_determinism_and_sorted(
    test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
):
    """Behavior 7: sorted output, same input → identical output (twice)."""
    a = compute_transfer_slices(
        test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
    )
    b = compute_transfer_slices(
        test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
    )
    assert a == b
    # Each list must be sorted lexicographically.
    for k, v in a.items():
        assert v == sorted(v), f"{k} not sorted: {v}"


def test_8_halt_gate_metric(
    test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
):
    """Behavior 8: len(test_drug_novel) is the halt-gate input."""
    result = compute_transfer_slices(
        test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
    )
    # Per fixture: pert_ids 13, 14, 15, 16, 17, 18, 20 are drug-novel → 7
    assert len(result["test_drug_novel"]) == 7
    # Sanity: test_in_pdg should have 11, 12, 19 → 3
    assert len(result["test_in_pdg"]) == 3
    # Sum equals total test pert_ids.
    assert (len(result["test_drug_novel"]) + len(result["test_in_pdg"])
            == len(test_pert_ids))


def test_9_specific_pert_ids_in_correct_slice(
    test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
):
    """Sanity: explicit per-row placement matches the fixture layout doc."""
    result = compute_transfer_slices(
        test_pert_ids, dili_canonical_df, pdg_drug_names, train_scaffolds_with_acyclic
    )
    # in_pdg slice: pert_ids 11, 12, 19
    assert set(result["test_in_pdg"]) == {"DILIST_0011", "DILIST_0012", "DILIST_0019"}
    # drug_novel slice: pert_ids 13, 14, 15, 16, 17, 18, 20
    assert set(result["test_drug_novel"]) == {
        "DILIST_0013", "DILIST_0014", "DILIST_0015",
        "DILIST_0016", "DILIST_0017", "DILIST_0018", "DILIST_0020",
    }
    # scaffold_novel slice: drug_novel rows whose scaffold ∉ train_scaffolds.
    # train_scaffolds = {"Sa", "Sb", "Sc", ""}. Sx and Sy are novel.
    # → 14, 15, 20
    assert set(result["test_drug_and_scaffold_novel"]) == {
        "DILIST_0014", "DILIST_0015", "DILIST_0020",
    }
