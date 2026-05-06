"""Unit tests for `src/data/scaffold_split.py` (SPLIT-01).

In-memory pandas fixtures — no real-data dependency.

Behaviors covered (per 02-01-PLAN.md):
  1. Output keys exactly {"train", "val", "test"}.
  2. All values are list[str], every element matches `^DILIST_\\d{4}$`.
  3. Determinism — twice with same args yields identical dict.
  4. Disjointness — pairwise empty intersection.
  5. Scaffold disjointness — no scaffold crosses train/test (acyclic
     drugs treated as singleton scaffolds, never merged).
  6. Stratification — per-partition class rate within ±5pp of global.
  7. Empty-scaffold handling — acyclic drugs placed deterministically.
  8. 80/10/10 ratios within tolerance on 50-row fixture.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from src.data.scaffold_split import _to_dilist_id, scaffold_split

DILIST_RE = re.compile(r"^DILIST_\d{4}$")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_50() -> pd.DataFrame:
    """50 rows, 25/25 class split, varied scaffold counts so stratification
    is exercisable. Includes 6 acyclic rows (scaffold == "") so empty-scaffold
    handling is exercised.

    Layout:
      - scaffold "Sa": 8 rows (4 pos / 4 neg)  — large group, will dominate one partition
      - scaffold "Sb": 6 rows (3 pos / 3 neg)
      - scaffold "Sc": 5 rows (3 pos / 2 neg)
      - scaffold "Sd": 4 rows (2 pos / 2 neg)
      - scaffold "Se": 4 rows (2 pos / 2 neg)
      - scaffold "Sf": 3 rows (2 pos / 1 neg)
      - scaffold "Sg": 3 rows (1 pos / 2 neg)
      - scaffold "Sh": 3 rows (2 pos / 1 neg)
      - scaffold "Si": 3 rows (1 pos / 2 neg)
      - scaffold "Sj": 3 rows (2 pos / 1 neg)
      - scaffold "Sk": 2 rows (1 pos / 1 neg)
      - acyclic ""  : 6 rows (2 pos / 4 neg)  — six singletons
      Total: 50 rows, 25 pos, 25 neg.
    """
    rows = []
    pid = 1

    def add_group(scaffold: str, pos: int, neg: int):
        nonlocal pid
        for _ in range(pos):
            rows.append({"pert_id": pid, "scaffold": scaffold, "dili_binary": 1})
            pid += 1
        for _ in range(neg):
            rows.append({"pert_id": pid, "scaffold": scaffold, "dili_binary": 0})
            pid += 1

    add_group("Sa", 4, 4)
    add_group("Sb", 3, 3)
    add_group("Sc", 3, 2)
    add_group("Sd", 2, 2)
    add_group("Se", 2, 2)
    add_group("Sf", 2, 1)
    add_group("Sg", 1, 2)
    add_group("Sh", 2, 1)
    add_group("Si", 1, 2)
    add_group("Sj", 2, 1)
    add_group("Sk", 1, 1)
    # 6 acyclic rows — 2 pos / 4 neg
    add_group("", 2, 4)

    df = pd.DataFrame(rows)
    assert len(df) == 50
    assert int(df["dili_binary"].sum()) == 25
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_helper_to_dilist_id_zero_pads():
    """The DILIST_{n:04d} helper format is locked in CONTEXT.md planner_prelim_findings #4."""
    assert _to_dilist_id(1) == "DILIST_0001"
    assert _to_dilist_id(42) == "DILIST_0042"
    assert _to_dilist_id(1234) == "DILIST_1234"
    # Numpy int64 should also work (pandas pert_id dtype)
    import numpy as np
    assert _to_dilist_id(np.int64(7)) == "DILIST_0007"


def test_1_keys_exactly_train_val_test(df_50):
    out = scaffold_split(df_50, seed=42)
    assert set(out.keys()) == {"train", "val", "test"}, (
        f"Expected exactly {{'train','val','test'}}, got {set(out.keys())}"
    )


def test_2_values_are_dilist_id_strings(df_50):
    out = scaffold_split(df_50, seed=42)
    for split_name in ("train", "val", "test"):
        assert isinstance(out[split_name], list), f"{split_name} not a list"
        for pid in out[split_name]:
            assert isinstance(pid, str), f"{split_name} contains non-str: {pid!r}"
            assert DILIST_RE.match(pid), (
                f"{split_name} pert_id {pid!r} does not match DILIST_NNNN pattern"
            )


def test_3_determinism(df_50):
    a = scaffold_split(df_50, seed=42)
    b = scaffold_split(df_50, seed=42)
    assert a == b, "scaffold_split must be deterministic given seed"


def test_4_disjointness(df_50):
    out = scaffold_split(df_50, seed=42)
    train, val, test = set(out["train"]), set(out["val"]), set(out["test"])
    assert train & val == set(), f"train ∩ val: {train & val}"
    assert train & test == set(), f"train ∩ test: {train & test}"
    assert val & test == set(), f"val ∩ test: {val & test}"
    # And every input pert_id appears somewhere
    all_ids = train | val | test
    expected = {_to_dilist_id(p) for p in df_50["pert_id"].tolist()}
    assert all_ids == expected, "Some pert_ids missing or extra"


def test_5_scaffold_disjointness(df_50):
    """No scaffold crosses train/test boundary (acyclic = singleton scaffolds,
    so two acyclic drugs CAN be in different partitions even though both have
    scaffold==""; the assertion is per-pert_id, not on the literal "" string)."""
    out = scaffold_split(df_50, seed=42)
    pid_to_scaffold = dict(
        zip(
            (_to_dilist_id(p) for p in df_50["pert_id"]),
            df_50["scaffold"].tolist(),
        )
    )
    train_scaffolds = {pid_to_scaffold[p] for p in out["train"] if pid_to_scaffold[p] != ""}
    test_scaffolds = {pid_to_scaffold[p] for p in out["test"] if pid_to_scaffold[p] != ""}
    overlap = train_scaffolds & test_scaffolds
    assert overlap == set(), (
        f"Non-acyclic scaffolds appearing in both train and test: {overlap}"
    )
    # Per CONTEXT.md planner_prelim_findings #3: acyclic = singleton scaffold each.
    # That means individual acyclic pert_ids can land in different partitions
    # (NOT all forced into one partition). Verify at least the disjointness of
    # pert_id sets handles this — already covered in test_4. No further assertion
    # here beyond non-acyclic-scaffold disjointness above.


def test_6_stratification_within_5pp(df_50):
    """Stratification within ±5pp on the train partition (the only one with
    enough slots — val/test only have 5 each, so integer cardinality forces
    minimum ±10pp deviation on a 50/50 fixture). The 5pp invariant in
    02-CONTEXT.md applies on the real 1,118-row dataset.
    """
    out = scaffold_split(df_50, seed=42)
    pid_to_label = dict(
        zip(
            (_to_dilist_id(p) for p in df_50["pert_id"]),
            df_50["dili_binary"].tolist(),
        )
    )
    global_rate = df_50["dili_binary"].mean()  # 0.5

    # Train: must be within ±5pp (40 slots, plenty of room).
    train_rate = sum(pid_to_label[p] for p in out["train"]) / len(out["train"])
    assert abs(train_rate - global_rate) <= 0.05, (
        f"train rate {train_rate:.3f} more than 5pp from global {global_rate:.3f}"
    )
    # Val/test: tolerance ±15pp because cardinality (5 slots each) makes ±5pp
    # mathematically infeasible on a 50/50 fixture (closest is 40% or 60%).
    # On a real 100-row test partition, the 5pp rule is realistic.
    for split_name in ("val", "test"):
        ids = out[split_name]
        if not ids:
            continue
        rate = sum(pid_to_label[p] for p in ids) / len(ids)
        assert abs(rate - global_rate) <= 0.15, (
            f"{split_name} rate {rate:.3f} more than 15pp from global "
            f"{global_rate:.3f} (small-partition tolerance)"
        )


def test_7_empty_scaffold_handled_deterministically(df_50):
    """Acyclic (scaffold=='') rows are placed; the function does not crash and
    each acyclic row is placed exactly once. Two consecutive calls yield the
    same placement for each acyclic row."""
    out_a = scaffold_split(df_50, seed=42)
    out_b = scaffold_split(df_50, seed=42)
    acyclic_ids = {
        _to_dilist_id(p)
        for p, sc in zip(df_50["pert_id"], df_50["scaffold"])
        if sc == ""
    }
    # Each acyclic id is in exactly one partition (already covered by test_4)
    # but verify each appears somewhere.
    placed = set(out_a["train"]) | set(out_a["val"]) | set(out_a["test"])
    assert acyclic_ids.issubset(placed), "Some acyclic rows not placed"
    # Determinism for acyclic placements specifically.
    for sname in ("train", "val", "test"):
        assert {x for x in out_a[sname] if x in acyclic_ids} == {
            x for x in out_b[sname] if x in acyclic_ids
        }


def test_8_ratios_within_tolerance(df_50):
    out = scaffold_split(df_50, seed=42)
    n_total = sum(len(out[s]) for s in ("train", "val", "test"))
    assert n_total == 50
    train_frac = len(out["train"]) / n_total
    val_frac = len(out["val"]) / n_total
    test_frac = len(out["test"]) / n_total
    assert 0.75 <= train_frac <= 0.85, f"train_frac={train_frac:.3f} not in [0.75, 0.85]"
    assert 0.05 <= val_frac <= 0.15, f"val_frac={val_frac:.3f} not in [0.05, 0.15]"
    assert 0.05 <= test_frac <= 0.15, f"test_frac={test_frac:.3f} not in [0.05, 0.15]"


def test_9_missing_required_columns_raises():
    """Defensive: missing pert_id/scaffold/dili_binary → KeyError-ish."""
    bad_df = pd.DataFrame({"pert_id": [1, 2], "scaffold": ["A", "B"]})  # no dili_binary
    with pytest.raises((KeyError, ValueError)):
        scaffold_split(bad_df, seed=42)


def test_10_seed_changes_partition(df_50):
    """Different seeds may produce different partitions (but each is deterministic).

    Note: stratified-greedy assignment may produce identical partitions for some
    seeds because the algorithm is largely deterministic (only tie-breaks use
    the RNG). What we verify is that calling with the SAME seed twice gives
    identical output — already covered in test_3; here we just sanity-check that
    seed is plumbed through (no exception when changed)."""
    out_42 = scaffold_split(df_50, seed=42)
    out_7 = scaffold_split(df_50, seed=7)
    # Both should be valid partitions of the same 50 ids.
    for out in (out_42, out_7):
        all_ids = set(out["train"]) | set(out["val"]) | set(out["test"])
        assert len(all_ids) == 50
