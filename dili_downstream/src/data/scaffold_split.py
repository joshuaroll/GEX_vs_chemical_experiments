"""Murcko scaffold split (SPLIT-01).

Pure library. Phase 2's CLI driver in `scripts/build_phase2_splits.py` consumes
this. No file I/O here.

Locked by 02-CONTEXT.md §"Locked by Q6 / Q12":
    - Primary split: Murcko scaffold, 80/10/10, stratified on `dili_binary`
    - Seed 42 (community convention; exposed via kwarg)

Locked by 02-CONTEXT.md planner_prelim_findings:
    #3: Acyclic drugs (scaffold == "") are treated as singleton scaffolds
        each, so they never merge into a single oversize cluster.
    #4: All output pert_ids are formatted as `f"DILIST_{n:04d}"` strings
        (explicit prefix avoids collision with PDG's `drug_name` keys in any
        future merged dataframe).

Algorithm (greedy stratified scaffold-group assignment):
    1. Group rows by `scaffold`. Empty-string scaffolds become per-row
       singleton groups keyed `f"_acyclic_{pert_id}"`.
    2. Sort groups by size descending, tie-broken by group key string for
       determinism.
    3. For each group in order, assign the entire group to the partition
       that is most under-target. Stratification: among partitions still
       under-target, prefer the one whose dili_binary class-balance
       deviation from the global rate would be reduced most by adding
       this group. Tie-break with `random.Random(seed)`.

The companion module `cluster_split.py` (Task 2) imports `_to_dilist_id`
from here for ID-format consistency.
"""

from __future__ import annotations

import logging
import random
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["scaffold_split", "_to_dilist_id"]


REQUIRED_COLUMNS: frozenset[str] = frozenset({"pert_id", "scaffold", "dili_binary"})


def _to_dilist_id(pid: int | "Iterable[int]") -> str:
    """Format a DILIst integer pert_id as `DILIST_NNNN`.

    Parameters
    ----------
    pid : int (or numpy.int64)
        The DILIst-supplied integer ID. Cast via `int(pid)` so numpy scalars
        also work.

    Returns
    -------
    str of the form `DILIST_0001`.
    """
    return f"DILIST_{int(pid):04d}"


def _validate_inputs(df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(
            f"scaffold_split: input df missing required columns: {sorted(missing)}. "
            f"Saw: {df.columns.tolist()}"
        )
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError(
            f"scaffold_split: train+val+test fractions must sum to 1.0, "
            f"got {train_frac}+{val_frac}+{test_frac}={train_frac + val_frac + test_frac}"
        )
    if df["dili_binary"].isna().any():
        raise ValueError("scaffold_split: dili_binary contains NaN — Phase 1 invariant violated.")


def _build_scaffold_groups(df: pd.DataFrame) -> list[tuple[str, list[int], list[int]]]:
    """Group rows by scaffold, treating acyclic ("") rows as singletons.

    Returns a list of `(group_key, pert_ids, labels)` tuples, sorted by
    `(-size, group_key)` for determinism. `pert_ids` is the int pert_id list
    for the group; `labels` aligns 1:1 with pert_ids.
    """
    groups: dict[str, tuple[list[int], list[int]]] = {}
    for pid, scaffold, label in zip(
        df["pert_id"].tolist(), df["scaffold"].tolist(), df["dili_binary"].tolist()
    ):
        # Treat NaN scaffolds as acyclic too (Phase 1 emits "" but be defensive).
        if scaffold is None or (isinstance(scaffold, float) and pd.isna(scaffold)) or scaffold == "":
            key = f"_acyclic_{int(pid)}"
        else:
            key = str(scaffold)
        if key not in groups:
            groups[key] = ([], [])
        groups[key][0].append(int(pid))
        groups[key][1].append(int(label))

    items = [(k, pids, labels) for k, (pids, labels) in groups.items()]
    # Sort by size descending, then by key ascending for determinism.
    items.sort(key=lambda t: (-len(t[1]), t[0]))
    return items


def _pick_partition(
    group_size: int,
    group_pos: int,
    fills: dict[str, int],
    pos_counts: dict[str, int],
    targets_total: dict[str, int],
    targets_pos: dict[str, int],
    n_total: int,
    global_pos_rate: float,
    rng: random.Random,
) -> str:
    """Choose the best partition for the next group.

    Combined score per partition (lower is better):
        size_deficit_after_add  +  pos_deficit_after_add
    where deficits are squared positive amounts the partition is *under*
    its size and class targets. This encourages the algorithm to fill the
    smallest partitions (val/test) first when they're under-target on either
    axis, while large groups still flow into train. Partitions at-or-over
    target are heavily penalized via a large constant offset so they're
    only chosen when no other partition is under-target.

    Tie-break with the seeded RNG over a sorted candidate list (determinism).
    """
    EXCESS_PENALTY = 10 * n_total * n_total  # any partition over-target gets pushed last

    def cost(p: str) -> float:
        new_fill = fills[p] + group_size
        new_pos = pos_counts[p] + group_pos
        # Size deficit: positive if under target, 0 if at/over.
        size_deficit_now = max(0, targets_total[p] - fills[p])
        size_deficit_after = max(0, targets_total[p] - new_fill)
        # Reward *reducing* size deficit (closing the gap).
        size_score = size_deficit_after ** 2
        # Penalize over-shooting size target.
        if new_fill > targets_total[p]:
            size_score += EXCESS_PENALTY * (new_fill - targets_total[p])
        # Class-balance: target_pos[p] is targets_total[p] * global_pos_rate.
        # Deficit penalized when partition is positively under-represented.
        pos_target = targets_pos[p]
        pos_deficit_after = (new_pos - pos_target) ** 2
        # Bonus for closing positive-class gap.
        return float(size_score + pos_deficit_after)

    scored = [(cost(p), p) for p in ("train", "val", "test")]
    min_cost = min(s for s, _ in scored)
    candidates = sorted([p for s, p in scored if s == min_cost])
    if len(candidates) == 1:
        return candidates[0]
    # Final RNG tie-break — preserves determinism via seeded rng.
    return rng.choice(candidates)


def scaffold_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> dict[str, list[str]]:
    """Murcko scaffold split, stratified on `dili_binary` (SPLIT-01).

    Required columns on `df`:
        pert_id (int), scaffold (str; "" for acyclic), dili_binary (int 0/1).

    Acyclic rows (scaffold == "") become per-row singletons so they never
    merge — see 02-CONTEXT.md planner_prelim_findings #3.

    Returns
    -------
    dict
        `{"train": [DILIST_NNNN, ...], "val": [...], "test": [...]}`.
        All values are sorted lexicographically inside each partition for
        downstream-consumer determinism.
    """
    _validate_inputs(df, train_frac, val_frac, test_frac)

    n_total = len(df)
    if n_total == 0:
        return {"train": [], "val": [], "test": []}

    targets_total: dict[str, int] = {
        "train": int(round(n_total * train_frac)),
        "val": int(round(n_total * val_frac)),
        "test": n_total - int(round(n_total * train_frac)) - int(round(n_total * val_frac)),
    }

    global_pos_rate = float(df["dili_binary"].mean())
    n_total_pos = int(df["dili_binary"].sum())
    targets_pos: dict[str, int] = {
        "train": int(round(n_total_pos * train_frac)),
        "val": int(round(n_total_pos * val_frac)),
        "test": n_total_pos - int(round(n_total_pos * train_frac)) - int(round(n_total_pos * val_frac)),
    }
    log.debug(
        "scaffold_split: n_total=%d targets=%s targets_pos=%s seed=%d",
        n_total, targets_total, targets_pos, seed,
    )

    rng = random.Random(seed)

    # Group rows by scaffold (acyclic = singleton).
    groups = _build_scaffold_groups(df)
    n_groups = len(groups)
    n_acyclic = sum(1 for k, _, _ in groups if k.startswith("_acyclic_"))
    log.debug(
        "scaffold_split: %d scaffold groups (of which %d acyclic singletons)",
        n_groups, n_acyclic,
    )

    # Greedy stratified assignment.
    fills: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    pos_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    placements: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for group_key, pids, labels in groups:
        group_size = len(pids)
        group_pos = sum(labels)
        partition = _pick_partition(
            group_size, group_pos, fills, pos_counts,
            targets_total, targets_pos, n_total, global_pos_rate, rng,
        )
        placements[partition].extend(pids)
        fills[partition] += group_size
        pos_counts[partition] += group_pos

    # Format pert_ids and sort each partition for downstream determinism.
    out: dict[str, list[str]] = {
        sname: sorted(_to_dilist_id(p) for p in placements[sname])
        for sname in ("train", "val", "test")
    }

    # Sanity log
    for sname in ("train", "val", "test"):
        n = len(out[sname])
        rate = pos_counts[sname] / fills[sname] if fills[sname] else float("nan")
        log.info(
            "scaffold_split %s: n=%d (%.1f%% of total), pos_rate=%.3f (global=%.3f)",
            sname, n, 100.0 * n / n_total, rate, global_pos_rate,
        )

    return out
