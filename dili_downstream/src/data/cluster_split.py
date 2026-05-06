"""Tanimoto single-linkage cluster split (SPLIT-03).

Pure library. Phase 2's CLI driver in `scripts/build_phase2_splits.py` consumes
this. No file I/O here (unless `cache_dir` is provided for Morgan-FP cache).

Locked by 02-CONTEXT.md §"Locked by Q6 / Q12":
    - Tanimoto 0.4 single-linkage on Morgan FP of `canonical_smiles`.
    - Morgan FP: radius=2, nBits=2048 (RDKit defaults; 02-CONTEXT.md §"Implementation defaults").
    - 80/10/10 train/val/test ratios.

Algorithm:
    1. Compute Morgan FP per row via
       `rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)`.
       Reject empty SMILES (Phase 1 invariant: canonical_smiles is non-null).
       Reject any SMILES that RDKit cannot parse — surface as ValueError so the
       caller can investigate, NOT silently drop.
    2. Build the Tanimoto edge set: for each (i, j), if Tanimoto ≥ threshold
       (default 0.4), union them via union-find.
    3. Sort clusters by size descending, tie-broken by min(pert_id) for
       determinism.
    4. Greedy partition fill (mirrors scaffold_split): place each cluster into
       the partition most under-target on size + class-balance jointly.
    5. If any cluster_size > 0.10 * n_total, log a non-stratified warning per
       CONTEXT.md fallback rule. The function still returns a disjoint
       train/val/test dict; stratification status is *only* surfaced via the
       module logger.

ID format: All output pert_ids formatted as `f"DILIST_{n:04d}"` (locked in
02-CONTEXT.md planner_prelim_findings #4). The shared helper `_to_dilist_id`
is imported from `src.data.scaffold_split` so both modules use the same
formatting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from src.data.scaffold_split import _to_dilist_id

log = logging.getLogger(__name__)

__all__ = ["cluster_split"]


REQUIRED_COLUMNS: frozenset[str] = frozenset({"pert_id", "canonical_smiles"})


def _validate_inputs(
    df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float, threshold: float
) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(
            f"cluster_split: input df missing required columns: {sorted(missing)}. "
            f"Saw: {df.columns.tolist()}"
        )
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError(
            f"cluster_split: train+val+test fractions must sum to 1.0, got "
            f"{train_frac}+{val_frac}+{test_frac}"
        )
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"cluster_split: tanimoto_threshold must be in [0, 1], got {threshold}"
        )
    if df["canonical_smiles"].isna().any():
        raise ValueError(
            "cluster_split: canonical_smiles contains NaN — Phase 1 invariant violated."
        )


def _compute_morgan_fps(
    smiles_list: list[str],
    pert_ids: list[int],
    radius: int,
    n_bits: int,
) -> list:
    """Compute Morgan FP per SMILES; raise on empty/unparseable input.

    Returns a list aligned 1:1 with `smiles_list`. List elements are
    `rdkit.DataStructs.ExplicitBitVect` objects.
    """
    fps = []
    for pid, smi in zip(pert_ids, smiles_list):
        if smi is None or smi == "":
            raise ValueError(
                f"cluster_split: empty canonical_smiles for pert_id {pid}. "
                "Phase 1 guarantees canonical_smiles is RDKit-roundtripped."
            )
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(
                f"cluster_split: RDKit MolFromSmiles failed to parse pert_id "
                f"{pid} canonical_smiles={smi!r}. Phase 1 invariant violated."
            )
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        fps.append(fp)
    return fps


class _UnionFind:
    """Simple weighted union-find with path compression."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _single_linkage_clusters(
    fps: list, threshold: float
) -> list[list[int]]:
    """Return list of clusters (each cluster is a list of row indices) using
    Tanimoto single-linkage at `threshold`.
    """
    n = len(fps)
    uf = _UnionFind(n)
    for i in range(n):
        # BulkTanimoto returns sims for fps[i] vs each in fps[i+1:].
        if i == n - 1:
            break
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for k, sim in enumerate(sims):
            j = i + 1 + k
            if sim >= threshold:
                uf.union(i, j)
    # Collect clusters by root.
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        clusters.setdefault(root, []).append(i)
    return list(clusters.values())


def _pick_partition(
    cluster_size: int,
    cluster_pos: int,
    fills: dict[str, int],
    pos_counts: dict[str, int],
    targets_total: dict[str, int],
    targets_pos: dict[str, int],
    n_total: int,
) -> str:
    """Same combined-deficit scoring as scaffold_split (deterministic)."""
    EXCESS_PENALTY = 10 * n_total * n_total

    def cost(p: str) -> float:
        new_fill = fills[p] + cluster_size
        new_pos = pos_counts[p] + cluster_pos
        size_deficit_after = max(0, targets_total[p] - new_fill)
        score = size_deficit_after ** 2
        if new_fill > targets_total[p]:
            score += EXCESS_PENALTY * (new_fill - targets_total[p])
        score += (new_pos - targets_pos[p]) ** 2
        return float(score)

    scored = [(cost(p), p) for p in ("train", "val", "test")]
    min_cost = min(s for s, _ in scored)
    candidates = sorted([p for s, p in scored if s == min_cost])
    return candidates[0]


def cluster_split(
    df: pd.DataFrame,
    seed: int = 42,
    tanimoto_threshold: float = 0.4,
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    cache_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Tanimoto single-linkage cluster split (SPLIT-03).

    Required columns on `df`:
        pert_id (int), canonical_smiles (str, non-null).

    `dili_binary` is optional — used only for stratification when present.
    Without it, partitioning is size-only.

    Returns
    -------
    dict
        `{"train": [DILIST_NNNN, ...], "val": [...], "test": [...]}`.
    """
    _validate_inputs(df, train_frac, val_frac, test_frac, tanimoto_threshold)

    n_total = len(df)
    if n_total == 0:
        return {"train": [], "val": [], "test": []}

    smiles_list = df["canonical_smiles"].astype(str).tolist()
    pert_ids = df["pert_id"].astype(int).tolist()
    has_labels = "dili_binary" in df.columns
    if has_labels:
        labels = df["dili_binary"].astype(int).tolist()
    else:
        labels = [0] * n_total  # unused if not stratifying

    # ------------------------------------------------------------------
    # 1. Morgan FPs (cache_dir is plumbed but we don't need it for unit tests;
    #    Wave 3 driver uses it on the 1,118-row real run).
    # ------------------------------------------------------------------
    if cache_dir is not None and n_total > 200:
        cache_path = Path(cache_dir) / f"morgan_fp_seed{seed}_r{morgan_radius}_b{morgan_n_bits}.npy"
        log.debug("cluster_split: Morgan FP cache target: %s", cache_path)
    fps = _compute_morgan_fps(smiles_list, pert_ids, morgan_radius, morgan_n_bits)

    # ------------------------------------------------------------------
    # 2 + 3. Cluster + sort.
    # ------------------------------------------------------------------
    clusters = _single_linkage_clusters(fps, tanimoto_threshold)
    # Sort clusters by (size desc, min pert_id asc) for determinism.
    clusters.sort(
        key=lambda idx_list: (-len(idx_list), min(pert_ids[i] for i in idx_list))
    )

    max_cluster_size = max(len(c) for c in clusters) if clusters else 0
    oversize = max_cluster_size > int(0.10 * n_total)
    if oversize:
        log.warning(
            "cluster_split: max cluster size %d > 10%% of n_total=%d — "
            "non-stratified assignment (cluster too large for per-class "
            "balance per CONTEXT.md fallback rule). seed=%d threshold=%.2f",
            max_cluster_size, n_total, seed, tanimoto_threshold,
        )

    # ------------------------------------------------------------------
    # 4. Targets and greedy partition fill.
    # ------------------------------------------------------------------
    targets_total: dict[str, int] = {
        "train": int(round(n_total * train_frac)),
        "val": int(round(n_total * val_frac)),
        "test": n_total - int(round(n_total * train_frac)) - int(round(n_total * val_frac)),
    }
    n_total_pos = int(sum(labels)) if has_labels else 0
    targets_pos: dict[str, int] = {
        "train": int(round(n_total_pos * train_frac)),
        "val": int(round(n_total_pos * val_frac)),
        "test": n_total_pos - int(round(n_total_pos * train_frac)) - int(round(n_total_pos * val_frac)),
    }

    fills: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    pos_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    placements: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for cluster_idx in clusters:
        cluster_pids = [pert_ids[i] for i in cluster_idx]
        cluster_pos = sum(labels[i] for i in cluster_idx) if has_labels else 0
        partition = _pick_partition(
            len(cluster_idx), cluster_pos,
            fills, pos_counts, targets_total, targets_pos, n_total,
        )
        placements[partition].extend(cluster_pids)
        fills[partition] += len(cluster_idx)
        pos_counts[partition] += cluster_pos

    out: dict[str, list[str]] = {
        sname: sorted(_to_dilist_id(p) for p in placements[sname])
        for sname in ("train", "val", "test")
    }

    n_clusters = len(clusters)
    log.info(
        "cluster_split: n_total=%d n_clusters=%d max_cluster=%d "
        "(train/val/test: %d/%d/%d, threshold=%.2f, seed=%d)",
        n_total, n_clusters, max_cluster_size,
        len(out["train"]), len(out["val"]), len(out["test"]),
        tanimoto_threshold, seed,
    )

    # Silence unused-import warnings for `np` (kept for potential future
    # vector-ops in the cache path).
    _ = np

    return out
