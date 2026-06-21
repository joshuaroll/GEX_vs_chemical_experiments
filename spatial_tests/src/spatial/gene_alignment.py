"""Pure library: align a gene-expression profile to a target gene space.

Phase deliverable: GEX-profile alignment utility consumed by signature caching
(src/signatures/) and downstream classifier feature assembly (src/classifiers/).

This module handles the mapping step that precedes any DE computation:
    raw_profile (dict/array over source_genes) -> array ordered by target_genes

Alignment policy (locked):
    1. Intersection: genes present in both source and target are filled from
       the profile.
    2. Missing genes (in target but not source): filled per `missing` policy —
       either 0.0 ('zero') or NaN ('nan').
    3. Extra genes (in source but not target): silently dropped.
    4. Duplicate symbols in source_genes: resolved per `duplicates` policy —
       either first occurrence ('first') or arithmetic mean ('mean').

There is NO imputation model here — intersection/reindex only. For profiles
with low coverage (e.g., an L1000-landmark profile aligned to the full N_PDG
gene space), the many missing genes will be filled with the `missing` value.

# TODO(landmark-imputation): when upstream predicted signatures are L1000
# (978 genes) and the target is the full N_PDG (10716) gene space, a learned
# imputation model (e.g., Achilles CRISPR latent-space predictor or a simple
# linear regression trained on LINCS bulk RNA) should fill the 9738 unmeasured
# genes rather than using zeros/NaN. This seam is intentional — see
# 09_spatial_decisions.md (in 0_project_documents/downstream_tasks/) for the decision
# log on whether landmark→full-genome imputation is required, and which model
# to use.

DE convention:
    This module does NOT perform DE computation. The caller is responsible for
    computing  predicted_DE_region = predicted_treated(drug, region_basal) -
    region_basal  BEFORE calling align_to_gene_space, or AFTER alignment
    (both are equivalent, since alignment is a linear reindexing).

Hard rules honored:
    - Pure library: NO hardcoded absolute paths, NO real-data filenames.
    - No mock or synthetic labels — the caller provides real arrays.
    - Gene space constants (N_LANDMARK=978, N_PDG=10716) are imported from
      src/spatial/config.py but are NOT enforced here (the caller chooses
      the target gene space; this module is generic).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["align_to_gene_space", "coverage_fraction"]

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Profile input: either a dict {gene_symbol -> value} or a 1-D array paired
# with source_genes. Numpy arrays are preferred for downstream callers.
_MissingPolicy = Literal["zero", "nan"]
_DuplicatePolicy = Literal["first", "mean"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def coverage_fraction(
    source_genes: list[str],
    target_genes: list[str],
) -> float:
    """Fraction of target genes covered by source_genes (intersection / target).

    Duplicate symbols in either list are deduplicated before the computation,
    matching the effective coverage seen after alignment.

    Parameters
    ----------
    source_genes : list[str]
        Gene symbols available in the profile being aligned.
    target_genes : list[str]
        Gene symbols of the target space (defines the denominator).

    Returns
    -------
    float
        Value in [0.0, 1.0]. Returns 0.0 if target_genes is empty.

    Examples
    --------
    >>> coverage_fraction(["A", "B", "C"], ["B", "C", "D"])
    0.6666666666666666
    """
    if len(target_genes) == 0:
        return 0.0
    source_set = set(source_genes)
    target_set = set(target_genes)
    n_covered = len(source_set & target_set)
    return n_covered / len(target_set)


def align_to_gene_space(
    profile: np.ndarray | dict[str, float],
    source_genes: list[str],
    target_genes: list[str],
    *,
    missing: _MissingPolicy = "zero",
    duplicates: _DuplicatePolicy = "first",
) -> np.ndarray:
    """Align a gene-expression profile to a target gene ordering.

    Parameters
    ----------
    profile : np.ndarray or dict[str, float]
        If ndarray: shape (len(source_genes),), dtype float32 or float64.
        If dict: maps gene symbol -> float value. When dict is provided,
        `source_genes` is used to define the canonical order of keys (any
        keys present in the dict but absent from `source_genes` are ignored,
        matching the "extra genes dropped" policy).
    source_genes : list[str]
        Gene symbols for `profile`. Must satisfy len(source_genes) ==
        len(profile) when `profile` is a numpy array. Symbols may include
        duplicates (resolved per `duplicates` policy).
    target_genes : list[str]
        Desired output gene ordering. May contain symbols not in
        `source_genes` (filled per `missing` policy). Duplicates in
        `target_genes` are an error (raises ValueError) — the caller should
        pass a deduplicated target list.
    missing : {'zero', 'nan'}
        Fill value for target genes absent in source_genes:
            'zero' — fill with 0.0 (safe for DE computation; zero
                     differential expression for unmeasured genes).
            'nan'  — fill with NaN (caller must handle NaN before linear ops).
    duplicates : {'first', 'mean'}
        Resolution policy for duplicate gene symbols in source_genes:
            'first' — use the value at the first occurrence.
            'mean'  — use the arithmetic mean across all occurrences.

    Returns
    -------
    np.ndarray
        Shape (len(target_genes),), dtype float32. Values are ordered to
        match `target_genes` exactly.

    Raises
    ------
    ValueError
        If `profile` is an ndarray and len(source_genes) != len(profile).
        If `target_genes` contains duplicate symbols.
        If `missing` or `duplicates` are not recognized literals.
    TypeError
        If `profile` is not an ndarray or dict.

    Notes
    -----
    Coverage logging:
        When coverage < 0.5 (fewer than half of target genes are present in
        source), a WARNING is emitted via the module logger. The caller should
        treat this as a signal to investigate imputation (see TODO above).

    Low-coverage guidance:
        When aligning L1000 landmark profiles (978 genes) to a full N_PDG
        target (10716 genes), coverage ~ 0.09. Consider the landmark-imputation
        TODO before proceeding with 90% zero-filled vectors.
    """
    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    if missing not in ("zero", "nan"):
        raise ValueError(
            f"align_to_gene_space: `missing` must be 'zero' or 'nan', got {missing!r}."
        )
    if duplicates not in ("first", "mean"):
        raise ValueError(
            f"align_to_gene_space: `duplicates` must be 'first' or 'mean', got {duplicates!r}."
        )

    # Validate target has no duplicates (caller contract).
    if len(target_genes) != len(set(target_genes)):
        seen: set[str] = set()
        dups = [g for g in target_genes if g in seen or seen.add(g)]  # type: ignore[func-returns-value]
        raise ValueError(
            f"align_to_gene_space: target_genes contains duplicate symbols: "
            f"{sorted(set(dups))!r}. Pass a deduplicated target list."
        )

    # Coerce profile to a flat float64 array aligned to source_genes.
    if isinstance(profile, dict):
        raw: np.ndarray = np.array(
            [profile.get(g, np.nan) for g in source_genes],
            dtype=np.float64,
        )
    elif isinstance(profile, np.ndarray):
        if profile.ndim != 1:
            raise ValueError(
                f"align_to_gene_space: profile must be 1-D, got shape {profile.shape}."
            )
        if len(profile) != len(source_genes):
            raise ValueError(
                f"align_to_gene_space: len(source_genes)={len(source_genes)} != "
                f"len(profile)={len(profile)}."
            )
        raw = profile.astype(np.float64)
    else:
        raise TypeError(
            f"align_to_gene_space: `profile` must be np.ndarray or dict, "
            f"got {type(profile).__name__!r}."
        )

    # ------------------------------------------------------------------
    # 2. Resolve duplicates in source_genes -> {symbol: value}
    # ------------------------------------------------------------------
    if duplicates == "first":
        resolved: dict[str, float] = {}
        for sym, val in zip(source_genes, raw.tolist()):
            if sym not in resolved:
                resolved[sym] = float(val)
    else:  # 'mean'
        accum: dict[str, list[float]] = {}
        for sym, val in zip(source_genes, raw.tolist()):
            accum.setdefault(sym, []).append(float(val))
        resolved = {sym: float(np.mean(vals)) for sym, vals in accum.items()}

    # ------------------------------------------------------------------
    # 3. Coverage check + logging
    # ------------------------------------------------------------------
    cov = coverage_fraction(list(resolved.keys()), target_genes)
    if cov < 0.5 and len(target_genes) > 0:
        log.warning(
            "align_to_gene_space: low coverage %.1f%% (%d / %d target genes "
            "found in source). Missing genes will be filled with '%s'. "
            "Consider landmark imputation (see TODO in gene_alignment.py).",
            100.0 * cov,
            int(round(cov * len(target_genes))),
            len(target_genes),
            missing,
        )

    # ------------------------------------------------------------------
    # 4. Build output array in target-gene order
    # ------------------------------------------------------------------
    fill_val: float = 0.0 if missing == "zero" else float("nan")
    out = np.full(len(target_genes), fill_val, dtype=np.float32)
    for i, gene in enumerate(target_genes):
        val = resolved.get(gene)
        if val is not None:
            out[i] = np.float32(val)
        # else: leave fill_val (zero or NaN)

    n_filled = int(round(cov * len(target_genes)))
    n_missing = len(target_genes) - n_filled
    log.debug(
        "align_to_gene_space: filled %d / %d target genes; %d missing (policy='%s'); "
        "coverage=%.3f; duplicate_policy='%s'",
        n_filled,
        len(target_genes),
        n_missing,
        missing,
        cov,
        duplicates,
    )

    return out
