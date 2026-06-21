"""Pure library: pseudobulk aggregation of a spots-by-genes matrix.

Given a spots-by-genes expression matrix, a per-spot region-label array, and a
gene-name list, return a regions-by-genes basal-profile matrix by aggregating
spots that share the same region label.

This module is **pure**: no file I/O, no hardcoded absolute paths, no real-data
filenames, no AnnData hard dependency. An optional helper `from_anndata` is
provided for caller convenience but only imported lazily so this module can be
imported without anndata installed.

Aggregation modes
-----------------
'mean'   : arithmetic mean across spots in the region (NaN-aware, nanmean).
'median' : median across spots in the region (NaN-aware, nanmedian).
'sum'    : sum across spots in the region (NaN treated as 0 via nansum).

Empty-region handling
---------------------
If a label supplied in `region_labels` is absent from `spot_labels` the
corresponding row in the output is filled with NaN (for 'mean'/'median') or
zeros (for 'sum'). An absent region is not an error; callers can detect it via
`result.empty_regions`.

Region ordering
---------------
The output row order follows the sorted order of unique labels found across
*both* `spot_labels` and `region_labels`. Determinism is guaranteed: same
inputs → identical output regardless of Python dict-insertion order.
"""

from __future__ import annotations

import logging
from typing import Literal, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["pseudobulk", "PseudobulkResult"]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AggMode = Literal["mean", "median", "sum"]
_VALID_AGG_MODES: frozenset[str] = frozenset({"mean", "median", "sum"})


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class PseudobulkResult(NamedTuple):
    """Output of `pseudobulk`.

    Attributes
    ----------
    profiles : np.ndarray
        Shape ``(n_regions, n_genes)``, dtype float64. Row ``i`` is the
        aggregated basal profile for ``region_order[i]``.
    region_order : list[str]
        Sorted list of region labels. ``profiles[i]`` corresponds to
        ``region_order[i]``.
    gene_names : list[str]
        Gene names in column order, echoed from the input for caller
        convenience.
    empty_regions : list[str]
        Subset of ``region_order`` for which no spot carried that label.
        These rows in ``profiles`` are NaN (mean/median) or zero (sum).
    spot_counts : dict[str, int]
        Mapping from region label → number of spots contributing. Zero for
        empty regions.
    """

    profiles: np.ndarray
    region_order: list[str]
    gene_names: list[str]
    empty_regions: list[str]
    spot_counts: dict[str, int]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    matrix: np.ndarray,
    spot_labels: np.ndarray,
    gene_names: Sequence[str],
    region_labels: Optional[Sequence[str]],
    agg: str,
) -> None:
    if matrix.ndim != 2:
        raise ValueError(
            f"pseudobulk: matrix must be 2-D (spots x genes), got ndim={matrix.ndim}."
        )
    n_spots, n_genes = matrix.shape
    if len(spot_labels) != n_spots:
        raise ValueError(
            f"pseudobulk: len(spot_labels)={len(spot_labels)} != "
            f"matrix.shape[0]={n_spots}."
        )
    if len(gene_names) != n_genes:
        raise ValueError(
            f"pseudobulk: len(gene_names)={len(gene_names)} != "
            f"matrix.shape[1]={n_genes}."
        )
    if agg not in _VALID_AGG_MODES:
        raise ValueError(
            f"pseudobulk: agg={agg!r} is not a valid mode. "
            f"Choose from {sorted(_VALID_AGG_MODES)}."
        )
    if region_labels is not None and len(region_labels) == 0:
        raise ValueError(
            "pseudobulk: region_labels is an empty sequence — pass None to "
            "infer regions from spot_labels."
        )


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def pseudobulk(
    matrix: np.ndarray,
    spot_labels: np.ndarray | Sequence,
    gene_names: Sequence[str],
    *,
    agg: AggMode = "mean",
    region_labels: Optional[Sequence[str]] = None,
) -> PseudobulkResult:
    """Aggregate a spots-by-genes matrix into a regions-by-genes basal profile.

    Parameters
    ----------
    matrix : np.ndarray
        Shape ``(n_spots, n_genes)``. May contain NaN. Converted to float64
        internally; the input array is never modified.
    spot_labels : array-like of str
        Length ``n_spots``. Each element is the region label for that spot.
        Non-string values are coerced to ``str``.
    gene_names : sequence of str
        Length ``n_genes``. Gene identifiers in column order.
    agg : {'mean', 'median', 'sum'}
        Aggregation function applied across spots within each region.
        NaN handling: 'mean'/'median' use numpy nan-aware functions; 'sum'
        uses nansum (NaN treated as 0).
    region_labels : sequence of str, optional
        Explicit set of region labels to include in the output. Labels absent
        from ``spot_labels`` become empty regions (NaN/zero rows). If ``None``,
        the output includes exactly the labels present in ``spot_labels``.

    Returns
    -------
    PseudobulkResult
        See class docstring. Region rows are in sorted label order.

    Raises
    ------
    ValueError
        On shape mismatches, invalid ``agg`` mode, or empty ``region_labels``
        sequence (as opposed to ``None``).
    """
    # --- Coerce + validate ---------------------------------------------------
    mat = np.array(matrix, dtype=np.float64)
    spot_labels_arr = np.array([str(lbl) for lbl in spot_labels], dtype=object)
    gene_names_list: list[str] = [str(g) for g in gene_names]
    _validate_inputs(mat, spot_labels_arr, gene_names_list, region_labels, agg)

    n_spots, n_genes = mat.shape

    # --- Determine output region set ----------------------------------------
    unique_in_spots: list[str] = sorted(set(spot_labels_arr.tolist()))
    if region_labels is not None:
        extra = sorted(set(region_labels))
        all_regions: list[str] = sorted(set(unique_in_spots) | set(extra))
    else:
        all_regions = unique_in_spots

    n_regions = len(all_regions)

    # --- Aggregate -----------------------------------------------------------
    profiles = np.full((n_regions, n_genes), np.nan, dtype=np.float64)
    empty_regions: list[str] = []
    spot_counts: dict[str, int] = {}

    for i, region in enumerate(all_regions):
        mask = spot_labels_arr == region
        n_matching = int(mask.sum())
        spot_counts[region] = n_matching

        if n_matching == 0:
            empty_regions.append(region)
            if agg == "sum":
                profiles[i] = 0.0
            # else: leave as NaN (already set by np.full)
            log.debug("pseudobulk: region %r has no spots — row set to %s.",
                      region, "zeros" if agg == "sum" else "NaN")
            continue

        sub = mat[mask]  # shape (n_matching, n_genes)

        if agg == "mean":
            profiles[i] = np.nanmean(sub, axis=0)
        elif agg == "median":
            profiles[i] = np.nanmedian(sub, axis=0)
        elif agg == "sum":
            profiles[i] = np.nansum(sub, axis=0)

    log.info(
        "pseudobulk: agg=%s n_spots=%d n_regions=%d (%d empty) n_genes=%d",
        agg, n_spots, n_regions, len(empty_regions), n_genes,
    )

    return PseudobulkResult(
        profiles=profiles,
        region_order=all_regions,
        gene_names=gene_names_list,
        empty_regions=empty_regions,
        spot_counts=spot_counts,
    )


# ---------------------------------------------------------------------------
# Optional AnnData helper (lazy import — no hard dependency)
# ---------------------------------------------------------------------------


def from_anndata(
    adata,
    obs_col: str,
    *,
    agg: AggMode = "mean",
    region_labels: Optional[Sequence[str]] = None,
    layer: Optional[str] = None,
) -> PseudobulkResult:
    """Extract matrix and labels from an AnnData object and call `pseudobulk`.

    This is a convenience wrapper; no AnnData is imported at module level.
    AnnData must be installed in the calling environment.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. ``adata.obs[obs_col]`` must be a string-valued
        column of region labels. ``adata.var_names`` is used as ``gene_names``.
    obs_col : str
        Column in ``adata.obs`` to use as the per-spot region label.
    agg : {'mean', 'median', 'sum'}
        Forwarded to `pseudobulk`.
    region_labels : sequence of str, optional
        Forwarded to `pseudobulk`.
    layer : str, optional
        If given, use ``adata.layers[layer]`` instead of ``adata.X``.

    Returns
    -------
    PseudobulkResult
    """
    if obs_col not in adata.obs.columns:
        raise KeyError(
            f"from_anndata: obs_col={obs_col!r} not found in adata.obs. "
            f"Available columns: {adata.obs.columns.tolist()}"
        )

    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(
                f"from_anndata: layer={layer!r} not found in adata.layers. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        raw = adata.layers[layer]
    else:
        raw = adata.X

    # AnnData matrices may be scipy sparse — convert to dense ndarray.
    if hasattr(raw, "toarray"):
        mat = raw.toarray()
    else:
        mat = np.asarray(raw)

    spot_labels_arr = adata.obs[obs_col].astype(str).to_numpy()
    gene_names_list: list[str] = list(adata.var_names.astype(str))

    return pseudobulk(
        mat,
        spot_labels_arr,
        gene_names_list,
        agg=agg,
        region_labels=region_labels,
    )
