"""APAP per-zone validity anchor (WIRE-03 / Halt Gate 3).

Compute the predicted-vs-measured per-zone Pearson on the mouse APAP injury
anchor and enforce Halt Gate 3. This is the phase's only DIRECT validity
evidence: predicted DE (from the WIRE-01 rule-B mouse cache, D-02) vs measured
DE (APAP_zone - control_zone on GSE272564's matched arms, D-06), per zone, over
the 10,716 PDG-gene space intersected with mouse->human ortholog coverage and
Visium coverage (D-08).

Public API
----------
assign_zones(adata, annotation=None, markers=None)
    Per-spot zonation labels: published annotation if present, else canonical
    markers (pericentral Glul/Cyp2e1; periportal Sds/Cyp2f2) per D-07.
measured_zone_de(apap_zone, ctrl_zone, mouse_genes, human_genes, ortholog_table, ...)
    Measured DE = APAP_zone - ctrl_zone (mouse symbols), mouse->human
    ortholog-aligned to the target human space, absent genes FLAGGED via a
    present_mask (NOT zero-filled, D-08). Reports n_genes_compared.
zone_pearson(pred_de, measured_de, present_mask)
    Per-zone Pearson over the present (flagged) intersection only; raises
    ValueError if fewer than 2 present genes (mirrors region_diagnostics).
halt_gate_3_fires(per_zone_r, threshold=0.3, key_zone="pericentral")
    Halt Gate 3 predicate keyed to the pericentral zone (APAP's classical
    injury site, D-08); a fired gate is a stop-and-REFRAME, not a bug (D-09).

Hard rules honored
------------------
    - Pure-where-possible: NO hardcoded absolute paths, NO real-data filenames.
      The caller (scripts/run_apap_validation.py) provides real arrays/AnnData.
    - Validation-only: this module reads Visium + the predicted cache for the
      gate; it NEVER feeds the model and NEVER uses APAP data as basal input.
    - DE-rule discipline: measured DE is a differential (APAP_zone - ctrl_zone);
      raw expression is never used as the comparison quantity.
    - present_mask flag-not-zero (D-08): absent ortholog/coverage genes are
      flagged and EXCLUDED from the Pearson, never silently zero-filled (which
      would inflate/deflate the correlation, T-02-11).
    - scanpy/squidpy are NEVER imported at module level (Pitfall 9: numba/numpy
      conflict). The driver loads AnnData behind a function-local import.

Primary analog: src/spatial/eda/region_diagnostics.py::human_mouse_liver_correlation
(lines ~266-340) -- the ortholog-mapped, intersection-flagged Pearson template.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import pearsonr

from src.spatial.gene_alignment import align_to_gene_space

log = logging.getLogger(__name__)

__all__ = [
    "assign_zones",
    "measured_zone_de",
    "zone_pearson",
    "halt_gate_3_fires",
    "CANONICAL_ZONE_MARKERS",
]

# ---------------------------------------------------------------------------
# Canonical zonation markers (D-07)
# ---------------------------------------------------------------------------

#: Default per-zone marker gene sets (D-07). Pericentral = where APAP injury
#: classically acts (CYP2E1-mediated NAPQI formation); periportal = the
#: opposing zone. Symbols are mouse-cased (Glul, not GLUL) because the rodent
#: APAP anchor is mouse Visium.
CANONICAL_ZONE_MARKERS: Dict[str, List[str]] = {
    "pericentral": ["Glul", "Cyp2e1"],
    "periportal": ["Sds", "Cyp2f2"],
}

#: Spots whose best marker-set mean score is below this absolute floor are
#: labelled "unassigned" and excluded by the caller before pseudobulking. The
#: floor is documented and intentionally permissive (real Visium spots carry
#: some marker signal); the dominant-set rule does the actual assignment.
_UNASSIGNED_SCORE_FLOOR: float = 1e-9


# ---------------------------------------------------------------------------
# Zone assignment (D-07)
# ---------------------------------------------------------------------------


def assign_zones(
    adata,
    annotation: Optional[Sequence[str]] = None,
    markers: Optional[Dict[str, List[str]]] = None,
) -> np.ndarray:
    """Assign each spot a zonation label (D-07).

    If ``annotation`` (published per-spot labels) is provided, it is used
    verbatim. Otherwise each spot is scored against the canonical marker sets
    (pericentral Glul/Cyp2e1; periportal Sds/Cyp2f2) and labelled by whichever
    set has the higher mean expression of its *available* marker genes. Spots
    whose best score falls below a documented floor are labelled "unassigned"
    (the caller excludes these before pseudobulking). Unsupervised Leiden is the
    documented last resort and is NOT implemented here -- it is only reached when
    BOTH a published annotation is absent AND no marker genes are present, in
    which case a ValueError is raised so the caller can fall back explicitly.

    Parameters
    ----------
    adata : AnnData-like
        Object exposing ``X`` (spots x genes), ``var_names`` (gene symbols), and
        ``n_obs``. Duck-typed; no anndata import required here.
    annotation : sequence of str, optional
        Published per-spot zone labels (length ``n_obs``). When given, returned
        as-is (after a length check). This is the D-07 "published annotation if
        present" branch.
    markers : dict[str, list[str]], optional
        Marker sets per zone. Defaults to ``CANONICAL_ZONE_MARKERS``.

    Returns
    -------
    np.ndarray of str, shape (n_obs,)
        Per-spot zone labels in spot order.

    Raises
    ------
    ValueError
        If ``annotation`` length mismatches ``n_obs``; or if no marker genes are
        present in ``var_names`` (Leiden last-resort signalled to the caller).
    """
    n_obs = int(adata.n_obs)

    # --- D-07 branch 1: published annotation if present ---
    if annotation is not None:
        labels = list(annotation)
        if len(labels) != n_obs:
            raise ValueError(
                f"assign_zones: annotation length {len(labels)} != n_obs {n_obs}."
            )
        log.info("assign_zones: using published annotation (%d spots).", n_obs)
        return np.asarray([str(x) for x in labels], dtype=object)

    # --- D-07 branch 2: canonical markers ---
    markers = markers or CANONICAL_ZONE_MARKERS

    var_names = list(adata.var_names)
    var_idx = {g: i for i, g in enumerate(var_names)}

    # Dense expression matrix (sparse guard mirrors pseudobulk.py).
    raw = adata.X
    X = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
    X = np.asarray(X, dtype=np.float64)

    # Resolve which marker genes are actually present per zone.
    present_markers: Dict[str, List[int]] = {}
    for zone, genes in markers.items():
        cols = [var_idx[g] for g in genes if g in var_idx]
        present_markers[zone] = cols

    n_present_total = sum(len(c) for c in present_markers.values())
    if n_present_total == 0:
        raise ValueError(
            "assign_zones: no canonical marker genes present in var_names and no "
            "published annotation supplied. Leiden clustering is the documented "
            "last resort (D-07) but is not implemented here -- supply an "
            f"annotation or marker genes. Markers tried: {markers}."
        )

    zones = list(present_markers.keys())
    # Per-zone mean score over available markers; absent zones score -inf so a
    # spot is never assigned to a zone with no detectable markers.
    scores = np.full((n_obs, len(zones)), -np.inf, dtype=np.float64)
    for j, zone in enumerate(zones):
        cols = present_markers[zone]
        if cols:
            scores[:, j] = X[:, cols].mean(axis=1)

    best_j = np.argmax(scores, axis=1)
    best_score = scores[np.arange(n_obs), best_j]

    labels_arr = np.array(
        [zones[j] for j in best_j], dtype=object
    )
    # Floor: spots with no marker signal at all -> unassigned.
    unassigned = best_score <= _UNASSIGNED_SCORE_FLOOR
    if unassigned.any():
        labels_arr[unassigned] = "unassigned"

    n_unassigned = int(unassigned.sum())
    log.info(
        "assign_zones: marker-based labels for %d spots (%d unassigned). "
        "Markers present: %s",
        n_obs,
        n_unassigned,
        {z: len(c) for z, c in present_markers.items()},
    )
    return labels_arr


# ---------------------------------------------------------------------------
# Measured DE = APAP_zone - ctrl_zone, ortholog-aligned, present-mask (D-08)
# ---------------------------------------------------------------------------


def measured_zone_de(
    apap_zone: np.ndarray,
    ctrl_zone: np.ndarray,
    mouse_genes: Sequence[str],
    human_genes: Sequence[str],
    ortholog_table,
    *,
    agg: str = "mean",  # accepted for signature symmetry; subtraction is elementwise
) -> dict:
    """Measured DE for one zone: APAP_zone - ctrl_zone, mapped to human space.

    The two inputs are per-zone pseudobulk vectors in MOUSE symbol space (one
    arm each). DE is the elementwise difference (rule on measured DE: a
    treated - control differential, mirroring rule B's predicted side). The DE
    is then mapped mouse->human via the ortholog table and reindexed to the
    target human gene order. Human genes with no mouse ortholog (or whose mouse
    ortholog is absent from ``mouse_genes``) are FLAGGED via ``present_mask`` and
    excluded -- never zero-filled (D-08, T-02-11).

    Parameters
    ----------
    apap_zone, ctrl_zone : np.ndarray, shape (len(mouse_genes),)
        Per-zone pseudobulk for the APAP arm and the control arm, in mouse
        symbol space and the same gene order.
    mouse_genes : sequence of str
        Gene symbols for ``apap_zone`` / ``ctrl_zone`` (mouse).
    human_genes : sequence of str
        Target human gene order (e.g. the 10,716 PDG symbols). ``de`` and
        ``present_mask`` are returned in this order/length.
    ortholog_table : OrthologTable-like
        Object with a ``.pairs`` DataFrame carrying ``human_symbol`` and
        ``mouse_symbol`` columns (one-to-one orthologs, src.spatial.orthology).
    agg : str
        Accepted for caller symmetry; the DE subtraction is elementwise and
        does not aggregate here (aggregation happens upstream in pseudobulk).

    Returns
    -------
    dict with keys:
        de : np.ndarray, shape (len(human_genes),)
            Measured DE reindexed to human order. Absent positions are NaN.
        present_mask : np.ndarray of bool, shape (len(human_genes),)
            True where a mouse ortholog DE value was placed; False (flagged)
            otherwise.
        n_genes_compared : int
            present_mask.sum() -- the intersection size (D-08).

    Raises
    ------
    ValueError
        If ``apap_zone`` / ``ctrl_zone`` shapes mismatch ``mouse_genes``.
    """
    apap = np.asarray(apap_zone, dtype=np.float64)
    ctrl = np.asarray(ctrl_zone, dtype=np.float64)
    mouse_genes = [str(g) for g in mouse_genes]
    human_genes = [str(g) for g in human_genes]

    if apap.shape != (len(mouse_genes),):
        raise ValueError(
            f"measured_zone_de: apap_zone shape {apap.shape} != "
            f"(len(mouse_genes)={len(mouse_genes)},)."
        )
    if ctrl.shape != (len(mouse_genes),):
        raise ValueError(
            f"measured_zone_de: ctrl_zone shape {ctrl.shape} != "
            f"(len(mouse_genes)={len(mouse_genes)},)."
        )

    # Measured DE in mouse space (treated - control).
    de_mouse = apap - ctrl

    # Map mouse->human via the ortholog table (mirror region_diagnostics
    # h_to_m / index-lookup idiom). Build a human-keyed DE dict so we can
    # align_to_gene_space with NaN flagging.
    ot = ortholog_table.pairs
    h_to_m = dict(zip(ot["human_symbol"], ot["mouse_symbol"]))
    m_idx = {g: i for i, g in enumerate(mouse_genes)}

    # Human DE values keyed by human symbol, only where the mouse ortholog is
    # present in the measured vector.
    human_de_map: Dict[str, float] = {}
    for h_gene, m_gene in h_to_m.items():
        if m_gene in m_idx:
            human_de_map[h_gene] = float(de_mouse[m_idx[m_gene]])

    # Reindex to the target human order with NaN for absent (flag-not-zero,
    # D-08). align_to_gene_space(missing="nan") returns float32; absent -> NaN.
    aligned = align_to_gene_space(
        human_de_map,
        source_genes=list(human_de_map.keys()),
        target_genes=human_genes,
        missing="nan",
        duplicates="first",
    )
    de = np.asarray(aligned, dtype=np.float64)
    present_mask = ~np.isnan(de)
    n_compared = int(present_mask.sum())

    log.info(
        "measured_zone_de: %d human genes, %d with a present mouse ortholog DE "
        "value (intersection; %d flagged absent, not zero-filled).",
        len(human_genes),
        n_compared,
        len(human_genes) - n_compared,
    )

    return {
        "de": de,
        "present_mask": present_mask,
        "n_genes_compared": n_compared,
    }


# ---------------------------------------------------------------------------
# Per-zone Pearson (D-08) -- mirrors region_diagnostics <2 guard
# ---------------------------------------------------------------------------


def zone_pearson(
    pred_de: np.ndarray,
    measured_de: np.ndarray,
    present_mask: np.ndarray,
) -> dict:
    """Pearson r of predicted vs measured DE over the present (flagged) genes.

    Only positions where ``present_mask`` is True are compared -- absent genes
    are NEVER included (flag-not-zero, D-08). Mirrors the region_diagnostics
    ``< 2`` ValueError guard.

    Parameters
    ----------
    pred_de : np.ndarray
        Predicted DE in the target human order (from the WIRE-01 mouse cache).
    measured_de : np.ndarray
        Measured DE in the same order (from ``measured_zone_de``). May contain
        NaN at absent positions -- those are masked out by ``present_mask``.
    present_mask : np.ndarray of bool
        True where both vectors carry a comparable value.

    Returns
    -------
    dict with keys: pearson_r (float), p_value (float), n_genes_compared (int).

    Raises
    ------
    ValueError
        If fewer than 2 present genes (Pearson is undefined).
    """
    pred_de = np.asarray(pred_de, dtype=np.float64)
    measured_de = np.asarray(measured_de, dtype=np.float64)
    present_mask = np.asarray(present_mask, dtype=bool)

    idx = np.where(present_mask)[0]
    if idx.size < 2:
        raise ValueError(
            f"zone_pearson: only {idx.size} present gene(s) after masking; at "
            "least 2 are required for Pearson r. Check the ortholog map and the "
            "predicted/measured gene orders (D-08 flag-not-zero applied)."
        )

    r, p = pearsonr(pred_de[idx], measured_de[idx])
    return {
        "pearson_r": float(r),
        "p_value": float(p),
        "n_genes_compared": int(idx.size),
    }


# ---------------------------------------------------------------------------
# Halt Gate 3 (D-08 / D-09) -- keyed to the pericentral zone
# ---------------------------------------------------------------------------


def halt_gate_3_fires(
    per_zone_r: Dict[str, float],
    threshold: float = 0.3,
    key_zone: str = "pericentral",
) -> bool:
    """Halt Gate 3 predicate: pericentral predicted-vs-measured Pearson < 0.3.

    The gate is keyed to the pericentral zone -- where APAP injury classically
    acts (CYP2E1-mediated NAPQI) -- because reproducing the right *regional*
    effect is the actual spatial claim (D-08). All zones are reported by the
    caller regardless; only the key zone gates. A fired gate is a
    stop-and-REFRAME of the spatial claim (D-09), not a bug.

    Parameters
    ----------
    per_zone_r : dict[str, float]
        Mapping zone -> pearson_r.
    threshold : float
        Gate threshold (default 0.3, D-08).
    key_zone : str
        The gating zone (default "pericentral", D-08).

    Returns
    -------
    bool
        True if ``per_zone_r[key_zone] < threshold`` (gate fires).

    Raises
    ------
    KeyError
        If ``key_zone`` is absent from ``per_zone_r`` (a mis-keyed gate would
        silently mis-fire/mis-clear -- T-02-12; surface it loudly instead).
    """
    if key_zone not in per_zone_r:
        raise KeyError(
            f"halt_gate_3_fires: key_zone {key_zone!r} not in per_zone_r "
            f"(zones present: {sorted(per_zone_r)}). The gate cannot evaluate a "
            "zone it was never given -- check zone assignment (D-07)."
        )
    fires = per_zone_r[key_zone] < threshold
    log.info(
        "halt_gate_3_fires: %s r=%.4f %s %.2f -> gate %s",
        key_zone,
        per_zone_r[key_zone],
        "<" if fires else ">=",
        threshold,
        "FIRES (stop-and-REFRAME, D-09)" if fires else "CLEAR",
    )
    return bool(fires)
