"""Pure library: region diagnostics for spatial EDA (EDA-03).

Implements:
  - compute_moran_svgs: Moran's I spatially variable gene retention via squidpy
  - basal_similarity_matrix: pairwise Pearson r across zone pseudobulk profiles
  - svg_retention: fraction of raw Visium SVGs in the 10,716-gene model space
  - human_mouse_liver_correlation: ortholog-filtered human-vs-rodent basal Pearson r
  - ood_mahalanobis: ridge-regularized Mahalanobis OOD distance in the 978-gene
    landmark subspace (Pitfall 7: full 10,716-gene covariance is rank-deficient)

Hard rules honored:
    - Pure library: NO hardcoded absolute paths, NO real-data filenames.
    - No mock or synthetic labels -- the caller provides real data.
    - squidpy imported inside compute_moran_svgs (try/except ImportError) so this
      file imports cleanly in environments where squidpy is absent (Pitfall 9).
    - scanpy is NEVER imported at module level (Pitfall 9: numba/numpy conflict).
    - adata.obsm['spatial'] must be populated by the caller before compute_moran_svgs
      (see yu2022 coord loading note below).

Yu2022 coordinate note:
    The plan-06 script loads yu2022_liver from the figshare zip, extracts
    tissue_positions_list.csv, and assigns adata.obsm['spatial'] = coords.
    This library assumes obsm['spatial'] is present and raises KeyError with
    the available obsm keys if not (mirror pseudobulk.from_anndata KeyError style).

OOD method:
    Chosen method = Mahalanobis, 978-gene LINCS landmark subspace, alpha=1e-2
    (ridge regularisation). The caller must project both manifold and query to
    the 978-gene landmark subspace BEFORE calling ood_mahalanobis. This avoids
    the rank-deficient covariance of the full 10,716-gene space (Pitfall 7).
    The method name must be reported in P1_eda.md: "Mahalanobis, 978-gene
    landmark subspace, alpha=1e-2".
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import pearsonr

from src.spatial.gene_alignment import coverage_fraction

log = logging.getLogger(__name__)

__all__ = [
    "compute_moran_svgs",
    "basal_similarity_matrix",
    "svg_retention",
    "human_mouse_liver_correlation",
    "ood_mahalanobis",
]

# ---------------------------------------------------------------------------
# OOD method constant (reported in P1_eda.md)
# ---------------------------------------------------------------------------

OOD_METHOD = "Mahalanobis, 978-gene landmark subspace, alpha=1e-2"


# ---------------------------------------------------------------------------
# Task 1: Moran's I SVG retention, basal similarity, gene coverage
# ---------------------------------------------------------------------------


def compute_moran_svgs(
    adata,
    spatial_key: str = "spatial",
    n_neighs: int = 6,
) -> dict:
    """Compute Moran's I on a Visium AnnData and return SVG retention stats.

    Uses squidpy 1.8.2 API: spatial_neighbors_knn (not deprecated
    spatial_neighbors, Pitfall 4) and spatial_autocorr(mode='moran').

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. ``adata.obsm[spatial_key]`` must contain (x, y)
        pixel coordinates (Pitfall 5: populated by caller from
        tissue_positions_list.csv for yu2022, not by this library).
    spatial_key : str
        Key in ``adata.obsm`` for spatial coordinates.
    n_neighs : int
        Number of nearest neighbours for the spatial graph (default 6, the
        Visium hexagonal grid standard).

    Returns
    -------
    dict with keys:
        svg_count : int -- number of genes with Moran's I pval_norm < 0.05
        svg_fraction : float -- svg_count / total genes tested
        top_svgs : list[str] -- up to 20 gene names ranked by Moran's I

    Raises
    ------
    ImportError
        If squidpy is not installed. Guard with pytest.importorskip in tests.
    KeyError
        If adata.obsm[spatial_key] is absent.
    """
    # Guard: spatial coordinates must be present
    if spatial_key not in adata.obsm:
        available = list(adata.obsm.keys())
        raise KeyError(
            f"compute_moran_svgs: obsm[{spatial_key!r}] not found. "
            f"Available obsm keys: {available}. "
            "Populate adata.obsm['spatial'] from tissue_positions_list.csv "
            "before calling this function."
        )

    try:
        import squidpy as sq
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "compute_moran_svgs requires squidpy. "
            "Install with: pip install squidpy"
        ) from exc

    # Build k-NN spatial graph (n_neighs=6 for Visium hex grid)
    sq.gr.spatial_neighbors_knn(
        adata,
        spatial_key=spatial_key,
        n_neighs=n_neighs,
        key_added="spatial",
    )

    # Compute Moran's I for all genes; result stored in adata.uns['moranI']
    sq.gr.spatial_autocorr(
        adata,
        mode="moran",
        genes=None,
        connectivity_key="spatial_connectivities",
        transformation=True,
        copy=False,
    )

    moran_df = adata.uns["moranI"]  # DataFrame: index=gene, cols ['I', 'pval_norm', ...]
    svgs = moran_df[moran_df["pval_norm"] < 0.05]

    n_total = len(moran_df)
    n_svgs = len(svgs)

    log.info(
        "compute_moran_svgs: %d total genes tested, %d SVGs (pval_norm<0.05, %.1f%%)",
        n_total, n_svgs, 100.0 * n_svgs / n_total if n_total > 0 else 0.0,
    )

    return {
        "svg_count": n_svgs,
        "svg_fraction": n_svgs / n_total if n_total > 0 else 0.0,
        "top_svgs": svgs.nlargest(20, "I").index.tolist(),
    }


def basal_similarity_matrix(
    profiles: Union[dict, np.ndarray],
    labels: Optional[List[str]] = None,
    metric: str = "pearson",
) -> pd.DataFrame:
    """Compute pairwise similarity between zone pseudobulk profiles.

    Parameters
    ----------
    profiles : dict[str, np.ndarray] or np.ndarray
        If dict: keys are zone names, values are 1-D gene-expression vectors.
        If ndarray: shape (n_zones, n_genes); requires `labels` to name rows.
    labels : list[str], optional
        Zone names when `profiles` is an ndarray. Ignored when dict.
    metric : str
        'pearson' (default) or 'cosine'. Pearson is standard for GEX profiles.

    Returns
    -------
    pd.DataFrame
        Symmetric (n_zones x n_zones) similarity matrix. Index and columns are
        zone names.

    Notes
    -----
    Applies the pseudobulk.py sparse guard: if a matrix or array-like has a
    `toarray` method, it is converted to dense before computing similarity.
    """
    # Normalise input to (n_zones, n_genes) array + label list
    if isinstance(profiles, dict):
        labels_list = list(profiles.keys())
        mat = np.stack([profiles[k] for k in labels_list]).astype(np.float64)
    else:
        raw = profiles
        if hasattr(raw, "toarray"):  # sparse guard (mirrors pseudobulk.py)
            mat = raw.toarray()
        else:
            mat = np.asarray(raw, dtype=np.float64)
        if labels is None:
            labels_list = [str(i) for i in range(mat.shape[0])]
        else:
            labels_list = list(labels)

    n_zones = len(labels_list)
    sim = np.zeros((n_zones, n_zones), dtype=np.float64)

    if metric == "pearson":
        for i in range(n_zones):
            for j in range(i, n_zones):
                if i == j:
                    sim[i, j] = 1.0
                else:
                    r, _ = pearsonr(mat[i], mat[j])
                    sim[i, j] = r
                    sim[j, i] = r
    elif metric == "cosine":
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        mat_norm = mat / norms
        sim = mat_norm @ mat_norm.T
    else:
        raise ValueError(
            f"basal_similarity_matrix: metric={metric!r} not supported. "
            "Choose 'pearson' or 'cosine'."
        )

    log.info(
        "basal_similarity_matrix: %d zones, metric=%s",
        n_zones, metric,
    )

    return pd.DataFrame(sim, index=labels_list, columns=labels_list)


def svg_retention(
    svg_genes: Sequence[str],
    model_gene_space: Sequence[str],
) -> float:
    """Fraction of raw-Visium SVGs whose symbol is in the 10,716-gene model space.

    Parameters
    ----------
    svg_genes : sequence of str
        Gene symbols identified as spatially variable in the raw Visium data.
    model_gene_space : sequence of str
        The 10,716-gene MultiDCP model gene list.

    Returns
    -------
    float
        Coverage fraction in [0.0, 1.0].
    """
    frac = coverage_fraction(list(svg_genes), list(model_gene_space))
    log.info(
        "svg_retention: %d SVGs, %d in model space (%.1f%%)",
        len(svg_genes),
        int(round(frac * len(svg_genes))),
        100.0 * frac,
    )
    return frac


# ---------------------------------------------------------------------------
# Task 2: Ortholog-filtered cross-species correlation + OOD Mahalanobis
# ---------------------------------------------------------------------------


def human_mouse_liver_correlation(
    human_profile: np.ndarray,
    human_genes: List[str],
    mouse_profile: np.ndarray,
    mouse_genes: List[str],
    ortholog_table,
) -> dict:
    """Pearson r between human and mouse liver pseudobulk, one-to-one orthologs only.

    Maps human_symbol -> mouse_symbol via ortholog_table.pairs, keeps only genes
    present in BOTH profiles, then computes Pearson r over the matched vector.
    n_genes_compared is strictly < len(human_genes) when the ortholog table
    covers only a subset of human genes (XC-08 one-to-one filter).

    Parameters
    ----------
    human_profile : np.ndarray, shape (n_genes_human,)
        Pseudobulk expression vector for human liver (e.g., yu2022 whole-tissue).
    human_genes : list[str]
        Gene symbols corresponding to each position in human_profile.
    mouse_profile : np.ndarray, shape (n_genes_mouse,)
        Pseudobulk expression vector for mouse liver (e.g., GSE272564 APAP0h).
    mouse_genes : list[str]
        Gene symbols corresponding to each position in mouse_profile.
    ortholog_table : OrthologTable-like
        Object with a ``.pairs`` DataFrame having columns 'human_symbol' and
        'mouse_symbol' (one-to-one orthologs only, from src.spatial.orthology).

    Returns
    -------
    dict with keys:
        pearson_r : float
        p_value : float
        n_genes_compared : int (< len(human_genes) when ortholog coverage < 100%)

    Notes
    -----
    GSE272564 APAP0h has no published zone annotations; use whole-sample
    pseudobulk as the mouse reference and note this in P1_eda.md.
    Zone-level rodent correlation is deferred to later phases.
    """
    ot = ortholog_table.pairs  # DataFrame cols: human_symbol, mouse_symbol, ...

    # Build human->mouse symbol map (one-to-one; pairs already filtered)
    h_to_m = dict(zip(ot["human_symbol"], ot["mouse_symbol"]))

    # Index maps for O(1) lookup
    h_idx: dict[str, int] = {g: i for i, g in enumerate(human_genes)}
    m_idx: dict[str, int] = {g: i for i, g in enumerate(mouse_genes)}

    h_vals: list[float] = []
    m_vals: list[float] = []

    for h_gene, m_gene in h_to_m.items():
        if h_gene in h_idx and m_gene in m_idx:
            h_vals.append(float(human_profile[h_idx[h_gene]]))
            m_vals.append(float(mouse_profile[m_idx[m_gene]]))

    n_compared = len(h_vals)
    log.info(
        "human_mouse_liver_correlation: %d orthologs tested, %d matched in both profiles "
        "(%d human genes total)",
        len(h_to_m), n_compared, len(human_genes),
    )

    if n_compared < 2:
        raise ValueError(
            f"human_mouse_liver_correlation: only {n_compared} genes matched across "
            "human and mouse profiles via the ortholog table. At least 2 are required "
            "for Pearson r. Check that human_genes and mouse_genes use the same symbol "
            "convention as ortholog_table.pairs."
        )

    r, p = pearsonr(np.array(h_vals, dtype=np.float64), np.array(m_vals, dtype=np.float64))
    return {"pearson_r": float(r), "p_value": float(p), "n_genes_compared": n_compared}


def ood_mahalanobis(
    manifold: np.ndarray,
    query: np.ndarray,
    alpha: float = 1e-2,
) -> np.ndarray:
    """Ridge-regularized Mahalanobis distance of query points from the manifold.

    RECOMMENDED CALL: project both `manifold` and `query` to the 978 LINCS
    landmark genes BEFORE calling this function. The full 10,716-gene covariance
    (n=10 << d=10,716) is rank-deficient even with ridge regularisation; the
    978-gene landmark subspace is well-conditioned (Pitfall 7).

    Chosen method (report in P1_eda.md):
        "Mahalanobis, 978-gene landmark subspace, alpha=1e-2"

    Parameters
    ----------
    manifold : np.ndarray, shape (n_ref, d)
        Reference distribution (e.g., PDG cancer-line matrix, 10 cell lines).
        Caller must project to the 978-gene landmark subspace first.
    query : np.ndarray, shape (n_query, d)
        Query points (e.g., healthy liver pseudobulk, projected to 978 genes).
    alpha : float
        Ridge regularisation coefficient added to the diagonal of the covariance
        before inversion. Default 1e-2.

    Returns
    -------
    np.ndarray, shape (n_query,)
        Mahalanobis distance of each query row from the manifold mean.

    Notes
    -----
    cov = np.cov(manifold.T)  # (d, d)
    cov_reg = cov + alpha * np.eye(d)
    vi = np.linalg.inv(cov_reg)
    mean_ref = manifold.mean(axis=0)
    distances[i] = mahalanobis(query[i], mean_ref, vi)
    """
    manifold = np.asarray(manifold, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)

    if manifold.ndim != 2:
        raise ValueError(
            f"ood_mahalanobis: manifold must be 2-D (n_ref, d), got ndim={manifold.ndim}."
        )
    if query.ndim != 2:
        raise ValueError(
            f"ood_mahalanobis: query must be 2-D (n_query, d), got ndim={query.ndim}."
        )
    if manifold.shape[1] != query.shape[1]:
        raise ValueError(
            f"ood_mahalanobis: manifold.shape[1]={manifold.shape[1]} != "
            f"query.shape[1]={query.shape[1]}. "
            "Both must be projected to the same gene subspace before calling."
        )

    d = manifold.shape[1]
    cov = np.cov(manifold.T)  # (d, d); rank-deficient if n_ref << d
    cov_reg = cov + alpha * np.eye(d)
    vi = np.linalg.inv(cov_reg)
    mean_ref = manifold.mean(axis=0)

    distances = np.array([mahalanobis(q, mean_ref, vi) for q in query])

    log.info(
        "ood_mahalanobis: n_ref=%d, n_query=%d, d=%d, alpha=%g, "
        "mean_dist=%.4f (method: %s)",
        manifold.shape[0], len(query), d, alpha, float(distances.mean()), OOD_METHOD,
    )

    return distances
