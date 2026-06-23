"""Unit tests for `src/spatial/eda/region_diagnostics.py`.

In-memory fixtures only -- no real data files are read. Duck-typed AnnData
and OrthologTable-like objects are used as fixtures.

Behaviors covered:
  1. compute_moran_svgs returns a dict with keys svg_count/svg_fraction/top_svgs.
  2. human_mouse_liver_correlation uses only one-to-one orthologs
     (n_genes_compared < len(human_genes) when ortholog table is a strict subset).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.spatial.eda.region_diagnostics import (
    compute_moran_svgs,
    human_mouse_liver_correlation,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_moran_returns_dataframe() -> None:
    """compute_moran_svgs returns dict with svg_count/svg_fraction/top_svgs."""
    squidpy = pytest.importorskip("squidpy")

    # Build a minimal duck-typed AnnData with spatial coordinates.
    rng = np.random.default_rng(0)
    n_spots = 30
    n_genes = 10
    # Expression matrix (spots x genes)
    X = rng.standard_normal((n_spots, n_genes)).astype(np.float32)
    # Spatial coordinates (x, y) in pixel space
    spatial_coords = rng.uniform(0, 500, size=(n_spots, 2))
    # Gene names
    gene_names = [f"GENE{i}" for i in range(n_genes)]

    import anndata as ad
    adata = ad.AnnData(X=X)
    adata.var_names = gene_names
    adata.obsm["spatial"] = spatial_coords

    result = compute_moran_svgs(adata, spatial_key="spatial", n_neighs=6)

    assert isinstance(result, dict), "Expected a dict return from compute_moran_svgs"
    assert "svg_count" in result, "Result must have 'svg_count' key"
    assert "svg_fraction" in result, "Result must have 'svg_fraction' key"
    assert "top_svgs" in result, "Result must have 'top_svgs' key"


def test_cross_species_ortholog_filter() -> None:
    """n_genes_compared < len(human_genes) when ortholog table is a strict subset."""
    # Human profile: 6 genes
    human_genes = ["GAPDH", "TP53", "ACTB", "MYC", "EGFR", "CDH1"]
    human_profile = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)

    # Mouse profile: 4 genes (only 3 map to human via ortholog table)
    mouse_genes = ["Gapdh", "Trp53", "Actb", "Myc"]
    mouse_profile = np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float32)

    # Ortholog table: maps only 3 of the 6 human genes
    ortholog_pairs = pd.DataFrame({
        "human_symbol": ["GAPDH", "TP53", "ACTB"],
        "mouse_symbol": ["Gapdh", "Trp53", "Actb"],
    })
    # Duck-typed OrthologTable-like object
    ortholog_table = SimpleNamespace(pairs=ortholog_pairs)

    result = human_mouse_liver_correlation(
        human_profile, human_genes, mouse_profile, mouse_genes, ortholog_table
    )

    assert "n_genes_compared" in result, "Result must have 'n_genes_compared' key"
    assert "pearson_r" in result, "Result must have 'pearson_r' key"
    assert result["n_genes_compared"] < len(human_genes), (
        f"n_genes_compared ({result['n_genes_compared']}) should be < "
        f"len(human_genes) ({len(human_genes)}) when ortholog table is a strict subset"
    )
