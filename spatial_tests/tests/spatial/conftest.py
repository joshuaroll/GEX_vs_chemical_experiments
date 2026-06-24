"""Shared Phase-2 fixtures for the spatial test suite.

Centralizes the fixtures reused across the four Phase-2 test files
(``test_region_signature.py``, ``test_model_load.py``, ``test_tox_head.py``,
``test_apap_validation.py``). The small region-signature fixtures
(``three_genes``, ``two_regions``, ``two_pert_ids``, ``small_manifest``,
``region_basal_map``) were moved here verbatim from ``test_region_signature.py``
so all Phase-2 files resolve them via pytest fixture discovery.

Purity gate
-----------
- No ``/raid`` paths anywhere in this file.
- No real-data filenames; the gene-order slice is a hand-written synthetic
  16-symbol tuple for shape contracts only (real symbols are read only by the
  ``@pytest.mark.gpu`` integration smoke, which is exempt from this gate).
- All arrays are constructed inline with small synthetic values.
- No import of torch or any model checkpoint utilities.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.spatial.region_signature import build_manifest


# ---------------------------------------------------------------------------
# Region-signature fixtures (moved from test_region_signature.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def three_genes():
    """Gene ID tuple with 3 elements (small fixture for readable assertions)."""
    return ("GENE_A", "GENE_B", "GENE_C")


@pytest.fixture
def two_regions():
    """Two region labels."""
    return ["periportal", "pericentral"]


@pytest.fixture
def two_pert_ids():
    """Two perturbation IDs."""
    return ["drug_X", "drug_Y"]


@pytest.fixture
def small_manifest(two_pert_ids, two_regions, three_genes):
    """Manifest for 2 pert_ids × 2 regions × 3 genes (multidcp_pdg)."""
    return build_manifest(
        pert_ids=two_pert_ids,
        regions=two_regions,
        gene_ids=list(three_genes),
        model_variant="multidcp_pdg",
    )


@pytest.fixture
def region_basal_map(two_regions, three_genes):
    """Synthetic basal vectors, shape (3,) each, for two regions."""
    rng = np.random.default_rng(42)
    return {
        region: rng.standard_normal(len(three_genes)).astype(np.float32)
        for region in two_regions
    }


# ---------------------------------------------------------------------------
# New Phase-2 fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_treated_control(three_genes):
    """Two float32 vectors (treated, control) with known rule-B difference.

    treated = [3, 5, 7], control = [1, 2, 3] -> de = treated - control = [2, 3, 4].
    Length matches ``three_genes`` so rule-B subtraction is checkable.
    """
    treated = np.array([3.0, 5.0, 7.0], dtype=np.float32)
    control = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert treated.shape == (len(three_genes),)
    return treated, control


@pytest.fixture
def gene_order_slice():
    """A fixed 16-symbol synthetic gene-order slice for shape contracts only.

    Hardcoded (NOT read from any real-data file) to keep the pure suite off
    ``/raid`` and off ``data/processed/...``. Real gene symbols are read only by
    the gpu-marked integration smoke.
    """
    return (
        "FLNC", "MAP2K4", "GLUL", "CYP2E1", "SDS", "CYP2F2",
        "ALB", "APOA1", "CYP3A4", "GAPDH", "ACTB", "TP53",
        "MYC", "EGFR", "CDH1", "PCNA",
    )


# ---------------------------------------------------------------------------
# Zone-marker AnnData fixture (WIRE-03, apap_validation)
# ---------------------------------------------------------------------------


class _FakeVarIndex:
    """Minimal stand-in for adata.var_names (array-like of str)."""

    def __init__(self, names):
        self._names = list(names)

    def astype(self, dtype):
        return _FakeVarIndex([dtype(n) for n in self._names])

    def get_loc(self, key):
        return self._names.index(key)

    def __iter__(self):
        return iter(self._names)

    def __contains__(self, key):
        return key in self._names

    def __len__(self):
        return len(self._names)


class _FakeAnnData:
    """Duck-typed AnnData-like object. No anndata package required.

    Carries an expression matrix, an ``obs`` DataFrame (with a planted ``zone``
    column for assertion), and ``var_names``. Mirrors the duck-typed AnnData
    style of ``test_pseudobulk.py``.
    """

    def __init__(self, X, obs_data, gene_names, layers=None):
        self.X = X
        self.obs = pd.DataFrame(obs_data)
        self.var_names = _FakeVarIndex(gene_names)
        self.layers = layers or {}
        self.n_obs = X.shape[0]
        self.n_vars = X.shape[1]


@pytest.fixture
def zone_marker_adata():
    """Tiny AnnData with planted pericentral / periportal zonation.

    Genes include the canonical zonation markers (D-07):
      - pericentral: Glul, Cyp2e1
      - periportal:  Sds, Cyp2f2

    Four spots: two with high pericentral markers, two with high periportal
    markers. The intended per-spot zone is recorded in ``obs['zone_truth']`` so
    a marker-based ``assign_zones`` can be checked against it.

    Gene columns: [Glul, Cyp2e1, Sds, Cyp2f2, Alb]
    """
    # rows = spots, cols = [Glul, Cyp2e1, Sds, Cyp2f2, Alb]
    X = np.array(
        [
            [9.0, 8.0, 0.5, 0.4, 3.0],  # spot 0: high pericentral
            [8.5, 9.0, 0.6, 0.3, 3.1],  # spot 1: high pericentral
            [0.4, 0.5, 9.0, 8.5, 3.0],  # spot 2: high periportal
            [0.3, 0.6, 8.0, 9.0, 3.2],  # spot 3: high periportal
        ],
        dtype=np.float64,
    )
    gene_names = ["Glul", "Cyp2e1", "Sds", "Cyp2f2", "Alb"]
    obs_data = {
        "zone_truth": ["pericentral", "pericentral", "periportal", "periportal"],
        "arm": ["control", "control", "control", "control"],
    }
    return _FakeAnnData(X=X, obs_data=obs_data, gene_names=gene_names)


@pytest.fixture
def ortholog_table_small():
    """Duck-typed OrthologTable with mouse->human one-to-one pairs.

    Mirrors the ``SimpleNamespace(pairs=DataFrame)`` style used in
    ``test_eda_region.py``.
    """
    pairs = pd.DataFrame(
        {
            "human_symbol": ["GLUL", "CYP2E1", "SDS", "CYP2F2"],
            "mouse_symbol": ["Glul", "Cyp2e1", "Sds", "Cyp2f2"],
        }
    )
    return SimpleNamespace(pairs=pairs)
