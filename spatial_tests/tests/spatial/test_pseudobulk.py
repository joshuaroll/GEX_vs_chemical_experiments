"""Unit tests for `src/data/pseudobulk.py`.

All fixtures are small in-memory synthetic matrices — no real data files are
read. This conforms to the project rule "tests may use small in-memory
synthetic fixtures only; never real data files."

Behaviors covered:
  1.  Mean aggregation: per-region means match hand-computed values.
  2.  Median aggregation: per-region medians match hand-computed values.
  3.  Sum aggregation: per-region sums match hand-computed values.
  4.  Region ordering: output rows appear in sorted label order.
  5.  Empty-region (label present in region_labels but absent from spot_labels):
      row is NaN for mean/median, zeros for sum; listed in empty_regions.
  6.  NaN inputs: nanmean/nanmedian/nansum ignore NaN spots; non-NaN positions
      still produce correct values.
  7.  Single-spot region: aggregation collapses to the spot's own values.
  8.  Single region (all spots same label): full-matrix aggregate correct.
  9.  spot_counts dict: correct count per region including zero for empty.
  10. gene_names echoed verbatim in result.
  11. region_labels=None infers regions from spot_labels only.
  12. region_labels explicit superset: extra labels become empty regions.
  13. Input matrix dtype coercion: integer input → float64 output.
  14. ValueError on ndim != 2.
  15. ValueError on len(spot_labels) mismatch.
  16. ValueError on len(gene_names) mismatch.
  17. ValueError on invalid agg mode.
  18. ValueError on empty region_labels sequence (not None).
  19. Determinism: identical inputs → identical output across two calls.
  20. from_anndata: extracts matrix + labels from a duck-typed AnnData-like
      object (no anndata package required for the test).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.spatial.pseudobulk import PseudobulkResult, from_anndata, pseudobulk


# ---------------------------------------------------------------------------
# Shared tiny fixture builders
# ---------------------------------------------------------------------------


def _make_matrix_and_labels() -> tuple[np.ndarray, list[str], list[str]]:
    """Return a 6-spot x 4-gene float matrix, labels, and gene_names.

    Layout:
        spot 0: region 'A'  values [1.0, 2.0, 3.0, 4.0]
        spot 1: region 'B'  values [5.0, 6.0, 7.0, 8.0]
        spot 2: region 'A'  values [3.0, 4.0, 5.0, 6.0]
        spot 3: region 'C'  values [9.0, 0.0, 1.0, 2.0]
        spot 4: region 'B'  values [7.0, 8.0, 9.0, 0.0]
        spot 5: region 'C'  values [1.0, 3.0, 5.0, 7.0]

    Expected means:
        A = [(1+3)/2, (2+4)/2, (3+5)/2, (4+6)/2] = [2.0, 3.0, 4.0, 5.0]
        B = [(5+7)/2, (6+8)/2, (7+9)/2, (8+0)/2] = [6.0, 7.0, 8.0, 4.0]
        C = [(9+1)/2, (0+3)/2, (1+5)/2, (2+7)/2] = [5.0, 1.5, 3.0, 4.5]
    """
    matrix = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [3.0, 4.0, 5.0, 6.0],
        [9.0, 0.0, 1.0, 2.0],
        [7.0, 8.0, 9.0, 0.0],
        [1.0, 3.0, 5.0, 7.0],
    ], dtype=np.float64)
    labels = ["A", "B", "A", "C", "B", "C"]
    gene_names = ["G1", "G2", "G3", "G4"]
    return matrix, labels, gene_names


# ---------------------------------------------------------------------------
# Tests 1-3: aggregation modes
# ---------------------------------------------------------------------------


def test_1_mean_aggregation_per_region():
    matrix, labels, gene_names = _make_matrix_and_labels()
    result = pseudobulk(matrix, labels, gene_names, agg="mean")

    assert isinstance(result, PseudobulkResult)
    idx = {r: i for i, r in enumerate(result.region_order)}

    np.testing.assert_allclose(result.profiles[idx["A"]], [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(result.profiles[idx["B"]], [6.0, 7.0, 8.0, 4.0])
    np.testing.assert_allclose(result.profiles[idx["C"]], [5.0, 1.5, 3.0, 4.5])


def test_2_median_aggregation_per_region():
    # Three-spot region median for odd-count: straightforward with 2-spot
    # regions the median equals the mean.
    matrix, labels, gene_names = _make_matrix_and_labels()
    result = pseudobulk(matrix, labels, gene_names, agg="median")

    idx = {r: i for i, r in enumerate(result.region_order)}
    # For 2 spots, median == mean.
    np.testing.assert_allclose(result.profiles[idx["A"]], [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(result.profiles[idx["B"]], [6.0, 7.0, 8.0, 4.0])
    np.testing.assert_allclose(result.profiles[idx["C"]], [5.0, 1.5, 3.0, 4.5])


def test_2b_median_odd_count():
    """3 spots per region: median picks the middle value, not the mean."""
    matrix = np.array([
        [1.0, 10.0],
        [3.0, 30.0],
        [2.0, 20.0],
    ], dtype=np.float64)
    labels = ["X", "X", "X"]
    gene_names = ["GA", "GB"]
    result = pseudobulk(matrix, labels, gene_names, agg="median")
    # Sorted values per gene: [1,2,3] and [10,20,30]; median = [2.0, 20.0]
    np.testing.assert_allclose(result.profiles[0], [2.0, 20.0])


def test_3_sum_aggregation_per_region():
    matrix, labels, gene_names = _make_matrix_and_labels()
    result = pseudobulk(matrix, labels, gene_names, agg="sum")

    idx = {r: i for i, r in enumerate(result.region_order)}
    np.testing.assert_allclose(result.profiles[idx["A"]], [4.0, 6.0, 8.0, 10.0])
    np.testing.assert_allclose(result.profiles[idx["B"]], [12.0, 14.0, 16.0, 8.0])
    np.testing.assert_allclose(result.profiles[idx["C"]], [10.0, 3.0, 6.0, 9.0])


# ---------------------------------------------------------------------------
# Test 4: region ordering
# ---------------------------------------------------------------------------


def test_4_region_ordering_is_sorted():
    # Labels deliberately in non-alphabetical insertion order.
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ], dtype=np.float64)
    labels = ["Zebra", "Alpha", "Mango"]
    gene_names = ["G1", "G2"]
    result = pseudobulk(matrix, labels, gene_names, agg="mean")

    assert result.region_order == sorted(["Zebra", "Alpha", "Mango"])
    # Verify row alignment matches the order.
    idx = {r: i for i, r in enumerate(result.region_order)}
    np.testing.assert_allclose(result.profiles[idx["Alpha"]], [3.0, 4.0])
    np.testing.assert_allclose(result.profiles[idx["Mango"]], [5.0, 6.0])
    np.testing.assert_allclose(result.profiles[idx["Zebra"]], [1.0, 2.0])


# ---------------------------------------------------------------------------
# Tests 5: empty-region handling
# ---------------------------------------------------------------------------


def test_5a_empty_region_mean_is_nan():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    labels = ["A", "A"]
    gene_names = ["G1", "G2"]
    result = pseudobulk(
        matrix, labels, gene_names, agg="mean",
        region_labels=["A", "B", "C"],
    )
    assert "B" in result.empty_regions
    assert "C" in result.empty_regions
    idx = {r: i for i, r in enumerate(result.region_order)}
    assert np.all(np.isnan(result.profiles[idx["B"]]))
    assert np.all(np.isnan(result.profiles[idx["C"]]))
    # The A row is still correct.
    np.testing.assert_allclose(result.profiles[idx["A"]], [2.0, 3.0])


def test_5b_empty_region_median_is_nan():
    matrix = np.array([[1.0, 2.0]], dtype=np.float64)
    labels = ["only"]
    gene_names = ["G1", "G2"]
    result = pseudobulk(
        matrix, labels, gene_names, agg="median",
        region_labels=["only", "ghost"],
    )
    idx = {r: i for i, r in enumerate(result.region_order)}
    assert "ghost" in result.empty_regions
    assert np.all(np.isnan(result.profiles[idx["ghost"]]))


def test_5c_empty_region_sum_is_zeros():
    matrix = np.array([[1.0, 2.0]], dtype=np.float64)
    labels = ["X"]
    gene_names = ["G1", "G2"]
    result = pseudobulk(
        matrix, labels, gene_names, agg="sum",
        region_labels=["X", "Y"],
    )
    idx = {r: i for i, r in enumerate(result.region_order)}
    assert "Y" in result.empty_regions
    np.testing.assert_allclose(result.profiles[idx["Y"]], [0.0, 0.0])


# ---------------------------------------------------------------------------
# Test 6: NaN inputs
# ---------------------------------------------------------------------------


def test_6_nan_inputs_ignored_in_mean():
    matrix = np.array([
        [1.0, np.nan],
        [3.0, 4.0],
        [np.nan, 2.0],
    ], dtype=np.float64)
    labels = ["R", "R", "R"]
    gene_names = ["G1", "G2"]
    result = pseudobulk(matrix, labels, gene_names, agg="mean")
    # G1: nanmean([1, 3, nan]) = 2.0; G2: nanmean([nan, 4, 2]) = 3.0
    np.testing.assert_allclose(result.profiles[0], [2.0, 3.0])


def test_6b_nan_inputs_ignored_in_sum():
    matrix = np.array([
        [1.0, np.nan],
        [np.nan, 4.0],
    ], dtype=np.float64)
    labels = ["R", "R"]
    gene_names = ["G1", "G2"]
    result = pseudobulk(matrix, labels, gene_names, agg="sum")
    # nansum treats NaN as 0.
    np.testing.assert_allclose(result.profiles[0], [1.0, 4.0])


# ---------------------------------------------------------------------------
# Test 7: single-spot region
# ---------------------------------------------------------------------------


def test_7_single_spot_region():
    matrix = np.array([[7.0, 8.0, 9.0]], dtype=np.float64)
    labels = ["solo"]
    gene_names = ["G1", "G2", "G3"]
    result = pseudobulk(matrix, labels, gene_names, agg="mean")
    np.testing.assert_allclose(result.profiles[0], [7.0, 8.0, 9.0])
    assert result.spot_counts["solo"] == 1


# ---------------------------------------------------------------------------
# Test 8: single region (all spots same label)
# ---------------------------------------------------------------------------


def test_8_single_region_all_spots():
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((10, 5)).astype(np.float64)
    labels = ["only"] * 10
    gene_names = [f"G{i}" for i in range(5)]
    result = pseudobulk(matrix, labels, gene_names, agg="mean")
    assert result.region_order == ["only"]
    np.testing.assert_allclose(result.profiles[0], np.nanmean(matrix, axis=0))


# ---------------------------------------------------------------------------
# Test 9: spot_counts dict
# ---------------------------------------------------------------------------


def test_9_spot_counts_correct():
    matrix, labels, gene_names = _make_matrix_and_labels()
    result = pseudobulk(
        matrix, labels, gene_names, agg="mean",
        region_labels=["A", "B", "C", "D"],
    )
    assert result.spot_counts["A"] == 2
    assert result.spot_counts["B"] == 2
    assert result.spot_counts["C"] == 2
    assert result.spot_counts["D"] == 0


# ---------------------------------------------------------------------------
# Test 10: gene_names echoed
# ---------------------------------------------------------------------------


def test_10_gene_names_echoed():
    matrix = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    labels = ["r"]
    gene_names = ["TP53", "GAPDH", "ACTB"]
    result = pseudobulk(matrix, labels, gene_names, agg="mean")
    assert result.gene_names == ["TP53", "GAPDH", "ACTB"]


# ---------------------------------------------------------------------------
# Tests 11-12: region_labels parameter
# ---------------------------------------------------------------------------


def test_11_region_labels_none_infers_from_spot_labels():
    matrix = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    labels = ["Z", "A", "Z"]
    gene_names = ["G1"]
    result = pseudobulk(matrix, labels, gene_names, agg="mean", region_labels=None)
    assert result.region_order == ["A", "Z"]
    assert result.empty_regions == []


def test_12_explicit_region_labels_superset_creates_empty_rows():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    labels = ["present", "present"]
    gene_names = ["G1", "G2"]
    result = pseudobulk(
        matrix, labels, gene_names, agg="mean",
        region_labels=["present", "absent1", "absent2"],
    )
    assert set(result.region_order) == {"present", "absent1", "absent2"}
    assert set(result.empty_regions) == {"absent1", "absent2"}
    idx = {r: i for i, r in enumerate(result.region_order)}
    np.testing.assert_allclose(result.profiles[idx["present"]], [2.0, 3.0])
    assert np.all(np.isnan(result.profiles[idx["absent1"]]))
    assert np.all(np.isnan(result.profiles[idx["absent2"]]))


# ---------------------------------------------------------------------------
# Test 13: integer input dtype → float64 output
# ---------------------------------------------------------------------------


def test_13_integer_input_coerced_to_float64():
    matrix = np.array([[1, 2], [3, 4]], dtype=np.int32)
    labels = ["A", "B"]
    gene_names = ["G1", "G2"]
    result = pseudobulk(matrix, labels, gene_names, agg="mean")
    assert result.profiles.dtype == np.float64
    idx = {r: i for i, r in enumerate(result.region_order)}
    np.testing.assert_allclose(result.profiles[idx["A"]], [1.0, 2.0])
    np.testing.assert_allclose(result.profiles[idx["B"]], [3.0, 4.0])


# ---------------------------------------------------------------------------
# Tests 14-18: error cases
# ---------------------------------------------------------------------------


def test_14_value_error_ndim_not_2():
    matrix_1d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="2-D"):
        pseudobulk(matrix_1d, ["A", "B", "C"], ["G1", "G2", "G3"])


def test_14b_value_error_ndim_3():
    matrix_3d = np.zeros((2, 3, 4))
    with pytest.raises(ValueError, match="2-D"):
        pseudobulk(matrix_3d, ["A", "B"], ["G1", "G2", "G3"])


def test_15_value_error_spot_labels_length_mismatch():
    matrix = np.zeros((5, 3))
    with pytest.raises(ValueError, match="spot_labels"):
        pseudobulk(matrix, ["A", "B"], ["G1", "G2", "G3"])


def test_16_value_error_gene_names_length_mismatch():
    matrix = np.zeros((3, 4))
    with pytest.raises(ValueError, match="gene_names"):
        pseudobulk(matrix, ["A", "B", "C"], ["G1", "G2"])


def test_17_value_error_invalid_agg_mode():
    matrix = np.zeros((2, 3))
    with pytest.raises(ValueError, match="agg"):
        pseudobulk(matrix, ["A", "B"], ["G1", "G2", "G3"], agg="variance")  # type: ignore[arg-type]


def test_18_value_error_empty_region_labels_sequence():
    matrix = np.zeros((2, 3))
    with pytest.raises(ValueError, match="region_labels"):
        pseudobulk(matrix, ["A", "B"], ["G1", "G2", "G3"], region_labels=[])


# ---------------------------------------------------------------------------
# Test 19: determinism
# ---------------------------------------------------------------------------


def test_19_determinism():
    rng = np.random.default_rng(7)
    matrix = rng.standard_normal((20, 10)).astype(np.float64)
    labels = [f"R{i % 4}" for i in range(20)]
    gene_names = [f"G{i}" for i in range(10)]

    r1 = pseudobulk(matrix, labels, gene_names, agg="mean")
    r2 = pseudobulk(matrix, labels, gene_names, agg="mean")

    np.testing.assert_array_equal(r1.profiles, r2.profiles)
    assert r1.region_order == r2.region_order
    assert r1.empty_regions == r2.empty_regions
    assert r1.spot_counts == r2.spot_counts


# ---------------------------------------------------------------------------
# Test 20: from_anndata with a duck-typed AnnData-like object
# ---------------------------------------------------------------------------


class _FakeVarIndex:
    """Minimal stand-in for adata.var_names (array-like of str)."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def astype(self, dtype: type) -> "_FakeVarIndex":
        return _FakeVarIndex([dtype(n) for n in self._names])

    def __iter__(self):
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


class _FakeAnnData:
    """Duck-typed AnnData-like object. No anndata package required."""

    def __init__(
        self,
        X: np.ndarray,
        obs_col_data: dict[str, list[str]],
        gene_names: list[str],
        layers: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        self.X = X
        self.obs = pd.DataFrame(obs_col_data)
        self.var_names = _FakeVarIndex(gene_names)
        self.layers: dict[str, np.ndarray] = layers or {}


def test_20_from_anndata_extracts_matrix_and_labels():
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ], dtype=np.float64)
    adata = _FakeAnnData(
        X=matrix,
        obs_col_data={"region": ["R1", "R2", "R1"]},
        gene_names=["GENE_A", "GENE_B"],
    )
    result = from_anndata(adata, obs_col="region", agg="mean")

    assert result.gene_names == ["GENE_A", "GENE_B"]
    assert result.region_order == ["R1", "R2"]
    idx = {r: i for i, r in enumerate(result.region_order)}
    # R1 spots: rows 0 and 2 → mean = [3.0, 4.0]
    np.testing.assert_allclose(result.profiles[idx["R1"]], [3.0, 4.0])
    # R2 spot: row 1 → [3.0, 4.0]
    np.testing.assert_allclose(result.profiles[idx["R2"]], [3.0, 4.0])


def test_20b_from_anndata_uses_layer_when_specified():
    X_base = np.zeros((3, 2), dtype=np.float64)
    layer_data = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float64)
    adata = _FakeAnnData(
        X=X_base,
        obs_col_data={"tissue": ["T1", "T1", "T2"]},
        gene_names=["GA", "GB"],
        layers={"normalized": layer_data},
    )
    result = from_anndata(adata, obs_col="tissue", agg="mean", layer="normalized")

    idx = {r: i for i, r in enumerate(result.region_order)}
    np.testing.assert_allclose(result.profiles[idx["T1"]], [20.0, 30.0])
    np.testing.assert_allclose(result.profiles[idx["T2"]], [50.0, 60.0])


def test_20c_from_anndata_raises_on_missing_obs_col():
    matrix = np.zeros((2, 2))
    adata = _FakeAnnData(
        X=matrix,
        obs_col_data={"region": ["A", "B"]},
        gene_names=["G1", "G2"],
    )
    with pytest.raises(KeyError, match="obs_col"):
        from_anndata(adata, obs_col="nonexistent", agg="mean")


def test_20d_from_anndata_raises_on_missing_layer():
    matrix = np.zeros((2, 2))
    adata = _FakeAnnData(
        X=matrix,
        obs_col_data={"region": ["A", "B"]},
        gene_names=["G1", "G2"],
    )
    with pytest.raises(KeyError, match="layer"):
        from_anndata(adata, obs_col="region", agg="mean", layer="nosuchlayer")


def test_20e_from_anndata_handles_sparse_matrix():
    """from_anndata must handle scipy-sparse-like objects (has .toarray())."""
    try:
        import scipy.sparse as sp
        sparse_matrix = sp.csr_matrix(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        )
    except ImportError:
        # scipy not installed — use a minimal duck-type stand-in.
        class _DenseFake:
            def __init__(self, arr):
                self._arr = arr
            def toarray(self):
                return self._arr
        sparse_matrix = _DenseFake(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        )

    adata = _FakeAnnData(
        X=sparse_matrix,
        obs_col_data={"region": ["A", "B", "A"]},
        gene_names=["G1", "G2"],
    )
    result = from_anndata(adata, obs_col="region", agg="mean")
    idx = {r: i for i, r in enumerate(result.region_order)}
    np.testing.assert_allclose(result.profiles[idx["A"]], [3.0, 4.0])
    np.testing.assert_allclose(result.profiles[idx["B"]], [3.0, 4.0])
