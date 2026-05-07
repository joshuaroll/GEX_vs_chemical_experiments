"""Unit tests for `src/data/wangli_lincs_lookup.py`.

Synthetic-only fixtures — no real LINCS h5 reads. The full 1.4 GB
`Bayesian_GSE92742_Level5_COMPZ_n361481x978.h5` is exercised only by the
Wave-2 CLI driver (Plan 01-04), per the project rule "no real-data unit tests".

Tests cover behaviors 1-12 in `01-02-PLAN.md`:
  1. open_lincs_h5 returns a working file handle for a valid fixture.
  2. open_lincs_h5 raises ValueError when /data is missing.
  3. open_lincs_h5 raises ValueError when /data shape disagrees with /colid x /rowid.
  4. lookup_inst_ids returns aligned (N,gene_count) matrix when all ids are present.
  5. lookup_inst_ids returns matrix + missing_ids list on partial-miss query.
  6. lookup_inst_ids returns (0, gene_count) matrix and full missing_ids on all-miss.
  7. lookup_inst_ids returns float32 dtype.
  8. lookup_inst_ids decodes bytes-stored colid (h5py default) under str query.
  9. crossvalidate_pearson returns 1.0 per gene for identical arrays.
  10. crossvalidate_pearson returns -1.0 per gene for sign-inverted arrays.
  11. crossvalidate_pearson returns NaN for zero-variance genes.
  12. crossvalidate_pearson returns shape (gene_count,) dtype float64.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data.wangli_lincs_lookup import (
    crossvalidate_pearson,
    lookup_inst_ids,
    open_lincs_h5,
)


# ---------------------------------------------------------------------------
# Fixture factories
# ---------------------------------------------------------------------------


def _write_h5(
    path: Path,
    *,
    n_profiles: int,
    n_genes: int,
    colid_dtype: str = "bytes",
    omit_data: bool = False,
    data_shape: tuple[int, int] | None = None,
    seed: int = 0,
) -> tuple[Path, np.ndarray]:
    """Write a minimal LINCS-shaped fixture h5 to `path`.

    Returns
    -------
    path : Path
        The fixture file path (echoed for caller convenience).
    data : np.ndarray
        The float32 (n_profiles, n_genes) matrix written to /data, so tests
        can assert against the same numbers the impl will read back.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_profiles, n_genes)).astype(np.float32)

    # colids are sequential `id_0000`, `id_0001`, ...
    colids_str = [f"id_{i:04d}" for i in range(n_profiles)]
    rowids_str = [f"gene_{j:03d}" for j in range(n_genes)]

    if colid_dtype == "bytes":
        colids = np.array([s.encode("utf-8") for s in colids_str], dtype="S16")
        rowids = np.array([s.encode("utf-8") for s in rowids_str], dtype="S16")
    elif colid_dtype == "str":
        # h5py variable-length unicode
        str_dtype = h5py.string_dtype(encoding="utf-8")
        colids = np.array(colids_str, dtype=str_dtype)
        rowids = np.array(rowids_str, dtype=str_dtype)
    else:
        raise ValueError(f"unsupported colid_dtype={colid_dtype}")

    with h5py.File(path, "w") as f:
        f.create_dataset("colid", data=colids)
        f.create_dataset("rowid", data=rowids)
        if not omit_data:
            shape = data_shape if data_shape is not None else data.shape
            payload = data if shape == data.shape else rng.standard_normal(shape).astype(np.float32)
            f.create_dataset("data", data=payload)
    return path, data


@pytest.fixture
def small_h5(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """A 100x5 fixture with bytes-stored colid (h5py default)."""
    return _write_h5(tmp_path / "small.h5", n_profiles=100, n_genes=5)


@pytest.fixture
def small_h5_str(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """A 100x5 fixture with variable-length-str colid."""
    return _write_h5(tmp_path / "small_str.h5", n_profiles=100, n_genes=5,
                     colid_dtype="str")


@pytest.fixture
def tiny_h5(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """A 10x5 fixture for open_lincs_h5 happy path."""
    return _write_h5(tmp_path / "tiny.h5", n_profiles=10, n_genes=5)


# ---------------------------------------------------------------------------
# Tests 1-3: open_lincs_h5
# ---------------------------------------------------------------------------


def test_open_lincs_h5_returns_file_handle(tiny_h5: tuple[Path, np.ndarray]) -> None:
    path, _ = tiny_h5
    f = open_lincs_h5(path)
    try:
        assert isinstance(f, h5py.File)
        assert "colid" in f
        assert "rowid" in f
        assert "data" in f
        assert f["data"].shape == (10, 5)
    finally:
        f.close()


def test_open_lincs_h5_raises_on_missing_dataset(tmp_path: Path) -> None:
    bad_path = tmp_path / "missing_data.h5"
    _write_h5(bad_path, n_profiles=10, n_genes=5, omit_data=True)
    with pytest.raises(ValueError, match=r"data"):
        open_lincs_h5(bad_path)


def test_open_lincs_h5_raises_on_shape_mismatch(tmp_path: Path) -> None:
    bad_path = tmp_path / "shape_mismatch.h5"
    # /colid is length 10 but /data is (15, 5) — should fail validation.
    _write_h5(bad_path, n_profiles=10, n_genes=5, data_shape=(15, 5))
    with pytest.raises(ValueError, match=r"shape"):
        open_lincs_h5(bad_path)


# ---------------------------------------------------------------------------
# Tests 4-8: lookup_inst_ids
# ---------------------------------------------------------------------------


def test_lookup_inst_ids_all_found_returns_aligned_matrix(
    small_h5: tuple[Path, np.ndarray],
) -> None:
    path, data = small_h5
    query = ["id_0050", "id_0010", "id_0099"]
    matrix, found_ids, missing_ids = lookup_inst_ids(path, query)

    assert matrix.shape == (3, 5)
    assert found_ids == query  # all found, in input order
    assert missing_ids == []
    # Check row-content alignment to input order, not row-index order.
    np.testing.assert_array_equal(matrix[0], data[50])
    np.testing.assert_array_equal(matrix[1], data[10])
    np.testing.assert_array_equal(matrix[2], data[99])


def test_lookup_inst_ids_partial_found_returns_missing_list(
    small_h5: tuple[Path, np.ndarray],
) -> None:
    path, data = small_h5
    query = ["id_0010", "id_NOTREAL", "id_0050"]
    matrix, found_ids, missing_ids = lookup_inst_ids(path, query)

    assert matrix.shape == (2, 5)
    assert found_ids == ["id_0010", "id_0050"]
    assert missing_ids == ["id_NOTREAL"]
    np.testing.assert_array_equal(matrix[0], data[10])
    np.testing.assert_array_equal(matrix[1], data[50])


def test_lookup_inst_ids_none_found_returns_empty_matrix(
    small_h5: tuple[Path, np.ndarray],
) -> None:
    path, _ = small_h5
    query = ["id_BAD1", "id_BAD2"]
    matrix, found_ids, missing_ids = lookup_inst_ids(path, query)

    assert matrix.shape == (0, 5)
    assert found_ids == []
    assert missing_ids == ["id_BAD1", "id_BAD2"]


def test_lookup_inst_ids_dtype_float32(small_h5: tuple[Path, np.ndarray]) -> None:
    path, _ = small_h5
    matrix, _, _ = lookup_inst_ids(path, ["id_0001", "id_0002"])
    assert matrix.dtype == np.float32


def test_lookup_inst_ids_decodes_bytes_colid(
    small_h5: tuple[Path, np.ndarray],
) -> None:
    """small_h5 stores colid as bytes (h5py default for fixed-length strings).

    Querying with str inputs must still work — the lookup must decode UTF-8
    internally.
    """
    path, data = small_h5
    matrix, found_ids, missing_ids = lookup_inst_ids(path, ["id_0042"])

    assert matrix.shape == (1, 5)
    assert found_ids == ["id_0042"]
    assert missing_ids == []
    np.testing.assert_array_equal(matrix[0], data[42])


def test_lookup_inst_ids_works_on_str_colid(
    small_h5_str: tuple[Path, np.ndarray],
) -> None:
    """Variable-length str colid should also work — covers the inverse of
    test_lookup_inst_ids_decodes_bytes_colid for symmetry."""
    path, data = small_h5_str
    matrix, found_ids, missing_ids = lookup_inst_ids(path, ["id_0007", "id_0042"])

    assert matrix.shape == (2, 5)
    assert found_ids == ["id_0007", "id_0042"]
    assert missing_ids == []
    np.testing.assert_array_equal(matrix[0], data[7])
    np.testing.assert_array_equal(matrix[1], data[42])


# ---------------------------------------------------------------------------
# Tests 9-12: crossvalidate_pearson
# ---------------------------------------------------------------------------


def test_crossvalidate_pearson_identical_arrays() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((20, 5)).astype(np.float32)
    pear = crossvalidate_pearson(arr, arr.copy())

    assert pear.shape == (5,)
    np.testing.assert_allclose(pear, np.ones(5), rtol=0, atol=1e-6)


def test_crossvalidate_pearson_inverted_signs() -> None:
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((20, 5)).astype(np.float32)
    pear = crossvalidate_pearson(arr, -arr)

    assert pear.shape == (5,)
    np.testing.assert_allclose(pear, -np.ones(5), rtol=0, atol=1e-6)


def test_crossvalidate_pearson_zero_variance_returns_nan() -> None:
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((20, 5)).astype(np.float32)
    # Make gene 2 constant in arr1 (zero variance) → Pearson undefined → NaN.
    arr1 = arr.copy()
    arr1[:, 2] = 7.0
    arr2 = arr.copy()

    pear = crossvalidate_pearson(arr1, arr2)
    assert pear.shape == (5,)
    assert np.isnan(pear[2])
    # The other genes are still finite (non-zero variance, possibly correlated
    # less than 1 since arr1 == arr2 except gene 2 — but per-gene Pearson is
    # computed independently, so genes 0/1/3/4 still equal exactly 1.0).
    other = np.delete(pear, 2)
    np.testing.assert_allclose(other, np.ones(4), rtol=0, atol=1e-6)


def test_crossvalidate_pearson_shape_and_dtype() -> None:
    rng = np.random.default_rng(3)
    arr1 = rng.standard_normal((20, 5)).astype(np.float32)
    arr2 = rng.standard_normal((20, 5)).astype(np.float32)
    pear = crossvalidate_pearson(arr1, arr2)

    assert pear.shape == (5,)
    assert pear.dtype == np.float64
