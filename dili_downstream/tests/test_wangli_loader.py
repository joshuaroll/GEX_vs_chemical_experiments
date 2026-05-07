"""Unit tests for `src/data/wangli_loader.py`.

In-memory fixture tests only. No real xlsx or pickle files are touched here —
the Wave-2 driver exercises the real Wang/Li files.

Tests cover the 10 behaviors in 01-01-PLAN.md:
  1. `load_inst_ids` returns a list of 6000 string inst_ids from a fabricated xlsx.
  2. Wrong row count → ValueError mentioning "6000".
  3. Whitespace around inst_ids is stripped.
  4. dict-format pickle parses to WangliDataset with N=10 (978 expr, 50 splits).
  5. DataFrame-format pickle parses to WangliDataset with same N and shapes.
  6. `dili_binary` is int8 with values strictly in {0, 1}.
  7. `split_flags` is bool dtype, shape (N, 50).
  8. `expressions` is float32, shape (N, 978).
  9. Pickle missing `dili_binary` → ValueError mentioning the missing key.
  10. inst_ids are returned as list[str] when the pickle exposes them, else None.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.wangli_loader import (
    WangliDataset,
    load_inst_ids,
    load_split_pickle,
)


# ---------------------------------------------------------------------------
# Helpers — fabricate synthetic Wang/Li-shaped artifacts
# ---------------------------------------------------------------------------

def _make_inst_id(i: int) -> str:
    """Fabricate a LINCS Level-5 inst_id matching the locked regex pattern.

    Pattern: `{plate}_{cell}_{time}H:BRD-K{8 digits}:{dose}` (paper format).
    """
    cells = ["MCF7", "HEPG2", "A549", "PC3", "HA1E"]
    cell = cells[i % len(cells)]
    plate = f"CGS{(i % 999):03d}"
    time = 6 if i % 2 == 0 else 24
    brd = f"BRD-K{(i % 100000000):08d}"
    dose = "10" if i % 3 == 0 else "3.33333"
    return f"{plate}_{cell}_{time}H:{brd}:{dose}"


def _write_xlsx_with_n_inst_ids(path: Path, n: int, *, with_whitespace: bool = False) -> None:
    """Write a single-column xlsx of `n` fabricated inst_ids."""
    ids = [_make_inst_id(i) for i in range(n)]
    if with_whitespace:
        ids = [f"   {x}  " for x in ids]
    df = pd.DataFrame({"profile_id": ids})
    df.to_excel(path, index=False)


def _make_dict_pickle(n: int = 10, *, include_inst_ids: bool = False) -> dict:
    """Build a dict-format Wang/Li pickle with N rows, 978 genes, 50 splits."""
    rng = np.random.default_rng(42)
    obj: dict = {
        "compound_names": [f"compound_{i}" for i in range(n)],
        "dili_binary": np.array([i % 2 for i in range(n)], dtype=np.int8),
        "expressions": rng.standard_normal((n, 978)).astype(np.float32),
        "split_flags": rng.integers(0, 2, size=(n, 50)).astype(bool),
    }
    if include_inst_ids:
        obj["inst_ids"] = [_make_inst_id(i) for i in range(n)]
    return obj


def _make_dataframe_pickle(n: int = 10, *, include_inst_ids: bool = False) -> pd.DataFrame:
    """Build a DataFrame-format Wang/Li pickle with N rows + 978 expr + 50 split cols."""
    rng = np.random.default_rng(42)
    cols: dict = {
        "compound_name": [f"compound_{i}" for i in range(n)],
        "DILI": [i % 2 for i in range(n)],
    }
    if include_inst_ids:
        cols["inst_id"] = [_make_inst_id(i) for i in range(n)]
    expr = rng.standard_normal((n, 978)).astype(np.float32)
    for j in range(978):
        cols[f"expression_{j}"] = expr[:, j]
    splits = rng.integers(0, 2, size=(n, 50)).astype(bool)
    for j in range(50):
        cols[f"split_{j:02d}"] = splits[:, j]
    return pd.DataFrame(cols)


def _dump_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ---------------------------------------------------------------------------
# Tests for load_inst_ids
# ---------------------------------------------------------------------------

def test_load_inst_ids_returns_6000_strings(tmp_path):
    """Test 1: fabricated xlsx with 6000 inst_ids → returns list of 6000 str."""
    xlsx = tmp_path / "ids.xlsx"
    _write_xlsx_with_n_inst_ids(xlsx, 6000)

    ids = load_inst_ids(xlsx)
    assert isinstance(ids, list)
    assert len(ids) == 6000
    assert all(isinstance(x, str) for x in ids)


def test_load_inst_ids_raises_on_wrong_row_count(tmp_path):
    """Test 2: xlsx with 5 rows → ValueError mentioning 6000."""
    xlsx = tmp_path / "ids.xlsx"
    _write_xlsx_with_n_inst_ids(xlsx, 5)

    with pytest.raises(ValueError, match=r"6000"):
        load_inst_ids(xlsx)


def test_load_inst_ids_strips_whitespace(tmp_path):
    """Test 3: leading/trailing whitespace on each inst_id is stripped."""
    xlsx = tmp_path / "ids.xlsx"
    _write_xlsx_with_n_inst_ids(xlsx, 6000, with_whitespace=True)

    ids = load_inst_ids(xlsx)
    # No leading/trailing whitespace anywhere
    assert all(x == x.strip() for x in ids)
    # And the first one matches the regex pattern (no spaces)
    assert not ids[0].startswith(" ")
    assert not ids[0].endswith(" ")


# ---------------------------------------------------------------------------
# Tests for load_split_pickle (dict format)
# ---------------------------------------------------------------------------

def test_load_split_pickle_dict_format(tmp_path):
    """Test 4: dict pickle with N=10 → WangliDataset; expr (10,978), splits (10,50)."""
    pkl = tmp_path / "split.pickle"
    _dump_pickle(_make_dict_pickle(n=10), pkl)

    ds = load_split_pickle(pkl)
    assert isinstance(ds, WangliDataset)
    assert len(ds.compound_names) == 10
    assert ds.expressions.shape == (10, 978)
    assert ds.split_flags.shape == (10, 50)
    assert ds.dili_binary.shape == (10,)


# ---------------------------------------------------------------------------
# Tests for load_split_pickle (DataFrame format)
# ---------------------------------------------------------------------------

def test_load_split_pickle_dataframe_format(tmp_path):
    """Test 5: DataFrame pickle (compound_name, DILI, 978 expr cols, 50 split cols)."""
    pkl = tmp_path / "split.pickle"
    _dump_pickle(_make_dataframe_pickle(n=10), pkl)

    ds = load_split_pickle(pkl)
    assert isinstance(ds, WangliDataset)
    assert len(ds.compound_names) == 10
    assert ds.expressions.shape == (10, 978)
    assert ds.split_flags.shape == (10, 50)
    assert ds.dili_binary.shape == (10,)


# ---------------------------------------------------------------------------
# Dtype invariants
# ---------------------------------------------------------------------------

def test_load_split_pickle_dili_binary_dtype(tmp_path):
    """Test 6: dili_binary is int8 with values strictly in {0, 1}."""
    pkl = tmp_path / "split.pickle"
    _dump_pickle(_make_dict_pickle(n=10), pkl)

    ds = load_split_pickle(pkl)
    assert ds.dili_binary.dtype == np.int8
    assert set(np.unique(ds.dili_binary).tolist()).issubset({0, 1})


def test_load_split_pickle_split_flags_dtype(tmp_path):
    """Test 7: split_flags dtype is bool, shape (N, 50)."""
    pkl = tmp_path / "split.pickle"
    _dump_pickle(_make_dict_pickle(n=10), pkl)

    ds = load_split_pickle(pkl)
    assert ds.split_flags.dtype == np.bool_
    assert ds.split_flags.shape == (10, 50)


def test_load_split_pickle_expressions_dtype(tmp_path):
    """Test 8: expressions dtype is float32, shape (N, 978)."""
    pkl = tmp_path / "split.pickle"
    _dump_pickle(_make_dict_pickle(n=10), pkl)

    ds = load_split_pickle(pkl)
    assert ds.expressions.dtype == np.float32
    assert ds.expressions.shape == (10, 978)


# ---------------------------------------------------------------------------
# Schema-error path
# ---------------------------------------------------------------------------

def test_load_split_pickle_raises_on_missing_keys(tmp_path):
    """Test 9: dict pickle missing `dili_binary` → ValueError mentioning that key."""
    pkl = tmp_path / "split.pickle"
    bad = _make_dict_pickle(n=10)
    del bad["dili_binary"]
    _dump_pickle(bad, pkl)

    with pytest.raises(ValueError, match=r"dili_binary|DILI|label"):
        load_split_pickle(pkl)


# ---------------------------------------------------------------------------
# Optional inst_ids field
# ---------------------------------------------------------------------------

def test_load_split_pickle_inst_ids_optional(tmp_path):
    """Test 10: inst_ids field is None when absent, list[str] of length N when present."""
    # No inst_ids
    pkl_a = tmp_path / "no_ids.pickle"
    _dump_pickle(_make_dict_pickle(n=10, include_inst_ids=False), pkl_a)
    ds_a = load_split_pickle(pkl_a)
    assert ds_a.inst_ids is None

    # With inst_ids — dict format
    pkl_b = tmp_path / "with_ids.pickle"
    _dump_pickle(_make_dict_pickle(n=10, include_inst_ids=True), pkl_b)
    ds_b = load_split_pickle(pkl_b)
    assert isinstance(ds_b.inst_ids, list)
    assert len(ds_b.inst_ids) == 10
    assert all(isinstance(x, str) for x in ds_b.inst_ids)

    # With inst_ids — DataFrame format
    pkl_c = tmp_path / "df_with_ids.pickle"
    _dump_pickle(_make_dataframe_pickle(n=10, include_inst_ids=True), pkl_c)
    ds_c = load_split_pickle(pkl_c)
    assert isinstance(ds_c.inst_ids, list)
    assert len(ds_c.inst_ids) == 10
    assert all(isinstance(x, str) for x in ds_c.inst_ids)
