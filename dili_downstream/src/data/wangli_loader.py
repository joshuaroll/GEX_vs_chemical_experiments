"""Pure library for parsing Wang/Li 2020's two source artifacts.

Phase 1 / Plan 01-01 deliverable. The CLI driver in `scripts/build_phase1_wangli.py`
(Wave 2) wires this with the LINCS h5 lookup and the SMILES resolver.

Inputs (per L1000_DILI repo + Li, Tong et al. 2020 §Methods):
    1. The Wang/Li transcriptomic-profiles xlsx — 6,000 LINCS Level-5 inst_ids
       (`{plate}_{cell}_{time}H:BRD-K{8 digits}:{dose}`) post Kennard-Stone
       selection from 23,791 mapped DILIst-LINCS profiles.
    2. The Wang/Li drug-split pickle (Synapse syn22910750) — per-profile
       (compound_name, DILI label, 978-gene Level-5 expressions, 50 train/test
       split flags). The pickle structure is not formally documented in the
       L1000_DILI README, so this loader does adaptive parsing on dict /
       DataFrame / list-of-dicts layouts.

Locked output schema (CONTEXT.md §Decisions):
    WangliDataset(
        compound_names: list[str]                  # length N
        dili_binary:    np.ndarray int8  (N,)      # values {0, 1}
        expressions:    np.ndarray float32 (N, 978)
        split_flags:    np.ndarray bool   (N, 50)  # True = train, False = test
        inst_ids:       Optional[list[str]]        # None if pickle omits them
    )

Hard rules honored:
    - Pure library: NO source-of-truth paths, NO hardcoded absolute paths.
    - Caller passes file paths in; the function reads them with the standard
      Python/pandas/pickle stack and validates schema.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class WangliDataset(NamedTuple):
    """Locked output of `load_split_pickle`.

    See module docstring for shape/dtype contract.
    """

    compound_names: list[str]
    dili_binary: np.ndarray
    expressions: np.ndarray
    split_flags: np.ndarray
    inst_ids: Optional[list[str]]


# ---------------------------------------------------------------------------
# Regex for LINCS Level-5 inst_id sniffing in the xlsx
# ---------------------------------------------------------------------------
#
# Real Wang/Li examples:
#   CGS001_MCF7_24H:BRD-K12345678:10
#   CPC020_HEPG2_6H:BRD-K01234567:3.33333
#
# We intentionally allow variable BRD digit counts and any non-whitespace dose
# token (handles "10", "3.33333", and any future dose representation).
_SIG_ID_REGEX = re.compile(r"^[A-Z0-9]+_[A-Z0-9]+_\d+H:BRD-[A-Z][0-9]+:\S+$")


# ---------------------------------------------------------------------------
# load_inst_ids
# ---------------------------------------------------------------------------


def load_inst_ids(xlsx_path: str | Path) -> list[str]:
    """Read Wang/Li's transcriptomic-profiles xlsx → 6,000 inst_id strings.

    The xlsx exposes the 6,000 inst_ids in a single `object`-dtype column. We
    sniff the column by matching values against `_SIG_ID_REGEX` (LINCS Level-5
    sig_id). Whitespace is stripped from every value before validation.

    Parameters
    ----------
    xlsx_path : path-like
        Path to the xlsx. Caller's responsibility to point at the real file.

    Returns
    -------
    list[str]
        Exactly 6,000 stripped inst_id strings, in source-row order.

    Raises
    ------
    ValueError
        If row count != 6000, or no column matches the sig_id pattern.
    """
    df = pd.read_excel(Path(xlsx_path), sheet_name=0)

    # Pick the first object-dtype column whose non-null entries match the
    # LINCS Level-5 sig_id regex on a small sample. Wang/Li's file has just
    # one such column, but we don't want to hardcode the column name.
    chosen_col: Optional[str] = None
    for col in df.columns:
        series = df[col].dropna().astype(str).str.strip()
        if len(series) == 0:
            continue
        sample = series.head(20)
        if all(_SIG_ID_REGEX.match(v) for v in sample):
            chosen_col = col
            break

    if chosen_col is None:
        raise ValueError(
            "Could not locate a LINCS Level-5 inst_id column in the xlsx. "
            f"Inspected columns: {df.columns.tolist()}. "
            "Expected pattern like '<plate>_<cell>_<time>H:BRD-K########:<dose>'."
        )

    raw = df[chosen_col].astype(str).tolist()
    stripped = [v.strip() for v in raw]
    if any(orig != stripped_v for orig, stripped_v in zip(raw, stripped)):
        log.warning(
            "load_inst_ids: stripped leading/trailing whitespace from one or "
            "more inst_ids in column %r", chosen_col,
        )

    if len(stripped) != 6000:
        raise ValueError(
            f"Expected 6000 inst_ids in {xlsx_path}, got {len(stripped)} "
            f"(column {chosen_col!r}). Wang/Li's published file has exactly 6000 rows."
        )

    return stripped


# ---------------------------------------------------------------------------
# load_split_pickle — adaptive parsing
# ---------------------------------------------------------------------------


# Aliases for the four mandatory fields plus the optional inst_ids field.
# Order in each list reflects preference (first match wins).
_COMPOUND_KEYS = ("compound_names", "compound_name", "drug_name", "drug_names")
_LABEL_KEYS = ("dili_binary", "DILI", "dili", "label", "labels", "y")
_EXPR_KEYS = ("expressions", "expression", "gene_expression", "X", "features")
_SPLIT_KEYS = ("split_flags", "splits", "split")
_INST_KEYS = ("inst_ids", "inst_id", "profile_ids", "profile_id", "sig_ids")


def _first_present(d: dict, keys: tuple[str, ...]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def _coerce_dili_binary(arr: Any, n: int) -> np.ndarray:
    arr = np.asarray(arr).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(
            f"dili_binary length mismatch: got {arr.shape[0]}, expected {n}"
        )
    out = arr.astype(np.int8, copy=False)
    uniq = set(np.unique(out).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(
            f"dili_binary values must be in {{0, 1}}; got unique values {sorted(uniq)}"
        )
    return out


def _coerce_expressions(arr: Any, n: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape != (n, 978):
        raise ValueError(
            f"expressions shape must be ({n}, 978); got {arr.shape}"
        )
    return arr.astype(np.float32, copy=False)


def _coerce_split_flags(arr: Any, n: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape != (n, 50):
        raise ValueError(
            f"split_flags shape must be ({n}, 50); got {arr.shape}"
        )
    return arr.astype(bool, copy=False)


def _parse_dict(obj: dict) -> WangliDataset:
    """Adaptive parse for dict-of-arrays layout."""
    log.debug("load_split_pickle: dict layout, keys=%s", sorted(obj.keys()))

    # ------- compound_names -------
    cname_key = _first_present(obj, _COMPOUND_KEYS)
    if cname_key is None:
        raise ValueError(
            "Pickle dict missing compound name field "
            f"(searched {_COMPOUND_KEYS}, got keys {sorted(obj.keys())})"
        )
    compound_names = [str(x) for x in obj[cname_key]]
    n = len(compound_names)

    # ------- dili_binary -------
    label_key = _first_present(obj, _LABEL_KEYS)
    if label_key is None:
        raise ValueError(
            "Pickle dict missing dili_binary / DILI / label field "
            f"(searched {_LABEL_KEYS}, got keys {sorted(obj.keys())})"
        )
    dili_binary = _coerce_dili_binary(obj[label_key], n)

    # ------- expressions -------
    expr_key = _first_present(obj, _EXPR_KEYS)
    if expr_key is None:
        raise ValueError(
            "Pickle dict missing expressions field "
            f"(searched {_EXPR_KEYS}, got keys {sorted(obj.keys())})"
        )
    expressions = _coerce_expressions(obj[expr_key], n)

    # ------- split_flags -------
    split_key = _first_present(obj, _SPLIT_KEYS)
    if split_key is not None:
        split_flags = _coerce_split_flags(obj[split_key], n)
    else:
        # Try to assemble from 50 separate split_NN keys.
        per_split = sorted(
            k for k in obj.keys()
            if isinstance(k, str) and re.match(r"^split_\d+$", k)
        )
        if len(per_split) != 50:
            raise ValueError(
                "Pickle dict missing split_flags field "
                f"(searched {_SPLIT_KEYS} and `split_NN` keys; "
                f"found {len(per_split)} per-split keys, expected 50; "
                f"got keys {sorted(obj.keys())})"
            )
        split_flags = np.column_stack([np.asarray(obj[k]).reshape(-1) for k in per_split])
        split_flags = _coerce_split_flags(split_flags, n)

    # ------- inst_ids (optional) -------
    inst_key = _first_present(obj, _INST_KEYS)
    if inst_key is not None:
        raw = obj[inst_key]
        if len(raw) != n:
            raise ValueError(
                f"inst_ids length {len(raw)} does not match N={n}"
            )
        inst_ids: Optional[list[str]] = [str(x) for x in raw]
    else:
        inst_ids = None

    return WangliDataset(
        compound_names=compound_names,
        dili_binary=dili_binary,
        expressions=expressions,
        split_flags=split_flags,
        inst_ids=inst_ids,
    )


def _parse_dataframe(df: pd.DataFrame) -> WangliDataset:
    """Adaptive parse for DataFrame-of-rows layout."""
    log.debug("load_split_pickle: DataFrame layout, columns=%s", df.columns.tolist())

    cols_lower = {c.lower(): c for c in df.columns}

    # ------- compound_names -------
    cname_col: Optional[str] = None
    for k in _COMPOUND_KEYS:
        if k.lower() in cols_lower:
            cname_col = cols_lower[k.lower()]
            break
    if cname_col is None:
        raise ValueError(
            "Pickle DataFrame missing compound name column "
            f"(searched {_COMPOUND_KEYS}, got columns {df.columns.tolist()})"
        )
    compound_names = [str(x) for x in df[cname_col].tolist()]
    n = len(compound_names)

    # ------- dili_binary -------
    label_col: Optional[str] = None
    for k in _LABEL_KEYS:
        if k.lower() in cols_lower:
            label_col = cols_lower[k.lower()]
            break
    if label_col is None:
        raise ValueError(
            "Pickle DataFrame missing dili_binary / DILI / label column "
            f"(searched {_LABEL_KEYS}, got columns {df.columns.tolist()})"
        )
    dili_binary = _coerce_dili_binary(df[label_col].to_numpy(), n)

    # ------- expressions: any of `expression_\d+`, `gene_\d+`, or pure-int names -------
    expr_pat = re.compile(r"^(?:expression_|gene_)?(\d+)$")
    expr_pairs: list[tuple[int, str]] = []
    for c in df.columns:
        m = expr_pat.match(str(c))
        if m:
            expr_pairs.append((int(m.group(1)), c))
    if len(expr_pairs) != 978:
        raise ValueError(
            "Pickle DataFrame missing expressions columns "
            f"(expected 978 matching `^(expression_|gene_)?\\d+$`; "
            f"found {len(expr_pairs)}; columns sample={df.columns.tolist()[:8]})"
        )
    expr_pairs.sort(key=lambda p: p[0])
    expressions = _coerce_expressions(
        df[[c for _, c in expr_pairs]].to_numpy(), n
    )

    # ------- split_flags: 50 columns matching `split_\d+` -------
    split_pat = re.compile(r"^split_(\d+)$")
    split_pairs: list[tuple[int, str]] = []
    for c in df.columns:
        m = split_pat.match(str(c))
        if m:
            split_pairs.append((int(m.group(1)), c))
    if len(split_pairs) != 50:
        raise ValueError(
            "Pickle DataFrame missing split_flags columns "
            f"(expected 50 matching `^split_\\d+$`; found {len(split_pairs)})"
        )
    split_pairs.sort(key=lambda p: p[0])
    split_flags = _coerce_split_flags(
        df[[c for _, c in split_pairs]].to_numpy(), n
    )

    # ------- inst_ids (optional) -------
    inst_col: Optional[str] = None
    for k in _INST_KEYS:
        if k.lower() in cols_lower:
            inst_col = cols_lower[k.lower()]
            break
    if inst_col is not None:
        inst_ids: Optional[list[str]] = [str(x) for x in df[inst_col].tolist()]
    else:
        inst_ids = None

    return WangliDataset(
        compound_names=compound_names,
        dili_binary=dili_binary,
        expressions=expressions,
        split_flags=split_flags,
        inst_ids=inst_ids,
    )


def _parse_list_of_dicts(rows: list) -> WangliDataset:
    """Adaptive parse for list-of-per-profile-dicts layout.

    Each row is expected to expose:
        compound_name, DILI/label/dili_binary, expressions (978-vec), split_flags (50-vec),
        and optionally inst_id.
    """
    log.debug("load_split_pickle: list-of-dicts layout, n=%d", len(rows))
    if not rows or not isinstance(rows[0], dict):
        raise ValueError(
            "Pickle is a non-empty list but rows are not dicts; "
            f"first row type={type(rows[0]).__name__ if rows else 'empty'}"
        )

    # Sniff the field names from the first row using the same alias tables.
    first = rows[0]
    cname_key = _first_present(first, _COMPOUND_KEYS)
    if cname_key is None:
        raise ValueError(
            "List-of-dicts pickle: rows missing compound name field "
            f"(searched {_COMPOUND_KEYS}, got keys {sorted(first.keys())})"
        )
    label_key = _first_present(first, _LABEL_KEYS)
    if label_key is None:
        raise ValueError(
            "List-of-dicts pickle: rows missing dili_binary / DILI / label field "
            f"(searched {_LABEL_KEYS}, got keys {sorted(first.keys())})"
        )
    expr_key = _first_present(first, _EXPR_KEYS)
    if expr_key is None:
        raise ValueError(
            "List-of-dicts pickle: rows missing expressions field "
            f"(searched {_EXPR_KEYS}, got keys {sorted(first.keys())})"
        )
    split_key = _first_present(first, _SPLIT_KEYS)
    if split_key is None:
        raise ValueError(
            "List-of-dicts pickle: rows missing split_flags field "
            f"(searched {_SPLIT_KEYS}, got keys {sorted(first.keys())})"
        )
    inst_key = _first_present(first, _INST_KEYS)

    n = len(rows)
    compound_names = [str(r[cname_key]) for r in rows]
    dili_binary = _coerce_dili_binary(
        np.asarray([r[label_key] for r in rows]), n
    )
    expressions = _coerce_expressions(
        np.stack([np.asarray(r[expr_key]) for r in rows]), n
    )
    split_flags = _coerce_split_flags(
        np.stack([np.asarray(r[split_key]) for r in rows]), n
    )
    inst_ids = (
        [str(r[inst_key]) for r in rows] if inst_key is not None else None
    )

    return WangliDataset(
        compound_names=compound_names,
        dili_binary=dili_binary,
        expressions=expressions,
        split_flags=split_flags,
        inst_ids=inst_ids,
    )


def load_split_pickle(pickle_path: str | Path) -> WangliDataset:
    """Load Wang/Li's drug-split pickle (Synapse syn22910750) → WangliDataset.

    Adaptive parser: dispatches on the unpickled object's type. The L1000_DILI
    README does not formally document the pickle layout, so we sniff at runtime.

    Supported layouts:
        - dict-of-arrays (preferred): keys among `compound_names`/`compound_name`,
          `dili_binary`/`DILI`/`label`, `expressions`/`gene_expression`,
          `split_flags` or 50 `split_NN` keys, optional `inst_ids`.
        - DataFrame-of-rows: columns `compound_name`, `DILI`/`label`,
          978 expression columns matching `^(expression_|gene_)?\\d+$`,
          50 columns matching `^split_\\d+$`, optional `inst_id`.
        - list-of-dicts: each row a dict with the field aliases above.

    Parameters
    ----------
    pickle_path : path-like

    Returns
    -------
    WangliDataset

    Raises
    ------
    ValueError
        On unrecognized layout, missing required fields, or shape/dtype
        violations of the locked schema.
    """
    with open(Path(pickle_path), "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        return _parse_dict(obj)
    if isinstance(obj, pd.DataFrame):
        return _parse_dataframe(obj)
    if isinstance(obj, list):
        return _parse_list_of_dicts(obj)

    raise ValueError(
        "Unrecognized Wang/Li drug-split pickle structure: "
        f"type={type(obj).__name__} (expected dict, DataFrame, or list of dicts)"
    )
