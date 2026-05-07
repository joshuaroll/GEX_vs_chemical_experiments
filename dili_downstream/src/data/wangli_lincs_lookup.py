"""Pure library: open the local LINCS Level-5 h5, look up inst_ids, run a
per-gene Pearson cross-validation against published Level-5 expressions.

Phase 1 / Plan 01-02 deliverable. The CLI driver in
`scripts/build_phase1_wangli.py` (Wave 2) wires this with `wangli_loader`
(Plan 01-01) and `wangli_smiles_resolver` (Plan 01-03) against the real
Wang/Li xlsx + pickle and the local 1.4 GB Level-5 h5.

CONTEXT.md decisions implemented here:
    - Use the local Bayesian Level-5 h5 as LINCS source. The "Bayesian_"
      prefix is unusual — Phase 1 documents whether values match published
      Level 5 numerically via a per-gene Pearson check on a 100-profile
      sample of `pickle_expression vs h5_extracted` (the pickle's published
      gene values are Wang/Li's source-of-truth Level 5).
    - Fall back to fresh GEO download only if cross-validation Pearson
      < 0.99 averaged or > 1% inst_ids miss. The fallback policy is the
      Wave-2 driver's call; this library only reports.

h5 schema (verified empirically):
    /colid (n_profiles,)         object/bytes-or-str — LINCS sig_id strings
    /rowid (n_genes,)            object/bytes-or-str — gene symbols
    /data  (n_profiles, n_genes) float32              — z-score matrix

Hard rules honored:
    - Pure library: NO source-of-truth paths, NO hardcoded absolute paths.
    - Caller passes the h5 path in; the function reads it via h5py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from scipy import stats

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_id_array(arr: np.ndarray) -> list[str]:
    """Decode an h5py-loaded id array (bytes or str) to a list[str].

    h5py defaults to bytes for fixed-length string datasets and to
    variable-length-utf8 for `h5py.string_dtype()`. We accept both.
    """
    if arr.dtype.kind == "S" or (arr.size > 0 and isinstance(arr[0], (bytes, np.bytes_))):
        return [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr]
    # Already string-like (object dtype with str entries, or h5py vlen-str)
    return [str(x) for x in arr]


# ---------------------------------------------------------------------------
# open_lincs_h5
# ---------------------------------------------------------------------------


def open_lincs_h5(h5_path: str | Path) -> h5py.File:
    """Open the LINCS Level-5 h5 read-only and validate schema.

    Validates that `/colid`, `/rowid`, `/data` datasets exist and that
    `/data.shape == (len(/colid), len(/rowid))`.

    The caller is responsible for closing the returned handle (we do not
    use a context manager so callers can pass the live handle around).

    Parameters
    ----------
    h5_path : path-like

    Returns
    -------
    h5py.File
        Read-only handle.

    Raises
    ------
    ValueError
        If any required dataset is missing, or shapes disagree.
    """
    f = h5py.File(str(h5_path), "r")
    try:
        keys = list(f.keys())
        for required in ("colid", "rowid", "data"):
            if required not in f:
                raise ValueError(
                    f"Missing required dataset {required!r} in {h5_path}; "
                    f"got top-level keys {keys}. Need colid/rowid/data."
                )
        n_profiles = f["colid"].shape[0]
        n_genes = f["rowid"].shape[0]
        data_shape = f["data"].shape
        if data_shape != (n_profiles, n_genes):
            raise ValueError(
                f"shape mismatch in {h5_path}: /data has shape {data_shape} "
                f"but /colid is ({n_profiles},) and /rowid is ({n_genes},). "
                f"Expected /data shape ({n_profiles}, {n_genes})."
            )
        log.info(
            "open_lincs_h5: opened %s with /data shape %s", h5_path, data_shape,
        )
    except Exception:
        f.close()
        raise
    return f


# ---------------------------------------------------------------------------
# lookup_inst_ids
# ---------------------------------------------------------------------------


def lookup_inst_ids(
    h5_path: str | Path,
    inst_ids: Sequence[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Look up `inst_ids` in the h5's /colid and pull their /data rows.

    Builds a single `colid_str -> row_index` dict by decoding /colid once,
    then indexes /data by sorted-row-index batches for sequential h5 access
    and re-permutes the result back into input order.

    Parameters
    ----------
    h5_path : path-like
    inst_ids : sequence of str
        Query inst_ids in the order the caller wants the matrix rows.

    Returns
    -------
    matrix : np.ndarray, dtype float32, shape (len(found_ids), n_genes)
        Rows of /data for the found inst_ids, aligned to input query order
        (skipping missing ids).
    found_ids : list[str]
        The subset of `inst_ids` that exist in /colid, in input order.
    missing_ids : list[str]
        The subset NOT found in /colid, in input order.
    """
    f = open_lincs_h5(h5_path)
    try:
        n_profiles = f["colid"].shape[0]
        n_genes = f["rowid"].shape[0]

        # Decode /colid once (~361k strings for the real file — modest).
        colid_strs = _decode_id_array(f["colid"][:])
        colid_to_row: dict[str, int] = {s: i for i, s in enumerate(colid_strs)}

        # Pass 1: split query into found / missing in input order.
        found_input_pos: list[int] = []  # positions within input order
        found_row_idx: list[int] = []    # /data row indices, parallel to above
        found_ids: list[str] = []
        missing_ids: list[str] = []
        for qid in inst_ids:
            row = colid_to_row.get(qid)
            if row is None:
                missing_ids.append(qid)
            else:
                found_input_pos.append(len(found_ids))
                found_row_idx.append(row)
                found_ids.append(qid)

        n_found = len(found_ids)
        log.info(
            "lookup_inst_ids: query=%d found=%d missing=%d (h5=%d profiles x %d genes)",
            len(inst_ids), n_found, len(missing_ids), n_profiles, n_genes,
        )
        if len(inst_ids) > 0 and len(missing_ids) / max(len(inst_ids), 1) > 0.01:
            log.warning(
                "lookup_inst_ids: missing rate %.2f%% exceeds 1%% threshold "
                "(missing=%d of %d) — Wave-2 driver should consider GEO-fallback.",
                100.0 * len(missing_ids) / len(inst_ids),
                len(missing_ids), len(inst_ids),
            )

        if n_found == 0:
            matrix = np.empty((0, n_genes), dtype=np.float32)
            return matrix, found_ids, missing_ids

        # Pass 2: fancy-indexing /data requires a sorted, deduplicated index
        # vector for h5py. Sort, fetch, then permute back to input order.
        sorted_order = np.argsort(np.asarray(found_row_idx, dtype=np.int64), kind="stable")
        sorted_rows = np.asarray(found_row_idx, dtype=np.int64)[sorted_order]
        # h5py supports list-of-indices fancy indexing on the leading axis.
        fetched = f["data"][sorted_rows.tolist(), :]
        # Build inverse permutation: row i of the output corresponds to
        # input position i, which had row found_row_idx[i] in /data, and that
        # row sits at position np.where(sorted_order == i)[0][0] in `fetched`.
        inverse = np.empty_like(sorted_order)
        inverse[sorted_order] = np.arange(n_found)
        matrix = np.asarray(fetched, dtype=np.float32)[inverse, :]

        return matrix, found_ids, missing_ids
    finally:
        f.close()


# ---------------------------------------------------------------------------
# crossvalidate_pearson
# ---------------------------------------------------------------------------


def crossvalidate_pearson(
    h5_extracted: np.ndarray,
    pickle_expressions: np.ndarray,
) -> np.ndarray:
    """Per-gene Pearson correlation between two (sample_n, gene_count) matrices.

    Used by the Wave-2 driver to verify the local Bayesian h5's Level-5
    z-scores match Wang/Li's pickle-published expressions — a numerical
    sanity check for the unusual "Bayesian_" prefix on the local file.

    Parameters
    ----------
    h5_extracted : np.ndarray, shape (sample_n, gene_count)
        Sample of /data rows pulled by `lookup_inst_ids`.
    pickle_expressions : np.ndarray, shape (sample_n, gene_count)
        Wang/Li pickle's `expressions` field for the same sample.

    Returns
    -------
    np.ndarray, dtype float64, shape (gene_count,)
        Pearson r per gene. NaN for genes where either input has zero
        variance (Pearson undefined).

    Raises
    ------
    ValueError
        If shapes disagree or inputs aren't 2D.
    """
    h5_arr = np.asarray(h5_extracted)
    pk_arr = np.asarray(pickle_expressions)
    if h5_arr.ndim != 2 or pk_arr.ndim != 2:
        raise ValueError(
            f"crossvalidate_pearson expects 2D arrays; got shapes "
            f"{h5_arr.shape} and {pk_arr.shape}"
        )
    if h5_arr.shape != pk_arr.shape:
        raise ValueError(
            f"crossvalidate_pearson shape mismatch: h5 {h5_arr.shape} vs "
            f"pickle {pk_arr.shape}"
        )

    n_samples, gene_count = h5_arr.shape
    out = np.empty(gene_count, dtype=np.float64)

    # Promote to float64 for numerical stability of the Pearson computation.
    h5_f64 = h5_arr.astype(np.float64, copy=False)
    pk_f64 = pk_arr.astype(np.float64, copy=False)

    for g in range(gene_count):
        x = h5_f64[:, g]
        y = pk_f64[:, g]
        # Zero-variance check first: if either column is constant, Pearson
        # is undefined → NaN. (scipy raises a warning + returns NaN itself,
        # but doing the check up-front keeps the test-suite output clean.)
        if x.std() == 0.0 or y.std() == 0.0:
            out[g] = np.nan
            continue
        r, _p = stats.pearsonr(x, y)
        out[g] = r

    return out
