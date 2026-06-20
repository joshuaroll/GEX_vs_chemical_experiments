"""Pure library: build a human-mouse-rat one-to-one ortholog map.

Phase deliverable: cross-species ortholog table used by the gene-alignment
seam (src/spatial/gene_alignment.py) to restrict multi-species comparisons
to unambiguous one-to-one gene pairs.

Policy (locked):
    - Input: a cached Ensembl BioMart TSV (8 columns, produced by
      scripts/build_orthologs.py) or an already-parsed pandas DataFrame.
    - Output: an OrthologTable with a `pairs` DataFrame (one row per kept
      human gene) and a dropped_fraction metric (XC-08).
    - Only rows where BOTH mmusculus_homolog_orthology_type AND
      rnorvegicus_homolog_orthology_type equal "ortholog_one2one" are kept.
    - Many-to-many orthologs are DROPPED (not collapsed).
    - Rows with empty mouse or rat Ensembl IDs are dropped.
    - The dropped_fraction is always reported; a warning is emitted when the
      fraction is unexpectedly high (>= 0.80).

Hard rules honored:
    - Pure library: NO hardcoded absolute paths, NO real-data filenames.
    - No mock or synthetic labels — the caller provides a real cached TSV or
      an already-fetched DataFrame.
    - Network fetch lives behind a cached-TSV boundary; scripts/build_orthologs.py
      does the GET; this module only reads and filters.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Final, NamedTuple, Union

import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["build_one2one_orthologs", "ortholog_report", "OrthologTable"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Column names in the BioMart 8-column TSV produced by scripts/build_orthologs.py
_BIOMART_COLS: Final[list[str]] = [
    "ensembl_gene_id",
    "external_gene_name",
    "mmusculus_homolog_ensembl_gene",
    "mmusculus_homolog_associated_gene_name",
    "mmusculus_homolog_orthology_type",
    "rnorvegicus_homolog_ensembl_gene",
    "rnorvegicus_homolog_associated_gene_name",
    "rnorvegicus_homolog_orthology_type",
]

#: Threshold above which the dropped fraction triggers a warning.
_HIGH_DROPPED_THRESHOLD: Final[float] = 0.80

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class OrthologTable(NamedTuple):
    """Human-mouse-rat one-to-one ortholog map.

    Produced by ``build_one2one_orthologs``; consumed by the gene-alignment
    seam and the MANIFEST / P0_orthologs.md report.

    Attributes
    ----------
    pairs : pd.DataFrame
        Columns: human_ensembl, human_symbol, mouse_ensembl, mouse_symbol,
        rat_ensembl, rat_symbol.  One row per human gene kept (one2one in
        BOTH mouse AND rat).  Index is reset (integer, starting at 0).
    n_input : int
        Rows in the raw BioMart TSV (or DataFrame) before filtering.  This
        is the denominator for ``dropped_fraction``.
    n_one2one : int
        Rows kept after the mutual one2one filter and ID-completeness filter.
        Must equal ``len(pairs)``.
    dropped_fraction : float
        ``(n_input - n_one2one) / n_input``.  Reported per XC-08 (cross-
        species ortholog discipline).  Value in [0.0, 1.0].
    """

    pairs: object          # pd.DataFrame; typed as object for NamedTuple compat
    n_input: int
    n_one2one: int
    dropped_fraction: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_one2one_orthologs(
    source: Union[pd.DataFrame, pathlib.Path, str],
) -> OrthologTable:
    """Filter a BioMart TSV (or DataFrame) to mutual human-mouse-rat one-to-one orthologs.

    Only rows where BOTH species carry ``orthology_type == "ortholog_one2one"``
    are retained.  Rows with empty mouse or rat Ensembl IDs are dropped first.
    Many-to-many orthologs are DROPPED, never collapsed (XC-08).

    Parameters
    ----------
    source : pd.DataFrame, pathlib.Path, or str
        Either an already-parsed BioMart DataFrame (8 columns matching
        ``_BIOMART_COLS``) or a filesystem path to a tab-separated BioMart
        TSV with the same 8 columns (with or without a header row).

    Returns
    -------
    OrthologTable
        Frozen NamedTuple with ``pairs`` (one row per kept human gene),
        ``n_input``, ``n_one2one``, and ``dropped_fraction``.

    Raises
    ------
    TypeError
        If ``source`` is not a DataFrame, Path, or str.
    ValueError
        If the loaded TSV/DataFrame does not have the expected 8 columns.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "ensembl_gene_id": ["ENSG001", "ENSG002"],
    ...     "external_gene_name": ["GENE1", "GENE2"],
    ...     "mmusculus_homolog_ensembl_gene": ["ENSMUSG001", "ENSMUSG002"],
    ...     "mmusculus_homolog_associated_gene_name": ["Gene1", "Gene2"],
    ...     "mmusculus_homolog_orthology_type": ["ortholog_one2one", "ortholog_one2many"],
    ...     "rnorvegicus_homolog_ensembl_gene": ["ENSRNOG001", "ENSRNOG002"],
    ...     "rnorvegicus_homolog_associated_gene_name": ["Gene1", "Gene2"],
    ...     "rnorvegicus_homolog_orthology_type": ["ortholog_one2one", "ortholog_one2one"],
    ... })
    >>> tbl = build_one2one_orthologs(df)
    >>> tbl.n_one2one
    1
    >>> tbl.dropped_fraction
    0.5
    """
    # ------------------------------------------------------------------
    # 1. Load DataFrame
    # ------------------------------------------------------------------
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, (pathlib.Path, str)):
        path = pathlib.Path(source)
        # Sniff header: if first field is "ensembl_gene_id" it has a header.
        df = pd.read_csv(
            path,
            sep="\t",
            header=0,
            names=None,
            dtype=str,
            na_values=[""],
            keep_default_na=True,
        )
        # After reading with header=0, check if column names match.
        if list(df.columns) != _BIOMART_COLS:
            # Try headerless read (8 raw columns).
            df = pd.read_csv(
                path,
                sep="\t",
                header=None,
                names=_BIOMART_COLS,
                dtype=str,
                na_values=[""],
                keep_default_na=True,
            )
    else:
        raise TypeError(
            f"build_one2one_orthologs: `source` must be a DataFrame, Path, or str; "
            f"got {type(source).__name__!r}."
        )

    # ------------------------------------------------------------------
    # 2. Validate expected columns are present
    # ------------------------------------------------------------------
    missing_cols = [c for c in _BIOMART_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"build_one2one_orthologs: input is missing expected columns: "
            f"{missing_cols!r}.  Got columns: {list(df.columns)!r}."
        )

    n_input = len(df)

    # ------------------------------------------------------------------
    # 3. Drop rows where mouse or rat Ensembl ID is empty/NaN
    # ------------------------------------------------------------------
    df = df.fillna("")
    id_present = (
        (df["mmusculus_homolog_ensembl_gene"].str.strip() != "")
        & (df["rnorvegicus_homolog_ensembl_gene"].str.strip() != "")
    )
    df = df[id_present]

    # ------------------------------------------------------------------
    # 4. Apply one2one filter (XC-08) — keep rows where BOTH types == one2one
    # ------------------------------------------------------------------
    mask = (
        (df["mmusculus_homolog_orthology_type"] == "ortholog_one2one")
        & (df["rnorvegicus_homolog_orthology_type"] == "ortholog_one2one")
    )
    kept = df[mask].copy()

    n_one2one = len(kept)
    dropped_fraction = 1.0 - n_one2one / max(n_input, 1)

    # ------------------------------------------------------------------
    # 5. Build the canonical pairs DataFrame
    # ------------------------------------------------------------------
    pairs = kept[
        [
            "ensembl_gene_id",
            "external_gene_name",
            "mmusculus_homolog_ensembl_gene",
            "mmusculus_homolog_associated_gene_name",
            "rnorvegicus_homolog_ensembl_gene",
            "rnorvegicus_homolog_associated_gene_name",
        ]
    ].rename(
        columns={
            "ensembl_gene_id": "human_ensembl",
            "external_gene_name": "human_symbol",
            "mmusculus_homolog_ensembl_gene": "mouse_ensembl",
            "mmusculus_homolog_associated_gene_name": "mouse_symbol",
            "rnorvegicus_homolog_ensembl_gene": "rat_ensembl",
            "rnorvegicus_homolog_associated_gene_name": "rat_symbol",
        }
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 6. Warn if dropped fraction is unexpectedly high (mirrors
    #    gene_alignment.py:231-240 warn-on-low idiom)
    # ------------------------------------------------------------------
    if dropped_fraction >= _HIGH_DROPPED_THRESHOLD:
        log.warning(
            "build_one2one_orthologs: high dropped fraction %.1f%% "
            "(%d input rows, %d one2one kept). "
            "Verify the BioMart TSV contains the expected orthology columns "
            "and that the Ensembl release is current.",
            100.0 * dropped_fraction,
            n_input,
            n_one2one,
        )
    else:
        log.debug(
            "build_one2one_orthologs: %d input rows → %d one2one kept "
            "(dropped %.1f%%)",
            n_input,
            n_one2one,
            100.0 * dropped_fraction,
        )

    return OrthologTable(
        pairs=pairs,
        n_input=n_input,
        n_one2one=n_one2one,
        dropped_fraction=dropped_fraction,
    )


def ortholog_report(table: OrthologTable) -> str:
    """Return a markdown summary table of the ortholog filtering result (XC-08).

    Used by Plan 04 to write P0_orthologs.md.  Also emits a log.warning when
    the dropped fraction is >= 80% (unexpectedly high).

    Parameters
    ----------
    table : OrthologTable
        Result from ``build_one2one_orthologs``.

    Returns
    -------
    str
        A markdown table string with three rows: n_input, n_one2one, and
        dropped_fraction (formatted as a percentage).

    Examples
    --------
    >>> import pandas as pd
    >>> pairs = pd.DataFrame(columns=["human_ensembl", "human_symbol",
    ...                               "mouse_ensembl", "mouse_symbol",
    ...                               "rat_ensembl", "rat_symbol"])
    >>> tbl = OrthologTable(pairs=pairs, n_input=100, n_one2one=45,
    ...                     dropped_fraction=0.55)
    >>> print(ortholog_report(tbl))  # doctest: +NORMALIZE_WHITESPACE
    | Metric | Value |
    |--------|-------|
    | n_input | 100 |
    | n_one2one | 45 |
    | dropped_fraction | 55.00% |
    """
    if table.dropped_fraction >= _HIGH_DROPPED_THRESHOLD:
        log.warning(
            "ortholog_report: dropped fraction is %.1f%% — "
            "unusually high; review filter settings or BioMart query.",
            100.0 * table.dropped_fraction,
        )

    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| n_input | {table.n_input} |",
        f"| n_one2one | {table.n_one2one} |",
        f"| dropped_fraction | {100.0 * table.dropped_fraction:.2f}% |",
    ]
    return "\n".join(lines)
