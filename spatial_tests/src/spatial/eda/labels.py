"""Pure library: load DILI label files and apply Wang/Li binary encoding.

Phase deliverable: shared label-loading utilities consumed by the floor
(plan 03) and ceiling label-alignment (plan 04) pipelines.

Policy (locked):
    - DILIrank: header=1 (skip FDA title row); vDILI-Concern column is
      case-normalized before matching; Ambiguous-DILI-concern rows excluded;
      vMost + vMOST + vLess -> 1; vNo -> 0 (D-04).
    - DILIst: trailing space on "DILIst Classification " stripped via
      df.columns.str.strip() (Pitfall 3).
    - Warn via log.warning when excluded (Ambiguous) fraction > 0.30.
    - name_lower added for join key consistency.

Hard rules honored:
    - Pure library: NO hardcoded absolute paths, NO real-data filenames.
    - No mock or synthetic labels — the caller provides a real path.
    - pd.read_excel(path, ...) is called so that tests can monkeypatch it.
"""

from __future__ import annotations

import logging
from typing import Final, NamedTuple

import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["load_dilirank", "load_dilist", "LabelTable"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: vDILI-Concern values that map to binary 1 (positive, hepatotoxic)
_POS_VALUES: Final[frozenset[str]] = frozenset(
    {"vmost-dili-concern", "vless-dili-concern"}
)

#: vDILI-Concern values that map to binary 0 (negative, non-hepatotoxic)
_NEG_VALUES: Final[frozenset[str]] = frozenset({"vno-dili-concern"})

#: Excluded-fraction threshold above which a warning is emitted.
_HIGH_EXCLUDED_THRESHOLD: Final[float] = 0.30

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class LabelTable(NamedTuple):
    """Binary DILI label table with encoding metadata.

    Attributes
    ----------
    df : pd.DataFrame
        Filtered and encoded compound rows (Ambiguous excluded).
    n_total : int
        Rows in the raw file before filtering.
    n_positive : int
        Rows encoded as 1 (hepatotoxic).
    n_negative : int
        Rows encoded as 0 (non-hepatotoxic).
    n_excluded : int
        Rows excluded (Ambiguous or unknown concern level).
    source : str
        Source label for provenance ('dilirank' or 'dilist').
    """

    df: object          # pd.DataFrame; typed as object for NamedTuple compat
    n_total: int
    n_positive: int
    n_negative: int
    n_excluded: int
    source: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dilirank(path: str) -> pd.DataFrame:
    """Load DILIrank.xlsx and return a binary-encoded compound DataFrame.

    Reads the Excel file with header=1 (skipping the FDA title row), normalizes
    the vDILI-Concern column (lowercase + strip), excludes Ambiguous rows, and
    adds a binary dili_binary column per the Wang/Li convention (D-04).

    Parameters
    ----------
    path : str
        Filesystem path to dilirank.xlsx (e.g. "data/raw/labels/dilirank/dilirank.xlsx").

    Returns
    -------
    pd.DataFrame
        Columns: LTKBID, CompoundName, name_lower, dili_binary, SeverityClass.
        Ambiguous-DILI-concern rows are excluded. n rows = n_positive + n_negative.

    Notes
    -----
    Pitfall 1: Must use header=1; row 0 is the FDA title row, not column names.
    Pitfall 2: Case variation ("vMOST" vs "vMost") normalized by .str.lower().
    """
    df = pd.read_excel(path, header=1)

    n_total = len(df)

    # Normalize case variants (Pitfall 2: vMOST vs vMost)
    concern = df["vDILI-Concern"].str.lower().str.strip()

    pos_mask = concern.isin(_POS_VALUES)
    neg_mask = concern.isin(_NEG_VALUES)
    keep_mask = pos_mask | neg_mask

    n_positive = int(pos_mask[keep_mask].sum())
    n_negative = int(neg_mask[keep_mask].sum())
    n_excluded = n_total - keep_mask.sum()

    # Warn if excluded fraction is unexpectedly high (mirrors orthology pattern)
    excluded_fraction = n_excluded / max(n_total, 1)
    if excluded_fraction > _HIGH_EXCLUDED_THRESHOLD:
        log.warning(
            "load_dilirank: high excluded fraction %.1f%% "
            "(%d total, %d excluded as Ambiguous/unknown). "
            "Expected ~26%% for DILIrank 2.0.",
            100.0 * excluded_fraction,
            n_total,
            n_excluded,
        )
    else:
        log.debug(
            "load_dilirank: %d total -> %d positive, %d negative, %d excluded (%.1f%%)",
            n_total,
            n_positive,
            n_negative,
            n_excluded,
            100.0 * excluded_fraction,
        )

    df = df[keep_mask].copy()
    df["dili_binary"] = pos_mask[keep_mask].astype(int).values
    df["name_lower"] = df["CompoundName"].str.lower().str.strip()

    return df[["LTKBID", "CompoundName", "name_lower", "dili_binary", "SeverityClass"]]


def load_dilist(path: str) -> pd.DataFrame:
    """Load DILIst.xlsx and return a binary-encoded compound DataFrame.

    Strips trailing whitespace from all column names before accessing
    "DILIst Classification" (Pitfall 3: the on-disk column has a trailing space).

    Parameters
    ----------
    path : str
        Filesystem path to dilist.xlsx (e.g. "data/raw/labels/dilist/dilist.xlsx").

    Returns
    -------
    pd.DataFrame
        Columns: name_lower, dili_binary.
    """
    df = pd.read_excel(path)
    # Pitfall 3: strip trailing space on column names
    df.columns = df.columns.str.strip()

    df["dili_binary"] = df["DILIst Classification"].astype(int)
    # On-disk column is "CompoundName" (not "Compound"); both variants tolerated.
    name_col = "CompoundName" if "CompoundName" in df.columns else "Compound"
    df["name_lower"] = df[name_col].str.lower().str.strip()

    return df[["name_lower", "dili_binary"]]
