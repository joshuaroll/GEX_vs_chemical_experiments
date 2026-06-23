"""Pure library: join compound names to canonical SMILES via a 3-layer cascade.

Phase deliverable: SMILES join utility consumed by the floor pipeline
(plan 03) to provide chemical structures for ECFP4 fingerprinting (D-03).

Join cascade policy (locked):
    1. Layer 1 — dili_canonical.csv (canonical_smiles, join key: drug_name
       normalized to name_lower). Covers ~49.5% of DILIrank.
    2. Layer 2 — drugbank_smiles_index.csv (name_lower key; fills residual NaN).
    3. Layer 3 — TDC fallback (callable | None, default None); only invoked
       if residual gap > 10% of input rows. Does NOT hard-require pytdc —
       caller provides the callable.

Coverage reporting:
    - covered fraction = smiles.notna().mean() over all input rows.
    - log.warning if covered fraction < 0.40 (acquisition risk flag from
      RESEARCH.md Security Domain and open question 2).

Hard rules honored:
    - Pure library: NO hardcoded absolute paths, NO real-data filenames.
    - No mock or synthetic labels — caller provides real CSV paths.
    - pd.read_csv(path, ...) is called so that tests can patch paths via
      real temp files (not monkeypatched).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["join_smiles_cascade"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Coverage threshold below which a warning is emitted.
_LOW_COVERAGE_THRESHOLD: float = 0.40

#: Gap threshold above which the TDC fallback is triggered (if provided).
_TDC_FALLBACK_GAP_THRESHOLD: float = 0.10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def join_smiles_cascade(
    labels_df: pd.DataFrame,
    dili_canonical_path: str,
    drugbank_path: str,
    *,
    tdc_fallback: Optional[Callable[[list[str]], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Join compound name_lower keys to canonical SMILES in priority-cascade order.

    Layer 1: dili_canonical.csv (canonical_smiles column, join key: name_lower
    derived from drug_name). Layer 2: drugbank_smiles_index.csv (name_lower is
    already the join key). Layer 3: optional tdc_fallback callable, invoked only
    when residual gap > 10% of input rows.

    Parameters
    ----------
    labels_df : pd.DataFrame
        Must contain a "name_lower" column (lowercase, stripped compound names).
    dili_canonical_path : str
        Path to dili_canonical.csv (columns: drug_name, canonical_smiles, ...).
    drugbank_path : str
        Path to drugbank_smiles_index.csv (columns: name_lower, name, smiles).
    tdc_fallback : callable | None
        Optional layer-3 callable. Signature: f(name_lower_list: list[str]) ->
        pd.DataFrame with columns [name_lower, smiles]. Called only if still-
        missing fraction > 0.10 after layers 1+2. Default None (no network call).

    Returns
    -------
    pd.DataFrame
        Input labels_df with an added "smiles" column. Rows without a SMILES
        match have smiles=NaN. Row count and order are preserved.

    Notes
    -----
    Coverage flag: log.warning when covered fraction < 0.40 (acquisition risk).
    """
    if "name_lower" not in labels_df.columns:
        raise ValueError(
            "join_smiles_cascade: labels_df must contain a 'name_lower' column. "
            f"Got columns: {list(labels_df.columns)!r}."
        )

    # ------------------------------------------------------------------
    # Layer 1: dili_canonical (canonical_smiles, join key: drug_name -> name_lower)
    # ------------------------------------------------------------------
    c1 = pd.read_csv(dili_canonical_path)
    c1["name_lower"] = c1["drug_name"].str.lower().str.strip()
    merged = labels_df.merge(
        c1[["name_lower", "canonical_smiles"]].rename(
            columns={"canonical_smiles": "smiles"}
        ),
        on="name_lower",
        how="left",
    )

    # ------------------------------------------------------------------
    # Layer 2: drugbank_smiles_index (name_lower already the join key)
    # ------------------------------------------------------------------
    missing_mask = merged["smiles"].isna()
    if missing_mask.any():
        c2 = pd.read_csv(drugbank_path)  # cols: name_lower, name, smiles
        residual = merged[missing_mask][["name_lower"]].merge(
            c2[["name_lower", "smiles"]], on="name_lower", how="left"
        )
        merged.loc[missing_mask, "smiles"] = residual["smiles"].values

    # ------------------------------------------------------------------
    # Layer 3: TDC fallback (only if provided and residual gap > 10%)
    # ------------------------------------------------------------------
    still_missing = merged["smiles"].isna()
    still_missing_fraction = still_missing.sum() / max(len(merged), 1)
    if tdc_fallback is not None and still_missing_fraction > _TDC_FALLBACK_GAP_THRESHOLD:
        missing_names = merged.loc[still_missing, "name_lower"].tolist()
        log.debug(
            "join_smiles_cascade: invoking TDC fallback for %d still-missing names "
            "(gap=%.1f%% > %.0f%% threshold).",
            len(missing_names),
            100.0 * still_missing_fraction,
            100.0 * _TDC_FALLBACK_GAP_THRESHOLD,
        )
        tdc_result = tdc_fallback(missing_names)
        if "name_lower" in tdc_result.columns and "smiles" in tdc_result.columns:
            tdc_map = tdc_result.set_index("name_lower")["smiles"]
            for idx in merged.index[still_missing]:
                name = merged.at[idx, "name_lower"]
                if name in tdc_map:
                    merged.at[idx, "smiles"] = tdc_map[name]

    # ------------------------------------------------------------------
    # Coverage reporting (gene_alignment.py:231-240 style)
    # ------------------------------------------------------------------
    covered = merged["smiles"].notna().mean()
    n_covered = merged["smiles"].notna().sum()
    n_input = len(merged)

    if covered < _LOW_COVERAGE_THRESHOLD:
        log.warning(
            "join_smiles_cascade: low SMILES coverage %.1f%% (%d / %d input rows). "
            "Floor classifier will run on the covered subset only. "
            "Consider running the TDC fallback to improve coverage.",
            100.0 * covered,
            n_covered,
            n_input,
        )
    else:
        log.info(
            "join_smiles_cascade: SMILES coverage %.1f%% (%d / %d input rows).",
            100.0 * covered,
            n_covered,
            n_input,
        )

    return merged
