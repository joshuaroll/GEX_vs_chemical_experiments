"""Render `results/tables/P1_data_summary.md` from `data/processed/dili_canonical.csv`.

Pure library. The CLI driver (`scripts/summarize_phase1.py`) handles file I/O and the
DILIst-total lookup; here we only depend on a canonical DataFrame and an explicit
`dilist_total` integer (so unit tests don't need to touch xlsx).

Required markdown sections (locked — used as grep-checks in 01-02-PLAN.md):
    # Phase 1 Data Summary
    ## Class balance
    ## SMILES resolution rate
    ## DILIrank severity populated
    ## D_DILI ∩ LINCS
    ## D_DILI ∩ PDG
"""

from __future__ import annotations

import pandas as pd


def summary_stats(canonical_df: pd.DataFrame, dilist_total: int) -> dict:
    """Compute the seven Phase 1 summary statistics.

    Parameters
    ----------
    canonical_df : the 9-column dili_canonical.csv as a DataFrame.
    dilist_total : total number of DILIst rows (the denominator for the SMILES
        resolution rate). Pass the row count of `data/raw/DILIst/dilist.xlsx`.

    Returns
    -------
    dict with keys:
        n_canonical : int
        dilist_total : int
        class_balance : {'positive': int, 'negative': int, 'positive_frac': float}
        smiles_resolution_rate : float  (n_canonical / dilist_total)
        severity_populated : {'count': int, 'frac': float}  (frac of n_canonical)
        intersect_lincs : {'count': int, 'frac': float}     (frac of n_canonical)
        intersect_pdg   : {'count': int, 'frac': float}     (frac of n_canonical)
    """
    n = len(canonical_df)
    pos = int((canonical_df["dili_binary"] == 1).sum())
    neg = int((canonical_df["dili_binary"] == 0).sum())
    pos_frac = pos / n if n else 0.0

    sev_count = int(canonical_df["dili_severity"].notna().sum())
    lincs_count = int(canonical_df["in_lincs"].sum())
    pdg_count = int(canonical_df["in_pdg"].sum())

    return {
        "n_canonical": n,
        "dilist_total": int(dilist_total),
        "class_balance": {
            "positive": pos,
            "negative": neg,
            "positive_frac": pos_frac,
        },
        "smiles_resolution_rate": (n / dilist_total) if dilist_total else 0.0,
        "severity_populated": {
            "count": sev_count,
            "frac": (sev_count / n) if n else 0.0,
        },
        "intersect_lincs": {
            "count": lincs_count,
            "frac": (lincs_count / n) if n else 0.0,
        },
        "intersect_pdg": {
            "count": pdg_count,
            "frac": (pdg_count / n) if n else 0.0,
        },
    }


def render_markdown(stats: dict) -> str:
    """Render `summary_stats(...)` output as the Phase 1 summary markdown.

    Section headers are LOCKED — they're grep-checked in the plan's verify block.
    """
    cb = stats["class_balance"]
    sev = stats["severity_populated"]
    lincs = stats["intersect_lincs"]
    pdg = stats["intersect_pdg"]
    n = stats["n_canonical"]
    dilist_total = stats["dilist_total"]
    res_rate = stats["smiles_resolution_rate"]

    lines = [
        "# Phase 1 Data Summary",
        "",
        f"Generated from `data/processed/dili_canonical.csv` ({n:,} rows).",
        "",
        "## Class balance",
        "",
        f"DILI-concern (`dili_binary == 1`): **{cb['positive']:,}** "
        f"({cb['positive_frac']:.3%})",
        "",
        f"No DILI-concern (`dili_binary == 0`): **{cb['negative']:,}** "
        f"({1 - cb['positive_frac']:.3%})",
        "",
        f"Total: **{n:,}** small-molecule rows.",
        "",
        "## SMILES resolution rate",
        "",
        f"**{n:,} / {dilist_total:,} = {res_rate:.3%}** of DILIst rows resolved "
        f"to canonical SMILES (per Q13 small-molecule scoping).",
        "",
        "The 161 unresolved entries (biologics, polymers/inorganics, obsolete or "
        "industrial chemicals) are logged in "
        "`data/processed/dili_smiles_resolution_failures.csv` and excluded from "
        "`dili_canonical.csv`.",
        "",
        "## DILIrank severity populated",
        "",
        f"**{sev['count']:,} / {n:,} = {sev['frac']:.3%}** of canonical rows have "
        "a `dili_severity` label from DILIrank 2.0 (`vDILI-Concern`).",
        "",
        "Severity is `NaN` for drugs that aren't in DILIrank 2.0; matching uses "
        "lowercased compound name with salt-suffix stripping (mirrors Plan 01's "
        "SMILES resolver tactic so 'abacavir' [DILIst] hits 'abacavir sulfate' "
        "[DILIrank]).",
        "",
        "## D_DILI ∩ LINCS",
        "",
        f"**{lincs['count']:,} / {n:,} = {lincs['frac']:.3%}** of canonical rows "
        "have a drug name (lowercased) present in the LINCS L1000 Phase II "
        "`pert_iname` set "
        "(`GSE70138_Broad_LINCS_sig_info_2017-03-06.txt`, 1,826 unique drugs).",
        "",
        "## D_DILI ∩ PDG",
        "",
        f"**{pdg['count']:,} / {n:,} = {pdg['frac']:.3%}** of canonical rows have "
        "a drug name (lowercased) present in the PDG perturbation set "
        "(`MultiDCP/data/all_drugs_pdg.csv` ≅ `pdg_brddrugfiltered.pkl` "
        "pert_id column).",
        "",
    ]
    return "\n".join(lines)
