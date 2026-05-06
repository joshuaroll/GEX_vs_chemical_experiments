#!/usr/bin/env python3
"""Build data/processed/dili_canonical.csv from Plan 01 outputs + DILIrank + LINCS + PDG.

Usage:  python scripts/build_dili_canonical.py

Inputs (paths are relative to the dili_downstream repo root, except the LINCS / PDG
sources which live in the MultiDCP repo):
    data/raw/DILIst/dilist.xlsx
    data/raw/DILIrank/dilirank.xlsx               (sheet_name='version 2', header=1)
    data/processed/dilist_smiles_resolved.csv
    /raid/home/joshua/projects/MultiDCP/MultiDCP/data/pert_transcriptom/level3/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt
    /raid/home/joshua/projects/MultiDCP/MultiDCP/data/pdg_brddrugfiltered.pkl
    (optional fast-path) /raid/home/joshua/projects/MultiDCP/MultiDCP/data/all_drugs_pdg.csv

Output:
    data/processed/dili_canonical.csv  (9 columns, locked order)

Invariant assertions are run after the build; the script exits non-zero if any fail.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

# Make the repo importable regardless of where this is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.build_dili_canonical import CANONICAL_COLUMNS, build_canonical  # noqa: E402

# Source-of-truth paths (locked in MANIFEST.md / 01-CONTEXT.md)
DILIST_XLSX = REPO_ROOT / "data" / "raw" / "DILIst" / "dilist.xlsx"
DILIRANK_XLSX = REPO_ROOT / "data" / "raw" / "DILIrank" / "dilirank.xlsx"
RESOLVED_CSV = REPO_ROOT / "data" / "processed" / "dilist_smiles_resolved.csv"
LINCS_SIG_INFO = Path(
    "/raid/home/joshua/projects/MultiDCP/MultiDCP/data/pert_transcriptom/"
    "level3/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt"
)
PDG_PKL = Path(
    "/raid/home/joshua/projects/MultiDCP/MultiDCP/data/pdg_brddrugfiltered.pkl"
)
PDG_CSV_FASTPATH = Path(
    "/raid/home/joshua/projects/MultiDCP/MultiDCP/data/all_drugs_pdg.csv"
)
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "dili_canonical.csv"


def load_lincs_inames_lower(sig_info_path: Path) -> set[str]:
    """Return the set of unique lowercased `pert_iname` values from the LINCS sig_info TSV.

    `pert_iname` is the human-readable drug name (1,826 unique in GSE70138). This is
    the canonical join key for the `in_lincs` flag — `pert_id` in this file is a
    BRD compound ID, which doesn't match DILIst CompoundName.
    """
    if not sig_info_path.exists():
        raise FileNotFoundError(
            f"LINCS sig_info TSV missing: {sig_info_path}. "
            "This is the source-of-truth for `in_lincs` per 01-CONTEXT.md."
        )
    df = pd.read_csv(sig_info_path, sep="\t", usecols=["pert_iname"])
    inames = set(df["pert_iname"].astype(str).str.strip().str.lower().unique())
    inames.discard("")
    inames.discard("nan")
    return inames


def load_pdg_inames_lower(pkl_path: Path, csv_fastpath: Path) -> set[str]:
    """Return the set of lowercased PDG drug names.

    Prefer the fast CSV (`all_drugs_pdg.csv` has clean `drug_name`/`cpd_smiles`
    columns); fall back to the pickle if the CSV is unavailable. The pickle's
    `pert_id` column actually holds drug names (e.g. "flutamide"), not BRD IDs.
    """
    if csv_fastpath.exists():
        df = pd.read_csv(csv_fastpath, usecols=["drug_name"])
        names = set(df["drug_name"].astype(str).str.strip().str.lower().unique())
    else:
        with pkl_path.open("rb") as fh:
            obj = pickle.load(fh)
        if hasattr(obj, "columns") and "pert_id" in obj.columns:
            names = set(obj["pert_id"].astype(str).str.strip().str.lower().unique())
        elif isinstance(obj, dict):
            # Walk dict-of-DataFrames; collect any pert_id columns.
            names = set()
            for v in obj.values():
                if hasattr(v, "columns") and "pert_id" in v.columns:
                    names.update(v["pert_id"].astype(str).str.strip().str.lower().unique())
            if not names:
                raise RuntimeError(
                    f"Could not locate a 'pert_id' column anywhere in {pkl_path}"
                )
        else:
            raise RuntimeError(f"Unrecognized PDG pickle structure: {type(obj)}")
    names.discard("")
    names.discard("nan")
    return names


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("build_dili_canonical")

    log.info("Loading DILIst from %s", DILIST_XLSX)
    dilist_df = pd.read_excel(DILIST_XLSX)
    log.info("DILIst: %d rows, columns=%s", len(dilist_df), dilist_df.columns.tolist())

    log.info("Loading DILIrank 2.0 from %s (sheet_name='version 2', header=1)", DILIRANK_XLSX)
    dilirank_df = pd.read_excel(DILIRANK_XLSX, sheet_name="version 2", header=1)
    log.info("DILIrank: %d rows, columns=%s", len(dilirank_df), dilirank_df.columns.tolist())

    log.info("Loading resolved-SMILES table from %s", RESOLVED_CSV)
    resolved_df = pd.read_csv(RESOLVED_CSV)
    log.info("Resolved SMILES: %d rows", len(resolved_df))

    log.info("Loading LINCS pert_inames from %s", LINCS_SIG_INFO)
    lincs_inames_lower = load_lincs_inames_lower(LINCS_SIG_INFO)
    log.info("LINCS unique pert_inames: %d", len(lincs_inames_lower))

    log.info("Loading PDG drug names (fastpath %s, fallback %s)", PDG_CSV_FASTPATH, PDG_PKL)
    pdg_inames_lower = load_pdg_inames_lower(PDG_PKL, PDG_CSV_FASTPATH)
    log.info("PDG unique drug names: %d", len(pdg_inames_lower))

    log.info("Building canonical 9-column table")
    df = build_canonical(
        dilist_df=dilist_df,
        dilirank_df=dilirank_df,
        resolved_df=resolved_df,
        lincs_inames_lower=lincs_inames_lower,
        pdg_inames_lower=pdg_inames_lower,
    )

    # ------------------------------------------------------------------
    # Invariant assertions (fail loud)
    # ------------------------------------------------------------------
    assert df.columns.tolist() == CANONICAL_COLUMNS, (
        f"Wrong columns: {df.columns.tolist()} (expected {CANONICAL_COLUMNS})"
    )
    for col in ("pert_id", "drug_name", "dili_binary", "canonical_smiles"):
        assert df[col].notna().all(), f"NaN in non-nullable column: {col}"
    assert df["pert_id"].is_unique, "pert_id is not unique"
    assert df["dili_binary"].isin([0, 1]).all(), "dili_binary has non-binary values"
    assert df["in_lincs"].dtype == bool, f"in_lincs not bool: {df['in_lincs'].dtype}"
    assert df["in_pdg"].dtype == bool, f"in_pdg not bool: {df['in_pdg'].dtype}"

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    log.info("Wrote %s (%d rows)", OUTPUT_CSV, len(df))

    # ------------------------------------------------------------------
    # Stats summary (printed for the commit log)
    # ------------------------------------------------------------------
    class_balance = df["dili_binary"].value_counts().to_dict()
    severity_pop = int(df["dili_severity"].notna().sum())
    n_lincs = int(df["in_lincs"].sum())
    n_pdg = int(df["in_pdg"].sum())
    print(
        f"OK rows={len(df)} "
        f"class_balance={class_balance} "
        f"severity_populated={severity_pop} "
        f"in_lincs={n_lincs} "
        f"in_pdg={n_pdg}"
    )


if __name__ == "__main__":
    main()
