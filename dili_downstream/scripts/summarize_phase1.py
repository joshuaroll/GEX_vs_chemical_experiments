#!/usr/bin/env python3
"""Render `results/tables/P1_data_summary.md` from `data/processed/dili_canonical.csv`.

Usage:  python scripts/summarize_phase1.py

Produces the Phase 1 deliverable markdown summary with these required sections:
    # Phase 1 Data Summary
    ## Class balance
    ## SMILES resolution rate
    ## DILIrank severity populated
    ## D_DILI ∩ LINCS
    ## D_DILI ∩ PDG
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.summarize_phase1 import render_markdown, summary_stats  # noqa: E402

CANONICAL_CSV = REPO_ROOT / "data" / "processed" / "dili_canonical.csv"
DILIST_XLSX = REPO_ROOT / "data" / "raw" / "DILIst" / "dilist.xlsx"
OUTPUT_MD = REPO_ROOT / "results" / "tables" / "P1_data_summary.md"


def main() -> None:
    if not CANONICAL_CSV.exists():
        raise FileNotFoundError(
            f"{CANONICAL_CSV} missing. Run `python scripts/build_dili_canonical.py` first."
        )

    canonical_df = pd.read_csv(CANONICAL_CSV)
    dilist_total = len(pd.read_excel(DILIST_XLSX))

    stats = summary_stats(canonical_df, dilist_total=dilist_total)
    md = render_markdown(stats)

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(md)

    # Print stats to stdout for the commit log.
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
