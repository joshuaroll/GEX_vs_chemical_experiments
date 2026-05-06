#!/usr/bin/env python3
"""Resolve SMILES for DILIst drugs via DrugBank XML index. Writes 3 CSVs.

Usage:
    python scripts/resolve_smiles.py [--rebuild-index]

Outputs (relative to dili_downstream repo root):
    data/processed/drugbank_smiles_index.csv
    data/processed/dilist_smiles_resolved.csv
    data/processed/dili_smiles_resolution_failures.csv

Behavior:
    - The DrugBank XML parse (~1.5 GB streaming) is cached to
      ``drugbank_smiles_index.csv``. Re-runs reuse the cache unless
      ``--rebuild-index`` is passed.
    - Exits non-zero with a clear message if resolution rate falls below 90%
      (the DATA-04 target). Does NOT auto-stub or fabricate.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.drugbank_smiles_index import build_index, write_index  # noqa: E402
from src.data.resolve_smiles import resolve_dilist  # noqa: E402

DRUGBANK_XML = Path(
    "/raid/home/joshua/projects/MultiDCP/MultiDCP/data/drugbank_data/full_database.xml"
)
DILIST_XLSX = REPO_ROOT / "data" / "raw" / "DILIst" / "dilist.xlsx"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
INDEX_CSV = PROCESSED_DIR / "drugbank_smiles_index.csv"
RESOLVED_CSV = PROCESSED_DIR / "dilist_smiles_resolved.csv"
FAILURES_CSV = PROCESSED_DIR / "dili_smiles_resolution_failures.csv"

RESOLUTION_RATE_GATE = 0.90  # DATA-04 hard gate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the DrugBank SMILES index from the XML even if a cached "
        "CSV exists (default: reuse cached index when available).",
    )
    return p.parse_args()


def load_or_build_index(rebuild: bool) -> pd.DataFrame:
    """Return the (name_lower, name, smiles) index, building it if needed."""
    if INDEX_CSV.exists() and not rebuild:
        print(f"[resolve_smiles] Reusing cached index: {INDEX_CSV}")
        df = pd.read_csv(INDEX_CSV)
        print(f"[resolve_smiles]   {len(df):,} drugs in cached index.")
        return df

    if not DRUGBANK_XML.exists():
        raise FileNotFoundError(
            f"DrugBank XML not found at {DRUGBANK_XML}. "
            "Verify MultiDCP/data/drugbank_data/full_database.xml is present."
        )

    print(f"[resolve_smiles] Building index from {DRUGBANK_XML} (~1.5 GB streaming parse)...")
    t0 = time.perf_counter()
    df = build_index(DRUGBANK_XML)
    elapsed = time.perf_counter() - t0
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX_CSV, index=False)
    print(
        f"[resolve_smiles] Built index: {len(df):,} drugs with SMILES "
        f"in {elapsed:.1f}s. Cached → {INDEX_CSV}"
    )
    return df


def main() -> int:
    args = parse_args()

    if not DILIST_XLSX.exists():
        print(f"FAIL: DILIst input missing: {DILIST_XLSX}", file=sys.stderr)
        return 2

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: index
    index_df = load_or_build_index(args.rebuild_index)

    # Step 2: load DILIst
    dilist_df = pd.read_excel(DILIST_XLSX)
    print(f"[resolve_smiles] Loaded DILIst: {len(dilist_df):,} drugs from {DILIST_XLSX}")

    # Step 3: resolve
    resolved_df, failures_df = resolve_dilist(dilist_df, index_df)

    # Step 4: write outputs
    resolved_df.to_csv(RESOLVED_CSV, index=False)
    failures_df.to_csv(FAILURES_CSV, index=False)
    print(f"[resolve_smiles] Wrote {RESOLVED_CSV} ({len(resolved_df):,} rows)")
    print(f"[resolve_smiles] Wrote {FAILURES_CSV} ({len(failures_df):,} rows)")

    # Step 5: report + gate
    total = len(dilist_df)
    rate = len(resolved_df) / total if total else 0.0
    print(f"[resolve_smiles] Resolved {len(resolved_df)} / {total} = {rate:.1%}")
    if not failures_df.empty:
        print("[resolve_smiles] Failure-reason counts:")
        for reason, count in failures_df["reason"].value_counts().items():
            print(f"    {reason}: {count}")

    if rate < RESOLUTION_RATE_GATE:
        print(
            f"FAIL: resolution rate {rate:.1%} < 90% target. "
            f"See {FAILURES_CSV} and either improve the resolver or escalate per D-04.",
            file=sys.stderr,
        )
        return 1

    print(f"[resolve_smiles] PASS: rate {rate:.1%} ≥ {RESOLUTION_RATE_GATE:.0%} gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
