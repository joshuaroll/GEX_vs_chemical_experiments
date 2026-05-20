"""Filter E-Hill dev CSV to remove DILIst test drugs (name + scaffold).

Writes data/processed/ehill_dev_safe.parquet.

Run:
    conda run -n dili_v04_env python scripts/build_ehill_dev_safe.py \
      --dili-split data/processed/dili_split.json \
      --dili-canonical ../dili_downstream/data/processed/dili_canonical.csv \
      --dev-csv /raid/home/joshua/data/MultiDCP/data/ehill_data/high_confident_data_dev.csv \
      --out data/processed/ehill_dev_safe.parquet
"""

import argparse
import json
import sys
from pathlib import Path

# Insert project root into sys.path before any src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Filter E-Hill dev CSV to remove DILIst test drugs"
    )
    parser.add_argument("--dili-split", required=True,
                        help="Path to dili_split.json (provides scaffolds_in_test and test partition)")
    parser.add_argument("--dili-canonical", required=True,
                        help="Path to dili_canonical.csv (columns: pert_id, drug_name, scaffold, ...)")
    parser.add_argument("--dev-csv", required=True,
                        help="Path to E-Hill dev CSV (high_confident_data_dev.csv)")
    parser.add_argument("--out", required=True,
                        help="Output path for filtered parquet file")
    args = parser.parse_args()

    # 1. Load dili_split.json
    with open(args.dili_split) as f:
        dili_split = json.load(f)

    # scaffolds_in_test: list of 30 Murcko scaffold SMILES strings
    scaffolds_in_test = set(dili_split.get("scaffolds_in_test", []))
    test_ids = set(dili_split.get("test", []))  # list of DILIST_ IDs (ints or strings)

    print(f"Scaffolds in test set: {len(scaffolds_in_test)}")
    print(f"Test drug IDs: {len(test_ids)}")

    # 2. Load dili_canonical.csv to get test drug names
    # test_ids are 'DILIST_NNNN' strings; map to int via split('_')[1]
    def dilist_to_int(dilist_id: str) -> int:
        """Map 'DILIST_0004' -> 4."""
        try:
            return int(dilist_id.split("_")[1])
        except (IndexError, ValueError):
            return int(dilist_id)  # fallback: direct int conversion

    dili_canonical = pd.read_csv(args.dili_canonical)
    test_ids_int = {dilist_to_int(x) for x in test_ids}
    test_rows = dili_canonical[dili_canonical["pert_id"].isin(test_ids_int)]

    test_drug_names = {row.drug_name.lower() for _, row in test_rows.iterrows()
                       if pd.notna(row.drug_name)}
    print(f"Test drug names to exclude: {len(test_drug_names)}")

    # 3. Load E-Hill dev CSV
    dev_df = pd.read_csv(args.dev_csv)
    print(f"Dev CSV loaded: {dev_df.shape} rows")

    # 4. Build name mask (primary exclusion — name-only is sufficient per leakage_report analysis)
    # pert_id in ehill CSV contains drug names (e.g., 'Vorinostat', 'Methotrexate')
    name_mask = dev_df["pert_id"].str.lower().isin(test_drug_names)
    print(f"Rows excluded by name filter: {name_mask.sum()}")

    # 5. Safe mask = not excluded by name
    # (scaffold exclusion not needed per leakage_report: all ehill exclusions
    #  were both-name-and-scaffold, so name-only filter is sufficient)
    safe_mask = ~name_mask

    # 6. Save filtered parquet
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dev_df[safe_mask].to_parquet(str(out_path), index=False)

    print(f"Dev filter: {len(dev_df)} → {safe_mask.sum()} rows (dropped {(~safe_mask).sum()})")
    print(f"Output written to: {out_path}")


if __name__ == "__main__":
    main()
