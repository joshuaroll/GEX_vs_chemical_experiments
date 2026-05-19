"""Build a fixed Bemis-Murcko scaffold split on DILIst (1,118 drugs).

Output: data/processed/dili_split.json
  {"train": [pert_ids...], "val": [...], "test": [...], "scaffolds_in_test": [...]}

The test scaffolds list is the load-bearing artifact for the leakage filter
(scaffold_split is then consumed by upstream_filter to exclude E-Hill/LINCS
training rows whose Murcko scaffold falls in the DILIst test set).

Reuses src.data.scaffold_split for the partitioning logic (copied from v0.5).

Run:
  conda run -n dili_v04_env python scripts/build_dili_scaffold_split.py \\
    --canonical ../dili_downstream/data/processed/dili_canonical.csv \\
    --out data/processed/dili_split.json \\
    --test-frac 0.15 --val-frac 0.1 --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root (parent of scripts/) is on sys.path so `src` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.scaffold_split import scaffold_split


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a fixed Bemis-Murcko scaffold split on DILIst."
    )
    p.add_argument("--canonical", type=Path, required=True,
                   help="Path to dili_canonical.csv (must have pert_id, scaffold, dili_binary).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSON path (e.g. data/processed/dili_split.json).")
    p.add_argument("--test-frac", type=float, default=0.15,
                   help="Fraction of drugs for test set (default: 0.15).")
    p.add_argument("--val-frac", type=float, default=0.10,
                   help="Fraction of drugs for val set (default: 0.10).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for scaffold assignment (default: 42).")
    args = p.parse_args()

    # Derive train_frac so the three fractions sum to 1.0
    train_frac = 1.0 - args.test_frac - args.val_frac
    if train_frac <= 0:
        raise ValueError(
            f"test_frac ({args.test_frac}) + val_frac ({args.val_frac}) >= 1.0; "
            "no room for train set."
        )

    df = pd.read_csv(args.canonical)

    # Validate required columns are present
    required_cols = {"pert_id", "scaffold", "dili_binary"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"dili_canonical.csv is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    print(
        f"Loaded {len(df)} drugs from {args.canonical} "
        f"(train_frac={train_frac:.2f}, val_frac={args.val_frac}, "
        f"test_frac={args.test_frac}, seed={args.seed})"
    )

    # Call the v0.5 scaffold_split API.
    # Returns {"train": ["DILIST_0001", ...], "val": [...], "test": [...]}.
    splits = scaffold_split(
        df,
        seed=args.seed,
        train_frac=train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )

    # Build test_scaffolds: unique Murcko scaffolds from test-set drugs.
    # The splits use DILIST_NNNN IDs; map back via the integer pert_id.
    # DILIST_NNNN -> int(NNNN) to match the pert_id column.
    def dilist_to_int(dilist_id: str) -> int:
        return int(dilist_id.split("_")[1])

    test_int_ids = {dilist_to_int(pid) for pid in splits["test"]}
    test_scaffolds = sorted(
        {s for s in df[df["pert_id"].isin(test_int_ids)]["scaffold"].dropna() if s != ""}
    )

    payload = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "scaffolds_in_test": test_scaffolds,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(
        f"Wrote {args.out}: "
        f"train={len(payload['train'])} val={len(payload['val'])} "
        f"test={len(payload['test'])} test_scaffolds={len(payload['scaffolds_in_test'])}"
    )


if __name__ == "__main__":
    main()
