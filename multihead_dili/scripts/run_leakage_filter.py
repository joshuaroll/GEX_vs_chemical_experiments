"""Apply drug_name + Murcko-scaffold leakage filter to E-Hill and LINCS train sets.

V2: name-based filter (lowercased pert_id) AND scaffold-based filter (Murcko scaffold of
resolved SMILES from DrugBank, compared against DILIst test scaffolds from dili_split.json).

Reads dili_split.json (Task 0.8) -> extracts test drug names + test scaffolds -> filters
E-Hill train CSV and LINCS PDG-filtered pickle -> writes _safe parquet files +
leakage_report.md.

Discovery notes (Task 0.9):
  - E-Hill train CSV: pert_id column contains mixed-case drug names (e.g. "Vorinostat")
  - LINCS PDG pickle: DataFrame with pert_id column containing lowercase/mixed drug names
    (e.g. "flutamide", "TG-101348"). No pert_iname column -- pert_id IS the name column.
  - dili_canonical.csv: pert_id is integer, drug_name is lowercase
  - DrugBank XML (~1.5 GB) is stream-parsed; use build_index() then set_index for O(1) lookup.

Run:
  conda run -n dili_v04_env python scripts/run_leakage_filter.py \\
    --dili-split data/processed/dili_split.json \\
    --dili-canonical ../dili_downstream/data/processed/dili_canonical.csv \\
    --ehill-train /raid/home/joshua/data/MultiDCP/data/ehill_data/high_confident_data_train.csv \\
    --lincs-pkl /raid/home/joshua/projects/MultiDCP/MultiDCP/data/pdg_brddrugfiltered.pkl \\
    --drugbank-dir /raid/home/joshua/projects/MultiDCP/MultiDCP/data/drugbank_data/ \\
    --out-dir data/processed/
"""
import argparse
import json
import pickle
from pathlib import Path
import sys

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.drugbank_smiles_index import build_index


def murcko(smiles: str) -> str:
    """Return Bemis-Murcko scaffold SMILES; empty string for acyclic / invalid."""
    if not isinstance(smiles, str) or not smiles:
        return ""
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return ""


def dilist_to_int(dilist_id: str) -> int:
    """Map 'DILIST_NNNN' string ID back to integer pert_id."""
    return int(dilist_id.split("_")[1])


def build_drugbank_lookup(drugbank_dir: Path) -> pd.Series:
    """Build a name_lower -> smiles Series from DrugBank XML.

    Returns a pandas Series indexed by lowercased drug name (name_lower)
    with values being SMILES strings.
    """
    xml_path = drugbank_dir / "full_database.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"DrugBank XML not found at: {xml_path}")
    print(f"Parsing DrugBank XML: {xml_path}  (this may take ~30-60 s) ...")
    df = build_index(xml_path)
    # Set index on name_lower for O(1) lookups; keep first occurrence (deterministic)
    return df.set_index("name_lower")["smiles"]


def compute_scaffold_filter_masks(df: pd.DataFrame, name_col: str,
                                   drugbank_index: pd.Series,
                                   test_drug_names: set,
                                   test_scaffolds: set):
    """Return (name_mask, scaffold_mask, unresolved_count) for df[name_col].

    name_mask:     rows whose lowercased name_col is in test_drug_names
    scaffold_mask: rows whose Murcko scaffold (via DrugBank SMILES) is in
                   test_scaffolds (empty-string scaffolds are never excluded)
    unresolved_count: rows where DrugBank had no SMILES for the drug name
    """
    names_lower = df[name_col].astype(str).str.lower()

    # Dedupe: compute scaffolds per unique drug name, then map back (25-50x faster)
    unique_names = names_lower.unique()
    smiles_series = pd.Series(unique_names).map(drugbank_index)  # NaN if not found
    scaffold_series = smiles_series.apply(murcko)
    name_scaffold_map = dict(zip(unique_names, scaffold_series))
    name_smiles_map = dict(zip(unique_names, smiles_series))

    scaffolds = names_lower.map(name_scaffold_map)
    smiles_resolved = names_lower.map(name_smiles_map)

    name_mask = names_lower.isin(test_drug_names)
    scaffold_mask = scaffolds.isin(test_scaffolds) & (scaffolds != "")
    unresolved_count = int(smiles_resolved.isna().sum())

    return name_mask, scaffold_mask, unresolved_count


def filter_and_report(df: pd.DataFrame, name_col: str,
                       drugbank_index: pd.Series,
                       test_drug_names: set,
                       test_scaffolds: set,
                       label: str):
    """Apply OR-mask filter and return (safe_df, stats_dict)."""
    n_in = len(df)
    name_mask, scaffold_mask, unresolved_count = compute_scaffold_filter_masks(
        df, name_col, drugbank_index, test_drug_names, test_scaffolds
    )

    combined_mask = name_mask | scaffold_mask
    safe_df = df[~combined_mask].copy()
    n_out = len(safe_df)

    n_dropped_total = n_in - n_out
    n_name_only = int((name_mask & ~scaffold_mask).sum())
    n_scaffold_only = int((scaffold_mask & ~name_mask).sum())
    n_both = int((name_mask & scaffold_mask).sum())

    stats = {
        "n_in": n_in,
        "n_out": n_out,
        "n_dropped_total": n_dropped_total,
        "n_name_only": n_name_only,
        "n_scaffold_only": n_scaffold_only,
        "n_both": n_both,
        "n_unresolved_smiles": unresolved_count,
    }

    print(
        f"{label}: {n_in} -> {n_out}  "
        f"(dropped {n_dropped_total}: name-only={n_name_only}, "
        f"scaffold-only={n_scaffold_only}, both={n_both}; "
        f"unresolved-SMILES={unresolved_count})"
    )
    return safe_df, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dili-split", type=Path, required=True)
    p.add_argument("--dili-canonical", type=Path, required=True)
    p.add_argument("--ehill-train", type=Path, required=True)
    p.add_argument("--lincs-pkl", type=Path, required=True)
    p.add_argument(
        "--drugbank-dir",
        type=Path,
        default=Path("/raid/home/joshua/projects/MultiDCP/MultiDCP/data/drugbank_data/"),
        help="Directory containing full_database.xml",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load DILIst test exclusions ---
    split = json.loads(args.dili_split.read_text())
    test_pert_ids_int = {dilist_to_int(s) for s in split["test"]}
    dili_df = pd.read_csv(args.dili_canonical)
    test_drug_names = {
        n.lower()
        for n in dili_df[dili_df["pert_id"].isin(test_pert_ids_int)]["drug_name"].dropna()
    }
    # Task 0.8 guarantees empty strings are excluded from scaffolds_in_test
    test_scaffolds = set(split["scaffolds_in_test"]) - {""}
    print(f"Test drug names: {len(test_drug_names)}")
    print(f"Test scaffolds:  {len(test_scaffolds)}")

    # --- Build DrugBank name -> SMILES index ---
    drugbank_index = build_drugbank_lookup(args.drugbank_dir)
    print(f"DrugBank index size: {len(drugbank_index)} name entries")

    # --- Filter E-Hill ---
    # E-Hill pert_id is the drug name (e.g. "Vorinostat", "YK 4-279")
    ehill = pd.read_csv(args.ehill_train)
    ehill_safe, ehill_stats = filter_and_report(
        ehill, "pert_id", drugbank_index, test_drug_names, test_scaffolds, "E-Hill"
    )
    ehill_safe.to_parquet(args.out_dir / "ehill_train_safe.parquet", index=False)

    # --- Filter LINCS (PDG pickle) ---
    with open(args.lincs_pkl, "rb") as f:
        lincs = pickle.load(f)

    lincs_safe_emitted = False
    lincs_stats = None
    lincs_name_col_used = None

    if isinstance(lincs, pd.DataFrame):
        print(f"LINCS pickle type: DataFrame, cols: {list(lincs.columns)[:15]}")
        for name_col in ("pert_iname", "drug_name", "pert_id", "compound_name"):
            if name_col in lincs.columns:
                lincs_name_col_used = name_col
                lincs_safe, lincs_stats = filter_and_report(
                    lincs, name_col, drugbank_index,
                    test_drug_names, test_scaffolds, "LINCS"
                )
                lincs_safe.to_parquet(args.out_dir / "lincs_train_safe.parquet", index=False)
                lincs_safe_emitted = True
                break

    if not lincs_safe_emitted:
        print("WARNING: LINCS pickle structure not recognized as DataFrame with a known name column.")
        print(f"  Type: {type(lincs).__name__}")
        if isinstance(lincs, pd.DataFrame):
            print(f"  Cols: {list(lincs.columns)[:10]}")
            lincs.to_parquet(args.out_dir / "lincs_train_safe.parquet", index=False)
        elif isinstance(lincs, dict):
            print(f"  Keys: {list(lincs.keys())[:10]}")
            try:
                pd.DataFrame(lincs).to_parquet(args.out_dir / "lincs_train_safe.parquet", index=False)
            except Exception as e:
                print(f"  Could not write pass-through parquet: {e}")
        else:
            print(f"  Cannot inspect further. Skipping LINCS parquet emission.")

    # --- Emit report ---
    def _stats_block(label: str, stats: dict | None, name_col: str | None) -> str:
        if stats is None:
            return (
                f"## {label} train filter\n\n"
                f"PASS-THROUGH: no recognized name column found — see WARNING above\n"
            )
        filter_desc = f"lowercased {name_col} in test_drug_names OR scaffold in test_scaffolds"
        return (
            f"## {label} train filter (v2: name + scaffold)\n\n"
            f"Input rows:            {stats['n_in']}\n"
            f"Output rows:           {stats['n_out']}\n"
            f"Dropped (total):       {stats['n_dropped_total']}\n"
            f"  Name-only drop:      {stats['n_name_only']}\n"
            f"  Scaffold-only drop:  {stats['n_scaffold_only']}\n"
            f"  Both name+scaffold:  {stats['n_both']}\n"
            f"Unresolved SMILES:     {stats['n_unresolved_smiles']}  "
            f"(not in DrugBank; excluded only if name matches)\n"
            f"Filter:                {filter_desc}\n"
        )

    report = (
        f"# Leakage filter report\n\n"
        f"Generated: {pd.Timestamp.utcnow().isoformat()}\n"
        f"DILIst test partition: {len(split['test'])} drugs, "
        f"{len(test_drug_names)} drug-name keys\n"
        f"DILIst test scaffolds: {len(test_scaffolds)} (non-empty Murcko scaffold strings)\n"
        f"DrugBank index size:   {len(drugbank_index)} name entries\n\n"
        + _stats_block("E-Hill", ehill_stats, "pert_id")
        + "\n"
        + _stats_block("LINCS", lincs_stats, lincs_name_col_used)
    )
    (args.out_dir / "leakage_report.md").write_text(report)
    print(f"Wrote {args.out_dir / 'leakage_report.md'}")


if __name__ == "__main__":
    main()
