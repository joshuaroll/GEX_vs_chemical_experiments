"""
v2 Phase 1 — Wang/Li 8-layer DNN on Measured vs Predicted GEX
==============================================================
Decomposes the v1 negative finding (0.5930 chemistry-only AUROC vs Wang/Li's
published 0.798 measured-GEX AUROC) into two components:
  - Predicted-vs-measured GEX quality gap (Run A vs Run B)
  - Classifier capacity gap (Wang/Li 8-layer DNN vs v1's small MLP heads)

Run A: measured LINCS DE (978d, mean-pooled per drug) + Wang/Li 8-layer DNN
Run B: MultiDCP-predicted GEX (919d, from dili_features.parquet) + same DNN

Both runs use:
  - Same ~628 LINCS-intersect DILIst drugs
  - Same scaffold-novel split (from dili_split.json, filtered to intersect)
  - Same random split (generated from intersect drugs)
  - 5-fold CV (stratified) × 5 seeds
  - Balanced class weights (Wang/Li protocol)
  - Checkpointing on val_monitor_f (Wang/Li protocol)

Usage:
  cd /raid/home/joshua/projects/GEX_vs_chemical_experiments/multihead_dili
  conda run -n dili_v04_env python src/v2/run_v2_p1.py [--out-dir data/processed/v2_predictions]

Outputs:
  data/processed/v2_predictions/run_{A|B}_split_{scaffold|random}_fold{0..4}_seed{0..4}.parquet
  results/tables/v2_p1_summary.md
"""

import os
import sys
import json
import argparse
import warnings
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# Note: TF 2.21 on this box has CUDA_ERROR_UNSUPPORTED_PTX_VERSION when Keras initializes
# Dense layers with GPU random seeds. DNN is small (676K params) so CPU training is adequate.
# Run on CPU to avoid the PTX incompatibility.
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only

# ---------------------------------------------------------------------------
# Paths (absolute, relative to this file's location when run from project root)
# ---------------------------------------------------------------------------

PROJ_ROOT = Path(__file__).resolve().parents[2]  # multihead_dili/
DILI_DOWNSTREAM = PROJ_ROOT.parent / "dili_downstream"

WANGLI_PROFILES = DILI_DOWNSTREAM / "data/processed/wangli_profiles.csv"
WANGLI_DE = DILI_DOWNSTREAM / "data/processed/wangli_measured_de.npy"
DILI_CANONICAL = DILI_DOWNSTREAM / "data/processed/dili_canonical.csv"
DILI_FEATURES = PROJ_ROOT / "data/processed/dili_features.parquet"
DILI_SPLIT = PROJ_ROOT / "data/processed/dili_split.json"

# ---------------------------------------------------------------------------
# Import evaluate helpers (DeLong + bootstrap from v1 evaluate_dili.py)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(PROJ_ROOT / "src"))
from stage2.evaluate_dili import delong_paired_test, bootstrap_auroc_ci


# ---------------------------------------------------------------------------
# Step 1: Build per-drug measured DE matrix
# ---------------------------------------------------------------------------

def build_measured_features(profiles_path: Path, de_path: Path):
    """
    Mean-pool LINCS DE profiles per compound.
    Returns:
      measured_per_drug: dict {compound_name_lower -> (978,) float32}
      measured_labels:   dict {compound_name_lower -> 0|1}
    """
    print("[Step 1] Building per-drug measured DE matrix...")
    profiles = pd.read_csv(profiles_path)
    de_matrix = np.load(de_path)  # (N_profiles, 978)

    assert de_matrix.shape[0] == len(profiles), (
        f"Profile count mismatch: {de_matrix.shape[0]} rows vs {len(profiles)} profiles"
    )

    measured_per_drug = {}
    measured_labels = {}
    for drug_name, grp in profiles.groupby("compound_name"):
        indices = grp.index.tolist()
        measured_per_drug[drug_name.lower()] = de_matrix[indices].mean(axis=0).astype(np.float32)
        labels = grp["dili_binary"].unique()
        assert len(labels) == 1, f"Inconsistent labels for drug '{drug_name}': {labels}"
        measured_labels[drug_name.lower()] = int(labels[0])

    print(f"  Unique compounds with measured DE: {len(measured_per_drug)}")
    return measured_per_drug, measured_labels


# ---------------------------------------------------------------------------
# Step 2: Find LINCS-intersect subset of DILIst
# ---------------------------------------------------------------------------

def build_intersect(
    measured_per_drug: dict,
    measured_labels: dict,
    dili_canonical_path: Path,
    features_path: Path,
):
    """
    Join wangli per-drug dict with dili_canonical and dili_features.
    Returns intersect_df: rows where drug is in wangli AND in dili_features.
    """
    print("[Step 2] Finding LINCS-intersect DILIst drugs...")
    dili = pd.read_csv(dili_canonical_path)
    features_df = pd.read_parquet(features_path)

    # Lowercase name matching
    dili["name_lower"] = dili["drug_name"].str.lower()
    dili["in_wangli"] = dili["name_lower"].isin(measured_per_drug.keys())
    dili["wangli_name"] = dili["name_lower"].map(
        lambda x: x if x in measured_per_drug else None
    )

    # Feature pert_ids
    feat_pert_ids = set(features_df["pert_id"].tolist())

    # Intersect: must be in wangli AND in features_df
    intersect = dili[dili["in_wangli"] & dili["pert_id"].isin(feat_pert_ids)].copy()
    intersect = intersect.reset_index(drop=True)

    # Sanity: check label consistency between wangli and dili_canonical
    mismatches = 0
    for _, row in intersect.iterrows():
        wang_label = measured_labels[row["wangli_name"]]
        dili_label = int(row["dili_binary"])
        if wang_label != dili_label:
            mismatches += 1
            print(f"  WARNING: Label mismatch for {row['drug_name']}: "
                  f"wangli={wang_label} dili_canonical={dili_label}")
    if mismatches == 0:
        print("  Label consistency check: PASS (no mismatches)")
    else:
        print(f"  WARNING: {mismatches} label mismatches found — check data!")

    print(f"  Intersect: {len(intersect)} of {len(dili)} DILIst drugs found in Wang/Li data")
    print(f"  Label balance: pos={int(intersect['dili_binary'].sum())} "
          f"neg={int((intersect['dili_binary'] == 0).sum())}")
    return intersect, features_df


# ---------------------------------------------------------------------------
# Step 3: Filter scaffold split to intersect drugs + build random split
# ---------------------------------------------------------------------------

def build_splits(intersect_df: pd.DataFrame, split_path: Path, random_seed: int = 42):
    """
    Filter the scaffold split to intersect drugs.
    Build a random split (stratified 70/10/20) for secondary comparison.

    Returns two dicts:
      scaffold_split: {'train': [...pert_id ints], 'val': [...], 'test': [...]}
      random_split:   {'train': [...pert_id ints], 'val': [...], 'test': [...]}
    """
    print("[Step 3] Building filtered scaffold split + random split...")
    raw_split = json.load(open(split_path))

    def to_int(s):
        return int(s.split("_")[1])

    intersect_ids = set(intersect_df["pert_id"].tolist())

    scaffold_split = {
        "train": [to_int(x) for x in raw_split["train"] if to_int(x) in intersect_ids],
        "val":   [to_int(x) for x in raw_split["val"]   if to_int(x) in intersect_ids],
        "test":  [to_int(x) for x in raw_split["test"]  if to_int(x) in intersect_ids],
    }

    print(f"  Scaffold split (filtered): "
          f"train={len(scaffold_split['train'])} "
          f"val={len(scaffold_split['val'])} "
          f"test={len(scaffold_split['test'])}")

    if len(scaffold_split["test"]) < 30:
        print(f"  WARNING: test set < 30 drugs ({len(scaffold_split['test'])}). "
              "Consider relaxing filter. Proceeding anyway.")

    # Random split: stratified 70/10/20 on intersect drugs
    all_ids = intersect_df["pert_id"].values
    all_labels = intersect_df["dili_binary"].values

    rng = np.random.default_rng(random_seed)
    # Stratified: separate pos/neg, shuffle each, split
    pos_ids = all_ids[all_labels == 1]
    neg_ids = all_ids[all_labels == 0]
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    def stratified_split_ids(ids, fracs=(0.70, 0.10, 0.20)):
        n = len(ids)
        n_train = int(n * fracs[0])
        n_val   = int(n * fracs[1])
        return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]

    pos_tr, pos_va, pos_te = stratified_split_ids(pos_ids)
    neg_tr, neg_va, neg_te = stratified_split_ids(neg_ids)

    random_split = {
        "train": list(np.concatenate([pos_tr, neg_tr])),
        "val":   list(np.concatenate([pos_va, neg_va])),
        "test":  list(np.concatenate([pos_te, neg_te])),
    }
    print(f"  Random split: "
          f"train={len(random_split['train'])} "
          f"val={len(random_split['val'])} "
          f"test={len(random_split['test'])}")

    return scaffold_split, random_split


# ---------------------------------------------------------------------------
# Step 4: Build feature matrices
# ---------------------------------------------------------------------------

def get_measured_features(
    pert_ids: list,
    intersect_df: pd.DataFrame,
    measured_per_drug: dict,
    measured_labels: dict,
):
    """Build (X, y) for measured DE features."""
    X, y = [], []
    for pid in pert_ids:
        row = intersect_df[intersect_df["pert_id"] == pid].iloc[0]
        wname = row["wangli_name"]
        X.append(measured_per_drug[wname])
        y.append(measured_labels[wname])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def get_predicted_features(
    pert_ids: list,
    features_df: pd.DataFrame,
    feat_gex_cols: list,
):
    """Build (X, y) for predicted GEX features."""
    pid_to_row = features_df.set_index("pert_id")
    X, y = [], []
    for pid in pert_ids:
        row = pid_to_row.loc[pid]
        X.append(row[feat_gex_cols].values.astype(np.float32))
        y.append(int(row["dili_binary"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 5: Build 5-fold CV indices for a given split
# ---------------------------------------------------------------------------

def make_cv_folds(
    split: dict,
    intersect_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> list:
    """
    Create 5-fold CV over the TRAIN+VAL portion of the split.
    Test set is held out (fixed from scaffold/random split definition).
    Returns list of dicts: [{'train': idx_array, 'val': idx_array, 'test': idx_array}, ...]
    where indices are into the COMBINED array of (train+val+test) drugs.
    """
    all_ids = np.array(split["train"] + split["val"] + split["test"])
    test_ids = np.array(split["test"])
    trainval_ids = np.array(split["train"] + split["val"])

    # Build label array aligned with all_ids
    pid_to_label = dict(zip(intersect_df["pert_id"].values, intersect_df["dili_binary"].values))
    all_labels = np.array([pid_to_label[pid] for pid in all_ids])
    trainval_labels = np.array([pid_to_label[pid] for pid in trainval_ids])

    # Map each pert_id to its index in all_ids
    id_to_idx = {pid: i for i, pid in enumerate(all_ids)}
    test_indices = np.array([id_to_idx[pid] for pid in test_ids])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_indices = []
    for fold_train_rel, fold_val_rel in skf.split(trainval_ids, trainval_labels):
        fold_train_ids = trainval_ids[fold_train_rel]
        fold_val_ids   = trainval_ids[fold_val_rel]
        fold_train_idx = np.array([id_to_idx[pid] for pid in fold_train_ids])
        fold_val_idx   = np.array([id_to_idx[pid] for pid in fold_val_ids])
        fold_indices.append({
            "train": fold_train_idx,
            "val":   fold_val_idx,
            "test":  test_indices,
        })
    return fold_indices, all_ids, all_labels


# ---------------------------------------------------------------------------
# Step 6: Aggregate predictions + compute metrics
# ---------------------------------------------------------------------------

def aggregate_predictions(pred_dir: Path, run_label: str, split_label: str) -> pd.DataFrame:
    """Load all fold × seed predictions for a (run, split) cell."""
    frames = []
    for f in pred_dir.glob(f"run_{run_label}_split_{split_label}_fold*_seed*.parquet"):
        frames.append(pd.read_parquet(f))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def compute_cell_metrics(df: pd.DataFrame, n_boot: int = 10_000) -> dict:
    """
    Compute pooled AUROC + 95% bootstrap CI for a (run, split) cell.
    Bootstrap resamples at drug level (mean proba per drug).
    """
    y = df["true_label"].values.astype(int)
    p = df["predicted_proba"].values.astype(float)

    auroc_pooled = float(roc_auc_score(y, p))
    auprc_pooled = float(average_precision_score(y, p))

    # Drug-level mean proba
    drug_df = df.groupby("pert_id").agg(
        true_label=("true_label", "first"),
        mean_proba=("predicted_proba", "mean"),
    ).reset_index()
    y_drug = drug_df["true_label"].values.astype(int)
    p_drug = drug_df["mean_proba"].values.astype(float)

    boot_mean, ci_lo, ci_hi = bootstrap_auroc_ci(y_drug, p_drug, n_boot=n_boot, rng_seed=42)

    return {
        "auroc_pooled": auroc_pooled,
        "auprc_pooled": auprc_pooled,
        "boot_auroc_mean": boot_mean,
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
        "n_drugs": int(drug_df.shape[0]),
        "n_predictions": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Step 7: Write summary
# ---------------------------------------------------------------------------

def write_v2_p1_summary(
    results: dict,
    intersect_count: int,
    scaffold_split: dict,
    random_split: dict,
    out_path: Path,
):
    """Write results/tables/v2_p1_summary.md"""
    print(f"[Step 7] Writing summary to {out_path}...")

    # DeLong paired tests (Run A vs Run B, per split)
    delong_notes = {}
    for split_label in ["scaffold", "random"]:
        ra = results.get(("A_measured", split_label))
        rb = results.get(("B_predicted", split_label))
        if ra is None or rb is None:
            delong_notes[split_label] = "N/A (missing data)"
            continue
        # We need paired predictions — load them
        pred_dir = out_path.parent.parent.parent / "data/processed/v2_predictions"
        df_a = aggregate_predictions(pred_dir, "A_measured", split_label)
        df_b = aggregate_predictions(pred_dir, "B_predicted", split_label)
        if df_a is None or df_b is None:
            delong_notes[split_label] = "N/A (missing predictions)"
            continue

        # Align by drug
        da = df_a.groupby("pert_id").agg(
            true_label=("true_label", "first"),
            mean_proba=("predicted_proba", "mean")
        ).reset_index()
        db = df_b.groupby("pert_id").agg(
            true_label=("true_label", "first"),
            mean_proba=("predicted_proba", "mean")
        ).reset_index()
        merged = da.merge(db, on="pert_id", suffixes=("_a", "_b"))
        y   = merged["true_label_a"].values.astype(int)
        p_a = merged["mean_proba_a"].values
        p_b = merged["mean_proba_b"].values
        auc_a, auc_b, z, p_val = delong_paired_test(y, p_a, p_b)
        delong_notes[split_label] = f"ΔAUROC={auc_a - auc_b:+.4f}, z={z:.3f}, p={p_val:.4f}"

    lines = [
        "# v2 Phase 1 Summary — Wang/Li 8-Layer DNN: Measured vs Predicted GEX",
        "",
        "**Date:** 2026-05-20",
        "**Purpose:** Decompose v1 negative finding into predicted-vs-measured gap and "
        "classifier-capacity gap.",
        "",
        "## Drug counts",
        "",
        f"- Total DILIst drugs: 1118",
        f"- LINCS-intersect (in Wang/Li profiles + dili_features): **{intersect_count}**",
        f"- Scaffold split (filtered): "
        f"train={len(scaffold_split['train'])} val={len(scaffold_split['val'])} "
        f"test={len(scaffold_split['test'])}",
        f"- Random split (70/10/20 stratified): "
        f"train={len(random_split['train'])} val={len(random_split['val'])} "
        f"test={len(random_split['test'])}",
        "",
        "## Architecture",
        "",
        "Wang/Li 8-layer DNN: 512→256→128→64→32→16→8→1(sigmoid), ELU, Adam, BCE loss,",
        "balanced class weights, checkpoint on val_monitor_f (Wang/Li criterion),",
        "EarlyStopping(patience=5) on val_loss, max 100 epochs, batch_size=128.",
        "Run A input_dim=978 (measured LINCS DE); Run B input_dim=919 (MultiDCP predicted GEX).",
        "",
        "## Headline Results",
        "",
        "| Run | Split | AUROC (pooled) | 95% CI | n_drugs | vs Run-A (DeLong) |",
        "|-----|-------|---------------|--------|---------|-------------------|",
    ]

    for split_label in ["scaffold", "random"]:
        split_name = "scaffold-novel" if split_label == "scaffold" else "random"
        for run_label, run_name in [("A_measured", "A: measured GEX (978d)"), ("B_predicted", "B: predicted GEX (919d)")]:
            key = (run_label, split_label)
            if key not in results:
                lines.append(f"| {run_name} | {split_name} | N/A | N/A | N/A | N/A |")
                continue
            m = results[key]
            delong_str = "—" if run_label == "A_measured" else delong_notes.get(split_label, "N/A")
            lines.append(
                f"| {run_name} | {split_name} "
                f"| {m['auroc_pooled']:.4f} "
                f"| [{m['ci_95_lo']:.4f}, {m['ci_95_hi']:.4f}] "
                f"| {m['n_drugs']} "
                f"| {delong_str} |"
            )

    lines += [
        "",
        "## Comparison to v1 and Wang/Li published",
        "",
        "| Method | Split | AUROC |",
        "|--------|-------|-------|",
        "| v1 chemistry-only (MolFormer + mlp2) | scaffold-novel | 0.5930 |",
        "| v1 chemistry-only (MolFormer + mlp2) | random | 0.6645 |",
        "| Wang/Li 2020 published (their split) | random-ish | 0.798 |",
    ]

    # Add v2 numbers for comparison
    for split_label in ["scaffold", "random"]:
        split_name = "scaffold-novel" if split_label == "scaffold" else "random"
        for run_label, run_name in [("A_measured", "v2 Run A: measured GEX 8-layer DNN"),
                                    ("B_predicted", "v2 Run B: predicted GEX 8-layer DNN")]:
            key = (run_label, split_label)
            if key in results:
                m = results[key]
                lines.append(f"| {run_name} | {split_name} | {m['auroc_pooled']:.4f} |")

    lines += [
        "",
        "## Interpretation",
        "",
    ]

    # Generate interpretation
    ra_scaf = results.get(("A_measured", "scaffold"))
    rb_scaf = results.get(("B_predicted", "scaffold"))
    ra_rand = results.get(("A_measured", "random"))
    rb_rand = results.get(("B_predicted", "random"))
    v1_scaf = 0.5930
    v1_rand = 0.6645
    wangli_published = 0.798

    if ra_scaf and rb_scaf and ra_rand and rb_rand:
        measured_vs_pred_gap_scaf = ra_scaf["auroc_pooled"] - rb_scaf["auroc_pooled"]
        measured_vs_pred_gap_rand = ra_rand["auroc_pooled"] - rb_rand["auroc_pooled"]
        classifier_gap_scaf = ra_scaf["auroc_pooled"] - v1_scaf
        classifier_gap_rand = ra_rand["auroc_pooled"] - v1_rand

        lines += [
            f"### Gap decomposition (scaffold-novel split)",
            "",
            f"- v1 chemistry-only AUROC: {v1_scaf:.4f}",
            f"- Wang/Li published AUROC: {wangli_published:.4f}",
            f"- Total gap to explain: {wangli_published - v1_scaf:+.4f}",
            "",
            f"- Run A (measured GEX + 8-layer DNN): {ra_scaf['auroc_pooled']:.4f}",
            f"- Run B (predicted GEX + 8-layer DNN): {rb_scaf['auroc_pooled']:.4f}",
            "",
            f"- **Classifier-capacity component** (Run A vs v1 chemistry-only): "
            f"{classifier_gap_scaf:+.4f}",
            f"- **Predicted-vs-measured GEX component** (Run A vs Run B): "
            f"{measured_vs_pred_gap_scaf:+.4f}",
            "",
        ]

        if abs(measured_vs_pred_gap_scaf) > abs(classifier_gap_scaf):
            dom_component = "predicted-vs-measured GEX quality"
        else:
            dom_component = "classifier capacity (8-layer DNN vs small MLP head)"

        lines += [
            f"**Dominant component:** {dom_component}",
            "",
            "### Implications for v2",
            "",
        ]

        if ra_scaf["auroc_pooled"] > 0.65:
            lines += [
                f"Run A achieves {ra_scaf['auroc_pooled']:.4f} AUROC on scaffold-novel split — "
                "substantially above v1 chemistry-only (0.5930). This suggests that MEASURED "
                "LINCS GEX provides significant DILI signal that MultiDCP-predicted GEX fails to capture.",
                "Priority for v2: improve predicted signature quality (DILI-tuned MultiDCP, "
                "or use measured LINCS GEX where available).",
            ]
        elif ra_scaf["auroc_pooled"] < 0.60:
            lines += [
                f"Run A achieves only {ra_scaf['auroc_pooled']:.4f} AUROC on scaffold-novel split. "
                "Even measured LINCS GEX does not substantially improve on chemistry-only. "
                "This suggests scaffold-novelty is the primary challenge — "
                "GEX signal (measured or predicted) does not generalize well to novel scaffolds.",
            ]
        else:
            lines += [
                f"Run A achieves {ra_scaf['auroc_pooled']:.4f} AUROC — moderate improvement over "
                f"v1 chemistry-only (0.5930). Measured GEX adds some signal on scaffold-novel drugs.",
            ]

    lines += [
        "",
        "## Sanity note",
        "",
        "Wang/Li published 0.798 AUROC was on PROFILE-LEVEL training (N=5517 profiles, "
        "multiple profiles per drug treated as independent). Drug-level training (N=628 drugs) "
        "is more conservative and avoids within-drug label leakage but yields lower AUROC.",
        "All AUROCs in this experiment reflect drug-unit predictions (one test label per drug).",
        "",
    ]
    if ra_rand:
        lines.append(f"Run A random AUROC = {ra_rand['auroc_pooled']:.4f} "
                     "(Wang/Li published 0.798 used profile-level, not drug-level).")
    if ra_scaf:
        lines.append(f"Run A scaffold AUROC = {ra_scaf['auroc_pooled']:.4f} "
                     "(expected to be below random — scaffold OOD is harder).")

    lines += [
        "",
        "## Method notes",
        "",
        "- Wang/Li DNN.ipynb uses random splits (no scaffold filter), balanced class weights,",
        "  custom monitor_f metric, and 30 epochs. This reproduction uses same architecture",
        "  but adds EarlyStopping(patience=5) on val_loss, max 100 epochs, and evaluates on",
        "  our scaffold-novel and random splits for direct comparability to v1 results.",
        "- The Wang/Li 0.798 AUROC was on their dataset/split (not scaffold-novel DILIst).",
        "  Direct numeric comparison is informative but not apples-to-apples.",
        "- Predictions cached to data/processed/v2_predictions/ (gitignored).",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"  Written: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="v2 P1 — Wang/Li DNN on measured vs predicted GEX")
    parser.add_argument("--out-dir", default="data/processed/v2_predictions",
                        help="Directory for prediction parquets")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--skip-scaffold", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--skip-run-a", action="store_true")
    parser.add_argument("--skip-run-b", action="store_true")
    args = parser.parse_args()

    out_dir = PROJ_ROOT / args.out_dir
    results_path = PROJ_ROOT / "results/tables/v2_p1_summary.md"

    print("=" * 70)
    print("v2 Phase 1 — Wang/Li 8-layer DNN: Measured vs Predicted GEX")
    print("=" * 70)

    # Step 1: Build measured features
    measured_per_drug, measured_labels = build_measured_features(WANGLI_PROFILES, WANGLI_DE)

    # Step 2: Find intersect
    intersect_df, features_df = build_intersect(
        measured_per_drug, measured_labels, DILI_CANONICAL, DILI_FEATURES
    )

    # Predicted GEX columns
    feat_gex_cols = [c for c in features_df.columns if c.startswith("feat_gex_")]
    print(f"  Predicted GEX columns: {len(feat_gex_cols)}")

    # Step 3: Build splits
    scaffold_split, random_split = build_splits(intersect_df, DILI_SPLIT)

    # Sanity: verify label agreement between measured and predicted for intersect
    print("[Sanity] Verifying label agreement between pipelines...")
    mismatch_count = 0
    for pid in intersect_df["pert_id"].values[:50]:  # spot-check first 50
        row = intersect_df[intersect_df["pert_id"] == pid].iloc[0]
        feat_row = features_df[features_df["pert_id"] == pid]
        if len(feat_row) == 0:
            continue
        dili_label = int(feat_row["dili_binary"].iloc[0])
        wang_label = measured_labels[row["wangli_name"]]
        if dili_label != wang_label:
            mismatch_count += 1
    print(f"  Label mismatches (spot-check 50 drugs): {mismatch_count}")

    # Collect results
    all_results_list = []
    cell_results = {}

    def run_cell(run_label: str, split_label: str, split: dict):
        """Run one (run, split) cell with 5-fold CV x N seeds."""
        print(f"\n{'='*60}")
        print(f"Cell: Run {run_label} | Split {split_label}")
        print(f"{'='*60}")

        fold_indices, all_ids, all_labels = make_cv_folds(
            split, intersect_df, n_splits=args.n_folds, seed=42
        )

        if run_label == "A_measured":
            X_all, y_all = get_measured_features(all_ids.tolist(), intersect_df, measured_per_drug, measured_labels)
        else:
            X_all, y_all = get_predicted_features(all_ids.tolist(), features_df, feat_gex_cols)

        print(f"  X shape: {X_all.shape}  y balance: pos={int(y_all.sum())} neg={int((y_all==0).sum())}")

        from src.v2.wangli_8layer import train_with_cv
        with tempfile.TemporaryDirectory() as tmpdir:
            fold_results = train_with_cv(
                X_all, y_all, all_ids,
                fold_indices=fold_indices,
                seeds=args.seeds,
                run_label=run_label,
                split_label=split_label,
                out_dir=out_dir,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                tmpdir=tmpdir,
            )
        all_results_list.extend(fold_results)

        # Aggregate
        df_all_preds = aggregate_predictions(out_dir, run_label, split_label)
        if df_all_preds is not None:
            metrics = compute_cell_metrics(df_all_preds, n_boot=args.n_boot)
            cell_results[(run_label, split_label)] = metrics
            print(f"\n  => Pooled AUROC: {metrics['auroc_pooled']:.4f} "
                  f"[{metrics['ci_95_lo']:.4f}, {metrics['ci_95_hi']:.4f}]")

            # Sanity gate
            if run_label == "A_measured" and metrics["auroc_pooled"] < 0.55:
                print(f"\n  *** SANITY GATE TRIGGERED: Run A AUROC = {metrics['auroc_pooled']:.4f} < 0.55 ***")
                print("  *** Stopping to debug. Check data pipeline. ***")
                # Write partial summary and exit
                write_v2_p1_summary(
                    cell_results, len(intersect_df), scaffold_split, random_split, results_path
                )
                sys.exit(1)

    # Execute cells
    if not args.skip_scaffold:
        if not args.skip_run_a:
            run_cell("A_measured", "scaffold", scaffold_split)
        if not args.skip_run_b:
            run_cell("B_predicted", "scaffold", scaffold_split)

    if not args.skip_random:
        if not args.skip_run_a:
            run_cell("A_measured", "random", random_split)
        if not args.skip_run_b:
            run_cell("B_predicted", "random", random_split)

    # Write results summary
    if cell_results:
        write_v2_p1_summary(
            cell_results, len(intersect_df), scaffold_split, random_split, results_path
        )

    print("\n" + "=" * 70)
    print("v2 Phase 1: COMPLETE")
    for (run, split), m in sorted(cell_results.items()):
        print(f"  {run} | {split}: AUROC={m['auroc_pooled']:.4f} [{m['ci_95_lo']:.4f}, {m['ci_95_hi']:.4f}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
