#!/usr/bin/env python3
"""
Phase 4 — HG3 check + P4 ablation summary generation.

Reads data/processed/P4_runs.parquet and all prediction parquets.
Checks halt gate 3: embed-only (var1) random-split AUROC >= 0.55.
Writes results/tables/P4_ablation_summary.md.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import glob

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

VARIANT_NAMES = {
    1: "embed-only (MolFormer 768d)",
    2: "gex-only (MODEL_GEX 919d)",
    3: "dose-only (MODEL_DOSE 1d)",
    4: "embed+gex (1687d)",
    5: "embed+dose (769d)",
    6: "gex+dose (920d)",
    7: "all-three (1688d) ← headline",
}

HG3_THRESHOLD = 0.55


def compute_auroc_from_files(pattern: str) -> list:
    from sklearn.metrics import roc_auc_score
    files = glob.glob(pattern)
    aurocs = []
    for f in files:
        df = pd.read_parquet(f)
        if len(df["true_label"].unique()) >= 2:
            aurocs.append(roc_auc_score(df["true_label"], df["predicted_proba"]))
    return aurocs


def main():
    base = _PROJECT_ROOT
    runs_path = base / "data/processed/P4_runs.parquet"
    pred_dir = base / "data/processed/predictions"
    out_path = base / "results/tables/P4_ablation_summary.md"

    if not runs_path.exists():
        # Recompute from individual parquets
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            matthews_corrcoef, balanced_accuracy_score,
        )
        rows = []
        for f in sorted(pred_dir.glob("*.parquet")):
            df = pd.read_parquet(f)
            if len(df["true_label"].unique()) < 2:
                continue
            rows.append({
                "variant": df["variant"].iloc[0],
                "head_depth": df["head_depth"].iloc[0],
                "split_type": df["split_type"].iloc[0],
                "fold": df["fold"].iloc[0],
                "seed": df["seed"].iloc[0],
                "auroc": roc_auc_score(df["true_label"], df["predicted_proba"]),
                "auprc": average_precision_score(df["true_label"], df["predicted_proba"]),
                "mcc": matthews_corrcoef(df["true_label"], df["predicted_label"]),
                "balanced_acc": balanced_accuracy_score(df["true_label"], df["predicted_label"]),
            })
        df_runs = pd.DataFrame(rows)
        df_runs.to_parquet(runs_path, index=False)
        print(f"Rebuilt P4_runs.parquet: {len(df_runs)} rows")
    else:
        df_runs = pd.read_parquet(runs_path)
        print(f"Loaded P4_runs.parquet: {len(df_runs)} rows")

    # -------------------------------------------------------------------
    # HG3: embed-only (var1) random-split AUROC
    # -------------------------------------------------------------------
    hg3_mask = (df_runs["variant"] == 1) & (df_runs["split_type"] == "random")
    hg3_aurocs = df_runs.loc[hg3_mask, "auroc"].dropna()
    hg3_mean = hg3_aurocs.mean()
    hg3_std = hg3_aurocs.std()
    hg3_n = len(hg3_aurocs)
    hg3_pass = hg3_mean >= HG3_THRESHOLD

    print(f"\nHG3 — embed-only (var1) random-split AUROC:")
    print(f"  mean={hg3_mean:.4f}  std={hg3_std:.4f}  n={hg3_n}")
    print(f"  threshold={HG3_THRESHOLD}  verdict={'PASS' if hg3_pass else 'HALT'}")

    # -------------------------------------------------------------------
    # Per-variant per-head summary table
    # -------------------------------------------------------------------
    summary_rows = []
    for v in sorted(df_runs["variant"].unique()):
        for h in ["linear", "mlp1", "mlp2"]:
            for s in ["scaffold", "random"]:
                mask = (df_runs["variant"] == v) & (df_runs["head_depth"] == h) & (df_runs["split_type"] == s)
                sub = df_runs[mask]
                if len(sub) == 0:
                    continue
                summary_rows.append({
                    "variant": v,
                    "variant_name": VARIANT_NAMES.get(v, str(v)),
                    "head_depth": h,
                    "split_type": s,
                    "n_runs": len(sub),
                    "auroc_mean": sub["auroc"].mean(),
                    "auroc_std": sub["auroc"].std(),
                    "auprc_mean": sub["auprc"].mean() if "auprc" in sub.columns else float("nan"),
                    "mcc_mean": sub["mcc"].mean() if "mcc" in sub.columns else float("nan"),
                    "bal_acc_mean": sub["balanced_acc"].mean() if "balanced_acc" in sub.columns else float("nan"),
                })

    df_summary = pd.DataFrame(summary_rows)

    # -------------------------------------------------------------------
    # Scaffold-split top-line (variants 1, 2, 3, 7 across all heads)
    # -------------------------------------------------------------------
    print("\nTop-line (scaffold split, mean across heads):")
    for v in [1, 2, 3, 7]:
        mask = (df_runs["variant"] == v) & (df_runs["split_type"] == "scaffold")
        sub = df_runs[mask]
        print(f"  var{v} ({VARIANT_NAMES[v][:25]}): AUROC={sub['auroc'].mean():.4f}±{sub['auroc'].std():.4f}")

    # -------------------------------------------------------------------
    # Write markdown summary
    # -------------------------------------------------------------------
    lines = [
        "# Phase 4 Ablation Summary — 7-Way Pathway Ablation DILI Classifier",
        "",
        f"**Date:** 2026-05-20",
        f"**Total runs:** {len(df_runs)}",
        f"**Grid:** 7 variants × 3 heads × 2 splits × 5 folds × 3 seeds = 630",
        "",
        "---",
        "",
        "## Halt Gate 3 (HG3)",
        "",
        f"**Check:** embed-only (var1, MolFormer 768d) random-split AUROC ≥ {HG3_THRESHOLD}",
        f"**Result:** mean={hg3_mean:.4f} ± {hg3_std:.4f} (n={hg3_n} runs)",
        f"**Verdict:** {'PASS ✓' if hg3_pass else 'HALT ✗ — chemistry shortcut baseline broken; consult before Phase 5'}",
        "",
        "---",
        "",
        "## Per-Variant Per-Head AUROC (mean ± std over 5 folds × 3 seeds = 15 runs)",
        "",
        "### Scaffold-Novel Split (primary — leakage discipline)",
        "",
        "| Var | Variant | Head | AUROC mean | AUROC std | AUPRC mean | Bal.Acc mean |",
        "|-----|---------|------|-----------|----------|-----------|------------|",
    ]

    for _, row in df_summary[df_summary["split_type"] == "scaffold"].iterrows():
        lines.append(
            f"| {int(row['variant'])} | {row['variant_name']} | {row['head_depth']} "
            f"| {row['auroc_mean']:.4f} | {row['auroc_std']:.4f} "
            f"| {row['auprc_mean']:.4f} | {row['bal_acc_mean']:.4f} |"
        )

    lines += [
        "",
        "### Random Split (comparability baseline)",
        "",
        "| Var | Variant | Head | AUROC mean | AUROC std | AUPRC mean | Bal.Acc mean |",
        "|-----|---------|------|-----------|----------|-----------|------------|",
    ]

    for _, row in df_summary[df_summary["split_type"] == "random"].iterrows():
        lines.append(
            f"| {int(row['variant'])} | {row['variant_name']} | {row['head_depth']} "
            f"| {row['auroc_mean']:.4f} | {row['auroc_std']:.4f} "
            f"| {row['auprc_mean']:.4f} | {row['bal_acc_mean']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Top-Line Preview (scaffold split, mean across all heads)",
        "",
        "| Variant | AUROC mean | AUROC std |",
        "|---------|-----------|----------|",
    ]

    for v in sorted(VARIANT_NAMES.keys()):
        mask = (df_runs["variant"] == v) & (df_runs["split_type"] == "scaffold")
        sub = df_runs[mask]
        if len(sub) == 0:
            continue
        lines.append(
            f"| var{v}: {VARIANT_NAMES[v]} | {sub['auroc'].mean():.4f} | {sub['auroc'].std():.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Feature Dimensions Reference",
        "",
        "| Var | Pathway | Input dim |",
        "|-----|---------|----------|",
        "| 1 | embed-only (MolFormer) | 768 |",
        "| 2 | gex-only (MODEL_GEX DE) | 919 |",
        "| 3 | dose-only (MODEL_DOSE E-Hill) | 1 |",
        "| 4 | embed+gex | 1687 |",
        "| 5 | embed+dose | 769 |",
        "| 6 | gex+dose | 920 |",
        "| 7 | all-three (headline) | 1688 |",
        "",
        "Note: feat_gex = 919-dim (not 978 as design doc stated; 919 landmark genes overlap "
        "between gene_vector.csv and lincs_train_safe.parquet with header=None).",
        "",
        "---",
        "",
        "## HG3 Full Details",
        "",
        "```",
        f"embed-only (var1) random-split AUROC: mean={hg3_mean:.4f} std={hg3_std:.4f} n={hg3_n}",
        f"HG3 threshold: {HG3_THRESHOLD}",
        f"HG3 verdict: {'PASS' if hg3_pass else 'HALT'}",
        "```",
        "",
    ]

    if not hg3_pass:
        lines += [
            "## HALT_REASON_3 (HG3 FAIL)",
            "",
            f"embed-only (var1=MolFormer) random-split AUROC = {hg3_mean:.4f} < threshold {HG3_THRESHOLD}.",
            "The chemistry-shortcut hypothesis is broken: MolFormer 768-dim embedding cannot beat 0.55 AUROC",
            "on a random split. This calls into question the whole three-pathway experimental framing.",
            "DO NOT proceed to Phase 5 until the user reviews this result.",
            "",
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote: {out_path}")

    # Write HALT_REASON_3.md if needed
    if not hg3_pass:
        halt_path = _PROJECT_ROOT / ".planning/phases/04-dili-consumer-ablation/HALT_REASON_3.md"
        halt_path.write_text(
            f"# HALT GATE 3 FIRED\n\n"
            f"**Date:** 2026-05-20\n"
            f"**Gate:** HG3 — embed-only (var1=MolFormer) random-split AUROC ≥ {HG3_THRESHOLD}\n\n"
            f"**Result:** mean={hg3_mean:.4f} ± {hg3_std:.4f} (n={hg3_n})\n\n"
            f"The chemistry-shortcut hypothesis is broken. MolFormer alone cannot achieve "
            f"AUROC ≥ 0.55 on a random split. This is below random chance plus a minimal margin, "
            f"indicating the molecular structure embedding is not capturing DILI signal.\n\n"
            f"**Action:** Stop. Do NOT execute Phase 5. Consult with user before proceeding.\n"
        )
        print(f"Wrote HALT_REASON_3.md")

    return hg3_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
