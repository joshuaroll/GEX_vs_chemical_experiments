#!/usr/bin/env python3
"""
Phase 4 — 7-way pathway ablation DILI classifier grid driver.

Trains 630 small DILI classifiers:
  7 variants × 3 head depths × 2 splits × 5 folds × 3 seeds = 630

Feature parquet (1688 feature dims):
  feat_dose (1) + feat_gex_0..918 (919) + feat_embed_0..767 (768)

Each run emits a prediction parquet under data/processed/predictions/.
Aggregate run-level metrics written to data/processed/P4_runs.parquet.

Usage:
    python src/stage2/train_dili_classifier.py \
        --features data/processed/dili_features.parquet \
        --split    data/processed/dili_split.json \
        --output-dir data/processed/predictions/ \
        --n-workers 6 \
        --wandb-project joshroll/MultiDCP_multihead_dili \
        --wandb-group multihead_dili

    # Dry run (prints grid, no training):
    python src/stage2/train_dili_classifier.py --dry-run
"""
import sys, os, argparse, json, random, hashlib, time, logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# GPU: not needed for tiny classifiers, but keep the CLAUDE.md hygiene rule
# ---------------------------------------------------------------------------
for i, a in enumerate(sys.argv):
    if a == "--gpu" and i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
        break

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score,
)
import multiprocessing as mp
from functools import partial

# ---------------------------------------------------------------------------
# Add project root to path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from src.stage2.classifiers import build_head

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VARIANTS = {
    1: "embed-only",
    2: "gex-only",
    3: "dose-only",
    4: "embed+gex",
    5: "embed+dose",
    6: "gex+dose",
    7: "all-three",
}
HEADS = ["linear", "mlp1", "mlp2"]
SPLITS = ["scaffold", "random"]
N_FOLDS = 5
SEEDS = [0, 1, 2]

# Locked HPs (Phase 4 spec)
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------
def set_seed(variant: int, head: str, split: str, fold: int, seed: int):
    # Hash the tuple to a stable integer seed
    key = f"v{variant}_{head}_{split}_f{fold}_s{seed}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 31)
    random.seed(h)
    np.random.seed(h)
    torch.manual_seed(h)
    return h


# ---------------------------------------------------------------------------
# Build variant→column-name list from the loaded parquet
# ---------------------------------------------------------------------------
def build_variant_cols(df: pd.DataFrame) -> dict:
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    embed_cols = sorted(c for c in feat_cols if c.startswith("feat_embed"))
    gex_cols   = sorted(c for c in feat_cols if c.startswith("feat_gex"))
    dose_cols  = [c for c in feat_cols if c == "feat_dose"]

    return {
        1: embed_cols,               # 768
        2: gex_cols,                 # 919
        3: dose_cols,                # 1
        4: embed_cols + gex_cols,    # 1687
        5: embed_cols + dose_cols,   # 769
        6: gex_cols + dose_cols,     # 920
        7: feat_cols,                # 1688 (all, in parquet column order)
    }


# ---------------------------------------------------------------------------
# Build scaffold split folds
# ---------------------------------------------------------------------------
def _parse_dilist_id(x) -> int:
    """Convert DILIST_XXXX string or plain int to int."""
    if isinstance(x, str):
        return int(x.replace("DILIST_", ""))
    return int(x)


def build_scaffold_folds(df: pd.DataFrame, split_json: dict, n_folds: int = 5):
    """
    Returns a list of n_folds (train_idx, val_idx, test_idx) tuples.
    test set is always the fixed held-out set from dili_split.json.
    CV is over train+val.

    dili_split.json uses DILIST_XXXX string IDs; parquet uses integer pert_id.
    """
    test_ids = {_parse_dilist_id(x) for x in split_json["test"]}
    trainval_ids = ({_parse_dilist_id(x) for x in split_json["train"]} |
                    {_parse_dilist_id(x) for x in split_json["val"]})

    df_tv = df[df["pert_id"].isin(trainval_ids)].copy()
    df_test = df[df["pert_id"].isin(test_ids)].copy()

    test_idx = df_test.index.tolist()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = []
    for fold_i, (tr, va) in enumerate(skf.split(df_tv, df_tv["dili_binary"])):
        tr_idx = df_tv.iloc[tr].index.tolist()
        va_idx = df_tv.iloc[va].index.tolist()
        folds.append((tr_idx, va_idx, test_idx))
    return folds


# ---------------------------------------------------------------------------
# Build random split folds
# ---------------------------------------------------------------------------
def build_random_folds(df: pd.DataFrame, n_folds: int = 5):
    """80/10/10 random stratified split, 5 folds."""
    from sklearn.model_selection import train_test_split

    folds = []
    for fold_i in range(n_folds):
        rng = 42 + fold_i
        idx_all = df.index.tolist()
        labels = df["dili_binary"].tolist()

        # First split: 80% train+val, 20% test
        tr_va_idx, te_idx = train_test_split(
            idx_all, test_size=0.2, stratify=labels, random_state=rng
        )
        labels_tv = df.loc[tr_va_idx, "dili_binary"].tolist()
        # Second split: 80/20 of trainval -> 64% train, 16% val (80/10/10 overall)
        tr_idx, va_idx = train_test_split(
            tr_va_idx, test_size=0.125, stratify=labels_tv, random_state=rng
        )
        folds.append((tr_idx, va_idx, te_idx))
    return folds


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def train_one_run(args: dict, wandb_run=None) -> dict:
    """
    Train one (variant, head, split, fold, seed) cell.
    Returns metrics dict.
    """
    variant = args["variant"]
    head_name = args["head"]
    split_type = args["split_type"]
    fold = args["fold"]
    seed = args["seed"]
    df = args["df"]
    variant_cols = args["variant_cols"]
    folds_scaffold = args["folds_scaffold"]
    folds_random = args["folds_random"]
    output_dir = Path(args["output_dir"])

    set_seed(variant, head_name, split_type, fold, seed)

    # --- Get fold split ---
    if split_type == "scaffold":
        tr_idx, va_idx, te_idx = folds_scaffold[fold]
    else:
        tr_idx, va_idx, te_idx = folds_random[fold]

    cols = variant_cols[variant]
    X = df[cols].values.astype(np.float32)  # [N, D]
    y = df["dili_binary"].values.astype(np.float32)

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    # Class weight for pos_weight in BCEWithLogitsLoss
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    input_dim = X_tr.shape[1]
    model = build_head(head_name, input_dim)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # DataLoaders
    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_count = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for X_b, y_b in tr_loader:
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            for X_b, y_b in va_loader:
                logits = model(X_b)
                val_losses.append(criterion(logits, y_b).item())
            val_loss = np.mean(val_losses)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on test set
    model.eval()
    X_te_t = torch.from_numpy(X_te)
    with torch.no_grad():
        logits_te = model(X_te_t).numpy()
    proba_te = 1.0 / (1.0 + np.exp(-logits_te))  # sigmoid
    pred_te = (proba_te >= 0.5).astype(int)

    # Metrics
    te_idx_list = te_idx if isinstance(te_idx, list) else list(te_idx)
    df_te = df.iloc[te_idx_list] if hasattr(te_idx, '__len__') else df.loc[te_idx]

    # Recompute y_te from df in case index ordering matters
    y_te_final = df.iloc[te_idx_list]["dili_binary"].values if isinstance(te_idx, list) else df.loc[te_idx, "dili_binary"].values

    auroc = roc_auc_score(y_te_final, proba_te) if len(np.unique(y_te_final)) >= 2 else float("nan")
    auprc = average_precision_score(y_te_final, proba_te) if len(np.unique(y_te_final)) >= 2 else float("nan")
    mcc   = matthews_corrcoef(y_te_final, pred_te)
    bal_acc = balanced_accuracy_score(y_te_final, pred_te)

    # Build output parquet
    df_pred = pd.DataFrame({
        "pert_id": df.iloc[te_idx_list]["pert_id"].values if isinstance(te_idx, list) else df.loc[te_idx, "pert_id"].values,
        "drug_name": df.iloc[te_idx_list]["drug_name"].values if isinstance(te_idx, list) else df.loc[te_idx, "drug_name"].values,
        "true_label": y_te_final,
        "predicted_proba": proba_te,
        "predicted_label": pred_te,
        "fold": fold,
        "seed": seed,
        "variant": variant,
        "head_depth": head_name,
        "split_type": split_type,
    })

    fname = f"var{variant}_head{head_name}_split{split_type}_fold{fold}_seed{seed}.parquet"
    df_pred.to_parquet(output_dir / fname, index=False)

    run_metrics = {
        "variant": variant,
        "variant_name": VARIANTS[variant],
        "head_depth": head_name,
        "split_type": split_type,
        "fold": fold,
        "seed": seed,
        "input_dim": input_dim,
        "auroc": auroc,
        "auprc": auprc,
        "mcc": mcc,
        "balanced_acc": bal_acc,
        "output_file": fname,
    }

    if wandb_run is not None:
        try:
            wandb_run.log({
                "variant": variant,
                "head_depth": head_name,
                "split_type": split_type,
                "fold": fold,
                "seed": seed,
                "auroc": auroc,
                "auprc": auprc,
                "mcc": mcc,
                "balanced_acc": bal_acc,
            })
        except Exception:
            pass

    return run_metrics


# ---------------------------------------------------------------------------
# Worker function for multiprocessing (no wandb in subprocess)
# ---------------------------------------------------------------------------
def _worker(args: dict) -> dict:
    """Worker entry point (called by Pool.imap)."""
    t0 = time.time()
    result = train_one_run(args, wandb_run=None)
    result["wall_sec"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 4 — 630-run DILI classifier grid")
    parser.add_argument("--features", default="data/processed/dili_features.parquet")
    parser.add_argument("--split", default="data/processed/dili_split.json")
    parser.add_argument("--output-dir", default="data/processed/predictions/")
    parser.add_argument("--n-workers", type=int, default=6)
    parser.add_argument("--wandb-project", default="joshroll/MultiDCP_multihead_dili")
    parser.add_argument("--wandb-group", default="multihead_dili")
    parser.add_argument("--dry-run", action="store_true", help="Print grid config and exit")
    parser.add_argument("--variant", type=int, default=None, help="Run only this variant (1-7)")
    parser.add_argument("--head", default=None, help="Run only this head depth")
    parser.add_argument("--gpu", default=None, help="GPU index (kept for CLAUDE.md compliance)")
    args = parser.parse_args()

    # Total grid
    total = len(VARIANTS) * len(HEADS) * len(SPLITS) * N_FOLDS * len(SEEDS)
    print(f"[P4] Grid: {len(VARIANTS)} variants × {len(HEADS)} heads × {len(SPLITS)} splits "
          f"× {N_FOLDS} folds × {len(SEEDS)} seeds = {total} runs")

    if args.dry_run:
        for v in VARIANTS:
            for h in HEADS:
                for s in SPLITS:
                    for f in range(N_FOLDS):
                        for seed in SEEDS:
                            print(f"  var{v}_{h}_{s}_fold{f}_seed{seed}")
        return

    # Load data
    print(f"[P4] Loading features from {args.features}")
    df = pd.read_parquet(args.features)
    print(f"[P4] Loaded: {df.shape}")

    with open(args.split) as fh:
        split_json = json.load(fh)

    variant_cols = build_variant_cols(df)
    for v, name in VARIANTS.items():
        print(f"  var{v} ({name}): {len(variant_cols[v])} dims")

    # Build folds (once, reused across all runs)
    print("[P4] Building scaffold folds...")
    folds_scaffold = build_scaffold_folds(df, split_json, N_FOLDS)
    print("[P4] Building random folds...")
    folds_random = build_random_folds(df, N_FOLDS)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter variants/heads if requested
    variants_to_run = [args.variant] if args.variant else list(VARIANTS.keys())
    heads_to_run = [args.head] if args.head else HEADS

    # Build job list
    jobs = []
    for v in variants_to_run:
        for h in heads_to_run:
            for s in SPLITS:
                for f in range(N_FOLDS):
                    for seed in SEEDS:
                        fname = f"var{v}_head{h}_split{s}_fold{f}_seed{seed}.parquet"
                        if (output_dir / fname).exists():
                            continue  # Skip already done (resume support)
                        jobs.append({
                            "variant": v,
                            "head": h,
                            "split_type": s,
                            "fold": f,
                            "seed": seed,
                            "df": df,
                            "variant_cols": variant_cols,
                            "folds_scaffold": folds_scaffold,
                            "folds_random": folds_random,
                            "output_dir": str(output_dir),
                        })

    print(f"[P4] {len(jobs)} runs to execute (skipping existing outputs)")

    # WandB init (main process only)
    wandb_run = None
    try:
        import wandb
        entity, project = args.wandb_project.split("/", 1)
        wandb_run = wandb.init(
            entity=entity,
            project=project,
            group=args.wandb_group,
            name="P4_ablation_grid",
            config={
                "n_variants": len(VARIANTS),
                "n_heads": len(HEADS),
                "n_splits": len(SPLITS),
                "n_folds": N_FOLDS,
                "n_seeds": len(SEEDS),
                "total_runs": total,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
            },
        )
    except Exception as e:
        print(f"[P4] WandB init failed (non-fatal): {e}")

    # Run grid
    t_start = time.time()
    all_metrics = []

    n_workers = min(args.n_workers, len(jobs), mp.cpu_count() - 1)
    n_workers = max(n_workers, 1)

    if n_workers > 1 and len(jobs) > 0:
        print(f"[P4] Parallel execution with {n_workers} workers")
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker, jobs)):
                all_metrics.append(result)
                if (i + 1) % 50 == 0 or (i + 1) == len(jobs):
                    elapsed = time.time() - t_start
                    print(f"  [{i+1}/{len(jobs)}] elapsed={elapsed:.0f}s "
                          f"last: var{result['variant']} {result['head_depth']} "
                          f"{result['split_type']} f{result['fold']} s{result['seed']} "
                          f"AUROC={result['auroc']:.4f}")
    else:
        print(f"[P4] Sequential execution")
        for i, job in enumerate(jobs):
            result = _worker(job)
            all_metrics.append(result)
            if (i + 1) % 50 == 0 or (i + 1) == len(jobs):
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{len(jobs)}] elapsed={elapsed:.0f}s "
                      f"AUROC={result['auroc']:.4f}")

    elapsed_total = time.time() - t_start
    print(f"[P4] Grid complete: {len(all_metrics)} runs in {elapsed_total:.0f}s "
          f"({elapsed_total/60:.1f} min)")

    # Also collect any pre-existing runs (for aggregate)
    existing_parquets = sorted(output_dir.glob("*.parquet"))
    print(f"[P4] Total prediction files: {len(existing_parquets)}")

    # Aggregate metrics
    if all_metrics:
        df_runs = pd.DataFrame(all_metrics)
        runs_path = Path(args.features).parent / "P4_runs.parquet"
        # Merge with any existing P4_runs.parquet
        if runs_path.exists():
            df_existing = pd.read_parquet(runs_path)
            df_runs = pd.concat([df_existing, df_runs], ignore_index=True)
            df_runs = df_runs.drop_duplicates(
                subset=["variant", "head_depth", "split_type", "fold", "seed"]
            )
        df_runs.to_parquet(runs_path, index=False)
        print(f"[P4] P4_runs.parquet: {len(df_runs)} rows -> {runs_path}")

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    print("[P4] DONE")


if __name__ == "__main__":
    main()
