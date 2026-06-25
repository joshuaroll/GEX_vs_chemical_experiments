#!/usr/bin/env python3
"""Reproduce the Wang/Li 2020 DILI DNN benchmark by retraining, and contrast the
published (leaky) profile-level split against a drug-disjoint split.

Two evaluations, SAME architecture, SAME training recipe:

  1. Published leaky split: train on usage==Training, test on usage==Test.
     A drug's profiles can appear in BOTH sides -> drug-identity leakage.
     Report per-seed AUROC, mean +/- std, and 10-seed probability-ensemble AUROC.
     Target: ensemble in [0.78, 0.82] vs published 0.798.

  2. Drug-disjoint split: StratifiedGroupKFold(5) grouped by compound_name
     (lowercased) over the full 5517 profiles. No compound appears in both
     train and test. Collect out-of-fold predictions, report profile-level AUROC.
     Same 10-seed treatment (mean +/- std).

Architecture (DNN.ipynb / P2_wangli_reproduction.md):
  978 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 1, ELU hidden, sigmoid out.
  Adam, BCE, balanced class weights. Up to 100 epochs, EarlyStopping on val_loss
  (patience 5), select best-val-loss epoch. Input = 978-d Bayesian COMPZ DE as-is.

CUDA hygiene: CUDA_VISIBLE_DEVICES is set BEFORE importing torch (see bottom).
Avoid GPU 1 (busy). Auto-pick least-used among {2,3} by default.

Real data only:
  data/processed/wangli_measured_de.npy   (5517, 978) float32
  data/processed/wangli_profiles.csv      5517 rows, aligned row-for-row to the npy
both in the sibling dili_downstream project.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DILI_DOWNSTREAM = Path(
    "/raid/home/joshua/projects/GEX_vs_chemical_experiments/dili_downstream"
)
DE_PATH = DILI_DOWNSTREAM / "data" / "processed" / "wangli_measured_de.npy"
CSV_PATH = DILI_DOWNSTREAM / "data" / "processed" / "wangli_profiles.csv"

N_SEEDS = 10
MAX_EPOCHS = 100
PATIENCE = 5
N_FOLDS = 5
VAL_FRAC = 0.1  # held-out fraction of the training pool, for EarlyStopping
LAYER_DIMS = [512, 256, 128, 64, 32, 16, 8]


def pick_gpu() -> str:
    """Least-used GPU among {2,3} (never GPU 1; leave others free)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        used = {}
        for line in out.strip().splitlines():
            idx, mem = line.split(",")
            used[int(idx.strip())] = int(mem.strip())
        cands = [g for g in (2, 3) if g in used]
        if not cands:
            return "2"
        return str(min(cands, key=lambda g: used[g]))
    except Exception:
        return "2"


def build_model(torch, nn):
    layers = []
    prev = 978
    for h in LAYER_DIMS:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ELU())
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train_one(torch, nn, X_tr, y_tr, X_te, device, seed, monitor="loss"):
    """Train on (X_tr, y_tr) with an internal val split for EarlyStopping;
    return predicted probabilities for X_te. Balanced class weights via BCE
    pos_weight = n_neg / n_pos on the training pool.

    monitor: "loss" selects best epoch by min val BCE loss; "auroc" selects
    best epoch by max val AUROC (closer to Wang/Li's custom val_monitor_f)."""
    from sklearn.metrics import roc_auc_score
    g = torch.Generator().manual_seed(seed)
    n = X_tr.shape[0]
    perm = torch.randperm(n, generator=g).numpy()
    n_val = max(1, int(round(VAL_FRAC * n)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    Xtr = torch.tensor(X_tr[tr_idx], dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr[tr_idx], dtype=torch.float32, device=device).unsqueeze(1)
    Xval = torch.tensor(X_tr[val_idx], dtype=torch.float32, device=device)
    yval = torch.tensor(y_tr[val_idx], dtype=torch.float32, device=device).unsqueeze(1)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=device)

    n_pos = float((y_tr[tr_idx] == 1).sum())
    n_neg = float((y_tr[tr_idx] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)

    torch.manual_seed(seed)
    model = build_model(torch, nn).to(device)
    opt = torch.optim.Adam(model.parameters())
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    yval_np = y_tr[val_idx]
    batch = 256
    best_metric = float("inf") if monitor == "loss" else -float("inf")
    best_state = None
    bad = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        bperm = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0], batch):
            bi = bperm[i:i + batch]
            opt.zero_grad()
            loss = crit(model(Xtr[bi]), ytr[bi])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vlogits = model(Xval)
            vloss = crit(vlogits, yval).item()
            vprob = torch.sigmoid(vlogits).cpu().numpy().ravel()
        if monitor == "loss":
            improved = vloss < best_metric
            cur = vloss
        else:
            cur = roc_auc_score(yval_np, vprob)
            improved = cur > best_metric
        if improved:
            best_metric = cur
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xte)).cpu().numpy().ravel()
    return probs


def run_leaky(torch, nn, device, X, y, usage, monitor="loss"):
    from sklearn.metrics import roc_auc_score
    tr = usage == "Training"
    te = usage == "Test"
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]
    per_seed = []
    prob_stack = []
    for s in range(N_SEEDS):
        p = train_one(torch, nn, X_tr, y_tr, X_te, device, seed=1000 + s,
                      monitor=monitor)
        per_seed.append(float(roc_auc_score(y_te, p)))
        prob_stack.append(p)
    ens = np.mean(np.vstack(prob_stack), axis=0)
    ens_auc = float(roc_auc_score(y_te, ens))
    return {
        "per_seed": per_seed,
        "mean": float(np.mean(per_seed)),
        "std": float(np.std(per_seed)),
        "ensemble": ens_auc,
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
    }


def run_drug_disjoint(torch, nn, device, X, y, groups, monitor="loss"):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    per_seed = []
    for s in range(N_SEEDS):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                    random_state=2000 + s)
        oof = np.zeros(len(y), dtype=np.float64)
        for tr_idx, te_idx in sgkf.split(X, y, groups):
            p = train_one(torch, nn, X[tr_idx], y[tr_idx], X[te_idx],
                          device, seed=2000 + s, monitor=monitor)
            oof[te_idx] = p
        per_seed.append(float(roc_auc_score(y, oof)))
    return {
        "per_seed": per_seed,
        "mean": float(np.mean(per_seed)),
        "std": float(np.std(per_seed)),
        "n_folds": N_FOLDS,
        "n_total": int(len(y)),
        "n_groups": int(pd.Series(groups).nunique()),
    }


def main() -> int:
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--monitor", choices=["loss", "auroc"], default="auroc",
                    help="early-stop / best-epoch selection metric on internal val. "
                         "auroc mirrors Wang/Li's custom val_monitor_f (their DNN.ipynb "
                         "does not select on plain val_loss); loss is the literal "
                         "Keras EarlyStopping(val_loss) recipe.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"device={device}")

    X = np.load(DE_PATH).astype(np.float32)
    df = pd.read_csv(CSV_PATH)
    assert len(df) == X.shape[0] == 5517, (len(df), X.shape)
    y = df["dili_binary"].to_numpy().astype(np.int64)
    usage = df["usage"].to_numpy()
    groups = df["compound_name"].astype(str).str.lower().to_numpy()
    print(f"[info] X={X.shape} pos={int((y==1).sum())} neg={int((y==0).sum())} "
          f"compounds={pd.Series(groups).nunique()}")

    print(f"[run] leaky published split (monitor={args.monitor}) ...")
    leaky = run_leaky(torch, nn, device, X, y, usage, monitor=args.monitor)
    print(f"  mean={leaky['mean']:.4f}+/-{leaky['std']:.4f} "
          f"ensemble={leaky['ensemble']:.4f}")

    print(f"[run] drug-disjoint StratifiedGroupKFold (monitor={args.monitor}) ...")
    disj = run_drug_disjoint(torch, nn, device, X, y, groups, monitor=args.monitor)
    print(f"  mean={disj['mean']:.4f}+/-{disj['std']:.4f}")

    results = {"monitor": args.monitor, "leaky": leaky, "drug_disjoint": disj}
    out = args.out or str(Path(__file__).resolve().parent.parent
                          / "results" / "tables" / "P2_wangli_check_results.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    # CUDA hygiene: set CUDA_VISIBLE_DEVICES BEFORE importing torch.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = pick_gpu()
        print(f"[info] auto-picked GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
    sys.exit(main())
