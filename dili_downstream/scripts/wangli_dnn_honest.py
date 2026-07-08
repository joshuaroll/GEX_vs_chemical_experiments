#!/usr/bin/env python
"""Honest Li/Tong 8-layer DNN on measured GEX, with an inflation-decomposition ablation.

Li/Tong 2020 (Front. Bioeng. Biotechnol., 10.3389/fbioe.2020.562677) report AUROC ~0.798 for a
fully-connected 8-layer DNN on 978 LINCS L1000 landmark genes. Their published code inflates that
number three ways, each of which this harness can turn on or off independently:

  (1) LEAKY PROFILE SPLIT  -- random profile-level 80/20; a drug's many profiles scatter across both
      sides, so 96.8% of test drugs are also in train and the per-drug label leaks. (--split profile)
  (2) TEST-SET SELECTION   -- they fit with validation_data=(X_test, y_test) + ModelCheckpoint(
      monitor='val_monitor_f', save_best_only=True), i.e. they keep the epoch that scores best ON THE
      TEST SET. Early stopping on the test set. (--selection test)
  (3) PROFILE-LEVEL SCORING -- roc_auc_score over every test PROFILE, counting each of a held-out
      drug's correlated profiles as an independent example (pseudoreplication). (--score profile)

The honest fix, and the axis this script exposes:
  --split {profile,drug,scaffold}  strict drug-disjoint / Murcko-scaffold-disjoint groups vs the leaky
                                    profile split.
  --selection {test,honest}        honest = early-stop / keep-best-epoch by AUROC on a group-disjoint
                                    VALIDATION set carved from the TRAIN groups (disjoint from BOTH
                                    train and test); test = mimic Li/Tong and select on the test set.
  --score {profile,drug}           drug = one mean-probability prediction per compound before AUROC;
                                    profile = per-row AUROC.

The 5-config inflation decomposition (each consecutive drop = one fix's contribution):
  C0  profile  | score profile | selection test    -> reproduces their ~0.78-0.80 headline
  C1  drug     | score profile | selection test     -> their "drug-based split" analog
  C2  drug     | score profile | selection honest   -> removes selection inflation
  C3  drug     | score drug     | selection honest   -> fully honest (expect ~0.59)
  C4  scaffold | score drug     | selection honest   -> strongest OOD

Comparability: this rides the SAME split + inner-validation + drug/profile-AUROC + permute-control
machinery as scripts/tox_finetune_multidcp.py (inner_split, auroc_profile_and_drug, the
GroupShuffleSplit(n_splits=N, test_size=0.2, random_state=0) drug/scaffold resamples, and the fixed
usage=='Training'/'Test' profile split), copied verbatim below WITH ATTRIBUTION, so numbers merge
directly with results/tables/P_tox_finetune_multidcp_v3.csv. It uses the SAME data file and the SAME
5,141-profile subset (all 621 SMILES featurize under MultiDCP's convert_smile_to_feature, so v3's
featurizability filter drops nothing -> the full 5,141 profiles, in the same row order, are the subset).
Only the classifier changes: the small ToxHead becomes Li/Tong's faithful 8-layer DNN.

Architecture (per shared context / their DNN.ipynb create_model with the BatchNorm+Dropout variant):
  978 -> Dense(512,ELU) -> BatchNorm -> Dropout(0.2)
      -> Dense(256,ELU) -> BatchNorm -> Dropout(0.2)
      -> Dense(128,ELU) -> BatchNorm
      -> Dense(64,ELU) -> Dense(32,ELU) -> Dense(16,ELU) -> Dense(8,ELU) -> Dense(1) [sigmoid via BCE].
  Loss BCE, optimizer Adam, balanced class weights (pos_weight = n_neg/n_pos on the train pool).
  NOTE: the published DNN.ipynb weights that reproduce 0.798 are pure Dense/ELU (no BatchNorm/Dropout);
  --arch plain builds that variant if the BatchNorm+Dropout default underperforms on the C0 gate.

CUDA hygiene: --gpu is parsed and CUDA_VISIBLE_DEVICES is set BEFORE `import torch`. --gpu auto (default)
auto-picks the freest GPU via nvidia-smi (we occupy exactly one, leaving the rest free).

Real data only. Does not commit; writes CSVs under results/tables/. Does not import or alter
tox_finetune_multidcp.py.
"""
from __future__ import annotations
import argparse, os, sys, copy, subprocess


def pick_gpu() -> str:
    """Freest GPU by used memory (we occupy exactly one, leaving the others free)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True)
        used = {}
        for line in out.strip().splitlines():
            idx, mem = line.split(",")
            used[int(idx.strip())] = int(mem.strip())
        if not used:
            return "0"
        return str(min(used, key=lambda g: used[g]))
    except Exception:
        return "0"


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--gpu", default="auto",
                help="CUDA device index, or 'auto' to pick the freest GPU via nvidia-smi (default)")
ap.add_argument("--split", default="profile,drug,scaffold",
                help="comma list of {profile,drug,scaffold} (profile = Li/Tong's leaky usage split)")
ap.add_argument("--score", default="profile,drug",
                help="comma list of {profile,drug}; drug aggregates to one mean-prob per compound")
ap.add_argument("--selection", default="test,honest",
                help="comma list of {test,honest}; test = select best epoch on the TEST set (Li/Tong), "
                     "honest = select on a group-disjoint val carved from TRAIN groups")
ap.add_argument("--seeds", type=int, default=3, help="repeats on the fixed profile Test split (>=3)")
ap.add_argument("--n-resamples", type=int, default=10,
                help="resampled drug/scaffold-disjoint GroupShuffleSplits (matches v3)")
ap.add_argument("--arch", choices=["bn", "plain"], default="bn",
                help="bn = shared-context 8-layer DNN with BatchNorm+Dropout(0.2) (default); "
                     "plain = published DNN.ipynb pure Dense/ELU (the variant that hits ~0.798)")
ap.add_argument("--batch-size", type=int, default=256, help="minibatch size (matches the validated retrain)")
ap.add_argument("--max-epochs", type=int, default=100)
ap.add_argument("--patience", type=int, default=10, help="early-stop patience on the selection AUROC")
ap.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
ap.add_argument("--permute-labels", action="store_true",
                help="negative control: shuffle the per-drug label map; every honest arm should give ~0.5")
ap.add_argument("--smoke", action="store_true", help="tiny build+train sanity check, no CSV written")
ap.add_argument("--tag", default="", help="suffix for the output CSV filenames")
args = ap.parse_args()

if args.gpu == "auto":
    args.gpu = pick_gpu()
    print(f"[gpu] auto-picked freest GPU {args.gpu}")
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # BEFORE importing torch (CUDA hygiene)

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score

REPO = Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/dili_downstream")
NPZ = REPO / "data/processed/wangli_multidcp_finetune.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# C0-C4 labels for the inflation-decomposition figure; anything else is a "custom" grid cell.
CONFIG_LABELS = {
    ("profile", "profile", "test"): "C0",
    ("drug", "profile", "test"): "C1",
    ("drug", "profile", "honest"): "C2",
    ("drug", "drug", "honest"): "C3",
    ("scaffold", "drug", "honest"): "C4",
}


# ============================================================================================
# COPIED VERBATIM (with attribution) from scripts/tox_finetune_multidcp.py so the honest DNN
# rides the identical split / inner-validation / scoring / permute machinery as the v3 numbers.
# ============================================================================================
def inner_split(tr, y, groups, seed, frac=0.15):
    """Carve an early-stopping validation set from tr.

    [VERBATIM from tox_finetune_multidcp.py:308-329]

    Group-disjoint when `groups` is given (val holds out whole compounds/scaffolds) so val AUROC
    tracks generalization rather than within-group memorization -> early stopping fires meaningfully
    and finetunes don't grind to the epoch cap. Random stratified for the profile-level split, where
    leakage is inherent to the split design. Falls back to random if a group holdout leaves one class
    empty on either side.
    """
    if groups is None:
        return train_test_split(tr, test_size=frac, random_state=seed, stratify=y[tr])
    g = groups[tr]
    uniq = np.unique(g)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(frac * len(uniq))))
    val_groups = set(uniq[:n_val].tolist())
    va_mask = np.array([x in val_groups for x in g])
    va, tr2 = tr[va_mask], tr[~va_mask]
    if len(va) == 0 or len(tr2) == 0 or len(np.unique(y[va])) < 2 or len(np.unique(y[tr2])) < 2:
        return train_test_split(tr, test_size=frac, random_state=seed, stratify=y[tr])
    return tr2, va


def auroc_profile_and_drug(y_te, pred_te, drugs_te):
    """Profile-level AUROC (paper's unit) and drug-level AUROC (mean prediction per drug, one label
    per drug -- the honest per-drug unit since DILI is a drug property). NaN if a level has one class.

    [VERBATIM from tox_finetune_multidcp.py:332-339]
    """
    prof = roc_auc_score(y_te, pred_te) if len(np.unique(y_te)) > 1 else np.nan
    df = pd.DataFrame({"d": drugs_te, "p": pred_te, "y": y_te})
    g = df.groupby("d").agg(p=("p", "mean"), y=("y", "first"))
    drug = roc_auc_score(g["y"].values, g["p"].values) if g["y"].nunique() > 1 else np.nan
    return prof, drug
# ============================================================================================
# END verbatim block.
# ============================================================================================


class WangLiDNN(nn.Module):
    """Li/Tong's faithful 8-layer DNN. Emits a raw logit; sigmoid is applied downstream (via
    BCEWithLogitsLoss in training, and explicitly for the probability we score/aggregate)."""

    def __init__(self, in_dim=978, arch="bn"):
        super().__init__()
        dims = [512, 256, 128, 64, 32, 16, 8]
        # per shared-context arch: BatchNorm after the first 3 ELUs, Dropout(0.2) after the first 2.
        bn_after = {512, 256, 128} if arch == "bn" else set()
        drop_after = {512, 256} if arch == "bn" else set()
        layers = []
        prev = in_dim
        for h in dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            if h in bn_after:
                layers.append(nn.BatchNorm1d(h))
            if h in drop_after:
                layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_dnn8(Xtr, ytr, Xva, yva, Xte, seed):
    """Train the 8-layer DNN on (Xtr,ytr); early-stop / keep the best epoch by AUROC on (Xva,yva);
    return sigmoid PROBABILITIES for Xte.

    Balanced class weights via BCEWithLogitsLoss(pos_weight = n_neg/n_pos) on the train pool.
    The caller decides what (Xva,yva) is: for honest selection it is a group-disjoint val carved
    from the train groups (disjoint from both train and test); for test-set selection it is (Xte,yte),
    reproducing Li/Tong's validation_data=(X_test,y_test) + best-checkpoint behaviour.
    """
    torch.manual_seed(seed)
    np_rng = np.random.RandomState(seed)
    model = WangLiDNN(Xtr.shape[1], arch=args.arch).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_pos = float((ytr == 1).sum())
    n_neg = float((ytr == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)

    bs = args.batch_size
    n = Xtr_t.shape[0]
    best, best_state, bad = -1.0, None, 0
    for ep in range(args.max_epochs):
        model.train()
        order = np_rng.permutation(n)
        for s in range(0, n, bs):
            bb = order[s:s + bs]
            if len(bb) < 2:
                continue  # BatchNorm needs >1 sample in train mode; skip a size-1 trailing minibatch
            idx = torch.as_tensor(bb, device=DEVICE)
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_logit = model(Xva_t).cpu().numpy()
        try:
            va_auc = roc_auc_score(yva, va_logit) if len(np.unique(yva)) > 1 else -1.0
        except ValueError:
            va_auc = -1.0
        if va_auc > best + 1e-4:
            best, best_state, bad = va_auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte_t)).cpu().numpy()


def load_data():
    d = np.load(NPZ, allow_pickle=True)
    D = {k: d[k] for k in d.files}
    y = D["label"].astype(int)
    N = len(y)
    if args.permute_labels:
        # negative control, shuffle labels at the DRUG level (preserve one-label-per-drug) so any
        # above-chance score reveals leakage. [logic copied from tox_finetune_multidcp.py:355-365]
        rng = np.random.RandomState(0)
        udrugs = np.unique(D["compound_name"])
        drug_lab = {dd: int(y[D["compound_name"] == dd][0]) for dd in udrugs}
        shuffled = rng.permutation(np.array([drug_lab[dd] for dd in udrugs]))
        newmap = dict(zip(udrugs, shuffled))
        y = np.array([newmap[dd] for dd in D["compound_name"]], dtype=int)
        D["label"] = y
        print("[permute] labels shuffled at drug level (negative control; expect ~0.5 everywhere)")
    _mnan = int(np.isnan(D["measured_modz"]).any(axis=1).sum())
    if _mnan:
        print(f"[WARN] measured arm: {_mnan}/{N} profiles have NaN MODZ -> zero-filled.")
    measured = np.nan_to_num(D["measured_modz"].astype(np.float32))
    print(f"[data] N={N} pos={int(y.sum())} neg={int((1 - y).sum())} "
          f"cells={len(set(D['cell_id']))} compounds={len(set(D['compound_name']))} arch={args.arch}")
    return D, y, measured, N


def build_resamples(split, D, y, N):
    """Resampled (seed, tr, te, split_groups) tuples for one split axis.
    [split-building logic copied from tox_finetune_multidcp.py:458-471]"""
    if split == "profile":
        # Li/Tong's fixed profile-level Training/Test split; repeat over seeds for a small CI vs 0.798.
        tr_all = np.where(D["usage"] == "Training")[0]
        te_all = np.where(D["usage"] == "Test")[0]
        return [(s, tr_all, te_all, None) for s in range(args.seeds)]
    # drug- or scaffold-disjoint: N resampled GroupShuffleSplits (held-out 20% of groups).
    split_groups = D["compound_name"] if split == "drug" else D["scaffold"]
    gss = GroupShuffleSplit(n_splits=args.n_resamples, test_size=0.2, random_state=0)
    return [(s, tr, te, split_groups) for s, (tr, te) in enumerate(gss.split(np.zeros(N), y, split_groups))]


def run_one(split, selection, measured, y, drugs, resamples):
    """Train + score every resample for one (split, selection). Returns per-resample
    (profile_auroc, drug_auroc) lists. Both metrics come from the SAME predictions."""
    profs, drugaucs = [], []
    for s, tr, te, split_groups in resamples:
        if selection == "honest":
            # group-disjoint val carved from TRAIN groups: disjoint from tr2 (holds out whole
            # groups) AND from te (val subset of tr, and tr is group-disjoint from te for drug/
            # scaffold). For the leaky profile split (groups=None) this is a random stratified carve.
            tr2, va = inner_split(tr, y, split_groups, s)
            train_idx, val_idx = tr2, va
        else:  # test  -- Li/Tong: train on the full train pool, select best epoch on the TEST set
            train_idx, val_idx = tr, te
        pred = train_dnn8(measured[train_idx], y[train_idx], measured[val_idx], y[val_idx],
                          measured[te], s)
        prof, drug = auroc_profile_and_drug(y[te], pred, drugs[te])
        profs.append(prof)
        drugaucs.append(drug)
    return profs, drugaucs


def write(rows, per_rows):
    tag = f"_{args.tag}" if args.tag else ""
    out = REPO / f"results/tables/P_wangli_dnn_honest{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(per_rows).to_csv(
        REPO / f"results/tables/P_wangli_dnn_honest{tag}_per_resample.csv", index=False)


def smoke(measured, y, drugs, D, N):
    print("[smoke] building 8-layer DNN and running a 3-epoch drug-split sanity check ...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr, te = next(gss.split(np.zeros(N), y, D["compound_name"]))
    tr2, va = inner_split(tr, y, D["compound_name"], 0)
    _me, _mp = args.max_epochs, args.patience
    args.max_epochs, args.patience = 3, 3
    pred = train_dnn8(measured[tr2], y[tr2], measured[va], y[va], measured[te], 0)
    args.max_epochs, args.patience = _me, _mp
    prof, drug = auroc_profile_and_drug(y[te], pred, drugs[te])
    print(f"[smoke] pred.shape={pred.shape} finite={np.isfinite(pred).all()} "
          f"range=[{pred.min():.3f},{pred.max():.3f}]  profile_auroc={prof:.3f} drug_auroc={drug:.3f}")
    print(f"[smoke] train={len(tr2)} val={len(va)} test={len(te)}  "
          f"val disjoint from train (groups)={set(D['compound_name'][tr2]).isdisjoint(set(D['compound_name'][va]))} "
          f"val disjoint from test (groups)={set(D['compound_name'][va]).isdisjoint(set(D['compound_name'][te]))}")


def main():
    D, y, measured, N = load_data()
    drugs = D["compound_name"]
    if args.smoke:
        smoke(measured, y, drugs, D, N)
        return

    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    scores = [s.strip() for s in args.score.split(",") if s.strip()]
    selections = [s.strip() for s in args.selection.split(",") if s.strip()]

    rows, per_rows = [], []
    for split in splits:
        resamples = build_resamples(split, D, y, N)
        for selection in selections:
            profs, drugaucs = run_one(split, selection, measured, y, drugs, resamples)
            for s, (p, dd) in enumerate(zip(profs, drugaucs)):
                per_rows.append(dict(split=split, selection=selection, feature="measured",
                                     resample=s, profile_auroc=p, drug_auroc=dd))
            prof_mean, prof_sd = float(np.nanmean(profs)), float(np.nanstd(profs))
            drug_mean, drug_sd = float(np.nanmean(drugaucs)), float(np.nanstd(drugaucs))
            for score in scores:
                head_mean = drug_mean if score == "drug" else prof_mean
                head_sd = drug_sd if score == "drug" else prof_sd
                rows.append(dict(
                    config=CONFIG_LABELS.get((split, score, selection), "custom"),
                    split=split, score=score, selection=selection, feature="measured",
                    n_resamples=len(resamples),
                    auroc_mean=head_mean, auroc_sd=head_sd,
                    profile_auroc_mean=prof_mean, profile_auroc_sd=prof_sd,
                    drug_auroc_mean=drug_mean, drug_auroc_sd=drug_sd,
                    permute=bool(args.permute_labels)))
                print(f"  [{rows[-1]['config']:>6}] split={split:8s} score={score:7s} "
                      f"sel={selection:6s}  headline={head_mean:.4f}±{head_sd:.4f}  "
                      f"(profile={prof_mean:.4f} drug={drug_mean:.4f}, n={len(resamples)})", flush=True)
            write(rows, per_rows)  # incremental save so a late crash preserves completed cells
    write(rows, per_rows)
    print(f"[done] wrote results/tables/P_wangli_dnn_honest"
          f"{('_' + args.tag) if args.tag else ''}.csv (+ _per_resample.csv)")


if __name__ == "__main__":
    main()
