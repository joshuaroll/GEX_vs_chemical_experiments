#!/usr/bin/env python
"""Toxicity-tuned MultiDCP signature (design conditions E/F) vs frozen DE vs structure.

Honest caveat: a tox-tuned signature is still a function of SMILES, so a win shows the
gene-expression bottleneck is a useful INDUCTIVE BIAS, not that omics adds structure-independent
information. A loss shows the bottleneck adds nothing even when optimized for tox.

Variants (all share the SAME head: BatchNorm1d -> Linear(.,128) -> ReLU -> Dropout(0.3) -> Linear(128,1)):
  structure   ECFP4 -> head                         (baseline)
  frozen_DE   frozen engine DE -> head              (= the predicted-DE arm, SGD head)
  tuned_E     engine + head trained end-to-end on tox (engine gets tox gradient)
  tuned_F     tuned_E + anchor MSE(pred_treated, frozen pred_treated)  (keep signature biology-shaped)

Drug-disjoint and scaffold-disjoint 5-fold CV x N seeds; internal val early stopping (AUROC).
Halt gate: if tuned <= structure on both organs, the bottleneck adds nothing over structure.
"""
from __future__ import annotations
import argparse, os, sys, copy

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="1")
ap.add_argument("--organs", default="liver,kidney")
ap.add_argument("--splits", default="drug,scaffold")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--max-epochs", type=int, default=100)
ap.add_argument("--patience", type=int, default=15)
ap.add_argument("--anchor", type=float, default=1.0, help="tuned_F anchor weight")
ap.add_argument("--variants", default="structure,frozen_DE,tuned_E,tuned_F")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn

REPO = "/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests"
MULTIDCP = "/raid/home/joshua/projects/MultiDCP/MultiDCP"
GENE_VECTOR = f"{MULTIDCP}/data/gene_vector.csv"
CKPT_DIR = f"{REPO}/results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, REPO)
sys.path[:0] = [f"{MULTIDCP}/models", f"{MULTIDCP}/utils"]

import multidcp
from data_utils import convert_smile_to_feature, create_mask_feature, read_gene
from multidcp_ae_utils import initialize_model_registry
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from scripts.three_way_comparison import load_liver, load_kidney, smi2ikey14
from src.spatial.structure_encoders import get_encoder

LOADERS = {"liver": load_liver, "kidney": load_kidney}
GENE_T = read_gene(GENE_VECTOR, DEVICE)


def fresh_engine(organ):
    reg = initialize_model_registry()
    reg.update({"num_gene": 978, "cell_id_input_dim": 978, "pert_idose_input_dim": 1,
                "dropout": 0.3, "linear_encoder_flag": True})
    m = multidcp.MultiDCPOriginal(device=DEVICE, model_param_registry=reg).to(DEVICE).double()
    m.load_state_dict(torch.load(f"{CKPT_DIR}/multidcp_{organ}_linear_sd42.pt", map_location=DEVICE))
    return m


def can_feat(s):
    try:
        convert_smile_to_feature([s], DEVICE); return True
    except Exception:
        return False


class ToxHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm1d(in_dim), nn.Linear(in_dim, 128), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(128, 1)).double()

    def forward(self, x):
        return self.net(x).squeeze(1)


def engine_de(engine, smis, basal_vec):
    """Predicted DE [B,978] and predicted treated [B,978] for a SMILES list (one forward)."""
    d = convert_smile_to_feature(list(smis), DEVICE)
    m = create_mask_feature(d, DEVICE)
    B = len(smis)
    basal = torch.as_tensor(np.tile(basal_vec, (B, 1)), dtype=torch.float64, device=DEVICE)
    dose = torch.ones(B, 1, dtype=torch.float64, device=DEVICE)
    pred = engine(d, GENE_T, m, basal, dose)
    return pred - basal, pred


def train_static(Xtr, ytr, Xva, yva, Xte, seed):
    torch.manual_seed(seed)
    head = ToxHead(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float64, device=DEVICE)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float64, device=DEVICE)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float64, device=DEVICE)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float64, device=DEVICE)
    best, best_state, bad = -1, None, 0
    for ep in range(args.max_epochs):
        head.train(); opt.zero_grad()
        loss = lossf(head(Xtr_t), ytr_t); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            va = roc_auc_score(yva, head(Xva_t).cpu().numpy())
        if va > best + 1e-4:
            best, best_state, bad = va, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        return head(Xte_t).cpu().numpy()


BS = 16  # gene-gene attention is O(B * 978^2); full-batch OOMs, so mini-batch the engine


@torch.no_grad()
def predict_tuned(engine, head, smis, basal_vec):
    """Chunked head-logits for a SMILES list (no grad)."""
    out = []
    for i in range(0, len(smis), BS):
        de, _ = engine_de(engine, smis[i:i + BS], basal_vec)
        out.append(head(de).cpu().numpy())
    return np.concatenate(out)


def train_tuned(engine, tr_smis, ytr, va_smis, yva, te_smis, basal_vec, seed, anchor):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    head = ToxHead(978).to(DEVICE)
    opt = torch.optim.Adam([{"params": engine.parameters(), "lr": 1e-4},
                            {"params": head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    tr_smis = list(tr_smis)
    # frozen predicted-treated anchor, precomputed per-drug (chunked, no grad), indexed per minibatch
    if anchor > 0:
        with torch.no_grad():
            frozen_treated = torch.cat([engine_de(engine, tr_smis[i:i + BS], basal_vec)[1]
                                        for i in range(0, len(tr_smis), BS)]).detach()
    best, best_state, bad = -1, None, 0
    idx = np.arange(len(tr_smis))
    for ep in range(args.max_epochs):
        engine.train(); head.train()
        rng.shuffle(idx)
        for s in range(0, len(idx), BS):
            b = idx[s:s + BS]
            opt.zero_grad()
            de_b, treated_b = engine_de(engine, [tr_smis[i] for i in b], basal_vec)
            yb = torch.as_tensor(ytr[b], dtype=torch.float64, device=DEVICE)
            loss = lossf(head(de_b), yb)
            if anchor > 0:
                loss = loss + anchor * ((treated_b - frozen_treated[b]) ** 2).mean()
            loss.backward(); opt.step()
        engine.eval(); head.eval()
        va = roc_auc_score(yva, predict_tuned(engine, head, va_smis, basal_vec))
        if va > best + 1e-4:
            best, best_state, bad = va, (copy.deepcopy(engine.state_dict()),
                                         copy.deepcopy(head.state_dict())), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    engine.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    engine.eval(); head.eval()
    return predict_tuned(engine, head, te_smis, basal_vec)


def folds_iter(split, y, groups, seed):
    if split == "drug":
        yield from StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    else:
        yield from StratifiedGroupKFold(args.folds).split(np.zeros(len(y)), y, groups)


def main():
    ecfp = get_encoder("ecfp4")
    rows = []
    for organ in args.organs.split(","):
        df = LOADERS[organ]().dropna(subset=["smiles"]).copy()
        df["ik"] = df["smiles"].map(smi2ikey14)
        df = df.dropna(subset=["ik"]).drop_duplicates("ik").reset_index(drop=True)
        df = df[df["smiles"].map(can_feat)].reset_index(drop=True)
        if args.smoke:
            df = df.sample(120, random_state=0).reset_index(drop=True)
        smis = df["smiles"].tolist()
        y = df["label"].to_numpy(int)
        d = np.load(f"{REPO}/data/processed/organ_train/{organ}.npz", allow_pickle=True)
        basal_vec = d["x2"].astype(np.float64).mean(0)
        Xstruct = np.asarray(ecfp(smis)[0], float)
        scaf = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) or s
                         for s in smis])
        # frozen DE (static), computed once
        eng0 = fresh_engine(organ)
        eng0.eval()
        with torch.no_grad():
            DEfroz = np.vstack([engine_de(eng0, smis[i:i+128], basal_vec)[0].cpu().numpy()
                                for i in range(0, len(smis), 128)])
        print(f"\n[{organ}] n={len(y)} ({int(y.sum())}/{int((1-y).sum())})")

        for split in args.splits.split(","):
            for variant in args.variants.split(","):
                aurocs = []
                for seed in range(args.seeds):
                    oof = np.zeros(len(y))
                    for tr, te in folds_iter(split, y, scaf, seed):
                        tr2, va = train_test_split(tr, test_size=0.15, random_state=seed,
                                                   stratify=y[tr])
                        if variant in ("structure", "frozen_DE"):
                            X = Xstruct if variant == "structure" else DEfroz
                            oof[te] = train_static(X[tr2], y[tr2], X[va], y[va], X[te], seed)
                        else:
                            eng = fresh_engine(organ)
                            anchor = args.anchor if variant == "tuned_F" else 0.0
                            oof[te] = train_tuned(eng, [smis[i] for i in tr2], y[tr2],
                                                  [smis[i] for i in va], y[va],
                                                  [smis[i] for i in te], basal_vec, seed, anchor)
                            del eng; torch.cuda.empty_cache()
                    aurocs.append(roc_auc_score(y, oof))
                mu, sd = float(np.mean(aurocs)), float(np.std(aurocs))
                rows.append(dict(organ=organ, split=split, variant=variant, n=len(y),
                                 auroc=mu, auroc_sd=sd))
                print(f"  {split:8s} {variant:10s} AUROC={mu:.3f}±{sd:.3f}")

    write(rows)


def write(rows):
    df = pd.DataFrame(rows)
    L = ["# Tox-tuned MultiDCP signature (conditions E/F) vs frozen DE vs structure", "",
         "All variants share one head (BatchNorm -> 128 -> 1). structure=ECFP4; frozen_DE=frozen "
         "engine DE; tuned_E=engine+head end-to-end on tox; tuned_F=tuned_E + anchor to frozen "
         f"predicted-treated (w={args.anchor}). {args.folds}-fold x {args.seeds} seeds, internal-val "
         "early stopping. **Caveat:** tuned signature is still a function of SMILES; a win = useful "
         "bottleneck inductive bias, not structure-independent information.", ""]
    for split in sorted(df["split"].unique()):
        L += [f"## {split}-disjoint", "",
              "| organ | structure | frozen_DE | tuned_E | tuned_F |", "|---|---|---|---|---|"]
        for organ in df["organ"].unique():
            g = df[(df.split == split) & (df.organ == organ)].set_index("variant")
            cell = lambda v: (f"{g.loc[v,'auroc']:.3f}±{g.loc[v,'auroc_sd']:.3f}"
                              if v in g.index else "-")
            L.append(f"| {organ} | {cell('structure')} | {cell('frozen_DE')} | "
                     f"{cell('tuned_E')} | {cell('tuned_F')} |")
        L.append("")
    L += ["## Reading",
          "- tuned_E/F > structure on both organs and splits => the GEX bottleneck is a useful "
          "inductive bias for tox (worth pursuing).",
          "- tuned ~ structure => tox-tuning does not help; the bottleneck adds nothing over structure "
          "even when optimized for the endpoint (HALT GATE).",
          "- tuned > frozen_DE => tox gradient to the encoder helps vs frozen; isolates the tuning effect.", ""]
    Path(f"{REPO}/results/tables/P4_tox_tuned.md").write_text("\n".join(L) + "\n")
    df.to_csv(f"{REPO}/results/tables/P4_tox_tuned.csv", index=False)
    print("\nwrote results/tables/P4_tox_tuned.md")


if __name__ == "__main__":
    main()
