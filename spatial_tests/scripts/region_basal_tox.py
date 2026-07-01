#!/usr/bin/env python
"""Basal-sensitivity of the organ engine + does basal-conditioning help toxicity?

The spatial premise is "combine drug response across locations", where location =
a measured basal context. Prerequisite: does the organ-trained engine's predicted
DE even RESPOND to the basal? We condition predicted DE for the tox drugs on several
REAL, in-distribution measured basals (per-cell-line mean x2, e.g. liver PHH vs
HEPG2; kidney HA1E vs HEK293 -- genuinely different tissue-relevant contexts, no OOD
problem), and measure (1) how correlated predicted DE is across basals (sensitivity),
(2) toxicity AUROC per basal, concat-across-basals, vs structure.

Note (honest): each per-basal predicted DE is still a function of SMILES (the basal is
drug-independent), so it is structure-ceilinged; this tests the spatial MECHANISM
(does context change the feature), not a way to beat structure.
"""
from __future__ import annotations
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="1")
ap.add_argument("--encoder", default="linear")
ap.add_argument("--organs", default="liver,kidney")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import pandas as pd
import torch

REPO = "/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests"
MULTIDCP = "/raid/home/joshua/projects/MultiDCP/MultiDCP"
GENE_VECTOR = f"{MULTIDCP}/data/gene_vector.csv"
CKPT_DIR = f"{REPO}/results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, REPO)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from scripts.three_way_comparison import load_liver, load_kidney, smi2ikey14
from src.spatial.structure_encoders import get_encoder

LOADERS = {"liver": load_liver, "kidney": load_kidney}
# basals per organ: top cell lines (real measured controls) + organ mean
BASAL_CELLS = {"liver": ["PHH", "HEPG2", "JHH5"], "kidney": ["HA1E", "HEK293"]}
N_SEEDS, N_FOLDS = 5, 5


def build_engine():
    sys.path[:0] = [f"{MULTIDCP}/models", f"{MULTIDCP}/utils"]
    import multidcp
    from data_utils import convert_smile_to_feature, create_mask_feature, read_gene
    from multidcp_ae_utils import initialize_model_registry
    reg = initialize_model_registry()
    reg.update({"num_gene": 978, "cell_id_input_dim": 978, "pert_idose_input_dim": 1,
                "dropout": 0.3, "linear_encoder_flag": args.encoder == "linear"})
    model = multidcp.MultiDCPOriginal(device=DEVICE, model_param_registry=reg).to(DEVICE).double()
    gene_t = read_gene(GENE_VECTOR, DEVICE)

    def feat(s):
        d = convert_smile_to_feature(list(s), DEVICE); return d, create_mask_feature(d, DEVICE)
    return model, feat, gene_t


def _ok(feat, s):
    try:
        feat([s]); return True
    except Exception:
        return False


def organ_basals(organ):
    d = np.load(f"{REPO}/data/processed/organ_train/{organ}.npz", allow_pickle=True)
    x2, cell = d["x2"].astype(np.float64), d["cell"]
    basals = {"organ_mean": x2.mean(0)}
    for c in BASAL_CELLS[organ]:
        if (cell == c).sum() >= 20:
            basals[c] = x2[cell == c].mean(0)
    return basals


@torch.no_grad()
def predicted_de_multi(organ, smis, basals):
    """Return {basal_name: predicted DE [n,978]} for all basals in ONE pass per drug."""
    model, feat, gene_t = build_engine()
    model.load_state_dict(torch.load(f"{CKPT_DIR}/multidcp_{organ}_{args.encoder}_sd42.pt",
                                     map_location=DEVICE)); model.eval()
    names = list(basals)
    B = {n: torch.as_tensor(basals[n][None], dtype=torch.float64, device=DEVICE) for n in names}
    out = {n: np.full((len(smis), 978), np.nan, np.float64) for n in names}
    for i, s in enumerate(smis):
        if not _ok(feat, s):
            continue
        drug, mask = feat([s])
        dose = torch.ones(1, 1, dtype=torch.float64, device=DEVICE)
        for n in names:
            pred = model(drug, gene_t, mask, B[n], dose, epoch=0)
            pred = (pred[0] if isinstance(pred, tuple) else pred)[0].cpu().numpy()
            out[n][i] = pred - basals[n]
    return out


def cv_auroc(X, y):
    aurs = []
    for seed in range(N_SEEDS):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
        aurs.append(roc_auc_score(y, oof))
    return float(np.mean(aurs)), float(np.std(aurs))


def main():
    enc = get_encoder("chemberta")
    L = ["# Basal-sensitivity of the organ engine + basal-conditioned toxicity", "",
         "Predicted DE conditioned on several REAL in-distribution measured basals (per-cell-line "
         "mean x2). Tests whether the organ-trained engine responds to basal context (the spatial "
         "'combine across locations' prerequisite) and whether it helps toxicity. Drug-disjoint 5x5.", ""]
    for organ in args.organs.split(","):
        df = LOADERS[organ]().dropna(subset=["smiles"]).copy()
        df["ik"] = df["smiles"].map(smi2ikey14)
        df = df.dropna(subset=["ik"]).drop_duplicates("ik").reset_index(drop=True)
        smis = df["smiles"].tolist()
        basals = organ_basals(organ)
        DEs = predicted_de_multi(organ, smis, basals)
        Semb, oks = enc(smis); Semb, oks = np.asarray(Semb), np.asarray(oks, bool)
        good = oks & np.all([np.isfinite(DEs[n]).all(1) for n in DEs], axis=0)
        y = df["label"].to_numpy(int)[good]
        S = Semb[good].astype(float)
        DEs = {n: DEs[n][good].astype(float) for n in DEs}
        names = list(DEs)

        # (1) basal sensitivity: mean per-sample Pearson of predicted DE across basal pairs
        print(f"\n[{organ}] basal-sensitivity of predicted DE (per-sample Pearson across basals):")
        L += [f"## {organ} — basal sensitivity (per-sample Pearson of predicted DE across basals)",
              "Closer to 1.000 = engine ignores the basal (spatial conditioning is moot).", "",
              "| basal A | basal B | mean per-sample Pearson |", "|---|---|---|"]
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                na, nb = names[a], names[b]
                r = np.mean([pearsonr(DEs[na][i], DEs[nb][i])[0] for i in range(len(y))])
                print(f"   {na:10s} vs {nb:10s}: {r:.4f}")
                L.append(f"| {na} | {nb} | {r:.4f} |")

        # (2) toxicity: structure, per-basal predicted DE, concat-across-basals
        print(f"[{organ}] toxicity AUROC:")
        feats = {"structure": S}
        for n in names:
            feats[f"predDE @ {n}"] = DEs[n]
        feats["predDE concat-all-basals"] = np.hstack([DEs[n] for n in names])
        L += ["", f"## {organ} — toxicity AUROC (n={len(y)}, {int(y.sum())}/{int((1-y).sum())})",
              "| feature | AUROC |", "|---|---|"]
        for n, X in feats.items():
            m, sd = cv_auroc(X, y)
            print(f"   {n:28s}: {m:.3f}±{sd:.3f}")
            L.append(f"| {n} | {m:.3f}±{sd:.3f} |")
        L.append("")

    L += ["## Reading",
          "- If predicted DE is ~collinear across very different basals (PHH vs HEPG2 / HA1E vs HEK293), "
          "the engine ignores basal context -> region-resolved conditioning cannot produce distinct "
          "location features, and the spatial 'combine across locations' arm is moot.",
          "- Even if basal-sensitive, per-basal predicted DE is a function of SMILES (drug-independent "
          "basal), so it stays structure-ceilinged; concat-across-basals cannot exceed structure.", ""]
    from pathlib import Path
    Path(f"{REPO}/results/tables/P4_region_basal_tox.md").write_text("\n".join(L) + "\n")
    print("\nwrote results/tables/P4_region_basal_tox.md")


if __name__ == "__main__":
    main()
