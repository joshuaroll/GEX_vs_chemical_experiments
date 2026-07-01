#!/usr/bin/env python
"""Diagnostic: is MultiDCP's PREDICTED DE carrying the same toxicity signal as
JUST the MEASURED DE? On the SAME drugs (tox-labeled drugs that also have measured
LINCS DE in the organ corpus), compare three toxicity features through the same
small fixed head:

  structure    = ChemBERTa
  predicted DE = organ engine output: engine(smiles, organ_basal) - organ_basal  [978]
  measured DE  = the real per-drug mean (x1 - x2) from the organ LINCS corpus     [978]

If measured DE >> predicted DE, the engine is losing toxicity signal (the model is
poorly connected). If predicted ~ measured, the engine faithfully reproduces it.
Drug-disjoint 5-fold x 5 seeds; AUROC / AUPRC / accuracy / balanced-acc / F1.
"""
from __future__ import annotations
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="1")
ap.add_argument("--arch", default="multidcp")
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
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             balanced_accuracy_score, f1_score)
from scripts.three_way_comparison import load_liver, load_kidney, smi2ikey14
from src.spatial.structure_encoders import get_encoder

LOADERS = {"liver": load_liver, "kidney": load_kidney}
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

    def fwd(d, m, basal):
        dose = torch.ones(basal.shape[0], 1, dtype=torch.float64, device=DEVICE)
        o = model(d, gene_t, m, basal, dose, epoch=0)
        return o[0] if isinstance(o, tuple) else o
    return model, feat, fwd


def _ok(feat, s):
    try:
        feat([s]); return True
    except Exception:
        return False


@torch.no_grad()
def predicted_de(organ, smis):
    d = np.load(f"{REPO}/data/processed/organ_train/{organ}.npz", allow_pickle=True)
    basal_vec = d["x2"].astype(np.float64).mean(0)
    model, feat, fwd = build_engine()
    model.load_state_dict(torch.load(f"{CKPT_DIR}/{args.arch}_{organ}_{args.encoder}_sd42.pt",
                                     map_location=DEVICE)); model.eval()
    out = np.full((len(smis), 978), np.nan, np.float64)
    for i, s in enumerate(smis):
        if not _ok(feat, s):
            continue
        drug, mask = feat([s])
        basal = torch.as_tensor(basal_vec[None], dtype=torch.float64, device=DEVICE)
        out[i] = (fwd(drug, mask, basal).cpu().numpy()[0] - basal_vec)
    return out


def measured_de_map(organ):
    """ikey14 -> mean measured DE (x1-x2) over that drug's organ LINCS profiles."""
    d = np.load(f"{REPO}/data/processed/organ_train/{organ}.npz", allow_pickle=True)
    de = (d["x1"].astype(np.float64) - d["x2"].astype(np.float64))
    ik = np.array([smi2ikey14(s) or "NA" for s in d["smiles"]])
    m = {}
    for k in np.unique(ik):
        if k == "NA":
            continue
        m[k] = de[ik == k].mean(0)
    return m


def cv_oof(X, y, seed):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def scores(y, X):
    out = {m: [] for m in ("auroc", "auprc", "acc", "bal_acc", "f1")}
    for s in range(N_SEEDS):
        oof = cv_oof(X, y, s); pred = (oof >= 0.5).astype(int)
        out["auroc"].append(roc_auc_score(y, oof)); out["auprc"].append(average_precision_score(y, oof))
        out["acc"].append(accuracy_score(y, pred)); out["bal_acc"].append(balanced_accuracy_score(y, pred))
        out["f1"].append(f1_score(y, pred))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}


def main():
    enc = get_encoder("chemberta")
    rows = []
    for organ in args.organs.split(","):
        tox = LOADERS[organ]().dropna(subset=["smiles"]).copy()
        tox["ik"] = tox["smiles"].map(smi2ikey14)
        tox = tox.dropna(subset=["ik"]).drop_duplicates("ik").reset_index(drop=True)
        mde = measured_de_map(organ)
        tox = tox[tox["ik"].isin(mde)].reset_index(drop=True)  # only drugs with measured DE
        smis = tox["smiles"].tolist()
        Xpred = predicted_de(organ, smis)
        Semb, oks = enc(smis); Semb, oks = np.asarray(Semb), np.asarray(oks, bool)
        good = oks & np.isfinite(Xpred).all(1)
        tox, smis = tox[good].reset_index(drop=True), [s for s, g in zip(smis, good) if g]
        y = tox["label"].to_numpy(int)
        feats = {"structure": Semb[good].astype(float),
                 "predicted DE": Xpred[good].astype(float),
                 "measured DE": np.vstack([mde[k] for k in tox["ik"]]).astype(float)}
        for name, X in feats.items():
            sc = scores(y, X)
            rows.append(dict(organ=organ, n=len(y), pos=int(y.sum()), neg=int((1 - y).sum()),
                             feature=name, **{k: v[0] for k, v in sc.items()},
                             **{k + "_sd": v[1] for k, v in sc.items()}))
            print(f"[{organ}/{name}] n={len(y)} AUROC={sc['auroc'][0]:.3f}±{sc['auroc'][1]:.3f} "
                  f"AUPRC={sc['auprc'][0]:.3f} bal_acc={sc['bal_acc'][0]:.3f} F1={sc['f1'][0]:.3f}")

    df = pd.DataFrame(rows)
    L = ["# Predicted DE vs JUST measured DE vs structure — same drugs (overlap set)", "",
         "Only tox drugs that ALSO have measured LINCS DE in the organ corpus, so predicted and "
         "measured DE are scored on the IDENTICAL drug set. Same small head (L2 logreg, balanced), "
         "drug-disjoint 5-fold x 5 seeds. predicted DE = engine(smiles, organ_basal) - organ_basal; "
         "measured DE = per-drug mean (x1 - x2).", "",
         "| Organ | n (pos/neg) | feature | AUROC | AUPRC | accuracy | balanced acc | F1 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        oc = r["organ"] if r["feature"] == "structure" else ""
        nc = f"{r['n']} ({r['pos']}/{r['neg']})" if r["feature"] == "structure" else ""
        L.append(f"| {oc} | {nc} | {r['feature']} | {r['auroc']:.3f}±{r['auroc_sd']:.3f} | "
                 f"{r['auprc']:.3f} | {r['acc']:.3f} | {r['bal_acc']:.3f} | {r['f1']:.3f} |")
    L += ["", "## Reading",
          "- predicted vs measured DE on the SAME drugs: if close, the engine faithfully carries the "
          "toxicity-relevant DE signal (good connection); if measured >> predicted, the engine loses it.",
          "- Both vs structure: whether any DE (real or predicted) beats chemical structure for tox.", ""]
    from pathlib import Path
    Path(f"{REPO}/results/tables/P4_predicted_vs_measured_de.md").write_text("\n".join(L) + "\n")
    df.to_csv(f"{REPO}/results/tables/P4_predicted_vs_measured_de.csv", index=False)
    print("\nwrote results/tables/P4_predicted_vs_measured_de.md")


if __name__ == "__main__":
    main()
