#!/usr/bin/env python
"""Connect the tox predictor to MultiDCP's INTERNAL representation, not its DE output.

The current stage-2 connection collapses MultiDCP to its 978-gene predicted DE. Here
we instead tap the penultimate per-gene hidden layer `out` = model.multidcp(...)[0]
[B,978,128] (right BEFORE relu+linear_final collapses it to the 978 DE scalars), pool
it over genes into a compact drug-effect embedding, and use THAT as the tox feature.

Features compared (same small head, drug-disjoint 5-fold x 5 seeds, full tox set):
  structure        = ChemBERTa
  predicted DE     = model output (linear_final(relu(out))) - basal           [978]
  internal mean    = mean over genes of relu(out)                             [128]
  internal meanmax = concat(mean, max) over genes of relu(out)                [256]
  structure+internal = concat
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
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             balanced_accuracy_score, f1_score)
from scripts.three_way_comparison import load_liver, load_kidney, smi2ikey14
from src.spatial.structure_encoders import get_encoder

LOADERS = {"liver": load_liver, "kidney": load_kidney}
N_SEEDS, N_FOLDS, N_BOOT = 5, 5, 10000


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


@torch.no_grad()
def engine_features(organ, smis):
    """Return predicted DE [n,978], internal mean [n,128], internal meanmax [n,256], ok mask."""
    d = np.load(f"{REPO}/data/processed/organ_train/{organ}.npz", allow_pickle=True)
    basal_vec = d["x2"].astype(np.float64).mean(0)
    model, feat, gene_t = build_engine()
    model.load_state_dict(torch.load(f"{CKPT_DIR}/multidcp_{organ}_{args.encoder}_sd42.pt",
                                     map_location=DEVICE)); model.eval()
    basal = torch.as_tensor(basal_vec[None], dtype=torch.float64, device=DEVICE)
    DE = np.full((len(smis), 978), np.nan, np.float64)
    IM = np.full((len(smis), 128), np.nan, np.float64)
    IMM = np.full((len(smis), 256), np.nan, np.float64)
    for i, s in enumerate(smis):
        if not _ok(feat, s):
            continue
        drug, mask = feat([s])
        out = model.multidcp(drug, gene_t, mask, basal, dose_t(1), epoch=0)[0]  # [1,978,128]
        h = model.relu(out)                                                     # activated penultimate
        de = model.linear_final(h).squeeze(-1)[0].cpu().numpy() - basal_vec     # [978]
        hm = h.mean(1)[0].cpu().numpy()                                          # [128]
        hx = h.max(1).values[0].cpu().numpy()                                    # [128]
        DE[i], IM[i], IMM[i] = de, hm, np.concatenate([hm, hx])
    return DE, IM, IMM


def dose_t(n):
    return torch.ones(n, 1, dtype=torch.float64, device=DEVICE)


def cv_oof(X, y, seed):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def scores(y, X):
    out = {m: [] for m in ("auroc", "auprc", "acc", "bal_acc", "f1")}; oof0 = None
    for s in range(N_SEEDS):
        oof = cv_oof(X, y, s); pred = (oof >= 0.5).astype(int)
        out["auroc"].append(roc_auc_score(y, oof)); out["auprc"].append(average_precision_score(y, oof))
        out["acc"].append(accuracy_score(y, pred)); out["bal_acc"].append(balanced_accuracy_score(y, pred))
        out["f1"].append(f1_score(y, pred))
        if s == 0: oof0 = oof
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}, oof0


def boot_lift(y, oa, ob, seed=0):
    rng = np.random.RandomState(seed); n = len(y); d = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        d.append(roc_auc_score(y[idx], ob[idx]) - roc_auc_score(y[idx], oa[idx]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    enc = get_encoder("chemberta")
    rows, lifts = [], []
    for organ in args.organs.split(","):
        df = LOADERS[organ]().dropna(subset=["smiles"]).copy()
        df["ik"] = df["smiles"].map(smi2ikey14)
        df = df.dropna(subset=["ik"]).drop_duplicates("ik").reset_index(drop=True)
        smis = df["smiles"].tolist()
        DE, IM, IMM = engine_features(organ, smis)
        Semb, oks = enc(smis); Semb, oks = np.asarray(Semb), np.asarray(oks, bool)
        good = oks & np.isfinite(DE).all(1)
        y = df["label"].to_numpy(int)[good]
        S, DE, IM, IMM = Semb[good].astype(float), DE[good], IM[good], IMM[good]
        feats = {"structure": S, "predicted DE": DE, "internal mean(128)": IM,
                 "internal meanmax(256)": IMM, "structure+internal": np.hstack([S, IM])}
        oofs = {}
        for name, X in feats.items():
            sc, oof0 = scores(y, X); oofs[name] = oof0
            rows.append(dict(organ=organ, n=len(y), pos=int(y.sum()), neg=int((1 - y).sum()),
                             feature=name, **{k: v[0] for k, v in sc.items()},
                             **{k + "_sd": v[1] for k, v in sc.items()}))
            print(f"[{organ}/{name}] n={len(y)} AUROC={sc['auroc'][0]:.3f}±{sc['auroc'][1]:.3f} "
                  f"AUPRC={sc['auprc'][0]:.3f} F1={sc['f1'][0]:.3f}")
        # lift of internal-mean and predicted-DE over structure
        for nm in ("internal mean(128)", "predicted DE", "structure+internal"):
            lift = boot_lift(y, oofs["structure"], oofs[nm])
            lifts.append(dict(organ=organ, feature=nm, lift=lift[0], lo=lift[1], hi=lift[2]))
            print(f"   {nm} vs structure: +lift={lift[0]:+.3f} CI[{lift[1]:+.3f},{lift[2]:+.3f}]")

    L = ["# Internal-representation connection vs DE-output connection (toxicity)", "",
         "Tapping MultiDCP's penultimate per-gene hidden (relu(out) [B,978,128]) pooled over genes, "
         "vs its 978-gene DE output, vs structure. Same head (L2 logreg, balanced), drug-disjoint "
         "5-fold x 5 seeds, full tox set.", "",
         "| Organ | n (pos/neg) | feature | AUROC | AUPRC | accuracy | balanced acc | F1 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        oc = r["organ"] if r["feature"] == "structure" else ""
        nc = f"{r['n']} ({r['pos']}/{r['neg']})" if r["feature"] == "structure" else ""
        L.append(f"| {oc} | {nc} | {r['feature']} | {r['auroc']:.3f}±{r['auroc_sd']:.3f} | "
                 f"{r['auprc']:.3f} | {r['acc']:.3f} | {r['bal_acc']:.3f} | {r['f1']:.3f} |")
    L += ["", "## Lift over structure floor (AUROC, paired bootstrap)",
          "| Organ | feature | +lift | 95% CI |", "|---|---|---|---|"]
    for r in lifts:
        L.append(f"| {r['organ']} | {r['feature']} | **{r['lift']:+.3f}** | [{r['lo']:+.3f}, {r['hi']:+.3f}] |")
    L += ["", "## Reading",
          "- internal mean/meanmax = the model's learned hidden representation (richer than the DE scalar).",
          "- If internal >> predicted DE, the DE-output connection was discarding useful signal; if "
          "internal ~ predicted DE ~ structure, the model's representation is structure-ceilinged "
          "(all are deterministic functions of SMILES).", ""]
    from pathlib import Path
    Path(f"{REPO}/results/tables/P4_internal_rep_tox.md").write_text("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(f"{REPO}/results/tables/P4_internal_rep_tox.csv", index=False)
    print("\nwrote results/tables/P4_internal_rep_tox.md")


if __name__ == "__main__":
    main()
