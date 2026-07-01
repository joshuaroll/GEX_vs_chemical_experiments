#!/usr/bin/env python
"""Stage 2: organ toxicity prediction from organ-trained predicted DE.

For each organ with both a trained engine and tox labels (liver, kidney):
  expression = organ engine's predicted DE for the drug, in a representative organ
               basal context: DE = engine(smiles, organ_basal) - organ_basal  [978].
  structure  = ChemBERTa embedding (floor).
  both       = concat.
Same small fixed head (L2 logistic regression, balanced), drug-disjoint 5-fold x 5
seeds. Reports AUROC, AUPRC, accuracy, balanced-acc, F1 + the +expression lift over
structure with a paired bootstrap CI. This is the payoff: does organ-grounded
PREDICTED expression beat chemical structure at predicting organ toxicity?
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
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests"
MULTIDCP = "/raid/home/joshua/projects/MultiDCP/MultiDCP"
CHEMOE = "/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src"
GENE_VECTOR = f"{MULTIDCP}/data/gene_vector.csv"
CKPT_DIR = f"{REPO}/results/checkpoints"
CACHE = f"{REPO}/data/processed/latent_cache"
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


def build_engine(arch):
    if arch == "multidcp":
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
    raise SystemExit("only multidcp arch wired for stage2 (the better engine)")


@torch.no_grad()
def predict_de(organ, smis):
    """Predicted DE [n,978] for SMILES, in the organ's mean-basal context."""
    cache = f"{CACHE}/stage2_predDE_{args.arch}_{organ}.npz"
    d = np.load(f"{REPO}/data/processed/organ_train/{organ}.npz", allow_pickle=True)
    basal_vec = d["x2"].astype(np.float64).mean(0)  # representative organ basal
    model, feat, fwd = build_engine(args.arch)
    model.load_state_dict(torch.load(f"{CKPT_DIR}/{args.arch}_{organ}_{args.encoder}_sd42.pt",
                                     map_location=DEVICE)); model.eval()
    out, ok = [], []
    for i in range(0, len(smis), 64):
        s = list(smis[i:i + 64])
        good = [x for x in s if _ok(feat, x)]
        ok += [x in set(good) for x in s]
        if not good:
            continue
        drug, mask = feat(good)
        basal = torch.as_tensor(np.tile(basal_vec, (len(good), 1)), dtype=torch.float64, device=DEVICE)
        pred = fwd(drug, mask, basal).cpu().numpy()
        out.append(pred - basal_vec)  # DE = pred - basal
    return np.vstack(out).astype(np.float32), np.array(ok)


def _ok(feat, s):
    try:
        feat([s]); return True
    except Exception:
        return False


def cv_oof(X, y, seed):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def evaluate(y, feats):
    out = {k: {m: [] for m in ("auroc", "auprc", "acc", "bal_acc", "f1")} for k in feats}
    oof0 = {}
    for k, X in feats.items():
        for s in range(N_SEEDS):
            oof = cv_oof(X, y, s); pred = (oof >= 0.5).astype(int)
            out[k]["auroc"].append(roc_auc_score(y, oof))
            out[k]["auprc"].append(average_precision_score(y, oof))
            out[k]["acc"].append(accuracy_score(y, pred))
            out[k]["bal_acc"].append(balanced_accuracy_score(y, pred))
            out[k]["f1"].append(f1_score(y, pred))
            if s == 0: oof0[k] = oof
    return out, oof0


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
    rows = []
    for organ in args.organs.split(","):
        df = LOADERS[organ]().dropna(subset=["smiles"]).copy()
        df["ikey14"] = df["smiles"].map(smi2ikey14)
        df = df.dropna(subset=["ikey14"]).drop_duplicates("ikey14").reset_index(drop=True)
        smis = df["smiles"].tolist()
        Xe, oke = predict_de(organ, smis)            # Xe rows = oke.sum(), original order
        Semb, oks = enc(smis)                         # structure, oks per smile
        oke, oks = np.asarray(oke, bool), np.asarray(oks, bool)
        keep = oke & oks
        y = df["label"].to_numpy(int)[keep]
        Xs = np.asarray(Semb)[keep].astype(float)
        Xexp = Xe[oks[oke]].astype(float)            # among oke-True rows, keep also-oks-True
        assert len(Xs) == len(Xexp) == len(y), (len(Xs), len(Xexp), len(y))
        feats = {"structure": Xs, "expression": Xexp, "both": np.hstack([Xs, Xexp])}
        res, oof0 = evaluate(y, feats)
        lift = boot_lift(y, oof0["structure"], oof0["both"])
        row = dict(organ=organ, n=len(y), pos=int(y.sum()), neg=int((1 - y).sum()),
                   lift=lift[0], lo=lift[1], hi=lift[2])
        for k in feats:
            for m in ("auroc", "auprc", "acc", "bal_acc", "f1"):
                row[f"{k}_{m}"] = float(np.mean(res[k][m]))
                row[f"{k}_{m}_sd"] = float(np.std(res[k][m]))
        rows.append(row)
        print(f"[{organ}] n={row['n']} ({row['pos']}/{row['neg']}) | "
              f"AUROC struct={row['structure_auroc']:.3f} expr={row['expression_auroc']:.3f} "
              f"both={row['both_auroc']:.3f} | +lift={lift[0]:+.3f} CI[{lift[1]:+.3f},{lift[2]:+.3f}]")

    L = ["# Stage 2: organ toxicity from organ-trained predicted DE", "",
         f"expression = **{args.arch}** organ-engine predicted DE (in the organ mean-basal context) | "
         "structure = ChemBERTa | head = L2 logistic regression, drug-disjoint 5-fold x 5 seeds. "
         "Engines & labels overlap only for liver, kidney.", "",
         "| Organ | n (pos/neg) | feature | AUROC | AUPRC | accuracy | balanced acc | F1 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for k, lab in [("structure", "structure"), ("expression", "predicted DE"), ("both", "both")]:
            organ_c = r["organ"] if k == "structure" else ""
            n_c = f"{r['n']} ({r['pos']}/{r['neg']})" if k == "structure" else ""
            L.append(f"| {organ_c} | {n_c} | {lab} | "
                     f"{r[k+'_auroc']:.3f}±{r[k+'_auroc_sd']:.3f} | {r[k+'_auprc']:.3f} | "
                     f"{r[k+'_acc']:.3f} | {r[k+'_bal_acc']:.3f} | {r[k+'_f1']:.3f} |")
    L += ["", "## +expression lift over the structure floor (AUROC, paired bootstrap)",
          "| Organ | structure AUROC | both AUROC | +lift | 95% CI |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['organ']} | {r['structure_auroc']:.3f} | {r['both_auroc']:.3f} | "
                 f"**{r['lift']:+.3f}** | [{r['lo']:+.3f}, {r['hi']:+.3f}] |")
    L += ["", "## Reading",
          "- The headline is +expression lift: does organ-trained predicted DE add toxicity signal over "
          "chemical structure (drug-disjoint, honest)?",
          "- expression here is PREDICTED (organ engine, from SMILES) so it generalizes to any drug; the "
          "feature is the predicted perturbation in the organ's representative basal context.",
          "- Brain has an engine but no tox labels (SIDER block); heart has labels but no engine (no LINCS "
          "cardiac data). Both excluded.", ""]
    from pathlib import Path
    Path(f"{REPO}/results/tables/P4_stage2_toxicity.md").write_text("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(f"{REPO}/results/tables/P4_stage2_toxicity.csv", index=False)
    print("\nwrote results/tables/P4_stage2_toxicity.md")


if __name__ == "__main__":
    main()
