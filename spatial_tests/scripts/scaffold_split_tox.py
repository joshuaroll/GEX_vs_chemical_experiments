#!/usr/bin/env python
"""Scaffold-disjoint vs drug-disjoint toxicity: does structure's edge shrink when
analog/scaffold leakage is removed?

Same features (structure=ChemBERTa, predicted DE=organ engine output, both) scored
under TWO CV schemes:
  drug-disjoint     : StratifiedKFold over deduplicated drugs (a scaffold can span folds)
  scaffold-disjoint : StratifiedGroupKFold grouped by Bemis-Murcko scaffold (no scaffold
                      in train and test; acyclic molecules are singletons)
5 folds x 5 seeds. If structure drops MORE than predicted DE under scaffold-disjoint,
structure's strength was partly analog memorization; if both drop together, the
expression signal is no more mechanistic than structure.
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
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from scripts.three_way_comparison import load_liver, load_kidney, smi2ikey14
from src.spatial.structure_encoders import get_encoder

LOADERS = {"liver": load_liver, "kidney": load_kidney}
N_SEEDS, N_FOLDS = 5, 5


def scaffold_of(smiles, idx):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        m = Chem.MolFromSmiles(smiles)
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else ""
    except Exception:
        sc = ""
    return sc if sc else f"_acyclic_{idx}"  # acyclic -> singleton group


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
def predicted_de(organ, smis):
    d = np.load(f"{REPO}/data/processed/organ_train/{organ}.npz", allow_pickle=True)
    basal_vec = d["x2"].astype(np.float64).mean(0)
    model, feat, gene_t = build_engine()
    model.load_state_dict(torch.load(f"{CKPT_DIR}/multidcp_{organ}_{args.encoder}_sd42.pt",
                                     map_location=DEVICE)); model.eval()
    basal = torch.as_tensor(basal_vec[None], dtype=torch.float64, device=DEVICE)
    out = np.full((len(smis), 978), np.nan, np.float64)
    for i, s in enumerate(smis):
        if not _ok(feat, s):
            continue
        drug, mask = feat([s])
        dose = torch.ones(1, 1, dtype=torch.float64, device=DEVICE)
        pred = model(drug, gene_t, mask, basal, dose, epoch=0)
        pred = (pred[0] if isinstance(pred, tuple) else pred)[0].cpu().numpy()
        out[i] = pred - basal_vec
    return out


def cv_auroc(X, y, splitter_factory, groups=None):
    aurs = []
    for seed in range(N_SEEDS):
        sp = splitter_factory(seed)
        oof = np.zeros(len(y))
        it = sp.split(X, y, groups) if groups is not None else sp.split(X, y)
        for tr, te in it:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
        aurs.append(roc_auc_score(y, oof))
    return float(np.mean(aurs)), float(np.std(aurs))


def main():
    enc = get_encoder("chemberta")
    rows = []
    for organ in args.organs.split(","):
        df = LOADERS[organ]().dropna(subset=["smiles"]).copy()
        df["ik"] = df["smiles"].map(smi2ikey14)
        df = df.dropna(subset=["ik"]).drop_duplicates("ik").reset_index(drop=True)
        smis = df["smiles"].tolist()
        DE = predicted_de(organ, smis)
        Semb, oks = enc(smis); Semb, oks = np.asarray(Semb), np.asarray(oks, bool)
        good = oks & np.isfinite(DE).all(1)
        df = df[good].reset_index(drop=True)
        smis = [s for s, g in zip(smis, good) if g]
        y = df["label"].to_numpy(int)
        S, DE = Semb[good].astype(float), DE[good].astype(float)
        scaf = np.array([scaffold_of(s, i) for i, s in enumerate(smis)])
        n_scaf = len(set(scaf))
        feats = {"structure": S, "predicted DE": DE, "both": np.hstack([S, DE])}

        drug_sp = lambda seed: StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        scaf_sp = lambda seed: StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
        for name, X in feats.items():
            dm, ds = cv_auroc(X, y, drug_sp)
            sm, ss = cv_auroc(X, y, scaf_sp, groups=scaf)
            rows.append(dict(organ=organ, n=len(y), n_scaffolds=n_scaf, feature=name,
                             drug_auroc=dm, drug_sd=ds, scaf_auroc=sm, scaf_sd=ss))
            print(f"[{organ}/{name}] drug-disjoint={dm:.3f}±{ds:.3f} | "
                  f"scaffold-disjoint={sm:.3f}±{ss:.3f} | drop={dm-sm:+.3f}")

    L = ["# Scaffold-disjoint vs drug-disjoint toxicity (does structure's edge survive?)", "",
         "Same features, two CV schemes. drug-disjoint = StratifiedKFold over drugs (scaffolds can "
         "span folds). scaffold-disjoint = StratifiedGroupKFold grouped by Bemis-Murcko scaffold "
         "(acyclic = singletons). 5 folds x 5 seeds, AUROC.", "",
         "| Organ | n (scaffolds) | feature | drug-disjoint AUROC | scaffold-disjoint AUROC | drop |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        oc = f"{r['organ']}" if r["feature"] == "structure" else ""
        nc = f"{r['n']} ({r['n_scaffolds']})" if r["feature"] == "structure" else ""
        L.append(f"| {oc} | {nc} | {r['feature']} | {r['drug_auroc']:.3f}±{r['drug_sd']:.3f} | "
                 f"{r['scaf_auroc']:.3f}±{r['scaf_sd']:.3f} | {r['drug_auroc']-r['scaf_auroc']:+.3f} |")
    # structure - predicted DE gap under each split
    L += ["", "## structure − predicted DE gap, by split (does structure's edge shrink?)",
          "| Organ | gap drug-disjoint | gap scaffold-disjoint |", "|---|---|---|"]
    by = {(r["organ"], r["feature"]): r for r in rows}
    for organ in args.organs.split(","):
        s, d = by[(organ, "structure")], by[(organ, "predicted DE")]
        L.append(f"| {organ} | {s['drug_auroc']-d['drug_auroc']:+.3f} | {s['scaf_auroc']-d['scaf_auroc']:+.3f} |")
    L += ["", "## Reading",
          "- If structure drops MORE than predicted DE under scaffold-disjoint (gap shrinks), "
          "structure's edge was partly analog memorization, and expression is relatively more "
          "mechanistic. If both drop together (gap stable), structure's edge is real generalization.", ""]
    from pathlib import Path
    Path(f"{REPO}/results/tables/P4_scaffold_split_tox.md").write_text("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(f"{REPO}/results/tables/P4_scaffold_split_tox.csv", index=False)
    print("\nwrote results/tables/P4_scaffold_split_tox.md")


if __name__ == "__main__":
    main()
