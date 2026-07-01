#!/usr/bin/env python
"""Do the DE+structure combination PROPERLY and re-test vs structure alone.

Fixes the ML gaps in stage2's "both":
  - tune L2 strength C via nested CV (LogisticRegressionCV) instead of fixed C=1
  - PCA-reduce the 978-d DE block before fusion so it can't dilute the 384-d structure
  - late fusion (average of a structure model and a DE model's probabilities), which
    avoids the early-fusion dimensionality imbalance entirely

Feature blocks: structure = ChemBERTa (384), predicted DE = organ engine @ organ-mean
basal (978). Drug-disjoint 5-fold x 5 seeds, AUROC (folds depend only on y+seed, so
they're identical across feature sets -> valid late fusion). All scalers/PCA/C-tuning
fit inside each training fold (no leakage).
"""
from __future__ import annotations
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="1")
ap.add_argument("--encoder", default="linear")
ap.add_argument("--organs", default="liver,kidney")
ap.add_argument("--pca", type=int, default=30)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import pandas as pd
import torch

REPO = "/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests"
MULTIDCP = "/raid/home/joshua/projects/MultiDCP/MultiDCP"
GENE_VECTOR = f"{MULTIDCP}/data/gene_vector.csv"
CKPT_DIR = f"{REPO}/results/checkpoints"
CACHE = f"{REPO}/data/processed/latent_cache"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, REPO)

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scripts.three_way_comparison import load_liver, load_kidney, smi2ikey14
from src.spatial.structure_encoders import get_encoder

LOADERS = {"liver": load_liver, "kidney": load_kidney}
N_SEEDS, N_FOLDS = 5, 5
CS = np.logspace(-3, 3, 13)


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
def get_features(organ):
    """Cache structure [n,384], predicted DE [n,978], labels."""
    cache = f"{CACHE}/improved_feats_{organ}.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["S"], z["DE"], z["y"]
    df = LOADERS[organ]().dropna(subset=["smiles"]).copy()
    df["ik"] = df["smiles"].map(smi2ikey14)
    df = df.dropna(subset=["ik"]).drop_duplicates("ik").reset_index(drop=True)
    smis = df["smiles"].tolist()
    d = np.load(f"{REPO}/data/processed/organ_train/{organ}.npz", allow_pickle=True)
    basal_vec = d["x2"].astype(np.float64).mean(0)
    model, feat, gene_t = build_engine()
    model.load_state_dict(torch.load(f"{CKPT_DIR}/multidcp_{organ}_{args.encoder}_sd42.pt",
                                     map_location=DEVICE)); model.eval()
    DE = np.full((len(smis), 978), np.nan, np.float64)
    basal = torch.as_tensor(basal_vec[None], dtype=torch.float64, device=DEVICE)
    for i, s in enumerate(smis):
        if not _ok(feat, s):
            continue
        drug, mask = feat([s])
        dose = torch.ones(1, 1, dtype=torch.float64, device=DEVICE)
        p = model(drug, gene_t, mask, basal, dose, epoch=0)
        DE[i] = ((p[0] if isinstance(p, tuple) else p)[0].cpu().numpy()) - basal_vec
    enc = get_encoder("chemberta")
    S, oks = enc(smis); S, oks = np.asarray(S), np.asarray(oks, bool)
    good = oks & np.isfinite(DE).all(1)
    S, DE, y = S[good].astype(float), DE[good].astype(float), df["label"].to_numpy(int)[good]
    np.savez(cache, S=S, DE=DE, y=y)
    return S, DE, y


def oof(X, y, seed, est_factory):
    o = np.zeros(len(y))
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y):
        est = est_factory(); est.fit(X[tr], y[tr]); o[te] = est.predict_proba(X[te])[:, 1]
    return o


def lrcv():
    return LogisticRegressionCV(Cs=CS, cv=5, scoring="roc_auc", max_iter=4000,
                                class_weight="balanced")


def factories(n_struct):
    scaled_lrcv = lambda: make_pipeline(StandardScaler(), lrcv())
    de_pca = lambda: make_pipeline(StandardScaler(), PCA(n_components=args.pca), lrcv())
    # early fusion, tuned + DE block PCA'd (structure passthrough-scaled)
    ct = lambda: ColumnTransformer([
        ("struct", StandardScaler(), slice(0, n_struct)),
        ("de", make_pipeline(StandardScaler(), PCA(n_components=args.pca)), slice(n_struct, None)),
    ])
    both_tuned = lambda: Pipeline([("ct", ct()), ("clf", lrcv())])
    both_naive = lambda: make_pipeline(StandardScaler(),
                                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    return scaled_lrcv, de_pca, both_naive, both_tuned


def main():
    rows = []
    for organ in args.organs.split(","):
        S, DE, y = get_features(organ)
        both = np.hstack([S, DE])
        struct_f, de_f, both_naive_f, both_tuned_f = factories(S.shape[1])
        variants = {
            "structure (tuned)": (S, struct_f),
            f"DE PCA{args.pca} (tuned)": (DE, de_f),
            "both naive (C=1)": (both, both_naive_f),
            f"both early+PCA{args.pca} (tuned)": (both, both_tuned_f),
        }
        aur = {}
        seed_oofs = {}
        for name, (X, fac) in variants.items():
            per_seed = []
            for sd in range(N_SEEDS):
                o = oof(X, y, sd, fac); per_seed.append(roc_auc_score(y, o))
                if name in ("structure (tuned)", f"DE PCA{args.pca} (tuned)"):
                    seed_oofs.setdefault(name, {})[sd] = o
            aur[name] = (float(np.mean(per_seed)), float(np.std(per_seed)))
        # late fusion = average of tuned structure + tuned DE probs (same folds)
        lf = []
        for sd in range(N_SEEDS):
            o = 0.5 * (seed_oofs["structure (tuned)"][sd] + seed_oofs[f"DE PCA{args.pca} (tuned)"][sd])
            lf.append(roc_auc_score(y, o))
        aur["both LATE fusion (avg)"] = (float(np.mean(lf)), float(np.std(lf)))

        print(f"\n[{organ}] n={len(y)} ({int(y.sum())}/{int((1-y).sum())})")
        for k, (m, sd) in aur.items():
            print(f"   {k:32s}: {m:.3f}±{sd:.3f}")
        for k, (m, sd) in aur.items():
            rows.append(dict(organ=organ, n=len(y), variant=k, auroc=m, sd=sd))

    df = pd.DataFrame(rows)
    L = ["# Properly combining DE + structure (fixing the naive-fusion gap)", "",
         "structure=ChemBERTa(384), DE=engine predicted DE(978) @ organ-mean basal. All C tuned by "
         "nested LogisticRegressionCV (roc_auc, inner cv=5); DE PCA-reduced before fusion; late "
         "fusion = avg of tuned structure+DE probs. Drug-disjoint 5-fold x 5 seeds.", "",
         "| Organ | n | variant | AUROC |", "|---|---|---|---|"]
    for r in rows:
        oc = r["organ"] if r["variant"] == "structure (tuned)" else ""
        nc = r["n"] if r["variant"] == "structure (tuned)" else ""
        L.append(f"| {oc} | {nc} | {r['variant']} | {r['auroc']:.3f}±{r['sd']:.3f} |")
    L += ["", "## Reading",
          "- 'both naive (C=1)' is the old stage-2 method; compare to the tuned/PCA/late-fusion "
          "variants. If proper fusion now EXCEEDS structure, the earlier both<structure was a "
          "methodology artifact (DE block diluting structure under uniform L2). If it still ties "
          "structure, the ceiling is real.", ""]
    from pathlib import Path
    Path(f"{REPO}/results/tables/P4_improved_both.md").write_text("\n".join(L) + "\n")
    df.to_csv(f"{REPO}/results/tables/P4_improved_both.csv", index=False)
    print("\nwrote results/tables/P4_improved_both.md")


if __name__ == "__main__":
    main()
