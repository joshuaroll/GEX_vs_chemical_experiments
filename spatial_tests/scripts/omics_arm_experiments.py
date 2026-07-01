#!/usr/bin/env python
"""Can the MEASURED-omics arm be made toxicity-informative on its own?

The fusion diagnostic showed measured DE is structure-independent (negative R^2 to structure)
but its standalone tox classifier is at chance (0.51-0.53) with raw 978-gene L2 logreg on n~200,
so prob-fusion has nothing to extract. This asks: does a better representation or model lift the
omics arm above chance? If yes -> fusion is worth engineering; if no -> the omics signal is not in
this data (as represented) and the negative holds for the right reason.

Per organ (measured-DE overlap set):
  A. omics-alone AUROC over representation x model grid
       reps:   raw978 | pca30 | kbest100 (supervised, in-pipeline, leakage-safe)
       models: logreg(L2) | rf(300) | mlp(64)
  B. oracle-inflation sanity: oracle(structure, PERMUTED omics) — how much of the 0.90 oracle
     is just label-informed selection over an independent-but-uninformative stream.
  C. if best omics-alone > 0.57, nonlinear JOINT fusion (MLP on [structure (+) omics]) vs structure,
     vs late/stacker — does feature-level nonlinear fusion beat structure when prob-fusion didn't?
"""
from __future__ import annotations
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="1")
ap.add_argument("--organs", default="liver,kidney")
ap.add_argument("--struct", default="ecfp4", help="structure encoder for the fusion test")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import pandas as pd
from pathlib import Path

REPO = "/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests"
sys.path.insert(0, REPO)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scripts.three_way_comparison import load_liver, load_kidney, smi2ikey14
from src.spatial.structure_encoders import get_encoder

LOADERS = {"liver": load_liver, "kidney": load_kidney}
N_SEEDS, N_FOLDS = 3, 5


def measured_de_map(organ):
    d = np.load(f"{REPO}/data/processed/organ_train/{organ}.npz", allow_pickle=True)
    de = (d["x1"].astype(np.float64) - d["x2"].astype(np.float64))
    ik = np.array([smi2ikey14(s) or "NA" for s in d["smiles"]])
    return {k: de[ik == k].mean(0) for k in np.unique(ik) if k != "NA"}


def model(name):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, early_stopping=True, random_state=0)
    raise ValueError(name)


def rep_pipe(rep, mdl):
    steps = [StandardScaler()]
    if rep == "pca30":
        steps.append(PCA(30))
    elif rep == "kbest100":
        steps.append(SelectKBest(f_classif, k=100))
    elif rep != "raw":
        raise ValueError(rep)
    steps.append(model(mdl))
    return make_pipeline(*steps)


def oof(pipe_factory, X, y, seed):
    o = np.zeros(len(y))
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y):
        clf = pipe_factory()
        clf.fit(X[tr], y[tr]); o[te] = clf.predict_proba(X[te])[:, 1]
    return o


def auroc_over_seeds(pipe_factory, X, y):
    a = [roc_auc_score(y, oof(pipe_factory, X, y, s)) for s in range(N_SEEDS)]
    return float(np.mean(a)), float(np.std(a))


def main():
    enc = get_encoder(args.struct)
    grid_rows, fuse_rows = [], []
    for organ in args.organs.split(","):
        tox = LOADERS[organ]().dropna(subset=["smiles"]).copy()
        tox["ik"] = tox["smiles"].map(smi2ikey14)
        tox = tox.dropna(subset=["ik"]).drop_duplicates("ik").reset_index(drop=True)
        mde = measured_de_map(organ)
        tox = tox[tox["ik"].isin(mde)].reset_index(drop=True)
        smis = tox["smiles"].tolist()
        S, ok = enc(smis); S, ok = np.asarray(S), np.asarray(ok, bool)
        tox = tox[ok].reset_index(drop=True)
        y = tox["label"].to_numpy(int)
        Xs = S[ok].astype(float)
        Xm = np.vstack([mde[k] for k in tox["ik"]]).astype(float)
        print(f"\n[{organ}] n={len(y)} ({int(y.sum())}/{int((1-y).sum())}) struct={args.struct}")

        # A. omics-alone grid
        best = (None, None, -1)
        for rep in ("raw", "pca30", "kbest100"):
            for mdl in ("logreg", "rf", "mlp"):
                mu, sd = auroc_over_seeds(lambda r=rep, m=mdl: rep_pipe(r, m), Xm, y)
                grid_rows.append(dict(organ=organ, arm="measured omics", rep=rep, model=mdl,
                                      auroc=mu, auroc_sd=sd))
                print(f"  omics {rep:9s} {mdl:7s} AUROC={mu:.3f}±{sd:.3f}")
                if mu > best[2]:
                    best = (rep, mdl, mu)
        # structure reference
        s_mu, s_sd = auroc_over_seeds(lambda: rep_pipe("raw", "logreg"), Xs, y)
        print(f"  structure raw     logreg  AUROC={s_mu:.3f}±{s_sd:.3f}")

        # B. oracle-inflation sanity (structure + permuted omics prob)
        ps = oof(lambda: rep_pipe("raw", "logreg"), Xs, y, 0)
        pm = oof(lambda r=best[0], m=best[1]: rep_pipe(r, m), Xm, y, 0)
        rng = np.random.RandomState(0); pm_perm = pm[rng.permutation(len(pm))]
        orc_real = roc_auc_score(y, np.where(np.abs(y - ps) <= np.abs(y - pm), ps, pm))
        orc_perm = roc_auc_score(y, np.where(np.abs(y - ps) <= np.abs(y - pm_perm), ps, pm_perm))
        print(f"  ORACLE real={orc_real:.3f}  ORACLE(struct+PERMUTED omics)={orc_perm:.3f} "
              f"(gap over structure that is pure inflation: {orc_perm - s_mu:+.3f})")

        # C. nonlinear joint fusion using the BEST omics rep, vs structure / late / stacker
        Xfuse = np.hstack([Xs, Xm])
        f_mlp, f_mlp_sd = auroc_over_seeds(lambda: rep_pipe("raw", "mlp"), Xfuse, y)
        s_mlp, s_mlp_sd = auroc_over_seeds(lambda: rep_pipe("raw", "mlp"), Xs, y)
        late = np.mean([roc_auc_score(y, 0.5 * (oof(lambda: rep_pipe("raw", "logreg"), Xs, y, s)
                       + oof(lambda r=best[0], m=best[1]: rep_pipe(r, m), Xm, y, s)))
                        for s in range(N_SEEDS)])
        fuse_rows.append(dict(organ=organ, n=len(y), struct=args.struct,
                              best_omics_rep=best[0], best_omics_model=best[1], best_omics_auroc=best[2],
                              structure_logreg=s_mu, structure_mlp=s_mlp,
                              joint_concat_mlp=f_mlp, late_fusion=late,
                              oracle_real=orc_real, oracle_permuted=orc_perm))
        print(f"  FUSION  struct(mlp)={s_mlp:.3f}  joint-concat(mlp)={f_mlp:.3f}  late={late:.3f}  "
              f"| best omics {best[0]}/{best[1]}={best[2]:.3f}")

    write(grid_rows, fuse_rows)


def write(grid_rows, fuse_rows):
    L = ["# Can the measured-omics arm be made toxicity-informative? (+ honest fusion test)", "",
         "Measured-DE overlap set. omics-alone over representation x model; then oracle-inflation "
         "sanity and a nonlinear JOINT (concat) fusion vs structure. 3 seeds x 5-fold.", "",
         "## A. omics-alone AUROC (representation x model)", "",
         "| organ | rep | model | AUROC |", "|---|---|---|---|"]
    for r in grid_rows:
        L.append(f"| {r['organ']} | {r['rep']} | {r['model']} | {r['auroc']:.3f}±{r['auroc_sd']:.3f} |")
    L += ["", "## B/C. fusion + oracle-inflation", "",
          "| organ | n | best omics (rep/model) | structure (logreg / mlp) | joint concat-MLP | late | "
          "oracle real | oracle PERMUTED-omics |", "|---|---|---|---|---|---|---|---|"]
    for r in fuse_rows:
        L.append(f"| {r['organ']} | {r['n']} | {r['best_omics_rep']}/{r['best_omics_model']} "
                 f"({r['best_omics_auroc']:.3f}) | {r['structure_logreg']:.3f} / {r['structure_mlp']:.3f} | "
                 f"{r['joint_concat_mlp']:.3f} | {r['late_fusion']:.3f} | {r['oracle_real']:.3f} | "
                 f"{r['oracle_permuted']:.3f} |")
    L += ["", "## Reading",
          "- If best omics-alone stays ~chance across all reps/models, the measured omics is not tox-"
          "informative as represented (cell context / aggregation / n), and fusion has nothing to add.",
          "- oracle PERMUTED-omics ~ oracle real => the high oracle is label-selection INFLATION, not "
          "real complementary signal.",
          "- joint concat-MLP > structure-mlp => nonlinear feature-level fusion finds signal that linear "
          "prob-fusion missed (the user's hypothesis); ~equal => it does not, on this data.", ""]
    Path(f"{REPO}/results/tables/P4_omics_arm.md").write_text("\n".join(L) + "\n")
    pd.DataFrame(grid_rows).to_csv(f"{REPO}/results/tables/P4_omics_arm_grid.csv", index=False)
    pd.DataFrame(fuse_rows).to_csv(f"{REPO}/results/tables/P4_omics_arm_fusion.csv", index=False)
    print("\nwrote results/tables/P4_omics_arm.md")


if __name__ == "__main__":
    main()
