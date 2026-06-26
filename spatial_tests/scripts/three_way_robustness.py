#!/usr/bin/env python
"""Robustness checks for the three-way comparison's expression arm.

The drug-level raw result used 978 expression dims on ~210 drugs (p>>n), which
may underserve expression. Two fairer tests, same organs (liver/kidney/heart),
same fair evaluation, reusing the validated loaders/join:

  CHECK 1 -- PCA-20 expression (drug level): reduce expression to ~its effective
            rank (Phase 1 measured ~18.5) BEFORE the classifier. PCA fit on the
            training fold only (no leakage). Tests whether a low-dim expression
            signal exists that raw 978-dim LR washes out.
  CHECK 2 -- profile-level drug-disjoint: keep ALL LINCS profiles (more samples)
            instead of mean-aggregating to one vector per drug. StratifiedGroupKFold
            by drug (no drug in train and test). Tests whether the drug-level
            aggregation threw away usable signal.

Headline per organ/check: the +expression LIFT (structure+expression minus
structure), with a bootstrap CI. For profile level the bootstrap resamples DRUGS
(groups), not profiles, to respect non-independence.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests")
DD = REPO / "../dili_downstream/data/processed"
sys.path.insert(0, str(REPO))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

from scripts.three_way_comparison import (
    load_liver, load_kidney, load_heart, build_matrices, load_lincs_expression,
    smi2ikey14, norm_name)
from src.spatial.eda.fingerprints import smiles_to_ecfp4

FIGDIR = Path("/raid/home/joshua/claude_memory/downstream_2026/visuals/three_way_comparison")
ORGANS = {"liver": load_liver, "kidney": load_kidney, "heart": load_heart}
N_SEEDS, N_FOLDS, N_BOOT, N_PCA = 5, 5, 5000, 20


def _lr():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


# ---------------- CHECK 1: drug-level, PCA-20 expression ----------------
def check1_pca(y, Xs, Xe):
    feats = ["structure", "expression_pca20", "both_pca20"]
    aur = {f: [] for f in feats}; oof0 = {}
    for s in range(N_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        oof = {f: np.zeros(len(y)) for f in feats}
        for tr, te in skf.split(Xs, y):
            ssc = StandardScaler().fit(Xs[tr])
            Str, Ste = ssc.transform(Xs[tr]), ssc.transform(Xs[te])
            oof["structure"][te] = _lr().fit(Str, y[tr]).predict_proba(Ste)[:, 1]
            esc = StandardScaler().fit(Xe[tr])
            pca = PCA(n_components=N_PCA, random_state=0).fit(esc.transform(Xe[tr]))
            Ztr, Zte = pca.transform(esc.transform(Xe[tr])), pca.transform(esc.transform(Xe[te]))
            oof["expression_pca20"][te] = _lr().fit(Ztr, y[tr]).predict_proba(Zte)[:, 1]
            Btr, Bte = np.hstack([Str, Ztr]), np.hstack([Ste, Zte])
            oof["both_pca20"][te] = _lr().fit(Btr, y[tr]).predict_proba(Bte)[:, 1]
        for f in feats:
            aur[f].append(roc_auc_score(y, oof[f]))
        if s == 0:
            oof0 = dict(oof)
    return aur, oof0, "structure", "both_pca20"


# ---------------- CHECK 2: profile-level, drug-disjoint ----------------
def build_profiles(lab_drug, y_drug, Xs_drug):
    """Expand to profile level: each LINCS profile of each kept drug is a row."""
    prof = pd.read_csv(DD / "wangli_profiles.csv")
    de = np.load(DD / "wangli_measured_de.npy")
    prof["ikey14"] = prof["smiles"].map(smi2ikey14)
    ik2lab = {ik: int(yy) for ik, yy in zip(lab_drug["ikey14"], y_drug)}
    ik2fp = {ik: Xs_drug[i] for i, ik in enumerate(lab_drug["ikey14"])}
    keep = prof["ikey14"].isin(ik2lab)
    sub = prof[keep]
    rows = [prof.index.get_loc(i) for i in sub.index]
    Xe = de[rows].astype(float)
    Xs = np.vstack([ik2fp[ik] for ik in sub["ikey14"]]).astype(float)
    y = np.array([ik2lab[ik] for ik in sub["ikey14"]], int)
    groups = sub["ikey14"].to_numpy()
    return y, Xs, Xe, groups


def check2_profile(y, Xs, Xe, groups):
    feats = {"structure": Xs, "expression": Xe, "structure+expression": np.hstack([Xs, Xe])}
    aur = {k: [] for k in feats}; oof0 = {}
    for s in range(N_SEEDS):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        oof = {k: np.zeros(len(y)) for k in feats}
        for k, X in feats.items():
            for tr, te in sgkf.split(X, y, groups):
                sc = StandardScaler().fit(X[tr])
                oof[k][te] = _lr().fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
            aur[k].append(roc_auc_score(y, oof[k]))
        if s == 0:
            oof0 = dict(oof)
    return aur, oof0, "structure", "structure+expression"


def paired_bootstrap(y, oof_a, oof_b, groups=None, seed=0):
    """+expr lift CI: AUROC(b) - AUROC(a). Resample groups if given, else rows."""
    rng = np.random.RandomState(seed)
    diffs = []
    if groups is not None:
        uniq = np.unique(groups)
        g2idx = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(N_BOOT):
            gsamp = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([g2idx[g] for g in gsamp])
            if len(np.unique(y[idx])) < 2:
                continue
            diffs.append(roc_auc_score(y[idx], oof_b[idx]) - roc_auc_score(y[idx], oof_a[idx]))
    else:
        n = len(y)
        for _ in range(N_BOOT):
            idx = rng.randint(0, n, n)
            if len(np.unique(y[idx])) < 2:
                continue
            diffs.append(roc_auc_score(y[idx], oof_b[idx]) - roc_auc_score(y[idx], oof_a[idx]))
    d = np.array(diffs)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    drug_de, _ = load_lincs_expression()
    out = []
    for organ, loader in ORGANS.items():
        lab, y, Xs, Xe = build_matrices(loader(), drug_de)

        # CHECK 1
        a1, o1, ka, kb = check1_pca(y, Xs, Xe)
        lift1 = paired_bootstrap(y, o1[ka], o1[kb])
        out.append(dict(organ=organ, check="drug-level PCA-20 expr", n=len(y),
                        struct=np.mean(a1["structure"]), expr=np.mean(a1["expression_pca20"]),
                        both=np.mean(a1["both_pca20"]), lift=lift1[0], lo=lift1[1], hi=lift1[2]))

        # CHECK 2
        yp, Xsp, Xep, gp = build_profiles(lab, y, Xs)
        a2, o2, qa, qb = check2_profile(yp, Xsp, Xep, gp)
        lift2 = paired_bootstrap(yp, o2[qa], o2[qb], groups=gp)
        out.append(dict(organ=organ, check="profile-level drug-disjoint", n=len(yp),
                        n_drugs=len(np.unique(gp)),
                        struct=np.mean(a2["structure"]), expr=np.mean(a2["expression"]),
                        both=np.mean(a2["structure+expression"]), lift=lift2[0], lo=lift2[1], hi=lift2[2]))

        print(f"[{organ}] PCA20  struct={np.mean(a1['structure']):.3f} expr={np.mean(a1['expression_pca20']):.3f} "
              f"both={np.mean(a1['both_pca20']):.3f} lift={lift1[0]:+.3f} CI[{lift1[1]:+.3f},{lift1[2]:+.3f}]")
        print(f"[{organ}] PROF   ({len(yp)} prof / {len(np.unique(gp))} drugs) "
              f"struct={np.mean(a2['structure']):.3f} expr={np.mean(a2['expression']):.3f} "
              f"both={np.mean(a2['structure+expression']):.3f} lift={lift2[0]:+.3f} CI[{lift2[1]:+.3f},{lift2[2]:+.3f}]")

    df = pd.DataFrame(out)
    # ---- figure: +expression lift with CI, drug-raw (prior) vs PCA20 vs profile ----
    prior = {"liver": -0.048, "kidney": -0.005, "heart": +0.042}  # from P1_three_way (drug-level raw)
    prior_ci = {"liver": (-0.118, 0.024), "kidney": (-0.049, 0.038), "heart": (-0.024, 0.109)}
    organs = list(ORGANS)
    fig, ax = plt.subplots(figsize=(8.6, 5.0)); w = 0.26
    series = [("drug raw (978d)", "#bbbbbb", prior, prior_ci),
              ("drug PCA-20", "#f5b041", {o: df[(df.organ == o) & (df.check.str.contains("PCA"))]["lift"].iloc[0] for o in organs},
               {o: (df[(df.organ == o) & (df.check.str.contains("PCA"))]["lo"].iloc[0],
                    df[(df.organ == o) & (df.check.str.contains("PCA"))]["hi"].iloc[0]) for o in organs}),
              ("profile-level", "#2c7fb8", {o: df[(df.organ == o) & (df.check.str.contains("profile"))]["lift"].iloc[0] for o in organs},
               {o: (df[(df.organ == o) & (df.check.str.contains("profile"))]["lo"].iloc[0],
                    df[(df.organ == o) & (df.check.str.contains("profile"))]["hi"].iloc[0]) for o in organs})]
    for i, (lbl, col, lifts, cis) in enumerate(series):
        x = np.arange(len(organs)) + (i - 1) * w
        vals = [lifts[o] for o in organs]
        err = [[lifts[o] - cis[o][0] for o in organs], [cis[o][1] - lifts[o] for o in organs]]
        ax.bar(x, vals, w, yerr=err, capsize=3, label=lbl, color=col)
    ax.axhline(0, c="black", lw=1)
    ax.set_xticks(np.arange(len(organs))); ax.set_xticklabels(organs)
    ax.set_ylabel("+expression lift over structure (AUROC, 95% CI)")
    ax.set_title("Does expression add over structure? Robustness of the +expression lift\n"
                 "(bars crossing 0 = no significant lift)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGDIR / "expression_lift_robustness.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    L = ["# Expression-arm robustness checks (three-way comparison)", "",
         "Tests whether 'measured expression adds nothing over structure' survives a fairer "
         "setup than the raw 978-dim drug-level LR (p>>n). Same organs, same fair evaluation.", "",
         "## +expression lift (structure+expression minus structure), by organ and method", "",
         "| Organ | method | n | structure | expression | both | **+expr lift** | 95% CI |",
         "|---|---|---|---|---|---|---|---|",
         "| liver | drug-level raw 978d (prior) | 234 | 0.575 | 0.456 | 0.527 | **-0.048** | [-0.118, +0.024] |",
         "| kidney | drug-level raw 978d (prior) | 203 | 0.680 | 0.493 | 0.659 | **-0.005** | [-0.049, +0.038] |",
         "| heart | drug-level raw 978d (prior) | 221 | 0.585 | 0.514 | 0.597 | **+0.042** | [-0.024, +0.109] |"]
    for r in out:
        nlab = f"{r['n']}" + (f" prof/{r['n_drugs']} drugs" if "n_drugs" in r and pd.notna(r.get("n_drugs")) else "")
        L.append(f"| {r['organ']} | {r['check']} | {nlab} | {r['struct']:.3f} | {r['expr']:.3f} | "
                 f"{r['both']:.3f} | **{r['lift']:+.3f}** | [{r['lo']:+.3f}, {r['hi']:+.3f}] |")
    L += ["", "Lift CI excluding 0 => expression adds signal over structure. PCA reduces expression to "
          f"{N_PCA} components (fit on train fold only). Profile-level keeps all LINCS profiles, "
          "StratifiedGroupKFold by drug, bootstrap resamples DRUGS not profiles.", "",
          "## Figure", f"- `{FIGDIR}/expression_lift_robustness.png`", ""]
    (FIGDIR / "ROBUSTNESS.md").write_text("\n".join(L))
    (REPO / "results/tables/P1_three_way_robustness.md").write_text("\n".join(L))
    df.to_csv(REPO / "results/tables/P1_three_way_robustness.csv", index=False)
    print(f"\nwrote {FIGDIR}/ROBUSTNESS.md + figure, results/tables/P1_three_way_robustness.{{md,csv}}")


if __name__ == "__main__":
    main()
