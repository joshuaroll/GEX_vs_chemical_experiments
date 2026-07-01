#!/usr/bin/env python
"""DILIrank SeverityClass (0-8) prediction: structure vs expression vs both.

All three feature arms predict the ordinal severity CLASS (not the binary label). Liver only
(DILIrank is the only severity source). Compares feature sets AND two classifiers so "best
predictive model" is a real comparison. Two cohorts:
  FULL  all severity drugs (0-8, includes the SeverityClass=0 "no-DILI")  -> partly recapitulates
        the binary "is it toxic" split plus grading.
  POS   toxicants only (SeverityClass >= 1)  -> the sharper test: grade "how bad" with the
        binary is-it-toxic component removed (where structure's chemotype edge lived).

expression = organ engine predicted DE (all drugs; near-circular caveat) with a measured-DE
sub-comparison on the LINCS overlap (the honest omics test). Drug-disjoint 5-fold x 3 seeds,
folds stratified on coarse severity bins (raw class is imbalanced in the tail).

Two output tables per the request:
  (1) ACCURACY   exact accuracy + within-1 (ordinal-tolerant) accuracy
  (2) OTHER METRICS  balanced acc, macro-F1, quadratic-weighted Cohen kappa, Spearman rho, MAE
Metrics that reward getting the ordinal right (QWK, Spearman, MAE, within-1) are the ones to
judge "best predictive model" on; exact 9-class accuracy is reported but is dominated by the
common classes.
"""
from __future__ import annotations
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="1")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--folds", type=int, default=5)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import pandas as pd
from pathlib import Path
import torch

REPO = "/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests"
MULTIDCP = "/raid/home/joshua/projects/MultiDCP/MultiDCP"
GENE_VECTOR = f"{MULTIDCP}/data/gene_vector.csv"
CKPT_DIR = f"{REPO}/results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, REPO)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             cohen_kappa_score, mean_absolute_error)
from scipy.stats import spearmanr
from scripts.three_way_comparison import _name_cascade, norm_name, smi2ikey14
from src.spatial.structure_encoders import get_encoder


def load_severity():
    f = _name_cascade()
    dr = pd.read_excel(f"{REPO}/data/raw/labels/dilirank/dilirank.xlsx", header=1)
    dr["name_lower"] = dr["CompoundName"].map(norm_name)
    dr = dr.dropna(subset=["name_lower"]).drop_duplicates("name_lower")
    dr["smiles"] = dr["name_lower"].map(f)
    dr = dr.dropna(subset=["smiles"])
    dr["sev"] = pd.to_numeric(dr["SeverityClass"], errors="coerce")
    dr = dr.dropna(subset=["sev"]); dr["sev"] = dr["sev"].astype(int)
    dr["ik"] = dr["smiles"].map(smi2ikey14)
    dr = dr.dropna(subset=["ik"]).drop_duplicates("ik").reset_index(drop=True)
    return dr[["name_lower", "sev", "smiles", "ik"]]


def build_engine():
    sys.path[:0] = [f"{MULTIDCP}/models", f"{MULTIDCP}/utils"]
    import multidcp
    from data_utils import convert_smile_to_feature, create_mask_feature, read_gene
    from multidcp_ae_utils import initialize_model_registry
    reg = initialize_model_registry()
    reg.update({"num_gene": 978, "cell_id_input_dim": 978, "pert_idose_input_dim": 1,
                "dropout": 0.3, "linear_encoder_flag": True})
    m = multidcp.MultiDCPOriginal(device=DEVICE, model_param_registry=reg).to(DEVICE).double()
    m.load_state_dict(torch.load(f"{CKPT_DIR}/multidcp_liver_linear_sd42.pt", map_location=DEVICE))
    m.eval()
    gene_t = read_gene(GENE_VECTOR, DEVICE)

    def feat(s):
        d = convert_smile_to_feature(list(s), DEVICE); return d, create_mask_feature(d, DEVICE)

    def fwd(d, mask, basal):
        dose = torch.ones(basal.shape[0], 1, dtype=torch.float64, device=DEVICE)
        o = m(d, gene_t, mask, basal, dose, epoch=0)
        return o[0] if isinstance(o, tuple) else o
    return feat, fwd


@torch.no_grad()
def predicted_de(smis):
    d = np.load(f"{REPO}/data/processed/organ_train/liver.npz", allow_pickle=True)
    basal_vec = d["x2"].astype(np.float64).mean(0)
    feat, fwd = build_engine()
    out = np.full((len(smis), 978), np.nan)
    for i, s in enumerate(smis):
        try:
            dd, mm = feat([s])
        except Exception:
            continue
        basal = torch.as_tensor(basal_vec[None], dtype=torch.float64, device=DEVICE)
        out[i] = fwd(dd, mm, basal).cpu().numpy()[0] - basal_vec
    return out


def measured_de_map():
    d = np.load(f"{REPO}/data/processed/organ_train/liver.npz", allow_pickle=True)
    de = d["x1"].astype(np.float64) - d["x2"].astype(np.float64)
    ik = np.array([smi2ikey14(s) or "NA" for s in d["smiles"]])
    return {k: de[ik == k].mean(0) for k in np.unique(ik) if k != "NA"}


def coarse_bin(sev):
    # stratification bins so folds are balanced despite the imbalanced tail
    b = np.zeros_like(sev)
    b[(sev >= 1) & (sev <= 3)] = 1
    b[(sev >= 4) & (sev <= 6)] = 2
    b[sev >= 7] = 3
    return b


def clf_factory(kind):
    if kind == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000, class_weight="balanced"))
    return make_pipeline(StandardScaler(),
                         RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=0))


def evaluate(X, y, kind, seed):
    strat = coarse_bin(y)
    pred = np.zeros(len(y), int)
    exp = np.zeros(len(y))  # expected severity (proba-weighted) for a smooth rank
    for tr, te in StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(X, strat):
        clf = clf_factory(kind); clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
        proba = clf.predict_proba(X[te])
        exp[te] = proba @ clf.classes_.astype(float)
    return dict(
        accuracy=accuracy_score(y, pred),
        within1=float(np.mean(np.abs(pred - y) <= 1)),
        bal_acc=balanced_accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro"),
        qwk=cohen_kappa_score(y, pred, weights="quadratic"),
        spearman=float(spearmanr(exp, y).statistic),
        mae=mean_absolute_error(y, pred),
    )


def run_cohort(feats, y, name):
    rows = []
    for fname, X in feats.items():
        for kind in ("logreg", "rf"):
            ms = [evaluate(X, y, kind, s) for s in range(args.seeds)]
            agg = {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms])))
                   for k in ms[0]}
            rows.append(dict(cohort=name, feature=fname, model=kind, n=len(y), **agg))
            print(f"  [{name}] {fname:12s} {kind:7s} acc={agg['accuracy'][0]:.3f} "
                  f"within1={agg['within1'][0]:.3f} QWK={agg['qwk'][0]:.3f} "
                  f"rho={agg['spearman'][0]:.3f} MAE={agg['mae'][0]:.3f}")
    return rows


def main():
    dr = load_severity()
    smis = dr["smiles"].tolist()
    sev = dr["sev"].to_numpy(int)
    print(f"severity drugs with SMILES: n={len(dr)}  sev dist={np.bincount(sev).tolist()}")
    Xcb = np.asarray(get_encoder("chemberta")(smis)[0], float)
    Xpred = predicted_de(smis)
    mde = measured_de_map()
    ok_pred = np.isfinite(Xpred).all(1)

    rows = []
    # ---- predicted-DE triple (all severity drugs) ----
    for cohort, mask in [("FULL", ok_pred), ("POS(sev>=1)", ok_pred & (sev >= 1))]:
        idx = np.where(mask)[0]
        feats = {"structure": Xcb[idx], "expression(pred DE)": Xpred[idx],
                 "both": np.hstack([Xcb[idx], Xpred[idx]])}
        rows += run_cohort(feats, sev[idx], cohort)

    # ---- measured-DE triple (LINCS overlap — the honest omics test) ----
    ov = dr["ik"].isin(mde).to_numpy() & ok_pred
    Xmeas_full = np.full((len(dr), 978), np.nan)
    Xmeas_full[ov] = np.vstack([mde[k] for k in dr["ik"][ov]])
    for cohort, mask in [("FULL/measured-overlap", ov), ("POS/measured-overlap", ov & (sev >= 1))]:
        idx = np.where(mask)[0]
        feats = {"structure": Xcb[idx], "expression(meas DE)": Xmeas_full[idx],
                 "both": np.hstack([Xcb[idx], Xmeas_full[idx]])}
        rows += run_cohort(feats, sev[idx], cohort)

    write(rows)


def write(rows):
    df = pd.DataFrame(rows)
    cohorts = ["FULL", "POS(sev>=1)", "FULL/measured-overlap", "POS/measured-overlap"]
    L = ["# DILIrank SeverityClass (0-8) prediction: structure vs expression vs both", "",
         "All arms predict the ordinal severity CLASS. Liver only. chemberta structure; expression = "
         "engine predicted DE (near-circular caveat) or measured LINCS DE (overlap, the honest omics "
         "test). Drug-disjoint 5-fold x 3 seeds; folds stratified on coarse severity bins. "
         "mean +/- std over seeds.", "",
         "## Table 1 - prediction accuracy", "",
         "exact = exact 9-class accuracy; within-1 = predicted class within +/-1 of true (ordinal-tolerant).", "",
         "| cohort | feature | model | n | exact accuracy | within-1 accuracy |",
         "|---|---|---|---|---|---|"]
    def cell(r, k): return f"{r[k][0]:.3f}±{r[k][1]:.3f}"
    for c in cohorts:
        for _, r in df[df.cohort == c].iterrows():
            L.append(f"| {c} | {r['feature']} | {r['model']} | {r['n']} | "
                     f"{cell(r,'accuracy')} | {cell(r,'within1')} |")
    L += ["", "## Table 2 - model-selection metrics", "",
          "QWK = quadratic-weighted Cohen kappa (ordinal agreement; 0=chance, 1=perfect); rho = "
          "Spearman of proba-weighted expected severity vs true; MAE = mean |pred-true| classes "
          "(lower better); balanced acc + macro-F1 handle class imbalance. Best predictive model = "
          "highest QWK / rho / within-1, lowest MAE.", "",
          "| cohort | feature | model | balanced acc | macro-F1 | QWK | Spearman rho | MAE |",
          "|---|---|---|---|---|---|---|---|"]
    for c in cohorts:
        for _, r in df[df.cohort == c].iterrows():
            L.append(f"| {c} | {r['feature']} | {r['model']} | {cell(r,'bal_acc')} | "
                     f"{cell(r,'macro_f1')} | {cell(r,'qwk')} | {cell(r,'spearman')} | {cell(r,'mae')} |")
    L += ["", "## Reading",
          "- Compare the three feature arms within a cohort+model: does expression or both beat "
          "structure on QWK / rho / within-1 (the ordinal-quality metrics)?",
          "- FULL includes SeverityClass=0, so structure's edge there partly reflects the binary "
          "is-it-toxic split; POS(sev>=1) is the cleaner 'grade how bad among toxicants' test.",
          "- measured-overlap rows are the honest omics test (real biology, smaller n); predicted-DE "
          "rows are the deployable-from-SMILES feature (near-circular with structure).", ""]
    Path(f"{REPO}/results/tables/P4_severity_class.md").write_text("\n".join(L) + "\n")
    df.to_csv(f"{REPO}/results/tables/P4_severity_class.csv", index=False)
    print("\nwrote results/tables/P4_severity_class.md")


if __name__ == "__main__":
    main()
