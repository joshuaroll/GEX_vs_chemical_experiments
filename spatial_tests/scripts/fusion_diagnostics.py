#!/usr/bin/env python
"""Is there complementary signal between structure and omics to fuse at all?

Reframes the negative. The predicted-DE arm is near-circular: the engine is a map
SMILES -> DE, so predicted DE carries no information beyond SMILES, and structure is
another SMILES map. The real test of "omics adds info" needs MEASURED DE (independent
biology). Before engineering fusion, we ask the decisive question: is there any
complementary signal to fuse?

Per organ, on the measured-DE overlap set, with structure = ChemBERTa AND ECFP4:
  1. REDUNDANCY   how much of predicted/measured DE is linearly recoverable from structure
                  (Ridge multi-output, 5-fold R^2). predicted-DE R^2 -> 1 confirms circularity.
  2. COMPLEMENTARITY  per-drug OOF errors: is measured DE right where structure is wrong?
  3. FUSION UPPER/REAL BOUNDS  on the two probability streams (OOF):
        - structure alone
        - late fusion (mean prob), parameter-free
        - learned stacker (LR on [p_struct, p_meas], nested CV)  <- legitimate "combine properly"
        - ORACLE (per-drug pick the prob closer to the truth) <- unachievable upper bound on ANY fusion
     If oracle ~ structure -> no fusion can help (negative is real).
     If oracle >> structure but stacker ~ structure -> fusion design is the lever.
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
from pathlib import Path
import torch

REPO = "/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests"
MULTIDCP = "/raid/home/joshua/projects/MultiDCP/MultiDCP"
GENE_VECTOR = f"{MULTIDCP}/data/gene_vector.csv"
CKPT_DIR = f"{REPO}/results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, REPO)

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, KFold
from sklearn.metrics import roc_auc_score, r2_score
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
    d = np.load(f"{REPO}/data/processed/organ_train/{organ}.npz", allow_pickle=True)
    de = (d["x1"].astype(np.float64) - d["x2"].astype(np.float64))
    ik = np.array([smi2ikey14(s) or "NA" for s in d["smiles"]])
    return {k: de[ik == k].mean(0) for k in np.unique(ik) if k != "NA"}


def redundancy_r2(Xs, Yde, seed=0):
    """Mean 5-fold R^2 of a Ridge multi-output map structure -> DE (variance-weighted)."""
    keep = Yde.std(0) > 1e-8
    Y = Yde[:, keep]
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    pred = cross_val_predict(pipe, Xs, Y, cv=KFold(5, shuffle=True, random_state=seed))
    # variance-weighted mean R2 across output dims
    return float(r2_score(Y, pred, multioutput="variance_weighted"))


def oof(X, y, seed):
    o = np.zeros(len(y))
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); o[te] = clf.predict_proba(X[te])[:, 1]
    return o


def stacker_auroc(ps, pe, y, seed):
    """Legitimate learned fusion: LR on the two OOF prob streams, evaluated by CV on those streams."""
    Z = np.column_stack([ps, pe])
    o = np.zeros(len(y))
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(Z, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Z[tr], y[tr]); o[te] = clf.predict_proba(Z[te])[:, 1]
    return roc_auc_score(y, o)


def oracle_auroc(ps, pe, y):
    """Per-drug pick the prob closer to the true label. Unachievable upper bound on ANY fusion."""
    po = np.where(np.abs(y - ps) <= np.abs(y - pe), ps, pe)
    return roc_auc_score(y, po)


def main():
    cb = get_encoder("chemberta"); fp = get_encoder("ecfp4")
    rows, red = [], []
    for organ in args.organs.split(","):
        tox = LOADERS[organ]().dropna(subset=["smiles"]).copy()
        tox["ik"] = tox["smiles"].map(smi2ikey14)
        tox = tox.dropna(subset=["ik"]).drop_duplicates("ik").reset_index(drop=True)
        mde = measured_de_map(organ)
        tox = tox[tox["ik"].isin(mde)].reset_index(drop=True)
        smis = tox["smiles"].tolist()
        Xpred = predicted_de(organ, smis)
        Scb, okc = cb(smis); Sfp, okf = fp(smis)
        Scb, Sfp = np.asarray(Scb), np.asarray(Sfp)
        good = np.asarray(okc, bool) & np.asarray(okf, bool) & np.isfinite(Xpred).all(1)
        tox = tox[good].reset_index(drop=True)
        y = tox["label"].to_numpy(int)
        Xpred = Xpred[good]
        Xmeas = np.vstack([mde[k] for k in tox["ik"]])
        struct = {"chemberta": Scb[good].astype(float), "ecfp4": Sfp[good].astype(float)}
        print(f"\n[{organ}] measured-overlap n={len(y)} ({int(y.sum())}/{int((1-y).sum())})")

        # 1) redundancy of predicted / measured DE w.r.t. each structure encoder
        for sname, Xs in struct.items():
            r_pred = redundancy_r2(Xs, Xpred)
            r_meas = redundancy_r2(Xs, Xmeas)
            red.append(dict(organ=organ, structure=sname, r2_struct_to_predDE=r_pred,
                            r2_struct_to_measDE=r_meas))
            print(f"  redundancy R^2  {sname}->predDE={r_pred:+.3f}  {sname}->measDE={r_meas:+.3f}")

        # 2+3) complementarity + fusion bounds, structure vs MEASURED DE (the real test)
        for sname, Xs in struct.items():
            au_s, au_m, au_late, au_stk, au_orc = [], [], [], [], []
            comp = None
            for seed in range(N_SEEDS):
                ps = oof(Xs, y, seed); pm = oof(Xmeas, y, seed)
                au_s.append(roc_auc_score(y, ps)); au_m.append(roc_auc_score(y, pm))
                au_late.append(roc_auc_score(y, 0.5 * (ps + pm)))
                au_stk.append(stacker_auroc(ps, pm, y, seed))
                au_orc.append(oracle_auroc(ps, pm, y))
                if seed == 0:
                    es = (np.round(ps) != y); em = (np.round(pm) != y)
                    comp = dict(struct_wrong=int(es.sum()), meas_wrong=int(em.sum()),
                                both_wrong=int((es & em).sum()),
                                meas_right_where_struct_wrong=int((es & ~em).sum()),
                                struct_right_where_meas_wrong=int((~es & em).sum()),
                                err_corr=float(np.corrcoef(es.astype(float), em.astype(float))[0, 1]))
            m = lambda a: (float(np.mean(a)), float(np.std(a)))
            row = dict(organ=organ, structure=sname, n=len(y),
                       struct=m(au_s), meas=m(au_m), late=m(au_late), stacker=m(au_stk),
                       oracle=m(au_orc), **comp)
            rows.append(row)
            print(f"  fuse[{sname}]  struct={m(au_s)[0]:.3f}  meas={m(au_m)[0]:.3f}  "
                  f"late={m(au_late)[0]:.3f}  stacker={m(au_stk)[0]:.3f}  ORACLE={m(au_orc)[0]:.3f} "
                  f"| meas_right_where_struct_wrong={comp['meas_right_where_struct_wrong']}/"
                  f"{comp['struct_wrong']}  err_corr={comp['err_corr']:+.2f}")

    write(rows, red)


def write(rows, red):
    L = ["# Fusion diagnostics: is there complementary structure+omics signal to fuse?", "",
         "Measured-DE overlap set per organ. structure = ChemBERTa / ECFP4. "
         "REDUNDANCY = 5-fold variance-weighted R^2 of a Ridge map structure->DE (how much of the "
         "omics vector is just structure). FUSION on the two OOF probability streams: late = mean prob; "
         "stacker = LR on [p_struct, p_meas] (legitimate); ORACLE = per-drug pick the prob closer to "
         "the label (unachievable upper bound on ANY fusion). 5 seeds.", "",
         "## Redundancy (how much of DE is linearly recoverable from structure)", "",
         "| organ | structure | R^2 struct->predicted DE | R^2 struct->measured DE |",
         "|---|---|---|---|"]
    for r in red:
        L.append(f"| {r['organ']} | {r['structure']} | {r['r2_struct_to_predDE']:+.3f} | "
                 f"{r['r2_struct_to_measDE']:+.3f} |")
    L += ["", "> predicted-DE R^2 near 1 = predicted DE is structure re-encoded (the near-circular arm). "
          "measured-DE R^2 well below 1 = real biology carries structure-independent variance.", "",
          "## Fusion of structure + MEASURED DE (the real test)", "",
          "| organ | structure | n | structure | measured | late fusion | learned stacker | ORACLE (UB) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda t: f"{t[0]:.3f}±{t[1]:.3f}"
        L.append(f"| {r['organ']} | {r['structure']} | {r['n']} | {f(r['struct'])} | {f(r['meas'])} | "
                 f"{f(r['late'])} | {f(r['stacker'])} | {f(r['oracle'])} |")
    L += ["", "## Complementarity (seed-0 OOF errors)", "",
          "| organ | structure | struct wrong | meas wrong | both wrong | meas right where struct wrong | err corr |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['organ']} | {r['structure']} | {r['struct_wrong']} | {r['meas_wrong']} | "
                 f"{r['both_wrong']} | {r['meas_right_where_struct_wrong']} | {r['err_corr']:+.2f} |")
    L += ["", "## Reading",
          "- **ORACLE ~ structure** => no per-drug combination of these two streams beats structure; "
          "the negative is real and fusion engineering cannot rescue it on this data.",
          "- **ORACLE >> structure but stacker ~ structure** => complementary signal EXISTS but linear "
          "prob-fusion misses it => invest in fusion design (MLP / gating / interactions).",
          "- **stacker > structure** => a better fusion already helps; report it.",
          "- err_corr near 1 = the two models fail on the same drugs (no complementarity); near 0 = "
          "independent failures (fusable).", ""]
    Path(f"{REPO}/results/tables/P4_fusion_diagnostics.md").write_text("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(f"{REPO}/results/tables/P4_fusion_diagnostics.csv", index=False)
    pd.DataFrame(red).to_csv(f"{REPO}/results/tables/P4_fusion_redundancy.csv", index=False)
    print("\nwrote results/tables/P4_fusion_diagnostics.md")


if __name__ == "__main__":
    main()
