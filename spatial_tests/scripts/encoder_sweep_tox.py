#!/usr/bin/env python
"""Encoder sweep: is the structure > expression negative robust to the structure encoder?

The P4 three-way comparison (P4_stage2_toxicity) rode on a single structure encoder
(ChemBERTa). This sweeps the structure arm across learned embeddings and classical
fingerprints while holding everything else fixed:

    structure  = <encoder>(smiles)          # swept: chemberta / unimol_v1 / unimol_v2 /
                                             #        ecfp4 / ecfp6 / maccs / atompair /
                                             #        topotorsion / rdkit_fp
    expression = organ engine predicted DE   # CONSTANT per organ (DE = engine(smi,basal) - basal)
    both       = concat(structure, expression)

Same small fixed head (L2 logistic regression, balanced), drug-disjoint 5-fold x 5 seeds.
All encoders are scored on the IDENTICAL drug set per organ (engine-featurizable AND
valid for every encoder), so the expression arm is byte-identical across the sweep and
the only thing that moves is the structure representation. Reports, per (organ, encoder):
structure / expression / both AUROC and the +expression lift over structure with a paired
bootstrap CI. If structure stays >= expression across encoders and no encoder yields a
CI-clearing +lift, the "expression is structure-ceilinged" finding is not a ChemBERTa
artifact.
"""
from __future__ import annotations
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="1")
ap.add_argument("--arch", default="multidcp")
ap.add_argument("--engine-encoder", default="linear",
                help="engine's internal linear_encoder_flag + checkpoint tag (NOT the structure arm)")
ap.add_argument("--encoders",
                default="chemberta,ecfp4,ecfp6,maccs,atompair,topotorsion,rdkit_fp,unimol_v1,unimol_v2")
ap.add_argument("--organs", default="liver,kidney")
ap.add_argument("--encode-only", action="store_true",
                help="only compute+cache structure embeddings (one encoder per process avoids "
                     "GPU OOM from holding several encoder models at once); skip engine + eval")
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
CACHE = f"{REPO}/data/processed/latent_cache"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, REPO)
Path(CACHE).mkdir(parents=True, exist_ok=True)

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
    """The predicted-DE organ engine (MultiDCP original, linear cell encoder)."""
    sys.path[:0] = [f"{MULTIDCP}/models", f"{MULTIDCP}/utils"]
    import multidcp
    from data_utils import convert_smile_to_feature, create_mask_feature, read_gene
    from multidcp_ae_utils import initialize_model_registry
    reg = initialize_model_registry()
    reg.update({"num_gene": 978, "cell_id_input_dim": 978, "pert_idose_input_dim": 1,
                "dropout": 0.3, "linear_encoder_flag": args.engine_encoder == "linear"})
    model = multidcp.MultiDCPOriginal(device=DEVICE, model_param_registry=reg).to(DEVICE).double()
    gene_t = read_gene(GENE_VECTOR, DEVICE)

    def feat(s):
        d = convert_smile_to_feature(list(s), DEVICE)
        return d, create_mask_feature(d, DEVICE)

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
def predict_de_full(organ, smis):
    """Predicted DE expanded to full input order: (predDE_full [n,978], oke [n] bool)."""
    d = np.load(f"{REPO}/data/processed/organ_train/{organ}.npz", allow_pickle=True)
    basal_vec = d["x2"].astype(np.float64).mean(0)
    model, feat, fwd = build_engine()
    model.load_state_dict(torch.load(
        f"{CKPT_DIR}/{args.arch}_{organ}_{args.engine_encoder}_sd42.pt", map_location=DEVICE))
    model.eval()
    n = len(smis)
    full = np.zeros((n, 978), np.float32)
    oke = np.zeros(n, bool)
    for i in range(0, n, 64):
        idx = list(range(i, min(i + 64, n)))
        good = [j for j in idx if _ok(feat, smis[j])]
        if not good:
            continue
        drug, mask = feat([smis[j] for j in good])
        basal = torch.as_tensor(np.tile(basal_vec, (len(good), 1)),
                                dtype=torch.float64, device=DEVICE)
        pred = fwd(drug, mask, basal).cpu().numpy()
        for k, j in enumerate(good):
            full[j] = (pred[k] - basal_vec).astype(np.float32); oke[j] = True
    return full, oke


def encode(organ, name, smis):
    """Structure embedding [n,d] + ok [n], cached per (organ, encoder)."""
    cf = f"{CACHE}/enc_{organ}_{name}.npz"
    if os.path.exists(cf):
        z = np.load(cf); return z["emb"], z["ok"]
    emb, ok = get_encoder(name)(smis)
    emb, ok = np.asarray(emb, np.float32), np.asarray(ok, bool)
    np.savez(cf, emb=emb, ok=ok)
    return emb, ok


def cv_oof(X, y, seed):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
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


def organ_smis(organ):
    df = LOADERS[organ]().dropna(subset=["smiles"]).copy()
    df["ikey14"] = df["smiles"].map(smi2ikey14)
    df = df.dropna(subset=["ikey14"]).drop_duplicates("ikey14").reset_index(drop=True)
    return df


def main():
    encs = [e.strip() for e in args.encoders.split(",") if e.strip()]

    if args.encode_only:
        for organ in args.organs.split(","):
            smis = organ_smis(organ)["smiles"].tolist()
            for name in encs:
                emb, ok = encode(organ, name, smis)
                print(f"[encode] {organ}/{name}: {emb.shape} ok={int(ok.sum())}/{len(ok)}")
        return

    rows = []
    for organ in args.organs.split(","):
        df = organ_smis(organ)
        smis = df["smiles"].tolist()

        predDE, oke = predict_de_full(organ, smis)
        emb_ok = {}
        common = oke.copy()
        for name in encs:
            emb, ok = encode(organ, name, smis)
            emb_ok[name] = (emb, ok)
            common &= ok
        y = df["label"].to_numpy(int)[common]
        Xexp = predDE[common].astype(float)
        print(f"[{organ}] common drug set n={common.sum()} "
              f"({int(y.sum())}/{int((1 - y).sum())}); expression dim={Xexp.shape[1]}")

        # expression arm is identical across encoders -> evaluate once
        exp_res, exp_oof = evaluate(y, {"expression": Xexp})
        exp_auroc = float(np.mean(exp_res["expression"]["auroc"]))
        exp_auroc_sd = float(np.std(exp_res["expression"]["auroc"]))

        for name in encs:
            emb, _ = emb_ok[name]
            Xs = emb[common].astype(float)
            res, oof0 = evaluate(y, {"structure": Xs, "both": np.hstack([Xs, Xexp])})
            oof0["expression"] = exp_oof["expression"]
            lift_both = boot_lift(y, oof0["structure"], oof0["both"])
            lift_exp = boot_lift(y, oof0["structure"], oof0["expression"])
            row = dict(
                organ=organ, encoder=name, dim=Xs.shape[1], n=int(common.sum()),
                pos=int(y.sum()), neg=int((1 - y).sum()),
                structure_auroc=float(np.mean(res["structure"]["auroc"])),
                structure_auroc_sd=float(np.std(res["structure"]["auroc"])),
                expression_auroc=exp_auroc, expression_auroc_sd=exp_auroc_sd,
                both_auroc=float(np.mean(res["both"]["auroc"])),
                both_auroc_sd=float(np.std(res["both"]["auroc"])),
                structure_auprc=float(np.mean(res["structure"]["auprc"])),
                both_auprc=float(np.mean(res["both"]["auprc"])),
                lift_both=lift_both[0], lift_both_lo=lift_both[1], lift_both_hi=lift_both[2],
                lift_exp=lift_exp[0], lift_exp_lo=lift_exp[1], lift_exp_hi=lift_exp[2],
            )
            rows.append(row)
            print(f"  {name:12s} dim={row['dim']:4d} | struct={row['structure_auroc']:.3f} "
                  f"expr={exp_auroc:.3f} both={row['both_auroc']:.3f} | "
                  f"+lift(both-struct)={lift_both[0]:+.3f} [{lift_both[1]:+.3f},{lift_both[2]:+.3f}]")

    write_tables(rows, encs)


def write_tables(rows, encs):
    L = ["# Encoder sweep: structure vs DE vs DE+structure (does the negative survive the encoder?)",
         "",
         f"structure = swept encoder | expression = **{args.arch}** organ-engine predicted DE "
         "(DE = engine(smiles, organ-mean basal) - basal), IDENTICAL across encoders per organ | "
         "head = L2 logistic regression (balanced), drug-disjoint 5-fold x 5 seeds. Every encoder "
         "scored on the same drug set per organ (engine-featurizable AND valid for all encoders).", ""]
    for organ in args.organs.split(","):
        orows = [r for r in rows if r["organ"] == organ]
        if not orows:
            continue
        r0 = orows[0]
        L += [f"## {organ} (n={r0['n']}, {r0['pos']}/{r0['neg']}; expression AUROC "
              f"{r0['expression_auroc']:.3f}±{r0['expression_auroc_sd']:.3f}, constant)", "",
              "| encoder | dim | structure AUROC | both AUROC | struct − expr | "
              "+lift (both − struct) | 95% CI |",
              "|---|---|---|---|---|---|---|"]
        for r in sorted(orows, key=lambda x: -x["structure_auroc"]):
            gap = r["structure_auroc"] - r["expression_auroc"]
            L.append(f"| {r['encoder']} | {r['dim']} | "
                     f"{r['structure_auroc']:.3f}±{r['structure_auroc_sd']:.3f} | "
                     f"{r['both_auroc']:.3f}±{r['both_auroc_sd']:.3f} | {gap:+.3f} | "
                     f"**{r['lift_both']:+.3f}** | [{r['lift_both_lo']:+.3f}, {r['lift_both_hi']:+.3f}] |")
        L.append("")
    L += ["## Reading",
          "- **struct − expr** > 0 for every encoder = structure beats predicted expression regardless of "
          "representation; the expression ceiling is not a ChemBERTa artifact.",
          "- **+lift (both − struct)** with a CI that includes 0 = adding predicted DE to structure does not "
          "help. A single encoder with a CI clearing 0 would be the counter-signal to chase.",
          "- expression AUROC is constant within an organ by construction (same engine DE, same drug set, "
          "same head) — only the structure representation moves across rows.",
          "- Encoders: chemberta (learned LM), unimol_v1/v2 (3D pretrained), ecfp4/ecfp6 (Morgan r2/r3), "
          "maccs (167 keys), atompair, topotorsion, rdkit_fp (topological).", ""]
    Path(f"{REPO}/results/tables/P4_encoder_sweep.md").write_text("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(f"{REPO}/results/tables/P4_encoder_sweep.csv", index=False)
    print("\nwrote results/tables/P4_encoder_sweep.md")


if __name__ == "__main__":
    main()
