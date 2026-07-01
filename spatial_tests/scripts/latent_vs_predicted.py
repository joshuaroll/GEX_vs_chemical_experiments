#!/usr/bin/env python
"""Head-to-head: MultiDCP LATENT (global_features, 306-d) vs full PREDICTED DE
(rule-B treated - inert-control, 10,716 genes; and the 978-landmark subset) as
the expression feature for per-organ toxicity prediction.

Both features come off the SAME forward pass in the SAME fixed periportal basal,
so the only thing that differs is which representation is read out. Structure
(ChemBERTa) is carried as a reference column. Same fixed head (L2 logistic
regression, balanced), drug-level 5-fold x 5 seeds, per organ.

Answers: which representation of the model's drug response predicts DILI better?
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests")
sys.path.insert(0, str(REPO))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from scripts.three_way_comparison import load_liver, load_kidney, load_heart, smi2ikey14
from scripts.extract_multidcp_latent import MultiDCPLatentExtractor, load_reference_basal
from scripts.landmark_overlap import landmark_symbols
from src.spatial.structure_encoders import get_encoder

FIGDIR = Path("/raid/home/joshua/claude_memory/downstream_2026/visuals/three_way_comparison")
CACHE = REPO / "data/processed/latent_cache"
ORGANS = {"liver": load_liver, "kidney": load_kidney, "heart": load_heart}
N_SEEDS, N_FOLDS = 5, 5
INERT = "C"  # rule-B inert control reference (matches region_signature.INERT_CONTROL_SMILES)


def drug_table(loader):
    df = loader().dropna(subset=["smiles"]).copy()
    df["ikey14"] = df["smiles"].map(smi2ikey14)
    df = df.dropna(subset=["ikey14"]).drop_duplicates("ikey14")
    return df[["ikey14", "label", "smiles"]].reset_index(drop=True)


def cv_auroc(X, y):
    aurs = []
    for seed in range(N_SEEDS):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced"))
            clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
        aurs.append(roc_auc_score(y, oof))
    return float(np.mean(aurs)), float(np.std(aurs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="chemberta")
    ap.add_argument("--gpu", default="1")
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]  # shield lazy cache_region_de module-level argparse
    CACHE.mkdir(parents=True, exist_ok=True)

    tables = {o: drug_table(l) for o, l in ORGANS.items()}
    uniq = pd.concat(tables.values()).drop_duplicates("ikey14").set_index("ikey14")["smiles"]
    keys = list(uniq.index); smis = [uniq[k] for k in keys]
    print(f"unique drugs across organs: {len(keys)}")

    # ---- latent (306) + predicted DE (10,716), one forward per drug, cached ----
    cache_f = CACHE / "latent_and_de.npz"
    if cache_f.exists():
        z = np.load(cache_f, allow_pickle=True)
        gf = {k: g for k, g in zip(z["keys"], z["G"])}
        de = {k: d for k, d in zip(z["keys"], z["DE"])}
        gene_order = list(z["gene_order"])
        print(f"loaded cache: {len(gf)} drugs, DE dim {z['DE'].shape[1]}")
    else:
        print("building extractor + control pass ...")
        ext = MultiDCPLatentExtractor(gpu=args.gpu)
        print(f"strict-load 0/0: {ext.strict_ok}")
        ext.set_basal(load_reference_basal(ext))
        gene_order = list(ext.gene_order)
        _, ctrl = ext.extract_full(INERT)        # rule-B control, once
        gf, de = {}, {}
        for i, k in enumerate(keys):
            try:
                g, treated = ext.extract_full(uniq[k])
                gf[k] = g; de[k] = (treated - ctrl).astype(np.float32)
            except Exception as e:
                print(f"  skip {k}: {e}")
            if (i + 1) % 100 == 0: print(f"  {i+1}/{len(keys)}")
        good = [k for k in keys if k in de]
        np.savez(cache_f, keys=np.array(good),
                 G=np.vstack([gf[k] for k in good]),
                 DE=np.vstack([de[k] for k in good]),
                 gene_order=np.array(gene_order))
        print(f"cached {len(good)} drugs -> {cache_f.name}")

    # landmark subset indices within the 10,716 model gene order
    lm = landmark_symbols()
    lm_idx = np.array([i for i, gname in enumerate(gene_order) if gname in lm])
    print(f"landmark genes in model space: {len(lm_idx)}/{len(lm)}")

    # ---- structure latent ----
    print(f"structure encoder: {args.encoder}")
    enc = get_encoder(args.encoder)
    Semb, ok = enc(smis)
    struct = {k: Semb[i] for i, k in enumerate(keys) if ok[i] and k in de}

    feats = {
        "structure": lambda k: struct[k],
        "latent306": lambda k: gf[k],
        "predDE_full": lambda k: de[k],
        "predDE_lm978": lambda k: de[k][lm_idx],
    }

    rows = []
    for organ, tab in tables.items():
        tab = tab[tab["ikey14"].isin(struct) & tab["ikey14"].isin(de)]
        y = tab["label"].to_numpy(int); ks = list(tab["ikey14"])
        row = dict(organ=organ, n=len(y), pos=int(y.sum()), neg=int((1-y).sum()))
        for name, fn in feats.items():
            X = np.vstack([fn(k) for k in ks]).astype(float)
            m, s = cv_auroc(X, y)
            row[name] = m; row[f"{name}_sd"] = s
        rows.append(row)
        print(f"[{organ}] n={row['n']} ({row['pos']}/{row['neg']})  "
              f"struct={row['structure']:.3f}  latent306={row['latent306']:.3f}  "
              f"predDE_full={row['predDE_full']:.3f}  predDE_lm978={row['predDE_lm978']:.3f}")

    df = pd.DataFrame(rows)

    # ---- figure: latent vs predicted-DE per organ (structure as reference) ----
    organs = list(ORGANS); fig, ax = plt.subplots(figsize=(9.2, 5.0))
    series = [("structure", "structure (ChemBERTa)", "#7fb3d5"),
              ("latent306", "MultiDCP latent (306)", "#f5b041"),
              ("predDE_full", "predicted DE (10,716)", "#2c7fb8"),
              ("predDE_lm978", "predicted DE (978 landmark)", "#1a5276")]
    w = 0.2
    for i, (key, lab, c) in enumerate(series):
        sub = df.set_index("organ").loc[organs]
        x = np.arange(len(organs)) + (i - 1.5) * w
        ax.bar(x, sub[key], w, yerr=sub[f"{key}_sd"], capsize=2, label=lab, color=c)
        for xi, v in zip(x, sub[key]): ax.text(xi, v + .008, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set_xticks(np.arange(len(organs)))
    ax.set_xticklabels([f"{o}\n(n={int(df[df.organ==o].n.iloc[0])})" for o in organs])
    ax.set_ylim(0.4, 0.85); ax.set_ylabel("drug-level CV AUROC (mean +/- std, 5 seeds)")
    ax.set_title("Latent vs predicted-DE as the expression feature (per organ)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIGDIR / "latent_vs_predicted_by_organ.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- markdown ----
    L = ["# Latent vs predicted-DE: which representation predicts toxicity better?", "",
         f"Both features come off the SAME MultiDCP forward in the SAME fixed periportal basal. "
         f"Expression options: **latent** = global_features (306) | **predicted DE** = rule-B "
         f"treated(drug) - treated(inert '{INERT}'), full 10,716 genes and the 978-landmark subset "
         f"({len(lm_idx)} landmarks present in model space). Structure (**{args.encoder}**) is a "
         f"reference. Head = L2 logistic regression, drug-level 5-fold x 5 seeds.", "",
         "| Organ | n (pos/neg) | structure | latent (306) | predicted DE (10,716) | predicted DE (978 lm) |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['organ']} | {r['n']} ({r['pos']}/{r['neg']}) | "
                 f"{r['structure']:.3f}±{r['structure_sd']:.3f} | "
                 f"{r['latent306']:.3f}±{r['latent306_sd']:.3f} | "
                 f"{r['predDE_full']:.3f}±{r['predDE_full_sd']:.3f} | "
                 f"{r['predDE_lm978']:.3f}±{r['predDE_lm978_sd']:.3f} |")
    L += ["", "## Notes",
          "- Same forward, same basal: the only difference is which representation is read out "
          "(gating-network input latent vs the gene-output DE). A latent-vs-DE gap is therefore "
          "purely representational, not a context confound.",
          "- Predicted DE is rule-B (treated - inert-control), the spatial-arm DE convention; both "
          "operands stay in the model's output manifold.",
          "- Single fixed periportal basal (no within-organ region resolution yet); this isolates "
          "the representation question from the location-combination question.", "",
          "## Figure", f"- `{FIGDIR}/latent_vs_predicted_by_organ.png`", ""]
    (REPO / "results/tables/P1_latent_vs_predicted.md").write_text("\n".join(L))
    df.to_csv(REPO / "results/tables/P1_latent_vs_predicted.csv", index=False)
    print("\nwrote results/tables/P1_latent_vs_predicted.md + figure")


if __name__ == "__main__":
    main()
