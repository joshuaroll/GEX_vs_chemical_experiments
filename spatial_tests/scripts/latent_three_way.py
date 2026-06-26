#!/usr/bin/env python
"""Three-way comparison with LATENT representations, split by body region.

structure         = learned molecular-encoder embedding (default chemberta; swappable
                    to unimol_v1/v2/ecfp4 via --encoder)
expression        = MultiDCP-CheMoE latent global_features [306] (model's internal
                    representation; per-drug signal is the 128-d drug block -- see note)
structure + expr  = concatenation

Same fixed toxicity head (L2 logistic regression, balanced), drug-level fair CV
(5 folds x 5 seeds), per organ. Headline: the +expression lift (both - structure)
with a paired bootstrap CI. Uses MultiDCP-PREDICTED latent (from SMILES), so it does
NOT need the measured-LINCS join -> larger drug sets than the measured comparison.

Brain still excluded (label-to-structure join blocked).
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
from src.spatial.structure_encoders import get_encoder

FIGDIR = Path("/raid/home/joshua/claude_memory/downstream_2026/visuals/three_way_comparison")
CACHE = REPO / "data/processed/latent_cache"
ORGANS = {"liver": load_liver, "kidney": load_kidney, "heart": load_heart}
N_SEEDS, N_FOLDS, N_BOOT = 5, 5, 10000


def drug_table(loader):
    df = loader().dropna(subset=["smiles"]).copy()
    df["ikey14"] = df["smiles"].map(smi2ikey14)
    df = df.dropna(subset=["ikey14"]).drop_duplicates("ikey14")
    return df[["ikey14", "label", "smiles"]].reset_index(drop=True)


def cv_oof(X, y, seed):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def evaluate(y, Xs, Xe):
    feats = {"structure": Xs, "expression": Xe, "structure+expression": np.hstack([Xs, Xe])}
    aur = {k: [] for k in feats}; oof0 = {}
    for k, X in feats.items():
        for s in range(N_SEEDS):
            oof = cv_oof(X, y, s); aur[k].append(roc_auc_score(y, oof))
            if s == 0: oof0[k] = oof
    return aur, oof0


def bootstrap_lift(y, oa, ob, seed=0):
    rng = np.random.RandomState(seed); n = len(y); d = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        d.append(roc_auc_score(y[idx], ob[idx]) - roc_auc_score(y[idx], oa[idx]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="chemberta", help="structure encoder (chemberta/ecfp4/unimol_v1/unimol_v2)")
    ap.add_argument("--gpu", default="3")
    args = ap.parse_args()
    # cache_region_de (imported lazily inside MultiDCPLatentExtractor) runs a strict
    # module-level argparse; clear our extra args from sys.argv so it doesn't choke.
    sys.argv = [sys.argv[0]]
    CACHE.mkdir(parents=True, exist_ok=True)

    tables = {o: drug_table(l) for o, l in ORGANS.items()}
    uniq = pd.concat(tables.values()).drop_duplicates("ikey14").set_index("ikey14")["smiles"]
    keys = list(uniq.index); smis = [uniq[k] for k in keys]
    print(f"unique drugs across organs: {len(keys)}")

    # ---- expression latent (MultiDCP global_features), cached ----
    exp_cache = CACHE / "global_features.npz"
    if exp_cache.exists():
        z = np.load(exp_cache, allow_pickle=True)
        gf = {k: v for k, v in zip(z["keys"], z["G"])}
        print(f"loaded {len(gf)} cached global_features")
    else:
        gf = {}
    todo = [k for k in keys if k not in gf]
    if todo:
        print(f"extracting MultiDCP latent for {len(todo)} drugs ...")
        ext = MultiDCPLatentExtractor(gpu=args.gpu)
        ext.set_basal(load_reference_basal(ext))
        for i, k in enumerate(todo):
            try:
                gf[k] = ext.extract(uniq[k])
            except Exception as e:
                gf[k] = None
            if (i + 1) % 100 == 0: print(f"  {i+1}/{len(todo)}")
        good = {k: v for k, v in gf.items() if v is not None}
        np.savez(exp_cache, keys=np.array(list(good)), G=np.vstack(list(good.values())))
        gf = good

    # ---- structure latent ----
    print(f"structure encoder: {args.encoder}")
    enc = get_encoder(args.encoder)
    Semb, ok = enc(smis)
    struct = {k: Semb[i] for i, k in enumerate(keys) if ok[i] and gf.get(k) is not None}

    # ---- per-organ evaluation ----
    rows = []
    for organ, tab in tables.items():
        tab = tab[tab["ikey14"].isin(struct) & tab["ikey14"].isin(gf)]
        y = tab["label"].to_numpy(int)
        Xs = np.vstack([struct[k] for k in tab["ikey14"]]).astype(float)
        Xe = np.vstack([gf[k] for k in tab["ikey14"]]).astype(float)
        aur, oof0 = evaluate(y, Xs, Xe)
        lift = bootstrap_lift(y, oof0["structure"], oof0["structure+expression"])
        rows.append(dict(organ=organ, n=len(y), pos=int(y.sum()), neg=int((1-y).sum()),
                         struct=np.mean(aur["structure"]), struct_sd=np.std(aur["structure"]),
                         expr=np.mean(aur["expression"]), expr_sd=np.std(aur["expression"]),
                         both=np.mean(aur["structure+expression"]), both_sd=np.std(aur["structure+expression"]),
                         lift=lift[0], lo=lift[1], hi=lift[2]))
        print(f"[{organ}] n={len(y)} ({int(y.sum())}/{int((1-y).sum())})  "
              f"struct={np.mean(aur['structure']):.3f} expr={np.mean(aur['expression']):.3f} "
              f"both={np.mean(aur['structure+expression']):.3f} +lift={lift[0]:+.3f} CI[{lift[1]:+.3f},{lift[2]:+.3f}]")

    df = pd.DataFrame(rows)
    # ---- figure ----
    organs = list(ORGANS); fig, ax = plt.subplots(figsize=(8.4, 5.0)); w = 0.26
    cols = {"struct": "#7fb3d5", "expr": "#f5b041", "both": "#2c7fb8"}
    for i, (key, lab) in enumerate([("struct", "structure"), ("expr", "expression (MultiDCP latent)"),
                                    ("both", "structure+expression")]):
        sub = df.set_index("organ").loc[organs]
        x = np.arange(len(organs)) + (i - 1) * w
        ax.bar(x, sub[key], w, yerr=sub[f"{key}_sd"], capsize=3, label=lab, color=cols[key])
        for xi, v in zip(x, sub[key]): ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set_xticks(np.arange(len(organs)))
    ax.set_xticklabels([f"{o}\n(n={int(df[df.organ==o].n.iloc[0])})" for o in organs])
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("drug-level CV AUROC (mean +/- std, 5 seeds)")
    ax.set_title(f"Latent three-way by organ ({args.encoder} structure vs MultiDCP-latent expression)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGDIR / "latent_three_way_by_organ.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    L = ["# Latent three-way comparison, split by body region", "",
         f"structure = **{args.encoder}** embedding | expression = **MultiDCP-CheMoE latent (global_features, 306d)** "
         "| head = L2 logistic regression, drug-level 5-fold x 5 seeds.", "",
         "| Organ | n (pos/neg) | structure | expression (MultiDCP latent) | both | +expr lift | 95% CI |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['organ']} | {r['n']} ({r['pos']}/{r['neg']}) | {r['struct']:.3f} | {r['expr']:.3f} | "
                 f"{r['both']:.3f} | **{r['lift']:+.3f}** | [{r['lo']:+.3f}, {r['hi']:+.3f}] |")
    L += ["", "## Notes / honest caveats",
          "- Expression = MultiDCP global_features. Validated: within one basal context the cell_hidden + dose "
          "blocks are CONSTANT across drugs (std ~1.8e-8), so the per-drug expression signal is the 128-d drug "
          "block, i.e. MultiDCP's own (expression-trained) structure encoder. So this compares an external "
          "encoder vs MultiDCP's drug encoder; a near-zero lift means they carry the same information.",
          "- Uses MultiDCP-PREDICTED latent (from SMILES) -> no measured-LINCS join needed -> larger n than the "
          "measured comparison (P1_three_way).",
          f"- Structure encoder is pluggable: --encoder chemberta|ecfp4|unimol_v1|unimol_v2 (UniMol needs unimol_tools).",
          "- Brain excluded (label-to-structure join blocked: SIDER is CID-keyed, no SOC file on disk).", "",
          "## Figure", f"- `{FIGDIR}/latent_three_way_by_organ.png`", ""]
    (FIGDIR / f"LATENT_{args.encoder}.md").write_text("\n".join(L))
    (REPO / f"results/tables/P1_latent_three_way_{args.encoder}.md").write_text("\n".join(L))
    df.to_csv(REPO / f"results/tables/P1_latent_three_way_{args.encoder}.csv", index=False)
    print(f"\nwrote results/tables/P1_latent_three_way_{args.encoder}.md + figure")


if __name__ == "__main__":
    main()
