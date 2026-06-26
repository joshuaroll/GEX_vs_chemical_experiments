#!/usr/bin/env python
"""Three-way feature comparison, SPLIT BY BODY REGION (organ).

The core project question, per organ: does measured gene expression add toxicity
signal over chemical structure alone?

  structure          = 2048-bit ECFP4 (Morgan r=2) from SMILES
  expression         = per-drug mean measured LINCS L1000 DE (978-dim, treated-diseased)
  structure + expr   = concatenation

Same classifier (L2 logistic regression, balanced), same drug-level fair evaluation
(one row per drug; 5-fold stratified CV x seeds -> AUROC mean +/- std). The headline
is the +expression LIFT (both - structure) with a paired bootstrap CI on out-of-fold
predictions over the SAME drugs.

Organs: liver (DILIrank), kidney (DIRIL), heart (DICTrank). Brain (SIDER) cannot run
on current on-disk data (CID-keyed, no shared identifier with LINCS; needs a CID->
structure map + MedDRA SOC file) -- reported as a gap, not silently dropped.

Join key across all sources: InChIKey-14 (RDKit) from SMILES. Liver/heart SMILES come
from the name->dili_canonical->drugbank cascade; kidney has inline SMILES.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

REPO = Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests")
DD = REPO / "../dili_downstream/data/processed"
sys.path.insert(0, str(REPO))
from src.spatial.eda.fingerprints import smiles_to_ecfp4

FIGDIR = Path("/raid/home/joshua/claude_memory/downstream_2026/visuals/three_way_comparison")
N_SEEDS, N_FOLDS, N_BOOT = 5, 5, 10000


def smi2ikey14(smi):
    if not isinstance(smi, str) or not smi.strip(): return None
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try: ik = Chem.MolToInchiKey(m)
    except Exception: return None
    return ik.split("-")[0] if ik else None


def norm_name(s):
    return s.lower().strip() if isinstance(s, str) else None


# ---------------- LINCS measured expression: per-drug mean DE by InChIKey-14 ----------------
def load_lincs_expression():
    prof = pd.read_csv(DD / "wangli_profiles.csv")
    de = np.load(DD / "wangli_measured_de.npy")          # (5517, 978), row-aligned to prof
    assert de.shape[0] == len(prof), (de.shape, len(prof))
    prof["ikey14"] = prof["smiles"].map(smi2ikey14)
    drug_de, n_prof = {}, {}
    for ik, idx in prof.dropna(subset=["ikey14"]).groupby("ikey14").groups.items():
        rows = [prof.index.get_loc(i) for i in idx]
        drug_de[ik] = de[rows].mean(0)
        n_prof[ik] = len(rows)
    return drug_de, n_prof


# ---------------- SMILES name->structure cascade (liver, heart) ----------------
def _name_cascade():
    c1 = pd.read_csv(DD / "dili_canonical.csv"); c1["name_lower"] = c1["drug_name"].map(norm_name)
    c1map = c1.dropna(subset=["name_lower"]).drop_duplicates("name_lower").set_index("name_lower")["canonical_smiles"]
    c2 = pd.read_csv(DD / "drugbank_smiles_index.csv").dropna(subset=["name_lower"]).drop_duplicates("name_lower").set_index("name_lower")["smiles"]
    def f(nl):
        if nl in c1map.index and isinstance(c1map[nl], str): return c1map[nl]
        if nl in c2.index and isinstance(c2[nl], str): return c2[nl]
        return None
    return f


# ---------------- per-organ label tables -> [ikey14, label, smiles] ----------------
def load_liver():
    f = _name_cascade()
    dr = pd.read_excel(REPO / "data/raw/labels/dilirank/dilirank.xlsx", header=1)
    con = dr["vDILI-Concern"].astype(str).str.lower().str.strip()
    POS, NEG = {"vmost-dili-concern", "vless-dili-concern"}, {"vno-dili-concern"}
    dr = dr[con.isin(POS | NEG)].copy()
    dr["label"] = con[con.isin(POS | NEG)].isin(POS).astype(int).values
    dr["name_lower"] = dr["CompoundName"].map(norm_name)
    dr = dr.dropna(subset=["name_lower"]).drop_duplicates("name_lower")
    dr["smiles"] = dr["name_lower"].map(f)
    return dr[["name_lower", "label", "smiles"]]


def load_kidney():
    dk = pd.read_excel(REPO / "data/raw/labels/diril/diril_dataset_508.xlsx", sheet_name="A. DIRIL (317)")
    tox = dk["My Findings  (Toxicity)"].astype(str).str.strip().str.lower()
    dk["label"] = np.where(tox.str.startswith("nephro"), 1, np.where(tox.str.contains("non"), 0, np.nan))
    dk = dk[dk["label"].notna()].copy(); dk["label"] = dk["label"].astype(int)
    dk["name_lower"] = dk["name"].map(norm_name)
    return dk[["name_lower", "label", "smiles"]]


def load_heart():
    f = _name_cascade()
    dh = pd.read_excel(REPO / "data/raw/labels/dictrank/dictrank_dataset_508.xlsx", sheet_name="Table S1")
    def m(x):
        x = str(x).lower().strip()
        return 0 if x == "no" else (1 if x in ("most", "less") else np.nan)
    dh["label"] = dh["DICT _ Concern"].map(m)
    dh = dh[dh["label"].notna()].copy(); dh["label"] = dh["label"].astype(int)
    dh["name_lower"] = dh["Active Ingredient(s)"].map(norm_name)
    dh = dh.dropna(subset=["name_lower"]).drop_duplicates("name_lower")
    dh["smiles"] = dh["name_lower"].map(f)
    return dh[["name_lower", "label", "smiles"]]


ORGANS = {"liver": load_liver, "kidney": load_kidney, "heart": load_heart}


# ---------------- build feature matrices for one organ ----------------
def build_matrices(lab, drug_de):
    lab = lab.copy()
    lab["ikey14"] = lab["smiles"].map(smi2ikey14)
    lab = lab[lab["ikey14"].notna() & lab["ikey14"].isin(drug_de)].drop_duplicates("ikey14").reset_index(drop=True)
    Xs, valid = smiles_to_ecfp4(lab["smiles"].tolist())               # (n, 2048), (n,)
    lab = lab[valid].reset_index(drop=True)
    Xs = Xs[valid].astype(float)
    y = lab["label"].to_numpy(int)
    Xe = np.vstack([drug_de[ik] for ik in lab["ikey14"]]).astype(float)  # (n, 978)
    return lab, y, Xs, Xe


def cv_oof(X, y, seed):
    """Out-of-fold predicted probabilities (drug-level)."""
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def evaluate(y, Xs, Xe):
    feats = {"structure": Xs, "expression": Xe, "structure+expression": np.hstack([Xs, Xe])}
    aurocs = {k: [] for k in feats}
    oof0 = {}  # seed-0 OOF for the paired bootstrap
    for k, X in feats.items():
        for s in range(N_SEEDS):
            oof = cv_oof(X, y, s)
            aurocs[k].append(roc_auc_score(y, oof))
            if s == 0: oof0[k] = oof
    return aurocs, oof0


def paired_bootstrap_lift(y, oof_struct, oof_both, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y); diffs = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        diffs.append(roc_auc_score(y[idx], oof_both[idx]) - roc_auc_score(y[idx], oof_struct[idx]))
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), len(diffs)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    drug_de, n_prof = load_lincs_expression()
    print(f"LINCS per-drug mean DE: {len(drug_de)} drugs")

    rows, lift_rows = [], []
    for organ, loader in ORGANS.items():
        lab, y, Xs, Xe = build_matrices(loader(), drug_de)
        aur, oof0 = evaluate(y, Xs, Xe)
        lift_m, lo, hi, nb = paired_bootstrap_lift(y, oof0["structure"], oof0["structure+expression"])
        for k in aur:
            rows.append(dict(organ=organ, n=len(y), pos=int(y.sum()), neg=int((1 - y).sum()),
                             feature=k, auroc=np.mean(aur[k]), sd=np.std(aur[k])))
        lift_rows.append(dict(organ=organ, n=len(y), pos=int(y.sum()), neg=int((1 - y).sum()),
                              struct=np.mean(aur["structure"]), expr=np.mean(aur["expression"]),
                              both=np.mean(aur["structure+expression"]),
                              lift=lift_m, lo=lo, hi=hi, nboot=nb))
        print(f"[{organ}] n={len(y)} ({int(y.sum())}/{int((1-y).sum())})  "
              f"struct={np.mean(aur['structure']):.3f} expr={np.mean(aur['expression']):.3f} "
              f"both={np.mean(aur['structure+expression']):.3f}  "
              f"+expr lift={lift_m:+.3f} CI[{lo:+.3f},{hi:+.3f}]")

    df = pd.DataFrame(rows)
    # ---- grouped bar figure, split by organ ----
    organs = list(ORGANS); feats = ["structure", "expression", "structure+expression"]
    colors = {"structure": "#7fb3d5", "expression": "#f5b041", "structure+expression": "#2c7fb8"}
    fig, ax = plt.subplots(figsize=(8.4, 5.0)); w = 0.26
    for i, fkey in enumerate(feats):
        sub = df[df.feature == fkey].set_index("organ").loc[organs]
        x = np.arange(len(organs)) + (i - 1) * w
        ax.bar(x, sub["auroc"], w, yerr=sub["sd"], capsize=3, label=fkey, color=colors[fkey])
        for xi, (v, e) in zip(x, zip(sub["auroc"], sub["sd"])):
            ax.text(xi, v + e + .008, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(0.5, ls="--", c="grey", lw=1, label="chance")
    ax.set_xticks(np.arange(len(organs)))
    ax.set_xticklabels([f"{o}\n(n={int(df[df.organ==o].n.iloc[0])})" for o in organs])
    ax.set_ylabel("drug-level CV AUROC (mean +/- std, 5 seeds)"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Three-way feature comparison by organ\nstructure vs expression vs both (measured LINCS)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(FIGDIR / "three_way_by_organ.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- markdown ----
    L = ["# Three-way feature comparison, split by body region", "",
         "Core question per organ: does measured gene expression ADD toxicity signal over "
         "chemical structure alone? Drug-level fair evaluation (one row per drug; 5-fold "
         "stratified CV x 5 seeds; L2 logistic regression, balanced). Join: InChIKey-14.", "",
         "## Headline -- AUROC by organ and feature set", "",
         "| Organ | n (pos/neg) | structure | expression | structure+expression | **+expr lift (both - structure)** | 95% CI |",
         "|---|---|---|---|---|---|---|"]
    for r in lift_rows:
        L.append(f"| {r['organ']} | {r['n']} ({r['pos']}/{r['neg']}) | {r['struct']:.3f} | {r['expr']:.3f} | "
                 f"{r['both']:.3f} | **{r['lift']:+.3f}** | [{r['lo']:+.3f}, {r['hi']:+.3f}] |")
    L += ["", "Lift CI excluding 0 => expression adds signal over structure on that organ "
          "(paired bootstrap, same drugs, seed-0 OOF, " + f"{N_BOOT} resamples).", "",
          "## Brain (SIDER) -- CANNOT RUN on current data",
          "SIDER is keyed by PubChem CID with no SMILES/InChIKey/name and no MedDRA SOC column on "
          "disk; LINCS profiles carry no CID. No shared identifier => no join. Needs (1) a CID->"
          "structure (InChIKey) resolver and (2) a MedDRA SOC hierarchy file to define the "
          "nervous-system label. Reported, not silently dropped.", "",
          "## Method notes / caveats",
          "- Expression = per-drug MEAN of that drug's measured LINCS L1000 DE profiles (treated-diseased).",
          "- Drug-level evaluation is drug-disjoint by construction (one row per drug).",
          "- class_weight=balanced; AUROC is threshold-free. Liver/heart are positive-skewed; kidney is best-balanced.",
          "- Liver/heart SMILES via name->dili_canonical->drugbank cascade; kidney has inline SMILES.",
          "- This uses MEASURED expression. Predicted expression (MultiDCP) is the separate follow-on source.", "",
          "## Figure",
          f"- `{FIGDIR}/three_way_by_organ.png` -- grouped AUROC bars per organ", ""]
    (FIGDIR / "SUMMARY.md").write_text("\n".join(L))
    (REPO / "results/tables/P1_three_way_comparison.md").write_text("\n".join(L))
    df.to_csv(REPO / "results/tables/P1_three_way_comparison.csv", index=False)
    print(f"\nwrote {FIGDIR}/SUMMARY.md + figure, and results/tables/P1_three_way_comparison.{{md,csv}}")


if __name__ == "__main__":
    main()
