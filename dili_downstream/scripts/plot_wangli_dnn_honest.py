#!/usr/bin/env python
"""Inflation-decomposition figure for the honest Li/Tong 8-layer DNN.

Reads results/tables/P_wangli_dnn_honest_full.csv (+ the v3 merge target
results/tables/P_tox_finetune_multidcp_v3.csv) and renders two panels:

  A. Waterfall of the C0..C4 decomposition: the ~0.798 headline stepping down as each of the three
     inflation mechanisms (leaky split, test-set selection, profile-level scoring) is removed, then
     the scaffold-OOD step. A chance line at 0.5 anchors the drop.
  B. Honest drug-disjoint drug-level AUROC for every feature arm, so the measured-GEX DNN can be
     read against structure / ChemBERTa / MultiDCP on the same axis.

No git commit. Writes results/figures/P_wangli_dnn_honest_decomposition.png (+ .pdf).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/dili_downstream")
HONEST = REPO / "results/tables/P_wangli_dnn_honest_full.csv"
V3 = REPO / "results/tables/P_tox_finetune_multidcp_v3.csv"
OUT = REPO / "results/figures/P_wangli_dnn_honest_decomposition.png"

INK = "#1b2733"
DROP = "#c1443c"
BAR = "#5b8bb0"
HL = "#c1443c"
GREY = "#9aa7b2"


def get(df, config):
    r = df[df["config"] == config].iloc[0]
    return float(r["auroc_mean"]), float(r["auroc_sd"])


def main():
    h = pd.read_csv(HONEST)
    v3 = pd.read_csv(V3)

    labels = ["C0\nleaky split\nprofile score\ntest select",
              "C1\ndrug-disjoint\nprofile score\ntest select",
              "C2\ndrug-disjoint\nprofile score\nhonest select",
              "C3\ndrug-disjoint\ndrug score\nhonest select",
              "C4\nscaffold-disjoint\ndrug score\nhonest select"]
    vals, sds = zip(*[get(h, c) for c in ["C0", "C1", "C2", "C3", "C4"]])
    vals, sds = np.array(vals), np.array(sds)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---- Panel A: waterfall ----
    x = np.arange(len(vals))
    axA.plot(x, vals, "-", color=INK, lw=1.4, zorder=2)
    axA.errorbar(x, vals, yerr=sds, fmt="o", color=INK, ms=8, capsize=4, lw=1.4, zorder=3)
    axA.axhline(0.5, color=GREY, lw=1.0, ls="--", zorder=1)
    axA.text(len(vals) - 1, 0.5, " chance", va="bottom", ha="right", color=GREY, fontsize=9)
    axA.axhspan(0.78, 0.82, color="#e8b04b", alpha=0.18, zorder=0)
    axA.text(0, 0.815, " their published band (0.78-0.82)", va="bottom", ha="left",
             color="#a5791f", fontsize=8.5)
    for i, (v, s) in enumerate(zip(vals, sds)):
        axA.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=10, fontweight="bold", color=INK)
    fixes = ["remove leaky split", "honest val selection", "per-drug scoring", "scaffold OOD"]
    for i in range(1, len(vals)):
        d = vals[i] - vals[i - 1]
        xm = i - 0.5
        ym = (vals[i] + vals[i - 1]) / 2
        axA.annotate(f"{d:+.3f}\n{fixes[i-1]}", (xm, ym), textcoords="offset points",
                     xytext=(0, -26), ha="center", fontsize=8.2, color=DROP)
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=8.0)
    axA.set_ylim(0.40, 0.88)
    axA.set_ylabel("AUROC", fontsize=11)
    axA.set_title("A. Inflation decomposition: the same 8-layer DNN, one fix at a time",
                  fontsize=11, loc="left")
    axA.spines[["top", "right"]].set_visible(False)

    # ---- Panel B: honest drug-disjoint drug-level AUROC across arms ----
    dd = v3[v3["split"] == "drug"].set_index("variant")
    arms = [
        ("chemberta_meta", "ChemBERTa + cell/dose (ft)"),
        ("tuned_E", "tuned MultiDCP (predGEX)"),
        ("chemberta_ft", "ChemBERTa (ft)"),
        ("structure", "structure / ECFP4"),
        ("chemberta_frozen", "ChemBERTa (frozen)"),
        ("frozen_pred", "frozen MultiDCP"),
        ("cell_dose_only", "cell/dose only"),
        ("__dnn__", "measured GEX, 8-layer DNN"),
        ("measured", "measured GEX, small head"),
    ]
    names, means, errs, colors = [], [], [], []
    dnn_mean, dnn_sd = get(h, "C3")
    for key, nm in arms:
        if key == "__dnn__":
            m, e = dnn_mean, dnn_sd
            colors.append(HL)
        else:
            m = float(dd.loc[key, "drug_auroc"]); e = float(dd.loc[key, "drug_sd"])
            colors.append(HL if key == "measured" else BAR)
        names.append(nm); means.append(m); errs.append(e)
    y = np.arange(len(names))[::-1]
    axB.barh(y, means, xerr=errs, color=colors, height=0.66, capsize=3,
             error_kw=dict(lw=1.0, ecolor=INK))
    axB.axvline(0.5, color=GREY, lw=1.0, ls="--")
    for yi, m in zip(y, means):
        axB.text(m + 0.012, yi, f"{m:.3f}", va="center", fontsize=8.6, color=INK)
    axB.set_yticks(y)
    axB.set_yticklabels(names, fontsize=8.6)
    axB.set_xlim(0.40, 0.68)
    axB.set_xlabel("drug-disjoint, drug-level AUROC", fontsize=10.5)
    axB.set_title("B. Honest ceiling: no feature beats ~0.60; measured GEX is at chance",
                  fontsize=11, loc="left")
    axB.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Li/Tong DILI 8-layer DNN on measured GEX: the 0.798 headline is inflation, not signal",
        fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[done] wrote {OUT}")
    print(f"[done] wrote {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
