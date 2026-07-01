#!/usr/bin/env python
"""Lab-meeting figures for the structure-vs-expression toxicity result.

Reads the committed P4 result CSVs and renders four figures into lab_meetings/figures/:
  fig1_headline_threeway   structure / expression / both, per organ (P4_stage2_toxicity)
  fig2_encoder_sweep       per-encoder structure AUROC vs the flat expression bar (P4_encoder_sweep)
  fig3_measured_control    structure / predicted DE / measured DE on the overlap set (P4_predicted_vs_measured_de)
  fig4_lift_forest         +lift (both - structure) with 95% CI, all 18 encoder cells (P4_encoder_sweep)

Reproducible: figures are a pure function of the committed tables. No model calls.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests")
TAB = REPO / "results" / "tables"

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default=str(REPO / "lab_meetings" / "figures"))
args = ap.parse_args()
OUT = Path(args.outdir); OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "figure.dpi": 150, "savefig.bbox": "tight"})
ORGANS = ["liver", "kidney"]
C = {"structure": "#2c6fbb", "expression": "#d1495b", "both": "#6a4c93",
     "predicted DE": "#d1495b", "measured DE": "#e8a33d"}


def _bars_with_labels(ax, x, heights, errs, colors, width, labels=None):
    b = ax.bar(x, heights, width, yerr=errs, color=colors, capsize=3,
               error_kw=dict(lw=1, alpha=0.7))
    for xi, h in zip(x, heights):
        ax.text(xi, h + 0.006, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    return b


def fig1_headline():
    df = pd.read_csv(TAB / "P4_stage2_toxicity.csv").set_index("organ")
    feats = ["structure", "expression", "both"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.25
    xbase = np.arange(len(ORGANS))
    for j, f in enumerate(feats):
        h = [df.loc[o, f"{f}_auroc"] for o in ORGANS]
        e = [df.loc[o, f"{f}_auroc_sd"] for o in ORGANS]
        x = xbase + (j - 1) * w
        ax.bar(x, h, w, yerr=e, color=C[f], capsize=3, label=f,
               error_kw=dict(lw=1, alpha=0.7))
        for xi, hi in zip(x, h):
            ax.text(xi, hi + 0.006, f"{hi:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, ls=":", c="grey", lw=1, label="chance")
    ax.set_xticks(xbase)
    ax.set_xticklabels([f"{o}\n(n={int(df.loc[o,'n'])}, {int(df.loc[o,'pos'])}/{int(df.loc[o,'neg'])})"
                        for o in ORGANS])
    ax.set_ylabel("AUROC (drug-disjoint, 5-fold x 5 seeds)")
    ax.set_ylim(0.45, 0.80)
    ax.set_title("Predicted expression does not beat chemical structure")
    ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.savefig(OUT / "fig1_headline_threeway.png"); plt.close(fig)


def fig2_encoder_sweep():
    df = pd.read_csv(TAB / "P4_encoder_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, organ in zip(axes, ORGANS):
        d = df[df.organ == organ].sort_values("structure_auroc", ascending=False)
        x = np.arange(len(d))
        expr = d["expression_auroc"].iloc[0]
        ax.bar(x, d["structure_auroc"], 0.62, yerr=d["structure_auroc_sd"],
               color=C["structure"], capsize=2, error_kw=dict(lw=0.8, alpha=0.6))
        for xi, hi in zip(x, d["structure_auroc"]):
            ax.text(xi, hi + 0.006, f"{hi:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(expr, ls="--", c=C["expression"], lw=1.6,
                   label=f"expression (predicted DE) = {expr:.3f}")
        ax.axhline(0.5, ls=":", c="grey", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(d["encoder"], rotation=45, ha="right", fontsize=8.5)
        ax.set_title(f"{organ} (n={int(d['n'].iloc[0])})")
        ax.set_ylim(0.45, 0.78)
        ax.legend(fontsize=8.5, loc="upper right", frameon=False)
    axes[0].set_ylabel("structure AUROC")
    fig.suptitle("Structure beats predicted expression across 9 encoders (expression arm held fixed)",
                 fontsize=12, fontweight="bold")
    fig.savefig(OUT / "fig2_encoder_sweep.png"); plt.close(fig)


def fig3_measured_control():
    df = pd.read_csv(TAB / "P4_predicted_vs_measured_de.csv")
    feats = ["structure", "predicted DE", "measured DE"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    w = 0.25
    xbase = np.arange(len(ORGANS))
    for j, f in enumerate(feats):
        h, e = [], []
        for o in ORGANS:
            r = df[(df.organ == o) & (df.feature == f)].iloc[0]
            h.append(r["auroc"]); e.append(r["auroc_sd"])
        x = xbase + (j - 1) * w
        ax.bar(x, h, w, yerr=e, color=C[f], capsize=3, label=f, error_kw=dict(lw=1, alpha=0.7))
        for xi, hi in zip(x, h):
            ax.text(xi, hi + 0.006, f"{hi:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, ls=":", c="grey", lw=1, label="chance")
    ns = {o: int(df[df.organ == o]["n"].iloc[0]) for o in ORGANS}
    ax.set_xticks(xbase)
    ax.set_xticklabels([f"{o} (overlap n={ns[o]})" for o in ORGANS])
    ax.set_ylabel("AUROC (same drugs, drug-disjoint)")
    ax.set_ylim(0.45, 0.78)
    ax.set_title("Measured expression also fails to beat structure (kidney)\n"
                 "→ not a prediction-fidelity artifact")
    ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.savefig(OUT / "fig3_measured_control.png"); plt.close(fig)


def fig4_lift_forest():
    df = pd.read_csv(TAB / "P4_encoder_sweep.csv")
    # order: liver block then kidney block, each sorted by lift
    blocks = []
    for organ in ORGANS:
        d = df[df.organ == organ].sort_values("lift_both")
        blocks.append((organ, d))
    rows = sum(len(d) for _, d in blocks)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * rows + 1.4))
    y = 0; yticks, ylabels = [], []
    organ_color = {"liver": "#2c6fbb", "kidney": "#6a4c93"}
    for organ, d in blocks:
        for _, r in d.iterrows():
            lo, hi, m = r["lift_both_lo"], r["lift_both_hi"], r["lift_both"]
            ax.plot([lo, hi], [y, y], c=organ_color[organ], lw=2, alpha=0.8)
            ax.plot(m, y, "o", c=organ_color[organ], ms=5)
            yticks.append(y); ylabels.append(f"{organ}: {r['encoder']}")
            y += 1
        y += 0.6  # gap between organ blocks
    ax.axvline(0, ls="--", c="black", lw=1.2)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_xlabel("+lift  (both − structure)  AUROC, 95% CI")
    ax.set_title("Adding predicted DE to structure helps in 0 / 18 cells\n(every CI crosses zero)")
    ax.margins(y=0.02)
    fig.savefig(OUT / "fig4_lift_forest.png"); plt.close(fig)


def main():
    fig1_headline(); fig2_encoder_sweep(); fig3_measured_control(); fig4_lift_forest()
    for p in sorted(OUT.glob("fig*.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()
