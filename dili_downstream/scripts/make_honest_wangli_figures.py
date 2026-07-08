#!/usr/bin/env python
"""Lab-meeting figures for the honest Li/Tong (Wang/Li 2020) DILI reproduction.

Reads two result tables and writes three presentation-ready PNGs:

  FIG 1  inflation decomposition (waterfall)  -- the money figure
  FIG 2  honest drug-level head-to-head across every feature source
  FIG 3  leaky-vs-honest paired bars per model

Inputs
  results/tables/P_wangli_dnn_honest_full.csv    (8-layer DNN, C0..C4 decomposition)
  results/tables/P_tox_finetune_multidcp_v3.csv  (small-head feature-source comparison)

Outputs
  results/figures/fig1_inflation_decomposition.png
  results/figures/fig2_honest_head_to_head.png
  results/figures/fig3_leaky_vs_honest.png

Colorblind-safe palette (validated with the dataviz skill validator, worst
all-pairs CVD delta-E 16.2). No git commit is made by this script.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = "/raid/home/joshua/projects/GEX_vs_chemical_experiments/dili_downstream"
TABLES = os.path.join(ROOT, "results", "tables")
FIGDIR = os.path.join(ROOT, "results", "figures")
HONEST_CSV = os.path.join(TABLES, "P_wangli_dnn_honest_full.csv")
V3_CSV = os.path.join(TABLES, "P_tox_finetune_multidcp_v3.csv")

# --------------------------------------------------------------------------- #
# Palette (light chart surface #fcfcfb) -- from the dataviz reference instance
# --------------------------------------------------------------------------- #
BLUE = "#2a78d6"    # honest signal retained / drug-disjoint split
RED = "#e34948"     # inflation / leaky evaluation
ORANGE = "#eb6834"  # scaffold-disjoint split
INK = "#0b0b0b"     # primary ink
INK2 = "#52514e"    # secondary ink
MUTED = "#898781"   # axis / muted labels
GRID = "#e1e0d9"    # hairline gridline
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
RED_DARK = "#b0322f"  # ink tone for red annotations

PUBLISHED = 0.798   # Li/Tong 2020 published headline AUROC
CHANCE = 0.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
    "font.size": 11,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 1.0,
    "axes.titlecolor": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# Human-readable model names --------------------------------------------------
DISPLAY = {
    "structure": "Structure (GIN)",
    "chemberta_meta": "ChemBERTa (meta)",
    "chemberta_ft": "ChemBERTa (fine-tuned)",
    "chemberta_frozen": "ChemBERTa (frozen)",
    "tuned_E": "MultiDCP (tuned)",
    "frozen_pred": "MultiDCP pred (frozen)",
    "cell_dose_only": "Cell + dose only",
    "measured": "Measured GEX (small head)",
    "honest_dnn": "Li/Tong 8-layer DNN (measured)",
}


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(length=0)


# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
def load_data():
    honest = pd.read_csv(HONEST_CSV)
    v3 = pd.read_csv(V3_CSV)
    return honest, v3


def honest_row(honest, split, score, selection):
    m = (honest["split"] == split) & (honest["score"] == score) & (honest["selection"] == selection)
    r = honest[m]
    assert len(r) == 1, f"expected 1 row for {split}/{score}/{selection}, got {len(r)}"
    return r.iloc[0]


def v3_row(v3, split, variant):
    m = (v3["split"] == split) & (v3["variant"] == variant)
    r = v3[m]
    assert len(r) == 1, f"expected 1 row for {split}/{variant}, got {len(r)}"
    return r.iloc[0]


# --------------------------------------------------------------------------- #
# FIG 1 -- inflation decomposition waterfall
# --------------------------------------------------------------------------- #
def make_fig1(honest, out):
    # C0..C4 in order, with the fix removed at each step.
    configs = [
        ("C0", "profile", "profile", "test"),
        ("C1", "drug", "profile", "test"),
        ("C2", "drug", "profile", "honest"),
        ("C3", "drug", "drug", "honest"),
        ("C4", "scaffold", "drug", "honest"),
    ]
    vals, sds = [], []
    for _, split, score, sel in configs:
        r = honest_row(honest, split, score, sel)
        vals.append(float(r["auroc_mean"]))
        sds.append(float(r["auroc_sd"]))

    fix_labels = [
        "remove leaky split (drug-disjoint)",
        "honest val selection (group-disjoint)",
        "drug-level scoring (per compound)",
        "scaffold-disjoint (OOD)",
    ]
    tick_labels = [
        "C0\nleaky profile\nprofile . test",
        "C1\ndrug-disjoint\nprofile . test",
        "C2\ndrug-disjoint\nprofile . honest",
        "C3\ndrug-disjoint\ndrug . honest",
        "C4\nscaffold-disjoint\ndrug . honest",
    ]

    x = list(range(len(vals)))
    base = 0.45
    w = 0.62

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    _style_axes(ax)

    # blue bars = AUROC (honest signal retained)
    ax.bar(x, [v - base for v in vals], bottom=base, width=w,
           color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)

    # red floating rects = inflation removed by each successive fix
    for i in range(1, len(vals)):
        drop_lo, drop_hi = vals[i], vals[i - 1]
        ax.add_patch(Rectangle((x[i] - w / 2, drop_lo), w, drop_hi - drop_lo,
                               facecolor=RED, alpha=0.85, edgecolor=SURFACE,
                               linewidth=1.2, zorder=3))
        # thin dashed connector from previous bar top across to this drop's top
        ax.plot([x[i - 1] + w / 2, x[i] - w / 2], [drop_hi, drop_hi],
                color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)

    # legend / key block in the empty upper-middle, no inline callouts (avoids
    # collisions with the short C3/C4 bars). Two aligned text columns.
    ax.annotate("Red bar = inflation removed by each fix:",
                xy=(2.45, 0.792), ha="left", va="top", fontsize=9.5,
                fontweight="bold", color=RED_DARK, zorder=6)
    yrow = 0.760
    for i in range(1, len(vals)):
        delta = vals[i - 1] - vals[i]
        ax.annotate(fix_labels[i - 1], xy=(2.45, yrow), ha="left", va="top",
                    fontsize=9.0, color=INK2, zorder=6)
        ax.annotate(f"−{delta:.3f}", xy=(4.35, yrow), ha="right", va="top",
                    fontsize=9.0, color=RED_DARK, fontweight="bold", zorder=6)
        yrow -= 0.029

    # error bars + AUROC value labels on the blue bar tops
    ax.errorbar(x, vals, yerr=sds, fmt="none", ecolor=INK, elinewidth=1.4,
                capsize=4, capthick=1.4, zorder=6)
    for xi, v, sd in zip(x, vals, sds):
        ax.annotate(f"{v:.3f}", xy=(xi, v + sd + 0.006), ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=INK, zorder=6)

    # reference lines
    ax.axhline(PUBLISHED, color=INK2, lw=1.3, ls=(0, (2, 2)), zorder=1)
    ax.annotate(f"Published Li/Tong headline = {PUBLISHED:.3f}",
                xy=(-0.02, PUBLISHED + 0.006), ha="left", va="bottom",
                fontsize=9.5, color=INK2, fontstyle="italic")
    ax.axhline(CHANCE, color=MUTED, lw=1.3, ls=(0, (5, 4)), zorder=1)
    ax.annotate("chance = 0.50", xy=(len(vals) - 1 + 0.55, CHANCE), ha="left",
                va="center", fontsize=9.5, color=MUTED, fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8.6)
    ax.set_ylim(base, 0.83)
    ax.set_xlim(-0.6, len(vals) - 1 + 1.15)
    ax.set_ylabel("Test AUROC", fontsize=12)
    ax.set_title("Inflation decomposition: the 0.798 headline collapses to chance once each\n"
                 "inflation mechanism is removed (8-layer DNN, measured GEX)",
                 fontsize=13.5, fontweight="bold", loc="left", pad=12)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# FIG 2 -- honest drug-level head-to-head across feature sources
# --------------------------------------------------------------------------- #
def make_fig2(honest, v3, out):
    v3_variants = ["structure", "chemberta_meta", "chemberta_ft", "chemberta_frozen",
                   "tuned_E", "frozen_pred", "cell_dose_only", "measured"]

    # drug-level AUROC on the two honest, group-disjoint splits
    drug = {}
    scaf = {}
    for var in v3_variants:
        drug[var] = (float(v3_row(v3, "drug", var)["drug_auroc"]),
                     float(v3_row(v3, "drug", var)["drug_sd"]))
        scaf[var] = (float(v3_row(v3, "scaffold", var)["drug_auroc"]),
                     float(v3_row(v3, "scaffold", var)["drug_sd"]))
    # honest 8-layer DNN: C3 (drug-disjoint) and C4 (scaffold-disjoint), drug-scored
    c3 = honest_row(honest, "drug", "drug", "honest")
    c4 = honest_row(honest, "scaffold", "drug", "honest")
    drug["honest_dnn"] = (float(c3["drug_auroc_mean"]), float(c3["drug_auroc_sd"]))
    scaf["honest_dnn"] = (float(c4["drug_auroc_mean"]), float(c4["drug_auroc_sd"]))

    models = v3_variants + ["honest_dnn"]
    # order by drug-disjoint drug-level AUROC, best first
    models.sort(key=lambda m: drug[m][0], reverse=True)

    x = list(range(len(models)))
    w = 0.38

    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    _style_axes(ax)

    d_mean = [drug[m][0] for m in models]
    d_sd = [drug[m][1] for m in models]
    s_mean = [scaf[m][0] for m in models]
    s_sd = [scaf[m][1] for m in models]
    b = 0.40

    bars_d = ax.bar([xi - w / 2 for xi in x], [v - b for v in d_mean], bottom=b, width=w,
                    color=BLUE, edgecolor=SURFACE, linewidth=1.0, zorder=3,
                    label="Drug-disjoint split")
    bars_s = ax.bar([xi + w / 2 for xi in x], [v - b for v in s_mean], bottom=b, width=w,
                    color=ORANGE, edgecolor=SURFACE, linewidth=1.0, zorder=3,
                    label="Scaffold-disjoint split")
    ax.errorbar([xi - w / 2 for xi in x], d_mean, yerr=d_sd, fmt="none",
                ecolor=INK, elinewidth=1.1, capsize=3, capthick=1.1, zorder=5)
    ax.errorbar([xi + w / 2 for xi in x], s_mean, yerr=s_sd, fmt="none",
                ecolor=INK, elinewidth=1.1, capsize=3, capthick=1.1, zorder=5)

    for xi, v, sd in zip([xi - w / 2 for xi in x], d_mean, d_sd):
        ax.annotate(f"{v:.2f}", xy=(xi, v + sd + 0.004), ha="center", va="bottom",
                    fontsize=8.2, color=INK2)
    for xi, v, sd in zip([xi + w / 2 for xi in x], s_mean, s_sd):
        ax.annotate(f"{v:.2f}", xy=(xi, v + sd + 0.004), ha="center", va="bottom",
                    fontsize=8.2, color=INK2)

    ax.axhline(CHANCE, color=MUTED, lw=1.4, ls=(0, (5, 4)), zorder=2)
    ax.annotate("chance = 0.50", xy=(len(models) - 1 + 0.55, CHANCE), ha="left",
                va="center", fontsize=9.5, color=MUTED, fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in models], rotation=32, ha="right", fontsize=9.2)
    ax.set_xlim(-0.7, len(models) - 1 + 1.05)
    ax.set_ylim(b, 0.70)
    ax.set_ylabel("Drug-level AUROC (mean ± sd)", fontsize=12)
    ax.set_title("Honest head-to-head: under group-disjoint splits, drug-level AUROC clusters\n"
                 "near chance and no feature source beats plain chemical structure",
                 fontsize=13.5, fontweight="bold", loc="left", pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# FIG 3 -- leaky vs honest paired bars per model
# --------------------------------------------------------------------------- #
def make_fig3(honest, v3, out):
    v3_variants = ["structure", "chemberta_meta", "chemberta_ft", "chemberta_frozen",
                   "tuned_E", "frozen_pred", "cell_dose_only", "measured"]

    leaky = {}   # test split, profile scoring (their headline regime)
    hon = {}     # drug-disjoint split, drug scoring (fully honest)
    for var in v3_variants:
        tr = v3_row(v3, "test", var)
        dr = v3_row(v3, "drug", var)
        leaky[var] = (float(tr["profile_auroc"]), float(tr["profile_sd"]))
        hon[var] = (float(dr["drug_auroc"]), float(dr["drug_sd"]))
    c0 = honest_row(honest, "profile", "profile", "test")   # leaky profile
    c3 = honest_row(honest, "drug", "drug", "honest")        # fully honest
    leaky["honest_dnn"] = (float(c0["profile_auroc_mean"]), float(c0["profile_auroc_sd"]))
    hon["honest_dnn"] = (float(c3["drug_auroc_mean"]), float(c3["drug_auroc_sd"]))

    models = v3_variants + ["honest_dnn"]
    models.sort(key=lambda m: hon[m][0], reverse=True)

    x = list(range(len(models)))
    w = 0.38

    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    _style_axes(ax)

    l_mean = [leaky[m][0] for m in models]
    l_sd = [leaky[m][1] for m in models]
    h_mean = [hon[m][0] for m in models]
    h_sd = [hon[m][1] for m in models]

    ax.bar([xi - w / 2 for xi in x], l_mean, width=w, color=RED, edgecolor=SURFACE,
           linewidth=1.0, zorder=3, label="Leaky profile split (profile scoring)")
    ax.bar([xi + w / 2 for xi in x], h_mean, width=w, color=BLUE, edgecolor=SURFACE,
           linewidth=1.0, zorder=3, label="Honest drug-disjoint (drug scoring)")
    ax.errorbar([xi - w / 2 for xi in x], l_mean, yerr=l_sd, fmt="none",
                ecolor=INK, elinewidth=1.1, capsize=3, capthick=1.1, zorder=5)
    ax.errorbar([xi + w / 2 for xi in x], h_mean, yerr=h_sd, fmt="none",
                ecolor=INK, elinewidth=1.1, capsize=3, capthick=1.1, zorder=5)

    for xi, v, sd in zip([xi - w / 2 for xi in x], l_mean, l_sd):
        ax.annotate(f"{v:.2f}", xy=(xi, v + sd + 0.006), ha="center", va="bottom",
                    fontsize=8.2, color=INK2)
    for xi, v, sd in zip([xi + w / 2 for xi in x], h_mean, h_sd):
        ax.annotate(f"{v:.2f}", xy=(xi, v + sd + 0.006), ha="center", va="bottom",
                    fontsize=8.2, color=INK2)

    ax.axhline(CHANCE, color=MUTED, lw=1.4, ls=(0, (5, 4)), zorder=2)
    ax.annotate("chance = 0.50", xy=(len(models) - 1 + 0.55, CHANCE), ha="left",
                va="center", fontsize=9.5, color=MUTED, fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in models], rotation=32, ha="right", fontsize=9.2)
    ax.set_xlim(-0.7, len(models) - 1 + 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("AUROC (mean ± sd)", fontsize=12)
    ax.set_title("Leaky vs honest: every model's headline AUROC collapses toward chance\n"
                 "when the profile-level leak is closed and scoring is per compound",
                 fontsize=13.5, fontweight="bold", loc="left", pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    honest, v3 = load_data()

    f1 = os.path.join(FIGDIR, "fig1_inflation_decomposition.png")
    f2 = os.path.join(FIGDIR, "fig2_honest_head_to_head.png")
    f3 = os.path.join(FIGDIR, "fig3_leaky_vs_honest.png")

    make_fig1(honest, f1)
    make_fig2(honest, v3, f2)
    make_fig3(honest, v3, f3)

    for p in (f1, f2, f3):
        ok = os.path.exists(p) and os.path.getsize(p) > 0
        print(f"{'WROTE' if ok else 'FAILED'}  {p}  ({os.path.getsize(p) if ok else 0} bytes)")


if __name__ == "__main__":
    main()
