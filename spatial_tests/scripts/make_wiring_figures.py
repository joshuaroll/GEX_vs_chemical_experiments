#!/usr/bin/env python
"""Visual framework diagrams for the structure-vs-omics toxicity wiring.

Renders two schematics into lab_meetings/figures/:
  fig_engine.png     MultiDCP engine internals: 4 input streams -> fuse -> predicted DE
  fig_framework.png  experimental wiring: SMILES -> {structure, predicted DE} + measured DE (side),
                     three arms -> shared head -> {binary, severity}, colored by information source.
Pure drawing; no data. Blue = function of SMILES (structure-redundant); green = independent biology.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests/lab_meetings/figures")
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GREEN, GREY, ORANGE, PURPLE = "#cfe0f3", "#cde8d4", "#e8e8e8", "#f6dfc4", "#e2d5ef"
EDGE = "#3a3a3a"


def box(ax, x, y, w, h, text, fc=GREY, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.25",
                                fc=fc, ec=EDGE, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=5)


def arrow(ax, x1, y1, x2, y2, style="-|>", lw=1.4, color=EDGE, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 lw=lw, color=color, ls=ls, shrinkA=2, shrinkB=2, zorder=1))


# ============================================================ Figure 1: engine
def fig_engine():
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 97, "MultiDCP engine — how predicted differential expression (DE) is produced",
            ha="center", fontsize=13, fontweight="bold")

    # inputs (top row)
    box(ax, 3, 82, 20, 8, "SMILES\n→ molecular graph\n(atom 62-d, bond 6-d)", BLUE, 8.5)
    box(ax, 27, 82, 20, 8, "cell basal GEX\n[978]", GREEN, 9)
    box(ax, 51, 82, 15, 8, "dose\n[1]", GREY, 9)
    box(ax, 70, 82, 27, 8, "gene vector\n[978 × 128]  (fixed)", GREY, 9)

    # embed row
    box(ax, 3, 66, 20, 8, "NeuralFingerprint\n(graph conv)\n→ drug [128]", BLUE, 8.5)
    box(ax, 27, 66, 20, 8, "cell encoder\n(Linear / Transformer)\n→ cell [50]", GREEN, 8.5)
    box(ax, 51, 66, 15, 8, "Linear\n→ dose [16]", GREY, 8.5)
    box(ax, 70, 66, 27, 8, "Linear\n→ gene [978 × d]", GREY, 8.5)
    for x in (13, 37, 58.5, 83.5):
        arrow(ax, x, 82, x, 74)

    # attention (drug x gene)
    box(ax, 30, 52, 45, 7, "DrugGeneAttention  (2 layers, 4 heads)\ngene ⨯ drug-atoms", PURPLE, 9)
    arrow(ax, 13, 66, 40, 59)     # drug -> attn
    arrow(ax, 83.5, 66, 70, 59)   # gene -> attn

    # fuse
    box(ax, 20, 39, 60, 7,
        "concat per gene:  [ drug ⊕ gene(attn) ⊕ cell(50) ⊕ dose ]   → [978 × D]", GREY, 9)
    arrow(ax, 52, 52, 52, 46)              # attn -> fuse
    arrow(ax, 37, 66, 30, 46)              # cell -> fuse
    arrow(ax, 58.5, 66, 62, 46)            # dose -> fuse

    # decoder
    box(ax, 27, 26, 46, 7, "ReLU → Linear₁ → [978 × hid]  →  ReLU → Linear_final → [978]", GREY, 9)
    arrow(ax, 50, 39, 50, 33)
    box(ax, 30, 14, 40, 7, "predicted TREATED expression  [978]", ORANGE, 9.5, bold=True)
    arrow(ax, 50, 26, 50, 21)
    box(ax, 30, 3, 40, 7, "− basal  →  predicted DE  [978]", ORANGE, 9.5, bold=True)
    arrow(ax, 50, 14, 50, 10)

    # training note
    ax.add_patch(FancyBboxPatch((74, 22), 24, 22, boxstyle="round,pad=0.2,rounding_size=0.3",
                                fc="#fff6e6", ec=ORANGE, lw=1.4))
    ax.text(86, 40, "Trained self-supervised", ha="center", fontsize=9, fontweight="bold")
    ax.text(86, 33.5, "loss = MSE(pred, treated x1)\nno toxicity labels\nheld-out DE-Pearson\n"
            "0.68 (kidney) – 0.76 (liver)", ha="center", fontsize=8)
    arrow(ax, 74, 30, 70, 18, color=ORANGE, ls="--")

    ax.text(2, 47, "blue = from SMILES   green = measured cell biology", fontsize=8, color="#555")
    fig.savefig(OUT / "fig_engine.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ============================================================ Figure 2: framework
def fig_framework():
    fig, ax = plt.subplots(figsize=(13.5, 9.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 97.5, "Experimental framework — structure vs expression vs both",
            ha="center", fontsize=13, fontweight="bold")

    # ---- training role of measured LINCS profiles: they TRAIN the engine ----
    box(ax, 2, 70, 22, 9, "LINCS measured profiles\n(treated x1, basal x2)", GREEN, 8.5)
    ax.text(13, 66.5, "role = TRAIN the engine\n(self-supervised, no tox labels)",
            ha="center", fontsize=7.5, color="#2e7d32", style="italic")

    # ---- SMILES -> two deployable, SMILES-derived arms ----
    box(ax, 40, 89, 20, 6, "drug  (SMILES)", BLUE, 10, bold=True)
    box(ax, 6, 78, 30, 7, "STRUCTURE encoder\nChemBERTa / ECFP4 / UniMol / …", BLUE, 8.5)
    box(ax, 44, 78, 30, 9, "MultiDCP engine\n(frozen or tox-tuned)\nlatent layer  →  (or predicted DE)",
        BLUE, 8.5)
    arrow(ax, 45, 89, 21, 85)
    arrow(ax, 53, 89, 58, 87)
    arrow(ax, 24, 74.5, 44, 82, color="#2e7d32", ls="--")  # measured profiles -> train engine
    ax.text(33, 79.5, "trains", fontsize=7.5, color="#2e7d32", rotation=18)

    # ---- deployable arms ----
    box(ax, 8, 60, 26, 6, "arm 1: structure", BLUE, 9, bold=True)
    box(ax, 44, 60, 30, 6, "arm 2: expression = engine LATENT", BLUE, 8.5, bold=True)
    box(ax, 44, 51.5, 30, 5.5, "arm 3: both = structure ⊕ latent", PURPLE, 8.5, bold=True)
    arrow(ax, 20, 78, 20, 66)              # structure -> arm1
    arrow(ax, 59, 78, 59, 66)              # engine latent -> arm2
    arrow(ax, 22, 78, 50, 57)              # structure -> arm3
    arrow(ax, 59, 66, 62, 57)             # arm2 feeds into arm3 (both)

    # ---- control branch: measured DE fed DIRECTLY (upper bound, not deployable) ----
    box(ax, 77, 60, 21, 8, "measured DE  [978]\n(mean x1 − x2)", GREEN, 8)
    arrow(ax, 24, 72, 82, 68, color="#2e7d32", ls=":")   # profiles -> measured DE feature
    ax.text(88, 56.5, "CONTROL / upper bound (cond. D):\ndoes the MODALITY carry signal?\n"
            "needs LINCS — NOT deployable", ha="center", fontsize=7, color="#2e7d32", style="italic")

    # ---- shared head ----
    box(ax, 28, 38, 44, 8,
        "shared small HEAD\nlogreg / RandomForest / MLP\n(fusion: concat · late · stacker · MLP)",
        ORANGE, 9)
    arrow(ax, 20, 60, 40, 46)              # arm1 -> head
    arrow(ax, 59, 51.5, 52, 46)            # arm3 -> head
    arrow(ax, 62, 60, 58, 46)             # arm2 -> head
    arrow(ax, 82, 60, 66, 46, color="#2e7d32", ls=":")   # measured-DE control -> head

    # ---- targets ----
    box(ax, 22, 26, 26, 6, "binary DILI / DIKI\n(is it toxic?)", GREY, 9, bold=True)
    box(ax, 54, 26, 26, 6, "DILIrank Severity 0–8\n(how bad?)", GREY, 9, bold=True)
    arrow(ax, 46, 38, 35, 32)
    arrow(ax, 54, 38, 65, 32)

    # ---- verdict banner ----
    ax.add_patch(FancyBboxPatch((5, 11.5), 90, 9.5, boxstyle="round,pad=0.2,rounding_size=0.3",
                                fc="#fdeaea", ec="#c0392b", lw=1.5))
    ax.text(50, 16.2, "RESULT: structure is the ceiling on every target. The engine latent AND its DE "
            "output are both functions of SMILES (blue) → structure-ceilinged.\nThe measured-DE "
            "CONTROL (green) is at/below chance → the modality itself carries no extractable tox "
            "signal here. Only genuinely independent biology (measured DE in the\nright cell context) "
            "or a separate readout (dose-response head) could add information.",
            ha="center", fontsize=8.3, color="#7b241c")

    ax.text(50, 7, "eval: drug-disjoint & scaffold-disjoint CV · 5-fold × ≥3 seeds · paired bootstrap CI",
            ha="center", fontsize=8, color="#444")
    ax.add_patch(FancyBboxPatch((5, 0.3), 44, 4.6, boxstyle="round,pad=0.15", fc="white", ec="#999"))
    ax.add_patch(plt.Rectangle((7, 1.7), 2, 1.8, fc=BLUE, ec=EDGE))
    ax.text(10, 2.6, "function of SMILES (deployable arms)", fontsize=7.3, va="center")
    ax.add_patch(plt.Rectangle((32, 1.7), 2, 1.8, fc=GREEN, ec=EDGE))
    ax.text(35, 2.6, "measured biology (control)", fontsize=7.3, va="center")

    fig.savefig(OUT / "fig_framework.png", dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_engine(); fig_framework()
    for p in ("fig_engine.png", "fig_framework.png"):
        print("wrote", OUT / p)
