#!/usr/bin/env python
"""Spatial-transcriptomics EDA: expression matrix x physical coordinates.

Complements the tabular "bracket" EDA (which bounds the toxicity-label signal
and never touches coordinates). This pass characterizes the *input* structure
and follows the canonical squidpy/scanpy Visium workflow so the figures read as
standard spatial transcriptomics to a domain audience:

  1. QC in space            -- is signal even across the slide, or a technical gradient?
  2. Cell-type proportions  -- continuous deconvolution overlays (a 55um spot is a MIXTURE)
  3. Spatial domains         -- argmax dominant-type label (shown WITH the caveat above)
  4. Spatially variable genes (Moran's I) + on-tissue overlay (visual confirmation)
  5. Neighborhood enrichment -- which domains physically touch (z-score, permutation null)
  6. Co-occurrence vs distance
  7. Ripley's L             -- does a cell type self-cluster vs spatial randomness?
  8. Centrality scores
  9. Project bridge         -- region-basal distinctness in the MultiDCP 10,716 space

Conventions follow squidpy docs + sc-best-practices.org: sq.pl.spatial_scatter
for physically-meaningful spots, colorblind-safe cmaps (viridis/magma continuous,
tab10 categorical), rare classes surfaced via groups=, diverging RdBu_r centred at
0 for enrichment. No H&E is shown because this CELLxGENE-schema h5ad ships without
the histology image (uns['spatial'] absent); stated in the write-up rather than faked.

Figures + an explained SUMMARY.md are written to a flat folder for email copy-paste.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests")
sys.path.insert(0, str(REPO))

from src.spatial.eda.region_diagnostics import compute_moran_svgs, basal_similarity_matrix
from src.spatial.pseudobulk import from_anndata
from src.spatial.gene_alignment import align_to_gene_space, coverage_fraction

ORGANS = {
    "heart": dict(
        path=REPO / "data/raw/spatial/kuppe_heart/Visium_control_P1.h5ad",
        domain_col="cell_type_original",
        qc_cols=["n_counts", "n_genes", "percent.mt"],
        deconv=["Cardiomyocyte", "Fibroblast", "Endothelial", "Pericyte", "Myeloid", "vSMCs"],
        organ_label="heart (Kuppe control donor P1)",
        liver_ref="0.988-0.998 (near-identical)",
    ),
}
ROBUST_MIN_SPOTS = 20  # below this a pseudobulk / point-pattern is noise, not architecture


def _overlay(a, color, out, *, title=None, cmap="viridis", size=22, ncols=None,
             groups=None, figsize=None):
    """sq.pl.spatial_scatter wrapper (circle mode; this h5ad has no H&E image)."""
    import squidpy as sq
    cols = color if isinstance(color, list) else [color]
    ncols = ncols or min(3, len(cols))
    nrows = int(np.ceil(len(cols) / ncols))
    figsize = figsize or (4.6 * ncols, 4.3 * nrows)
    sq.pl.spatial_scatter(
        a, color=color, shape=None, img=False, size=size, cmap=cmap,
        ncols=ncols, figsize=figsize, title=title, groups=groups,
        legend_loc="right margin", colorbar=True,
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ", default="heart", choices=list(ORGANS))
    ap.add_argument("--figdir",
                    default="/raid/home/joshua/claude_memory/downstream_2026/visuals/heart_spatial_eda")
    args = ap.parse_args()

    import anndata as ad
    import squidpy as sq

    cfg = ORGANS[args.organ]
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    figs = []  # (filename, caption)

    # ---- load ----
    a = ad.read_h5ad(cfg["path"])
    a.obsm["spatial"] = np.asarray(a.obsm["X_spatial"])
    dom = cfg["domain_col"]
    a.obs[dom] = a.obs[dom].astype("category")
    comp = a.obs[dom].value_counts()
    robust = [r for r in comp.index if comp[r] >= ROBUST_MIN_SPOTS]
    rare = [r for r in comp.index if r not in robust]
    print(f"[load] {a.shape}; robust domains {robust}")

    # ---- 1. QC in space ----
    f = "01_qc_in_space.png"
    _overlay(a, cfg["qc_cols"], figdir / f, cmap="viridis", ncols=3)
    figs.append((f, "QC mapped onto tissue coordinates. Each dot is a ~55 um Visium spot; "
                    "color = total RNA (n_counts), genes detected (n_genes), and mitochondrial "
                    "fraction (percent.mt). We are checking signal is even across the slide, not "
                    "pooling at one edge (a capture/permeabilization artifact) or flagging a "
                    "degraded/necrotic band (high mito)."))

    # ---- 2. cell-type proportions (continuous deconvolution -- the honest composition view) ----
    f = "02_celltype_proportions.png"
    _overlay(a, cfg["deconv"], figdir / f, cmap="magma", ncols=3)
    figs.append((f, "Cell-type composition from deconvolution. A Visium spot mixes multiple cells, "
                    "so each panel is the ESTIMATED PROPORTION (0-1) of one cell type per spot, not a "
                    "hard call. This is the faithful view of tissue architecture: cardiomyocyte-rich "
                    "muscle with vascular/fibroblast niches threaded through it."))

    # ---- 3. spatial domains (argmax -- shown WITH the mixture caveat) ----
    f = "03_domains_argmax.png"
    _overlay(a, dom, figdir / f, size=22, figsize=(7.4, 6.6))
    figs.append((f, f"Dominant cell type per spot (argmax of the proportions in fig 02). "
                    f"CAVEAT: argmax discards the mixture -- a spot that is 0.44 cardiomyocyte / "
                    f"0.32 fibroblast is called 'cardiomyocyte'. Shown for orientation; the "
                    f"proportion maps (fig 02) are the honest version."))
    # rare niches surfaced (cardiomyocyte greyed out)
    f = "03b_domains_rare_niches.png"
    _overlay(a, dom, figdir / f, groups=rare + [r for r in robust if r != comp.index[0]],
             size=30, figsize=(7.4, 6.6))
    figs.append((f, "Same domains with the dominant cardiomyocyte background greyed out (squidpy "
                    "groups=) so the sparse vascular/immune/fibroblast niches are visible. "
                    "Spot counts are tiny for some types (see table) -- read with care."))

    # ---- 4. Moran's I spatially variable genes + on-tissue overlay ----
    moran = compute_moran_svgs(a, spatial_key="spatial", n_neighs=6)
    symbols = set((REPO / "data/processed/spatial/multidcp_10716_symbols.txt").read_text().split())
    top = moran["top_svgs"]
    in_model = [g for g in top if g in symbols]
    top_plot = [g for g in top[:6] if g in a.var_names]
    f = "04_svg_top_genes_in_space.png"
    _overlay(a, top_plot, figdir / f, cmap="magma", ncols=3)
    figs.append((f, "Top spatially variable genes (highest Moran's I) plotted on the tissue. "
                    "Moran's I flags genes whose expression forms spatial PATCHES rather than "
                    "salt-and-pepper noise; this overlay is the visual confirmation that the "
                    "statistic reflects a real pattern. Cardiac genes (MYH7, NPPA/NPPB, TNNI3) lead."))

    # ---- spatial graph for neighborhood stats ----
    sq.gr.spatial_neighbors(a, coord_type="generic", n_neighs=6)

    # ---- 5. neighborhood enrichment (diverging, centred at 0) ----
    sq.gr.nhood_enrichment(a, cluster_key=dom, seed=0, show_progress_bar=False)
    f = "05_nhood_enrichment.png"
    sq.pl.nhood_enrichment(a, cluster_key=dom, cmap="RdBu_r", vmin=-50, vmax=50,
                           figsize=(6, 5.5), title="Neighborhood enrichment (z-score)")
    plt.savefig(figdir / f, dpi=150, bbox_inches="tight"); plt.close("all")
    figs.append((f, "Which domains physically sit next to which. Cell type x cell type z-score vs a "
                    "permutation null: red = neighbors more than chance, blue = spatially segregated, "
                    "diagonal = self-aggregation."))

    # ---- 6. co-occurrence vs distance ----
    try:
        sq.gr.co_occurrence(a, cluster_key=dom, show_progress_bar=False)
        f = "06_co_occurrence.png"
        sq.pl.co_occurrence(a, cluster_key=dom, clusters=str(comp.index[0]), figsize=(7, 4))
        plt.savefig(figdir / f, dpi=150, bbox_inches="tight"); plt.close("all")
        figs.append((f, "Starting from a cardiomyocyte spot, how much more/less likely is each other "
                        "type nearby as you walk outward. Ratio > 1 = drawn together at that radius, "
                        "< 1 = repelled; the crossing point is a characteristic interaction length."))
    except Exception as e:
        print(f"[cooc] skipped: {e}")

    # ---- 7-8. Ripley's L + centrality on robust domains only (singletons break point patterns) ----
    a_rob = a[a.obs[dom].isin(robust)].copy()
    a_rob.obs[dom] = a_rob.obs[dom].cat.remove_unused_categories()
    sq.gr.spatial_neighbors(a_rob, coord_type="generic", n_neighs=6)
    try:
        sq.gr.ripley(a_rob, cluster_key=dom, mode="L")
        f = "07_ripley_L.png"
        sq.pl.ripley(a_rob, cluster_key=dom, mode="L", figsize=(6.5, 4.5))
        plt.savefig(figdir / f, dpi=150, bbox_inches="tight"); plt.close("all")
        figs.append((f, "Is each cell type clumped, evenly spread, or random? Ripley's L(r) vs radius; "
                        "the pale grey curve is the random (CSR) reference from simulations. A type's "
                        "curve well above it = self-clustered (cardiomyocyte fills the muscle), near it = "
                        "random. Robust domains only (>=20 spots)."))
    except Exception as e:
        print(f"[ripley] skipped: {e}")
    try:
        sq.gr.centrality_scores(a_rob, cluster_key=dom)
        f = "08_centrality_scores.png"
        sq.pl.centrality_scores(a_rob, cluster_key=dom, figsize=(9, 3.2))
        plt.savefig(figdir / f, dpi=150, bbox_inches="tight"); plt.close("all")
        figs.append((f, "Per-domain graph roles: degree centrality (a hub touching many types), "
                        "clustering coefficient (tight self-patches), closeness (centrally embedded)."))
    except Exception as e:
        print(f"[centrality] skipped: {e}")

    # ---- 9. PROJECT BRIDGE: region-basal distinctness in 10,716 model space ----
    pb = from_anndata(a, obs_col=dom, agg="mean")
    target = sorted(symbols)
    aligned = {}
    for i, lab in enumerate(pb.region_order):
        if str(lab) in [str(e) for e in pb.empty_regions]:
            continue
        v = align_to_gene_space(pb.profiles[i].astype(float), list(pb.gene_names), target, missing="zero")
        aligned[str(lab)] = v
    cov = coverage_fraction(list(pb.gene_names), target)
    sim = basal_similarity_matrix(aligned, metric="pearson")
    f = "09_region_basal_similarity_modelspace.png"
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(sim.values, vmin=float(sim.values[sim.values < 1].min()), vmax=1.0, cmap="magma")
    ax.set_xticks(range(len(sim))); ax.set_yticks(range(len(sim)))
    ax.set_xticklabels(sim.columns, rotation=90, fontsize=8); ax.set_yticklabels(sim.index, fontsize=8)
    for i in range(len(sim)):
        for j in range(len(sim)):
            ax.text(j, i, f"{sim.values[i,j]:.3f}", ha="center", va="center", fontsize=6, color="white")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Region basal similarity in 10,716 model space (Pearson r)", fontsize=10)
    fig.tight_layout(); fig.savefig(figdir / f, dpi=150, bbox_inches="tight"); plt.close(fig)
    figs.append((f, "The project bridge: how distinct are region basal profiles in the gene space the "
                    "MultiDCP model actually sees. Lower Pearson = more regional contrast for a "
                    "region-resolved predicted signature to exploit."))

    off = sim.values[~np.eye(len(sim), dtype=bool)]
    rsim = sim.loc[robust, robust]
    roff = rsim.values[~np.eye(len(rsim), dtype=bool)]

    # ---- write the explained summary ----
    L = [f"# Spatial EDA -- {cfg['organ_label']}", "",
         "Whole-transcriptome Visium, basal (healthy) tissue. This characterizes the spatial "
         "INPUT structure; it complements (does not replace) the tabular toxicity bracket.", "",
         f"- Spots x genes: **{a.n_obs} x {a.n_vars}**  |  coords in `obsm['X_spatial']`",
         f"- No H&E image in this h5ad (CELLxGENE schema strips it); spots shown without histology.",
         f"- Domain label `{dom}`: {a.obs[dom].nunique()} cell types (argmax of deconvolution).", "",
         "## How to read these figures (one line each)"]
    for fn, cap in figs:
        L.append(f"- **{fn}** -- {cap.split('.')[0]}.")
    L += ["", "## Tissue architecture (spot counts per dominant type)",
          "| Domain | spots | frac | robust(>=20) |", "|---|---|---|---|"]
    for k in comp.index:
        L.append(f"| {k} | {comp[k]} | {comp[k]/a.n_obs:.3f} | {'yes' if k in robust else 'no'} |")
    L += ["",
          "## Spatial autocorrelation (Moran's I)",
          f"- SVGs (pval_norm < 0.05): **{moran['svg_count']}** ({moran['svg_fraction']*100:.1f}% of genes tested)",
          f"- Top SVGs: {', '.join(top[:12])}",
          f"- Top-20 SVG retention in 10,716 model space: **{len(in_model)}/{len(top)}**",
          "- Moran's I is computed on the spatial graph (k=6 neighbours); the SVG set shifts if k changes, "
          "so treat it as a strong graph-conditional baseline and confirm hits visually (fig 04).", "",
          "## Project bridge -- region basal distinctness in the model's gene space",
          f"- {cfg['organ_label'].split(' ')[0].capitalize()} Visium -> 10,716 coverage: **{cov*100:.1f}%**",
          f"- ALL regions off-diagonal Pearson: min {off.min():.3f}, mean {off.mean():.3f}, max {off.max():.3f} "
          f"(min driven by 1-2 spot rare niches)",
          f"- **ROBUST regions (>=20 spots: {', '.join(robust)}) off-diagonal Pearson: "
          f"min {roff.min():.3f}, mean {roff.mean():.3f}, max {roff.max():.3f}**",
          f"- Liver reference (yu2022 zones): {cfg['liver_ref']}", "",
          "## Interpretation & caveats",
          "- **Argmax vs mixture.** Domains (fig 03) are the dominant type per spot; the spot is really a "
          "mixture (fig 02). Don't read 'this spot is a fibroblast' literally.",
          "- **Tiny-region pseudobulk is noise.** A 1-2 spot 'region' gives an unstable mean; the headline "
          "contrast uses robust regions (>=20 spots) only.",
          f"- **Dominated section.** {comp.index[0]} is {comp[comp.index[0]]/a.n_obs*100:.0f}% of spots, so heart "
          "'regions' are sparse cell-type niches, not clean anatomical zones like liver zonation.",
          "- **Single section, single donor.** P1 only; P7/P8/P17 controls are on disk for replication.",
          "- **No batch integration / no histology QC against an image** (image absent).", "",
          "## Figures (full captions)"]
    for fn, cap in figs:
        L.append(f"- `{figdir / fn}`\n  - {cap}")
    L += ["", f"_Run elapsed: {time.time()-t0:.1f}s_"]

    (figdir / "SUMMARY.md").write_text("\n".join(L))
    (REPO / f"results/tables/P1_spatial_eda_{args.organ}.md").write_text("\n".join(L))
    print(f"[bridge] ALL min {off.min():.3f} mean {off.mean():.3f} | ROBUST min {roff.min():.3f} mean {roff.mean():.3f}")
    print(f"=== wrote SUMMARY.md + {len(figs)} figures in {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
