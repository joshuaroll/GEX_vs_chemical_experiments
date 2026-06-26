#!/usr/bin/env python
"""Story-first spatial EDA: a feasibility-gate chain for the predicted region arm.

The predicted, region-resolved toxicity signature can only work if a chain of
necessary conditions holds on the INPUT tissue. This script measures each link
and reports PASS/FAIL, so the place the chain breaks is the finding:

  GATE 1  Are regions distinguishable in expression -- ABOVE the within-region
          noise floor? (split each region in half; between-region distance vs
          within-region split-half distance. The anchor that makes r mean something.)
  GATE 2  Is region a real axis of variance, or swamped by depth/donor?
          (PVCA-lite: % PC variance attributable to region vs sequencing depth.)
  GATE 3  Does the region-discriminating signal survive into the 10,716 model
          gene space -- and is it biology, not mito/ribo/technical?
  BRIDGE  Region-basal pairwise similarity in the model gene space (lower = more
          contrast for the model to exploit).
  (GATE 4 cross-species, GATE 5 OOD: scaffolded; need rodent region labels.)

The canonical niche battery (nhood/co-occurrence/Ripley) is demoted to one
supporting line -- it describes cell-type architecture, which this model does
not use.

Regions:
  liver  -- published anatomical zones (yu2022 obs['category']).
  kidney -- cortex vs medulla from canonical markers (NON-circular; not clustering).
Figures + a narrative SUMMARY.md go to a flat folder for email copy-paste.
"""
from __future__ import annotations
import argparse, sys, tarfile, tempfile, zipfile, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests")
sys.path.insert(0, str(REPO))
from src.spatial.gene_alignment import align_to_gene_space, coverage_fraction
from src.spatial.pseudobulk import from_anndata
from src.spatial.eda.region_diagnostics import basal_similarity_matrix

SYMBOLS = sorted(set((REPO / "data/processed/spatial/multidcp_10716_symbols.txt").read_text().split()))
SYMSET = set(SYMBOLS)
TECH_RE = re.compile(r"^(MT-|RPL|RPS|MRPL|MRPS|MTRNR|HBA|HBB|MALAT1|NEAT1)")
KIDNEY_MARKERS = dict(
    cortex=["NPHS1", "NPHS2", "PODXL", "LRP2", "SLC34A1", "SLC5A12", "CUBN", "MIOX"],
    medulla=["UMOD", "SLC12A1", "AQP2", "AQP3", "SLC14A2", "CLDN8", "FXYD4"],
)


# --------------------------- loaders ---------------------------
def _read_10x_h5(h5_path):
    import scanpy as sc
    a = sc.read_10x_h5(str(h5_path)); a.var_names_make_unique(); return a


def _attach_positions(adata, pos_csv):
    df = pd.read_csv(pos_csv, header=None,
                     names=["barcode", "in_tissue", "row", "col", "pix_row", "pix_col"]).set_index("barcode")
    common = adata.obs.index.intersection(df.index)
    adata = adata[common].copy()
    adata.obsm["spatial"] = df.loc[common, ["pix_col", "pix_row"]].to_numpy(float)
    return adata


def load_liver(tmp):
    zp = REPO / "data/raw/spatial/yu2022_liver/L5_upload.zip"
    with zipfile.ZipFile(zp) as zf: zf.extractall(tmp)
    h5 = next(Path(tmp).rglob("filtered_feature_bc_matrix.h5"))
    a = _read_10x_h5(h5)
    cat = pd.read_csv(next(Path(tmp).rglob("l5_category.csv")), index_col=0)
    a.obs = a.obs.join(cat, how="left")
    a.obs["region"] = a.obs["category"].astype("object")
    pos = next(Path(tmp).rglob("tissue_positions_list.csv"))
    a = _attach_positions(a, pos)
    a = a[a.obs["region"].notna()].copy()
    return a


def load_kidney(tmp, gsm="GSM6047774_V19S25-016_XY01_18-0006"):
    raw = REPO / "data/raw/spatial/lake_kpmp_kidney/GSE183456_RAW.tar"
    with tarfile.open(raw) as t: t.extract(f"{gsm}.tar.gz", tmp)
    with tarfile.open(Path(tmp) / f"{gsm}.tar.gz") as t: t.extractall(Path(tmp) / gsm)
    h5 = next((Path(tmp) / gsm).rglob("filtered_feature_bc_matrix.h5"))
    a = _read_10x_h5(h5)
    pos = next((Path(tmp) / gsm).rglob("tissue_positions_list.csv"))
    a = _attach_positions(a, pos)
    return a


def assign_kidney_regions(a_lognorm):
    """Cortex vs medulla from canonical markers (external, non-circular)."""
    import scanpy as sc
    scores = {}
    for reg, mk in KIDNEY_MARKERS.items():
        present = [g for g in mk if g in a_lognorm.var_names]
        sc.tl.score_genes(a_lognorm, present, score_name=f"{reg}_score")
        scores[reg] = a_lognorm.obs[f"{reg}_score"].to_numpy()
    region = np.where(scores["cortex"] >= scores["medulla"], "cortex", "medulla")
    a_lognorm.obs["region"] = region
    return [g for r in KIDNEY_MARKERS for g in KIDNEY_MARKERS[r] if g in a_lognorm.var_names]


# --------------------------- metrics ---------------------------
def preprocess(a):
    import scanpy as sc
    a.layers["counts"] = a.X.copy()
    a.obs["total_counts"] = np.asarray(a.X.sum(1)).ravel()
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    return a


def _pb(mat, mask):
    return np.asarray(mat[mask].mean(0)).ravel()


def gate1_separability(a, region_col="region", n_splits=8, min_spots=40, seed=0):
    import scipy.sparse as sp
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    labs = a.obs[region_col].astype(str).to_numpy()
    regs = [r for r in pd.unique(labs) if (labs == r).sum() >= min_spots]
    rng = np.random.RandomState(seed)
    within = []
    for r in regs:
        idx = np.where(labs == r)[0]
        for _ in range(n_splits):
            perm = rng.permutation(idx); h = len(perm) // 2
            a1, a2 = _pb(X, perm[:h]), _pb(X, perm[h:2 * h])
            within.append(np.corrcoef(a1, a2)[0, 1])
    full = {r: _pb(X, np.where(labs == r)[0]) for r in regs}
    between = [np.corrcoef(full[regs[i]], full[regs[j]])[0, 1]
               for i in range(len(regs)) for j in range(i + 1, len(regs))]
    wf, bt = float(np.mean(within)), float(np.mean(between))
    ratio = (1 - bt) / max(1 - wf, 1e-9)  # between-region distance in units of within-region noise
    return dict(regions=regs, within_floor=wf, within_sd=float(np.std(within)),
                between=bt, between_sd=float(np.std(between)),
                gap=wf - bt, ratio=ratio, within_all=within, between_all=between)


def gate2_variance_partition(a, region_col="region", n_pcs=30):
    import scanpy as sc
    b = a.copy()
    sc.pp.highly_variable_genes(b, n_top_genes=2000)
    b = b[:, b.var.highly_variable].copy()
    sc.pp.scale(b, max_value=10); sc.pp.pca(b, n_comps=n_pcs)
    pcs = b.obsm["X_pca"]; var = b.uns["pca"]["variance_ratio"]
    labs = pd.Categorical(a.obs[region_col].astype(str))
    depth = np.log1p(a.obs["total_counts"].to_numpy())
    reg_eta, dep_r2 = [], []
    for k in range(pcs.shape[1]):
        y = pcs[:, k]
        # region eta^2 (one-way ANOVA)
        grand = y.mean(); ss_t = ((y - grand) ** 2).sum()
        ss_b = sum((y[labs == g].mean() - grand) ** 2 * (labs == g).sum() for g in labs.categories)
        reg_eta.append(ss_b / max(ss_t, 1e-12))
        # depth R^2
        dd = depth - depth.mean()
        dep_r2.append((np.dot(dd, y - y.mean()) ** 2) / max((dd ** 2).sum() * ((y - y.mean()) ** 2).sum(), 1e-12))
    reg_eta, dep_r2, var = np.array(reg_eta), np.array(dep_r2), np.array(var)
    region_pct = float((var * reg_eta).sum() / var.sum())
    depth_pct = float((var * dep_r2).sum() / var.sum())
    return dict(region_pct=region_pct, depth_pct=depth_pct,
                residual_pct=max(0.0, 1 - region_pct - depth_pct))


def gate3_discriminating_genes(a, region_col="region", topn=50):
    from sklearn.feature_selection import f_classif
    import scipy.sparse as sp
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    y = a.obs[region_col].astype(str).to_numpy()
    F, _ = f_classif(X, y)
    genes = np.array(a.var_names)
    order = np.argsort(np.nan_to_num(F))[::-1][:topn]
    top = genes[order]; topF = np.nan_to_num(F)[order]
    in_model = np.array([g in SYMSET for g in top])
    tech = np.array([bool(TECH_RE.match(g)) for g in top])
    return dict(top=list(top), topF=list(topF), in_model=in_model, tech=tech,
                retention=float(in_model.mean()), tech_frac=float(tech.mean()))


def bridge(a, region_col="region"):
    pb = from_anndata(a, obs_col=region_col, agg="mean")
    aligned = {}
    for i, lab in enumerate(pb.region_order):
        if str(lab) in [str(e) for e in pb.empty_regions]: continue
        aligned[str(lab)] = align_to_gene_space(pb.profiles[i].astype(float),
                                                 list(pb.gene_names), SYMBOLS, missing="zero")
    cov = coverage_fraction(list(pb.gene_names), SYMBOLS)
    sim = basal_similarity_matrix(aligned, metric="pearson")
    return cov, sim


# --------------------------- figures ---------------------------
def fig_region_map(a, out, title):
    c = a.obsm["spatial"]; labs = a.obs["region"].astype(str)
    cats = list(pd.unique(labs)); pal = plt.cm.tab10(np.linspace(0, 1, max(10, len(cats))))
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    for i, cc in enumerate(cats):
        m = (labs == cc).to_numpy()
        ax.scatter(c[m, 0], c[m, 1], s=10, color=pal[i % len(pal)], label=f"{cc} (n={m.sum()})", linewidths=0)
    ax.legend(fontsize=8, markerscale=2, loc="center left", bbox_to_anchor=(1, .5), frameon=False)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_separability(g1, out, organ):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    parts = ax.violinplot([g1["within_all"], g1["between_all"]], showmeans=True)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["within-region\n(split-half noise floor)", "between-region"])
    ax.set_ylabel("Pearson r (region pseudobulks)")
    ax.set_title(f"GATE 1 separability -- {organ}\nwithin {g1['within_floor']:.4f} vs between "
                 f"{g1['between']:.4f}  (between-dist = {g1['ratio']:.1f}x within-noise)")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_variance(g2, out, organ):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    vals = [g2["region_pct"], g2["depth_pct"], g2["residual_pct"]]
    ax.bar(["region", "seq depth", "residual"], vals,
           color=["#2c7fb8", "#d95f0e", "#bbbbbb"])
    for i, v in enumerate(vals): ax.text(i, v + .01, f"{v*100:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("share of top-PC variance"); ax.set_ylim(0, 1)
    ax.set_title(f"GATE 2 variance partition -- {organ}")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_disc_genes(g3, out, organ):
    n = min(20, len(g3["top"])); top = g3["top"][:n][::-1]
    F = g3["topF"][:n][::-1]; inm = g3["in_model"][:n][::-1]; tech = g3["tech"][:n][::-1]
    colors = ["#cccccc" if t else ("#2c7fb8" if m else "#d95f0e") for m, t in zip(inm, tech)]
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.barh(range(n), F, color=colors)
    ax.set_yticks(range(n)); ax.set_yticklabels(top, fontsize=8)
    ax.set_xlabel("ANOVA F (region-discriminating)")
    ax.set_title(f"GATE 3 top region genes -- {organ}\nblue=in model space, orange=dropped, grey=technical")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_bridge(sim, out, organ):
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(sim.values, vmin=float(sim.values[sim.values < 1].min()), vmax=1, cmap="magma")
    ax.set_xticks(range(len(sim))); ax.set_yticks(range(len(sim)))
    ax.set_xticklabels(sim.columns, rotation=90, fontsize=8); ax.set_yticklabels(sim.index, fontsize=8)
    for i in range(len(sim)):
        for j in range(len(sim)):
            ax.text(j, i, f"{sim.values[i,j]:.3f}", ha="center", va="center", fontsize=6, color="white")
    fig.colorbar(im, ax=ax, shrink=.8); ax.set_title(f"Region basal similarity in 10,716 space -- {organ}")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------- driver ---------------------------
def run(organ, figroot):
    figdir = Path(figroot) / f"{organ}_spatial_story"; figdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        if organ == "liver":
            a = load_liver(tmp); a = preprocess(a); markers = []
            region_def = "published anatomical zones (yu2022 L5 category)"
        elif organ == "kidney":
            a = load_kidney(tmp); a = preprocess(a)
            markers = assign_kidney_regions(a)
            region_def = "cortex vs medulla from canonical markers (external, non-circular)"
        else:
            raise ValueError(organ)

    comp = a.obs["region"].value_counts()
    g1 = gate1_separability(a); g2 = gate2_variance_partition(a); g3 = gate3_discriminating_genes(a)
    cov, sim = bridge(a)
    off = sim.values[~np.eye(len(sim), dtype=bool)]

    fig_region_map(a, figdir / "00_region_map.png", f"{organ}: regions")
    fig_separability(g1, figdir / "01_gate1_separability.png", organ)
    fig_variance(g2, figdir / "02_gate2_variance.png", organ)
    fig_disc_genes(g3, figdir / "03_gate3_discriminating_genes.png", organ)
    fig_bridge(sim, figdir / "04_bridge_similarity_modelspace.png", organ)

    # verdicts (thresholds are interpretive anchors, not hard project gates)
    v1 = "PASS" if g1["ratio"] >= 5 and g1["gap"] >= 0.02 else ("WEAK" if g1["ratio"] >= 2 else "FAIL")
    v2 = "PASS" if g2["region_pct"] >= 0.10 else ("WEAK" if g2["region_pct"] >= 0.03 else "FAIL")
    v3 = "PASS" if g3["retention"] >= 0.7 and g3["tech_frac"] <= 0.3 else "WEAK"

    L = [f"# Spatial story EDA -- {organ}", "",
         f"Region definition: {region_def}.",
         f"Spots x genes: **{a.n_obs} x {a.n_vars}**.  Region spot counts: "
         + ", ".join(f"{k}={v}" for k, v in comp.items()) + ".", "",
         "The predicted region-resolved arm needs each gate to hold. Verdict per gate:", "",
         f"## GATE 1 -- are regions distinguishable above the within-region noise floor?  [{v1}]",
         f"- within-region split-half Pearson (noise ceiling): **{g1['within_floor']:.4f}** +/- {g1['within_sd']:.4f}",
         f"- between-region Pearson: **{g1['between']:.4f}** +/- {g1['between_sd']:.4f}",
         f"- gap (within - between): **{g1['gap']:.4f}**;  between-region distance = **{g1['ratio']:.1f}x** the within-region noise",
         f"- read: two halves of the SAME region differ by {1-g1['within_floor']:.4f} (distance); two DIFFERENT "
         f"regions differ by {1-g1['between']:.4f}. Signal is the ratio of these.", "",
         f"## GATE 2 -- is region a real axis of variance, or swamped by depth?  [{v2}]",
         f"- region explains **{g2['region_pct']*100:.1f}%** of top-PC expression variance",
         f"- sequencing depth explains {g2['depth_pct']*100:.1f}%; residual {g2['residual_pct']*100:.1f}%", "",
         f"## GATE 3 -- does the region signal survive into the model gene space, as biology?  [{v3}]",
         f"- top-50 region-discriminating genes retained in 10,716 space: **{g3['retention']*100:.0f}%**",
         f"- technical (mito/ribo/MALAT1/Hb) fraction of top-50: **{g3['tech_frac']*100:.0f}%**",
         f"- top genes: {', '.join(g3['top'][:15])}", "",
         "## BRIDGE -- region-basal similarity in the model gene space",
         f"- {organ} Visium -> 10,716 coverage: **{cov*100:.1f}%**",
         f"- off-diagonal Pearson: min {off.min():.3f}, mean {off.mean():.3f}, max {off.max():.3f}", "",
         "## GATE 4 (cross-species) / GATE 5 (OOD) -- scaffolded",
         "- needs region-matched rodent labels (liver mouse = whole-sample only, r=0.927 prior; "
         "kidney mouse = .rds not yet extracted). OOD Mahalanobis from the cancer-line manifold to follow.", "",
         "## Caveats",
         ("- Region was MARKER-DEFINED (cortex/medulla), so the seed markers re-appear among the "
          "top discriminators by construction; the non-marker genes in the list and the whole-"
          "transcriptome separability are the independent signal. Single section (GSM6047774); "
          "23 sections + donor axis available for the variance gate."
          if organ == "kidney" else
          "- Regions are the published L5 zone labels; cluster definition is external to the "
          "separability test. Single section (yu2022 L5)."),
         "- Pearson on log1p pseudobulks; depth regressed only as a single covariate (a fuller "
         "PVCA would add donor/section). Thresholds for PASS/WEAK/FAIL are interpretive anchors.", "",
         "## Supporting (demoted niche battery)",
         "- cell-type niche architecture (nhood/co-occurrence/Ripley) is in the per-organ "
         "`*_spatial_eda` figures; it describes tissue architecture but the model conditions on "
         "region pseudobulks, not niches, so it is context not evidence.", "",
         "## Figures",
         f"- `{figdir}/00_region_map.png` -- regions in space (orientation)",
         f"- `{figdir}/01_gate1_separability.png` -- GATE 1 headline (within vs between)",
         f"- `{figdir}/02_gate2_variance.png` -- GATE 2 variance partition",
         f"- `{figdir}/03_gate3_discriminating_genes.png` -- GATE 3 top region genes + retention",
         f"- `{figdir}/04_bridge_similarity_modelspace.png` -- region basals in model space", ""]
    (figdir / "SUMMARY.md").write_text("\n".join(L))
    (REPO / f"results/tables/P1_spatial_story_{organ}.md").write_text("\n".join(L))
    print(f"[{organ}] GATE1 within {g1['within_floor']:.4f} between {g1['between']:.4f} ratio {g1['ratio']:.1f}x [{v1}] "
          f"| GATE2 region {g2['region_pct']*100:.1f}% depth {g2['depth_pct']*100:.1f}% [{v2}] "
          f"| GATE3 retain {g3['retention']*100:.0f}% tech {g3['tech_frac']*100:.0f}% [{v3}]")
    print(f"[{organ}] wrote {figdir}/SUMMARY.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ", required=True, choices=["liver", "kidney"])
    ap.add_argument("--figroot", default="/raid/home/joshua/claude_memory/downstream_2026/visuals")
    args = ap.parse_args()
    run(args.organ, args.figroot)
