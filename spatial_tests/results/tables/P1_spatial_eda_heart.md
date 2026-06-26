# Spatial EDA -- heart (Kuppe control donor P1)

Whole-transcriptome Visium, basal (healthy) tissue. This characterizes the spatial INPUT structure; it complements (does not replace) the tabular toxicity bracket.

- Spots x genes: **4269 x 15730**  |  coords in `obsm['X_spatial']`
- No H&E image in this h5ad (CELLxGENE schema strips it); spots shown without histology.
- Domain label `cell_type_original`: 8 cell types (argmax of deconvolution).

## How to read these figures (one line each)
- **01_qc_in_space.png** -- QC mapped onto tissue coordinates.
- **02_celltype_proportions.png** -- Cell-type composition from deconvolution.
- **03_domains_argmax.png** -- Dominant cell type per spot (argmax of the proportions in fig 02).
- **03b_domains_rare_niches.png** -- Same domains with the dominant cardiomyocyte background greyed out (squidpy groups=) so the sparse vascular/immune/fibroblast niches are visible.
- **04_svg_top_genes_in_space.png** -- Top spatially variable genes (highest Moran's I) plotted on the tissue.
- **05_nhood_enrichment.png** -- Which domains physically sit next to which.
- **06_co_occurrence.png** -- Starting from a cardiomyocyte spot, how much more/less likely is each other type nearby as you walk outward.
- **07_ripley_L.png** -- Is each cell type clumped, evenly spread, or random? Ripley's L(r) vs radius; the pale grey curve is the random (CSR) reference from simulations.
- **08_centrality_scores.png** -- Per-domain graph roles: degree centrality (a hub touching many types), clustering coefficient (tight self-patches), closeness (centrally embedded).
- **09_region_basal_similarity_modelspace.png** -- The project bridge: how distinct are region basal profiles in the gene space the MultiDCP model actually sees.

## Tissue architecture (spot counts per dominant type)
| Domain | spots | frac | robust(>=20) |
|---|---|---|---|
| Cardiomyocyte | 4053 | 0.949 | yes |
| Fibroblast | 118 | 0.028 | yes |
| vSMCs | 50 | 0.012 | yes |
| Myeloid | 26 | 0.006 | yes |
| Endothelial | 15 | 0.004 | no |
| Cycling.cells | 4 | 0.001 | no |
| Mast | 2 | 0.000 | no |
| Pericyte | 1 | 0.000 | no |

## Spatial autocorrelation (Moran's I)
- SVGs (pval_norm < 0.05): **2329** (14.8% of genes tested)
- Top SVGs: FTH1, FTL, MALAT1, NPPB, ACTA1, MYH7, MTRNR2L12, MYL2, NPPA, HBB, HMOX1, TNNI3
- Top-20 SVG retention in 10,716 model space: **15/20**
- Moran's I is computed on the spatial graph (k=6 neighbours); the SVG set shifts if k changes, so treat it as a strong graph-conditional baseline and confirm hits visually (fig 04).

## Project bridge -- region basal distinctness in the model's gene space
- Heart Visium -> 10,716 coverage: **83.4%**
- ALL regions off-diagonal Pearson: min 0.555, mean 0.816, max 0.972 (min driven by 1-2 spot rare niches)
- **ROBUST regions (>=20 spots: Cardiomyocyte, Fibroblast, vSMCs, Myeloid) off-diagonal Pearson: min 0.929, mean 0.951, max 0.972**
- Liver reference (yu2022 zones): 0.988-0.998 (near-identical)

## Interpretation & caveats
- **Argmax vs mixture.** Domains (fig 03) are the dominant type per spot; the spot is really a mixture (fig 02). Don't read 'this spot is a fibroblast' literally.
- **Tiny-region pseudobulk is noise.** A 1-2 spot 'region' gives an unstable mean; the headline contrast uses robust regions (>=20 spots) only.
- **Dominated section.** Cardiomyocyte is 95% of spots, so heart 'regions' are sparse cell-type niches, not clean anatomical zones like liver zonation.
- **Single section, single donor.** P1 only; P7/P8/P17 controls are on disk for replication.
- **No batch integration / no histology QC against an image** (image absent).

## Figures (full captions)
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/heart_spatial_eda/01_qc_in_space.png`
  - QC mapped onto tissue coordinates. Each dot is a ~55 um Visium spot; color = total RNA (n_counts), genes detected (n_genes), and mitochondrial fraction (percent.mt). We are checking signal is even across the slide, not pooling at one edge (a capture/permeabilization artifact) or flagging a degraded/necrotic band (high mito).
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/heart_spatial_eda/02_celltype_proportions.png`
  - Cell-type composition from deconvolution. A Visium spot mixes multiple cells, so each panel is the ESTIMATED PROPORTION (0-1) of one cell type per spot, not a hard call. This is the faithful view of tissue architecture: cardiomyocyte-rich muscle with vascular/fibroblast niches threaded through it.
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/heart_spatial_eda/03_domains_argmax.png`
  - Dominant cell type per spot (argmax of the proportions in fig 02). CAVEAT: argmax discards the mixture -- a spot that is 0.44 cardiomyocyte / 0.32 fibroblast is called 'cardiomyocyte'. Shown for orientation; the proportion maps (fig 02) are the honest version.
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/heart_spatial_eda/03b_domains_rare_niches.png`
  - Same domains with the dominant cardiomyocyte background greyed out (squidpy groups=) so the sparse vascular/immune/fibroblast niches are visible. Spot counts are tiny for some types (see table) -- read with care.
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/heart_spatial_eda/04_svg_top_genes_in_space.png`
  - Top spatially variable genes (highest Moran's I) plotted on the tissue. Moran's I flags genes whose expression forms spatial PATCHES rather than salt-and-pepper noise; this overlay is the visual confirmation that the statistic reflects a real pattern. Cardiac genes (MYH7, NPPA/NPPB, TNNI3) lead.
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/heart_spatial_eda/05_nhood_enrichment.png`
  - Which domains physically sit next to which. Cell type x cell type z-score vs a permutation null: red = neighbors more than chance, blue = spatially segregated, diagonal = self-aggregation.
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/heart_spatial_eda/06_co_occurrence.png`
  - Starting from a cardiomyocyte spot, how much more/less likely is each other type nearby as you walk outward. Ratio > 1 = drawn together at that radius, < 1 = repelled; the crossing point is a characteristic interaction length.
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/heart_spatial_eda/07_ripley_L.png`
  - Is each cell type clumped, evenly spread, or random? Ripley's L(r) vs radius; the pale grey curve is the random (CSR) reference from simulations. A type's curve well above it = self-clustered (cardiomyocyte fills the muscle), near it = random. Robust domains only (>=20 spots).
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/heart_spatial_eda/08_centrality_scores.png`
  - Per-domain graph roles: degree centrality (a hub touching many types), clustering coefficient (tight self-patches), closeness (centrally embedded).
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/heart_spatial_eda/09_region_basal_similarity_modelspace.png`
  - The project bridge: how distinct are region basal profiles in the gene space the MultiDCP model actually sees. Lower Pearson = more regional contrast for a region-resolved predicted signature to exploit.

_Run elapsed: 38.9s_