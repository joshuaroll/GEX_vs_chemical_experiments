# Spatial Arm — Data and Code Manifest

**Version:** v1.0 (Phase 0 complete)
**Generated:** 2026-06-20
**Last updated:** 2026-06-20 (Phase 0 — dataset acquisition & MANIFEST)

This manifest is the **single source of truth for what data, code, and weights are
inside the spatial arm**. Update it at the end of every phase. The data-path test
(`tests/test_data_paths.py`) reads from this implicitly — keep them in sync.

---

## Source code (SHA-pinned)

| File/Checkpoint | Path | Source SHA | Purpose |
|----------------|------|-----------|---------|
| MultiDCP-CheMoE checkpoint | `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt` | `fbee15faade904cacd6484832cea211ebd4b28186ca40def40af3fea27c7d2a0` (SHA256) | Frozen CheMoE backbone (Conditions S-C, S-F); loaded in Phase 2 |
| MultiDCP-PDG checkpoint | `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/trained_models/chemoe_kpgt_MCF7_fold0/best_model.pt` | `8e9f0e441d76524bb4af25b819dc4730196b8b25abb5e5c6727a54a8bb7356d1` (SHA256) | Frozen PDG backbone (Conditions S-B, S-E); loaded in Phase 2 |
| MultiDCP repo | `/raid/home/joshua/projects/MultiDCP` | `871b8de0332045a3ad5ab3a39e689014317dadfe` (git SHA) | Source code for MultiDCP; pinned at this commit per sibling MANIFEST |

> **Note (XC-04):** Frozen checkpoints are recorded here for provenance only.
> `src/spatial/region_signature.py` holds a `NotImplementedError` seam — actual
> loading is Phase 2. Do NOT load these in Phase 0 or 1.

---

## Data files (SHA256-pinned)

### Human spatial (basal input, whole-transcriptome Visium)

| File | Path | SHA256 | Source | License | Whole-transcriptome | Notes |
|------|------|--------|--------|---------|--------------------|----|
| Yu 2022 liver L5 | `data/raw/spatial/yu2022_liver/L5_upload.zip` | `9643ff84165237a3ded3fd4dae88d552ac5eb948a82f7a69efd6c143ea6cfc39` | figshare 22321447 (DOI 10.6084/m9.figshare.22321447.v1) | GPL-3.0+ | Y | Leiden L5 categories; filtered_feature_bc_matrix.h5 inside zip |
| Yu 2022 liver L18 | `data/raw/spatial/yu2022_liver/L18_upload.zip` | `901a425bf061ce58748334e2a1c974fe11c0d29d8000f9348836c784eeac8520` | figshare 22321447 (DOI 10.6084/m9.figshare.22321447.v1) | GPL-3.0+ | Y | Leiden L18 categories |
| Andrews/Teichmann liver | `data/raw/spatial/andrews_liver/GSE185477_RAW.tar` | `7c73fd3deab5a0403a6d6dc036101b673ba197ee561d5594c329adc62da1b097` | GEO: GSE185477 | CC BY 4.0 | Y | Human liver Visium; 33514 genes; backup to Yu |
| Lake/KPMP kidney | `data/raw/spatial/lake_kpmp_kidney/GSE183456_RAW.tar` | `3437708ddd98c4a6d20282e5ea92804ec98bfa4ede89d87b409ba00a0608c457` | GEO: GSE183456 + GSE183279 (superseries) | see source (KPMP DUA — human review required before redistribution) | Y | KPMP Visium kidney; nested tar.gz/h5; 33514 genes |
| Abedini kidney | `data/raw/spatial/abedini_kidney/GSE211785_RAW.tar` | `9441b4e1c270944705b91a1ff9b46013173000dfcdc2fe93397cfbb2a942b9e9` | GEO: GSE211785 | see source (GEO license) | Y | AUDIT 2026-06-21: RAW.tar = images(.tif)+spatial(.json) only, NO counts; spatial counts ship separately as GSE211785_EXPORT_ST_counts.rds.gz (not fetched). Demoted to usable_as_input=False; lake_kpmp is the primary kidney input. |
| Canela kidney | `data/raw/spatial/canela_kidney/GSE202327_RAW.tar` | `74fc6d2a09e11eac7519144ffdd66a2997d46e295185df1a694bb57ae4e1ddda` | GEO: GSE202327 | see source (GEO license) | Y | long-read isoform spatial (.bb/SQANTI .gtf); no standard Visium gene x spot matrix. AUDIT 2026-06-21: demoted to usable_as_input=False (cannot feed the DE-rule count input). |
| Maynard DLPFC (metadata) | `data/raw/spatial/maynard_dlpfc/metadata_spatialLIBD.csv` | `d7a9c357dedeea2cf804810d1de8ad25f742acc2280c78bc9d03d7e91523e290` | spatialLIBD / LieberInstitute | Artistic-2.0 | Y | Metadata CSV only; full Visium objects require spatialLIBD R download. AUDIT 2026-06-21: demoted to usable_as_input=False (no expression on disk); chen_brain_mtg is the human brain input. |
| Chen brain MTG | `data/raw/spatial/chen_brain_mtg/GSE220442_counts_and_images.tar.gz` | `9c54b5607216ce9935bc3768844e154f28489fddb182a86b0703a5c2619db1bf` | GEO: GSE220442 | see source (GEO license) | Y | 10x Visium, 6 MTG sections (3 AD + 3 control), 36,601 genes; per-spot layer annotations in GSE220442_metadata.csv.gz. CORRECTED 2026-06-21 — prior GSE200474 was a wrong-accession error (ALS iPSC motor-neuron bulk RNA-seq, not Visium). |
| Kanemaru heart | `data/raw/spatial/kanemaru_heart/` | (no expected_files) | EGA: EGAS00001006330 | see source | Y | Heart deferred per ROADMAP; EGA controlled-access (DAC approval needed); directory empty. AUDIT 2026-06-21: demoted to usable_as_input=False. |
| Kuppe heart (control P1) | `data/raw/spatial/kuppe_heart/Visium_control_P1.h5ad` | `6da1eb233088630383e17efdf008aec526b0fef386d52be2d18a2a8f8224a58d` | Zenodo: 10.5281/zenodo.6578047 | CC BY 4.0 | Y | ADDED 2026-06-21. Public Visium, non-transplanted donor heart (basal). 4269 spots × 15730 genes, obsm['X_spatial']; coverage 0.834 of MultiDCP 10716 (> 0.80). Heart ACTIVATED 2026-06-21: usable_as_input=True. |
| Kuppe heart (control P7) | `data/raw/spatial/kuppe_heart/Visium_control_P7.h5ad` | `fb6d891966c8cbbaced295a57d03773997e9d018781c9e4eaf7568da99263aa1` | Zenodo: 10.5281/zenodo.6578047 | CC BY 4.0 | Y | Public Visium control donor heart (basal). |
| Kuppe heart (control P8) | `data/raw/spatial/kuppe_heart/Visium_control_P8.h5ad` | `c0eca744ef23d05b0858b0def7eb6f980a736c903a7bfed06a4e43b49b5eb75a` | Zenodo: 10.5281/zenodo.6578047 | CC BY 4.0 | Y | Public Visium control donor heart (basal). |
| Kuppe heart (control P17) | `data/raw/spatial/kuppe_heart/Visium_control_P17.h5ad` | `71f0b306910319ddb33bf491e43f0f21e69307c6b2c49850360357d297754b1c` | Zenodo: 10.5281/zenodo.6578047 | CC BY 4.0 | Y | Public Visium control donor heart (basal). |

### Mouse/rodent spatial (basal context Visium)

| File | Path | SHA256 | Source | License | Whole-transcriptome | Notes |
|------|------|--------|--------|---------|--------------------|----|
| GSE272564 mouse liver ctrl | `data/raw/spatial/gse272564_mouse_liver_ctrl/GSE272564_RAW.tar` | `0f6c15d33e8a8e28a1803883026451a16704c03e31aa6743c6f52c82d5236c35` | GEO: GSE272564 | see source (GEO license) | Y | Mouse Visium liver; 32245 genes; control arm used as basal context |
| GSE252772 mouse kidney | `data/raw/spatial/gse252772_mouse_kidney/GSE252772_RAW.tar` | `dd1f20418788fb6e752585b111879f6a8ea57e0106ff9b758432064bf18f2cbd` | GEO: GSE252772 | see source (GEO license) | Y | Mouse Visium kidney; .rds.gz format (R objects); no h5 extraction yet |
| GSE233983 mouse brain | `data/raw/spatial/gse233983_mouse_brain/GSE233983_RAW.tar` | `2988c54cd2ce3f50c439092f9bdfd4e8787c7305cbedd1393390c26532ee7629` | GEO: GSE233983 | see source (GEO license) | Y | Mouse Visium brain; 32245 genes; h5 files directly in tar |

### Validation anchors (mouse APAP liver, NOT basal input)

| File | Path | SHA256 | Source | License | Whole-transcriptome | Notes |
|------|------|--------|--------|---------|--------------------|----|
| GSE280652 APAP liver | `data/raw/spatial/gse280652_apap_liver/GSE280652_RAW.tar` | `984d90518b3c275c8f6f45353da3e5b57efef085e5821984a2dd3cc5d414e7aa` | GEO: GSE280652 | see source (GEO license) | Y | Validation-only (APAP-treated); never used as basal input |
| GSE272564 APAP liver | `data/raw/spatial/gse272564_apap_liver/GSE272564_RAW.tar` | `0f6c15d33e8a8e28a1803883026451a16704c03e31aa6743c6f52c82d5236c35` | GEO: GSE272564 | see source (GEO license) | Y | Validation-only (APAP-treated); same RAW.tar as mouse liver ctrl (shared download) |

### Toxicity label sets

| File | Path | SHA256 | Source | License | Notes |
|------|------|--------|--------|---------|-------|
| DILIst | `data/raw/labels/dilist/dilist.xlsx` | `4331ee9d16ae7641488161e4dc2c603c29e06baa8667dd16e7f3d635366e7e5e` | FDA NCTR LTKB | FDA (public) | 1279 drugs × binary DILI; SHA-verified copy from sibling dili_downstream |
| DILIrank 2.0 | `data/raw/labels/dilirank/dilirank.xlsx` | `1ca1352ff727af68e68e250eae2ed775bca8492335140ac0afd2233248694993` | FDA NCTR LTKB | FDA (public) | 1337 drugs × severity; SHA-verified copy from sibling dili_downstream |
| SIDER meddra_all_se | `data/raw/labels/sider/meddra_all_se.tsv.gz` | `119b2f5319a9398da83e5fe3419889010dbacf8d3eef590251b00c025e2b3f99` | SIDER 4.1 (embl.de) | see source | Brain SOC label source; SOC filter applied at use-time (Phase 1) |
| DIRIL | `data/raw/labels/diril/diril_dataset_508.xlsx` | `602edfda3da6b62bb52f8a88cd8fff5783f0635608dd0f2b0eb44299195b50dc` | FDA (Drug-Induced Renal Injury List); Connor et al., Drug Discov Today 29(4) 2024 | FDA (public) | Kidney nephrotoxicity labels; sheet "A. DIRIL (317)", 317 drugs × binary DIRI (My Findings: 171 Nephrotoxic / 146 Non-Nephrotoxic, no NaN), all with SMILES. RESOLVED 2026-06-21 from FDA (prior Elsevier-CDN candidate 404'd). |
| DICTrank | `data/raw/labels/dictrank/dictrank_dataset_508.xlsx` | `c50e55f6de01233dca24df749a1bff4fe0745d791b4d97ba05ebd7d9d193eb92` | FDA (Drug-Induced Cardiotoxicity Rank); Qu et al., Drug Discov Today 28(11) 2023 | FDA (public) | Heart cardiotoxicity labels; sheet "Table S1", 1318 drugs ranked DICT-concern (most 341 / less ~527 / no 343 / ambiguous 107). NO SMILES column (keyed by drug/active-ingredient name -> needs structure join). Heart DEFERRED per ROADMAP; acquired 2026-06-21 for readiness. |

### Ortholog map (processed, gitignored)

| File | Path | SHA256 | Source | Notes |
|------|------|--------|--------|-------|
| BioMart raw TSV | `data/raw/spatial/biomart/orthologs_raw_116_20260620.tsv` | `607dd457581753ce2247905633f4fd8def813bfb42fba44322ca8ce30b7a5551` | Ensembl BioMart release 116 (2026-06-20) | 219938 rows; XC-10: release+date baked into filename |
| One2one ortholog pairs | `data/processed/spatial/orthologs_h_m_r_one2one.tsv` | `b25404a9d79dde3dadd94645426556bbe2df2ffc21cdf12a2f5bd05aed7ec12c` | Filtered from BioMart raw; see orthology.py | 15956 human-mouse-rat one2one pairs; 92.75% dropped (expected) |

### Gene space reference (processed, gitignored)

| File | Path | SHA256 | Source | Notes |
|------|------|--------|--------|-------|
| MultiDCP 10716 symbols | `data/processed/spatial/multidcp_10716_symbols.txt` | (compute at use-time) | `MultiDCP/MultiDCP/data/pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv` columns | 10716 human gene symbols; N_PDG denominator for Halt Gate 1 coverage check |

---

## Environment

| Item | Value |
|------|-------|
| Conda environment | `dili_v04_env` |
| squidpy version | `1.8.2` (installed 2026-06-20 via `pip install squidpy`) |
| scanpy version | `1.11.5` (pre-existing) |
| anndata version | `0.12.10` (pre-existing) |
| h5py version | `3.14.0` (pre-existing) |
| Env snapshot | `MANIFEST_env_snapshot.yml` (generated 2026-06-20) |
| Snapshot SHA256 | (compute at use-time) |

### squidpy install record

```bash
conda run -n dili_v04_env pip install squidpy
# Installed: squidpy 1.8.2
# Date: 2026-06-20
# Purpose: Moran's I spatial-autocorrelation QC (Phase 1 spatial-QC)
```

---

## Phase 0 Coverage Gate Evidence

See `results/tables/P0_coverage.md` for per-dataset coverage against the 10,716-gene
MultiDCP space.

**Halt Gate 1 status: NOT FIRED**
- Human basal Visium datasets with readable feature matrices: all >99% coverage (PASS)
- Rodent basal Visium datasets: genome-scale gene counts (>32000 genes; PASS)
- Rodent datasets show near-zero human-symbol coverage by design; cross-species alignment
  via ortholog map (Phase 2, not a gate violation)

---

## License review reminder (user_setup gate)

The following datasets have `License: see source` and require human review before any
redistribution:
- `lake_kpmp_kidney` — KPMP Data Use Agreement
- `abedini_kidney` — GEO standard (verify for redistribution)
- `canela_kidney` — GEO standard (verify for redistribution)
- `chen_brain_mtg` — GEO standard (verify for redistribution)
- `kanemaru_heart` — EGA access agreement
- Mouse/rodent GEO datasets — GEO standard terms

Record an SPDX-style tag + URL in the License column above before any redistribution.

---

## Phase 1 — Reused external inputs (v0.5 dili_downstream + PDG manifold)

> Recorded before first read per D-05; these are static external inputs, not
> live cross-project dependencies. Do NOT touch /raid/home/joshua/.planning (isolation rule).

| File | Path | SHA256 | Source | License | Whole-transcriptome | Notes |
|------|------|--------|--------|---------|--------------------|----|
| wangli measured DE | `../dili_downstream/data/processed/wangli_measured_de.npy` | `aaeb07ce13222af49c873be8652b01597a4c638119efd306d4f4d7c04b9397f5` | dili_downstream v0.5 Phase 1 output | see source | N/A | External input; shape (5517, 978) float32; ceiling DE matrix (D-05) |
| wangli profiles | `../dili_downstream/data/processed/wangli_profiles.csv` | `963ea3e836b061a6eb2fa066ef5634b4cf9e9cfda8537ea45e98ef1ec3cf872e` | dili_downstream v0.5 | see source | N/A | 5517 rows x 12 cols; profile metadata for the DE matrix |
| dili canonical SMILES | `../dili_downstream/data/processed/dili_canonical.csv` | `8e30f71278fad1391eacab0fc287d2898361249cd119ade810f86212243a3525` | dili_downstream v0.5 | see source | N/A | SMILES join layer 1; join key drug_name->name_lower |
| dilist SMILES resolved | `../dili_downstream/data/processed/dilist_smiles_resolved.csv` | `b57be7557b52fd6c3f094e718e560a522452895ee7c317a67084b1c83ebbdf26` | dili_downstream v0.5 | see source | N/A | SMILES join (DILIst secondary, D-04) |
| drugbank SMILES index | `../dili_downstream/data/processed/drugbank_smiles_index.csv` | `2ad9eb4d642f954dfdc8aa30deb0a4089b99292af0917af4df0e034c05134a5e` | DrugBank index via dili_downstream v0.5 | see source | N/A | 38917 rows; SMILES fallback layer 2; key name_lower |
| pdg diseased manifold | `/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv` | `e1e38f118c91be64065bdb8458479bfe3b0fc0a9c118a7e5352523f5a2306fb1` | MultiDCP_pdg | see source | N/A | (10, 10716) cancer-line manifold; EDA-03 OOD reference |
