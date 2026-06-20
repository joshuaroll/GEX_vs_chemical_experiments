# Phase 0: Dataset acquisition & MANIFEST - Research

**Researched:** 2026-06-20
**Domain:** Bioinformatics data acquisition — spatial transcriptomics (Visium), rodent toxicogenomics, per-organ toxicity label sets, cross-species ortholog tables, provenance/MANIFEST discipline
**Confidence:** HIGH on accession resolution and tooling; MEDIUM on exact rodent-spatial dataset selection (many valid candidates; final pick is a planning decision)

## Summary

Phase 0 is a pure data-and-provenance phase: get every planned input dataset onto disk, versioned with SHAs and licenses in `MANIFEST.md`, report gene-space coverage of each Visium dataset against the 10,716-gene MultiDCP space, and build a human–mouse–rat one-to-one ortholog map in a new `src/spatial/orthology.py`. No model code. The single binary outcome that matters for Halt Gate 1 is: is every planned input dataset (a) available and (b) whole-transcriptome? This research verified availability directly against live registries.

Two results change the plan relative to the upstream docs. First, **the Yu 2022 liver Figshare accession in DOC 09 and in the existing `datasets.py` is wrong on both counts**: the design doc/ROADMAP cite Figshare `10.6084/m9.figshare.17058105` (returns HTTP 404 — "Entity not found"), and the existing `src/spatial/datasets.py` still carries the even-more-wrong GEO `GSE189994` (an unrelated m6A study, per DOC 09). The actual data is Figshare article **22321447** (DOI `10.6084/m9.figshare.22321447.v1`, title "L5_L18_normalliver", GPL-3.0+, two files `L5_upload.zip` 366 MB + `L18_upload.zip` 1.47 GB, MD5s available) `[VERIFIED: figshare API]`. Second, **the existing `datasets.py`/`config.py` registry contains several of the exact accessions DOC 09 corrects** (Yu GSE189994, Maynard GSE144239, Lake/KPMP GSE211785-as-primary, Siletti mislabeled as MERFISH). These must be corrected in this phase *before* download scripts are written, which is exactly what DATA-01's acceptance criterion demands.

All other confirmed human/validation accessions resolve and are tagged whole-transcriptome Visium at NCBI GEO: Andrews/Teichmann liver backup (GSE185477), Lake/KPMP kidney (GSE183456 + GSE183279), Abedini kidney substitute (GSE211785), and both APAP mouse-liver validation series (GSE280652, GSE272564) `[VERIFIED: NCBI GEO]`. spatialLIBD (Maynard DLPFC), atlas.kpmp.org, Open TG-GATEs (DBArchive), NTP DrugMatrix, SIDER, and Ensembl BioMart all return HTTP 200 `[VERIFIED: HTTP HEAD]`. Rodent whole-transcriptome Visium for liver/kidney/brain exists abundantly on GEO on Visium-compatible platforms (GPL24247 mouse, GPL25947 rat); the design's claim that these "must be identified and verified in this phase" is satisfiable — concrete candidates are listed below — but the final selection (which healthy/control series per organ) is a planning decision, not a blocker.

**Primary recommendation:** Correct the three bad accessions in `datasets.py`/`config.py` first (Yu→figshare 22321447, Maynard→spatialLIBD, Lake/KPMP→GSE183456+GSE183279 primary with GSE211785 demoted to substitute, Siletti→snRNA-seq not MERFISH). Build a declarative download registry (one entry per dataset: URL/accession + access mechanism + expected files + license), download into `data/raw/spatial/<dataset>/`, compute SHA256s into `MANIFEST.md`, implement `orthology.py` against the live Ensembl BioMart `martservice` REST endpoint (verified working — returns `orthology_type` so one2one filtering is trivial), and add `squidpy` to `dili_v04_env`. Gate every dataset on "is it whole-transcriptome Visium" and write `HALT_REASON.md` if any input fails.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Dataset download / fetch | Download scripts (`scripts/`) | — | Side-effecting I/O; kept out of pure `src/spatial/` library per existing module discipline (gene_alignment.py is explicitly "pure library, no paths") |
| Dataset registry / metadata | `src/spatial/datasets.py` [extend] | `config.py` | Pure config NamedTuples already established; add access-mechanism + license + whole_transcriptome flag fields |
| Ortholog table construction | `src/spatial/orthology.py` [new] | `gene_alignment.py` [extend] | Pure-ish library; network fetch belongs behind a cache-file boundary so the function can run offline from a cached BioMart TSV |
| Gene-space coverage report | `gene_alignment.py` `coverage_fraction()` [exists] | `datasets.py` | The coverage primitive already exists and is tested; Phase 0 only wires it to real Visium gene lists |
| Provenance / SHA / license record | `MANIFEST.md` [new] | `scripts/` (sha helper) | Single source of truth; mirrors sibling `dili_downstream/MANIFEST.md` and `multihead_dili/MANIFEST.md` conventions |
| Data-path verification | `tests/` (`test_data_paths.py`) | `conftest.py` (`network` marker) | Sanity-only tests per hard rules ("never on model outputs") |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scanpy | 1.11.5 (installed) | Read Visium `.h5ad` / 10x `filtered_feature_bc_matrix.h5`, extract `.var_names` (gene lists) | Already in `dili_v04_env`; canonical scRNA/spatial reader `[VERIFIED: importlib.metadata]` |
| anndata | 0.12.10 (installed) | In-memory AnnData container for Visium objects | Already installed; backs scanpy `[VERIFIED]` |
| h5py | 3.14.0 (installed) | Low-level read of 10x `.h5` and Seurat-exported HDF5 | Already installed `[VERIFIED]` |
| requests | 2.32.3 (installed) | Figshare API, Ensembl BioMart REST, NCBI E-utilities, KPMP downloads | Already installed; sufficient for all REST fetches `[VERIFIED]` |
| pandas | 2.2.3 (installed) | Ortholog TSV parsing, label-set tables (DILIst/DIRIL/SIDER), coverage tables | Already installed `[VERIFIED]` |
| squidpy | latest (NOT installed) | Moran's I spatially-variable-gene QC (needed P1, but env add belongs in P0 per DATA-02) | Design doc §0 mandates it; canonical spatial-QC library `[CITED: design §0; constraints CON-spatial-qc]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pooch | NOT installed | Download-with-checksum helper (registry → fetch → verify SHA) | Optional convenience for download scripts; `requests` + `hashlib` suffices without it |
| GEOparse | NOT installed | Programmatic GEO supplementary-file listing/download | Optional; GEO supplementary files are also reachable by direct FTP/HTTPS URL without it |
| pybiomart | NOT installed | Pythonic wrapper over Ensembl BioMart | Optional — the raw `martservice` XML-over-`requests` call is verified working and has zero new deps (recommended over adding pybiomart) |
| tangram-sc | NOT installed | Reference-based imputation to promote a targeted panel to whole-transcriptome | OUT OF SCOPE for P0 default path; only if a MERFISH/Xenium panel is ever promoted to input (it is not, by decision) — design doc lists it as *optional* |
| openpyxl | check | Read `.xlsx` label sets (DILIst, DILIrank are Excel per sibling MANIFEST) | If a label set ships as Excel |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| raw `requests` + BioMart XML | `pybiomart` | pybiomart is friendlier but adds a dependency and an abstraction; the raw call is verified and self-documenting. Recommend raw. |
| `requests`+`hashlib` downloader | `pooch` | pooch gives free checksum-verified caching but is another dep; for ~10 datasets a 30-line helper is enough |
| Ensembl BioMart for orthologs | MGI + RGD flat files only | BioMart gives all three species in one query with `orthology_type`; MGI/RGD flat files are the authoritative per-species cross-references and are the right *reconciliation/audit* source. Recommend BioMart primary, MGI/RGD as a cross-check (DOC/SPEC name all three). |

**Installation (env additions for DATA-02):**
```bash
conda activate dili_v04_env
pip install squidpy          # required (P1 Moran's I QC; env add recorded in P0 MANIFEST)
# Optional convenience, not required:
# pip install pooch GEOparse openpyxl
# tangram-sc: only if a panel is ever promoted to input (not in default plan)
conda env export -n dili_v04_env > MANIFEST_env_snapshot.yml   # record in MANIFEST
```

**Version verification:** squidpy/pybiomart/pooch/GEOparse/loompy/tangram-sc are confirmed ABSENT in `dili_v04_env`; scanpy 1.11.5, anndata 0.12.10, h5py 3.14.0, requests 2.32.3, pandas 2.2.3, scikit-learn 1.8.0, scipy 1.16.3, rdkit 2024.09.4, torch 2.6.0+cu124, pytest 9.0.2 are confirmed PRESENT `[VERIFIED: importlib.metadata in dili_v04_env]`. Pin the squidpy version actually installed into MANIFEST at install time (do not pre-pin from training data).

## Architecture Patterns

### System Architecture Diagram

```
                          ┌─────────────────────────────────────────────┐
                          │  src/spatial/datasets.py  (registry, extend) │
                          │  one entry per dataset:                      │
   accession corrections  │  name, organ, species, platform, accession, │
   (Yu/Maynard/KPMP/      │  access_mechanism, expected_files, license,  │
    Siletti) ───────────▶ │  whole_transcriptome:bool, usable_as_input  │
                          └───────────────┬─────────────────────────────┘
                                          │ (declarative metadata)
                                          ▼
   ┌──────────────────┐   reads registry  ┌──────────────────────────────┐
   │ scripts/         │◀──────────────────│  download driver             │
   │  download_*.py   │                   │  per-mechanism fetchers:     │
   │  (side-effects)  │──── writes ──────▶│   • Figshare API (Yu)        │
   └──────────────────┘                   │   • GEO suppl HTTPS (KPMP,   │
            │                             │     APAP, rodent)            │
            ▼                             │   • spatialLIBD/Bioconductor │
   data/raw/spatial/<dataset>/           │   • KPMP atlas portal        │
   (untouched downloads)                 │   • Open TG-GATEs / DrugMatrix│
            │                             └──────────────────────────────┘
            │ scanpy/h5py read .var_names
            ▼
   ┌──────────────────────────┐   coverage_fraction(visium_genes,
   │ gene_alignment.py        │       MULTIDCP_10716)  ──▶ per-dataset %
   │  coverage_fraction()     │
   └──────────────────────────┘
            │
            ▼                         ┌─────────────────────────────────┐
   ┌──────────────────────────┐      │ src/spatial/orthology.py [new]  │
   │ Ensembl BioMart REST     │─────▶│  fetch human↔mouse↔rat,         │
   │  martservice XML query   │      │  keep ortholog_one2one only,    │
   │  (returns orthology_type)│      │  drop many2many, report dropped │
   └──────────────────────────┘      │  fraction; cache TSV to disk    │
            ▲                         └──────────────┬──────────────────┘
            │ cross-check                            │
   MGI + RGD flat files (audit)                      ▼
                                      data/processed/spatial/orthologs_h_m_r_one2one.tsv
            │
            ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ MANIFEST.md  ── paths, SHA256, license, version, env snapshot,    │
   │               frozen MultiDCP/CheMoE checkpoint paths+SHA          │
   └──────────────────────────────────────────────────────────────────┘
            │
            ▼
   tests/test_data_paths.py  (sanity: files exist, shapes/gene-lists nonempty,
                              ortholog map nonempty + one2one; @network where needed)
```

A reader can trace: corrected registry → download driver dispatches per access mechanism → raw files on disk → gene lists read → coverage computed against 10,716 space; in parallel BioMart → one2one ortholog TSV; both feed MANIFEST and the data-path test.

### Recommended Project Structure
```
spatial_tests/
├── src/spatial/
│   ├── datasets.py            # [EXTEND] correct accessions; add species/license/mechanism/whole_transcriptome fields
│   ├── gene_alignment.py      # [EXISTS] coverage_fraction() reused as-is
│   └── orthology.py           # [NEW] BioMart one2one ortholog builder + dropped-fraction report
├── scripts/                   # [NEW] side-effecting download drivers (one per mechanism or one driver reading registry)
│   ├── download_spatial.py
│   ├── download_labels.py
│   ├── build_orthologs.py     # thin CLI over orthology.py (writes cached TSV)
│   └── compute_sha256.py
├── configs/                   # [NEW] optional per-organ YAML (deferred-friendly; can be empty in P0)
├── data/raw/spatial/<dataset>/        # [EXISTS empty] untouched downloads
├── data/processed/spatial/            # [EXISTS empty] ortholog TSV, coverage tables
├── results/tables/            # [NEW] P0_coverage.md, P0_orthologs.md
├── tests/
│   └── test_data_paths.py     # [NEW] sanity only (verification target named in design doc)
└── MANIFEST.md                # [NEW] single source of truth
```

> **Test-path note:** The design doc + REQUIREMENTS name the verification target `tests/test_data_paths.py` (flat under `tests/`), but the existing 124 tests live under `tests/spatial/`. Recommend creating `tests/test_data_paths.py` exactly as named (matches the acceptance criterion and the sibling `dili_downstream` convention) while leaving `tests/spatial/` untouched. Confirm at planning whether to nest it as `tests/spatial/test_data_paths.py` instead — the acceptance criterion text says the former.

### Pattern 1: Declarative dataset registry → generic download driver
**What:** Each dataset is a frozen record carrying *how* to fetch it (access mechanism enum) plus expected files and license. A single driver dispatches on mechanism. Keeps `src/spatial/` pure and side-effects in `scripts/`.
**When to use:** Always here — there are ~10 datasets across 5 distinct access mechanisms (Figshare API, GEO supplementary HTTPS, Bioconductor/spatialLIBD, KPMP portal, ToxGenomics DBs).
**Example:**
```python
# src/spatial/datasets.py  (extend existing NamedTuple) — Source: existing datasets.py pattern
class SpatialDataset(NamedTuple):
    name: str
    organ: str
    species: str                  # NEW: "human" | "mouse" | "rat"
    platform: str
    accession: str                # CORRECTED values
    access_mechanism: str         # NEW: "figshare_api"|"geo_supp"|"spatialLIBD"|"kpmp"|"url"
    expected_files: tuple[str, ...]  # NEW
    license: str                  # NEW
    whole_transcriptome: bool     # NEW: the Halt-Gate-1 flag
    usable_as_input: bool
    region_annotation_source: str
    slug: str                     # NEW: canonical lowercase data/raw/spatial/<slug> dir name (single source of truth)
```

### Pattern 2: BioMart one2one ortholog query via raw REST (no new deps)
**What:** POST/GET an XML query to `https://www.ensembl.org/biomart/martservice`, request `*_homolog_ensembl_gene` + `*_homolog_orthology_type` for mouse and rat against the human dataset, filter rows where BOTH types == `ortholog_one2one`.
**When to use:** Building `orthology.py`. Verified working this session.
**Example:**
```python
# Source: [VERIFIED: live Ensembl martservice call, 2026-06-20] — chr21 sample returned
# 5665 rows, 180 one2one; full-genome query is the same query without the chromosome filter.
import requests
QUERY = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" datasetConfigVersion="0.6">
 <Dataset name="hsapiens_gene_ensembl" interface="default">
  <Attribute name="ensembl_gene_id"/>
  <Attribute name="external_gene_name"/>
  <Attribute name="mmusculus_homolog_ensembl_gene"/>
  <Attribute name="mmusculus_homolog_associated_gene_name"/>
  <Attribute name="mmusculus_homolog_orthology_type"/>
  <Attribute name="rnorvegicus_homolog_ensembl_gene"/>
  <Attribute name="rnorvegicus_homolog_associated_gene_name"/>
  <Attribute name="rnorvegicus_homolog_orthology_type"/>
 </Dataset>
</Query>"""
r = requests.get("https://www.ensembl.org/biomart/martservice", params={"query": QUERY}, timeout=180)
# Then: keep rows where mouse_type == rat_type == "ortholog_one2one"; report dropped fraction;
# CACHE the raw TSV to data/raw/spatial/biomart/ so orthology.py is reproducible offline.
```
> Pin the Ensembl release in MANIFEST (BioMart serves the current release; record release number + query date for reproducibility — this is the XC-10 time-leakage discipline applied to gene annotations).

### Pattern 3: Whole-transcriptome gate (Halt Gate 1) is a data property check, not a count
**What:** A dataset passes if its `.var_names` after read is genome-scale (≈15k–35k genes), not a fixed panel (~313/500/960). Compute coverage of the 10,716 space; Visium should be near-100%, panels are ~5%.
**When to use:** As the explicit gate in the data-path test and the coverage report.
**Example:**
```python
# Source: existing gene_alignment.py coverage_fraction (tested)
from src.spatial.gene_alignment import coverage_fraction
from src.spatial.config import N_PDG  # 10716
cov = coverage_fraction(visium_var_names, multidcp_10716_symbols)
assert cov > 0.80, f"Halt Gate 1: {name} covers only {cov:.1%} of the 10,716 space — not whole-transcriptome"
```

### Anti-Patterns to Avoid
- **Carrying the stale accessions forward:** `datasets.py` and `config.py` still contain GSE189994 (Yu), GSE144239 (Maynard), GSE211785-as-Lake/KPMP-primary, and Siletti-as-MERFISH. Writing download scripts before correcting these violates DATA-01's acceptance criterion and would download wrong/empty data. Fix the registry first.
- **Trusting DOC 09's Figshare DOI verbatim:** DOC 09's "correction" (figshare 17058105) is itself wrong (404). Verify accessions against live registries, not just against the correction doc.
- **Symbol-based coverage without normalization:** ~44 of 978 L1000 landmarks are obsolete HGNC aliases; do a one-time symbol normalization (alias→approved) at gene-list comparison time or coverage is understated (`[CITED: CON-gene-space]`).
- **Putting network I/O inside `src/spatial/` pure functions:** existing modules (gene_alignment.py) are explicitly pure with "NO hardcoded absolute paths." Keep fetch logic in `scripts/` and behind a cached-TSV boundary in `orthology.py`.
- **Letting drug-treated spots into a basal profile:** APAP series (GSE280652/GSE272564) are validation-only, never basal context (`[CITED: §7 no-leakage]`).
- **Two independent slugifications:** the download driver and the data-path test must NOT each compute their own directory name from `name`. Use the single `slug` field on `SpatialDataset` as the one source of truth for `data/raw/spatial/<slug>/` so a rename can never desynchronize them.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Read Visium / 10x matrices | Custom HDF5/MTX parser | `scanpy.read_visium` / `scanpy.read_10x_h5` / `anndata.read_h5ad` | Handles 10x HDF5 layout, feature types, spatial coords; already installed |
| Cross-species orthologs | Manual symbol-name matching | Ensembl BioMart `orthology_type` field | Symbol matching silently mis-pairs paralogs and many-to-many; BioMart gives explicit one2one classification |
| Spatially-variable-gene QC (P1) | Custom Moran's I | `squidpy.gr.spatial_autocorr(mode="moran")` | Correct spatial-weights handling; the design mandates squidpy |
| Checksum-verified download | Re-download loop logic | `hashlib.sha256` + a small registry (or `pooch`) | Trivial but error-prone; MANIFEST needs the SHA anyway |
| HGNC alias resolution | Hardcoded alias dict | HGNC/Ensembl alias table (or BioMart `external_synonym`) | 44 obsolete aliases are a known list but drift over releases |

**Key insight:** This phase is 90% "fetch the right bytes and record their provenance correctly." The genuinely hard part is *accession correctness* (already shown to be wrong in two places) and *ortholog one2one discipline* — both have authoritative sources; hand-rolling either is how silent data errors enter the pipeline.

## Runtime State Inventory

> This is a fresh data-acquisition phase, not a rename/refactor. Most categories are N/A. The one real "stored state" concern is the existing registry carrying wrong accessions.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | `src/spatial/datasets.py` + `config.py` hold STALE/WRONG accessions: Yu=GSE189994 (config line 68), Maynard=GSE144239 (line 122), Lake/KPMP lists GSE211785 as primary accession (line 95), Siletti labeled MERFISH usable_as_input=False but mislabeled vs snRNA-seq | Code edit: correct all four in the registry before download scripts. Yu→figshare 22321447. |
| Live service config | None | None — verified: no external service stores project state for P0 |
| OS-registered state | None | None |
| Secrets/env vars | None required for P0 (all sources are public, no auth) | None — KPMP/Figshare/GEO/BioMart need no token (KPMP atlas may require click-through ToS, see Open Questions) |
| Build artifacts | `dili_v04_env` will gain `squidpy`; record `conda env export` snapshot in MANIFEST | Env edit + MANIFEST record (DATA-02) |

## Common Pitfalls

### Pitfall 1: Stale accession propagation
**What goes wrong:** Download scripts pull GSE189994 (an m6A macrophage study) thinking it's Yu liver, or pull nothing (figshare 17058105 is 404).
**Why it happens:** Three layers disagree — `datasets.py` (GSE189994), DOC 09 "correction" (figshare 17058105, also wrong), and reality (figshare 22321447).
**How to avoid:** Single-source the corrected accessions into the registry, each VERIFIED against the live registry, then have scripts read only the registry.
**Warning signs:** Download returns HTML error pages, zero-byte files, or gene lists that don't look like liver.

### Pitfall 2: Targeted panel mistaken for whole-transcriptome
**What goes wrong:** A MERFISH (~500), Xenium (~313), or CosMx (~960) dataset is fed as basal input; zero-padding ~95% of a full-rank-trained encoder is severe OOD.
**Why it happens:** Some panel datasets ship convenient region annotations and look like "spatial data."
**How to avoid:** The `whole_transcriptome` flag + coverage>0.80 gate. Panels are annotation-only (`[CITED: CON-gene-space, C1]`).
**Warning signs:** Coverage of the 10,716 space < 10%.

### Pitfall 3: Many-to-many orthologs collapsed instead of dropped
**What goes wrong:** A human gene with two mouse orthologs gets averaged/picked, silently mis-aligning cross-species expression.
**Why it happens:** Convenience; one2one filtering loses genes.
**How to avoid:** Filter on `orthology_type == ortholog_one2one` for BOTH mouse and rat; DROP the rest; report dropped fraction (`[CITED: XC-08, CON-ortholog-alignment]`).
**Warning signs:** Ortholog table has more rows than unique human genes.

### Pitfall 4: HGNC alias drift understates coverage
**What goes wrong:** Coverage looks artificially low (~95.5% instead of ~100%) because 44 L1000 landmarks use obsolete symbols.
**Why it happens:** Visium `.var_names` use current symbols; the MultiDCP gene list uses 2017-era symbols.
**How to avoid:** Normalize symbols (alias→approved) before intersection (`[CITED: CON-gene-space]`).
**Warning signs:** A handful of "missing" genes that are actually present under a renamed symbol.

### Pitfall 5: Rodent "spatial" series with no usable healthy/basal tissue
**What goes wrong:** A picked GEO series is entirely disease/injury with no control spots, so there is no basal context to embed.
**Why it happens:** Most rodent spatial studies are disease-model studies; controls are present but not advertised in the title.
**How to avoid:** Inspect each candidate's sample metadata for sham/control/healthy spots before committing; record which samples are the basal source in MANIFEST.
**Warning signs:** All samples are treated/injured.

### Pitfall 6: License obligations not recorded
**What goes wrong:** Yu liver is GPL-3.0+; some KPMP/atlas data carry attribution or non-commercial terms. DATA-02 requires licenses in MANIFEST.
**Why it happens:** Easy to skip when focused on getting bytes.
**How to avoid:** Record the license string per dataset at download time (`[VERIFIED: figshare API shows Yu = "GPL 3.0+"]`).

## Code Examples

### Verified accession resolution (what to put in the corrected registry)
```text
# Source: [VERIFIED: NCBI GEO acc.cgi + Figshare API + HTTP HEAD, 2026-06-20]
HUMAN SPATIAL (basal input, whole-transcriptome Visium):
  liver   Yu 2022           figshare 22321447  (DOI 10.6084/m9.figshare.22321447.v1, GPL-3.0+,
                            files L5_upload.zip 366MB md5 3abf91538674c3b0cbc3f55b9c5b6074,
                            L18_upload.zip 1.47GB md5 31bbf855afd7e3ab26f65f8786575a0d)   [VERIFIED]
          backup            GSE185477 (Andrews/Teichmann) — GEO resolves, Visium human liver  [VERIFIED]
  kidney  Lake/KPMP 2023    GSE183456 (raw) + GSE183279 (superseries); atlas.kpmp.org HTTP 200 [VERIFIED]
          substitute        GSE211785 (Abedini 2024) — resolves, Visium kidney                 [VERIFIED]
  brain   Maynard 2021 DLPFC  spatialLIBD (Bioconductor HTTP 200; spatial.libd.org HTTP 200)    [VERIFIED]
                            NOT GSE144239 (squamous cell carcinoma)                             [CITED: DOC 09]

VALIDATION (mouse APAP liver Visium, validation-only, never basal):
  GSE280652  (resolves, Visium, mouse, APAP/acetaminophen liver)   [VERIFIED]
  GSE272564  (resolves, Visium, mouse, mid-lobular hepatocyte APAP) [VERIFIED]

RODENT SPATIAL (basal input — candidates to select+verify healthy spots in this phase):
  mouse liver   GPL24247 Visium; e.g. control arms of GSE326636 (HFD model), GSE272564 controls,
                or a dedicated zonation atlas. count≈74 mouse-liver-Visium series on GEO.   [VERIFIED candidates]
  mouse kidney  GPL24247 Visium; e.g. GSE252772 (sex/lifespan kidney atlas, has healthy),
                GSE227045/GSE227046 (injury+repair atlas, has baseline). count≈24.          [VERIFIED candidates]
  mouse brain   GPL24247 Visium; e.g. GSE233983 (aging brain atlas), GSE242214/15 (neurovascular atlas).
                count≈104.                                                                    [VERIFIED candidates]
  rat (any)     GPL25947 rat Visium exists (count≈69) incl. brain/kidney/heart series.        [VERIFIED]
  brain reference  Allen ABC Atlas (mouse) — region reference only, MERFISH panel (annotation-only). [CITED: §6]
```

### MANIFEST.md skeleton (mirrors sibling conventions)
```markdown
# Source: [VERIFIED: dili_downstream/MANIFEST.md + multihead_dili/MANIFEST.md formats]
## Source code (SHA-pinned)
| File/Repo | Source repo | Source SHA | Purpose |
| MultiDCP-CheMoE checkpoint | /raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt | <git SHA of MultiDCP repo> | frozen baseline (Conditions C/F/S-C) |
| MultiDCP-PDG checkpoint    | /raid/home/joshua/projects/MultiDCP/trained_models/<file>.pt    | 871b8de... | frozen baseline (Conditions B/E/S-B) |

## Data files (SHA256-pinned)
| File | Path | SHA256 | Source | License | Whole-transcriptome | Notes |
| Yu liver L5  | data/raw/spatial/yu2022_liver/L5_upload.zip | <sha256> | figshare 22321447 | GPL-3.0+ | Y | Leiden L5 categories |
| ...

## Environment
| dili_v04_env | conda env | `conda env export` snapshot @ <date> + squidpy <ver> |
```

> **Frozen-checkpoint note (for MANIFEST):** The MultiDCP-CheMoE weights are at `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt` and MultiDCP-PDG checkpoints under `/raid/home/joshua/projects/MultiDCP/trained_models/` `[VERIFIED: filesystem]`. Sibling `dili_downstream` pins the MultiDCP repo at SHA `871b8de`. Phase 0 records paths+SHAs in MANIFEST; the actual *loading* of these is Phase 2 (`region_signature.py` seam stays `NotImplementedError` until then, per XC-04).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Yu liver = GSE189994 (datasets.py) | figshare 22321447 (DOI .v1) | corrected this phase | Wrong → correct primary liver basal source |
| DOC 09 says Yu = figshare 17058105 | figshare 22321447 | corrected this research | Even the correction doc was stale; verify live |
| Lake/KPMP = GSE211785 (datasets.py) | GSE183456 + GSE183279 primary; GSE211785 = Abedini substitute | DOC 09 + this phase | Right kidney atlas + a documented fallback |
| Maynard = GSE144239 (datasets.py) | spatialLIBD / LieberInstitute | DOC 09 | GSE144239 is an unrelated SCC study |
| Siletti = MERFISH usable input | snRNA-seq (not spatial); ABC MERFISH is mouse-only | DOC 09 | No public human-brain MERFISH exists |

**Deprecated/outdated:**
- pybiomart-required ortholog fetch: not needed — raw `martservice` REST works with zero new deps.
- tangram-sc as a P0 dependency: out of scope; only relevant if a panel is promoted to input (it is not by decision).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The specific rodent-spatial GEO series to use per organ are a *planning* selection from the verified candidate pools, not pre-fixed | Standard Stack / Code Examples | Low — pools are large and verified to exist; wrong pick is recoverable in planning |
| A2 | KPMP atlas Visium is downloadable without authenticated credentials (portal returns 200; click-through ToS may apply) | Environment Availability | Medium — if it needs a manual click-through/login, that becomes a human-action step (XC-01 "stop and ask") |
| A3 | The MultiDCP 10,716-gene symbol list used for coverage is recoverable from the MultiDCP/PDGrapher repo (e.g. `pdg_de_genes_*` pkl / geneinfo_beta.txt) | Don't Hand-Roll / Pitfall 4 | Medium — exact file must be located+pinned in MANIFEST; coverage % depends on it |
| A4 | Open TG-GATEs and DrugMatrix bulk arrays are downloadable bulk files (not whole-transcriptome Visium); they are cross-species *label/anchor* sources, not basal spatial input, so the whole-transcriptome gate does not apply to them | Environment Availability | Low — they are explicitly bulk toxicogenomics by design |
| A5 | DIRIL / DICTrank / DNT-IVB / Lane-Ekins seizure label sets are obtainable from their publications' supplements | Open Questions | Medium — supplement download mechanics vary; DICTrank (heart) is deferred so lower urgency |
| A6 | A symbol-normalization step recovers the ~44 obsolete L1000 aliases | Pitfall 4 | Low — well-characterized, ~0.5% of landmarks |

## Open Questions (RESOLVED)

1. **Which exact rodent-spatial series per organ?**
   - What we know: whole-transcriptome rodent Visium exists abundantly (mouse GPL24247: liver≈74, kidney≈24, brain≈104; rat GPL25947≈69 series) `[VERIFIED: NCBI GEO]`.
   - What's unclear: which series have clean *healthy/control* spots usable as basal context, and whether to prefer one atlas per organ vs. control arms of disease studies.
   - Recommendation: planner picks one primary healthy series per organ (candidates: liver = a zonation/control series; kidney = GSE252772 lifespan atlas; brain = GSE233983 aging atlas), inspects sample metadata for control spots, and records the chosen basal samples in MANIFEST. Verify whole-transcriptome via coverage gate.
   - **RESOLVED:** Candidate healthy whole-transcriptome series per organ are recorded in the `datasets.py` registry notes (Plan 01 — rodent basal-context Visium entries: mouse liver GSE272564-control arm, mouse kidney GSE252772, mouse brain GSE233983). The final per-organ pick is confirmed at download time by inspecting sample metadata for control spots (Pitfall 5); it is a planning/download decision, **not** a Gate-1 blocker.

2. **KPMP kidney access mechanism.**
   - What we know: atlas.kpmp.org returns 200; GSE183456/GSE183279 are on GEO.
   - What's unclear: whether the analysis-ready Visium objects come cleanest from GEO supplementary or the KPMP portal (portal may require ToS click-through).
   - Recommendation: prefer GEO supplementary (no auth); fall back to KPMP portal; if portal needs login, flag as human-action per XC-01.
   - **RESOLVED:** Primary access = GSE183456 + GSE183279 via GEO supplementary (no auth, `access_mechanism="geo_supp"`/`"kpmp"`). atlas.kpmp.org is the `user_setup`/manual-action click-through ToS fallback gate in Plan 02 (Task 1 `kpmp` dispatch + the Plan 02 `user_setup` entry).

3. **Exact 10,716-gene MultiDCP symbol list file.**
   - What we know: `geneinfo_beta.txt` and `pdg_de_genes_per_cell_top20_40_80.pkl` exist locally `[VERIFIED: filesystem]`.
   - What's unclear: which file is the canonical 10,716 ordering CheMoE outputs.
   - Recommendation: locate+pin the canonical list in MANIFEST during P0 (needed for the coverage report); cross-check count == 10,716.
   - **RESOLVED:** Located/loaded in Plan 04 Task 1 — the executor checks `geneinfo_beta.txt` / `pdg_de_genes_*.pkl` in the MultiDCP/PDGrapher repo, asserts `len(symbols) == 10716` (== `N_PDG`), and records the exact chosen source file path in MANIFEST. The pin is a Plan 04 deliverable, not a Gate-1 blocker.

4. **Brain/kidney/heart human label-set download mechanics.**
   - What we know: SIDER (sideeffects.embl.de) returns 200; DILIst/DILIrank are FDA Excel (sibling MANIFEST has direct URLs+SHAs).
   - What's unclear: DIRIL (Connor 2024 Drug Discov Today supplement), DNT-IVB, Lane-Ekins seizure exact file URLs.
   - Recommendation: resolve each from its paper supplement during P0; DICTrank (heart) is deferred so lowest priority.
   - **PARTIALLY RESOLVED / ACCEPTED-OPEN:** Recorded as an explicit DECISION in Plan 02 Task 2 — **P0 brain toxicity labels = SIDER nervous-system SOC (serious terms) ONLY.** Lane-Ekins seizure and DNT-IVB are DEFERRED to Phase 1 (brain is the last organ sequenced: liver → kidney → brain). DIRIL (kidney) supplement URL is resolved at download time; if unresolved, the driver writes a TODO line into the HALT_REASON candidates note — **never fabricate labels** (XC-01). Liver (DILIst/DILIrank) labels resolve from the sibling MANIFEST direct URLs. DICTrank (heart) is deferred per ROADMAP.

5. **Test file location (`tests/test_data_paths.py` vs `tests/spatial/`).**
   - Recommendation: create flat `tests/test_data_paths.py` per the acceptance-criterion wording; confirm at planning.
   - **RESOLVED:** Flat `tests/test_data_paths.py` (Plan 01 Task 2), per the DATA-01 acceptance wording and the sibling `dili_downstream` convention; the 124 existing `tests/spatial/` tests are left untouched.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| scanpy | Read Visium gene lists | ✓ | 1.11.5 | — |
| anndata | Visium container | ✓ | 0.12.10 | — |
| h5py | 10x/Seurat HDF5 | ✓ | 3.14.0 | — |
| requests | Figshare/BioMart/GEO/KPMP fetch | ✓ | 2.32.3 | — |
| pandas | tables/labels/orthologs | ✓ | 2.2.3 | — |
| squidpy | Moran's I QC (env add in P0 per DATA-02) | ✗ | — | None — must `pip install`; required by P1, env-add belongs in P0 |
| pybiomart | ortholog fetch | ✗ | — | raw `requests` + BioMart XML (verified working) — no install needed |
| pooch | checksum download | ✗ | — | `requests`+`hashlib` |
| GEOparse | GEO suppl listing | ✗ | — | direct GEO HTTPS/FTP URLs |
| Figshare API | Yu liver | ✓ (HTTP 200, article 22321447) | — | — |
| Ensembl BioMart REST | orthologs | ✓ (HTTP 200; ortholog query returns rows) | current release | MGI/RGD flat files |
| NCBI GEO (acc.cgi/E-utilities) | accession verify + downloads | ✓ (200) | — | — |
| spatialLIBD (Bioconductor) | Maynard DLPFC | ✓ (200) | — | LieberInstitute GitHub CSV |
| atlas.kpmp.org | kidney | ✓ (200) | — | GEO GSE183456/GSE183279 |
| Open TG-GATEs (DBArchive) | rat tox anchors | ✓ (200) | — | — |
| NTP DrugMatrix | rat tox anchors | ✓ (200) | — | — |
| SIDER | brain SOC labels | ✓ (200) | — | — |
| MultiDCP-CheMoE checkpoint | MANIFEST record | ✓ (`.../MultiDCP_CheMoE_pdg/src/best_model.pt`) | — | — |
| GPU | none for P0 | n/a | — | P0 is data-only; no torch/GPU use |

**Missing dependencies with no fallback:** `squidpy` (must be installed — but this IS the DATA-02 deliverable, not a blocker).
**Missing dependencies with fallback:** pybiomart→raw REST; pooch→hashlib; GEOparse→direct URLs. None block the phase.

## Validation Architecture

> Nyquist validation is enabled (no `workflow.nyquist_validation: false` in config.json; key absent = enabled).

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 (installed in `dili_v04_env`) |
| Config file | none — markers registered in `spatial_tests/conftest.py` (`network` marker exists) |
| Quick run command | `conda run -n dili_v04_env python -m pytest tests/test_data_paths.py -x -q` |
| Full suite command | `conda run -n dili_v04_env python -m pytest tests/ -q` (preserves the 124 existing `tests/spatial/` tests) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Each planned dataset present on disk under `data/raw/spatial/<dataset>/` with expected files | unit (filesystem) | `pytest tests/test_data_paths.py::test_raw_datasets_present -x` | ❌ Wave 0 |
| DATA-01 | Registry accessions are the CORRECTED ones (no GSE189994/GSE144239; Yu=figshare 22321447) | unit | `pytest tests/test_data_paths.py::test_accessions_corrected -x` | ❌ Wave 0 |
| DATA-01 (Gate 1) | Each Visium basal dataset is whole-transcriptome (coverage > 0.80 of 10,716) | unit | `pytest tests/test_data_paths.py::test_whole_transcriptome_gate -x` | ❌ Wave 0 |
| DATA-02 | `MANIFEST.md` exists and lists path+SHA256+license for every raw dataset and the frozen checkpoints | unit (parse MANIFEST) | `pytest tests/test_data_paths.py::test_manifest_complete -x` | ❌ Wave 0 |
| DATA-02 | squidpy importable in `dili_v04_env`; env snapshot recorded | smoke | `pytest tests/test_data_paths.py::test_squidpy_available -x` | ❌ Wave 0 |
| DATA-03 | `orthology.py` builds a nonempty human-mouse-rat one2one table; dropped fraction reported | unit | `pytest tests/test_data_paths.py::test_ortholog_one2one -x` | ❌ Wave 0 |
| DATA-03 | Per-Visium-dataset coverage against 10,716 space reported in `results/tables/P0_coverage.md` | unit | `pytest tests/test_data_paths.py::test_coverage_report_exists -x` | ❌ Wave 0 |
| DATA-03 | `gene_alignment.coverage_fraction` still green (regression) | unit | `pytest tests/spatial/test_gene_alignment.py -x` | ✅ exists |

### Sampling Rate
- **Per task commit:** `conda run -n dili_v04_env python -m pytest tests/test_data_paths.py -x -q`
- **Per wave merge:** `conda run -n dili_v04_env python -m pytest tests/ -q` (full suite incl. 124 existing)
- **Phase gate:** full suite green AND `results/tables/P0_coverage.md` + `P0_orthologs.md` populated AND MANIFEST complete before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `tests/test_data_paths.py` — covers DATA-01/02/03 (new file; sanity-only per hard rule "tests never on model outputs")
- [ ] Network-gated tests use the existing `@pytest.mark.network` marker so CI/offline runs skip downloads (set `TDC_NETWORK_TESTS=1`-style gate or a new env flag)
- [ ] Fixture: a small cached BioMart TSV (or the chr21 sample) so `test_ortholog_one2one` runs offline and deterministically
- [ ] No framework install needed — pytest 9.0.2 present

## Project Constraints (from CLAUDE.md)

- **Real data only** — no mocking/stubbing/synthetic labels; **stop and ask if a data path is unavailable** (XC-01). Directly relevant: if any Visium/label source fails to resolve at download time, write `HALT_REASON.md` (Halt Gate 1).
- **Atomic commits** — one phase per commit; format `spatial: P0 - <short description>`.
- **Leave one GPU free** — not exercised in P0 (data-only) but honor if any incidental torch import occurs.
- **No fabricated model outputs** — `region_signature.py` `NotImplementedError` seam stays untouched in P0 (XC-04).
- **Extend, don't rebuild** `src/spatial/` (124 passing tests must stay green).
- **Ortholog discipline** — one2one only, many-to-many dropped, dropped fraction reported (XC-08).
- **Time-leakage discipline** — pin dataset + Ensembl-release versions in MANIFEST (XC-10).

## Sources

### Primary (HIGH confidence)
- NCBI GEO acc.cgi + E-utilities (esearch/esummary, db=gds) — verified GSE185477, GSE183456, GSE183279, GSE211785, GSE280652, GSE272564 resolve and carry Visium/organ/species keywords; enumerated mouse liver/kidney/brain and rat Visium candidate pools — 2026-06-20
- Figshare API (`api.figshare.com/v2/articles/22321447`) — Yu 2022 "L5_L18_normalliver", DOI 10.6084/m9.figshare.22321447.v1, GPL-3.0+, file names+sizes+MD5s; confirmed article 17058105 returns 404 — 2026-06-20
- Ensembl BioMart `martservice` REST — live ortholog query (human↔mouse↔rat) returns `orthology_type`; chr21 sample 5665 rows / 180 one2one — 2026-06-20
- HTTP HEAD (200) — spatialLIBD (Bioconductor + spatial.libd.org), atlas.kpmp.org, Open TG-GATEs (DBArchive), NTP DrugMatrix, SIDER, Ensembl BioMart — 2026-06-20
- Local filesystem — MultiDCP-CheMoE checkpoint at `MultiDCP_CheMoE_pdg/src/best_model.pt`; geneinfo_beta.txt, pdg_de_genes pkl; `dili_v04_env` package inventory via importlib.metadata — 2026-06-20
- `dili_downstream/MANIFEST.md` + `multihead_dili/MANIFEST.md` — MANIFEST format conventions (SHA-pinned code/data, env snapshot) — read 2026-06-20

### Secondary (MEDIUM confidence)
- DOC 09 `09_spatial_decisions.md` — accession corrections (Yu/Maynard/Lake/Siletti), data plan, validity plan; note its Figshare DOI for Yu (17058105) was found stale this session
- Design doc `downstream_spatial_xspecies_MDCPMoE_06172026.md` §0/§5/§6/§7/§11 — phase scope, structure, no-leakage, references
- Intel files (`constraints.md`, `context.md`, `SYNTHESIS.md`) — gene-space discipline, ortholog discipline, channel map

### Tertiary (LOW confidence)
- Specific rodent-spatial series picks (GSE252772, GSE233983, etc.) — verified to exist and be Visium, NOT yet verified to contain clean healthy/basal spots; planner must inspect sample metadata

## Metadata

**Confidence breakdown:**
- Accession resolution (human + validation): HIGH — verified live against GEO/Figshare/HTTP
- Yu liver correction (figshare 22321447): HIGH — Figshare API returned the object with files+MD5
- Rodent-spatial selection: MEDIUM — pools verified to exist; per-organ pick + healthy-spot verification deferred to planning
- Tooling/env: HIGH — package inventory and BioMart/Figshare REST verified in-session
- Label-set download mechanics (DIRIL/DNT/seizure): MEDIUM — sources reachable, exact supplement URLs to resolve at planning

**Research date:** 2026-06-20
**Valid until:** 2026-07-20 (accessions are stable; re-verify Figshare/BioMart endpoints if planning slips past this)

## RESEARCH COMPLETE

**Phase:** 0 - Dataset acquisition & MANIFEST
**Confidence:** HIGH (with MEDIUM on final rodent-spatial selection and a few label-set download URLs)

### Key Findings
- **Yu 2022 liver accession is wrong in TWO places and now resolved:** existing `datasets.py` has GSE189994 (unrelated), DOC 09's "correction" figshare 17058105 returns 404, and the *actual* data is Figshare **22321447** (DOI .v1, GPL-3.0+, L5/L18 zips with MD5s). [VERIFIED]
- **All confirmed human/validation accessions resolve and are whole-transcriptome Visium:** GSE185477, GSE183456+GSE183279, GSE211785, GSE280652, GSE272564 — verified at NCBI GEO. [VERIFIED]
- **Existing `datasets.py`/`config.py` still carry the stale accessions DOC 09 corrects** (Yu GSE189994, Maynard GSE144239, Lake/KPMP GSE211785-as-primary, Siletti mislabeled) — must be fixed before any download script (DATA-01 acceptance). [VERIFIED]
- **Rodent whole-transcriptome Visium exists abundantly** (mouse GPL24247: liver≈74/kidney≈24/brain≈104; rat GPL25947≈69); final per-organ healthy-series pick is a planning decision, not a blocker. [VERIFIED candidates]
- **Ortholog method is dependency-free and verified:** Ensembl BioMart `martservice` REST over `requests` returns `orthology_type`, so one2one filtering + dropped-fraction reporting needs no new package. `squidpy` is the only required env addition (DATA-02). [VERIFIED]

### File Created
`/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests/.planning/phases/00-dataset-acquisition-manifest/00-RESEARCH.md`

### Confidence Assessment
| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | env inventory + REST endpoints verified in-session |
| Architecture | HIGH | extends existing pure-library pattern; registry→driver well-fit |
| Pitfalls | HIGH | two real stale-accession bugs found and verified |
| Rodent-spatial selection | MEDIUM | pools verified; per-organ healthy-spot pick deferred to planning |

### Open Questions (RESOLVED — see § Open Questions (RESOLVED) above)
- Final rodent-spatial series per organ — RESOLVED (registry candidates recorded; final pick at download, not a Gate-1 blocker)
- KPMP access mechanism — RESOLVED (GEO supplementary primary; atlas.kpmp.org ToS click-through = user_setup fallback)
- Canonical 10,716-gene symbol-list file to pin — RESOLVED (located/pinned in Plan 04 Task 1)
- Exact supplement URLs for DIRIL / DNT-IVB / Lane-Ekins seizure label sets — PARTIALLY RESOLVED / ACCEPTED-OPEN (P0 brain = SIDER SOC only; seizure/DNT-IVB deferred to P1; DIRIL resolved-or-TODO, never fabricated)
- `tests/test_data_paths.py` flat vs `tests/spatial/` location — RESOLVED (flat `tests/test_data_paths.py`)

### Ready for Planning
Research complete. Planner can create PLAN.md: correct the registry accessions first, build registry→driver downloads, implement `orthology.py` on BioMart REST, install squidpy, write MANIFEST + `tests/test_data_paths.py`, and gate every input on the whole-transcriptome coverage check (Halt Gate 1).
