# `src.spatial` — Region-conditioned signature library

Per-region, basal-conditioned MultiDCP signatures for toxicity prediction. Each spatial
region is treated as a cell context: pseudobulk its healthy basal profile, feed it to a frozen
MultiDCP/CheMoE as the cell input, take the per-region predicted differential expression
(`predicted_DE = predicted_treated − region_basal`), and attention-pool the regions into a
feature for the downstream toxicity classifier.

This module is a sibling experiment to `dili_downstream` (it lives in `spatial_tests/`, not
inside it). It reuses the same frozen upstream checkpoints but keeps the spatial pipeline
isolated.

## Pipeline

1. **Region definition** — anatomically curated regions per organ (`config.REGION_DEFS`):
   liver portal/central zonation, kidney glomerular/tubular/vascular/interstitial niches,
   brain cortical layers + white matter.
2. **Pseudobulk** (`pseudobulk.py`) — aggregate spots/cells annotated to a region into a
   region × gene basal profile.
3. **Gene alignment** (`gene_alignment.py`) — reindex the region profile onto MultiDCP's gene
   space (N_LANDMARK=978 / N_PDG=10716), with a documented missing-gene policy and an
   imputation seam for targeted panels.
4. **Per-region signature** (`region_signature.py`) — frozen-model inference per region;
   DE and manifest logic implemented, the checkpoint-load/inference call left as a
   `NotImplementedError` seam (no fabricated outputs).
5. **Combine** (`region_combiner.py`) — attention-pool over region embeddings; exposes the
   per-region attention weights as a regional-attribution interpretability output. Mean-pool
   and concat baselines included.

## Files

```
src/spatial/
├── __init__.py
├── config.py            # REGION_DEFS, GENE_SPACE constants, AGGREGATIONS
├── datasets.py          # SPATIAL_DATASETS registry (Visium = input; MERFISH/Xenium = annotation)
├── pseudobulk.py        # region aggregation (pure)
├── gene_alignment.py    # reindex to MultiDCP gene space (pure)
├── region_signature.py  # frozen-inference interface (DE/manifest pure; inference seam)
├── region_combiner.py   # attention-pool combiner (torch)
└── README.md
tests/spatial/           # 124 in-memory fixture tests (pure logic only)
data/raw/spatial/        # raw dataset downloads (one subdir per dataset)
data/processed/spatial/  # pseudobulked, gene-aligned region basal profiles
```

## Design and decisions

- `0_project_documents/downstream_tasks/08_spatial_transcriptomics_integration.md` — design.
- `0_project_documents/downstream_tasks/09_spatial_decisions.md` — resolved questions, data
  plan, validity plan, and dataset-accession corrections.
- `0_project_documents/downstream_tasks/spatial_transcriptomics_healthy_datasets.md` — the
  curated dataset list (note: four accessions corrected in `09`).

## Hard rules

Real data only; no fabricated model outputs (the inference path raises rather than fakes).
Tests use small in-memory fixtures, never real data files. Pure libraries carry no hardcoded
absolute paths.

---

*Consolidated and relocated to `spatial_tests/` on 2026-06-17.*
