---
phase: 01-eda-the-bracket
plan: 05
subsystem: spatial-eda
tags: [squidpy, moran, spatial-autocorrelation, mahalanobis, ood, cross-species, orthologs, pearsonr, anndata]

# Dependency graph
requires:
  - phase: 01-eda-the-bracket/01-01
    provides: RED test contract (test_moran_returns_dataframe, test_cross_species_ortholog_filter)
  - phase: 01-eda-the-bracket/01-02
    provides: label loading, smiles_join, eda package structure
provides:
  - compute_moran_svgs: squidpy Moran's I SVG retention (svg_count, svg_fraction, top_svgs)
  - basal_similarity_matrix: pairwise Pearson r across liver-zone pseudobulk profiles
  - svg_retention: fraction of raw-Visium SVGs in the 10,716-gene MultiDCP model space
  - human_mouse_liver_correlation: ortholog-filtered human-rodent Pearson r (one-to-one only, XC-08)
  - ood_mahalanobis: ridge-regularized Mahalanobis OOD distance in 978-gene landmark subspace
affects: [01-06-run-p1-eda, phase-02-model-wiring, phase-04-per-organ-train-test]

# Tech tracking
tech-stack:
  added: [squidpy (lazy import inside function), scipy.spatial.distance.mahalanobis, scipy.stats.pearsonr]
  patterns:
    - squidpy imported inside function body (not at module level) to avoid ImportError where squidpy absent (Pitfall 9)
    - spatial_neighbors_knn + spatial_autocorr(mode='moran', connectivity_key='spatial_connectivities') -- squidpy 1.8.2 canonical API (Pitfall 4)
    - obsm['spatial'] KeyError with available-keys message if coordinate key missing (T-01-08 threat mitigation)
    - 978-gene landmark subspace projection before Mahalanobis inversion (Pitfall 7: rank-deficient full-10716 covariance)
    - one-to-one ortholog filter via ortholog_table.pairs['human_symbol'] -> ['mouse_symbol'] (XC-08)

key-files:
  created:
    - src/spatial/eda/region_diagnostics.py
  modified: []

key-decisions:
  - "OOD method = Mahalanobis with ridge alpha=1e-2 in the 978-gene LINCS landmark subspace (not full 10,716; not kNN). Rationale: n=10 reference points makes kNN poorly conditioned; landmark subspace avoids rank-deficient covariance. Method name baked into OOD_METHOD constant for P1_eda.md reporting."
  - "squidpy imported inside compute_moran_svgs body (try/except ImportError) so region_diagnostics.py imports cleanly in environments without squidpy. Tests use pytest.importorskip to skip cleanly."
  - "human_mouse_liver_correlation raises ValueError (not silent NaN) when < 2 genes match -- ensures the caller sees a diagnostic message rather than getting a degenerate Pearson r."
  - "basal_similarity_matrix accepts both dict[str, ndarray] and ndarray + labels; applies sparse toarray guard from pseudobulk.py pattern."

patterns-established:
  - "Lazy squidpy import: import inside function with try/except ImportError; module-level import avoided."
  - "Threat T-01-08 mitigation: KeyError with available obsm keys when spatial coords absent."
  - "978-gene landmark projection contract: caller responsibility before ood_mahalanobis; documented in function docstring and OOD_METHOD constant."

requirements-completed: [EDA-03]

# Metrics
duration: 3min
completed: 2026-06-23
---

# Phase 1 Plan 05: Region Diagnostics Summary

**Squidpy Moran's I SVG retention, zone-level basal similarity matrix, ortholog-filtered human-rodent Pearson r, and ridge-regularized Mahalanobis OOD in the 978-gene landmark subspace.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-23T03:27:53Z
- **Completed:** 2026-06-23T03:30:49Z
- **Tasks:** 2
- **Files modified:** 1 (created)

## Accomplishments

- `compute_moran_svgs`: builds a k-NN spatial graph with `spatial_neighbors_knn` (n_neighs=6, Visium hex default), runs `spatial_autocorr(mode='moran', connectivity_key='spatial_connectivities')`, reads `adata.uns['moranI']`, returns svg_count / svg_fraction / top_svgs (pval_norm < 0.05 threshold)
- `basal_similarity_matrix`: pairwise Pearson r (or cosine) across zone pseudobulk vectors, handles both dict and ndarray input, applies sparse toarray guard
- `svg_retention`: wraps `gene_alignment.coverage_fraction` to report fraction of raw-Visium SVGs present in the 10,716-gene MultiDCP model space
- `human_mouse_liver_correlation`: one-to-one ortholog filter via `ortholog_table.pairs`, matched gene pairs only, returns pearson_r / p_value / n_genes_compared (confirmed < len(human_genes) in test)
- `ood_mahalanobis`: ridge-regularized covariance inversion (alpha=1e-2) with method name locked to `OOD_METHOD = "Mahalanobis, 978-gene landmark subspace, alpha=1e-2"` for P1_eda.md reporting

## Task Commits

1. **Task 1 + Task 2: Moran's I, basal similarity, ortholog correlation, OOD Mahalanobis** - `365a847` (feat)

**Plan metadata:** (committed below with SUMMARY and STATE updates)

## Files Created/Modified

- `src/spatial/eda/region_diagnostics.py` - five public functions covering EDA-03 region + cross-species diagnostics

## Decisions Made

- OOD method: Mahalanobis with ridge alpha=1e-2 in the 978-gene LINCS landmark subspace (resolves "Claude's Discretion" from CONTEXT.md; Mahalanobis preferred over kNN for n=10 reference manifold per RESEARCH.md Pattern 8).
- squidpy imported lazily inside `compute_moran_svgs` (not at module level) to match Pitfall 9 guidance; tests use `pytest.importorskip("squidpy")`.
- `human_mouse_liver_correlation` raises `ValueError` when < 2 genes match rather than returning NaN silently.
- `basal_similarity_matrix` supports both `dict[str, ndarray]` and `ndarray + labels` inputs for caller flexibility.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes introduced. T-01-08 (missing spatial coords) and T-01-09 (rank-deficient covariance) mitigations are both implemented as specified in the plan threat register.

## Known Stubs

None. All five public functions are fully implemented. The caller is responsible for projecting manifold and query to the 978-gene landmark subspace before calling `ood_mahalanobis` (documented in function docstring and OOD_METHOD constant).

## Next Phase Readiness

- EDA-03 library complete. `run_p1_eda.py` (plan 01-06) can now call `compute_moran_svgs`, `basal_similarity_matrix`, `svg_retention`, `human_mouse_liver_correlation`, and `ood_mahalanobis` directly.
- Caller must: (a) load yu2022 Visium h5 and populate `adata.obsm['spatial']` from `tissue_positions_list.csv` before calling `compute_moran_svgs`; (b) project PDG manifold + Visium pseudobulk to 978 landmark genes before calling `ood_mahalanobis`.
- No blockers.

---

## Self-Check: PASSED

- `src/spatial/eda/region_diagnostics.py` exists on disk: FOUND
- `365a847` commit exists: FOUND (verified via git log)
- `test_moran_returns_dataframe`: PASS (2 passed in 4.22s)
- `test_cross_species_ortholog_filter`: PASS
- `spatial_neighbors_knn` count >= 1: 2 (PASS)
- `spatial_autocorr` count >= 1: 2 (PASS)
- `def basal_similarity_matrix` + `def svg_retention` count >= 2: 2 (PASS)
- `import scanpy` count == 0: 0 (PASS)
- `human_symbol` count >= 1: 4 (PASS)
- `mahalanobis` count >= 1: 11 (PASS)
- `978|landmark` count >= 1: 13 (PASS)
- `pearsonr` count >= 1: 3 (PASS)

---

*Phase: 01-eda-the-bracket*
*Completed: 2026-06-23*
