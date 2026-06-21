# spatial_tests

Spatial-transcriptomics comparison arm for the MultiDCP-CheMoE toxicity project. Sibling to
`dili_downstream` (not nested inside it). Isolated here so the spatial experiment can move at
its own pace without touching the core liver/kidney/brain pipeline.

## What this is

The core project predicts drug-induced gene expression by conditioning MultiDCP on a **cell
line's** basal expression profile. This arm asks whether conditioning on a **spatial region's**
basal profile instead — a hepatic zone, a kidney niche, a cortical layer — produces signatures
that predict toxicity better, and whether localizing the predicted insult to a tissue
compartment adds a mechanistic, interpretable signal the cell-line approach cannot give.

It is a **comparison arm**, not a replacement: region-conditioned signatures (S-B/S-C/S-F) are
benchmarked against the cell-line conditions (B/C) from `dili_downstream`. It is gated behind
the liver positive control — nothing here is the headline until the core method is shown to
work.

## Layout

```
spatial_tests/
├── conftest.py                  # pytest rootdir anchor (so `from src.spatial...` resolves)
├── src/
│   ├── __init__.py
│   └── spatial/                 # the module — see src/spatial/README.md
├── tests/spatial/               # 124 in-memory fixture tests
└── data/
    ├── raw/spatial/             # raw dataset downloads, one subdir per dataset
    └── processed/spatial/       # pseudobulked, gene-aligned region basal profiles
```

## Run the tests

```bash
cd spatial_tests
conda run -n dili_v04_env python -m pytest tests/spatial/ -q
```

## Data plan (summary — full version in 09_spatial_decisions.md)

Whole-transcriptome **Visium** datasets are the basal input. Targeted panels (MERFISH/Xenium)
are annotation-only or need imputation. Inputs: Lake/KPMP (kidney, GSE183456/279), Yu 2022 or
Andrews/Teichmann (liver), Maynard DLPFC via spatialLIBD (brain). Validation anchor: APAP
liver Visium (GSE280652/GSE272564) — the only drug-perturbed spatial data, and it is mouse.

## Relationship to the core project

- Frozen MultiDCP/CheMoE checkpoints are shared with `dili_downstream` (this arm does not
  retrain the upstream model).
- Splits, classifier conventions, and halt-gate logic mirror the core design.
- GSD source-of-truth design: `0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md` (start here — phases, baseline, cross-species plan, no-leakage discipline).
- Design and decisions live in `0_project_documents/downstream_tasks/` (`08`, `09`).

## Status

Scaffold complete, 124 tests passing. Inference path is a documented seam pending confirmed
checkpoint paths. Blocked from execution behind the core liver Halt Gate 1 and five open items
for the professor (see `09_spatial_decisions.md`).
