# Phase 1: EDA (the bracket) - Context

**Gathered:** 2026-06-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Bound the achievable result **before any model runs**: establish the structure-only
**floor**, the measured-biology **ceiling**, and the gap between them, plus region
distinguishability and human↔rodent basal concordance. Deliverable:
`results/tables/P1_eda.md`. Triggers **Halt Gate 2** when the floor-ceiling gap is
not significantly positive. **No model code, no training, no MultiDCP forward pass**
(that is Phase 2).

Requirements in scope: **EDA-01** (label + structure brackets, structure-only floor),
**EDA-02** (measured-biology ceiling), **EDA-03** (region + cross-species diagnostics).

**This pass is liver-human scoped** (see D-01). The per-organ halt gate gates liver
only here.

</domain>

<decisions>
## Implementation Decisions

### Scope (which organ/species this pass brackets)
- **D-01:** Run the full bracket for **liver, human only** this pass. The other
  organs (kidney, brain, heart — heart last) and rodent labels/structure-floor are
  deferred to their own training phases, matching the SPEC's per-organ halt gate
  ("the organ being started"). **Exception — cross-species basal diagnostics stay
  in for liver:** EDA-03 still computes human-vs-rodent *liver* basal correlation
  (ortholog-mapped, using the `GSE272564` control arm registered in P0) and the
  OOD distance of healthy liver tissue from the cancer-line manifold. Those need no
  rodent labels, so they belong in this pass. What is deferred is the rodent
  *toxicogenomics ceiling*, rodent *structure floor/labels*, and all non-liver organs.

### Halt Gate 2 operationalization + negative-result stance
- **D-02:** Gap metric is **AUROC**. The gate fires when the **95% paired bootstrap CI
  of (measured-ceiling AUROC − structure-floor AUROC) includes 0** — i.e. the gap is
  not significantly greater than zero. A fired gate means **stop-and-REFRAME, not
  abandon**: write `HALT_REASON.md` into the phase dir and reframe before training.
  **This confirms and LOCKS the ROADMAP "Proposed Decision #3" (negative result is
  publishable)** for the P1 gate. Default 10,000 resamples (matches the P4 convention).

### Structure-only floor (EDA-01)
- **D-03:** Floor = **logistic regression + random forest on ECFP4/Morgan
  fingerprints** (2048-bit, radius 2). Report per organ+species: label entropy, class
  balance, AUPRC base rate, and the logistic/RF floor AUROC. SMILES are joined by
  **reusing the sibling v0.5 resolved tables** (`dili_canonical.csv`,
  `dilist_smiles_resolved.csv`, `drugbank_smiles_index.csv`) first; fall back to
  TDC DILI / PubChem name lookup only for residual gaps. No re-resolution from scratch.

### Liver label set (EDA-01)
- **D-04:** **DILIrank is the primary** liver label for the bracket (FDA vDILIConcern
  → binary, Wang/Li convention). **DILIst is the secondary** expanded-coverage
  cross-check; report its base rates + floor alongside where cheap. DILIrank gates.

### Measured-biology ceiling (EDA-02)
- **D-05:** The human liver ceiling **reuses the v0.5 Wang/Li LINCS measured DE**
  (`dili_downstream/data/processed/wangli_measured_de.npy` +
  `wangli_profiles.csv`). Ceiling diagnostics: effective rank (PCA participation
  ratio), per-gene mutual information vs label, and a measured-signature-only baseline
  AUROC. **Provenance discipline:** record these as external inputs in this project's
  `MANIFEST.md` with SHA + license/source note — do not treat as a silent live
  cross-project read. Rodent toxicogenomics ceiling (Open TG-GATEs / DrugMatrix) is
  deferred to the rodent pass; any organ/species with no measured data is reported as
  **"no measured ceiling — floor only"**, not silently skipped.

### Claude's Discretion
- OOD-distance method: Mahalanobis vs kNN (SPEC §Phase 1 task 5 lists both — planner/
  researcher picks; report which).
- Exact bootstrap resample count if 10,000 is too slow (floor 2,000).
- Morgan fingerprint bit length / radius if 2048/r2 underperforms.
- Which annotation field in the liver basal `.h5ad` provides the published region
  labels (published-annotations-default is locked; researcher resolves the field).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase spec + requirements (source of truth)
- `/raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md` §"Phase 1 — Exploratory data analysis (the bracket)" (lines ~164-177) — the 5 EDA tasks, deliverable, and halt gate.
- `.planning/ROADMAP.md` §"Phase 1: EDA (the bracket)" + §"Proposed Decisions to Confirm at Phase Planning" — success criteria and the negative-result decision (now locked, see D-02).
- `.planning/REQUIREMENTS.md` — EDA-01, EDA-02, EDA-03 (definitions + acceptance).
- `.planning/STATE.md` — locked design decisions + the isolation note (NEVER read/write `/raid/home/joshua/.planning`).

### Phase 0 outputs to build on
- `results/tables/P0_coverage.md` — per-Visium gene-space coverage vs the 10,716-gene space.
- `results/tables/P0_orthologs.md` — human-mouse-rat one-to-one ortholog map + dropped fraction.
- `MANIFEST.md` — dataset paths/SHAs/licenses; extend it with the reused LINCS ceiling provenance (D-05).

### Reused sibling (v0.5 dili_downstream) artifacts — external inputs, log in MANIFEST
- `../dili_downstream/data/processed/wangli_measured_de.npy` — measured LINCS L1000 DE (the liver ceiling input).
- `../dili_downstream/data/processed/wangli_profiles.csv` — profile metadata for the above.
- `../dili_downstream/src/data/wangli_lincs_lookup.py` — reference loader for the measured DE.
- `../dili_downstream/data/processed/dili_canonical.csv`, `dilist_smiles_resolved.csv`, `drugbank_smiles_index.csv` — already-resolved SMILES for the structure join (D-03).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets (this repo, `src/spatial/`)
- `orthology.py` — human↔mouse↔rat one-to-one ortholog map (P0). Feeds EDA-03 cross-species basal correlation.
- `pseudobulk.py` — region pseudobulk; precedes Moran's I spatially-variable-gene retention (squidpy 1.8.2 installed).
- `region_combiner.py` — region handling; supports the basal-profile similarity matrix.
- `gene_alignment.py` — align Visium genes to the 10,716-gene space (the gene space the model sees).
- `datasets.py` — registry; liver basal input = `yu2022_liver`, rodent liver basal = `GSE272564` control arm, labels = `dilirank`/`dilist`/`diril`/`dictrank`.

### Established Patterns
- Real-data-only / provenance-before-progress: any reused or acquired data is logged in `MANIFEST.md` with SHA before use. `tests/test_data_paths.py` gates presence.
- DE rule: GEX features on differential expression; raw expression as feature/metric forbidden. (Ceiling uses measured DE, already DE.)

### Integration Points
- New EDA code writes to `results/tables/P1_eda.md` (+ any figures under `results/figures/`).
- SMILES join and measured-DE load cross the project boundary into `../dili_downstream/` — read as static inputs, copy/record into local MANIFEST; do NOT touch `/raid/home/joshua/.planning`.

</code_context>

<specifics>
## Specific Ideas

- Floor-ceiling gap reported per organ+species with the bootstrap CI explicit (D-02), so the halt-gate decision is reproducible from the table, not a judgment call.
- DILIrank gates; DILIst shown as a coverage sensitivity check (D-04).
- Honest reporting: organs/species lacking measured data are labeled "no measured ceiling — floor only", never silently dropped.

</specifics>

<deferred>
## Deferred Ideas

- **Kidney / brain / heart brackets** — same EDA machinery, run at each organ's training phase (heart sequenced last). Not scope creep; sequencing.
- **Rodent structure-floor + rodent labels** — deferred to the rodent pass.
- **Rodent toxicogenomics ceiling** (Open TG-GATEs / DrugMatrix) — acquire + bracket when the rodent arm starts.

None — discussion stayed within phase scope (deferrals above are sequencing, not new capabilities).

</deferred>

---

*Phase: 1-EDA (the bracket)*
*Context gathered: 2026-06-22*
