# REQUIREMENTS: Spatial Cross-Species Toxicity Prediction (MultiDCP-CheMoE)

Each SPEC phase P0..P7 maps one-to-one to a GSD phase. Primary source:
`/raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md` (SPEC §6).
Supporting detail attributed inline. v1 = all 8 phase requirements (this milestone).

---

## DATA — Dataset acquisition & MANIFEST (Phase P0)

### DATA-01 — Acquire all input datasets, versioned, on disk
- **Phase:** P0
- **Source:** SPEC §6 Phase 0; accession corrections from DOC 09.
- Acquire every input dataset on disk, versioned: whole-transcriptome Visium for basal context; rodent spatial; APAP drug-perturbed validation (GSE280652 / GSE272564); rodent toxicogenomics; human/per-organ toxicity labels (DILIst/DILIrank liver; DIRIL kidney; DICTrank heart deferred; SIDER/Lane-Ekins/DNT-IVB brain). No model code in this phase.
- **Corrected accessions (DOC 09):** Yu 2022 liver = Figshare 10.6084/m9.figshare.17058105 (GSE189994 is wrong); Lake/KPMP kidney = GSE183456 + GSE183279 (not GSE211785 = Abedini 2024 substitute); Maynard DLPFC = spatialLIBD / LieberInstitute (not GSE144239); no public human-brain MERFISH exists (Allen ABC MERFISH is mouse only).
- **Acceptance:** all planned datasets on disk and versioned; accession corrections applied before download scripts written.

### DATA-02 — MANIFEST.md and environment record
- **Phase:** P0
- Build `MANIFEST.md` (paths / versions / SHAs / licenses for every dataset and the frozen MultiDCP/CheMoE checkpoints). Add `squidpy` (required) and optionally `tangram-sc` to `dili_v04_env` and record additions in MANIFEST.

### DATA-03 — Gene-space coverage + ortholog map
- **Phase:** P0
- Implement `src/spatial/orthology.py` (new): human-mouse-rat one-to-one orthologs (Ensembl BioMart / MGI / RGD), many-to-many dropped, dropped fraction reported. Report per-Visium-dataset gene-space coverage against the 10,716-gene MultiDCP space.
- **Acceptance:** `tests/test_data_paths.py` green; gene-space coverage per Visium dataset reported; ortholog one-to-one fraction reported.

> **HALT GATE 1 (P0):** any planned input dataset unavailable or not whole-transcriptome → write `HALT_REASON.md`, stop and re-plan source.

---

## EDA — Exploratory bracket (Phase P1)

### EDA-01 — Label and structure-space brackets
- **Phase:** P1
- **Source:** SPEC §6 Phase 1.
- (1) Label entropy / class balance / AUPRC base rates per organ+species. (2) Structure-space diagnostics + structure-only floor (logistic + RF).

### EDA-02 — Measured-biology ceiling
- **Phase:** P1
- Where measured data exists: effective rank, per-gene mutual information, measured-signature-only baseline (the ceiling).

### EDA-03 — Region + cross-species diagnostics
- **Phase:** P1
- Region diagnostics: basal-profile similarity matrix, spatially-variable-gene retention (Moran's I via squidpy), gene coverage. Cross-species diagnostics: ortholog overlap, human-vs-rodent basal correlation, OOD distance of healthy tissue from the cancer-line manifold.
- **Acceptance:** `results/tables/P1_eda.md` reporting floor, ceiling, floor-ceiling gap, region distinguishability, human-rodent basal concordance.

> **HALT GATE 2 (P1):** floor-ceiling gap near zero for the organ → write `HALT_REASON.md`, reframe before training.

---

## WIRE — MultiDCP wiring & toxicity head (Phase P2)

### WIRE-01 — Frozen-model forward path (replace seam)
- **Phase:** P2
- **Source:** SPEC §6 Phase 2; channel map DOC 06 §1-§2.
- Resolve frozen MultiDCP/CheMoE checkpoint paths in MANIFEST; replace the `region_signature.py` `NotImplementedError` seam with the real call. Cache per-region predicted DE for the starting organ (human first), both species. Spatial DE rule: `predicted_treated(drug, region_basal) - region_basal` (PROPOSED divergence — confirm at planning).

### WIRE-02 — Toxicity head + smoke train
- **Phase:** P2
- Implement `src/spatial/tox_head.py` concat-MLP; confirm the attention combiner feeds it; per-condition zero-tensor masking. Smoke-train condition A; confirm wandb logging.

### WIRE-03 — APAP validity anchor
- **Phase:** P2
- Report predicted-vs-measured per-zone Pearson on the APAP single-drug anchor (GSE280652 / GSE272564).
- **Acceptance:** working forward pass; no NaNs; manifests align across regions and species; APAP per-zone Pearson reported.

> **HALT GATE 3 (P2):** predicted-vs-acetaminophen per-zone Pearson < 0.3 → write `HALT_REASON.md`, reframe spatial claim.

---

## SPLIT — Compound-aware splits & no leakage (Phase P3)

### SPLIT-01 — Within-species compound-aware splits
- **Phase:** P3
- **Source:** SPEC §6 Phase 3 + §7.
- Implement `src/spatial/splits.py`: Murcko scaffold split (80/10/10) and Tanimoto-0.4 cluster split per organ+species; no drug crosses partitions; class balance within 5 points of global rate.

### SPLIT-02 — Cross-species transfer split
- **Phase:** P3
- Transfer split holds the target species entirely out; a drug in both species keeps the same partition role (no shared-drug label leak across the species boundary).

### SPLIT-03 — Leakage audits + novel-slice sizing
- **Phase:** P3
- Leakage audits: scaffold overlap, LINCS Phase II time-leakage, ortholog-pairing consistency. Report drug-novel slice size per organ; Tanimoto train-vs-test histograms; no silent truncation.
- **Acceptance:** split JSONs; `results/tables/P3_splits.md` with balance, Tanimoto histograms, novel-slice sizes, leakage-audit results.

> **HALT GATE 4 (P3):** drug-novel slice < 30 for the organ → write `HALT_REASON.md`, adjust split or labels.

---

## TRAIN — Per-organ train/test, human first (Phase P4)

### TRAIN-01 — Headline condition matrix, ≥3 seeds
- **Phase:** P4
- **Source:** SPEC §6 Phase 4; condition matrix DOC 06 §4.
- Implement `src/spatial/train.py` / `eval.py`. Train conditions A, B/C, S-B/S-C, G/H, fusion; ≥3 seeds each; human first then rodent.

### TRAIN-02 — Metrics, contrasts, channel ablations
- **Phase:** P4
- Report AUROC, AUPRC, MCC on scaffold + cluster splits with paired bootstrap CIs (10,000 resamples) for key contrasts: B−A, S−B, fusion−best-single-channel. Channel-ablation baselines (dose-response-only, GEX-only) run every organ; fusion must beat both. Per-mechanism breakdown (endpoint-conflation guard).
- **Acceptance:** `results/tables/P4_headline_{organ}_{species}.md`.

> **HALT GATE 5 (P4):** B does not beat A by ≥ 0.02 AUROC; OR S does not beat/contextualize B → write `HALT_REASON.md`, consult before Phase 6+.

---

## XSPEC — Cross-species translatability (Phase P5)

### XSPEC-01 — Transfer experiments + structure-vs-omics
- **Phase:** P5
- **Source:** SPEC §6 Phase 5.
- Train rodent / test human and reverse, per organ. Structure-only transfer baseline vs omics-based transfer (ortholog-aligned). Translatability metric = transfer AUROC vs within-target-species AUROC; report the gap honestly.
- **Acceptance:** `results/tables/P5_translatability_{organ}.md` with within-species, transfer, and structure-baseline numbers.

> **No halt gate (P5):** poor transfer is a reportable result (publishable concept 3/4); record the gap honestly.

---

## CONF — Confound control & interpretability (Phase P6)

### CONF-01 — Cytotoxicity / viability confound control
- **Phase:** P6
- **Source:** SPEC §6 Phase 6; viability rationale DOC 06 D2/§6.
- Viability ablation: dose-response-only baseline + viability-masked variant per organ. Brain only: BBB subset evaluation + BBB-only-proxy ablation (must beat by ≥ 0.05 AUROC on BBB-permeable compounds).

### CONF-02 — Interpretability + routing diagnostic
- **Phase:** P6
- Regional attention attribution per organ; GSEA on top attributed genes; routing-permutation diagnostic for CheMoE.
- **Acceptance:** `results/tables/P6_*` and `results/figures/P6_*`.

> **HALT GATE 6 (P6):** routing-permutation ΔAUROC not more negative than a control-layer permutation → write `HALT_REASON.md`, reframe CheMoE claim honestly.

---

## ROBUST — Robustness & writeup (Phase P7)

### ROBUST-01 — Stability checks
- **Phase:** P7
- **Source:** SPEC §6 Phase 7 + §9.
- Cluster-split robustness across all conditions; hyperparameter sweep on the headline classifier.

### ROBUST-02 — Publication artifacts
- **Phase:** P7
- Final figures and tables; results narrative mapped to the four publishable concepts.
- **Acceptance:** `results/figures/` publication set; results narrative mapped to the four concepts. Arm ready to write up when (per organ, human then rodent): B beats A and S beats/matches B on the held-out slice; signature survives viability ablation; cross-species translatability number exists with structure-vs-omics; regional attention + GSEA give a plausible mechanism; cluster-split robustness + HP sweep show stability.

> **No halt gate (P7).**

---

## Cross-Cutting Requirements (apply across all phases)

- **XC-01 Real data only** — no mocking/stubbing/synthetic labels; stop and ask if a path is unavailable.
- **XC-02 DE rule** — spatial variant `predicted_treated(drug, region_basal) - region_basal` (PROPOSED); raw expression forbidden; top-k retained.
- **XC-03 No leakage** — scaffold split default, cluster split harder check, no drug crosses partitions, cross-species holds target species fully out.
- **XC-04 No fabricated outputs** — `NotImplementedError` seam in `region_signature.py` until checkpoints confirmed.
- **XC-05 CUDA hygiene** — set `--gpu`/`CUDA_VISIBLE_DEVICES` before `import torch`; auto-detect free device; always leave one GPU free.
- **XC-06 Seeds** — ≥3 per experimental cell; report mean ± std.
- **XC-07 Atomic commits** — one phase per commit; `spatial: P{n} - {short description}`.
- **XC-08 Ortholog discipline** — one-to-one only; many-to-many dropped; dropped fraction reported.
- **XC-09 Endpoint-conflation guard** — carry secondary mechanism/sub-endpoint labels at eval time; never report a single aggregate AUROC without per-mechanism breakdown.
- **XC-10 Time-leakage guard** — MultiDCP trained on LINCS Phase II; pin dataset versions in MANIFEST; report pre-release overlap.

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | P0 | Pending |
| DATA-02 | P0 | Pending |
| DATA-03 | P0 | Pending |
| EDA-01 | P1 | Pending |
| EDA-02 | P1 | Pending |
| EDA-03 | P1 | Pending |
| WIRE-01 | P2 | Pending |
| WIRE-02 | P2 | Pending |
| WIRE-03 | P2 | Pending |
| SPLIT-01 | P3 | Pending |
| SPLIT-02 | P3 | Pending |
| SPLIT-03 | P3 | Pending |
| TRAIN-01 | P4 | Pending |
| TRAIN-02 | P4 | Pending |
| XSPEC-01 | P5 | Pending |
| CONF-01 | P6 | Pending |
| CONF-02 | P6 | Pending |
| ROBUST-01 | P7 | Pending |
| ROBUST-02 | P7 | Pending |

**Coverage:** 19/19 requirements mapped across 8 phases. No orphans.
