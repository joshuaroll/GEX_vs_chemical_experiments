# PROJECT: Spatial Cross-Species Toxicity Prediction (MultiDCP-CheMoE)

## Identity

- **Project name:** Spatial Cross-Species Toxicity Prediction (MultiDCP-CheMoE)
- **Project root / code home:** `/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests/`
  - Subdirectory of the umbrella git repo `GEX_vs_chemical_experiments`. **It does NOT have its own `.git`** — commits go to the umbrella repo.
  - GitHub: https://github.com/joshuaroll/GEX_vs_chemical_experiments
- **Standalone GSD project.** Unrelated to and isolated from the halted "liver" v0.5 project rooted at `/raid/home/joshua/.planning`. Never read or write that root.
- **Code module:** The spatial code already exists at `src/spatial/` with **124 passing fixture tests**. This plan **EXTENDS** that module; it never rebuilds it.

## Source of Truth

| Document | Role |
|----------|------|
| `/raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md` | SPEC (precedence 0) — full design narrative, P0..P7 phases, halt gates |
| `/raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md` | DOC — resolved decisions + dataset-accession corrections |
| `/raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md` | DOC — channel-to-code map, three-channel architecture |
| `.planning/intel/SYNTHESIS.md` + per-type intel | Ingest synthesis driving this roadmap |

## Core Value

Answer one research question rigorously: **Does a predicted, region-resolved molecular response signature predict organ-specific drug toxicity better than chemical structure alone, and does that signal translate across species well enough that cheaper rodent data can stand in for human?**

Four standalone publishable concepts:
1. Region-conditioned predicted GEX beats cell-line-conditioned predicted GEX for toxicity.
2. Structure-vs-omics cross-species translatability.
3. Rodent-to-human transfer as a proxy (headline best case).
4. Honest negatives as method contributions (OOD failure, cytotoxicity confound, BBB confound — instrumented and reported).

**A rigorously instrumented negative result is an acceptable, publishable outcome.**

## Developer-Facing Success Metric (SPEC §9)

For the first organ (human, then rodent):
- The predicted spatial signature (condition **S**) beats structure-only (**A**) and matches or beats cell-line predicted GEX (**B**) on the held-out drug-novel slice;
- the signature survives the viability ablation;
- a cross-species translatability number exists with a structure-vs-omics comparison;
- regional attention attribution + GSEA give a mechanistically plausible story;
- cluster-split robustness + a hyperparameter sweep show stability.

A rigorously instrumented negative (OOD failure, cytotoxicity/BBB confounds, poor translatability) is an acceptable, publishable outcome.

## Target Runtime / Environment

- Python research pipeline; conda env **`dili_v04_env`**, augmented with **`squidpy`** (required, not currently installed) and optionally **`tangram-sc`**. Record additions in `MANIFEST.md`.
- GPU box is shared: **always leave one GPU free**; set `--gpu` / `CUDA_VISIBLE_DEVICES` **before importing torch**; auto-detect a free device via `nvidia-smi`.

## Experimental Conditions (the experimental axis)

| # | Condition | Signature source |
|---|-----------|------------------|
| A | Structure-only (lower bound) | none (zero tensor) |
| B / C | Cell-line predicted GEX | MultiDCP / MultiDCP-CheMoE on 9 LINCS lines |
| S-B / S-C | Spatial-region predicted GEX | per-region tissue basal context |
| G / H | Dose-response (predicted E-Hill) | frozen / tox-tuned |
| fusion | Three-channel | chem + region GEX + dose-response |

Channel-ablation baselines run every organ (dose-response-only, GEX-only); **fusion must beat both.** Cross-species axis is orthogonal: each condition runs within-human and within-rodent plus both transfer directions; structure-only transfer is the reference isolating the omics contribution to translatability.

Frozen chem encoders: ChemBERTa (768), MolFormer (768), GIN (300), UniMol (512), projected to a common downstream dim.

## Hard Rules (non-negotiable)

1. **Real data only.** No mocking, stubbing, synthetic labels, or placeholder DataLoaders. Stop and ask if a data path is unavailable.
2. **DE rule (spatial variant).** Spatial GEX feature = `predicted_DE_region = predicted_treated(drug, region_basal) - region_basal`. **DOCUMENTED divergence** from the parent project's `treated - diseased` anchor (see Decision DEC-de-rule-spatial-divergence, PROPOSED). Raw expression as a feature/metric is forbidden; top-k selection retained.
3. **No leakage / compound-aware splits.** No drug crosses train/test. Random splits forbidden. Murcko scaffold split is the floor; Tanimoto-0.4 cluster split is the harder OOD check. Cross-species transfer holds the target species fully out.
4. **No fabricated model outputs.** `src/spatial/region_signature.py` enforces a `NotImplementedError` seam until checkpoints are confirmed; replaced with the real call in Phase 2.
5. **CUDA hygiene.** Set `--gpu` / `CUDA_VISIBLE_DEVICES` before `import torch`; auto-detect free device; always leave one GPU free.
6. **Seeds.** Minimum 3 per experimental cell; report mean ± std.
7. **Atomic commits.** One phase per commit. Format: `spatial: P{n} - {short description}`.
8. **Ortholog discipline.** Human/mouse/rat aligned through one-to-one orthologs only (Ensembl / MGI / RGD). Many-to-many dropped (not collapsed); dropped fraction reported.

Additional cross-cutting guards: endpoint-conflation guard (carry secondary mechanism/sub-endpoint labels; never report a single aggregate AUROC without per-mechanism breakdown); time-leakage guard (MultiDCP trained on LINCS Phase II — pin dataset versions in MANIFEST, report pre-release overlap).

## Halt Gates (stop and discuss; write `HALT_REASON.md` into the phase directory; do not bypass)

| # | Phase | Trigger | Action |
|---|-------|---------|--------|
| 1 | P0 | Any planned input dataset unavailable or not whole-transcriptome | Stop and re-plan source |
| 2 | P1 | Floor-ceiling gap near zero for the organ | Reframe before training |
| 3 | P2 | Predicted-vs-acetaminophen (APAP) per-zone Pearson < 0.3 | Reframe spatial claim |
| 4 | P3 | Drug-novel slice < 30 for the organ | Adjust split or labels |
| 5 | P4 | B does not beat A by ≥ 0.02 AUROC; OR S does not beat/contextualize B | Consult before Phase 6+ |
| 6 | P6 | Routing-permutation ΔAUROC not more negative than control-layer permutation | Reframe CheMoE claim honestly |

P5 and P7 have no halt gate (poor transfer is itself a reportable result).

## Key Decisions

### Decided (design)
- **DEC-frozen-baseline** — MultiDCP / MultiDCP-CheMoE (Pham et al. 2022) is a frozen baseline, not retrained. The downstream toxicity head is the only trained component.
- **DEC-spatial-is-comparison-arm (Q1)** — Spatial is an additive comparison arm (S-B / S-C / S-F) benchmarked against cell-line conditions B/C; it does not replace them.
- **DEC-platform-visium-only (Q3/C1)** — Whole-transcriptome Visium as basal input (near-100% coverage of the 10,716-gene MultiDCP space; 934/978 L1000 landmarks present). Targeted panels (MERFISH/Xenium/CosMx) are annotation-only aids; a targeted panel may be promoted only via Tangram reference-based imputation when a paired whole-transcriptome snRNA-seq reference exists.
- **DEC-region-granularity-published (Q4)** — Published region annotations are the default; fall back to unsupervised Leiden domains only when no usable published annotation exists.
- **DEC-frozen-checkpoints-seam** — Inference path raises until checkpoints confirmed; `region_signature.py` `NotImplementedError` seam replaced in Phase 2.
- **DEC-ortholog-discipline** — One-to-one orthologs only; many-to-many dropped; dropped fraction reported.
- **DEC-per-organ-only** — Per-organ training only; no joint multi-organ model. Liver/kidney/brain, **plus heart (ACTIVATED 2026-06-21 per user direction; heart deferral rescinded)**. Heart data acquired + verified: public Kuppe et al. 2022 Visium control sections (Zenodo 6578047, CC BY 4.0) + FDA DICTrank labels. Heart is still sequenced last. Dose-response channel prototyped on liver first, then ported.
- **DEC-doseresponse-first-class-channel (D2)** — Dose-response (E-Hill / viability) is a first-class third channel in all organs, simultaneously a feature AND the cytotoxicity confound control. Fusion must beat both GEX-only and dose-response-only single-channel baselines.
- **DEC-concat-mlp-fixed** — Headline classifier is a fixed 3-layer concat-MLP (GELU, dropout, BN) over projected channels; architecture search out of scope. Capacity held constant across conditions (inactive channel = zero tensor of identical shape).

### PROPOSED (pending professor sign-off — design questions to confirm at phase planning, NOT locked)
- **DEC-de-rule-spatial-divergence (open item 3)** — Spatial DE rule = `predicted_treated(drug, region_basal) - region_basal`, a documented divergence from the parent `treated - diseased` rule. Confirm at P2/P4 planning.
- **DEC-indirect-validity-plus-apap (open item 2)** — Accept indirect validation (per-region predicted DE as a learned feature) as the primary validity path, with one direct empirical anchor: an APAP single-drug Visium benchmark (GSE280652 / GSE272564). No panel-wide drug-perturbed spatial dataset exists as of mid-2026. Confirm acceptability of the APAP stand-in.
- **DEC-negative-result-acceptable (open item 1)** — A negative/limitation result (predicted GEX degrades on OOD healthy-tissue contexts) is an acceptable, publishable outcome. Confirm whether negative = stop-and-reframe vs stop-and-abandon.

(Two further sign-offs are operational, not design-locking: approve adding `squidpy`/Tangram to the env; sign off the dataset-accession corrections before download scripts are written — both handled in P0 planning.)

## Out of Scope

- Retraining / fine-tuning MultiDCP / CheMoE (frozen baseline only).
- Joint multi-organ model (per-organ only).
- Heart deferral — RESCINDED 2026-06-21. Heart is now the 4th in-scope organ (public Kuppe Visium control + FDA DICTrank acquired); still sequenced last, after liver/kidney/brain.
- Non-DE metrics / raw-expression features.

## Constraints (implementation contracts)

- **CON-model-io** — Frozen MultiDCP I/O: SMILES→graph→128-d NeuralFingerprint (species-invariant); basal expression (978 or 10,716-d)→encoder→50-d cell context; 6-way one-hot dose→embedding; GEX head over 10,716 genes (PDG/CheMoE); dose-response head = scalar `ehill`; CheMoE GatingNetwork (306→128→4), top-2 of 4 generic experts (specialization emergent, routing-permutation diagnostic is the only check).
- **CON-tox-head** — chem_emb→proj_chem; region_pooled_DE (attention pool over regions, top-k of 10,716)→proj_gex; dose_response→proj_dr; concat→3-layer MLP→organ-tox logit. Region pooling = attention combiner in `src/spatial/region_combiner.py`, emits per-region attention weights as interpretability output. Inactive channel = zero tensor of identical shape.
- **CON-gene-space** — 10,716-gene MultiDCP space; Visium near-100% coverage, 934/978 L1000 landmarks; `fill_value=0.0` documented assumption (~30% of 10,716 may be absent → OOD factor). Do not conflate 978 / 12,328 / 10,716.
- **CON-ortholog-alignment** — Human-mouse-rat one-to-one orthologs (Ensembl BioMart / MGI / RGD), in `src/spatial/orthology.py` (new, P0) and `src/spatial/gene_alignment.py` (extend).
- **CON-splits** — Within species: Murcko scaffold (floor) + Tanimoto-0.4 cluster (harder OOD); no drug crosses partitions; class balance within 5 points of global rate. Across species: target species fully held out; shared drug keeps same partition role. Every split ships a Tanimoto train-vs-test histogram and a drug-novel-slice count.
- **CON-spatial-qc** — Confirm spatially variable genes survive pseudobulking (Moran's I via squidpy).
- **CON-code-module-layout** — Extend `src/spatial/` (do not rebuild). Existing: config.py, datasets.py [extend], pseudobulk.py, gene_alignment.py [extend], region_signature.py [seam], region_combiner.py. New: orthology.py [P0], tox_head.py [P2], splits.py [P3], train.py/eval.py [P4]. Data under `data/raw/spatial/` and `data/processed/spatial/`; configs/ per-organ YAML; results/{tables,figures}/; MANIFEST.md [P0].

---

*Created from ingest synthesis on 2026-06-20. Standalone GSD project; isolated from `/raid/home/joshua/.planning`.*
