# Requirements Intel

No PRDs were ingested. The SPEC (precedence 0, source of truth) carries a PRD-like
P0-P7 phase roadmap with deliverables, verification, and halt gates. Each phase
maps to one GSD phase for the downstream roadmapper. Requirements below are derived
from the SPEC phases; supporting detail from the two DOCs is attributed inline.

Primary source for all phase requirements:
/raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md

---

## REQ-P0-dataset-acquisition-manifest
- phase: P0
- source: SPEC §6 Phase 0; dataset-accession corrections from /raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md
- description: Acquire every input dataset (whole-transcriptome Visium for basal context; rodent spatial; APAP drug-perturbed validation; rodent toxicogenomics; human toxicity labels) on disk, versioned, with verified gene-space and ortholog map. No model code. Build MANIFEST.md (paths/versions/SHAs/licenses) and `src/spatial/orthology.py`. Add `squidpy` (and optionally `tangram-sc`) to env and record in MANIFEST.
- corrected accessions (from DOC 09): Yu 2022 liver = Figshare 10.6084/m9.figshare.17058105 (no GEO; GSE189994 is wrong); Lake/KPMP kidney = GSE183456 + GSE183279 (not GSE211785, which is Abedini 2024); Maynard DLPFC brain = spatialLIBD / LieberInstitute (not GSE144239); no public human-brain MERFISH exists (Allen ABC MERFISH is mouse only).
- acceptance: tests/test_data_paths.py green; gene-space coverage per Visium dataset against 10,716-gene space reported; ortholog one-to-one fraction reported.
- halt gate (P0): any planned input dataset unavailable or not whole-transcriptome -> stop and re-plan source.

## REQ-P1-eda-bracket
- phase: P1
- source: SPEC §6 Phase 1
- description: Bound the achievable result before any model runs; check leakage and species structure. Tasks: (1) label entropy / class balance / AUPRC base rates per organ+species; (2) structure-space diagnostics + structure-only floor (logistic + RF); (3) measured-biology ceiling where measured data exists (effective rank, per-gene MI, measured-signature-only baseline); (4) region diagnostics (basal-profile similarity matrix, spatially-variable-gene retention via Moran's I/squidpy, gene coverage); (5) cross-species diagnostics (ortholog overlap, human-vs-rodent basal correlation, OOD distance of healthy tissue from cancer-line manifold).
- acceptance: results/tables/P1_eda.md reporting floor, ceiling, floor-ceiling gap, region distinguishability, human-rodent basal concordance.
- halt gate (P1): floor-ceiling gap near zero for the organ -> reframe before training.

## REQ-P2-wiring-tox-head
- phase: P2
- source: SPEC §6 Phase 2; channel-to-code map from /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (§1-§2); validity steps from DOC 09 Validity plan
- description: End-to-end forward path from drug + region basal to organ-tox logit using the frozen baseline. Tasks: (1) resolve frozen MultiDCP/CheMoE checkpoint paths in MANIFEST; replace region_signature.py seam with real call; (2) cache per-region predicted DE for starting organ (human first), both species; (3) implement tox_head.py concat-MLP, confirm attention combiner feeds it, per-condition zero-tensor masking; (4) smoke-train condition A, confirm wandb logging.
- acceptance: working forward pass; no NaNs; manifests align across regions and species; predicted-vs-measured Pearson on APAP anchor reported.
- halt gate (P2): predicted-vs-measured per-zone Pearson < 0.3 on the acetaminophen anchor -> reframe spatial claim.

## REQ-P3-splits-no-leakage
- phase: P3
- source: SPEC §6 Phase 3 + §7
- description: Compound-aware splits per organ and species plus cross-species transfer split with leakage checks. Tasks: (1) Murcko scaffold split (80/10/10) and Tanimoto-0.4 cluster split per organ+species, no drug crosses partitions; (2) cross-species transfer split holds out target species entirely, no shared-drug label leak across species boundary; (3) leakage audits (scaffold overlap, LINCS Phase II time-leakage, ortholog-pairing consistency); (4) report drug-novel slice size per organ.
- acceptance: split JSONs; results/tables/P3_splits.md with balance, Tanimoto histograms, novel-slice sizes, leakage-audit results.
- halt gate (P3): drug-novel slice < 30 for the organ -> adjust split or labels.

## REQ-P4-per-organ-train-test
- phase: P4
- source: SPEC §6 Phase 4; condition matrix from /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (§4)
- description: Within-species headline per organ, human first then rodent. Train conditions A, B/C, S-B/S-C, G/H, fusion; 3 seeds each. Report AUROC, AUPRC, MCC on scaffold + cluster splits with paired bootstrap CIs (10,000 resamples) for key contrasts: B-minus-A, S-minus-B, fusion-minus-best-single-channel. Channel-ablation baselines (dose-response-only, GEX-only) run every organ; fusion must beat both.
- acceptance: results/tables/P4_headline_{organ}_{species}.md.
- halt gates (P4): B does not beat A by >= 0.02 AUROC; OR S-condition does not beat/contextualize B.

## REQ-P5-cross-species-translatability
- phase: P5
- source: SPEC §6 Phase 5
- description: Measure rodent-to-human transfer and structure-vs-omics comparison. Tasks: (1) train rodent / test human and reverse, per organ; (2) structure-only transfer baseline vs omics-based transfer (ortholog-aligned); (3) translatability metric = transfer AUROC vs within-target-species AUROC, report the gap.
- acceptance: results/tables/P5_translatability_{organ}.md with within-species, transfer, and structure-baseline numbers.
- halt gate (P5): none (poor transfer is a reportable result per publishable concept 4); record the gap honestly.

## REQ-P6-confound-interpretability
- phase: P6
- source: SPEC §6 Phase 6; viability-channel rationale from /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (D2, §6)
- description: Show the result is not generic cytotoxicity and produce the mechanistic story. Tasks: (1) viability ablation (dose-response-only baseline + viability-masked variant per organ); (2) brain only: BBB subset evaluation + BBB-only-proxy ablation (beat by >= 0.05 on permeable compounds); (3) regional attention attribution per organ; (4) GSEA on top attributed genes + routing-permutation diagnostic for CheMoE.
- acceptance: results/tables/P6_* and results/figures/P6_*.
- halt gate (P6): routing-permutation delta AUROC not more negative than a control-layer permutation -> reframe CheMoE claim honestly.

## REQ-P7-robustness-writeup
- phase: P7
- source: SPEC §6 Phase 7 + §9 success criteria
- description: Confirm stability and assemble publication artifacts. Tasks: cluster-split robustness across all conditions; hyperparameter sweep on headline classifier; final figures and tables; results narrative mapped to the four publishable concepts.
- acceptance: results/figures/ publication set; results narrative mapped to the four concepts. Arm ready to write up when (per organ, human then rodent): B beats A and S beats/matches B on held-out slice; signature survives viability ablation; cross-species translatability number exists with structure-vs-omics; regional attention + GSEA give plausible mechanism; cluster-split robustness + HP sweep show stability.
- halt gate (P7): none.

---

## Cross-cutting requirements (apply across all phases)
- source: SPEC §1 (8 hard rules), §7 (no-leakage detail), DOC 06 §6 (cross-organ pitfalls)
- Real data only; stop and ask if a path is unavailable. No mocking/synthetic labels.
- DE rule absolute (spatial variant per DEC-de-rule-spatial-divergence; raw expression forbidden).
- No data leakage: scaffold split default, cluster split harder check, no drug crosses partitions, cross-species holds target species fully out.
- No fabricated model outputs (NotImplementedError seam until checkpoints confirmed).
- CUDA hygiene: set --gpu/CUDA_VISIBLE_DEVICES before import torch; auto-detect free device; always leave one GPU free.
- Seeds: >= 3 per experimental cell; report mean and std.
- Atomic commits: one phase per commit, format `spatial: P{n} - {short description}`.
- Ortholog discipline: one-to-one orthologs only; many-to-many dropped and dropped fraction reported.
- Endpoint conflation guard (DOC 06): carry secondary mechanism/sub-endpoint labels at eval time; never report a single aggregate AUROC without per-mechanism breakdown.
- Time-leakage guard: MultiDCP trained on LINCS Phase II; pin dataset versions in MANIFEST; report pre-release overlap.
