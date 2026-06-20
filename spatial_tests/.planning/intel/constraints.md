# Constraints Intel

The SPEC is classified SPEC (precedence 0) and carries technical contracts
(model I/O dims, channel-to-projection map, code seams, ortholog discipline,
halt-gate thresholds). DOC 06 supplies the verified channel-to-code map and
output dimensionalities. Constraints below are extracted as implementation
contracts.

---

## CON-model-io — frozen MultiDCP I/O contract
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§3); confirmed in /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (§1, §8)
- type: api-contract
- content:
  - Drug input: SMILES -> molecular graph (atom 62, bond 6) -> differentiable NeuralFingerprint (128-d). Species-invariant. Not Morgan, not raw SMILES.
  - Cell context input: basal untreated expression vector (978 or 10,716-d) -> Linear/Transformer encoder -> 50-d. This is the generalization mechanism; any context with a basal profile can be embedded (spatial region pseudobulk, rodent tissue).
  - Dose input: 6-way one-hot -> embedding (linear baseline / 16->128 MLP for CheMoE).
  - GEX head: predicted treated expression over 10,716 genes (PDG/CheMoE variant); DE computed downstream. Original MultiDCP = 978 landmark.
  - Dose-response head: scalar `ehill` per (drug, cell, dose). EC50/Einf/Hill-slope are OFFLINE labels, not emitted. No separate viability head.
  - CheMoE: GatingNetwork (306->128->4), top-2 of 4 generic MLP experts over concat drug(128)+cell(50)+dose(128); specialization emergent (load-balancing loss only), NOT scaffold/assay-partitioned. "StructuredMoE" label overstates; routing-permutation diagnostic is the only check.

## CON-tox-head — downstream concat-MLP channel-to-projection map
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§3); /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (§2)
- type: schema
- content:
  - chem_emb (768/768/300/512 by encoder: ChemBERTa/MolFormer/GIN/UniMol) -> proj_chem
  - region_pooled_DE (attention-pool over regions, top-k of 10,716) -> proj_gex
  - dose_response (predicted ehill summary, potency/efficacy) -> proj_dr
  - concat -> 3-layer MLP (GELU, dropout, BN) -> organ-tox logit
  - Region pooling = attention combiner in src/spatial/region_combiner.py; emits per-region attention weights as interpretability output.
  - Capacity held constant: inactive channel = zero tensor of identical shape; only information content varies across conditions A..H/fusion.

## CON-gene-space — 10,716-gene MultiDCP space, coverage discipline
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md (C1); /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (§8)
- type: schema
- content:
  - Visium pseudobulk per region covers ~20k-33k genes; subset to 10,716-gene MultiDCP space = near-100% coverage. 934/978 (95.5%) L1000 landmarks present; 44 missing are obsolete HGNC aliases recoverable by one-time symbol normalization at pseudobulk time.
  - Targeted panels NOT usable as direct input: MERFISH ~500 (~4.7% coverage), Xenium ~313, CosMx standard ~960 — zero-padding ~95% of a full-rank-trained LinearEncoder input is severe OOD. Annotation-only.
  - fill_value=0.0 for missing genes is a documented modeling assumption; ~30% of the 10,716-gene space may be absent from Visium coverage (additional OOD factor). Mean-imputation from a hepatocyte atlas is an alternative fill strategy to consider.
  - Gene-count discipline: 978 landmark; 12,328 = full GSE70138 Level-5; 10,716 = PDGrapher subset (what CheMoE outputs). Do not conflate.

## CON-ortholog-alignment — cross-species ortholog tables
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§1.8, §5, §6 Phase 0)
- type: schema
- content: Human-mouse-rat one-to-one orthologs from Ensembl BioMart / MGI / RGD. Many-to-many dropped (not collapsed); dropped fraction reported. Implemented in src/spatial/orthology.py (new, Phase 0) and src/spatial/gene_alignment.py (extend).

## CON-splits — compound-aware split protocol (NFR)
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§7); /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (§6 pitfall 3)
- type: nfr
- content: Within species: Murcko scaffold split (floor), Tanimoto-0.4 cluster split (harder OOD check); no drug crosses partitions; class balance within 5 points of global rate. Across species: transfer direction holds target species fully out; a drug in both species keeps the same partition role. Every split ships a Tanimoto train-vs-test histogram and a drug-novel slice count; silent truncation not allowed. Scaffold-vs-random inflation ~0.03-0.15 (worse for small chemotype-clustered DIKI/DINT sets).

## CON-spatial-qc — spatially-variable-gene retention (NFR)
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md (Validity plan step 6, C3)
- type: nfr
- content: Confirm spatially variable genes survive pseudobulking (Moran's I via squidpy) so regions are not collapsed into near-identical basal vectors. squidpy is not currently installed and must be added to dili_v04_env.

## CON-halt-gate-thresholds — quantitative gates
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§8)
- type: nfr
- content:
  - P0: input dataset unavailable or not whole-transcriptome -> re-plan source.
  - P1: floor-ceiling gap near zero for the organ -> reframe before training.
  - P2: predicted-vs-acetaminophen per-zone Pearson < 0.3 -> reframe spatial claim.
  - P3: drug-novel slice < 30 -> adjust split or labels.
  - P4: B does not beat A by >= 0.02 AUROC; OR S does not beat B -> consult.
  - P6: routing-permutation delta AUROC not below control-layer permutation -> reframe CheMoE claim.
  - On any gate firing: write HALT_REASON.md into the phase directory; do not bypass.

## CON-code-module-layout — src/spatial/ extension contract
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§5)
- type: api-contract
- content: Extend existing src/spatial/ (124 passing fixture tests), do not rebuild. Existing: config.py, datasets.py [extend], pseudobulk.py, gene_alignment.py [extend orthologs], region_signature.py [seam], region_combiner.py. New: orthology.py [Phase 0], tox_head.py [Phase 2], splits.py [Phase 3], train.py/eval.py [Phase 4]. Data under data/raw/spatial/ and data/processed/spatial/; configs/ per-organ YAML; results/{tables,figures}/; MANIFEST.md [Phase 0].
