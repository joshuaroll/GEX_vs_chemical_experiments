# Decisions Intel

Synthesized from classified planning docs. No ADRs were ingested; all decisions
below are lead-researcher / design decisions extracted from the SPEC (source of
truth) and the two supporting DOCs. None are LOCKED ADRs. Items still requiring
professor sign-off are flagged as PROPOSED.

---

## DEC-frozen-baseline — MultiDCP / CheMoE is a frozen baseline
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md
- status: decided (design)
- decision: The baseline model (frozen MultiDCP / MultiDCP-CheMoE, Pham et al. 2022) is NOT retrained in this arm. The downstream toxicity head is the only trained component.
- scope: MultiDCP / MultiDCP-CheMoE baseline model; out of scope = retraining/fine-tuning the backbone.

## DEC-de-rule-spatial-divergence — region-basal DE anchor
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§1.2), /raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md (open item 3)
- status: PROPOSED (needs explicit note in REQUIREMENTS/PROJECT; flagged for professor sign-off in DOC 09)
- decision: Spatial GEX feature = `predicted_DE_region = predicted_treated(drug, region_basal) - region_basal`. This is a documented divergence from the parent project's `treated - diseased` anchor.
- scope: DE rule for the spatial arm. The non-spatial parent rule (treated - basal/diseased, top-k by |true DE|, raw expression forbidden) is unchanged for cell-line conditions.

## DEC-spatial-is-comparison-arm — additive, not replacement (Q1)
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md (Q1)
- status: decided (design, high confidence)
- decision: Spatial is an additional comparison arm (S-B / S-C / S-F) benchmarked against existing cell-line conditions B/C. It does not replace them. Headline = "region-conditioned vs cell-line-conditioned predicted GEX for toxicity."
- scope: experimental condition design. Agrees with SPEC §4 condition table.

## DEC-indirect-validity-plus-apap — validity path (Q2/C2)
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md (Q2/C2), /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (Phase 2)
- status: PROPOSED (acceptability of APAP stand-in flagged for professor; open item 2)
- decision: Accept indirect validation as the primary validity path (per-region predicted DE treated as a learned feature). Add one direct empirical anchor: an APAP single-drug Visium benchmark (GSE280652 / GSE272564). No panel-wide drug-perturbed spatial dataset exists as of mid-2026.
- scope: validity / Phase 2 verification.

## DEC-platform-visium-only — Visium as basal input (Q3/C1)
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md (Q3/C1)
- status: decided (design, high confidence)
- decision: Use whole-transcriptome Visium as basal input (near-100% coverage of the 10,716-gene MultiDCP space; 934/978 L1000 landmarks present). Targeted panels (MERFISH ~500, Xenium ~313, CosMx ~960) are region-annotation aids only. Conditional exception: a targeted panel may be promoted via Tangram reference-based imputation only if a paired whole-transcriptome snRNA-seq reference exists.
- scope: input platform selection. Agrees with SPEC Phase 0 dataset list.

## DEC-region-granularity-published — published annotations default (Q4)
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md (Q4)
- status: decided (design, high confidence)
- decision: Use published region annotations (not unsupervised spatial domains) as default. For Yu liver (Leiden cluster IDs) map to zonation via paper figures. Fall back to unsupervised Leiden domains only when no usable published annotation exists.
- scope: region definition.

## DEC-negative-result-acceptable — OOD risk framing (Q5/C3)
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md (Q5/C3), /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§2 concept 4, §9)
- status: PROPOSED (whether negative is publishable vs abandon flagged for professor; open item 1)
- decision: A negative or limitation result (predicted GEX degrades on OOD healthy-tissue contexts) is an acceptable, publishable scientific outcome. Gate the arm with a spatial Halt Gate (P2 Pearson < 0.3) so failure is detected and reported, not hidden.
- scope: project framing / halt-gate semantics.

## DEC-frozen-checkpoints-seam — no fabricated outputs
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§1.4)
- status: decided
- decision: The inference path raises until checkpoints are confirmed; `src/spatial/region_signature.py` enforces a NotImplementedError seam that must be replaced with the real call in Phase 2.
- scope: model wiring.

## DEC-ortholog-discipline — one-to-one orthologs only
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§1.8)
- status: decided
- decision: Human / mouse / rat expression aligned through one-to-one orthologs (Ensembl / MGI / RGD). Many-to-many orthologs are dropped (not collapsed); dropped fraction reported.
- scope: cross-species alignment.

## DEC-per-organ-only — no joint multi-organ model
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§10), /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (§7)
- status: decided
- decision: Per-organ training only; no joint multi-organ model. Liver/kidney/brain first; heart deferred. Dose-response channel prototyped on liver first then ported.
- scope: training scope.

## DEC-doseresponse-first-class-channel — three-channel fusion (D2)
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (D2)
- status: decided (lead-researcher call)
- decision: The dose-response (E-Hill / viability) channel is a first-class third channel in all organs, simultaneously a feature AND the cytotoxicity confound control. Fusion must beat both GEX-only and dose-response-only single-channel baselines.
- scope: fusion architecture / confound control.

## DEC-concat-mlp-fixed — deliberately simple classifier (D1/D3 spine)
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (§2), /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§3)
- status: decided
- decision: Headline classifier is a fixed 3-layer concat-MLP (GELU, dropout, BN) over projected channels; architecture search out of scope. Capacity held constant across conditions (inactive channel = zero tensor of identical shape).
- scope: classifier architecture.
