# Context Intel

Running notes keyed by topic, appended verbatim with source attribution.

---

## Topic: Research question and publishable concepts
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§2)
Core question: Does a predicted, region-resolved molecular response signature predict organ-specific drug toxicity better than chemical structure alone, and does that signal translate across species well enough that cheaper rodent data can stand in for human? Four standalone publishable concepts: (1) region-conditioned beats cell-line-conditioned; (2) structure-vs-omics translatability; (3) rodent-to-human transfer as proxy (headline best case); (4) honest negatives as method contributions (OOD failure, cytotoxicity confound, BBB confound instrumented and reported).

## Topic: Experimental conditions and cross-species axis
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§4)
Conditions: A (structure-only, zero tensor); B/C (cell-line predicted GEX, MultiDCP / CheMoE on 9 LINCS lines); S-B/S-C (spatial-region predicted GEX, per-region tissue basal); G/H (dose-response, predicted ehill frozen/tox-tuned); fusion (three-channel). Channel-ablation baselines every organ: dose-response-only, GEX-only; fusion must beat both. Cross-species axis orthogonal: each condition runs within-human and within-rodent plus transfer directions (train-rodent-test-human and reverse); structure-only transfer is the reference isolating omics contribution to translatability.

## Topic: Three-channel pipeline lineage (parent liver design)
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (§0-§4)
This DOC is the cross-organ design matrix that the kidney/brain per-organ docs reference. Three channels from the prof's drawing: chem+cell-line->dose-response (E-Hill viability), chem+cell-line->gene-expression (DE), chem->embedding. D1: slim per-organ docs referencing the liver template; milestone numbering kept aligned (M0-M12), dose-response appended M13-M14. D2: dose-response promoted to first-class channel in all organs, simultaneously feature AND confound control (Szalai cytotoxicity critique); liver gains conditions G/H gated after Halt Gate 3. D3: file locations. The spatial SPEC inherits this channel architecture and confound-control framing.

## Topic: Per-organ channel availability and value propositions
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (§3)
Predicted GEX and dose-response available for every organ; only condition D (measured upper bound) quality changes by LINCS coverage. Kidney = strongest case: HA1E (kidney-epithelial, in MultiDCP's 9 lines) is the in-distribution anchor; novel capability = extrapolating to RPTEC/TERT1 / HK-2 proximal-tubule basal (cannot be measured at LINCS scale). Brain = partial measured anchor (NPC/NEU sparse) + severe BBB confound; headline must be on BBB-permeable subset. Liver = positive control / most contested benchmark (MiniMol 0.956 on small saturated TDC.DILI; 0.80 is transcriptomic-signature-only ceiling, not overall DILI ceiling).

## Topic: OOD risk on healthy tissue (cancer-line manifold shift)
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md (Q5/C3)
The cancer-to-healthy-tissue shift is unmeasured; closest proxy (Li et al. 2024 benchmark) found no method generalized consistently across cell types, performance dominated by distribution distance. Cross-type Pearson r is upward-biased (Nicol et al. 2026). Mitigations: spatial Halt Gate (Pearson < 0.3); consider fine-tuning cell-context encoder on LINCS PHH basal (~77 compounds) as nearest in-distribution anchor; treat fill_value=0.0 as documented assumption (~30% genes absent from Visium); consider hepatocyte-atlas mean-imputation.

## Topic: Dataset plan and accession corrections
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md (Data plan, Data-reference corrections)
Usable Visium basal inputs (Y): Yu 2022 liver (Figshare), Andrews/Teichmann GSE185477, Lake/KPMP kidney (GSE183456+GSE183279), Abedini 2024 GSE211785 kidney (substitute), Maynard 2021 DLPFC (spatialLIBD), Kanemaru 2023 heart (E-MTAB-12975, deferred). Annotation-only (N): Wu/Moffitt liver MERFISH, Allen ABC mouse MERFISH, 10x Xenium brain, Muto CosMx kidney. Validation-only: GSE280652/GSE272564 APAP mouse liver. Corrections: Yu 2022 = PMID 36261431 / Figshare (GSE189994 is wrong, unrelated m6A study); Lake/KPMP = GSE183456+GSE183279 (GSE211785 is Abedini 2024); Maynard = spatialLIBD (GSE144239 is squamous cell carcinoma); no public human-brain MERFISH exists (Siletti 2023 is snRNA-seq; only mouse ABC MERFISH has 500-gene panel).

## Topic: Validity plan (indirect validation order)
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md (Validity plan)
Run in order with spatial Halt Gate as go/no-go: (1) APAP single-drug benchmark (predicted vs measured per-zone DE, GSE280652/GSE272564, r<0.3 = stop); (2) cell-line-anchor cross-check; (3) biomarker concordance per region (KIM-1/HAVCR1 proximal tubule for ATN; pericentral CYP2E1 for CYP-bioactivated hepatotoxins); (4) LINCS L1000 bulk DE as pharmacological upper bound (condition D); (5) downstream toxicity improvement (the actual claim, S beats B/C); (6) spatially-variable-gene retention QC.

## Topic: Toxicity labels per organ
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§6 Phase 0); /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (§8)
Liver: DILIst, DILIrank. Kidney: DIRIL (Connor 2024, 317 drugs 171+/146-, no standard split -> publish scaffold split). Heart: DICTrank. Brain: SIDER nervous-system SOC / Lane-Ekins seizure / DNT-IVB; NeuroTDPi bar ~0.84 (BBB 0.97 is BBB-only). Cheng 2025 JCIM kidney bar ~0.85-0.90 (transcriptomics 0.90 > chem 0.89; not a fused model).

## Topic: Open items needing professor sign-off
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md (Open items)
(1) Confirm negative spatial result is publishable (sets stop-and-reframe vs stop-and-abandon). (2) Confirm APAP single-drug benchmark acceptable as per-region validation stand-in. (3) Approve DE-rule divergence (predicted_treated - region_basal vs treated-diseased) with explicit REQUIREMENTS/PROJECT note. (4) Approve adding squidpy (optionally Tangram). (5) Sign off dataset-reference corrections before download scripts written. These are design questions, not workflow blockers; the SPEC has already adopted the proposed resolutions.

## Topic: Out of scope
- source: /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md (§10)
Retraining/fine-tuning MultiDCP/CheMoE (frozen baseline only); joint multi-organ model (per-organ only); heart as a starting organ (sequenced after liver/kidney/brain); non-DE metrics.

## Topic: Bibliographic corrections (repo-level, flagged not blocking)
- source: /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (§8)
MultiDCP cite = Pham T-H et al. 2022 Patterns 3:100441 (not Liu/Xie; reconcile/delete fabricated wu2022multidcp entry). "Wang 2020" -> "Li 2020" (Li T, Tong W, Front Bioeng Biotechnol 8:562677). Szalai -> 2019 NAR (not 2018 preprint). Gene counts: 978 / 12,328 / 10,716 distinct. ChemBioHepatox (Shou 2025) needs primary DOI.
