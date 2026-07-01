---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: **CORE THREE-WAY COMPARISON (structure vs DE vs DE+structure) COMPLETE for liver+kidney — CONSISTENT NEGATIVE FOR EXPRESSION (2026-06-30, committed efa8dac + d2efac6).** Honest drug-disjoint 5-fold x 5 seeds, ChemBERTa structure. Structure best in both organs (liver 0.714, kidney 0.654); predicted DE below; DE+structure gives no CI-clearing lift (liver -0.016 [-0.046,+0.012]; kidney +0.014 [-0.027,+0.055]). Negative robust across 6 probes: scaffold-disjoint (edge stable/widens), measured-DE also fails (kidney measured 0.527 ~ chance), internal hidden structure-ceilinged, engine weakly basal-sensitive (concat-across-basals can't exceed structure), tuned/late fusion still <= tuned structure (liver 0.769). Expression channel is structure-ceilinged. ENCODER SWEEP DONE (2026-06-30, committed 08909ae): the negative is NOT a ChemBERTa artifact — swept 9 structure encoders (chemberta, unimol_v1/v2, ecfp4/ecfp6, maccs, atompair, topotorsion, rdkit_fp) x 2 organs; +lift(both-struct) CI includes 0 in ALL 18 cells; structure > predicted DE in 16/18 (2 exceptions are the weakest encoders topotorsion/unimol_v1 falling below the flat expression bar); 3D UniMol does NOT beat cheap Morgan FPs (unimol_v1 worst both organs). Best per organ: liver chemberta 0.714, kidney ecfp4 0.705 (chemberta only 5th for kidney). See results/tables/P4_encoder_sweep.md. PRIOR (still valid): Phase-2 spatial predicted-region arm = closed negative (F4 instrumented-negative, Gate 4a Pearson 0.999987). NEXT (open forks): (2) brain (SIDER label block) + heart (no LINCS cardiac engine) still excluded -> only 2 organs; (3) write up the negative.**
stopped_at: P4 three-way suite complete + committed; nothing running. Six comparison tables in results/tables/P4_*.md (stage2_toxicity, scaffold_split_tox, internal_rep_tox, predicted_vs_measured_de, region_basal_tox, improved_both), scripts in scripts/{stage2_toxicity,scaffold_split_tox,internal_rep_tox,predicted_vs_measured_de_tox,region_basal_tox,improved_both_tox}.py. Engine+latent substrate committed efa8dac (train_organ_multidcp, build_organ_dataset, extract_multidcp_latent.extract_full, latent_vs_predicted, landmark_overlap; P1_latent_* + P3_* tables). Comparison committed d2efac6. Kidney improved_both is noisy (tuned structure 0.628 < naive both 0.660 on n=317) — small-n nested-CV instability, worth a glance if revisited. GPUs all free.
last_updated: "2026-06-30T22:15:00.000Z"
last_activity: 2026-07-01 -- tox-tuned signature (E/F): halt gate FIRED, structure>frozen>tuned on both organs x splits, tuning HURTS -> modeling side of expression-vs-structure closed; only lever left = tox-informative measured omics in right cell context (data problem). Committed 9e2d108. Earlier 2026-07-01: fusion/omics-signal investigation (6c538b5, f4b6a63, fe29dca). 2026-06-30: encoder sweep (08909ae), writeup (6e153a6), lab-meeting package (045f3dd).
progress:
  total_phases: 8
  completed_phases: 2
  total_plans: 11
  completed_plans: 11
  percent: 100
---

# STATE: Spatial Cross-Species Toxicity Prediction (MultiDCP-CheMoE)

> Living memory across sessions. Read this first.

## Project Reference

- **Project:** Spatial Cross-Species Toxicity Prediction (MultiDCP-CheMoE)
- **Root:** `/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests/` (subdir of umbrella repo `GEX_vs_chemical_experiments`; no own `.git`).
- **Core value:** Does a predicted, region-resolved molecular response signature predict organ-specific drug toxicity better than chemical structure alone, and does that signal translate across species (rodent → human)?
- **Current focus:** CORE feature comparison (structure vs DE vs DE+structure) — **COMPLETE for liver+kidney, consistent negative for expression (2026-06-30)**. The spatial predicted-region arm (Phase 2) stays closed-negative (F4). Next fork: encoder sweep, extend to brain/heart, or write up. See frontmatter status + Next action.
- **Isolation note:** This is a standalone GSD project. NEVER read/write `/raid/home/joshua/.planning` (separate, halted "liver" v0.5 project).

## Current Position

Phase: 01 (eda-the-bracket) — **COMPLETE 2026-06-23** (8/8 plans; verification 9/9). Halt Gate 2 fired as intended → reframed + documented; **halt LIFTED per D-10, Phase 2 UNBLOCKED**.
Plan: 8 of 8 executed (01-08 2×2-completion gap-closure DONE — the un-halt plan)

- **Phase:** 1 — DONE. Next: Phase 2 (MultiDCP wiring & toxicity head; carries HALT GATE 3).
- **Plan:** 01-08 EXECUTED + COMPLETE (Wave 4, gap-closure). Completed the floor×ceiling × leaky×disjoint **2×2** on 2648 profiles / 227 drugs (seed=42): **floor-leaky 0.9989, ceiling-leaky 0.9120, floor-disjoint 0.5461, ceiling-disjoint 0.6052.** Added a 95% paired-bootstrap CI on the profile-disjoint gap: **+0.0590, CI [0.0259, 0.0921]**, 10000/10000 resamples. Appended the dated RESOLUTION note to HALT_REASON.md (D-10). 5 atomic commits (RED→GREEN→wire→regenerate→SUMMARY).
- **Status:** **PHASE 1 COMPLETE — halt LIFTED (D-08/D-10).** The bracket is fully documented (2×2 + CIs); per D-08/D-10 the halt lifts on honest documentation regardless of the gap sign. Phase 2 (HALT GATE 3) is now UNBLOCKED. **This STATE.md is the durable un-halt record** (the RESOLUTION note in HALT_REASON.md is volatile — `run_p1_eda.py all` rewrites that file via `_write_halt_reason` on every run).
- **Progress:** `[####################] P1: 8/8 COMPLETE; halt LIFTED → Phase 2 unblocked`

**Halt Gate 2 result (as designed — fired, then documented/reframed, NOT a blocker anymore):** PRIMARY drug-level gate: structure floor 0.611, leakage-free drug-grouped measured ceiling 0.434, gap −0.177, 95% CI [−0.316, −0.033] → FIRES (still fires by design; driver still exits 1). **Completed-2×2 headline:** floor-leaky (0.999) ≥ ceiling-leaky (0.912) → the Wang/Li ~0.798 benchmark is largely **drug-identity memorization that chemical structure reproduces** ("explains the paper"); at the honest drug-disjoint level the measured ceiling shows a small real lift over structure (+0.059, CI [0.026, 0.092]) but does not clear the drug-level gate. See `results/tables/P1_eda.md` + `01-VERIFICATION.md`.

**Next action (2026-06-30):** the core three-way comparison is DONE for liver+kidney and committed (efa8dac substrate, d2efac6 the P4 suite); the finding is a consistent negative — predicted AND measured expression are structure-ceilinged (see frontmatter status for the six-probe breakdown). Open forks:
1. **Encoder sweep** — DONE (2026-06-30, committed 08909ae). `scripts/encoder_sweep_tox.py` swept 9 structure encoders x 2 organs holding the engine predicted-DE arm + drug set fixed. Negative is encoder-robust: +lift(both-struct) CI includes 0 in all 18 cells; structure > predicted DE 16/18; 3D UniMol does not beat Morgan FPs. Result table `results/tables/P4_encoder_sweep.md`. `unimol_tools 0.1.6` now installed in dili_v04_env (pulled numpy 2.2.6 + huggingface-hub 1.14 — full chain re-smoke-tested OK: engine import, fingerprints, chemberta, unimol v1/v2). Encoder additions + unimol-0.1.6 fix + OOM batch fix in `src/spatial/structure_encoders.py`.
2. **Extend beyond 2 organs** — brain blocked (SIDER CID-keyed → needs CID→structure + MedDRA SOC filter); heart blocked (DICTrank has no SMILES + no LINCS cardiac engine). Both are data-join work, bigger lift.
3. **Write up the negative** — DRAFTED (2026-06-30, committed 6e153a6): `results/P4_writeup.md` consolidates the 6 probes + 9-encoder sweep + measured-DE control + measured/latent three-way corroboration. Lab-meeting package committed 045f3dd (`lab_meetings/2026-07-01.md` + 4 figures). Negative-result framing pending professor sign-off (D3). Scope framing honestly per fork 4.
4. **Fusion / omics-signal investigation** — DONE 2026-07-01 (committed 6c538b5 + f4b6a63), in response to user challenge "omics should add info if fused properly". Findings: (a) predicted-DE arm is near-circular — structure linearly recovers 29-53% of predicted DE but has NEGATIVE R^2 to measured DE, so only measured omics tests the hypothesis; (b) measured-omics-alone stays at chance across 3 reps x 3 models (best kidney 0.557); (c) NO fusion beats structure (late/stacker/nonlinear concat-MLP); (d) the high oracle (0.90) is INFLATION — oracle(structure+PERMUTED omics)=0.89-0.90, real marginal complementarity +0.01-0.03 within noise. Conclusion: the bottleneck is the omics SIGNAL, not the combiner; fusion can't rescue an at-chance arm. Tables `P4_fusion_diagnostics.md`, `P4_omics_arm.md`.
5. **Tox-tuned signature (conditions E/F)** — DONE 2026-07-01 (committed 9e2d108), user-chosen. End-to-end fine-tuning the MultiDCP engine toward the tox label (tuned_E) + anchored variant (tuned_F) vs frozen DE vs structure, drug + scaffold split, 3 seeds x 5 folds. **PRE-REGISTERED HALT GATE FIRED:** structure > frozen_DE > tuned_E on both organs x both splits (drug: liver 0.712/0.578/0.559, kidney 0.652/0.560/0.532; scaffold: liver 0.709/0.608/0.560, kidney 0.668/0.602/0.533). Tox-tuning HURTS (overfits a 978-gene bottleneck on ~340 labels; anchor F partially protects, F>E everywhere). Table `P4_tox_tuned.md`. **This closes the MODELING side of expression-vs-structure** (frozen, tuned, fused, 9 encoders, 6 probes all negative). **Only remaining lever = tox-informative MEASURED omics in the right cell context** (kidney proximal-tubule RPTEC/TERT1 / HK-2; liver primary hepatocyte) — a DATA problem, largely non-existent at LINCS scale, not a modeling one.

Kidney `improved_both` noise (tuned structure 0.628 < naive 0.660 on n=317) is a known small-n nested-CV wrinkle; sanity-check if that table gets used in a writeup.

Historical context: this comparison was originally pivoted to MultiDCP `global_features` latents, but the P4 suite uses the direct predicted-DE output (978/N_PDG genes) — the honest, interpretable feature. VALIDATED CAVEAT still holds: global_features' per-drug signal is the 128-d drug block only (cell/dose constant), so a latent "expression" arm = MultiDCP's drug encoder. MEASURED-expression three-way (`P1_three_way_comparison.*`) and latent three-way (`P1_latent_three_way_chemberta.*`) agree with P4. Spatial predicted-region arm CLOSED negative (F4, `P2_gate4a_10ep.md`).

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases total | 8 |
| Phases complete | 1 (P0) |
| Requirements total | 19 |
| Halt gates | 6 (phases 0,1,2,3,4,6) |

## Accumulated Context

### Decisions (locked / design)

- Frozen MultiDCP/CheMoE baseline; only the downstream tox head trains.
- Spatial is an additive comparison arm (S-B/S-C/S-F vs cell-line B/C), not a replacement.
- Visium whole-transcriptome only as basal input; targeted panels are annotation-only.
- Published region annotations default; per-organ only; fixed 3-layer concat-MLP; dose-response is a first-class third channel and confound control.
- One-to-one orthologs only; report dropped fraction.
- **Heart ACTIVATED 2026-06-21 (per user direction; deferral rescinded)** — now the 4th in-scope organ. Input: public Kuppe et al. 2022 Visium control sections (`kuppe_heart`, 4×.h5ad, Zenodo 6578047, CC BY 4.0; coverage 0.834 of MultiDCP 10716, `usable_as_input=True`). Labels: FDA DICTrank (`data/raw/labels/dictrank/`, 1318 drugs, no SMILES → needs structure join). EGA-gated Kanemaru remains non-usable. Heart still sequenced last.

### Proposed decisions (PENDING professor sign-off — confirm at phase planning)

1. **Spatial DE-rule divergence** `predicted_treated(drug, region_basal) - region_basal` (vs parent `treated - diseased`). Confirm at P2/P4.
2. **APAP single-drug Visium anchor** (GSE280652/GSE272564) as primary direct validity stand-in. Confirm at P2.
3. **Negative result is publishable** (stop-and-reframe vs stop-and-abandon). Confirm before P1/P4 gates are acted on.

(Operational sign-offs handled in P0: approve adding `squidpy`/Tangram to env; sign off dataset-accession corrections before download scripts.)

### Plan 01-04 decisions (locked)

- **compute_ceiling drug-level aggregation**: profile-level LR OOF probs first, then mean per `compound_name` (lowercase) before `roc_auc_score`. Pitfall-8 compliance is internal to the library, not caller's responsibility.
- **gate_fires = ci_lower <= 0** (inclusive): matches D-02 "CI includes 0" wording.
- **NaN-sentinel for degenerate bootstrap draws**: `gaps[i] = nan` in loop; `valid_gaps = gaps[~np.isnan(gaps)]` after loop.
- **load_ceiling does not catch FileNotFoundError**: propagates so run_p1_eda.py can print "no measured ceiling -- floor only" per D-05.

### Plan 01-03 decisions (locked)

- **compute_floor uses float32 cast on fps**: sklearn LR/RF accept float32 natively; halves memory vs float64 at n_drugs scale.
- **auprc_base_rate == float(y.mean())**: PR no-skill baseline is positive prevalence by definition.
- **floor_probabilities takes seed: int (not Sequence)**: plan 06 needs one aligned OOF vector per seed; caller iterates seeds if needed.
- **RF n_jobs=-1**: deterministic via fixed random_state per seed; all cores used for cross_val_predict speed.

### Plan 01-02 decisions (locked)

- **labels.py returns plain DataFrames** (not LabelTable NamedTuple) — test contracts call `.set_index()` directly on return value; LabelTable defined for callers needing richer metadata.
- **SMILES dili_binary assignment** uses `pos_mask[keep_mask].astype(int).values` to prevent pandas index misalignment after filtering.
- **TDC fallback is a callable hook** in join_smiles_cascade (not a pytdc import) — keeps module pure, no network call at import time.
- **smiles_to_ecfp4** returns (fps (n,2048) uint8, valid_mask (n,) bool); zero-vector + valid_mask=False for MolFromSmiles None (T-01-03).
- **EDA-01 requirement met**: 5 plan-01 Nyquist tests now GREEN.

### Plan 01-01 decisions (locked)

- **Wave-0 Nyquist RED state confirmed**: 6 new EDA test files fail import on `src.spatial.eda` (expected; modules written in plans 02-05). 124 pre-existing tests unaffected.
- **All 6 SHA256s matched on-disk** before writing to MANIFEST.md (XC-01 clean; no drift).
- **tests/spatial/__init__.py pre-existed** as empty file; no action needed.

### Plan 01 decisions (locked)

- **slug field is single source of truth** for `data/raw/spatial/<slug>/` directory. Download driver and test suite must read `entry.slug`; no independent slugification.
- **test_raw_datasets_present** skips when RAW dir has no subdirectories (`.gitkeep` alone is not a trigger).
- **Stale accessions corrected** in `src/spatial/datasets.py`: Yu→figshare 22321447, Maynard→spatialLIBD, Lake/KPMP→GSE183456+GSE183279, Siletti→snRNA-seq.
- **Rodent basal-context candidates registered**: mouse liver (GSE272564 control arm), kidney (GSE252772), brain (GSE233983). Final healthy-spot confirmation at download (Pitfall 5).

### Plan 04 decisions (locked)

- **MultiDCP 10716-gene symbol list** sourced from `pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv` column names (verified len==10716, N_PDG confirmed).
- **Rodent coverage gate** = n_genes > 10000 (not human-symbol match); mouse/rat gene symbols differ; cross-species alignment is Phase 2 ortholog map.
- **Halt Gate 1: NOT FIRED** — human basal Visium >99% coverage; rodent datasets 32245 genes.
- **MANIFEST.md complete**: SHA256+license+Whole-transcriptome for all 13 datasets; both frozen-checkpoint SHAs; env snapshot with squidpy 1.8.2.
- **Phase 0 phase complete** — all DATA-01/02/03 requirements met; 132 tests pass.

### Plan 03 decisions (locked)

- **Ensembl release 116, query date 2026-06-20** — baked into raw TSV filename `orthologs_raw_116_20260620.tsv` (XC-10).
- **92.75% dropped fraction is expected** — BioMart returns all human genes including those with no homolog; strict mutual one2one yields ~15k of ~220k rows.
- **build_one2one_orthologs() accepts DataFrame, Path, or str** — offline testable against fixture without network.
- **Release probe regex** fixed to match `ensembl_mart_116` format (original matched `_gene_ensembl_NN`).

### Plan 02 decisions (locked)

- **P0 brain toxicity labels = SIDER meddra_all_se.tsv.gz (SOC filter at use-time)**; Lane-Ekins seizure and DNT-IVB DEFERRED to Phase 1.
- **DIRIL kidney supplement TODO note written** (Elsevier journal gate 404); kidney labels must be resolved before Phase 4. Never fabricated (XC-01). → **SUPERSEDED 2026-06-21:** DIRIL ACQUIRED from the FDA (`data/raw/labels/diril/diril_dataset_508.xlsx`, 317 drugs w/ SMILES + binary DIRI); TODO removed. The Elsevier candidate had the wrong article S-number.
- **DILIst/DILIrank acquired via sibling SHA-copy** (FDA bot protection blocked network; sibling has SHA-verified copies; data is real).
- **Figshare MD5 verification**: both Yu liver files (L5/L18) passed MD5 check.
- **13 spatial datasets downloaded**; KPMP GEO supplementary (4.87 GB) succeeded; KPMP portal ToS was not needed.
- **chen_brain_mtg expected_files corrected** in registry: was `GSE200474_RAW.tar` (404), now `GSE200474_Deseq2_...txt.gz` (actual GEO suppl file). → **SUPERSEDED 2026-06-21:** GSE200474 was a WRONG-ACCESSION error (it is ALS iPSC motor-neuron bulk RNA-seq, not MTG Visium). Corrected to **GSE220442** (real Chen 2022 MTG Visium; 6 sections, 36,601 genes, 99.8% coverage). `expected_files=("GSE220442_counts_and_images.tar.gz",)`.

### Plan 01-06 decisions (locked) — HALT GATE 2

- **Ceiling CV must be drug-grouped.** Original profile-level `StratifiedKFold` leaked drug identity (one drug = up to 784 profiles) → meaningless 0.56 ceiling. Fixed to `StratifiedGroupKFold` over compound + per-fold `StandardScaler` (now unit-consistent with the drug-level floor). Honest ceiling = 0.434. Regression test `test_ceiling_no_drug_leakage` added (RED→GREEN).
- **Halt Gate 2 FIRED** (gap −0.177, CI [−0.316, −0.033]) → **stop-and-REFRAME (D-02), user-confirmed.** Phase 2 BLOCKED.
- **Headline finding (reproducible in P1_eda.md):** Wang/Li-style benchmark is **+0.31 AUROC drug-leakage-inflated** (0.912 leaky vs 0.605 drug-disjoint at profile level). Honest profile-level measured ceiling (0.605) ≈ structure floor (0.611). Framing is "no measured lift over structure; benchmark inflated", NOT "structure beats biology".
- **Power:** 38 negative drugs; conservative min-detectable gap (0.198) > observed (0.177); gate significance rests on the paired bootstrap (thin margin). Reframe should expand negatives.
- **Driver now reports leakage decomposition + Hanley-McNeil power** so the headline is reproducible from `run_p1_eda.py all` (seed=42).

### Plan 01-05 decisions (locked)

- **OOD method = Mahalanobis, 978-gene landmark subspace, alpha=1e-2** (resolves "Claude's Discretion" from CONTEXT.md; baked into OOD_METHOD constant in region_diagnostics.py for P1_eda.md reporting).
- **squidpy imported lazily inside compute_moran_svgs** (try/except ImportError); module-level import avoided (Pitfall 9 compliance).
- **human_mouse_liver_correlation raises ValueError** (not silent NaN) when < 2 genes match -- diagnostic over silent failure.
- **basal_similarity_matrix accepts dict or ndarray+labels**; sparse toarray guard applied (mirrors pseudobulk.py pattern).

### Todos / watch items

- ~~DIRIL kidney labels~~ DONE 2026-06-21 — acquired from FDA (diril_dataset_508.xlsx). TODO note removed.
- DICTrank heart labels acquired (FDA, 1318 drugs) but have NO SMILES — needs a name→structure (PubChem/DrugBank) join before the structure-only baseline (Phase 4).
- `src/spatial/region_signature.py` holds the `NotImplementedError` seam — must be replaced with the real frozen-checkpoint call in Phase 2, not before.
- Extend `src/spatial/` (132 passing tests now); do not rebuild. New files: tox_head.py [P2], splits.py [P3], train.py/eval.py [P4].
- Demoted to usable_as_input=False (no usable standard counts): abedini_kidney (images-only tar), canela_kidney (long-read isoform), maynard_dlpfc (metadata-only), kanemaru_heart (EGA controlled, empty). chen_brain_mtg is NOW a real GSE220442 Visium input. A content/shape guard (`src/spatial/data_validation.py` + `tests/test_data_validation.py`) enforces counts-presence for every usable_as_input dataset and fires Halt Gate 1 in the driver on a present-but-empty/wrong file.
- 8 usable_as_input inputs, all counts-verified on disk: yu2022_liver, andrews_liver, lake_kpmp_kidney, chen_brain_mtg, kuppe_heart, gse272564_mouse_liver_ctrl, gse233983_mouse_brain, gse252772_mouse_kidney (.rds — needs R→anndata conversion).
- gse252772_mouse_kidney: R .rds.gz files only — no Python-readable feature matrix without conversion. Needs `rpy2` or `anndata2ri` at Phase 1.

### Blockers

- **Phase 3 BLOCKED — HALT GATE 3 fired (2026-06-24).** Do not start Phase 3 (splits) until a human gate-review decides the spatial reframe. The frozen CheMoE backbone does not produce region-resolved DE for healthy-liver basals (near-zonal-invariant), so the predicted region-resolved signature — the milestone's core bet — does not survive its first direct test as currently wired. Per D-09 this is stop-and-REFRAME (negative is publishable), not abandon. See `.planning/phases/02-multidcp-wiring-tox-head/HALT_REASON.md` + `results/tables/P2_apap_validation.md` + `02-VERIFICATION.md`. Reframe options to weigh at review: (a) test whether ANY conditioning signal beats structure even without zonal resolution (proceed to P3/P4 with S-C as a degenerate/region-pooled condition, reporting the spatial null honestly); (b) swap/seek a basal-context encoder that responds to healthy-tissue basals (re-opens S-B provenance / a non-cancer-trained backbone); (c) reframe the paper around the instrumented negative (frozen cancer-line GEX models do not transfer region structure to healthy tissue) per concept 4.
- ~~Phase 2 BLOCKED (Halt Gate 2)~~ RESOLVED 2026-06-23 (P1 un-halt, D-10).

## Session Continuity

- **Last activity:** 2026-06-25 (encoder smoke + Gate 4a; partial pass — see bullets below + the RESUME note)
- **Stopped at:** **Phase 2 EXECUTED + HALTED (Gate 3 fired)** via `/gsd-autonomous --to 2`. All 4 plans done + verified (4/4, 177 pure tests). Halt Gate 3 fired (pericentral r=+0.0166 < 0.3) → stop-and-reframe (D-09). Autonomous stopped at the scoped `--to 2` review point; lifecycle skipped (milestone not complete). Code review deferred (advisory; premature before the reframe — run `/gsd-code-review 2` if keeping the wiring).
- **Reframe decision (2026-06-24 gate review):** **FIX THE ENCODER — re-open Phase 2** (option b). See `02-REFRAME.md`. Phase 3 stays blocked.
- **DIAGNOSTICS DONE (2026-06-24) → VERDICT: DEAD-WEIGHTS (D), exact mechanism.** Two chained diagnostics: `P2_encoder_diagnostic.md` (b00ef98, ENCODER-FLAT: inputs differ, model ignores basal) then `P2_wiring_vs_weights.md` (00b9f55) isolated the cause. **(W) our wiring is correct** (upstream-native rebuild bypassing region_signature.py shows the same invariance; `_call_model` passes input_cell_gex correctly). **(M) no hidden mode** (CheMoE forward always uses the perturbed head; job_id ignored; no AE method). **Mechanism:** basal is alive out of the input Linear (Pearson 0.9996) then COLLAPSES inside the transformer cell-encoder — `repeat(1,1,32)` makes all 32 d_model channels identical so LayerNorm discards the basal magnitude, and the final `encoder.norm.weight`≈3.3e-3 crushes residual variance. Weights aren't zero — it's a collapsed composed function. The row-17 checkpoint cannot condition on basal (collapses ALL cell-conditioned conditions B/C/S-B/S-C, not just spatial). **The "fix the encoder" path is now fully characterized — see Blockers/02-REFRAME.**
- **DECISION (2026-06-24): F4 — fix + retrain the upstream encoder.** Professor signed off reversing `DEC-frozen-baseline` (scoped exception, recorded in PROJECT.md): apply the minimal encoder architecture fix in the parent `MultiDCP_CheMoE_pdg` project (`linear_encoder_flag=True` / replace `repeat(1,1,32)` with `Linear(1→32)`) + retrain, gate the new checkpoint on a basal-sensitivity check, then re-cache region DE here + re-run Halt Gate 3. New workstream in the parent project (`/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/`, own CLAUDE.md/planning).
- **F4 SCOPED + RUNNABLE (2026-06-24):** `02-F4-SCOPE.md` — fix is a config toggle (`--linear_encoder_flag`, no parent-source edit; `LinearEncoder` already exists). Data/script/GPUs verified. Validation gate defined (4a basal sensitivity + 4b ≤0.05 accuracy regression). Decided to run a SHORT SMOKE (8 epochs) first to test the key uncertainty (does training teach the encoder to use the basal?).
- **ENV (resolved 2026-06-25):** `mdcp_env` had drifted (setuptools 80 removed `declare_namespace`; wandb needs protobuf<6 / pydantic>=2.6). Left `mdcp_env` as found (setuptools 80.9.0). Cloned it to **`mdcp_f4`** and pinned setuptools<80 / protobuf<6 / pydantic 2.6–3 → training works. **Use `mdcp_f4` for the retrain, not `mdcp_env`.**
- **GATE 4A RESULT (2026-06-25): PARTIAL — encoder fixed, head not yet.** 8-epoch linear-encoder smoke (`best_model_linearenc_smoke.pt`). cell_hidden max|Δ| 1.04 (was 1.7e-8 → encoder fix WORKS) and pred max|Δ| 3.17e-2 both pass; but pred Pearson(periportal,pericentral)=0.999987 (gate wants <0.999) → predicted DE still near-collinear across zones, which is what Halt Gate 3 needs. So the head hasn't learned to turn cell_hidden into zone-distinct DE *shapes*. Result: `results/tables/P2_gate4a_smoke.md` (commit a6f904a), script `scripts/gate4a_basal_sensitivity.py`.
- **OPEN DECISION (awaiting user; they /clear'd to resume fresh):** is the head under-trained (best-val saves early ~epoch 1-2 → more epochs may fix) or is it a data/label limit (DE labels drug-dominated → won't fix)? **Cheap checks first** (see `/raid/home/joshua/claude_memory/downstream_2026/notes/2026-06-25-RESUME.md` "GATE 4A RESULT — WHY + PICK UP HERE"): (1) which epoch best-val saved; (2) can the prediction head reshape cell_hidden or only rescale it (read multidcp_chemoe_pdg.py head wiring); (3) run the REAL APAP per-zone gate on the smoke checkpoint instead of the proxy. Then choose: longer smoke (~40-50 ep, recommended) → re-gate; full ~8h run; or redirect to F3 (downstream basal contrast, not a predicted signal) / reframe to the instrumented negative.
- **Resume with:** read `/raid/home/joshua/claude_memory/downstream_2026/notes/2026-06-25-RESUME.md` (operational handoff, START at "GATE 4A RESULT — WHY + PICK UP HERE") + `2026-06-25-progress.md` (findings + professor email). Then the cheap checks above. F4 full-run command + downstream re-integration are in the RESUME note and `02-F4-SCOPE.md`.
