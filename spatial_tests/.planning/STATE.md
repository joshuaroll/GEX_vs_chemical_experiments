---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: **PHASE 2 EXECUTED + HALTED (2026-06-24). Engineering deliverables COMPLETE (verify 4/4, 177 pure tests). HALT GATE 3 FIRED — pericentral Pearson +0.0166 < 0.3 → stop-and-REFRAME (D-09). Phase 3 BLOCKED pending human gate-review of the reframe. Autonomous run stopped at --to 2 as scoped.**
stopped_at: Executed all 4 Phase-2 plans (W0 tests RED → W1 WIRE-01 seam+cache ∥ WIRE-02 tox_head+smoke-train → W2 WIRE-03 APAP gate). Forward path is real (MultiDCP_CheMoE_AE strict-load 0/0, liver DE cached human+mouse, tox_head smoke-trains loss 0.654→0.050). Halt Gate 3 FIRED: the frozen cancer-line CheMoE encoder is near-zonal-invariant on healthy-liver basals (per-zone DE differs by float32 epsilon 5.96e-08), so predicted DE has no regional contrast while measured APAP DE carries real pericentral injury zonation → pericentral r=+0.0166. Per D-09 = stop-and-reframe (not abandon). Awaiting gate-review reframe decision before Phase 3.
last_updated: "2026-06-24T00:30:00.000Z"
last_activity: 2026-06-24
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
- **Current focus:** Phase 02 — multidcp-wiring-tox-head **EXECUTED + HALTED (Gate 3 fired 2026-06-24)**. Next: human gate-review of the spatial reframe before Phase 3.
- **Isolation note:** This is a standalone GSD project. NEVER read/write `/raid/home/joshua/.planning` (separate, halted "liver" v0.5 project).

## Current Position

Phase: 01 (eda-the-bracket) — **COMPLETE 2026-06-23** (8/8 plans; verification 9/9). Halt Gate 2 fired as intended → reframed + documented; **halt LIFTED per D-10, Phase 2 UNBLOCKED**.
Plan: 8 of 8 executed (01-08 2×2-completion gap-closure DONE — the un-halt plan)

- **Phase:** 1 — DONE. Next: Phase 2 (MultiDCP wiring & toxicity head; carries HALT GATE 3).
- **Plan:** 01-08 EXECUTED + COMPLETE (Wave 4, gap-closure). Completed the floor×ceiling × leaky×disjoint **2×2** on 2648 profiles / 227 drugs (seed=42): **floor-leaky 0.9989, ceiling-leaky 0.9120, floor-disjoint 0.5461, ceiling-disjoint 0.6052.** Added a 95% paired-bootstrap CI on the profile-disjoint gap: **+0.0590, CI [0.0259, 0.0921]**, 10000/10000 resamples. Appended the dated RESOLUTION note to HALT_REASON.md (D-10). 5 atomic commits (RED→GREEN→wire→regenerate→SUMMARY).
- **Status:** **PHASE 1 COMPLETE — halt LIFTED (D-08/D-10).** The bracket is fully documented (2×2 + CIs); per D-08/D-10 the halt lifts on honest documentation regardless of the gap sign. Phase 2 (HALT GATE 3) is now UNBLOCKED. **This STATE.md is the durable un-halt record** (the RESOLUTION note in HALT_REASON.md is volatile — `run_p1_eda.py all` rewrites that file via `_write_halt_reason` on every run).
- **Progress:** `[####################] P1: 8/8 COMPLETE; halt LIFTED → Phase 2 unblocked`

**Halt Gate 2 result (as designed — fired, then documented/reframed, NOT a blocker anymore):** PRIMARY drug-level gate: structure floor 0.611, leakage-free drug-grouped measured ceiling 0.434, gap −0.177, 95% CI [−0.316, −0.033] → FIRES (still fires by design; driver still exits 1). **Completed-2×2 headline:** floor-leaky (0.999) ≥ ceiling-leaky (0.912) → the Wang/Li ~0.798 benchmark is largely **drug-identity memorization that chemical structure reproduces** ("explains the paper"); at the honest drug-disjoint level the measured ceiling shows a small real lift over structure (+0.059, CI [0.026, 0.092]) but does not clear the drug-level gate. See `results/tables/P1_eda.md` + `01-VERIFICATION.md`.

**Next action:** **HUMAN GATE-REVIEW of the Halt Gate 3 reframe — Phase 3 blocked until then.** Phase 2 executed cleanly and fired Halt Gate 3 as designed: predicted region-resolved DE has no zonal contrast (frozen cancer-line CheMoE encoder near-zonal-invariant on healthy-liver basals), pericentral r=+0.0166 vs measured APAP zonation. Decide the reframe (see Blockers for options a/b/c), then resume. Autonomous run stopped at --to 2 as scoped. Artifacts to review: `HALT_REASON.md`, `results/tables/P2_apap_validation.md`, `02-VERIFICATION.md`, `02-02-SUMMARY.md` (zonal-invariance flag). Resume after the decision with `/gsd-discuss-phase 3` (likely re-scoped per the reframe) or re-open Phase 2 wiring if option (b).

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

- **Last activity:** 2026-06-24
- **Stopped at:** **Phase 2 EXECUTED + HALTED (Gate 3 fired)** via `/gsd-autonomous --to 2`. All 4 plans done + verified (4/4, 177 pure tests). Halt Gate 3 fired (pericentral r=+0.0166 < 0.3) → stop-and-reframe (D-09). Autonomous stopped at the scoped `--to 2` review point; lifecycle skipped (milestone not complete). Code review deferred (advisory; premature before the reframe — run `/gsd-code-review 2` if keeping the wiring).
- **Reframe decision (2026-06-24 gate review):** **FIX THE ENCODER — re-open Phase 2** (option b). See `02-REFRAME.md`. Phase 3 stays blocked.
- **DIAGNOSTICS DONE (2026-06-24) → VERDICT: DEAD-WEIGHTS (D), exact mechanism.** Two chained diagnostics: `P2_encoder_diagnostic.md` (b00ef98, ENCODER-FLAT: inputs differ, model ignores basal) then `P2_wiring_vs_weights.md` (00b9f55) isolated the cause. **(W) our wiring is correct** (upstream-native rebuild bypassing region_signature.py shows the same invariance; `_call_model` passes input_cell_gex correctly). **(M) no hidden mode** (CheMoE forward always uses the perturbed head; job_id ignored; no AE method). **Mechanism:** basal is alive out of the input Linear (Pearson 0.9996) then COLLAPSES inside the transformer cell-encoder — `repeat(1,1,32)` makes all 32 d_model channels identical so LayerNorm discards the basal magnitude, and the final `encoder.norm.weight`≈3.3e-3 crushes residual variance. Weights aren't zero — it's a collapsed composed function. The row-17 checkpoint cannot condition on basal (collapses ALL cell-conditioned conditions B/C/S-B/S-C, not just spatial). **The "fix the encoder" path is now fully characterized — see Blockers/02-REFRAME.**
- **DECISION (2026-06-24): F4 — fix + retrain the upstream encoder.** Professor signed off reversing `DEC-frozen-baseline` (scoped exception, recorded in PROJECT.md): apply the minimal encoder architecture fix in the parent `MultiDCP_CheMoE_pdg` project (`linear_encoder_flag=True` / replace `repeat(1,1,32)` with `Linear(1→32)`) + retrain, gate the new checkpoint on a basal-sensitivity check, then re-cache region DE here + re-run Halt Gate 3. **NEXT: scope the parent-project fix + retrain (exact edit, training entrypoint, env, data, compute cost, validation gate) BEFORE launching — do not blind-launch a cross-project multi-hour run.** New workstream in the parent project (`/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/`, own CLAUDE.md/planning).
- **Resume with:** `/gsd-discuss-phase 2 --update` (re-scope Phase 2 to the encoder fix per 02-REFRAME.md) — start with the diagnostic. Review artifacts: `02-REFRAME.md`, `HALT_REASON.md`, `results/tables/P2_apap_validation.md`, `02-VERIFICATION.md`, `02-02-SUMMARY.md`, `scripts/cache_region_de.py` (basal pipeline to inspect).
