---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: halted
stopped_at: "Phase 1 executed (6/6 plans); HALT GATE 2 FIRED on 01-06 → stop-and-REFRAME (D-02). Phase 2 BLOCKED pending reframe."
last_updated: "2026-06-23T04:30:00Z"
last_activity: 2026-06-23
progress:
  total_phases: 8
  completed_phases: 1
  total_plans: 10
  completed_plans: 9
  percent: 90
---

# STATE: Spatial Cross-Species Toxicity Prediction (MultiDCP-CheMoE)

> Living memory across sessions. Read this first.

## Project Reference

- **Project:** Spatial Cross-Species Toxicity Prediction (MultiDCP-CheMoE)
- **Root:** `/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests/` (subdir of umbrella repo `GEX_vs_chemical_experiments`; no own `.git`).
- **Core value:** Does a predicted, region-resolved molecular response signature predict organ-specific drug toxicity better than chemical structure alone, and does that signal translate across species (rodent → human)?
- **Current focus:** Phase 01 — eda-the-bracket
- **Isolation note:** This is a standalone GSD project. NEVER read/write `/raid/home/joshua/.planning` (separate, halted "liver" v0.5 project).

## Current Position

Phase: 01 (eda-the-bracket) — EXECUTED, **HALTED (Halt Gate 2 fired)**
Plan: 6 of 6 (all executed)

- **Phase:** 1
- **Plan:** 01-06 COMPLETE (run_p1_eda.py + results/tables/P1_eda.md produced on real data)
- **Status:** **HALTED — Halt Gate 2 FIRED → stop-and-REFRAME (D-02). Phase 2 is BLOCKED.**
- **Progress:** `[##################  ] P1: 6/6 plans executed; outcome = halt/reframe`

**Halt Gate 2 result (leakage-corrected):** structure floor 0.611, leakage-free drug-grouped measured ceiling 0.434, gap −0.177, 95% CI [−0.316, −0.033] → FIRES. Robust headline finding: the Wang/Li-style benchmark is **+0.31 AUROC drug-leakage-inflated** (profile-level 0.912 leaky vs 0.605 drug-disjoint); honest measured ≈ structure at the fair level; the drug-aggregated gate is underpowered (38 negative drugs). See `results/tables/P1_eda.md` + `.planning/phases/01-eda-the-bracket/HALT_REASON.md`.

**Next action:** REFRAME (do NOT execute Phase 2). Discuss the reframe — center on (1) the benchmark drug-leakage finding, (2) a properly powered drug-disjoint comparison (expand negatives), (3) the unit of analysis (profile-level with drug-disjoint splits).

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

- **Phase 2 BLOCKED — Halt Gate 2 fired (D-02).** Do not start Phase 2 model wiring until the reframe is decided/approved. See `HALT_REASON.md`.

## Session Continuity

- **Last activity:** 2026-06-23
- **Stopped at:** Phase 1 fully executed (6/6 plans); Halt Gate 2 FIRED on 01-06 → reframe (D-02, user-confirmed).
- **Resume with:** REFRAME discussion (NOT Phase 2). Use `/gsd-discuss-phase` or a milestone reframe to address: benchmark drug-leakage finding, powering a drug-disjoint comparison (expand negatives), and the unit of analysis. Phase 1 deliverable: `results/tables/P1_eda.md`; halt record: `.planning/phases/01-eda-the-bracket/HALT_REASON.md`.
