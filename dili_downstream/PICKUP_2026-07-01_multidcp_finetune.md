# PICKUP — finetune a pretrained MultiDCP on the Li/Tong benchmark, get comparison numbers

**Date:** 2026-07-01
**Task (user intent, verbatim):** "use a pretrained MultiDCP model and finetune it on the same
dataset to see what our comparison numbers are." The dataset = the Li/Tong 2020 exact 6,000-profile
DILI benchmark (now acquired + reproduced). "Comparison numbers" = our recurring structure vs
expression vs both, now with a *pretrained-and-finetuned* MultiDCP as the expression arm, anchored
against the 0.798 benchmark.

## PROGRESS — session 3 (2026-07-02): honest verdict + rigor pass

**VERDICT (v2, paper-consistent protocol): a pretrained MultiDCP backbone does NOT flip the
structure ≥ expression finding.** On the honest per-drug metric and leakage-corrected splits,
structure (ECFP4) ≈ finetuned MultiDCP (~0.58–0.60 drug-level AUROC, overlapping error), and true
measured expression is at chance (0.46 drug / 0.50 scaffold). The earlier "structure collapses to
0.37, expression wins 0.69" was a **profile-level pseudoreplication artifact** (median 4 / mean 8.3
profiles per drug; Vorinostat = 14% of profiles). Full table: `results/tables/P_tox_finetune_multidcp.md`
+ `P_tox_finetune_multidcp_v2.csv`. The winning arm (tuned MultiDCP) is itself a SMILES model
(SMILES→predicted GEX→DILI; only SMILES varies per drug on disjoint splits), so it's structure-vs-structure.

**Rigor pass (two independent reviews — code + methodology):**
- Code review: NO train/test leakage; group-disjoint inner val, metric guards, anchor indexing,
  MultiDCP/ChemBERTa training parity all correct; dose one-hot faithful to original data_utils. One
  conditional finding (measured arm zero-fills NaN) verified not to bite (100% join) + loud guard added.
- Methodology review: drug-level metric decisive (dissolved the reversal); **dose IS a confound**
  (top >16µM bin 94.5% positive) — I was wrong to fully concede this earlier; the claim must be worded
  "GEX-pretrained SMILES bottleneck ≈ fingerprints," not "expression beats structure."

**v3 DONE — null FIRMED (fair baseline + controls).** Drug-level AUROC, disjoint splits: all real
predictors cluster ~0.57–0.60 (ECFP4, ChemBERTa frozen/ft, MultiDCP) — statistically indistinguishable;
measured expression + cell_dose_only + dist_shift at chance. **Decisive ablation:** `chemberta_meta`
(SMILES+cell+dose direct) == `tuned_E` (same inputs via predicted-GEX bottleneck), 0.603 vs 0.600 drug /
0.575 vs 0.599 scaffold → GEX bottleneck adds nothing over the inputs used directly; retires even the
narrow "bottleneck > fingerprints" claim. Permutation control ~0.5 everywhere (no leakage). Final table:
`results/tables/P_tox_finetune_multidcp.md`. **Gate 0 (single-cell scRatio follow-up): sci-Plex FAILS**
(30 DILIst-labeled drugs, 23+/7−); Tahoe-100M only option, parked (low prior). Original run details:

**v3 config (firm up the null) — completed bg jobs:**
- Main (GPU 3, `--tag v3`): adds fair pretrained-structure arms `chemberta_frozen`/`chemberta_ft`
  (SMILES-only) and `chemberta_meta` (SMILES + same cell+dose one-hot MultiDCP gets → isolates the GEX
  bottleneck), plus `cell_dose_only` confound floor; per-resample AUROCs saved (`..._v3_per_resample.csv`).
  Smoke: cell_dose_only ≈ chance (0.50 drug-level) — dose confound doesn't generalize drug-disjoint.
- Permute control (GPU 4, `--tag permute`): drug-level label shuffle; all arms should return ~0.5.
- ChemBERTa arm added to harness: `load_chemberta`/`chemberta_embed`/`train_chemberta` (DeepChem/ChemBERTa-77M-MLM,
  3.4M params, hidden 384, offline-cached), same head + lr scheme as the MultiDCP finetune.

**Flow-based method to compare (user request): scRatio** (Antipov, Palma … Theis; arXiv 2602.24201;
github theislab/scRatio). Flow-matching **density-ratio estimation** between distributions, for
single-cell genomics (treatment effect, batch correction, ComboSciplex drug efficiency). Bridge to
DILI: density ratio = distributional perturbation shift → an alternative expression feature. BLOCKER:
needs single-cell (distributional) data; the Li/Tong benchmark is bulk L1000 MODZ (one vector/profile),
so scRatio can't run on it directly. Direct-comparison paths (ranked): (A) acquire a single-cell
drug-perturbation dataset overlapping DILIst (sci-Plex/ComboSciplex/tox atlas) → per-drug scRatio shift
feature → same DILI harness (same head, drug-disjoint, drug-level AUROC) vs structure + MultiDCP; (B)
bulk proxy: treat each drug's L1000 profiles as a sample, treated-vs-DMSO ratio; (C) cheap classifier-trick
density-ratio baseline on MODZ to test whether the ratio framing adds over raw MODZ before investing in
flow matching. Rec: scope (A) as a follow-up milestone (this is the single-cell direction); (C) is a
cheap in-harness sanity check.

---

## PROGRESS — session 2 (2026-07-01, tox-finetune build + run launched)

**Decisions locked (user, via AskUserQuestion):** reading = **tox-finetune (E/F)** — attach a DILI head
to the pretrained MultiDCP and finetune end-to-end. Backbone = **the ORIGINAL pretrained MultiDCP**
(2023 release, not the 2025 retrains): `L5_split1_05032023.pt`, single backbone + 3 finetune seeds
for variance.

**Built + verified this session:**
- Pretrained ckpt is a `MultiDCP_AE` wrapper (`.multidcp` + `linear_final` + `decoder_linear`),
  **Transformer** cell encoder (`linear_encoder_flag=False`), **6-way one-hot dose** (`pert_idose_embed`
  6→4), num_gene=978. Loads via `initialize_model_registry()` (from `multidcp_ae_utils`) + update
  `{num_gene:978, pert_idose_input_dim:6, linear_encoder_flag:False}`. Smoke forward → `[8,978]` finite,
  mean≈0 std≈1.1 (predicted MODZ z-scores). `MultiDCP_AE(...)(cell, drug, gene, mask, dose, job_id='perturbed')`
  → `(pred[B,978], cell_hidden)`.
- Faithful encoding: basal = `adjusted_ccle_tcga_ad_tpm_log2.csv.loc[cell_id]` (978, CCLE/TCGA log2-TPM,
  **input context, NOT subtracted**); dose = 6-way canonical one-hot in lex-sorted string order, Wang/Li
  dose→nearest canonical (log10); **predicted feature = MultiDCP_AE output [B,978] directly = predicted DE**
  (LINCS L5 MODZ is z-scored differential natively; honors the DE rule, sidesteps 977-vs-978 symbol mismatch).
- Cached input table: `scripts/build_wangli_multidcp_inputs.py` → `data/processed/wangli_multidcp_finetune.npz`
  = **5,141 profiles** (of 5,517 Phase-1 / 6,000 orig; dropped 376 on 12 cells absent from CCLE basal incl.
  PHH, HEK293T). Fields: smiles, cell_basal[N,978], dose_onehot[N,6], scaffold (Murcko), label, usage,
  time_h, measured_modz[N,978] (joined 5141/5141). 57 cells, 621 compounds, 433 scaffolds, 3121/2020 pos/neg,
  4127 train / 1014 test. dataset-verification PASS_WITH_WARNINGS (npz object-array parse limit only).
  **Caveat:** ~58% of profiles are 6H exposures; the pretrained model trained on 24H only (domain gap;
  finetuning adapts).
- Harness: `scripts/tox_finetune_multidcp.py`. Variants: structure(ECFP4 2048) · measured(L5 MODZ) ·
  frozen_pred(frozen MultiDCP out) · **tuned_E**(finetune engine+head) · tuned_F(+anchor to frozen) ·
  both(ECFP4⊕tuned engine out). Head = BatchNorm→128→ReLU→Dropout(0.3)→1; engine lr 1e-4 / head lr 1e-3;
  BS=16; internal-val early stop (AUROC). Splits: `test` (Wang/Li profile Training/Test → vs 0.798),
  `drug` (compound-disjoint 5-fold GroupKFold), `scaffold` (Murcko-disjoint). 3 seeds.

**Run launched (background, GPU 1):** `--splits test,drug,scaffold --variants <all 6> --seeds 3 --folds 5
--max-epochs 60 --patience 10`. Log: `results/tables/tox_finetune_run.log`; CSV out:
`results/tables/P_tox_finetune_multidcp_full.csv`. ETA ~10-13 h (grouped-CV is the bulk).

**HEADLINE FINDING (profile-level Test split, static variants):** structure(ECFP4) **0.996** ·
measured 0.764 · frozen_pred 0.872(±0.081). structure≈1.0 is **leakage, not signal** — the profile-level
split lets the same compound appear in train+test, and ECFP4 memorizes each drug's label. So the
profile-level "compare to 0.798" is meaningful only for the measured/expression arms; the honest
structure-vs-expression verdict comes from the **drug-/scaffold-disjoint** splits (pending in the run).

**Resume:** if run finished → read `results/tables/P_tox_finetune_multidcp_full.csv` +
`...run.log`, write `results/tables/P_tox_finetune_multidcp.md` (structure vs expression vs both, 3 splits,
vs 0.798, with the profile-level leakage caveat), update `.planning/STATE.md`, commit
(`dili_downstream: tox-finetune pretrained MultiDCP on Wang/Li — comparison numbers`). If crashed → check
log tail; likely spots: `both` variant head in_dim=978+2048, GroupKFold edge cases.

---

## State — what is already done and verified (this session)

- **Exact benchmark dataset acquired** (public: GitHub `TingLi2016/L1000_DILI` @`010361f` +
  GEO GSE92742 standard Level-5 MODZ; NO Synapse). Extracted matrix:
  `dili_downstream/data/processed/wangli_6000_landmark.npz`
  = `X [6000,978] float32`, `sig_ids`, `gene_ids` (Entrez, gctx row order), `label` (3568/2432),
  `usage` (Training 4800 / Test 1200). finite, std 1.885, 6000/6000 sig_ids recovered.
  sha256 `07aa3e69…1ca7`.
- **0.798 reproduced to 3 decimals** on their Test split with their checkpoint
  (`data/raw/L1000_DILI/optimized_model.h5`, numpy ELU forward, raw MODZ no scaling):
  AUROC 0.7976 / Sens 0.839 / Spec 0.603 / Acc 0.743 → **halt-gate-1 PASS**. The old 0.51 was the
  Bayesian-vs-standard MODZ variant; resolved. See `results/tables/P2_wangli_standard_modz_reproduction.md`.
- Scripts (committed `fc769a8`): `scripts/download_gse92742.sh`, `scripts/extract_gse92742_6000.py`,
  `scripts/reproduce_wangli_standard_modz.py`. The 23 GB gctx was deleted after slicing; re-acquire
  via the download+extract scripts.

## The finetune task — decisions to confirm with the user FIRST

The Li/Tong 6,000 are **measured** profiles (drug × cell × dose → 978-gene MODZ). A pretrained
MultiDCP maps **(SMILES, cell-basal, dose) → predicted 978-gene expression**. So "finetune MultiDCP
on this dataset" has two readings — confirm which (or do both):

- **(E/F) tox-finetune (most likely intent):** attach a DILI head to a pretrained MultiDCP and
  finetune end-to-end on `(SMILES, cell, dose) → DILI label`, starting from a real pretrained
  backbone (vs the from-scratch organ engine we used in `spatial_tests/scripts/tox_tuned_signature.py`,
  which HURT: structure>frozen>tuned). Question this answers: does a *pretrained* backbone change
  that verdict?
- **(predicted-DE) expression-finetune:** continue MultiDCP's expression training on these 6,000
  profiles, then use predicted DE as the DILI feature — a predicted-vs-measured test on the benchmark.

**Comparison numbers to report** (same framework, on the Li/Tong data):
structure (ChemBERTa/ECFP4) · expression (finetuned-MultiDCP) · both — AUROC, on **two splits**:
(1) their profile-level Test split → compare directly to 0.798; (2) our honest **drug-disjoint**
(and scaffold-disjoint) → the leakage-corrected number. Anchor = measured-GEX DNN 0.798 (their
`optimized_model.h5`, already reproduced).

## Pretrained MultiDCP checkpoints (candidates)

`/raid/home/joshua/projects/MultiDCP/trained_models/` — use an **L5 (Level-5 expression)** one:
`L5_split1_01302025.pt`, `L5_split2_08272025_seed2.pt/seed3.pt`, `L5_split3_333.pt`
(also `basic_split*.pt`, `baseline_split*_seed*.pt`). Confirm which split/seed and its registry
config (num_gene, linear vs transformer encoder) matches the loader. The engine class is
`multidcp.MultiDCPOriginal` (`/raid/home/joshua/projects/MultiDCP/MultiDCP/models/multidcp.py`);
forward = `model(input_drug, input_gene, mask, input_cell_gex, input_pert_idose)` → predicted
treated [B,978]. Reuse the finetune harness pattern from
`spatial_tests/scripts/tox_tuned_signature.py` (mini-batch backprop, BS=16 to avoid the gene-attention
OOM; head = BatchNorm→128→1; engine lr 1e-4, head lr 1e-3; anchor variant = tuned_F).

## New plumbing required (the real work)

MultiDCP needs, per profile: (1) **SMILES** for the drug — resolve from the xlsx `CompoundName`
via `dili_downstream/src/data/` resolver / DrugBank cascade (`resolve_smiles.py`); (2) **cell basal
GEX** (`input_cell_gex`, 978-d) for each `cell_id_y` in the 6,000 (≈69 cell lines) — need an
untreated/basal profile per cell line (CCLE / LINCS controls; check what the organ_train build used,
`spatial_tests/scripts/build_organ_dataset.py` pulls x2 control from
`/raid/home/joshua/data/L1000_and_CMap/final_data/processed_data_978modz_detplate_09012024.h5`);
(3) **dose** from `pert_idose`. Gene order for predicted output must match the 978 landmark order
used everywhere (gctx row order; our `wangli_6000_landmark.npz` `gene_ids` is that order).

## Env + gotchas

- Env: `conda activate dili_v04_env`. **numpy is 2.2.6** (bumped by unimol_tools; pytdc conflicts —
  see memory `dili-v04-env-numpy2-bump`). cmapPy is NOT installed and fragile under numpy2 — we
  slice gctx with h5py directly.
- Raw MODZ, **no scaling**, is the Li/Tong preprocessing (scaled gave 0.755 vs 0.798). For the head
  use BatchNorm (as in tox_tuned) so scaling is learned.
- Profile-level vs drug-level: their 0.798 is **profile-level** (drug identity leaks across
  train/test). Our Phase-1 bracket showed it drops to ~0.67 drug-level / ~0.55 scaffold-level on a
  proxy corpus. Report BOTH so the finetuned number is interpretable.
- GPU: `--gpu` before `import torch`; leave one GPU free. Long finetunes → background + watcher.
- Prior context (why this matters): across binary + ordinal severity, frozen/tuned/fused, 9 encoders,
  6 probes, gene expression never beat structure (spatial_tests `results/P4_writeup.md`). Tox-tuning
  a *from-scratch* organ engine HURT. This experiment tests whether a *pretrained* backbone + the
  *benchmark's own data* changes that — the honest, faithful version of conditions E/F.

## First concrete steps

1. Confirm with user: E/F tox-finetune vs expression-finetune (or both); which pretrained ckpt.
2. Build the per-profile input table for the 6,000: SMILES (resolve), cell basal (per cell_id), dose.
   Verify SMILES resolution rate + basal coverage; report drops.
3. Load pretrained L5 checkpoint into `MultiDCPOriginal`; smoke-test a forward on 8 profiles.
4. Finetune (adapt `tox_tuned_signature.py`): structure / finetuned-MultiDCP / both, profile-level
   Test split + drug-disjoint + scaffold-disjoint, ≥3 seeds. AUROC vs 0.798.
5. dataset-verification on any new processed artifact; commit scripts + a results md; update this
   pickup / STATE.

## Key paths

- data: `dili_downstream/data/processed/wangli_6000_landmark.npz`,
  `data/raw/L1000_DILI/` (xlsx, optimized_model.h5), `data/raw/GSE92742/` (gene_info, sig_info).
- pretrained: `/raid/home/joshua/projects/MultiDCP/trained_models/L5_*.pt`
- engine code: `/raid/home/joshua/projects/MultiDCP/MultiDCP/models/multidcp.py` (+ `utils/data_utils.py`).
- finetune harness to copy: `spatial_tests/scripts/tox_tuned_signature.py`.
- basal source used before: `spatial_tests/scripts/build_organ_dataset.py`.
- repo branch: `spatial/gsd-bootstrap` (dili_downstream code lives in the umbrella repo; v0.5
  history is on `main` — consider cherry-picking `fc769a8` there).
