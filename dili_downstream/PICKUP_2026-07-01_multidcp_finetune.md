# PICKUP — finetune a pretrained MultiDCP on the Li/Tong benchmark, get comparison numbers

**Date:** 2026-07-01
**Task (user intent, verbatim):** "use a pretrained MultiDCP model and finetune it on the same
dataset to see what our comparison numbers are." The dataset = the Li/Tong 2020 exact 6,000-profile
DILI benchmark (now acquired + reproduced). "Comparison numbers" = our recurring structure vs
expression vs both, now with a *pretrained-and-finetuned* MultiDCP as the expression arm, anchored
against the 0.798 benchmark.

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
