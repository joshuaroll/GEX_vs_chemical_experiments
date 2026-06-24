---
phase: 02-multidcp-wiring-tox-head
plan: 03
subsystem: spatial-tox-head
tags: [WIRE-02, tox-head, concat-mlp, smoke-train, condition-A, wandb, cuda-hygiene]
requires:
  - src.spatial.region_combiner.AttentionPoolCombiner (d=10716)
  - src.spatial.eda.fingerprints.smiles_to_ecfp4
  - src.spatial.eda.labels.load_dilirank
  - src.spatial.eda.smiles_join.join_smiles_cascade
provides:
  - src.spatial.tox_head.ToxHead (concat-MLP organ-tox head)
  - src.spatial.tox_head.ToxHeadOutput (NamedTuple: logit, attn_weights)
  - scripts/smoke_train_condA.py (condition-A structure-only smoke-train driver)
affects:
  - Phase 4 (full multi-condition training will reuse ToxHead + the training loop)
  - Phase 6 (attn_weights passthrough feeds regional-attribution interpretability)
tech-stack:
  added: []
  patterns:
    - nn.Module + NamedTuple-output container (mirrors RegionCombinerOutput)
    - per-condition zero-tensor channel masking (no special-casing)
    - CUDA hygiene: CUDA_VISIBLE_DEVICES set before import torch (Hard Rule 5)
key-files:
  created:
    - src/spatial/tox_head.py
    - scripts/smoke_train_condA.py
    - results/tables/P2_smoke_condA.md
  modified: []
decisions:
  - "Chem channel = ECFP4 (2048-d) for the smoke-train only (RESEARCH A1 / Claude's Discretion); ChemBERTa-vs-ECFP4 headline choice deferred to P4."
  - "wandb: entity=joshroll, project=DILI_spatial_xspecies, group=spatial_p2_smoke (the 'joshroll/...' form is entity/project; wandb project names cannot contain '/')."
  - "Single logit shape (B,) via squeeze(-1); BatchNorm1d in the MLP requires batch >= 2 so trailing singleton batches are skipped in the smoke-train loop."
metrics:
  duration: "~10 min"
  completed: 2026-06-24
  tasks: 2
  files: 3
---

# Phase 2 Plan 03: ToxHead concat-MLP + condition-A smoke-train Summary

WIRE-02 concat-MLP organ-tox head (proj_gex/proj_chem/proj_dr -> 3-layer MLP -> single logit) with per-condition zero-tensor channel masking, fed by `AttentionPoolCombiner(d=10716).pooled`, plus a condition-A (structure-only, zero GEX channel) smoke-train that trains the head on real DILIrank labels with BCEWithLogits and wandb logging.

## What was built

**`src/spatial/tox_head.py`** (Task 1)
- `ToxHeadOutput(NamedTuple)`: `logit` (B,), `attn_weights` passthrough for P6 — mirrors `RegionCombinerOutput`.
- `ToxHead(nn.Module)`: `proj_gex`/`proj_chem`/`proj_dr` linear projections to a common `d_proj=256`; concat -> 3-layer MLP `Linear -> BatchNorm1d -> GELU -> Dropout -> Linear -> GELU -> Dropout -> Linear(.,1)` (CON-tox-head) -> single logit squeezed to `(B,)`.
- Per-condition zero-tensor masking: an inactive channel is fed as a zero tensor of identical shape with no special-casing — a zeroed channel projects to its layer bias (finite), so condition A (zero GEX) yields a finite logit and never a NaN. `__init__` validates dims (mirror `AttentionPoolCombiner`); `forward` shape-guards 2-D inputs.
- Turns the Wave-0 `tests/spatial/test_tox_head.py` RED -> GREEN (4 passed: zero-gex-finite-logit, combiner-feed, dr-optional, named-tuple). The test file was not edited.

**`scripts/smoke_train_condA.py`** (Task 2)
- CUDA hygiene (Hard Rule 5): `--gpu` parsed and `CUDA_VISIBLE_DEVICES` set *before* `import torch`; `nvidia-smi` auto-detect of a free GPU (least-used) that never claims the last free GPU (requires >=2 free, else CPU).
- Condition A: GEX channel = `torch.zeros(B, 10716)` (no cache, no frozen-model inference, RESEARCH Pattern 3); chem channel = ECFP4 (2048-d, `smiles_to_ecfp4`).
- Real-data only (Hard Rule 1): DILIrank binary labels + the sibling-resolved SMILES cascade (`dili_canonical` -> `drugbank`), the same wiring `run_p1_eda.py` uses — 499 drugs with valid ECFP4, 62.7% positive. No synthetic labels.
- `ToxHead` + Adam + `BCEWithLogitsLoss`, >=1 epoch, asserts the loss moves and that the loss stays finite each step (zero-GEX-path NaN guard). Logs per-epoch loss to wandb and writes an auditable `results/tables/P2_smoke_condA.md`.
- Verified run (`--epochs 5 --wandb-mode offline`): loss 0.6540 -> 0.0502 over 5 epochs, device=cuda (auto-picked a free GPU), wandb offline run synced, `smoke_exit=0`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] wandb project name with '/' rejected**
- **Found during:** Task 2 (first smoke-train run).
- **Issue:** `wandb.init(project="joshroll/DILI_spatial_xspecies")` raised `UsageError: Invalid project name ... cannot contain characters '/'`. The `joshroll/...` form in the project's WandB table is `entity/project`, not a single project string.
- **Fix:** Split into `WANDB_ENTITY="joshroll"` + `WANDB_PROJECT="DILI_spatial_xspecies"` and pass `entity=` separately to `wandb.init`.
- **Files modified:** `scripts/smoke_train_condA.py`.
- **Commit:** 77cd237 (folded into the Task 2 commit; the fix was applied before the first successful run was committed).

## Authentication Gates

None. wandb ran in offline mode (no login required); the design accepts offline logging per the plan's wandb note.

## Threat surface

No new threat surface beyond the plan's `<threat_model>`. The three registered threats are mitigated:
- T-02-07 (fabricated labels): real DILIrank labels + sibling-resolved SMILES, no synthetic data.
- T-02-08 (last-free-GPU grab): `nvidia-smi` auto-detect requires >=2 free GPUs and picks the least-used, always leaving one free.
- T-02-09 (NaN from unmasked zero channel): the forward keeps the zero-GEX path finite; `test_tox_head.py` asserts it and the smoke-train raises on a non-finite loss.

## Known Stubs

None that block the plan's goal. Two scoped-out items are intentional and documented:
- Dose-response channel zeroed this phase (`d_dr=0`); conditions G/H wire it later (out of scope per the plan).
- Chem channel = ECFP4 is a smoke-train-only choice (RESEARCH A1); the ChemBERTa-vs-ECFP4 headline encoder is a deferred P4 decision.

## Ownership boundary

Only `src/spatial/tox_head.py`, `scripts/smoke_train_condA.py`, and the generated `results/tables/P2_smoke_condA.md` were created. `src/spatial/region_signature.py`, `scripts/cache_region_de.py`, and `configs/liver_p2.yaml` (owned by 02-02) were not touched — confirmed via `git diff --name-only` over both commits.

## Commits

- cdef33c: feat(02-03): ToxHead concat-MLP + zero-channel masking (GREEN test_tox_head.py)
- 77cd237: feat(02-03): condition-A smoke-train driver (zero GEX, BCEWithLogits, wandb)

## Self-Check: PASSED

All created files exist on disk and both task commits (cdef33c, 77cd237) are in the git log.
