# DILI Downstream — Liver Toxicity Prediction (v0.4)

**Milestone:** v0.4 (parallel to v0.3 MultiDCP JSD Expert Divergence)
**Started:** 2026-05-05
**Status:** scaffolded; Phase 1 not yet started

## Quick links

- **Source-of-truth doc:** [`CLAUDE_DILI_DOWNSTREAM.md`](./CLAUDE_DILI_DOWNSTREAM.md)
- **Roadmap:** `/raid/home/joshua/.planning/ROADMAP_v04.md` (workspace planning, separate repo)
- **Progress tracker:** `/raid/home/joshua/projects/0_project_documents/v04_dili_downstream_progress.md`
- **Parent project plan:** `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/CLAUDE.md`
- **Umbrella repo:** [`GEX_vs_chemical_experiments`](../README.md) on GitHub: https://github.com/joshuaroll/GEX_vs_chemical_experiments
- **Manifest:** [`MANIFEST.md`](./MANIFEST.md)

## Headline grid

6 conditions (A/B/C/D/E/F) × 4 chemical encoders (ChemBERTa, MolFormer, GIN, UniMol) × 3 seeds = **72 downstream classifier runs** on the primary unified DILI-aware scaffold split, plus a TDC-DILI secondary slice and a cluster-split robustness slice.

## Directory map

```
data/
  raw/{DILIst,DILIrank}/         # FDA NCTR downloads
  processed/                     # dili_canonical.csv, signature caches, encoder caches
  splits/                        # unified_dili_aware_{scaffold,cluster}.json + tdc_dili_scaffold.json
src/
  splits/                        # build_unified_split.py + tdc_split.py
  models/upstream/               # SHA-pinned clones from MultiDCP/MultiDCP/models/
  signatures/                    # multidcp_signature.py, chemoe_signature.py, lincs_signature.py
  encoders/                      # chemberta_wrapper.py, molformer_wrapper.py, gin_wrapper.py, unimol_wrapper.py
  classifiers/                   # concat_classifier.py + upstream_dili_head.py (E/F joint training)
  ablations/                     # viability_mask.py, permutation_diagnostic.py
  train_upstream.py              # Phase 3 launcher (B/C/E/F × 3 seeds)
  train_classifier.py            # Phase 5 launcher (6 × 4 × 3 grid)
  eval.py
configs/
  upstream/                      # B_baseline.yaml, C_chemoe.yaml, E_baseline_dili.yaml, F_chemoe_dili.yaml
  downstream/                    # A..F per-condition configs
  encoders/                      # one yaml per chemical encoder
  sweep_classifier_hp.yaml       # Phase 9 wandb sweep
scripts/
  download_dilist.py             # FDA NCTR fetch
  auto_select_gpus.py            # nvidia-smi parser, leave-one-free
  env_setup.sh                   # clone mdcp_env -> dili_v04_env + extra deps
results/
  tables/                        # per-phase markdown tables
  figures/
tests/
  test_data_paths.py             # only sanity test in pipeline; runs before every phase
```

## Hard rules (from §0 of source-of-truth doc)

1. **Real data only** — no mocking, stubbing, synthetic labels, placeholder DataLoaders.
2. **DE rule** — all GEX-derived features computed on differential expression `treated − diseased`, top-k by `|true_DE|`. Raw expression as a feature or metric is forbidden.
3. **Compound-aware splits** — Murcko scaffold default, cluster split as robustness; **scaffold-similarity-based exclusion** of test scaffolds from upstream training (Option a).
4. **CUDA hygiene** — set `CUDA_VISIBLE_DEVICES` via argparse before `import torch`.
5. **Seeds** — minimum 3 per cell, mean ± std.
6. **WandB** — `joshroll/DILI_Downstream_v04`, run group `dili_downstream`.
7. **Atomic git commits** — one phase per commit. Format: `dili_downstream: P{n} — {short description}`.
8. **GPU policy** — auto-detect via `nvidia-smi`, **always leave one GPU free** (shared CPU constraint on this box).

## Decision gates

| # | Phase | Gate | Action if failed |
|---|---|---|---|
| 1 | 2 | `\|D_DILI_test \\ D_PDG\| < 30` | Halt; re-discuss with Joshua |
| 2 | 4 | Predicted-vs-LINCS DE Pearson < 0.3 averaged | Halt; flag |
| 3 | 5 | B doesn't beat A by ≥ 0.02 AUROC mean | Halt; consult before Phase 6+ |
| 4 | 5 | F doesn't beat C on drug-and-scaffold-novel slice by ≥ 0.01 AUROC | Reframe paper claim |
| 5 | 7 | Gate-permutation ΔAUROC ≮ control-layer ΔAUROC | Reframe CheMoE claim honestly |
