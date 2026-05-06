# GEX vs Chemical Experiments

**Author:** Joshua Rollins (CUNY Graduate Center, Lei Xie lab)
**Repo:** https://github.com/joshuaroll/GEX_vs_chemical_experiments

This repository hosts downstream evaluations that compare **predicted gene-expression (GEX) signatures** from the MultiDCP / MultiDCP-CheMoE family of models against **chemistry-only baselines** across multiple toxicity, ADME, PK, and binding endpoints.

The central question across every arm is the same: **does the predicted-signature path beat structure-only on out-of-distribution drugs, and where measured signatures are available, does it close the gap to the upper bound?**

## Active arms

| Arm | Status | Endpoint | Subdirectory |
|---|---|---|---|
| **dili_downstream** | scaffolded; Phase 1 not yet started | DILIst binary liver toxicity | [`dili_downstream/`](./dili_downstream/) |

## Planned arms (v0.5+)

| Arm | Endpoint | Notes |
|---|---|---|
| `kidney_downstream` | DIRIL | Separate v0.5 GSD spec |
| `cardio_downstream` | DICTrank | Separate v0.5 GSD spec |
| `neuro_downstream` | Zhao et al. / SIDER | Separate v0.5 GSD spec |
| `adme_downstream` | TDC ADME (HIA, Pgp, Bioavailability, BBB, CYP2C9, Caco2, Lipophilicity, Solubility, HydrationFreeEnergy) | Chemistry-only negative control |
| `pk_multitask` | fu, Vd, cl via Amitesh's PBPK | Multitask regression |
| `viability_downstream` | Cell viability across MultiDCP cell lines | Closely tied to DILI viability ablation |
| `pli_binding` | Protein-ligand affinity | Yoojean's PLI codebase + new chem encoder swap |

## Workspace coordination

This repository is one of three coordinated workspaces:

- **Planning:** `/raid/home/joshua/.planning/` — milestone definitions, roadmaps, GSD state. **Workspace-level**, covers all arms.
- **Code (this repo):** `/raid/home/joshua/projects/GEX_vs_chemical_experiments/` — the canonical home for all experiment code, configs, manifests, and per-arm results.
- **Project notes:** `/raid/home/joshua/projects/0_project_documents/` — source-of-truth design docs and progress trackers for each arm.

| Concern | Lives in |
|---|---|
| Experiment design docs (the "what and why") | `0_project_documents/` |
| Phase plans and gates (the "how and when") | `.planning/` |
| Code + manifests + results (the "what got built") | this repo |

## Hard rules (apply across all arms)

1. **Real data only** — no mocking, stubbing, or synthetic labels.
2. **DE rule** — all GEX-derived features computed on differential expression `treated − diseased`, top-k by `|true_DE|`. Raw expression as a feature or metric is forbidden.
3. **Compound-aware splits** — Murcko scaffold split as the default per arm; cluster split as robustness; scaffold-similarity-based exclusion of test scaffolds from upstream training.
4. **CUDA hygiene** — `--gpu` argparse before `import torch`. Auto-detect via `nvidia-smi`; **always leave one GPU free** (shared-CPU constraint on Joshua's box).
5. **Seeds** — minimum 3 per cell-condition, mean ± std.
6. **Atomic commits** — one phase per commit. Format: `{arm}: P{n} — {short description}`.
7. **MANIFEST.md per arm** — every arm pins its data SHA256s, code SHAs, and external resource versions.

## Quick start

```bash
# Inspect available arms
ls /raid/home/joshua/projects/GEX_vs_chemical_experiments/

# Enter a specific arm
cd dili_downstream/
cat README.md
cat MANIFEST.md
```

For the active DILI arm, see [`dili_downstream/README.md`](./dili_downstream/README.md).
