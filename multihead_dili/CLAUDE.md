# multihead_dili (multi-head MultiDCP DILI — parallel branch)

This branch implements the three-pathway DILI prediction design (see `0_project_documents/multihead_multidcp_dili_three_pathway_05192026.md`).

**Relationship to v0.5:** Sibling subdir, no code crossover. v0.5 code lives in `../dili_downstream/`. Reused artifacts (symlink or copy): `dili_canonical.csv`.

**Source-of-truth doc:** `/raid/home/joshua/projects/0_project_documents/multihead_multidcp_dili_three_pathway_05192026.md`

## Hard rules

1. **Real data only.** No mocking, no synthetic labels, no stubbed loaders.
2. **DE rule.** All GEX-derived features computed on differential expression (treated − diseased), top-k by |true_DE|.
3. **Leakage discipline (load-bearing).** Murcko-scaffold + drug_name filter applied to Stage-1 training data BEFORE training. Test scaffolds from DILIst never seen by MODEL_DOSE or MODEL_GEX.
4. **CUDA hygiene.** `--gpu` argparse before `import torch`. Always leave one GPU free.
5. **Atomic commits.** One commit per task. Format: `multihead_dili: P{n} — {short description}`.

## Architecture (one-liner)

Three independent models — MODEL_DOSE (MultiDCP-AE on E-Hill) + MODEL_GEX (MultiDCP-AE on LINCS) + MODEL_EMBED (frozen MolFormer) — features concatenated for a 7-way pathway ablation DILI classifier.

## Environment

```bash
conda activate dili_v04_env
cd /raid/home/joshua/projects/GEX_vs_chemical_experiments/multihead_dili
```

## WandB

Project: `joshroll/MultiDCP_multihead_dili`
Run groups: `model_dose`, `model_gex`, `multihead_dili`
