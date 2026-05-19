# Multi-Head MultiDCP DILI (parallel branch)

## What this is

End-to-end DILI predictor combining three independent encoder→prediction pathways (chem+cell→Hill scalar, chem+cell→GEX vector, chem→frozen MolFormer embedding). Test whether combining the three pathways beats any single pathway on scaffold-novel DILIst test drugs.

## Source of truth

`/raid/home/joshua/projects/0_project_documents/multihead_multidcp_dili_three_pathway_05192026.md` (design doc, committed at f921223)
`/raid/home/joshua/projects/0_project_documents/multihead_multidcp_dili_implementation_plan_05192026.md` (this plan)

## Relationship to v0.5

Parallel branch, sibling subdir to `dili_downstream/`. No code crossover by default. Reused artifact: `dili_canonical.csv` (1,118 DILIst drugs from v0.4 P1).

## Hard rules

(See `multihead_dili/CLAUDE.md` for the full list.)

## Halt gates (4 total)

1. (P1) MODEL_DOSE dev RMSE not beating predict-mean baseline
2. (P2) MODEL_GEX dev predicted-vs-measured Pearson < 0.2
3. (P4) embed-only AUROC < 0.55 on random split
4. (P5) all-three (variant 7) AUROC not ≥ 0.01 over best single-pathway → reframe paper claim
