# Three-way feature comparison, split by body region

Core question per organ: does measured gene expression ADD toxicity signal over chemical structure alone? Drug-level fair evaluation (one row per drug; 5-fold stratified CV x 5 seeds; L2 logistic regression, balanced). Join: InChIKey-14.

## Headline -- AUROC by organ and feature set

| Organ | n (pos/neg) | structure | expression | structure+expression | **+expr lift (both - structure)** | 95% CI |
|---|---|---|---|---|---|---|
| liver | 234 (191/43) | 0.575 | 0.456 | 0.527 | **-0.048** | [-0.118, +0.024] |
| kidney | 203 (114/89) | 0.680 | 0.493 | 0.659 | **-0.005** | [-0.049, +0.038] |
| heart | 221 (166/55) | 0.585 | 0.514 | 0.597 | **+0.042** | [-0.024, +0.109] |

Lift CI excluding 0 => expression adds signal over structure on that organ (paired bootstrap, same drugs, seed-0 OOF, 10000 resamples).

## Brain (SIDER) -- CANNOT RUN on current data
SIDER is keyed by PubChem CID with no SMILES/InChIKey/name and no MedDRA SOC column on disk; LINCS profiles carry no CID. No shared identifier => no join. Needs (1) a CID->structure (InChIKey) resolver and (2) a MedDRA SOC hierarchy file to define the nervous-system label. Reported, not silently dropped.

## Method notes / caveats
- Expression = per-drug MEAN of that drug's measured LINCS L1000 DE profiles (treated-diseased).
- Drug-level evaluation is drug-disjoint by construction (one row per drug).
- class_weight=balanced; AUROC is threshold-free. Liver/heart are positive-skewed; kidney is best-balanced.
- Liver/heart SMILES via name->dili_canonical->drugbank cascade; kidney has inline SMILES.
- This uses MEASURED expression. Predicted expression (MultiDCP) is the separate follow-on source.

## Figure
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/three_way_comparison/three_way_by_organ.png` -- grouped AUROC bars per organ
