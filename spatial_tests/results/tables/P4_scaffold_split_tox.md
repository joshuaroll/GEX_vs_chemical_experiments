# Scaffold-disjoint vs drug-disjoint toxicity (does structure's edge survive?)

Same features, two CV schemes. drug-disjoint = StratifiedKFold over drugs (scaffolds can span folds). scaffold-disjoint = StratifiedGroupKFold grouped by Bemis-Murcko scaffold (acyclic = singletons). 5 folds x 5 seeds, AUROC.

| Organ | n (scaffolds) | feature | drug-disjoint AUROC | scaffold-disjoint AUROC | drop |
|---|---|---|---|---|---|
| liver | 492 (390) | structure | 0.714±0.017 | 0.712±0.003 | +0.002 |
|  |  | predicted DE | 0.671±0.014 | 0.664±0.013 | +0.007 |
|  |  | both | 0.705±0.013 | 0.701±0.018 | +0.004 |
| kidney | 317 (251) | structure | 0.654±0.010 | 0.664±0.018 | -0.009 |
|  |  | predicted DE | 0.599±0.013 | 0.593±0.014 | +0.006 |
|  |  | both | 0.660±0.017 | 0.669±0.009 | -0.009 |

## structure − predicted DE gap, by split (does structure's edge shrink?)
| Organ | gap drug-disjoint | gap scaffold-disjoint |
|---|---|---|
| liver | +0.043 | +0.048 |
| kidney | +0.055 | +0.071 |

## Reading
- If structure drops MORE than predicted DE under scaffold-disjoint (gap shrinks), structure's edge was partly analog memorization, and expression is relatively more mechanistic. If both drop together (gap stable), structure's edge is real generalization.

