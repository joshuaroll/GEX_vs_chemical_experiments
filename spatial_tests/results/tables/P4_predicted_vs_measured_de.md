# Predicted DE vs JUST measured DE vs structure — same drugs (overlap set)

Only tox drugs that ALSO have measured LINCS DE in the organ corpus, so predicted and measured DE are scored on the IDENTICAL drug set. Same small head (L2 logreg, balanced), drug-disjoint 5-fold x 5 seeds. predicted DE = engine(smiles, organ_basal) - organ_basal; measured DE = per-drug mean (x1 - x2).

| Organ | n (pos/neg) | feature | AUROC | AUPRC | accuracy | balanced acc | F1 |
|---|---|---|---|---|---|---|---|
| liver | 197 (154/43) | structure | 0.533±0.024 | 0.774 | 0.686 | 0.555 | 0.797 |
|  |  | predicted DE | 0.513±0.036 | 0.767 | 0.648 | 0.543 | 0.764 |
|  |  | measured DE | 0.513±0.026 | 0.800 | 0.627 | 0.488 | 0.755 |
| kidney | 240 (132/108) | structure | 0.691±0.015 | 0.695 | 0.642 | 0.639 | 0.674 |
|  |  | predicted DE | 0.603±0.024 | 0.685 | 0.576 | 0.570 | 0.618 |
|  |  | measured DE | 0.527±0.030 | 0.580 | 0.528 | 0.523 | 0.568 |

## Reading
- predicted vs measured DE on the SAME drugs: if close, the engine faithfully carries the toxicity-relevant DE signal (good connection); if measured >> predicted, the engine loses it.
- Both vs structure: whether any DE (real or predicted) beats chemical structure for tox.

