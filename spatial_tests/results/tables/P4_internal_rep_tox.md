# Internal-representation connection vs DE-output connection (toxicity)

Tapping MultiDCP's penultimate per-gene hidden (relu(out) [B,978,128]) pooled over genes, vs its 978-gene DE output, vs structure. Same head (L2 logreg, balanced), drug-disjoint 5-fold x 5 seeds, full tox set.

| Organ | n (pos/neg) | feature | AUROC | AUPRC | accuracy | balanced acc | F1 |
|---|---|---|---|---|---|---|---|
| liver | 492 (307/185) | structure | 0.714±0.017 | 0.778 | 0.680 | 0.667 | 0.737 |
|  |  | predicted DE | 0.671±0.014 | 0.712 | 0.663 | 0.640 | 0.730 |
|  |  | internal mean(128) | 0.638±0.013 | 0.692 | 0.634 | 0.614 | 0.704 |
|  |  | internal meanmax(256) | 0.658±0.013 | 0.700 | 0.658 | 0.642 | 0.720 |
|  |  | structure+internal | 0.708±0.015 | 0.768 | 0.683 | 0.668 | 0.741 |
| kidney | 317 (171/146) | structure | 0.654±0.010 | 0.672 | 0.618 | 0.616 | 0.645 |
|  |  | predicted DE | 0.599±0.013 | 0.688 | 0.559 | 0.559 | 0.576 |
|  |  | internal mean(128) | 0.606±0.018 | 0.662 | 0.580 | 0.581 | 0.592 |
|  |  | internal meanmax(256) | 0.621±0.017 | 0.679 | 0.577 | 0.578 | 0.592 |
|  |  | structure+internal | 0.641±0.012 | 0.654 | 0.601 | 0.598 | 0.629 |

## Lift over structure floor (AUROC, paired bootstrap)
| Organ | feature | +lift | 95% CI |
|---|---|---|---|
| liver | internal mean(128) | **-0.074** | [-0.137, -0.012] |
| liver | predicted DE | **-0.037** | [-0.095, +0.022] |
| liver | structure+internal | **-0.014** | [-0.033, +0.004] |
| kidney | internal mean(128) | **-0.056** | [-0.131, +0.019] |
| kidney | predicted DE | **-0.051** | [-0.127, +0.025] |
| kidney | structure+internal | **-0.012** | [-0.038, +0.014] |

## Reading
- internal mean/meanmax = the model's learned hidden representation (richer than the DE scalar).
- If internal >> predicted DE, the DE-output connection was discarding useful signal; if internal ~ predicted DE ~ structure, the model's representation is structure-ceilinged (all are deterministic functions of SMILES).

