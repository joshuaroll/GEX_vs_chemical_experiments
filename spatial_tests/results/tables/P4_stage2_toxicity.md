# Stage 2: organ toxicity from organ-trained predicted DE

expression = **multidcp** organ-engine predicted DE (in the organ mean-basal context) | structure = ChemBERTa | head = L2 logistic regression, drug-disjoint 5-fold x 5 seeds. Engines & labels overlap only for liver, kidney.

| Organ | n (pos/neg) | feature | AUROC | AUPRC | accuracy | balanced acc | F1 |
|---|---|---|---|---|---|---|---|
| liver | 492 (307/185) | structure | 0.714±0.017 | 0.778 | 0.680 | 0.667 | 0.737 |
|  |  | predicted DE | 0.671±0.014 | 0.712 | 0.663 | 0.640 | 0.730 |
|  |  | both | 0.705±0.013 | 0.752 | 0.677 | 0.660 | 0.739 |
| kidney | 317 (171/146) | structure | 0.654±0.010 | 0.672 | 0.618 | 0.616 | 0.645 |
|  |  | predicted DE | 0.599±0.013 | 0.688 | 0.559 | 0.559 | 0.575 |
|  |  | both | 0.660±0.017 | 0.684 | 0.615 | 0.613 | 0.642 |

## +expression lift over the structure floor (AUROC, paired bootstrap)
| Organ | structure AUROC | both AUROC | +lift | 95% CI |
|---|---|---|---|---|
| liver | 0.714 | 0.705 | **-0.016** | [-0.046, +0.012] |
| kidney | 0.654 | 0.660 | **+0.014** | [-0.027, +0.055] |

## Reading
- The headline is +expression lift: does organ-trained predicted DE add toxicity signal over chemical structure (drug-disjoint, honest)?
- expression here is PREDICTED (organ engine, from SMILES) so it generalizes to any drug; the feature is the predicted perturbation in the organ's representative basal context.
- Brain has an engine but no tox labels (SIDER block); heart has labels but no engine (no LINCS cardiac data). Both excluded.

