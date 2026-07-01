# Basal-sensitivity of the organ engine + basal-conditioned toxicity

Predicted DE conditioned on several REAL in-distribution measured basals (per-cell-line mean x2). Tests whether the organ-trained engine responds to basal context (the spatial 'combine across locations' prerequisite) and whether it helps toxicity. Drug-disjoint 5x5.

## liver — basal sensitivity (per-sample Pearson of predicted DE across basals)
Closer to 1.000 = engine ignores the basal (spatial conditioning is moot).

| basal A | basal B | mean per-sample Pearson |
|---|---|---|
| organ_mean | PHH | -0.1659 |
| organ_mean | HEPG2 | 0.5842 |
| organ_mean | JHH5 | 0.5207 |
| PHH | HEPG2 | 0.2170 |
| PHH | JHH5 | 0.1924 |
| HEPG2 | JHH5 | 0.6284 |

## liver — toxicity AUROC (n=492, 307/185)
| feature | AUROC |
|---|---|
| structure | 0.714±0.017 |
| predDE @ organ_mean | 0.671±0.014 |
| predDE @ PHH | 0.652±0.018 |
| predDE @ HEPG2 | 0.670±0.014 |
| predDE @ JHH5 | 0.665±0.017 |
| predDE concat-all-basals | 0.681±0.017 |

## kidney — basal sensitivity (per-sample Pearson of predicted DE across basals)
Closer to 1.000 = engine ignores the basal (spatial conditioning is moot).

| basal A | basal B | mean per-sample Pearson |
|---|---|---|
| organ_mean | HA1E | 0.3864 |
| organ_mean | HEK293 | -0.0585 |
| HA1E | HEK293 | 0.3310 |

## kidney — toxicity AUROC (n=317, 171/146)
| feature | AUROC |
|---|---|
| structure | 0.654±0.010 |
| predDE @ organ_mean | 0.599±0.013 |
| predDE @ HA1E | 0.596±0.011 |
| predDE @ HEK293 | 0.619±0.024 |
| predDE concat-all-basals | 0.597±0.019 |

## Reading
- If predicted DE is ~collinear across very different basals (PHH vs HEPG2 / HA1E vs HEK293), the engine ignores basal context -> region-resolved conditioning cannot produce distinct location features, and the spatial 'combine across locations' arm is moot.
- Even if basal-sensitive, per-basal predicted DE is a function of SMILES (drug-independent basal), so it stays structure-ceilinged; concat-across-basals cannot exceed structure.

