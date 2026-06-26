# Expression-arm robustness checks (three-way comparison)

Tests whether 'measured expression adds nothing over structure' survives a fairer setup than the raw 978-dim drug-level LR (p>>n). Same organs, same fair evaluation.

## +expression lift (structure+expression minus structure), by organ and method

| Organ | method | n | structure | expression | both | **+expr lift** | 95% CI |
|---|---|---|---|---|---|---|---|
| liver | drug-level raw 978d (prior) | 234 | 0.575 | 0.456 | 0.527 | **-0.048** | [-0.118, +0.024] |
| kidney | drug-level raw 978d (prior) | 203 | 0.680 | 0.493 | 0.659 | **-0.005** | [-0.049, +0.038] |
| heart | drug-level raw 978d (prior) | 221 | 0.585 | 0.514 | 0.597 | **+0.042** | [-0.024, +0.109] |
| liver | drug-level PCA-20 expr | 234 | 0.575 | 0.393 | 0.525 | **-0.049** | [-0.100, -0.002] |
| liver | profile-level drug-disjoint | 2696 prof/234 drugs | 0.605 | 0.685 | 0.618 | **+0.049** | [-0.015, +0.121] |
| kidney | drug-level PCA-20 expr | 203 | 0.680 | 0.544 | 0.668 | **+0.004** | [-0.032, +0.040] |
| kidney | profile-level drug-disjoint | 1398 prof/203 drugs | 0.717 | 0.467 | 0.715 | **+0.001** | [-0.015, +0.016] |
| heart | drug-level PCA-20 expr | 221 | 0.585 | 0.526 | 0.609 | **+0.040** | [-0.018, +0.100] |
| heart | profile-level drug-disjoint | 2663 prof/221 drugs | 0.481 | 0.612 | 0.496 | **-0.014** | [-0.072, +0.032] |

Lift CI excluding 0 => expression adds signal over structure. PCA reduces expression to 20 components (fit on train fold only). Profile-level keeps all LINCS profiles, StratifiedGroupKFold by drug, bootstrap resamples DRUGS not profiles.

## Figure
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/three_way_comparison/expression_lift_robustness.png`
