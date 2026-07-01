# Latent three-way comparison, split by body region

structure = **chemberta** embedding | expression = **MultiDCP-CheMoE latent (global_features, 306d)** | head = L2 logistic regression, drug-level 5-fold x 5 seeds.

| Organ | n (pos/neg) | structure | expression (MultiDCP latent) | both | +expr lift | 95% CI |
|---|---|---|---|---|---|---|
| liver | 492 (307/185) | 0.714 | 0.637 | 0.712 | **-0.011** | [-0.020, -0.002] |
| kidney | 317 (171/146) | 0.611 | 0.614 | 0.612 | **-0.000** | [-0.018, +0.018] |
| heart | 614 (404/210) | 0.604 | 0.623 | 0.616 | **+0.009** | [-0.005, +0.024] |

## Notes / honest caveats
- Expression = MultiDCP global_features. Validated: within one basal context the cell_hidden + dose blocks are CONSTANT across drugs (std ~1.8e-8), so the per-drug expression signal is the 128-d drug block, i.e. MultiDCP's own (expression-trained) structure encoder. So this compares an external encoder vs MultiDCP's drug encoder; a near-zero lift means they carry the same information.
- Uses MultiDCP-PREDICTED latent (from SMILES) -> no measured-LINCS join needed -> larger n than the measured comparison (P1_three_way).
- Structure encoder is pluggable: --encoder chemberta|ecfp4|unimol_v1|unimol_v2 (UniMol needs unimol_tools).
- Brain excluded (label-to-structure join blocked: SIDER is CID-keyed, no SOC file on disk).

## Figure
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/three_way_comparison/latent_three_way_by_organ.png`
