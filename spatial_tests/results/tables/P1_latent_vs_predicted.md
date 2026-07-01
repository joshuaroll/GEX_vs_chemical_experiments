# Latent vs predicted-DE: which representation predicts toxicity better?

Both features come off the SAME MultiDCP forward in the SAME fixed periportal basal. Expression options: **latent** = global_features (306) | **predicted DE** = rule-B treated(drug) - treated(inert 'C'), full 10,716 genes and the 978-landmark subset (919 landmarks present in model space). Structure (**chemberta**) is a reference. Head = L2 logistic regression, drug-level 5-fold x 5 seeds.

| Organ | n (pos/neg) | structure | latent (306) | predicted DE (10,716) | predicted DE (978 lm) |
|---|---|---|---|---|---|
| liver | 492 (307/185) | 0.714±0.017 | 0.637±0.003 | 0.603±0.015 | 0.610±0.020 |
| kidney | 317 (171/146) | 0.611±0.016 | 0.614±0.017 | 0.573±0.009 | 0.577±0.008 |
| heart | 614 (404/210) | 0.604±0.025 | 0.623±0.013 | 0.597±0.010 | 0.616±0.008 |

## Notes
- Same forward, same basal: the only difference is which representation is read out (gating-network input latent vs the gene-output DE). A latent-vs-DE gap is therefore purely representational, not a context confound.
- Predicted DE is rule-B (treated - inert-control), the spatial-arm DE convention; both operands stay in the model's output manifold.
- Single fixed periportal basal (no within-organ region resolution yet); this isolates the representation question from the location-combination question.

## Figure
- `/raid/home/joshua/claude_memory/downstream_2026/visuals/three_way_comparison/latent_vs_predicted_by_organ.png`
