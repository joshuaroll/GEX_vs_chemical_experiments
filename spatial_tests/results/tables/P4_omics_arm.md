# Can the measured-omics arm be made toxicity-informative? (+ honest fusion test)

Measured-DE overlap set. omics-alone over representation x model; then oracle-inflation sanity and a nonlinear JOINT (concat) fusion vs structure. 3 seeds x 5-fold.

## A. omics-alone AUROC (representation x model)

| organ | rep | model | AUROC |
|---|---|---|---|
| liver | raw | logreg | 0.525±0.020 |
| liver | raw | rf | 0.483±0.007 |
| liver | raw | mlp | 0.382±0.019 |
| liver | pca30 | logreg | 0.459±0.025 |
| liver | pca30 | rf | 0.496±0.035 |
| liver | pca30 | mlp | 0.452±0.060 |
| liver | kbest100 | logreg | 0.506±0.012 |
| liver | kbest100 | rf | 0.461±0.016 |
| liver | kbest100 | mlp | 0.444±0.039 |
| kidney | raw | logreg | 0.524±0.035 |
| kidney | raw | rf | 0.522±0.008 |
| kidney | raw | mlp | 0.556±0.008 |
| kidney | pca30 | logreg | 0.527±0.010 |
| kidney | pca30 | rf | 0.518±0.031 |
| kidney | pca30 | mlp | 0.501±0.023 |
| kidney | kbest100 | logreg | 0.557±0.040 |
| kidney | kbest100 | rf | 0.520±0.001 |
| kidney | kbest100 | mlp | 0.547±0.009 |

## B/C. fusion + oracle-inflation

| organ | n | best omics (rep/model) | structure (logreg / mlp) | joint concat-MLP | late | oracle real | oracle PERMUTED-omics |
|---|---|---|---|---|---|---|---|
| liver | 197 | raw/logreg (0.525) | 0.616 / 0.551 | 0.604 | 0.593 | 0.912 | 0.901 |
| kidney | 240 | kbest100/logreg (0.557) | 0.681 / 0.610 | 0.641 | 0.672 | 0.923 | 0.894 |

## Reading
- If best omics-alone stays ~chance across all reps/models, the measured omics is not tox-informative as represented (cell context / aggregation / n), and fusion has nothing to add.
- oracle PERMUTED-omics ~ oracle real => the high oracle is label-selection INFLATION, not real complementary signal.
- joint concat-MLP > structure-mlp => nonlinear feature-level fusion finds signal that linear prob-fusion missed (the user's hypothesis); ~equal => it does not, on this data.

