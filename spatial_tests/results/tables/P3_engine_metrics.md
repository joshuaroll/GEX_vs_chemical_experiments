# Standard MultiDCP expression metrics — per-organ engines (held-out-drug test set)

Metrics from MultiDCP's `utils/metric.py` (per-sample). DE = pred−x2 vs x1−x2 (the perturbation; the meaningful signal). precision@k = overlap of the true top/bottom-100 genes with the predicted top/bottom-k. raw-treated Pearson shown for reference (inflated by the shared basal). Linear encoder, seed 42.

| Organ | Arch | n_test | Pearson DE | Spearman DE | RMSE DE | prec@100 pos | prec@100 neg | Pearson treated |
|---|---|---|---|---|---|---|---|---|
| liver | multidcp | 598 | 0.7566 | 0.7293 | 0.2524 | 0.571 | 0.575 | 0.9897 |
| liver | chemoe | 598 | 0.7223 | 0.6961 | 0.2998 | 0.548 | 0.552 | 0.9853 |
| kidney | multidcp | 853 | 0.6752 | 0.6428 | 0.2175 | 0.503 | 0.516 | 0.9918 |
| kidney | chemoe | 853 | 0.6614 | 0.6304 | 0.2308 | 0.503 | 0.501 | 0.9908 |
| brain | multidcp | 958 | 0.7929 | 0.7691 | 0.2007 | 0.602 | 0.610 | 0.9940 |
| brain | chemoe | 958 | 0.7816 | 0.7614 | 0.2148 | 0.600 | 0.598 | 0.9930 |

## precision@k (positive / negative genes), all k

| Organ | Arch | p@50 pos | p@50 neg | p@100 pos | p@100 neg | p@200 pos | p@200 neg |
|---|---|---|---|---|---|---|---|
| liver | multidcp | 0.709 | 0.718 | 0.571 | 0.575 | 0.385 | 0.389 |
| liver | chemoe | 0.678 | 0.683 | 0.548 | 0.552 | 0.371 | 0.372 |
| kidney | multidcp | 0.631 | 0.650 | 0.503 | 0.516 | 0.347 | 0.353 |
| kidney | chemoe | 0.627 | 0.622 | 0.503 | 0.501 | 0.345 | 0.346 |
| brain | multidcp | 0.742 | 0.762 | 0.602 | 0.610 | 0.403 | 0.403 |
| brain | chemoe | 0.743 | 0.736 | 0.600 | 0.598 | 0.400 | 0.401 |

Sanity: Pearson DE here should match each run's logged test_de_pearson (kidney/multidcp ≈ 0.671).

