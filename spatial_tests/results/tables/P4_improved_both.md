# Properly combining DE + structure (fixing the naive-fusion gap)

structure=ChemBERTa(384), DE=engine predicted DE(978) @ organ-mean basal. All C tuned by nested LogisticRegressionCV (roc_auc, inner cv=5); DE PCA-reduced before fusion; late fusion = avg of tuned structure+DE probs. Drug-disjoint 5-fold x 5 seeds.

| Organ | n | variant | AUROC |
|---|---|---|---|
| liver | 492 | structure (tuned) | 0.769±0.009 |
|  |  | DE PCA30 (tuned) | 0.624±0.010 |
|  |  | both naive (C=1) | 0.705±0.013 |
|  |  | both early+PCA30 (tuned) | 0.756±0.007 |
|  |  | both LATE fusion (avg) | 0.746±0.008 |
| kidney | 317 | structure (tuned) | 0.628±0.024 |
|  |  | DE PCA30 (tuned) | 0.587±0.022 |
|  |  | both naive (C=1) | 0.660±0.017 |
|  |  | both early+PCA30 (tuned) | 0.627±0.017 |
|  |  | both LATE fusion (avg) | 0.634±0.023 |

## Reading
- 'both naive (C=1)' is the old stage-2 method; compare to the tuned/PCA/late-fusion variants. If proper fusion now EXCEEDS structure, the earlier both<structure was a methodology artifact (DE block diluting structure under uniform L2). If it still ties structure, the ceiling is real.

