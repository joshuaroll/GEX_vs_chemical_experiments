# Phase 4 Ablation Summary — 7-Way Pathway Ablation DILI Classifier

**Date:** 2026-05-20
**Total runs:** 630
**Grid:** 7 variants × 3 heads × 2 splits × 5 folds × 3 seeds = 630

---

## Halt Gate 3 (HG3)

**Check:** embed-only (var1, MolFormer 768d) random-split AUROC ≥ 0.55
**Result:** mean=0.6536 ± 0.0252 (n=45 runs)
**Verdict:** PASS ✓

---

## Per-Variant Per-Head AUROC (mean ± std over 5 folds × 3 seeds = 15 runs)

### Scaffold-Novel Split (primary — leakage discipline)

| Var | Variant | Head | AUROC mean | AUROC std | AUPRC mean | Bal.Acc mean |
|-----|---------|------|-----------|----------|-----------|------------|
| 1 | embed-only (MolFormer 768d) | linear | 0.5897 | 0.0188 | 0.7016 | 0.5434 |
| 1 | embed-only (MolFormer 768d) | mlp1 | 0.5990 | 0.0192 | 0.7045 | 0.5490 |
| 1 | embed-only (MolFormer 768d) | mlp2 | 0.5977 | 0.0279 | 0.7090 | 0.5471 |
| 2 | gex-only (MODEL_GEX 919d) | linear | 0.5230 | 0.0291 | 0.6446 | 0.5117 |
| 2 | gex-only (MODEL_GEX 919d) | mlp1 | 0.4977 | 0.0228 | 0.6305 | 0.4926 |
| 2 | gex-only (MODEL_GEX 919d) | mlp2 | 0.5354 | 0.0286 | 0.6544 | 0.5202 |
| 3 | dose-only (MODEL_DOSE 1d) | linear | 0.4850 | 0.0438 | 0.6013 | 0.5091 |
| 3 | dose-only (MODEL_DOSE 1d) | mlp1 | 0.5243 | 0.0479 | 0.6144 | 0.5029 |
| 3 | dose-only (MODEL_DOSE 1d) | mlp2 | 0.4244 | 0.0481 | 0.5635 | 0.4450 |
| 4 | embed+gex (1687d) | linear | 0.5898 | 0.0202 | 0.7019 | 0.5415 |
| 4 | embed+gex (1687d) | mlp1 | 0.5891 | 0.0213 | 0.6940 | 0.5436 |
| 4 | embed+gex (1687d) | mlp2 | 0.5897 | 0.0323 | 0.6942 | 0.5633 |
| 5 | embed+dose (769d) | linear | 0.5879 | 0.0249 | 0.7027 | 0.5379 |
| 5 | embed+dose (769d) | mlp1 | 0.5938 | 0.0200 | 0.7001 | 0.5542 |
| 5 | embed+dose (769d) | mlp2 | 0.5866 | 0.0254 | 0.6931 | 0.5397 |
| 6 | gex+dose (920d) | linear | 0.5142 | 0.0375 | 0.6359 | 0.4994 |
| 6 | gex+dose (920d) | mlp1 | 0.5296 | 0.0226 | 0.6509 | 0.5102 |
| 6 | gex+dose (920d) | mlp2 | 0.5167 | 0.0470 | 0.6244 | 0.5075 |
| 7 | all-three (1688d) ← headline | linear | 0.5876 | 0.0209 | 0.6979 | 0.5406 |
| 7 | all-three (1688d) ← headline | mlp1 | 0.5906 | 0.0265 | 0.6971 | 0.5447 |
| 7 | all-three (1688d) ← headline | mlp2 | 0.5952 | 0.0222 | 0.6985 | 0.5423 |

### Random Split (comparability baseline)

| Var | Variant | Head | AUROC mean | AUROC std | AUPRC mean | Bal.Acc mean |
|-----|---------|------|-----------|----------|-----------|------------|
| 1 | embed-only (MolFormer 768d) | linear | 0.6385 | 0.0177 | 0.7128 | 0.5951 |
| 1 | embed-only (MolFormer 768d) | mlp1 | 0.6512 | 0.0247 | 0.7224 | 0.6109 |
| 1 | embed-only (MolFormer 768d) | mlp2 | 0.6713 | 0.0223 | 0.7478 | 0.6295 |
| 2 | gex-only (MODEL_GEX 919d) | linear | 0.5407 | 0.0428 | 0.6582 | 0.5265 |
| 2 | gex-only (MODEL_GEX 919d) | mlp1 | 0.5132 | 0.0298 | 0.6336 | 0.5123 |
| 2 | gex-only (MODEL_GEX 919d) | mlp2 | 0.5528 | 0.0243 | 0.6641 | 0.5321 |
| 3 | dose-only (MODEL_DOSE 1d) | linear | 0.5192 | 0.0794 | 0.6276 | 0.5063 |
| 3 | dose-only (MODEL_DOSE 1d) | mlp1 | 0.5076 | 0.0756 | 0.6216 | 0.5059 |
| 3 | dose-only (MODEL_DOSE 1d) | mlp2 | 0.5362 | 0.0574 | 0.6473 | 0.5220 |
| 4 | embed+gex (1687d) | linear | 0.6413 | 0.0252 | 0.7176 | 0.6131 |
| 4 | embed+gex (1687d) | mlp1 | 0.6503 | 0.0297 | 0.7233 | 0.6075 |
| 4 | embed+gex (1687d) | mlp2 | 0.6211 | 0.0293 | 0.7016 | 0.5978 |
| 5 | embed+dose (769d) | linear | 0.6372 | 0.0199 | 0.7153 | 0.6069 |
| 5 | embed+dose (769d) | mlp1 | 0.6469 | 0.0319 | 0.7265 | 0.6030 |
| 5 | embed+dose (769d) | mlp2 | 0.6549 | 0.0396 | 0.7319 | 0.6177 |
| 6 | gex+dose (920d) | linear | 0.5491 | 0.0410 | 0.6572 | 0.5307 |
| 6 | gex+dose (920d) | mlp1 | 0.5473 | 0.0598 | 0.6608 | 0.5264 |
| 6 | gex+dose (920d) | mlp2 | 0.5427 | 0.0575 | 0.6538 | 0.5249 |
| 7 | all-three (1688d) ← headline | linear | 0.6381 | 0.0384 | 0.7099 | 0.6134 |
| 7 | all-three (1688d) ← headline | mlp1 | 0.6494 | 0.0370 | 0.7216 | 0.6102 |
| 7 | all-three (1688d) ← headline | mlp2 | 0.6266 | 0.0525 | 0.7049 | 0.5855 |

---

## Top-Line Preview (scaffold split, mean across all heads)

| Variant | AUROC mean | AUROC std |
|---------|-----------|----------|
| var1: embed-only (MolFormer 768d) | 0.5955 | 0.0223 |
| var2: gex-only (MODEL_GEX 919d) | 0.5187 | 0.0308 |
| var3: dose-only (MODEL_DOSE 1d) | 0.4779 | 0.0617 |
| var4: embed+gex (1687d) | 0.5895 | 0.0246 |
| var5: embed+dose (769d) | 0.5894 | 0.0232 |
| var6: gex+dose (920d) | 0.5202 | 0.0369 |
| var7: all-three (1688d) ← headline | 0.5911 | 0.0230 |

---

## Feature Dimensions Reference

| Var | Pathway | Input dim |
|-----|---------|----------|
| 1 | embed-only (MolFormer) | 768 |
| 2 | gex-only (MODEL_GEX DE) | 919 |
| 3 | dose-only (MODEL_DOSE E-Hill) | 1 |
| 4 | embed+gex | 1687 |
| 5 | embed+dose | 769 |
| 6 | gex+dose | 920 |
| 7 | all-three (headline) | 1688 |

Note: feat_gex = 919-dim (not 978 as design doc stated; 919 landmark genes overlap between gene_vector.csv and lincs_train_safe.parquet with header=None).

---

## HG3 Full Details

```
embed-only (var1) random-split AUROC: mean=0.6536 std=0.0252 n=45
HG3 threshold: 0.55
HG3 verdict: PASS
```

