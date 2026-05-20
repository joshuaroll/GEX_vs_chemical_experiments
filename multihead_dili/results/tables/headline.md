# Phase 5 Headline Results — Multi-Head MultiDCP DILI

**Date:** 2026-05-20
**Bootstrap:** 10,000 resamples (drug-level mean-pooled predictions)
**DeLong test:** var7 (all-three) vs var1 (embed-only) — paired, two-sided

## Scaffold-Novel Split

| Variant | Head | AUROC (pooled) | 95% CI | AUPRC | ECE | n_drugs |
|---------|------|---------------|--------|-------|-----|---------|
| var1: embed-only (MolFormer 768d) | linear | 0.5852 | [0.5119, 0.6832] | 0.6911 | 0.0932 | 168 |
| var1: embed-only (MolFormer 768d) | mlp1 | 0.5844 | [0.5198, 0.6913] | 0.6877 | 0.1051 | 168 |
| var1: embed-only (MolFormer 768d) | mlp2 | 0.5930 | [0.5325, 0.7019] | 0.6970 | 0.0817 | 168 |
| var2: gex-only (MODEL_GEX 919d) | linear | 0.5227 | [0.4137, 0.5911] | 0.6332 | 0.0905 | 168 |
| var2: gex-only (MODEL_GEX 919d) | mlp1 | 0.5035 | [0.4097, 0.5861] | 0.6201 | 0.0894 | 168 |
| var2: gex-only (MODEL_GEX 919d) | mlp2 | 0.5333 | [0.4594, 0.6360] | 0.6372 | 0.0891 | 168 |
| var3: dose-only (MODEL_DOSE 1d) | linear | 0.4909 | [0.3639, 0.5498] | 0.5766 | 0.0938 | 168 |
| var3: dose-only (MODEL_DOSE 1d) | mlp1 | 0.5172 | [0.4502, 0.6361] | 0.6132 | 0.0912 | 168 |
| var3: dose-only (MODEL_DOSE 1d) | mlp2 | 0.4307 | [0.3348, 0.5094] | 0.5559 | 0.0910 | 168 |
| var4: embed+gex (1687d) | linear | 0.5819 | [0.5124, 0.6828] | 0.6944 | 0.0968 | 168 |
| var4: embed+gex (1687d) | mlp1 | 0.5804 | [0.5109, 0.6813] | 0.6860 | 0.0862 | 168 |
| var4: embed+gex (1687d) | mlp2 | 0.5769 | [0.5168, 0.6880] | 0.6718 | 0.1016 | 168 |
| var5: embed+dose (769d) | linear | 0.5830 | [0.5132, 0.6842] | 0.6915 | 0.0990 | 168 |
| var5: embed+dose (769d) | mlp1 | 0.5855 | [0.5133, 0.6843] | 0.6834 | 0.0816 | 168 |
| var5: embed+dose (769d) | mlp2 | 0.5775 | [0.5096, 0.6811] | 0.6785 | 0.1012 | 168 |
| var6: gex+dose (920d) | linear | 0.5151 | [0.4157, 0.5899] | 0.6222 | 0.0976 | 168 |
| var6: gex+dose (920d) | mlp1 | 0.5220 | [0.4541, 0.6288] | 0.6244 | 0.0826 | 168 |
| var6: gex+dose (920d) | mlp2 | 0.5188 | [0.4387, 0.6165] | 0.6117 | 0.1016 | 168 |
| var7: all-three (1688d) | linear | 0.5798 | [0.5063, 0.6784] | 0.6857 | 0.0830 | 168 |
| var7: all-three (1688d) | mlp1 | 0.5788 | [0.5053, 0.6791] | 0.6814 | 0.0809 | 168 |
| var7: all-three (1688d) | mlp2 | 0.5843 | [0.5120, 0.6862] | 0.6738 | 0.0947 | 168 |

## Random Split

| Variant | Head | AUROC (pooled) | 95% CI | AUPRC | ECE | n_drugs |
|---------|------|---------------|--------|-------|-----|---------|
| var1: embed-only (MolFormer 768d) | linear | 0.6364 | [0.5906, 0.6735] | 0.7047 | 0.0951 | 739 |
| var1: embed-only (MolFormer 768d) | mlp1 | 0.6445 | [0.5999, 0.6822] | 0.7113 | 0.0986 | 739 |
| var1: embed-only (MolFormer 768d) | mlp2 | 0.6645 | [0.6215, 0.7030] | 0.7295 | 0.0830 | 739 |
| var2: gex-only (MODEL_GEX 919d) | linear | 0.5394 | [0.4943, 0.5781] | 0.6522 | 0.1072 | 739 |
| var2: gex-only (MODEL_GEX 919d) | mlp1 | 0.5129 | [0.4729, 0.5585] | 0.6201 | 0.1186 | 739 |
| var2: gex-only (MODEL_GEX 919d) | mlp2 | 0.5441 | [0.5050, 0.5879] | 0.6499 | 0.1010 | 739 |
| var3: dose-only (MODEL_DOSE 1d) | linear | 0.5124 | [0.4935, 0.5798] | 0.6096 | 0.1055 | 739 |
| var3: dose-only (MODEL_DOSE 1d) | mlp1 | 0.5086 | [0.4664, 0.5519] | 0.6147 | 0.0965 | 739 |
| var3: dose-only (MODEL_DOSE 1d) | mlp2 | 0.5216 | [0.4773, 0.5624] | 0.6320 | 0.1111 | 739 |
| var4: embed+gex (1687d) | linear | 0.6410 | [0.5989, 0.6821] | 0.7076 | 0.1011 | 739 |
| var4: embed+gex (1687d) | mlp1 | 0.6437 | [0.5983, 0.6816] | 0.7118 | 0.0884 | 739 |
| var4: embed+gex (1687d) | mlp2 | 0.6153 | [0.5813, 0.6644] | 0.6932 | 0.1122 | 739 |
| var5: embed+dose (769d) | linear | 0.6361 | [0.5930, 0.6743] | 0.7098 | 0.0970 | 739 |
| var5: embed+dose (769d) | mlp1 | 0.6412 | [0.5990, 0.6803] | 0.7159 | 0.0895 | 739 |
| var5: embed+dose (769d) | mlp2 | 0.6513 | [0.6151, 0.6953] | 0.7236 | 0.0818 | 739 |
| var6: gex+dose (920d) | linear | 0.5475 | [0.5021, 0.5869] | 0.6538 | 0.1019 | 739 |
| var6: gex+dose (920d) | mlp1 | 0.5384 | [0.4941, 0.5781] | 0.6392 | 0.1003 | 739 |
| var6: gex+dose (920d) | mlp2 | 0.5398 | [0.4987, 0.5846] | 0.6468 | 0.0969 | 739 |
| var7: all-three (1688d) | linear | 0.6368 | [0.5940, 0.6766] | 0.7046 | 0.0815 | 739 |
| var7: all-three (1688d) | mlp1 | 0.6453 | [0.6003, 0.6813] | 0.7138 | 0.0838 | 739 |
| var7: all-three (1688d) | mlp2 | 0.6201 | [0.5914, 0.6731] | 0.6963 | 0.1001 | 739 |

## DeLong Paired Tests: var7 (all-three) vs var1 (embed-only) — Scaffold Split

| Head | AUROC var7 | AUROC var1 | ΔAUROC | z-stat | p-value |
|------|-----------|-----------|--------|--------|---------|
| linear | 0.5919 | 0.5982 | -0.0063 | -0.603 | 0.5464 |
| mlp1 | 0.5919 | 0.6065 | -0.0146 | -1.462 | 0.1437 |
| mlp2 | 0.5988 | 0.6183 | -0.0195 | -1.022 | 0.3066 |

\* p < 0.05 (two-sided DeLong test)

## Top-Line Summary

Best single-pathway (scaffold): **var1 embed-only mlp2** = 0.5930 [0.5325, 0.7019]
Best all-three (scaffold): **var7 all-three mlp2** = 0.5843 [0.5120, 0.6862]
ΔAUROC (var7 - best single): -0.0087

### HG4 Verdict

**HG4: REFRAME** — all-three (0.5843) does NOT beat best single (0.5930) by >= 0.01 AUROC on scaffold split (ΔAUROC = -0.0087). See `HALT_REASON_4.md`.
