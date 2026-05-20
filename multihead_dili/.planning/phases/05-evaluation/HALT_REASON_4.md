# HALT_REASON_4.md — HG4 Reframe (Negative Finding)

## Halt Gate 4 Status: REFRAME (NOT a blocker for Phase 6)

**Trigger:** all-three (var7) AUROC < best single-pathway AUROC + 0.01 on scaffold-novel split

## Numbers

- Best single-pathway (scaffold): var1 embed-only mlp2 = **0.5930** 95% CI [0.5325, 0.7019]
- All-three (scaffold): var7 mlp2 = **0.5843** 95% CI [0.5120, 0.6862]
- ΔAUROC (var7 − var1): **-0.0087** (threshold needed: +0.01)
- DeLong test (best head): z = -1.022, p = 0.3066

## Interpretation

All-three (var7) does NOT outperform embed-only (var1) by ≥ 0.01 AUROC on the
scaffold-novel split. In fact, embed-only marginally outperforms all-three.

Statistical interpretation:
- DeLong p = 0.3066 ≥ 0.05: the difference is NOT statistically significant.
- **Framing:** Multi-pathway MATCHES best single-pathway — chemistry shortcut is sufficient.
- Predicted GEX and predicted dose pathways add no measurable signal beyond chemistry alone.

## Publishable Framing

Honest headline (recommended):
> 'On scaffold-novel DILIst drugs, a frozen MolFormer chemistry encoder
> (0.5930 AUROC, 95% CI [0.5325, 0.7019]) matches a three-pathway model combining
> predicted dose-response, predicted gene-expression, and chemistry
> (0.5843 AUROC, 95% CI [0.5120, 0.6862];
> ΔAUROC = -0.0087, DeLong p = 0.3066).
> At this experimental scale, predicted-GEX and predicted-dose pathways add no
> measurable DILI signal beyond what MolFormer chemistry alone captures.'

## Why This Is Still Publishable

1. Clean negative finding with well-powered design (630 classifier runs, 5-fold CV, 3 seeds)
2. Demonstrates that predicted (noisy) GEX signatures do not improve on chemistry shortcut
3. Motivates v2: use measured LINCS GEX, larger E-Hill corpus, attention-based combiner
4. Consistent with literature: DILI is scaffold-driven; 37 unique E-Hill drugs is a small corpus

## Action

Phase 6 (milestone summary) proceeds. The milestone summary should adopt the honest framing above.
No paper claim inflation. The design doc anticipated this as an acceptable negative finding.

**This HALT_REASON does NOT block Phase 6. Proceed.**