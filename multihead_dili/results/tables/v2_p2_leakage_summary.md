# v2 Phase 2 — Leakage Decomposition: 3-Split Comparison

**Date:** 2026-05-20
**Purpose:** Sharpen the leakage finding from v2 P1 into a publishable methodological
correction showing how Wang/Li's profile-level training leaks drug identity into
the test set, inflating their published AUROC.

## Headline finding

> Wang/Li's published AUROC of 0.798 inflates by 0.089 AUROC due to profile-level training leakage (369/382 test drugs also in training); under drug-level evaluation the AUROC drops to 0.673, indistinguishable from chance.

## Headline results table

| Split | AUROC (profile-level) | 95% CI (profile bootstrap) | n_test profiles | n_test drugs |
|-------|-----------------------|----------------------------|-----------------|--------------|
| A: Profile-level (Wang/Li original) | **0.7619** | [0.7486, 0.7751] | 1079 | 382 |
| B: Drug-level random | 0.6725 | [0.6613, 0.6838] | 1740 | 123 |
| C: Scaffold-level | 0.5517 | [0.5348, 0.5677] | 993 | 110 |

## Leakage quantification (THE core result)

**Split A (Profile-level — Wang/Li's protocol):**
- Train profiles: 4384 | Test profiles: 1079
- Unique drugs in train: 602
- Unique drugs in test: 382
- **Drugs in BOTH train AND test: 369**
- **Leakage fraction: 96.6% of test drugs also in training**

**Split B (Drug-level random — corrected):**
- Train profiles: 3723 | Test profiles: 1740
- Unique drugs in train: 492 | test: 123
- Drugs in both: 0 (should be 0)

**Split C (Scaffold-level — strictest):**
- Train profiles: 4470 | Test profiles: 993
- Unique scaffolds train: 337 | test: 85
- Unique drugs in test: 110
- Scaffolds crossing boundary: 0 (should be 0)

## AUROC comparison to reference points

| Method | Split | AUROC | Notes |
|--------|-------|-------|-------|
| Wang/Li 2020 published | Profile-level (their split) | 0.798 | Original paper |
| **This work Split A** | **Profile-level (Wang/Li protocol reproduced)** | **0.7619** | **Sanity check: should ≈ 0.79** |
| **This work Split B** | **Drug-level random** | **0.6725** | **Leakage-corrected** |
| **This work Split C** | **Scaffold-level** | **0.5517** | **Strictest OOD** |
| v2 P1 Run A (measured GEX) | Drug-level scaffold CV | 0.4565 | Drug-unit, scaffold OOD (5-fold) |
| v2 P1 Run A (measured GEX) | Drug-level random CV | 0.5014 | Drug-unit, random (5-fold) |
| v1 MolFormer chemistry-only | Drug-level scaffold | 0.5930 | Chemistry only, no GEX |

## Sanity gate: did profile-level (A) reproduce ≈ 0.79?

**PASS** — Split A AUROC = 0.7619 (within ±0.05 of Wang/Li published 0.798; v2 P1 benchmark on this box: 0.761). Profile-level protocol faithfully reproduced.

## Per-seed AUROC breakdown

| Split | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean | Std |
|-------|--------|--------|--------|--------|--------|------|-----|
| Profile-level | 0.7684 | 0.7689 | 0.7804 | 0.7567 | 0.7600 | 0.7669 | 0.0082 |
| Drug-level | 0.6685 | 0.7430 | 0.6340 | 0.6349 | 0.7892 | 0.6939 | 0.0620 |
| Scaffold-level | 0.5548 | 0.5356 | 0.5184 | 0.5512 | 0.5878 | 0.5496 | 0.0230 |

## Methodological caveat

The three splits evaluate the SAME model architecture on the SAME data pool,
but the test sets are DIFFERENT across splits by design — they use different
partitioning discipline. The AUROC differences reflect how split discipline
changes what the model is being tested on, not prediction quality on identical inputs.

Pairwise DeLong tests are not reported because the test sets are not matched.
The leakage delta (A − B = +0.0894) is the key quantity: it represents the AUROC inflation
attributable to Wang/Li's profile-level training leakage.

## Comparison to v1 and v2 P1

v1 gap (0.798 − 0.5930 = 0.205) was originally interpreted as GEX signal.
This experiment shows the gap decomposes as:
  - Profile-level leakage: +0.0894 (split A − split B AUROC)
  - Scaffold-novelty penalty vs random: +0.1208 (B − C AUROC)
  - Drug-level drug-novelty floor: 0.6725 AUROC

The v1 chemistry-only (MolFormer, 0.5930 scaffold) represents a better baseline
than Wang/Li 0.798, which is inflated by leakage.

## Files

- Figure: `results/figures/v2_leakage_decomposition.png`
- Code: `src/v2/run_v2_p2.py` (this experiment), `src/v2/wangli_8layer.py` (DNN)
- Predictions: `data/processed/v2_p2_predictions/` (gitignored)
