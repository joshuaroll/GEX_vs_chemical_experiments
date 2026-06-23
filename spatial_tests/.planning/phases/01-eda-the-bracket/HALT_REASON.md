# HALT: Gate 2 Fired -- Structure Floor Significantly Exceeds Measured Ceiling (drug-level)

**Gap observed:** -0.1770 AUROC
**95% CI:** [-0.3161, -0.0326]
**gate_fires:** True (ci_upper < 0 -- CI entirely below 0 (gap significantly negative))

## Interpretation

The 95% paired bootstrap CI of (ceiling AUROC - floor AUROC) lies entirely below 0. On the drug-disjoint, leakage-free shared set, the structure-only ECFP4 floor significantly OUTperforms the measured LINCS L1000 DE ceiling for liver DILI prediction -- measured biology adds no drug-level generalizable lift over chemical structure here.

## Caveats (bound the strength of this conclusion)

- **Leakage-free ceiling.** The ceiling uses StratifiedGroupKFold over compound (a drug's profiles never straddle train/test). A prior profile-level CV inflated the ceiling via per-drug memorization (one drug carries up to 784 profiles) and was discarded.
- **Underpowered.** The shared set is heavily positive-skewed (few negative drugs), so the CI is wide and the gate is sensitive to small changes.
- **Unit of analysis.** At the profile level (Wang/Li's published setup) the measured DE reproduces their benchmark (AUROC ~0.79-0.93). The near-chance drug-level ceiling therefore reflects failure to generalize to HELD-OUT DRUGS on this small set -- and suggests the profile-level benchmark itself may be substantially drug-leakage-inflated -- rather than a total absence of measured-biology DILI signal.

## Decision per D-02

**stop-and-REFRAME** (negative result is publishable, D-02 locked). Do NOT proceed to Phase 2 model training. Reframe the research question before continuing.

**Bootstrap resamples (valid):** 10,000 / 10,000
**Logged:** run_p1_eda.py
