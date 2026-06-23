# HALT: Gate 2 Fired -- Floor-Ceiling AUROC Gap Not Significantly Positive

**Gap observed:** -0.0518 AUROC
**95% CI:** [-0.2135, 0.1164]
**gate_fires:** True (ci_lower <= 0 -- CI includes 0)

## Interpretation

The 95% paired bootstrap CI of (ceiling AUROC - floor AUROC) includes 0. This means the measured-biology signal (LINCS L1000 DE) does not provide a statistically distinguishable lift over the structure-only fingerprint baseline for liver DILI prediction on the shared drug set.

## Decision per D-02

**stop-and-REFRAME** (negative result is publishable, D-02 locked). Do NOT proceed to Phase 2 model training. Reframe the research question before continuing.

**Bootstrap resamples (valid):** 10,000 / 10,000
**Logged:** run_p1_eda.py
