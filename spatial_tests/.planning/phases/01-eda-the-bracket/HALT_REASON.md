# HALT: Gate 2 Fired -- No Measured Lift Over Structure; Benchmark Is Drug-Leakage-Inflated

**Gap observed (drug-agg):** -0.1770 AUROC
**95% CI:** [-0.3161, -0.0326]
**gate_fires:** True (ci_upper < 0 -- CI entirely below 0)

## Interpretation

The drug-aggregated, leakage-free measured ceiling does not exceed the structure-only ECFP4 floor (paired bootstrap CI of ceiling - floor is below 0). Measured LINCS L1000 DE provides no drug-level lift over chemical structure for liver DILI on this shared set. This is a refined, honest reading -- not 'structure beats biology' -- bounded by the caveats below.

## Caveats (bound the strength of this conclusion)

- **Headline = drug leakage.** Profile-level measured AUROC is 0.912 under a leaky (drug-in-train+test) split but only 0.605 under a drug-disjoint split (+0.307 inflation). The Wang/Li-style benchmark (~0.798) is substantially drug-leakage-inflated.
- **Measured ~= structure at the fair level.** The honest profile-level measured ceiling is comparable to the structure floor; the more-negative drug-aggregated gap is largely noise from collapsing many profiles onto few drugs.
- **Underpowered.** The shared set has few negative drugs; the conservative minimum detectable gap exceeds the observed gap, so the firing rests on the paired bootstrap with a thin margin (see P1_eda.md Power section).

## Decision per D-02

**stop-and-REFRAME** (negative result is publishable, D-02 locked). Do NOT proceed to Phase 2 model training. The reframe centers on the drug-leakage finding (the benchmark is inflated; measured DE does not generalize to held-out drugs above structure here) and on powering a future drug-disjoint comparison.

**Bootstrap resamples (valid):** 10,000 / 10,000
**Logged:** run_p1_eda.py
