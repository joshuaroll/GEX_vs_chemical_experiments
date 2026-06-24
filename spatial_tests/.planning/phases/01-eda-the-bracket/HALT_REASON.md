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

---

## RESOLUTION (2026-06-23) -- 2x2 documentation complete; Phase 2 unblocked (D-10)

The floor x ceiling, leaky x disjoint 2x2 is now complete in
`results/tables/P1_eda.md`. The previously missing **floor-leaky** cell was added
(`floor_profile_leaky_auroc`: LR on ECFP4, random `StratifiedKFold` profile split,
NO drug groups, seed=42 -- mirroring the ceiling's `_leakage_decomposition` leaky
path), and a 95% paired-bootstrap CI (10,000 resamples, seed=42) was added on the
profile-level drug-disjoint gap from aligned floor/ceiling OOF vectors over the SAME
kept profile rows and SAME StratifiedGroupKFold drug folds.

Per **D-08 / D-10**, the halt LIFTS by COMPLETING the honest bracket documentation
(the 2x2), regardless of the gap sign. **Phase 2 (Halt Gate 3) is now UNBLOCKED.**
The predicted, region-resolved signature is the real untested bet; a measured null
just raises the bar for it.

The un-halt is a project decision (D-10), **not a gate change**: the PRIMARY
drug-level Halt Gate 2 STILL FIRES (gap -0.1770 AUROC, 95% CI [-0.3161, -0.0326],
`gate_fires: True`) and the driver STILL exits 1. The original halt analysis above
stands verbatim as the record of why the drug-level gate fired; nothing in it
(the gap/CI numbers, `gate_fires:** True`, or the Decision per D-02 block) is altered.

**Diagnostic reading (the cell that "explains the paper"):** floor-leaky is high
(profile-level, on the kept set), confirming that the Wang/Li-style headline (~0.798)
is largely drug-identity memorization that chemical structure reproduces in its own
favorable (leaky) setup -- measured biology adds little there. At the fair
drug-disjoint level the gap is small (+0.059, profile-level) and the drug-aggregated
gate is negative and underpowered. See `results/tables/P1_eda.md` for all four cells.
