# HALT: Gate 3 Fired -- Predicted Region-Resolved DE Does Not Recover the Pericentral APAP Injury Pattern

**Date:** 2026-06-24T19:24:40Z
**Trigger:** pericentral predicted-vs-measured Pearson +0.0166 < 0.3 (Halt Gate 3, keyed to the pericentral zone, D-08).
**gate_fires:** True

## Per-zone predicted-vs-measured Pearson

| Zone | Pearson r | p-value | n genes compared |
|------|-----------|---------|------------------|
| pericentral | +0.0166 | 1.10e-01 | 9,272 |
| periportal | +0.0038 | 7.14e-01 | 9,272 |

## Interpretation

The predicted, region-resolved DE for acetaminophen (WIRE-01 rule-B MOUSE cache) does not correlate with the measured pericentral APAP injury DE (GSE272564 APAP24h - APAP0h, matched arms) above the 0.3 gate. The likely mechanism (flagged in 02-02-SUMMARY): the frozen MultiDCP-CheMoE backbone is near-zonal-invariant on healthy-liver basals -- its per-zone predicted vectors differ by ~float32 epsilon, so the predicted side carries essentially no pericentral-vs-periportal contrast to match the measured injury gradient. The prediction is dominated by drug + dose, not by the regional basal context. This is the project's instrumented OOD hypothesis surfacing as the gate result, not a wiring bug.

## Caveats (bound the conclusion)

- **Rodent anchor for a human headline (D-09).** The only drug-perturbed spatial anchor that exists as of mid-2026 is rodent (mouse APAP); the headline organ is human. The gate is rodent; per-region predicted DE remains the primary, indirect validity path for the human claim.
- **Pattern, not magnitude (D-05).** A fixed canonical dose was used for the predicted signature; the Pearson tests the DE pattern across genes, not magnitude. The null is about pattern recovery, not dose calibration.
- **Partial replication only (GSE280652, D-06 amendment).** GSE280652 has no matched control arm, so it is a weaker partial replication, not a second matched-DE anchor.

## Decision per D-09

**stop-and-REFRAME the spatial claim** (NOT abandon). Consistent with P1's negative-is-publishable precedent (D-02) and DEC-negative-result-acceptable: the predicted region-resolved signature does not recover the pericentral APAP pattern under the frozen backbone, which is itself the reportable finding. The reframe centers on the near-zonal-invariance of the frozen model on healthy basals (the cell-context encoder contributes little for these inputs) and on what a region-sensitive signature would require.

**Logged:** run_apap_validation.py
