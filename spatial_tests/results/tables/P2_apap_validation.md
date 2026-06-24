# P2 APAP Validity Anchor: Predicted-vs-Measured Per-Zone DE Pearson (WIRE-03 / Halt Gate 3)

**Generated:** 2026-06-24T19:24:40Z
**Predicted side:** WIRE-01 rule-B MOUSE cache (D-02), drug `acetaminophen`, 10716 PDG genes per zone.
**Measured side (primary, D-06):** GSE272564 matched arms -- measured DE per zone = pseudobulk(APAP24h_zone) - pseudobulk(APAP0h_zone), in mouse symbols, mapped mouse->human (one-to-one orthologs) and reindexed to the PDG order; absent genes FLAGGED (present_mask), never zero-filled (D-08).

## Per-zone predicted-vs-measured Pearson

| Zone | Pearson r | p-value | n genes compared (intersection) |
|------|-----------|---------|---------------------------------|
| pericentral | +0.0166 | 1.10e-01 | 9,272 |
| periportal | +0.0038 | 7.14e-01 | 9,272 |

### Spot counts per zone (arms split by GSM, Pitfall 6)

| Zone | control (APAP0h) spots | APAP (APAP24h) spots |
|------|------------------------|----------------------|
| pericentral | 1146 | 278 |
| periportal | 332 | 1374 |
| unassigned | 5 | 3 |

## Halt Gate 3 verdict

Gate is keyed to the **pericentral** zone (where APAP injury classically acts, D-08). Pericentral Pearson = +0.0166; threshold 0.3. Halt Gate 3: **FIRES** (pericentral r < 0.3 -> stop-and-REFRAME, D-09).

## Caveats

- **Rodent anchor for a human headline (D-09).** The gate is computed on mouse APAP Visium -- the only drug-perturbed spatial anchor that exists as of mid-2026 -- while the headline organ is human. This asymmetry is accepted (D-09): per-region predicted DE remains the primary, indirect validity path for the human claim; the rodent APAP gate is the direct stand-in.
- **GSE280652 is a partial replication only (D-06 amendment).** GSE280652 has NO matched control arm, so it cannot supply a matched treated-minus-control DE; it is a weaker partial replication, not an independent matched anchor. The primary, matched anchor is GSE272564 (APAP0h baseline vs APAP24h injury).
- **Pattern, not magnitude (D-05).** The predicted signature uses a fixed canonical dose; the Pearson tests the cross-gene DE pattern, not magnitude.
- **Near-zonal-invariance of the frozen backbone (02-02).** The cached per-zone predicted DE vectors are near-identical across pericentral vs periportal (~float32 epsilon): the frozen CheMoE cell-context encoder contributes little for healthy-liver basals, so the predicted side carries little regional contrast. If the pericentral Pearson is ~0, this is the mechanism -- an honest negative, reported plainly, not a bug to chase.


---
*Run elapsed: 7.4s*
