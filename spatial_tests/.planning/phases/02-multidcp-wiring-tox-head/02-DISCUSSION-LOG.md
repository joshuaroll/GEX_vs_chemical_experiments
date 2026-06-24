# Phase 2: MultiDCP wiring & toxicity head - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-24
**Phase:** 2-multidcp-wiring-tox-head
**Areas discussed:** DE-rule sign-off, Gene space, Variant scope, APAP anchor + Halt Gate 3, Dose conditioning

---

## Gene space

| Option | Description | Selected |
|--------|-------------|----------|
| 10,716 PDG | Matches CON-gene-space, PDG/CheMoE native output, region_combiner d=10716 | ✓ |
| 978 LINCS landmark | Smaller, no imputation OOD, but contradicts CON-gene-space | |
| Cache 10,716, defer head top-k | Cache full space, leave top-k size to planner | |

**User's choice:** 10,716 PDG (→ D-01)
**Notes:** The 978→10,716 imputation is the OOD factor to instrument, not avoid.

---

## DE-rule sign-off

First pass: user chose "Reconsider the rule" rather than confirming the seam default. Opened a thinking-partner fork between rule A (minus raw observed basal, the seam/MultiDCP-native convention) and rule B (minus the model's predicted control, bias-corrected, manifold-consistent with measured treated−control).

| Option | Description | Selected |
|--------|-------------|----------|
| B — minus predicted control | Bias-corrected; both terms in model output manifold; matches measured treated−control for APAP gate; +1 control pass/region | ✓ |
| A — minus raw basal (seam default) | MultiDCP-native, already coded, accepts observed-vs-predicted mismatch | |
| Decide after a quick check | Confirm model output type first, then pick | |

**User's choice:** B — minus predicted control (→ D-02)
**Notes:** Conditional on the researcher confirming the checkpoint emits absolute predicted treated (not a DE head). Replaces region_signature.py's current rule A + de_convention string. Resolves 09_spatial_decisions.md open item #3.

### Cache contents
| Option | Description | Selected |
|--------|-------------|----------|
| DE + predicted treated/control terms | Audit reconstruction error, recompute A↔B, APAP in either form (~2–3× size) | ✓ |
| Final DE only | Smallest; re-run model to switch A↔B or get absolute | |

**User's choice:** Cache DE + intermediate terms (→ D-03)

---

## Variant scope for P2

| Option | Description | Selected |
|--------|-------------|----------|
| PDG first, CheMoE same phase if clean | De-risk seam on PDG (S-B), add CheMoE (S-C) when clean | ✓ |
| PDG only this phase | Smallest; CheMoE later | |
| Both PDG + CheMoE together | S-B vs S-C ready earlier, more upfront work | |

**User's choice:** PDG first, CheMoE same phase if clean (→ D-04)
**Notes:** Both checkpoints SHA-pinned in MANIFEST and confirmed on disk.

---

## Dose conditioning

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed canonical 10µM / 24h | LINCS-standard reference; dose-response handled separately (G/H) | ✓ |
| Highest available dose | Strongest signal, risks saturation | |
| Aggregate across dose bins | Smooths choice, blurs signal, 6× passes | |

**User's choice:** Fixed canonical 10µM / 24h (→ D-05)
**Notes:** APAP Pearson tests DE pattern, not magnitude (in-vivo dose ≠ in-vitro µM).

---

## APAP anchor + Halt Gate 3

### Anchor dataset
| Option | Description | Selected |
|--------|-------------|----------|
| GSE272564 primary, GSE280652 replication | Matched ctrl+APAP arms; clean treated−control DE + independent second anchor | ✓ |
| GSE272564 only | Simplest | |
| GSE280652 only | Needs own control arm | |

### Zone definition
| Option | Description | Selected |
|--------|-------------|----------|
| Published annotation, else zonation markers | Published labels if present; else periportal/pericentral markers; Leiden last | ✓ |
| Marker-based only | Always derive from markers | |
| Coarse 2-zone | Force periportal/pericentral | |

### Pearson definition
| Option | Description | Selected |
|--------|-------------|----------|
| 10,716 genes; gate on pericentral zone | Across PDG genes (∩ Visium); gate keyed to APAP-target pericentral zone | ✓ |
| 978 landmark; mean across zones | More conservative on gene reliability | |
| 10,716; all zones must pass | Strictest | |

### Validity stance
| Option | Description | Selected |
|--------|-------------|----------|
| Accept rodent anchor + stop-and-reframe | Rodent APAP = direct-validity stand-in; fired gate = reframe (D-02/P1 precedent) | ✓ |
| Accept anchor, gate = hard stop | Stricter go/no-go | |
| Discuss further | — | |

**User's choices:** D-06 (GSE272564 primary + GSE280652 replication), D-07 (published-then-markers zones), D-08 (10,716 genes, gate on pericentral), D-09 (rodent anchor accepted, stop-and-reframe). Resolves 09_spatial_decisions.md open item #1.

---

## Claude's Discretion

- basal → cell-context projection (978/10,716 → 50-d encoder input).
- top-k size for region-pooled DE.
- wandb project/run naming, smoke-train epoch count.
- pericentral/periportal marker thresholds + zone-assignment procedure.
- extended cache on-disk layout.

## Deferred Ideas

- CheMoE routing-permutation diagnostic (Halt Gate 6) — Phase 6.
- Dose-response channel (G/H) — later.
- Compound-aware splits + leakage audits — Phase 3.
- Full multi-condition training (≥3 seeds) — Phase 4.
- Other organs (kidney/brain/heart, heart last) — their phases.
- Tangram targeted-panel promotion — out of scope.
