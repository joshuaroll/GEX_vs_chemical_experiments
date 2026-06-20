## Conflict Detection Report

Mode: new project. Precedence: ADR > SPEC > PRD > DOC (SPEC overridden to precedence 0).
Docs synthesized: 3 (1 SPEC, 2 DOC). No ADRs, no PRDs ingested.
Cycle detection: ran on cross-ref graph; no cycle among the 3 in-set docs (SPEC -> {06, 09}; neither DOC back-references the SPEC). Max depth well under 50.
Confidence: all 3 docs classified high; no UNKNOWN/low-confidence docs.

### BLOCKERS (0)

(none)

No LOCKED ADRs were ingested, so no LOCKED-vs-LOCKED contradiction is possible.
No existing project context to contradict (net-new project). No cycles. No
UNKNOWN-confidence-low docs. Nothing gates the workflow.

### WARNINGS (0)

(none)

No PRDs were ingested, so there are no competing acceptance-criteria variants.
The two DOCs agree with the SPEC on every overlapping claim (conditions, DE-rule
spatial divergence, P2 Pearson < 0.3 gate, dataset accessions, channel-to-code map).

### INFO (5)

[INFO] DOCs agree with SPEC, no precedence override needed
  Found: SPEC (precedence 0) and DOCs 06/09 (precedence 2) overlap on conditions, DE rule, halt-gate thresholds, dataset accessions, and the channel-to-code map.
  Note: All overlaps are consistent. The two DOCs add detail (resolved decisions, accession corrections, channel-to-code map) that the SPEC already adopted. No precedence tiebreak was triggered.

[INFO] DE-rule divergence is a documented, intentional change
  Found: SPEC §1.2 and DOC 09 (open item 3) both define the spatial DE anchor as predicted_treated(drug, region_basal) - region_basal, diverging from the parent project's treated - diseased rule.
  Note: This is a deliberate, attributed divergence (PROPOSED, pending professor sign-off per DOC 09), not a contradiction. Recorded in decisions.md (DEC-de-rule-spatial-divergence) and flagged for explicit REQUIREMENTS/PROJECT note downstream.

[INFO] "StructuredMoE" naming nuance reconciled
  Found: DOC 06 (§1, §8) notes the "StructuredMoE" label overstates the architecture (4 generic experts, top-2 routing, emergent specialization, not scaffold/assay-partitioned). SPEC §3 states the same (specialization emergent; routing-permutation diagnostic is the only check).
  Note: Consistent. The CON-model-io constraint and the P6 routing-permutation halt gate both reflect this; the name is retained but its meaning is documented as load-bearing-on-the-diagnostic.

[INFO] Condition-label scope differs by organ, not in conflict
  Found: SPEC uses A / B-C / S-B-S-C / G-H / fusion; DOC 09 references S-B/S-C/S-F; DOC 06 enumerates A-H + 3-ch across liver/kidney/brain.
  Note: These are consistent expansions for different organ scopes (DOC 06 is the cross-organ matrix; the SPEC is the spatial arm). No condition is defined two different ways. Synthesized condition set follows the SPEC.

[INFO] Open design questions await professor sign-off (non-blocking)
  Found: DOC 09 lists 5 open items (negative-result acceptability, APAP stand-in, DE-rule divergence note, squidpy/Tangram env addition, dataset-reference corrections).
  Note: These are design decisions, not workflow blockers. The SPEC has already adopted the proposed resolutions. Captured in decisions.md as PROPOSED status and in context.md for the roadmapper to surface to the user.
