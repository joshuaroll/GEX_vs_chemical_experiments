# Synthesis Summary

Single entry point for the downstream gsd-roadmapper. Mode: new project.
Generated from per-doc classifications in
.planning/intel/classifications/ plus the source documents.

---

## Doc counts by type
- SPEC: 1 (source of truth, precedence 0)
  - /raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md
- DOC: 2 (precedence 2, supporting detail; both agree with SPEC)
  - /raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md (resolved decisions + dataset-accession corrections)
  - /raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md (channel-to-code map)
- ADR: 0 / PRD: 0

All 3 classified high confidence. No UNKNOWN-confidence-low docs.
Cycle detection ran on the cross-ref graph: no cycle (SPEC -> {06, 09}; neither DOC back-references SPEC).

## Decisions
- 12 decisions extracted (decisions.md). No LOCKED ADRs.
- Decided (design): 8 — frozen baseline, comparison-arm (Q1), Visium-only platform (Q3/C1), published region granularity (Q4), frozen-checkpoint seam, ortholog discipline, per-organ-only, dose-response first-class channel (D2), fixed concat-MLP.
- PROPOSED (pending professor sign-off): 3 — spatial DE-rule divergence (open item 3), indirect-validity + APAP anchor (open item 2), negative-result acceptability (open item 1).

## Requirements
- 8 phase requirements (requirements.md), one per SPEC phase P0..P7, each mapping to one GSD phase:
  - REQ-P0-dataset-acquisition-manifest
  - REQ-P1-eda-bracket
  - REQ-P2-wiring-tox-head
  - REQ-P3-splits-no-leakage
  - REQ-P4-per-organ-train-test
  - REQ-P5-cross-species-translatability
  - REQ-P6-confound-interpretability
  - REQ-P7-robustness-writeup
- Plus a cross-cutting requirements block (8 hard rules + no-leakage detail + cross-organ pitfalls).
- 6 halt gates preserved (P0, P1, P2, P3, P4, P6); P5 and P7 have no gate. Each gate writes HALT_REASON.md into the phase directory.

## Constraints
- 8 constraints (constraints.md):
  - api-contract: CON-model-io (frozen MultiDCP I/O dims), CON-code-module-layout (src/spatial/ extension)
  - schema: CON-tox-head (channel-to-projection map), CON-gene-space (10,716-gene space + coverage), CON-ortholog-alignment
  - nfr: CON-splits (compound-aware protocol), CON-spatial-qc (Moran's I / squidpy), CON-halt-gate-thresholds

## Context topics
- 10 topics (context.md): research question + publishable concepts, experimental conditions + cross-species axis, three-channel pipeline lineage, per-organ channel availability, OOD risk, dataset plan + accession corrections, validity plan, per-organ tox labels, open items needing professor sign-off, out of scope, bibliographic corrections.

## Conflicts
- 0 blockers, 0 competing-variants, 5 auto-resolved/INFO.
- Detail: .planning/INGEST-CONFLICTS.md

## Per-type intel files
- .planning/intel/decisions.md
- .planning/intel/requirements.md
- .planning/intel/constraints.md
- .planning/intel/context.md

## Routing note for roadmapper
- Map P0..P7 one-to-one to GSD phases; preserve all 6 halt gates.
- Surface the 3 PROPOSED decisions (DE-rule divergence, APAP validity stand-in, negative-result acceptability) to the user for confirmation before/at phase planning — they are design questions awaiting professor sign-off, not blockers.
- Initialize as its own GSD project rooted at spatial_tests/. Do NOT touch /raid/home/joshua/.planning (separate halted liver project).
