# Phase 5 Context — Evaluation

**Phase:** 5 of 6
**Status:** In progress
**Created:** 2026-05-20

## Goal

Compute full metrics with DeLong paired tests and 10K bootstrap CIs; produce headline table, 7-way ablation figure; check halt gate 4.

## Phase 4 Top-Line Results (inputs to Phase 5)

### Scaffold-novel split (primary — leakage discipline), mean across all heads:

| Variant | AUROC mean | AUROC std |
|---------|-----------|----------|
| var1: embed-only (MolFormer 768d) | 0.5955 | 0.0223 |
| var2: gex-only (MODEL_GEX 919d) | 0.5187 | 0.0308 |
| var3: dose-only (MODEL_DOSE 1d) | 0.4779 | 0.0617 |
| var4: embed+gex (1687d) | 0.5895 | 0.0246 |
| var5: embed+dose (769d) | 0.5894 | 0.0232 |
| var6: gex+dose (920d) | 0.5202 | 0.0369 |
| var7: all-three (1688d) ← headline | 0.5911 | 0.0230 |

### Random split (var1 embed-only):

- AUROC = 0.6536 ± 0.0252 (n=45 runs); HG3 PASS

## Halt Gate 4 (HG4) — Load-Bearing

**Trigger:** all-three (var7) AUROC >= best single-pathway AUROC + 0.01 on scaffold-novel split
**Current point estimate:** embed-only (var1, 0.5955) > all-three (var7, 0.5911) by 0.0044
→ var7 is WORSE than var1 by 0.0044 — soft HG4 fire expected

Phase 5 must quantify via DeLong paired test + 10K bootstrap CI whether this difference is statistically significant.

- If p >= 0.05 (not significant): framing = "multi-pathway matches best single; chemistry shortcut is sufficient"
- If p < 0.05 (significant): framing = "chemistry significantly outperforms multi-pathway; no synergy observed"

Either outcome is publishable. Write HALT_REASON_4.md documenting the reframe; does NOT block Phase 6.

## Deliverables

- `src/stage2/evaluate_dili.py` — DeLong + bootstrap + ECE calibration utility
- `results/tables/headline.md` — primary results table (all 21 cells + DeLong p-values + CIs)
- `results/figures/ablation.png` — 7-way ablation strip/bar plot per head depth, per split
- `results/figures/comparison_v05.png` — comparison vs v0.5 if available; otherwise skip and note
- `results/tables/P5_evaluation_summary.md` — narrative of findings + halt-gate verdict
- `.planning/phases/05-evaluation/HALT_REASON_4.md` — required (negative finding, not a blocker)

## Inputs

- `data/processed/predictions/*.parquet` — 630 prediction files
- `data/processed/P4_runs.parquet` — aggregated per-run metrics (630 rows)

## Key References

- Design doc: `/raid/home/joshua/projects/0_project_documents/multihead_multidcp_dili_three_pathway_05192026.md`
- Progress tracker: `/raid/home/joshua/projects/0_project_documents/multihead_multidcp_dili_implementation_plan_05192026.md`
- Phase 4 summary: `results/tables/P4_ablation_summary.md`

## Tone Guidance

Honest reporting. Do NOT massage numbers or cherry-pick favorable cells. The expected outcome (embed-only matches all-three) is a clean negative finding. Headline framing (if HG4 fires):

> "On scaffold-novel DILIst drugs, a frozen MolFormer chemistry encoder (0.5955 AUROC) matches a
> three-pathway model combining predicted dose-response, predicted gene-expression, and chemistry
> (0.5911 AUROC; ΔAUROC = -0.004, 95% CI [...], DeLong p = ...). At this experimental scale,
> predicted-GEX and predicted-dose pathways add no measurable DILI signal beyond what MolFormer
> chemistry alone captures."
