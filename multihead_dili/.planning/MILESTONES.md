# Milestones (multihead_dili)

## v1.0 — Multi-head MultiDCP DILI baseline (COMPLETE)

**Started:** 2026-05-19
**Completed:** 2026-05-20
**Status:** COMPLETE (6/6 phases done)
**Goal:** Ship the 7-way pathway ablation result on scaffold-novel DILIst test drugs.

**Headline finding:**
On scaffold-novel DILIst drugs, a frozen MolFormer chemistry encoder (0.5930 AUROC, 95% CI
[0.5325, 0.7019]) matches a three-pathway model combining predicted dose-response, predicted
gene-expression, and chemistry (0.5843 AUROC; ΔAUROC = -0.0087, DeLong p = 0.307, NS).
Predicted GEX and predicted dose pathways add no measurable signal beyond chemistry alone.

**Halt gate summary:** HG1 PASS, HG2 PASS, HG3 PASS, HG4 REFRAME (not a blocker).

**Key commits:**
- Phase 0: e281e9f (data foundation)
- Phase 2: 9216d06, c567589, 92cc932 (MODEL_GEX)
- Phase 3: 6e0bbdb, 0b5fe40 (Stage-2 caching)
- Phase 4: 0d210aa (7-way ablation)
- Phase 5: ef4296b (evaluation + HG4 reframe)
- Phase 6: (this commit) (milestone summary)

**Milestone summary:** `results/tables/v1_milestone_summary.md`

---

## (no archived prior milestones)
