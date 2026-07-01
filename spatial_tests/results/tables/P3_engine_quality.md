# Stage 1: per-organ expression-engine quality (from scratch, 978 genes, drug-disjoint)

Held-out-drug **test DE-Pearson** (predicted DE = pred − x2 vs true DE = x1 − x2), per-sample mean. Reference points: gate-2 floor 0.20; honest measured drug-disjoint ceiling ≈0.60 (P1_eda).

| Organ | n train profiles | base MultiDCP (attention) | CheMoE (MoE) |
|---|---|---|---|
| liver | 2,698 | 0.750 (dev 0.743) | 0.722 (dev 0.708) |
| kidney | 3,898 | 0.671 (dev 0.668) | 0.653 (dev 0.654) |
| brain | 4,282 | 0.791 (dev 0.790) | 0.782 (dev 0.777) |
| heart | 0 | blocked (no LINCS cardiac line) | blocked |

## Reading
- All trained engines clear the gate-2 floor (0.20) by a wide margin and reach/exceed the measured ceiling (~0.60), so from-scratch per-organ training produces high-quality predicted DE on NEW drugs.
- base vs CheMoE: base MultiDCP's fixed gene-vector prior + drug-gene attention is expected to help on the small per-organ sets; the table shows the head-to-head.
- Encoder = linear (plain, no AE pretraining), seed 42. Next: seeds 2+ for mean±std; AE-pretraining if basal looks weak; then Stage 2 (predicted DE -> small fixed toxicity head, per organ).
- Heart needs external cardiac perturbation data (DrugMatrix) before it can be trained.

