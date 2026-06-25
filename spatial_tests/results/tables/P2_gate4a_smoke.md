# P2 Gate 4a — basal-sensitivity (linear-encoder smoke checkpoint)

_Device: cuda:0 (CUDA_VISIBLE_DEVICES='2')_

_Checkpoint: `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model_linearenc_smoke.pt`_

Builds `MultiDCP_CheMoE_AE` the upstream way with `linear_encoder_flag=True` (LinearEncoder MLP), strict-loads the 8-epoch best-val smoke checkpoint. Real human liver basals, frozen checkpoint, read-only. Same probe path as `diagnose_wiring_vs_weights.py`.

## 1. Strict load (linear-encoder variant)
- cell_encoder class: `LinearEncoder` (expect `LinearEncoder`)
- load_state_dict(strict=False): missing=0, unexpected=0 (must be 0/0 — else wrong encoder variant)

## 2. Real-basal probe (periportal vs pericentral human liver)
- input basals: Pearson(PP,PC) = 0.332578 (clearly different)
- **cell_hidden (50-d)**: max|Δ| = **1.0385e+00**, Pearson = 0.999088
- **prediction (10716-d)**: max|Δ| = **3.1696e-02**, Pearson(pred_PP, pred_PC) = 0.999987

## 3. Synthetic maximal-contrast probe (Pearson ~ -1.0)
- input Pearson(A,B) = -1.000000
- **cell_hidden max|Δ|** = **7.1088e-01**, Pearson = 0.997418 (must NOT collapse to ~0)

## 4. Verdict
Baseline (dead transformer, row-17): cell_hidden max|Δ| ~1.7e-8, pred max|Δ| ~3e-10, Pearson(pred_PP,pred_PC) 1.000000.

| criterion | threshold | observed | pass |
|---|---|---|---|
| cell_hidden max\|Δ\| | >= 1e-2 | 1.0385e+00 | True |
| prediction max\|Δ\| | >= 1e-3 | 3.1696e-02 | True |
| Pearson(pred_PP, pred_PC) | < 0.999 | 0.999987 | False |

### VERDICT: **FAIL** (one of three criteria missed: prediction Pearson)

## 5. Interpretation (the failure mode is NOT the dead-transformer one)

The auto-verdict is FAIL on the strict three-criterion rule, but the *reason* is
different from the row-17 baseline and matters for the redirect decision:

- The encoder is now **alive**, not inert. cell_hidden moves max|Δ| = 1.04 on the
  real basal (vs 1.7e-8 dead) and 0.71 on the synthetic Pearson -1.0 probe with no
  collapse. The linear-encoder swap fixed the inert-encoder problem.
- The prediction **does** respond: max|Δ| = 3.17e-2, ~1e8 larger than the dead
  transformer's 3e-10, and 30x over the 1e-3 threshold. Criteria 1 and 2 pass.
- The single failure is **Pearson(pred_PP, pred_PC) = 0.999987 vs < 0.999**: the
  basal shifts prediction *magnitude* on a handful of genes but does not reshape
  the global 10,716-gene pattern. The per-region predicted DE signatures would be
  near-collinear across zones (cosine ~1.0), so a downstream zonation/translatability
  signal built on these predictions would be weak.

Caveat: this is the **8-epoch smoke** checkpoint, not the converged model. The
encoder being live but only weakly steering the head is consistent with an
under-trained cell branch (the head has not yet learned to weight cell_hidden).
The strict gate fails as specified; the qualitative picture is "encoder fixed,
head not yet using it," not "architecture cannot respond."

_Implication: Gate 4a fails on the prediction-Pearson criterion, but the encoder
is no longer inert (the linear-encoder fix worked) — the prediction is just nearly
collinear across zones at 8 epochs. Do NOT commit the full ~8h F4 retrain on this
evidence alone; either run a short additional smoke (more epochs) to see if
pred-Pearson drops below 0.999 as the head learns to use cell_hidden, or redirect
to F3 / reframe if the cell branch stays cosmetically live but predictively
collinear._
