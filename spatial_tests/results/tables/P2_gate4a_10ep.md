# P2 Gate 4a — basal-sensitivity (linear-encoder smoke checkpoint)

_Device: cuda:0 (CUDA_VISIBLE_DEVICES='2')_

_Checkpoint: `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model_linearenc_10ep.pt`_

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

### VERDICT: **FAIL**

_Implication: Encoder still basal-insensitive after the linear-encoder swap: the signal is data/label-side, no architecture fix helps. Redirect to F3 / reframe rather than committing the full F4 retrain._
