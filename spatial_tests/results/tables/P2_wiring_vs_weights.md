# P2 wiring-vs-weights root-cause diagnostic

_Device: cuda:0 (CUDA_VISIBLE_DEVICES='1')_

Builds the model the UPSTREAM way (registry + `MultiDCP_CheMoE_AE`), strict-loads row-17, bypassing `region_signature.py`. Real data, frozen checkpoint, read-only.

## 1. Forward signature & graph trace (from source)
`MultiDCP_CheMoEBase.forward(input_drug, input_gene, mask, input_cell_gex, input_pert_idose, job_id='perturbed', epoch=0)` (multidcp_chemoe_pdg.py:345) delegates UNCONDITIONALLY to `MultiDCP_CheMoE.forward(...)` (line 350) — `job_id` is accepted for training-script compatibility and IGNORED; there is **no autoencoder branch** (docstring line 348-349).

`input_cell_gex` graph to the PREDICTION output (multidcp_chemoe_pdg.py):
- L244 `cell_hidden = self.cell_encoder(input_cell_gex, epoch)` -> [B,50]
- L257 `global_features = cat([drug_embed, cell_hidden, dose_embed])` -> [B,306]
- L260 gating uses global_features; L268-272 expert input = global_features ⊕ gene_embed -> [B,num_gene,434]; L289 weighted expert sum -> predictions [B,num_gene]. **cell_hidden feeds the prediction head directly** (no detach, no *0, no gate). So if the basal moved cell_hidden, it would move the prediction.

cell_encoder = TransformerEncoder (linear_encoder_flag=False), multidcp_pdg.py:103-128:
- L108 `cell_id_embed = Linear(10716->200)->Linear(200->50)` (input_cell_gex)
- L110-112 unsqueeze(-1).repeat(1,1,32): every one of 32 channels is a COPY of the 50-d embed
- L119 PositionalEncoding; L120 `nn.Transformer(d_model=32)` self-attn (has internal LayerNorm); L126 `max` over the 32 dim -> cell_hidden [B,50].


## 2. Upstream-native strict load (bypasses region_signature.py)
- load_state_dict(strict=False): missing=0, unexpected=0 (0/0 == the production strict load).

## 3. DECISIVE TEST — upstream-native forward responds to basal?
- input basals: Pearson(PP,PC) = 0.332578 (clearly different)
- **prediction**: max|Δ output| = **3.1144e-10**, Pearson(pred_PP, pred_PC) = 1.000000
- **cell_hidden (50-d)**: max|Δ| = **1.7370e-08**, Pearson = 0.966244
- Upstream-native call reproduces the wired behaviour (INVARIANT to basal). => NOT a wiring bug (W ruled out): the basal reaches the same dead path whether called via region_signature or natively.

## 4. Layer-by-layer trace inside cell_encoder (where signal dies)
- after `cell_id_embed` Linear(10716->200->50): max|Δ| = 3.2920e+00, Pearson = 0.999601
- after repeat-to-[B,50,32] + pos_encoder (pre-transformer): max|Δ| = 3.2920e+00, Pearson = 0.999599
- after `nn.Transformer` (d_model=32, internal LayerNorm): max|Δ| = 1.7901e-07, Pearson = 1.000000
- after `max(-1)` -> cell_hidden [B,50]: max|Δ| = 1.7370e-08, Pearson = 0.966244

## 5. AE / reconstruction path probe (M check)
- AE/reconstruct-like methods on MultiDCP_CheMoE_AE: NONE
- on inner MultiDCP_CheMoE: ['cell_decoder_dim']
- Source confirms (docstring L348-349): 'CheMoE doesn't use autoencoder mode, always predicts perturbed expression.' There is no second forward path / reconstruction head that consumes the basal differently. => (M) ruled out: no alternate mode would route cell context into a prediction.

## 6. Weight-magnitude audit (D check)
| param | shape | mean\|w\| | max\|w\| | frac\|w\|<1e-6 |
|---|---|---|---|---|
| `cell_encoder.cell_id_embed.0.weight` | (200, 10716) | 1.1860e-02 | 2.6705e-02 | 0.000 |
| `cell_encoder.cell_id_embed.0.bias` | (200,) | 1.0616e-03 | 3.5008e-03 | 0.000 |
| `cell_encoder.cell_id_embed.1.weight` | (50, 200) | 8.7057e-02 | 1.7780e-01 | 0.000 |
| `cell_encoder.cell_id_transformer.encoder.layers.0.self_attn.in_proj_weight` | (96, 32) | 2.1678e-01 | 4.3473e-01 | 0.000 |
| `cell_encoder.cell_id_transformer.encoder.layers.0.linear1.weight` | (128, 32) | 2.1510e-01 | 4.3694e-01 | 0.000 |
| `cell_encoder.cell_id_transformer.encoder.norm.weight` | (32,) | 3.3055e-03 | 1.2799e-02 | 0.000 |
| `cell_encoder.cell_id_transformer.encoder.norm.bias` | (32,) | 9.7849e-03 | 3.2890e-02 | 0.000 |

_Drug branch reference (known live):_
| param | shape | mean\|w\| | max\|w\| | frac\|w\|<1e-6 |
|---|---|---|---|---|
| `drug_fp.conv_layers.0.bias` | (1, 16) | 2.9917e-01 | 5.8338e-01 | 0.000 |
| `drug_fp.conv_layers.0.linear.weight` | (16, 62) | 1.5516e-01 | 6.3792e-01 | 0.000 |
| `drug_fp.conv_layers.0.degree_layer_list.0.weight` | (16, 68) | 1.4836e-01 | 2.9699e-01 | 0.000 |

_First head layers that consume cell_hidden (global_features cols 128:178):_
| layer | full mean\|w\| | cell-slice(128:178) mean\|w\| | drug-slice(0:128) mean\|w\| | dose-slice(178:306) mean\|w\| |
|---|---|---|---|---|
| gating.gate.0.weight (128, 306) | 7.1410e-02 | 7.8369e-02 | 7.1007e-02 | 6.9094e-02 |
| experts.0.mlp.0.weight (128, 434) | 5.6389e-02 | 5.8587e-02 | 5.2546e-02 | 4.8488e-02 |

## 7. Confirmatory probe — the transformer is the collapse point

Direct probe of the cell encoder's transformer block on two maximally-different
synthetic 50-d embeds (bypassing the input Linear entirely):

- INPUT embeds: max|Δ| = **19.02**, Pearson = **−1.0** (as different as possible)
- after `repeat(1,1,32)` + pos_encoder + `nn.Transformer` + `max(-1)` →
  cell_hidden: max|Δ| = **2.19e-08**
- transformer output: abs-mean 0.0143, value range pinned to **[−0.0022, 0.4124]**
  regardless of input — a near-fixed point.

So the transformer maps ANY 50-d embed (even Pearson −1.0) to the same
cell_hidden to float epsilon.

## Mechanism

The signal is **alive** out of the input Linear (§4: `cell_id_embed`
Linear(10716→200→50) gives max|Δ| = 3.29, Pearson 0.9996 between periportal and
pericentral) and survives the `repeat`+`pos_encoder` (max|Δ| 3.29). It **dies
inside `nn.Transformer`** (§4: max|Δ| collapses 3.29 → 1.79e-07, Pearson →
1.000000). Two compounding causes, both intrinsic to the frozen checkpoint:

1. **Constant-across-`d_model` input.** `multidcp_pdg.py:110-112` does
   `cell_id_embed.unsqueeze(-1).repeat(1,1,32)` — every one of the 32 `d_model`
   channels is an identical copy of the 50-d embedding. The transformer's
   internal LayerNorm normalizes over `d_model=32`; a constant-across-`d_model`
   vector normalizes to the same pattern irrespective of its magnitude, so the
   per-token scale (which carries the basal signal) is largely discarded.
2. **A near-zero final encoder LayerNorm scale.** `encoder.norm.weight` has
   mean|w| = 3.3e-3 (§6) — two orders of magnitude below the transformer's other
   weights (~0.2). After the residual blocks, the final norm rescales the output
   to a near-constant, saturating cell_hidden to a fixed point.

The cell-encoder **parameters are not literally zero** (§6: input Linear mean|w|
0.012-0.087, transformer attn/FF 0.215, all frac<1e-6 = 0.000), and the head's
cell-context slice is fully live (cell-slice mean|w| 0.078 ≥ drug-slice 0.071).
But the composed cell-encoder **function** is inert: it maps all basals to one
cell_hidden. The wiring is correct (W ruled out, §3: upstream-native call shows
the identical invariance) and there is no AE/reconstruction mode that would route
the basal differently (M ruled out, §5).

---

## VERDICT: DEAD-WEIGHTS (D) — cell-context encoder is functionally inert in the row-17 checkpoint

The checkpoint's frozen `TransformerEncoder` collapses every basal — even
maximally-different ones (Pearson −1.0) — to the same 50-d cell_hidden
(max|Δ| ≤ 2.2e-08), so the prediction is a function of (drug, dose) only. This is
not (W): the upstream-native forward, built the training way and bypassing
`region_signature.py`, reproduces the identical basal-invariance (§3, max|Δ|
output 3.1e-10). It is not (M): `MultiDCP_CheMoEBase.forward` always runs the
perturbed head, `job_id` is ignored, and there is no reconstruction/AE method on
the class (§5). It is the **(D)** class — the cell-context path is inert in the
checkpoint — with the precise mechanism being architectural collapse inside the
frozen transformer (constant-across-`d_model` input × near-zero final LayerNorm
scale), not zeroed head weights.

### Single concrete next action

**(D) — the row-17 checkpoint is unusable for cell-conditioning.** No edit to
`region_signature.py` or the basal pipeline can recover zonal contrast in the
*predicted* signature, because the frozen cell encoder discards the basal before
it reaches the head. Two paths, in order of fidelity to the project's intent:

- **F4 (sound):** source / re-derive a checkpoint whose cell encoder actually
  uses `input_cell_gex` — e.g. retrain with `linear_encoder_flag=True`
  (`LinearEncoder`, multidcp_pdg.py:55-84, bypasses the collapsing transformer),
  or fix the transformer construction so `d_model` carries distinct channels
  (project the 50-d embed with a learned `Linear(1→32)` instead of `repeat`).
  This re-opens the checkpoint hunt rather than the wiring.
- **F3 (workaround, no model change):** condition the cached region signature on
  a basal contrast computed OUTSIDE the model (carry `zone_basal −
  tissue_mean_basal` as the region feature), accepting that the "predicted" DE no
  longer reflects model-extracted cell context.

The wired pipeline and the APAP gate are NOT to be re-run for this; the verdict
stands on the upstream-native evidence above.
