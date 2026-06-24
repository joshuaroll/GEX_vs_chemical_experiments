# F4 scope — encoder fix + retrain plan

**Drafted:** 2026-06-24
**Type:** read-only scoping spike. No training launched, no parent-repo edit, no
production-source edit. Investigation + costed plan only.
**Authorization:** professor approved a SCOPED reversal of the frozen-baseline
rule to fix + retrain ONLY the MultiDCP-CheMoE cell-context encoder.

## Problem recap (already diagnosed, evidence in P2_wiring_vs_weights.md)

The row-17 checkpoint `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt`
(SHA256 `fbee15faade904cacd6484832cea211ebd4b28186ca40def40af3fea27c7d2a0`,
class `MultiDCP_CheMoE_AE`) has a functionally inert cell-context encoder. The
basal signal is alive out of the input Linear (10716→200→50, max|Δ| 3.29 between
periportal and pericentral) but collapses inside the transformer cell-encoder
(max|Δ| → 1.79e-7, Pearson → 1.0), so the predicted signature is a function of
(drug, dose) only. The checkpoint's state dict contains the **transformer** cell
encoder (`cell_id_transformer.encoder/decoder...` keys present), confirming it was
trained with `linear_encoder_flag=False`. Two compounding mechanical causes:
1. `cell_id_embed.unsqueeze(-1).repeat(1,1,32)` (`multidcp_pdg.py:110-112`) makes
   all 32 `d_model` channels identical, so the transformer's internal LayerNorm
   discards the per-token magnitude that carries the basal.
2. The final `encoder.norm.weight` ≈ 3.3e-3 (two orders below the other
   transformer weights) crushes residual variance to a near-fixed point.

---

## 1. The exact architecture edit

### Key finding: the fix already exists as a config toggle

`multidcp_pdg.py` defines **two** cell encoders:
- `TransformerEncoder` (`multidcp_pdg.py:86-128`) — the collapsing path the
  frozen checkpoint uses (`repeat(1,1,32)` at L110-112; the `nn.Transformer`
  at L97-99 / L120 where the signal dies).
- `LinearEncoder` (`multidcp_pdg.py:55-84`) — a plain 3-layer MLP
  `Linear(10716→200)→ReLU→Linear(200→100)→ReLU→Linear(100→50)→ReLU`. **No
  transformer, no LayerNorm, no `repeat` collapse.** Output is `[batch, 50]`,
  identical shape/semantics to the transformer encoder's `cell_hidden`.

The CheMoE model selects between them on a registry flag
(`multidcp_chemoe_pdg.py:170-173`):
```python
if self.linear_encoder_flag:
    self.cell_encoder = LinearEncoder(self.cell_id_input_dim)
else:
    self.cell_encoder = TransformerEncoder(self.cell_id_input_dim)
```
and the training script already exposes it as a CLI flag
(`multidcp_chemoe_ae_de_pdg.py:942-943`, `--linear_encoder_flag`, passed into
the registry at L1028).

So the fix is **a training-time flag, not a code change** to the parent repo. No
edit to `multidcp_pdg.py` or `multidcp_chemoe_pdg.py` is required for the
recommended path.

### Candidate edits (with pros/cons)

| Candidate | What it does | Pros | Cons | Parent-repo code change? |
|---|---|---|---|---|
| **A. `linear_encoder_flag=True` (RECOMMENDED)** | Swaps `TransformerEncoder` → `LinearEncoder` (MLP 10716→200→100→50). Bypasses the collapsing transformer entirely. | Zero new code; the toggle is tested upstream and is how the `run_mdcp_chemoe_pdg.sh` script is already configured to train. Lowest risk. MLP cannot collapse the per-token scale because there is no LayerNorm-over-channels. | Loses any (unrealized) capacity the transformer might have added. None observed — the transformer never used it. | None |
| **B. Replace `repeat(1,1,32)` with a learned `Linear(1→32)`** | Project the 50-d embed per token into 32 distinct channels (the commented-out `cell_id_embed_1` at `multidcp_pdg.py:96, 113` is exactly this, currently disabled) so LayerNorm-over-`d_model` no longer sees a constant. | Keeps the transformer; addresses mechanical cause #1 directly. | Edits production source (`multidcp_pdg.py`) — outside the authorized scope ("fix + retrain ONLY the encoder" is satisfied by A without touching shared source). Does not by itself fix cause #2 (near-zero final-norm), which is re-learned at training time anyway but adds risk. New, untested code path. | **Yes** (forbidden by this spike's constraints) |
| **C. Fix only the final-norm scale** | Re-initialize / unfreeze `encoder.norm.weight`. | Cheapest in params. | Addresses only cause #2; cause #1 (constant-across-`d_model`) still discards magnitude. Insufficient alone. Requires retrain regardless. | Effectively requires B too |

### Recommendation

**Candidate A: retrain with `linear_encoder_flag=True`.** It is the
minimal, lowest-risk change, requires no edit to any parent-repo source, and is
the configuration the upstream launcher (`run_mdcp_chemoe_pdg.sh`, which already
passes `--linear_encoder_flag`) was written for. The transformer encoder added no
measured value (it collapsed every basal); the MLP encoder is strictly safer for
preserving basal variance into `cell_hidden`.

### I/O contract impact (must stay load-compatible)

The forward signature is **unchanged**:
`forward(input_drug, input_gene, mask, input_cell_gex, input_pert_idose, job_id='perturbed', epoch=0)`
→ `(predictions[B, 10716], cell_hidden[B, 50])`
(`multidcp_chemoe_pdg.py:218, 296, 345-351`). The spatial caller
`region_signature._call_model` (`region_signature.py:779-788`) passes exactly
these kwargs and reads `pred[N_PDG]`. The downstream contract holds.

The new checkpoint's **state-dict keys change** inside the cell encoder
(`cell_encoder.cell_id_1/2/3.*` for the MLP vs `cell_encoder.cell_id_transformer.*`
for the transformer). This is expected for a fresh retrain (we are not reloading
old weights). The ONE required consequence in the spatial repo:
`region_signature.py:670` currently hardcodes `linear_encoder_flag=False` so it
strict-loads the transformer checkpoint. After retrain this must flip to `True`
so the registry builds `LinearEncoder` and `load_state_dict(strict=True)` stays
0-missing/0-unexpected. That is a one-line flag flip + checkpoint-path/SHA update
(see §5) — not a forward-signature change.

---

## 2. The retrain entrypoint + config + env

- **Env:** `conda activate mdcp_env` (parent CLAUDE.md L13-14). Confirmed as the
  training env; `dili_v04_env` is only used for the read-only spatial diagnostics.
- **Script:** `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/multidcp_chemoe_ae_de_pdg.py`
- **Launcher:** `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/scripts/run_mdcp_chemoe_pdg.sh`
  (already passes `--linear_encoder_flag`; defaults to `MAX_EPOCH=300`, batch 32,
  dropout 0.3, runs 9 cells × 5 folds).
- **Model selection / save:** `multidcp_chemoe_ae_de_pdg.py:813-818` — saves on
  best **dev DE all-genes Pearson** to `args.model_name`. Registry built at
  L1025-1028 with `num_gene=10716` (`GEX_SIZE`, L90), `pert_idose_input_dim=2`,
  `dropout`, `linear_encoder_flag` from CLI.
- **How the current `best_model.pt` was produced:** trained with the **transformer**
  encoder (state dict has `cell_id_transformer.*`), i.e. WITHOUT
  `--linear_encoder_flag`, by an earlier run (best_model.pt mtime 2026-01-17
  08:28; the separate HT29 50-epoch log finished 13:06 the same day). The exact
  original command is not recorded; the cell, fold, and dose-vocab match the
  spatial config's assumptions (single dose → 2-dim one-hot, Pitfall 4).

### Proposed retrain command (single representative cell, candidate A)

The spatial signature is **dose-agnostic and cell-context-driven**, so we need a
checkpoint whose cell encoder responds to basal; a single-cell training run that
reproduces the ~0.75 DE R² and passes the basal-sensitivity gate is sufficient
for the F4 deliverable (full 9-cell × 5-fold sweep is unnecessary for the spatial
cache and would multiply cost). Recommended:

```bash
conda activate mdcp_env
cd /raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src
CUDA_VISIBLE_DEVICES=1 python multidcp_chemoe_ae_de_pdg.py \
  --test_cell A549 --fold 1 --gpu 1 \
  --linear_encoder_flag \
  --dropout 0.3 --batch_size 32 --max_epoch 100 \
  --drug_file   /raid/home/joshua/projects/MultiDCP_pdg/data/all_drugs_pdg.csv \
  --gene_file   /raid/home/joshua/data/MultiDCP/data/gene_vector.csv \
  --data_pickle     /raid/home/joshua/projects/MultiDCP_pdg/data/pdg_brddrugfiltered.pkl \
  --diseased_pickle /raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered.pkl \
  --cell_ge_file /raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv \
  --all_cells    /raid/home/joshua/data/MultiDCP/data/ccle_tcga_ad_cells.p \
  --use_split_file \
  --splits_base_path /raid/home/public/chemoe_collab_102025/PDGrapher/data/processed/splits/chemical \
  --model_name best_model_linearenc.pt
```
`--gpu`/`CUDA_VISIBLE_DEVICES` chosen to leave others free (Hard Rule 5; all 8
GPUs idle at scoping time). Do NOT overwrite `best_model.pt` — write a new file so
the frozen baseline stays auditable until the gate passes. `--max_epoch 100` is a
proposal; raise toward the original 300 only if dev Pearson hasn't plateaued.

---

## 3. Data availability + compute cost + feasibility verdict

### Data (all present on disk, verified)
| Input | Path | Status |
|---|---|---|
| train pickle | `…/MultiDCP_pdg/data/pdg_brddrugfiltered.pkl` | present (15 GB) |
| diseased pickle | `…/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered.pkl` | present (15 GB) |
| drug file | `…/MultiDCP_pdg/data/all_drugs_pdg.csv` | present (381 KB) |
| gene vector | `/raid/home/joshua/data/MultiDCP/data/gene_vector.csv` | present (1.4 MB) |
| cell GE avg | `…/pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv` | present (2.0 MB) |
| all_cells | `/raid/home/joshua/data/MultiDCP/data/ccle_tcga_ad_cells.p` | present (303 KB) |
| splits | `/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed/splits/chemical/{A375,A549,…}` | present (10 cells) |

No missing inputs. The launcher's `GENE_FILE` points at
`/raid/home/joshua/data/MultiDCP/data/gene_vector.csv` (present); the script's
argparse default points at `/raid/home/yoyowu/...` (also present) — use the
explicit launcher path to avoid the yoyowu dependency.

### Compute cost (estimate from the Jan-17 HT29 log)
- The HT29 single-cell, 50-epoch run spanned roughly 08:28→13:06 same day in the
  surrounding artifacts (≈ 4–4.5 h wall clock), i.e. **~5 min/epoch on one GPU**
  at batch 32. tqdm rates were not captured to the log, so this is a timestamp
  estimate, not a measured it/s.
- **Single cell, 100 epochs: ~8 h on 1 GPU** (best-dev save means the usable
  checkpoint typically lands earlier). The LinearEncoder is cheaper than the
  transformer per step, so this is an upper bound.
- 1 GPU, ~30 GB free each (32 GB cards, all idle). Leaves 7 GPUs free → Hard
  Rule 5 satisfied trivially.
- Full 9-cell × 5-fold sweep would be ~45× this and is **not recommended** /
  not needed for the spatial cache.

### Feasibility verdict: **RUNNABLE.**
All data, the script, the env, the toggle, and free GPUs are present. The only
caveat is the lack of a recorded original command/seed for byte-exact
reproduction — not a blocker, because the acceptance gate (§4) is defined on the
NEW checkpoint's behavior + accuracy, not on matching the old run.

---

## 4. Validation gate (run on the NEW checkpoint BEFORE any downstream re-cache)

Reuse the basal-sensitivity logic in
`spatial_tests/scripts/diagnose_wiring_vs_weights.py` (§3 decisive test): build
the model the upstream way with `linear_encoder_flag=True`, strict-load the new
checkpoint, run the upstream-native forward on the two clearly-different real
liver basals (human periportal vs pericentral, Pearson ≈ 0.33), and measure how
much `cell_hidden` and the prediction move. Two conditions, BOTH must pass:

**Gate 4a — basal sensitivity (the fix worked).** The retrained encoder must move
its output for clearly-different basals, far above the current dead ~2e-8:
- `cell_hidden` (50-d): **max|Δ| ≥ 1e-2** between periportal and pericentral
  (current: 1.7e-8). Stretch target: comparable to the live drug branch's
  0.08–0.13.
- prediction (10716-d): **max|Δ| ≥ 1e-3** AND **Pearson(pred_PP, pred_PC) < 0.999**
  (current: max|Δ| 3.1e-10, Pearson 1.000000 — pred is byte-identical across
  basals today).
- Sanity on the maximal-contrast synthetic probe (P2 §7): two Pearson −1.0
  embeds must NOT map to the same `cell_hidden` (the transformer's fixed-point
  failure must be gone).

**Gate 4b — no accuracy regression (the fix didn't break prediction).** Using the
repo's own eval metric (`multidcp_chemoe_ae_de_pdg.py` DE diagnostics:
`allgenes_pearson_mean` for model selection, and **DE top-k R²** at k=20/40/80,
the "~0.75" reference):
- Test **DE top-20 R² ≥ 0.70** (current transformer checkpoint logged ≈ 0.74–0.75
  on HT29; require within ~0.05).
- Test **DE all-genes Pearson ≥ 0.65** (current dev all-genes Pearson plateaued
  ≈ 0.69).
- If 4b fails, the linear encoder under-fits → raise `--max_epoch`, or revisit
  candidate B (which needs parent-repo authorization beyond this spike).

Only when 4a AND 4b pass does the checkpoint advance to §5. If 4a passes but 4b
fails, do NOT cache — escalate.

---

## 5. Downstream re-integration (spatial project only)

After a checkpoint passes both gates (no change to `region_signature.py` forward
logic — the I/O contract holds; §1):

1. **Flip the encoder flag + repoint the checkpoint.** In
   `src/spatial/region_signature.py:670` set `linear_encoder_flag=True`; update
   the checkpoint path (L635/L680 region and `configs/liver_p2.yaml:checkpoint_path`)
   to the new `best_model_linearenc.pt`.
2. **Update `configs/liver_p2.yaml`:** new `checkpoint_path` + recomputed
   `checkpoint_sha256`. Re-verify the gene-order SHA is unchanged (gene set is
   unchanged → it is).
3. **Update `MANIFEST.md` row-17:** new path + new SHA256 + a note that this
   checkpoint reverses the frozen-baseline rule under professor authorization
   (cell-encoder fix), superseding `fbee15fa…`.
4. **Re-run the region DE cache (human + mouse):**
   `conda run -n dili_v04_env python scripts/cache_region_de.py --gpu auto --species all`
   (writes `data/processed/spatial/region_de_cache/{human,mouse}/`). The mouse
   cache must include the APAP perturbation ids for the gate.
5. **Re-run the APAP validity anchor (Halt Gate 3):**
   `conda run -n dili_v04_env python scripts/run_apap_validation.py …` — pericentral
   Pearson vs the GSE272564 APAP24h−APAP0h anchor, threshold 0.3 (D-08). With a
   live cell encoder this is the real test of whether zonal contrast now appears
   in the *predicted* signature; the previous fired-gate analysis was explicitly
   conditioned on the dead encoder.
6. **Re-confirm strict-load** (0 missing / 0 unexpected) on the new checkpoint as
   the load smoke-test before trusting the cache.

No code change is needed in `region_signature.py` beyond the flag flip + path/SHA
because the forward signature and output shapes are identical (§1).

---

## Summary verdict

- **Recommended minimal edit:** retrain with `--linear_encoder_flag` (registry
  `linear_encoder_flag=True`) → swaps the collapsing `TransformerEncoder` for the
  existing `LinearEncoder` (`multidcp_pdg.py:55-84`). **Zero parent-repo source
  change.** Spatial-side: one-line flag flip at `region_signature.py:670` + new
  checkpoint path/SHA in `configs/liver_p2.yaml` and `MANIFEST.md`.
- **Feasibility:** **RUNNABLE** — all data, script, `mdcp_env`, toggle, and 8 idle
  GPUs present. Caveat: original training command/seed not recorded (not a
  blocker; gate is behavior-based).
- **Compute:** ~5 min/epoch on 1 GPU; **single cell, 100 epochs ≈ 8 h, 1 GPU**
  (timestamp estimate, not measured it/s). Full sweep unnecessary.
- **Validation gate:** 4a basal sensitivity — `cell_hidden` max|Δ| ≥ 1e-2 and
  prediction max|Δ| ≥ 1e-3 with Pearson(pred_PP,pred_PC) < 0.999 (vs current
  ~2e-8 / 1.000000); 4b accuracy — Test DE top-20 R² ≥ 0.70 and DE all-genes
  Pearson ≥ 0.65 (vs current ~0.75 / ~0.69). Both must pass before re-caching.

## Honest uncertainty (this gates a real training run)

- The ~8 h cost is a timestamp-derived estimate; tqdm rates weren't logged.
  Measure it/s in the first epoch and re-estimate before committing to 300 epochs.
- The LinearEncoder *can* preserve basal variance, but it is not guaranteed the
  training signal will *teach* it to use the basal strongly — the DE label may be
  largely drug-driven in this dataset. Gate 4a is exactly the check for this; if
  4a fails even with the linear encoder, the issue is data/label-side, not
  architectural, and candidate B won't help either.
- 4b thresholds assume the transformer's ~0.75 R² is reproducible with the MLP
  encoder on the same data; if the MLP under-fits, raise epochs before concluding
  regression.
