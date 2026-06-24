# Phase 2: MultiDCP wiring & toxicity head - Pattern Map

**Mapped:** 2026-06-24
**Files analyzed:** 11 (3 src, 1 config, 1 script, 5 tests, 1 conftest)
**Analogs found:** 11 / 11 (every new/modified file has a strong in-repo analog)

All analog excerpts below are from this repo unless noted. Cross-project model code lives in
`/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/` and is a **read-only static import** (RESEARCH §Cross-project import landmine).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/spatial/region_signature.py` (MODIFY) | service (frozen-model wrapper + pure cache) | request-response (forward pass) + transform (DE) | itself (fill the 2 stubs); `src/spatial/region_diagnostics.py` (cross-project sys.path import idiom) | exact (self-extend) |
| `src/spatial/tox_head.py` (NEW) | model (trainable nn.Module) | transform (concat-MLP) | `src/spatial/region_combiner.py` (`AttentionPoolCombiner`) | exact (same nn.Module + NamedTuple-output style) |
| `src/spatial/apap_validation.py` (NEW) | service (validation analysis) | transform (zone DE → per-zone Pearson) | `src/spatial/eda/region_diagnostics.py::human_mouse_liver_correlation` + `eda/ceiling.py` | exact (ortholog-mapped Pearson) |
| `configs/liver_p2.yaml` (NEW) | config | — | path-block in `scripts/run_p1_eda.py` (lines 50-77) | role-match (no YAML exists yet) |
| `scripts/cache_region_de.py` (NEW) | script (driver) | batch (loop over drug×region) | `scripts/run_p1_eda.py` (driver skeleton, sys.path, MANIFEST assert, `_write_report`) | exact (driver template) |
| `scripts/smoke_train_condA.py` (NEW) | script (driver) | batch (train loop) | `scripts/run_p1_eda.py` (driver skeleton + argparse + CUDA-hygiene block to add) | role-match |
| `tests/spatial/test_region_signature.py` (MODIFY) | test | — | itself (extend) | exact |
| `tests/spatial/test_model_load.py` (NEW, gpu-marked) | test (real-inference smoke) | — | no in-repo gpu test yet; mirror fixture style of `test_region_signature.py` | role-match |
| `tests/spatial/test_tox_head.py` (NEW) | test | — | `tests/spatial/test_region_combiner.py` | exact |
| `tests/spatial/test_apap_validation.py` (NEW) | test | — | `tests/spatial/test_eda_region.py` + `test_pseudobulk.py` (AnnData fixtures) | exact |
| `tests/spatial/conftest.py` (NEW) | test (shared fixtures) | — | fixture block in `tests/spatial/test_region_signature.py` (lines 53-69) | role-match |

---

## Pattern Assignments

### `src/spatial/region_signature.py` (MODIFY — service, forward + transform)

**Analog:** itself (the file already has the pure DE/cache machinery; Phase 2 fills `load_model` + `_call_model` and flips rule A → rule B). Cross-project import idiom analog: `src/spatial/eda/region_diagnostics.py` (lazy/guarded import) and `scripts/run_p1_eda.py` lines 43-44 (sys.path).

**Six concrete changes, each with the line(s) to mirror or replace:**

1. **Generalize `N_LANDMARK` (D-01).** Keep the 978 default but stop hardcoding it as the only contract. Current (lines 67-70):
```python
N_LANDMARK: Final[int] = 978
"""Number of LINCS L1000 landmark genes. ..."""
```
Add alongside (mirror `region_combiner.py` lines 33-36, which already names both): `N_PDG: Final[int] = 10716`. The cacher takes `gene_ids` of any length — the only hard 978 assertion is the constructor docstring (line 487 "Must have `len(gene_ids) == N_LANDMARK`"); relax it so a 10,716 tuple is accepted. No code currently *enforces* 978, so this is a docstring + new-constant change, not a logic change.

2. **Rule A → rule B in `compute_de` (D-02).** Current body (lines 339-356) subtracts `region_basal`:
```python
pt = np.asarray(predicted_treated, dtype=np.float32)
rb = np.asarray(region_basal, dtype=np.float32)
# ... shape checks ...
return pt - rb
```
Rule B subtracts a **second predicted vector** (`predicted_control`), not the raw basal. Rename the second operand to `predicted_control` and keep the identical shape-check structure (lines 342-354 — copy verbatim, just rename `rb`→`pc` and the error strings). RESEARCH §Code-Examples "rule-B DE at the cache layer":
```python
de_vector = predicted_treated - predicted_control   # rule B (NOT - region_basal)
```

3. **Update the `de_convention` manifest string (D-02).** Current (line 285):
```python
de_convention="predicted_treated - region_basal",
```
becomes `de_convention="predicted_treated(drug) - predicted_control(region_basal)"`.

4. **Extend the cache to 3 vectors (D-03).** Current `RegionSignatureCache` (lines 155-169) holds `de_array` only:
```python
class RegionSignatureCache(NamedTuple):
    de_array: np.ndarray
    manifest: RegionSignatureManifest
```
Add `treated_array` and `control_array` (same `(n_pert_ids, n_regions, n_genes)` shape, float32). `assemble_cache` (lines 364-449) must allocate all three (`np.empty` at line 421) and fill `treated_array[i,j]=pred`, `control_array[i,j]=pred_control`, `de_array[i,j]=compute_de(pred, pred_control)`. Keep the exact double-loop + KeyError/ValueError guards (lines 423-440).

5. **Fill `load_model` (WIRE-01).** Replace the `NotImplementedError` (lines 550-557). Use RESEARCH §Code-Examples verbatim (it is `[VERIFIED: strict-load]`):
```python
MDCP_SRC = "/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src"
sys.path.insert(0, os.path.join(MDCP_SRC, "models"))
sys.path.insert(0, os.path.join(MDCP_SRC, "utils"))
from multidcp_ae_pdg_utils import initialize_model_registry
import multidcp_chemoe_pdg as mc
reg = initialize_model_registry()
reg.update({"num_gene": 10716, "pert_idose_input_dim": 2,
            "dropout": 0.3, "linear_encoder_flag": False})
model = mc.MultiDCP_CheMoE_AE(device=self.device, model_param_registry=reg).double()
state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
model.load_state_dict(state, strict=True)        # expect 0/0
model.eval()
for p in model.parameters(): p.requires_grad_(False)
self._model = model
```
Checkpoint path (MANIFEST row 17, D-04 amended): `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt` (SHA `fbee15f…`). **Do NOT** use the stale `dili_downstream/trained_models/...best.pt` path in the current docstring (lines 521-534) nor the `chemoe_kpgt_…` row-18 checkpoint (collapsed/incompatible).

6. **Fill `_call_model` (WIRE-01).** Replace `NotImplementedError` (lines 604-608). Contract from the existing docstring (line 593, keep it): "Return the raw predicted treated GEX … Do NOT subtract the basal here." RESEARCH §Code-Examples verbatim:
```python
def _call_model(self, smiles, region_basal):
    drug, mask = self._featurize_drug(smiles)              # Molecules + convert_smile_to_feature
    basal = torch.as_tensor(region_basal, dtype=torch.float64,
                            device=self.device).unsqueeze(0)   # [1, 10716], already 0-1 normalized by caller
    dose = torch.tensor([[1.0, 0.0]], dtype=torch.float64, device=self.device)  # fixed 2-dim (Pitfall 4)
    with torch.no_grad():
        pred, _ = self._model(input_cell_gex=basal, input_drug=drug,
                              input_gene=self._gene_tensor, mask=mask,
                              input_pert_idose=dose, job_id="perturbed", epoch=0)
    return pred.squeeze(0).float().cpu().numpy()           # [10716] absolute treated
```
The control pass (rule B operand) is the same call with `CONTROL_INPUT` (inert/empty drug per D-02 amendment — Open Q2). Cast everything to **float64** (model is `.double()`, Pitfall 2); normalize basal to manifold **[0,1]** *before* this call (Pitfall 1 — the silent-corruption bug).

**Purity discipline (RESEARCH §Cross-project import landmine):** keep `compute_de`/`assemble_cache`/`build_manifest`/`make_cache_key` import-clean (no torch). All torch + sys.path injection lives only inside `load_model`/`_call_model`/`_featurize_drug`, so the pure functions stay fixture-testable. This mirrors how `region_diagnostics.py` guards its squidpy import inside the function (lines 116-119), not at module top.

---

### `src/spatial/tox_head.py` (NEW — model, transform)

**Analog:** `src/spatial/region_combiner.py` (`AttentionPoolCombiner`, lines 93-160). Same conventions: `nn.Module`, a `NamedTuple` output container, shape-validated `forward`, dimension args validated in `__init__`.

**Output-container pattern** (mirror `RegionCombinerOutput`, region_combiner.py lines 70-85):
```python
class ToxHeadOutput(NamedTuple):
    logit: torch.Tensor           # (B,) or (B,1)
    attn_weights: torch.Tensor | None   # passthrough from the combiner, for P6
```

**`__init__` validation pattern** (mirror region_combiner.py lines 119-131):
```python
def __init__(self, d_gex: int = 10716, d_chem: int = 2048, d_dr: int = 0,
             d_proj: int = 256, hidden=(256, 64), dropout: float = 0.3):
    super().__init__()
    if d_gex <= 0: raise ValueError(...)
    self.proj_gex  = nn.Linear(d_gex,  d_proj)
    self.proj_chem = nn.Linear(d_chem, d_proj)
    self.proj_dr   = nn.Linear(d_dr,   d_proj) if d_dr > 0 else None
    # 3-layer MLP: GELU + dropout + batchnorm (CON-tox-head)
```

**Zero-tensor channel masking (WIRE-02 / RESEARCH Pattern 3):** condition A feeds `torch.zeros(B, d_gex)` to `proj_gex`; the dose-response channel is zero this phase. The head must accept a zeroed channel and still produce a finite logit. Keep the `forward` shape-guard idiom from region_combiner.py lines 148-152 (`if x.dim() < 2: raise ValueError(...)`).

**Combiner feed (WIRE-02 acceptance):** `AttentionPoolCombiner(d=10716)(x).pooled` → `proj_gex`. Pull `pooled` and `attn_weights` from the `RegionCombinerOutput` NamedTuple (region_combiner.py lines 159-160); pass `attn_weights` through to `ToxHeadOutput` for P6.

**Chem channel (RESEARCH A1, Claude's Discretion):** ECFP4 (2048-d, P1's `smiles_to_ecfp4`) → `proj_chem` for the smoke-train; defer ChemBERTa-vs-ECFP4 to P4.

---

### `src/spatial/apap_validation.py` (NEW — service, zone DE → per-zone Pearson)

**Primary analog:** `src/spatial/eda/region_diagnostics.py::human_mouse_liver_correlation` (lines 266-340) — this is a near-exact template: ortholog-mapped, intersection-only, `scipy.stats.pearsonr`, reports `n_genes_compared`, raises if `< 2` matched. The Halt-Gate-3 per-zone Pearson is this function applied per zone over predicted-vs-measured DE.

**Ortholog-mapped intersection Pearson** (mirror region_diagnostics.py lines 307-340, D-08 "flag-not-zero"):
```python
from scipy.stats import pearsonr
def zone_pearson(pred_de, measured_de, present_mask):
    idx = np.where(present_mask)[0]                 # absent genes flagged, NOT zeroed (D-08)
    if idx.size < 2: raise ValueError(...)          # mirror region_diagnostics.py:331-337
    r, p = pearsonr(pred_de[idx], measured_de[idx])
    return {"pearson_r": float(r), "p_value": float(p), "n_genes_compared": int(idx.size)}
# Halt Gate 3 (D-08/D-09): r['pericentral'] < 0.3 → write HALT_REASON.md, stop-and-REFRAME
```

**Mouse→human ortholog mapping (Pitfall 5, WIRE-03):** mouse Visium symbols → human via `OrthologTable.pairs` (`human_symbol`/`mouse_symbol` columns). Build the lookup exactly as region_diagnostics.py lines 310-313:
```python
h_to_m = dict(zip(ot["human_symbol"], ot["mouse_symbol"]))
h_idx = {g: i for i, g in enumerate(human_genes)}
m_idx = {g: i for i, g in enumerate(mouse_genes)}
```
Ortholog source: `data/processed/spatial/orthologs_h_m_r_one2one.tsv` (P0 output, 15,956 pairs) via `src.spatial.orthology.build_one2one_orthologs`.

**Zone assignment (D-07):** published annotation if present, else canonical markers (pericentral Glul/Cyp2e1; periportal Sds/Cyp2f2). Produce per-spot labels, then `pseudobulk.from_anndata(adata, obs_col="zone", agg="mean")` (pseudobulk.py lines 235-297) per arm.

**Measured DE = APAP_zone − ctrl_zone (WIRE-03):** pseudobulk each arm separately, subtract. **Pitfall 6:** GSE272564 carries BOTH arms in one `RAW.tar` — split spots by GEO sample metadata at load (control vs APAP) before pseudobulking, or treated leaks into the control reference. Then `align_to_gene_space(measured_de, mouse→human symbols, target=10716_symbols, missing="nan")` (gene_alignment.py lines 105-112) so absent genes are NaN-flagged, matching D-08.

**Scanpy import discipline (Pitfall 9):** keep `import scanpy` inside the load function, never at module top (mirror region_diagnostics.py line 116 guarded-import idiom; `ceiling.py` lines 25-31 explicitly avoids top-level scanpy).

---

### `configs/liver_p2.yaml` (NEW — config)

**Analog:** the path/constant block in `scripts/run_p1_eda.py` lines 50-77 (no YAML exists in the repo yet — this is the first config file; mirror the *content* of that block as YAML keys). Capture: checkpoint path (`MultiDCP_CheMoE_pdg/src/best_model.pt`), gene-order file (`data/processed/spatial/multidcp_10716_symbols.txt`), ortholog TSV, APAP `RAW.tar` paths, fixed dose one-hot `[1.0, 0.0]`, basal-normalization params (manifold [0,1]), smoke-train epochs, wandb project/run name, control-input definition (D-02). Pin both checkpoint SHA and gene-order SHA (Security Domain: assert SHA at load).

---

### `scripts/cache_region_de.py` (NEW — driver, batch)

**Analog:** `scripts/run_p1_eda.py` (the canonical driver template). Mirror these blocks exactly:

- **sys.path injection** (run_p1_eda.py lines 43-44):
```python
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
```
- **ROOT / paths block** (lines 50-77) — adapt to checkpoint + cache-output paths.
- **MANIFEST provenance assert** (lines 82-118, `_assert_manifest_rows`) — assert the checkpoint + gene-order file are MANIFEST rows before reading; `sys.exit(2)` if absent.
- **logging.basicConfig** (lines 125-130).
- **library imports after sys.path** with `# noqa: E402` (lines 136-157).
- Drives `RegionSignatureCacher(model_variant="multidcp_chemoe", gene_ids=...)` → `.load_model(ckpt)` → `.run(pert_ids, smiles_map, region_basal_map)`; writes the 3-vector cache (npy + manifest JSON) under `data/processed/spatial/region_de_cache/`.

**CUDA hygiene (Hard Rule 5, Pitfall 7) — NOT in run_p1_eda.py, must add here:** parse `--gpu` and set `os.environ['CUDA_VISIBLE_DEVICES']` **before** `import torch`; auto-detect a free device via `nvidia-smi`; never take the last free GPU. This is the one block with no in-repo analog (run_p1_eda.py is sklearn-only, no torch).

---

### `scripts/smoke_train_condA.py` (NEW — driver, batch)

**Analog:** `scripts/run_p1_eda.py` driver skeleton (argparse sub-command + `_write_report` lines 164-168) + the CUDA-hygiene block from `cache_region_de.py` above. Condition A feeds `torch.zeros(B, 10716)` to the GEX channel (RESEARCH Pattern 3 — no cache, no frozen-model inference), trains only `tox_head.py` on a tiny batch, logs to wandb (≥1 epoch, loss decreases). BCEWithLogits loss.

---

### `tests/spatial/test_region_signature.py` (MODIFY — test)

**Analog:** itself. Extend with `-k rule_b` (synthetic treated/control → `compute_de` subtraction) and `-k cache_three_vector` (assert `treated_array`/`control_array`/`de_array` shapes + `de_convention` string). Keep the in-memory fixture style + purity gate (header lines 19-27: no `/raid`, no torch). Reuse the existing fixtures (`three_genes`, `two_regions`, `two_pert_ids`, lines 53-69) — these move to `conftest.py`.

### `tests/spatial/test_model_load.py` (NEW — gpu-marked real-inference smoke)

**Analog:** no in-repo gpu test yet. Mark `@pytest.mark.gpu`; CANNOT mock (Hard Rule 1). Asserts: strict-load 0/0 into `MultiDCP_CheMoE_AE`; one forward → finite `[10716]` in ~[0,1] (normalization sanity, Pitfall 1). Mirror the small-synthetic-input fixture style of `test_region_signature.py`.

### `tests/spatial/test_tox_head.py` (NEW — test)

**Analog:** `tests/spatial/test_region_combiner.py`. Zero GEX channel → finite logit; per-channel mask shapes; `-k combiner_feed` for `[B,n_regions,10716]→[B,10716]→logit`.

### `tests/spatial/test_apap_validation.py` (NEW — test)

**Analog:** `tests/spatial/test_eda_region.py` + `test_pseudobulk.py` (AnnData fixtures). `-k zone` (marker→label), `-k measured_de` (APAP−ctrl + ortholog align + intersection size), `-k pearson_gate` (per-zone Pearson + Halt-Gate-3 `<0.3` pericentral trigger).

### `tests/spatial/conftest.py` (NEW — shared fixtures)

**Analog:** the fixture block in `test_region_signature.py` lines 53-69. Provide: tiny AnnData with zonation markers, synthetic treated/control vectors, a gene-order slice. Centralizes fixtures across the four Phase-2 test files.

---

## Shared Patterns

### Cross-project model import (sys.path injection, guarded)
**Source:** `scripts/run_p1_eda.py` lines 43-44 (repo sys.path); `src/spatial/eda/region_diagnostics.py` lines 116-119 (function-local guarded import).
**Apply to:** `region_signature.py` (`load_model`/`_call_model`/`_featurize_drug` only), `cache_region_de.py`, `test_model_load.py`.
```python
MDCP_SRC = "/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src"
sys.path.insert(0, os.path.join(MDCP_SRC, "models"))
sys.path.insert(0, os.path.join(MDCP_SRC, "utils"))
```
Keep it inside the impure methods so pure DE/cache functions stay torch-free and fixture-testable.

### Pure-library header + Hard-Rules block
**Source:** every `src/spatial/*.py` header (e.g. pseudobulk.py lines 1-30, ceiling.py lines 1-32, orthology.py lines 1-25).
**Apply to:** `tox_head.py`, `apap_validation.py`.
Open with a module docstring stating purpose, locked policy, and a "Hard rules honored" list (pure library / no hardcoded `/raid` in pure code / DE-rule / no hand-rolled metrics).

### NamedTuple output container
**Source:** `RegionCombinerOutput` (region_combiner.py 70-85), `PseudobulkResult` (pseudobulk.py 57-83), `OrthologTable` (orthology.py 63-89).
**Apply to:** `tox_head.py` (`ToxHeadOutput`), `region_signature.py` (extend `RegionSignatureCache`).

### Ortholog-mapped, intersection-flagged Pearson
**Source:** `region_diagnostics.py::human_mouse_liver_correlation` lines 307-340.
**Apply to:** `apap_validation.py::zone_pearson`. Reuse the `h_to_m`/`h_idx`/`m_idx` lookup, the `< 2` ValueError guard, and the `{pearson_r, p_value, n_genes_compared}` return dict (D-08: flag absent genes, do not zero).

### MANIFEST provenance-before-use + SHA pin
**Source:** `run_p1_eda.py` lines 82-118 (`_assert_manifest_rows`, `sys.exit(2)`).
**Apply to:** `cache_region_de.py`, `apap_validation.py` driver path. Assert checkpoint + gene-order + APAP `RAW.tar` are MANIFEST rows; assert checkpoint SHA at load (Security Domain: the kpgt-vs-CheMoE mixup is exactly the tampering risk this guards).

### Driver skeleton (argparse + logging + report writer)
**Source:** `run_p1_eda.py` lines 25-37 (imports), 125-130 (logging), 164-168 (`_write_report`).
**Apply to:** `cache_region_de.py`, `smoke_train_condA.py`.

---

## No Analog Found

No file in this phase lacks an in-repo analog. Two items are the lowest-coverage spots (role-match only, flagged for the planner):

| File / concern | Role | Reason / action |
|------|------|------|
| `configs/liver_p2.yaml` | config | First YAML in the repo. Mirror the path-constant *content* of `run_p1_eda.py` lines 50-77; no structural YAML precedent to copy. |
| CUDA-hygiene block (`--gpu` before `import torch`) | script idiom | No torch-using script exists in `scripts/` yet (run_p1_eda.py is sklearn-only). Build per Hard Rule 5 / Pitfall 7; no in-repo line to copy. The `mdcp_env`/`dili_v04_env` upstream scripts in `MultiDCP_CheMoE_pdg/src/` are the cross-project reference if needed. |
| `test_model_load.py` (gpu smoke) | test | First `@pytest.mark.gpu` real-inference test; cannot be mocked (Hard Rule 1). Mirror fixture style of `test_region_signature.py`; the *inference* contract comes from RESEARCH §Code-Examples (VERIFIED strict-load), not an in-repo test. |

---

## Metadata

**Analog search scope:** `src/spatial/`, `src/spatial/eda/`, `scripts/`, `tests/spatial/`, plus the verified cross-project model home `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/` (read-only).
**Files scanned (read in full or targeted):** `region_signature.py`, `region_combiner.py`, `pseudobulk.py`, `orthology.py`, `gene_alignment.py`, `eda/ceiling.py`, `eda/floor.py`, `eda/region_diagnostics.py`, `scripts/run_p1_eda.py`, `tests/spatial/test_region_signature.py`, `datasets.py` (APAP registry grep).
**Pattern extraction date:** 2026-06-24
