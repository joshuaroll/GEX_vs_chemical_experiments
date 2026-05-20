---
phase: 02
phase_name: model-gex-training
review_depth: standard
status: findings
files_reviewed: 4
findings:
  critical: 0
  warning: 3
  info: 4
  total: 7
reviewed_at: "2026-05-20"
---

# Phase 02 Code Review — MODEL_GEX Training

**Depth:** standard (per-file analysis with language-specific checks)
**Files reviewed:**
- `multihead_dili/src/eval/de_evaluator.py`
- `multihead_dili/src/eval/__init__.py`
- `multihead_dili/src/train/train_model_gex.py`
- `multihead_dili/scripts/eval_model_gex.py`

---

## Findings

### WR-01 — `train_model_gex.py`: Eval split does not match training split exactly (Warning)

**File:** `src/train/train_model_gex.py` (lines 251–261)  
**File:** `scripts/eval_model_gex.py` (lines 64–68)

**Issue:** The training script and the eval script use slightly different split logic:

Training script (`train_model_gex.py`):
```python
idx_train, idx_temp = train_test_split(idx, test_size=(dev_frac + test_frac), random_state=seed)
dev_share = dev_frac / (dev_frac + test_frac)
idx_dev, idx_test = train_test_split(idx_temp, test_size=(1 - dev_share), random_state=seed)
```

Eval script (`eval_model_gex.py`):
```python
idx_train, idx_temp = train_test_split(idx, test_size=0.15, random_state=args.seed)
dev_share = 0.10 / 0.15
idx_dev, _ = train_test_split(idx_temp, test_size=(1 - dev_share), random_state=args.seed)
```

The eval script hardcodes `test_size=0.15` (the combined dev+test fraction) rather than reading from arguments. This works because 0.10 + 0.05 = 0.15, and both scripts use `random_state=343`, so the dev sets coincide by value. However, the redundancy is fragile: if dev_frac or test_frac defaults change in `train_model_gex.py`, the eval script will silently evaluate on a different set.

**Impact:** The reported mean Pearson 0.3568 is correct for the current run (confirmed by the 16,452-row dev count matching). But the script cannot be trusted independently for future re-evaluation with different fractions.

**Fix:** Share a `build_splits()` utility function, or add `--dev_frac` and `--test_frac` args to `eval_model_gex.py` with defaults matching training.

---

### WR-02 — `train_model_gex.py`: `pickle` import unused (Warning)

**File:** `src/train/train_model_gex.py` (line 47)

```python
import pickle
```

`pickle` is imported but never used in the file. The import originates from the upstream `multidcp_ae_balanceloss.py` template copy-paste and was not cleaned up.

**Impact:** Minor — no functional harm, but creates confusion about whether pickle serialization is used (it is not; checkpoints are saved with `torch.save(model.state_dict(), ...)`). Misleads future maintainers about the serialization strategy.

**Fix:** Remove `import pickle`.

---

### WR-03 — `train_model_gex.py`: `dev_cell_ids` passed to `model_training_ae` but never used inside it (Warning)

**File:** `src/train/train_model_gex.py` (lines 92, 341)

Function signature:
```python
def model_training_ae(args, model, train_loader, dev_loader, dev_cell_ids, metrics_summary):
```

The `dev_cell_ids` parameter is passed in and declared in the function signature but is **never used inside `model_training_ae`**. The training loop instead builds `cell_ids_dev` from the DataLoader output:
```python
cell_ids_dev = []
...
cell_ids_dev.extend(list(cell_id_batch))
```

**Impact:** No functional impact — the dev cell IDs are correctly recovered from the DataLoader. However, the dead parameter is confusing and could mislead future readers into thinking the function uses a pre-partitioned cell list rather than recovering IDs from iteration.

**Fix:** Remove `dev_cell_ids` from the function signature.

---

### IN-01 — `train_model_gex.py`: All MultiDCP paths are hardcoded absolute paths (Info)

**File:** `src/train/train_model_gex.py` (lines 34–35, 182–189)

All upstream data and model paths are hardcoded to `/raid/home/joshua/projects/MultiDCP/MultiDCP/...`. This is expected given the SHA-pinned dependency design (SHA 871b8de is load-bearing for reproducibility), but creates a portability issue: the script cannot be used on any other machine without editing 7+ absolute paths.

**Noted, not blocking.** The MANIFEST.md correctly records the MultiDCP SHA. Future improvement: introduce a `MULTIDCP_ROOT` environment variable that defaults to the hardcoded path.

---

### IN-02 — `de_evaluator.py`: Spearman computed on random subsample for large datasets (Info)

**File:** `src/eval/de_evaluator.py` (lines 131–139)

```python
sample_indices = range(n_samples) if n_samples <= 5000 else np.random.choice(n_samples, 2000, replace=False)
```

For dev sets larger than 5,000 samples, Spearman correlation is estimated on a random 2,000-sample subset. This is intentional (Spearman is O(n log n) per sample) but means the Spearman metric for large cell lines (e.g., MCF7 with 2,931 dev samples) is an approximation — though at 2,931 < 5,000, the subsample threshold is not triggered in Phase 2.

**Noted for documentation.** If Phase 3 / Phase 4 use larger dev sets, this subsample may kick in and should be noted in results reporting.

---

### IN-03 — `eval_model_gex.py`: Dropout set to 0.0 at eval time (Info — design confirmation)

**File:** `scripts/eval_model_gex.py` (line 92)

```python
'dropout': 0.0,             # eval mode — no dropout
```

The eval script correctly sets dropout to 0.0 AND calls `model.eval()`. Using both `model.eval()` (which disables dropout/batchnorm stochasticity) and `dropout=0.0` (which initializes with no dropout layers) is correct but slightly redundant — `model.eval()` alone would suffice. The `dropout=0.0` ensures deterministic behavior even if `model.eval()` were accidentally omitted.

**No action needed.** Just documenting as defense-in-depth pattern.

---

### IN-04 — `train_model_gex.py`: `metrics_summary` collected but never written to disk (Info)

**File:** `src/train/train_model_gex.py` (lines 337–341)

```python
metrics_summary = defaultdict(list)
...
best_pearson = model_training_ae(args, model, train_loader, dev_loader, dev_cell_ids, metrics_summary)
```

The `metrics_summary` dict accumulates `pearson_list_ae_dev` and `rmse_list_ae_dev` across epochs but is never serialized to disk after training completes. All epoch metrics are logged to WandB, so this is not a data loss (WandB is the persistent store), but the `metrics_summary` dict exists without a consumer and creates a dead-code smell.

**Fix (optional):** Either write `metrics_summary` to `results/tables/P2_metrics_history.json` at the end of training, or remove the collection and rely solely on WandB for epoch history.

---

## Summary

No critical issues. Phase 2 code is functionally correct — halt gate 2 evaluation and checkpoint selection logic are sound. The three warnings are code-quality issues that should be addressed before Phase 3 to prevent confusion when the eval script is reused for Stage-2 feature caching:

1. **WR-01** (split mismatch fragility): Most important to fix before eval_model_gex.py is reused
2. **WR-02** (dead import): 1-line fix
3. **WR-03** (dead parameter): 1-line fix

Recommended action: Apply WR-01 through WR-03 fixes in the Phase 3 prep commit or as a standalone cleanup commit before Phase 3 begins.
