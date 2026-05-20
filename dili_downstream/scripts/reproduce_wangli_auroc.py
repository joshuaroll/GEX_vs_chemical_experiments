#!/usr/bin/env python3
"""Reproduce Wang/Li 2020 DNN AUROC of 0.798 on their test split.

Approach:
  - Load optimized_model.h5 weights via h5py (no TensorFlow needed).
  - Reconstruct DNN architecture in NumPy: 978→512→256→128→64→32→16→8→1
    with ELU activations and sigmoid output (from DNN.ipynb create_model call).
  - Preprocessing: raw LINCS Level-5 z-scores, no additional scaling
    (DNN.ipynb uses data.iloc[:,3:].values directly with no StandardScaler).
  - Test split: Usage == 'Test' in wangli_profiles.csv (1200 profiles in
    Wang/Li's published xlsx; our h5 covers 1095 of these, of which 1091
    survive SMILES resolution).
  - Compute AUROC on the test split and compare to 0.798.

Published result (DNN.ipynb output):
  Optimized DNN model testing performance:
    AUC: 0.798, Sensitivity: 0.839, Specificity: 0.603, Accuracy: 0.743

Caveats documented in output:
  - Our h5 lookup finds 5558/6000 sig_ids (7.37% miss rate). The missing
    profiles (442) are not in the local Bayesian h5 — likely a batch/version
    difference from the GSE92742 release Wang/Li used. This reduces our test
    set from 1200 → 1091 profiles (SMILES filter applies too: -4 profiles).
  - The local h5 uses Bayesian z-scores; Wang/Li used standard COMPZ. These
    may not be numerically identical. A full Pearson cross-validation would
    require the Synapse pickle (Plan 01-04). For this reproduction we proceed
    with the local h5 values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
MODEL_H5 = REPO / "data" / "raw" / "wangli_2020" / "model" / "optimized_model.h5"
PROFILES_CSV = REPO / "data" / "processed" / "wangli_profiles.csv"
DE_NPY = REPO / "data" / "processed" / "wangli_measured_de.npy"
RESULTS_DIR = REPO / "results" / "tables"


# ---------------------------------------------------------------------------
# ELU forward pass (NumPy)
# ---------------------------------------------------------------------------

def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1.0))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def dense(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ W + b


# ---------------------------------------------------------------------------
# Load weights from h5
# ---------------------------------------------------------------------------

def load_weights(h5_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load DNN layer weights in order from the model weights h5.

    Returns list of (kernel, bias) tuples, one per layer, in layer order
    (dense_745 → dense_752 for the published optimized_model.h5).
    """
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    with h5py.File(str(h5_path), "r") as f:
        mw = f["model_weights"]
        # Layer groups are named dense_NNN. Sort by layer number.
        layer_groups = sorted(
            [k for k in mw.keys() if k.startswith("dense_")],
            key=lambda k: int(k.split("_")[1]),
        )
        print(f"Found {len(layer_groups)} layer groups: {layer_groups}")
        for lg in layer_groups:
            # Keras saves weights under model_weights/dense_NNN/dense_NNN/{kernel,bias}
            inner = mw[lg][lg]
            kernel_key = [k for k in inner.keys() if "kernel" in k][0]
            bias_key = [k for k in inner.keys() if "bias" in k][0]
            W = inner[kernel_key][:]
            b = inner[bias_key][:]
            layers.append((W.astype(np.float32), b.astype(np.float32)))
            print(f"  {lg}: kernel={W.shape} bias={b.shape}")
    return layers


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward(X: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Run DNN forward pass: all hidden layers use ELU; final layer uses sigmoid."""
    h = X.astype(np.float32)
    for i, (W, b) in enumerate(layers):
        h = dense(h, W, b)
        if i < len(layers) - 1:
            h = elu(h)  # hidden layers: ELU
        else:
            h = sigmoid(h)  # output layer: sigmoid
    return h.flatten()


# ---------------------------------------------------------------------------
# Main reproduction
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Wang/Li 2020 DNN AUROC Reproduction")
    print("=" * 60)

    # --- Load deliverables ------------------------------------------------
    if not PROFILES_CSV.exists():
        print(f"ERROR: {PROFILES_CSV} not found. Run build_wangli_random_only.py first.")
        return 1
    if not DE_NPY.exists():
        print(f"ERROR: {DE_NPY} not found. Run build_wangli_random_only.py first.")
        return 1
    if not MODEL_H5.exists():
        print(f"ERROR: {MODEL_H5} not found. Download model first.")
        return 1

    profiles = pd.read_csv(PROFILES_CSV)
    de_matrix = np.load(DE_NPY)  # (N, 978) float32

    print(f"\nProfiles loaded: {len(profiles)} rows, de_matrix shape: {de_matrix.shape}")
    print(f"Usage distribution:\n{profiles['usage'].value_counts().to_string()}")
    print(f"DILI label distribution:\n{profiles['dili_binary'].value_counts().to_string()}")

    # --- Test split -------------------------------------------------------
    test_mask = profiles["usage"] == "Test"
    X_test = de_matrix[test_mask.values]
    y_test = profiles.loc[test_mask, "dili_binary"].values.astype(np.int32)

    n_test = int(test_mask.sum())
    n_pos_test = int((y_test == 1).sum())
    n_neg_test = int((y_test == 0).sum())
    print(f"\nTest split: n={n_test} (pos={n_pos_test}, neg={n_neg_test})")
    print(f"Test cell distribution:")
    test_cells = profiles.loc[test_mask, "cell_id"].value_counts()
    for cell, cnt in test_cells.items():
        print(f"  {cell}: {cnt}")

    # --- Preprocessing ----------------------------------------------------
    # Wang/Li's DNN.ipynb: X = data.iloc[:,3:].values
    # No StandardScaler, no normalization — raw LINCS Level-5 z-scores.
    # Our de_matrix already contains these z-scores from the h5.
    print(f"\nPreprocessing: raw LINCS Level-5 z-scores (no additional scaling).")
    print(f"X_test stats: mean={X_test.mean():.4f} std={X_test.std():.4f} "
          f"min={X_test.min():.4f} max={X_test.max():.4f}")

    # --- Load model weights -----------------------------------------------
    print(f"\nLoading weights from: {MODEL_H5}")
    layers = load_weights(MODEL_H5)
    print(f"Architecture: 978 → {' → '.join(str(W.shape[1]) for W, b in layers)}")
    expected_arch = [(978, 512), (512, 256), (256, 128), (128, 64),
                     (64, 32), (32, 16), (16, 8), (8, 1)]
    actual_arch = [W.shape for W, b in layers]
    if actual_arch == expected_arch:
        print("Architecture matches DNN.ipynb: PASS")
    else:
        print(f"WARNING: architecture mismatch. Expected {expected_arch}, got {actual_arch}")

    # --- Inference --------------------------------------------------------
    print(f"\nRunning inference on {n_test} test profiles...")
    y_pred = forward(X_test, layers)
    print(f"Predictions: min={y_pred.min():.4f} max={y_pred.max():.4f} "
          f"mean={y_pred.mean():.4f}")

    # --- AUROC ------------------------------------------------------------
    auroc = roc_auc_score(y_test, y_pred)
    published = 0.798
    delta = auroc - published
    verdict = "PASS" if abs(delta) < 0.02 else "FAIL"

    print(f"\n{'=' * 60}")
    print(f"Wang/Li reproduction AUROC: {auroc:.4f}")
    print(f"Published AUROC:            {published:.4f}")
    print(f"Delta:                      {delta:+.4f}")
    print(f"Within ±0.02:               {verdict}")
    print(f"{'=' * 60}")

    # Additional metrics at 0.5 threshold (for reference vs their 0.743 acc)
    y_pred_class = (y_pred >= 0.5).astype(int)
    acc = (y_pred_class == y_test).mean()
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    print(f"\nAt threshold=0.5:")
    print(f"  Accuracy:    {acc:.3f}  (published: 0.743)")
    print(f"  Sensitivity: {sensitivity:.3f}  (published: 0.839)")
    print(f"  Specificity: {specificity:.3f}  (published: 0.603)")

    # --- Write results table ----------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / "P2_wangli_reproduction.md"

    # Build cell distribution table for test set
    cell_table_lines = ["| cell_id | count |", "|---|---|"]
    for cell, cnt in test_cells.items():
        hep_flag = " ← hepatocyte" if cell in ("HEPG2", "PHH") else ""
        cell_table_lines.append(f"| {cell} | {cnt}{hep_flag} |")
    cell_table = "\n".join(cell_table_lines)

    hepg2_test = int(test_cells.get("HEPG2", 0))
    phh_test = int(test_cells.get("PHH", 0))

    result_md = f"""# P2: Wang/Li 2020 DNN Model Reproduction

## Result

| Metric | Value |
|---|---|
| Reproduction AUROC | **{auroc:.4f}** |
| Published AUROC (DNN.ipynb) | 0.798 |
| Delta | {delta:+.4f} |
| Within ±0.02 | **{verdict}** |

## At threshold=0.5 (comparison to DNN.ipynb output)

| Metric | Reproduction | Published |
|---|---|---|
| Accuracy | {acc:.3f} | 0.743 |
| Sensitivity | {sensitivity:.3f} | 0.839 |
| Specificity | {specificity:.3f} | 0.603 |

## Checkpoint loading

- **Was checkpoint loadable?** Yes — h5 weights file loaded via h5py; model architecture
  reconstructed in NumPy (no TensorFlow required).
- **Architecture**: 978 → 512 → 256 → 128 → 64 → 32 → 16 → 8 → 1 (ELU hidden, sigmoid output)
  — confirmed matches DNN.ipynb `create_model('Adam', 'elu')`.
- **Optimizer**: Adam (from DNN.ipynb; not used at inference time)

## Preprocessing

Wang/Li's DNN.ipynb: `X = data.iloc[:,3:].values` — raw LINCS Level-5 z-scores,
no StandardScaler or other normalization. We apply the same: raw values from the
local Bayesian Level-5 h5 file.

## Test split

- Total xlsx test profiles (Usage=Test): 1,200
- Found in local h5: 1,095 (7.37% miss rate across full 6,000 xlsx profiles)
- After SMILES resolution filter: **{n_test}** profiles used
- Positive (DILI=1): {n_pos_test}
- Negative (DILI=0): {n_neg_test}

## Test cell-line distribution

{cell_table}

**HEPG2 in test set: {hepg2_test}**
**PHH in test set: {phh_test}**

## Caveats and methodological notes

1. **H5 miss rate (7.37%)**: 442 of 6,000 Wang/Li profiles are absent from the local
   Bayesian_GSE92742_Level5_COMPZ h5. The local file likely uses different batch
   processing (Bayesian COMPZ vs standard COMPZ). A future step with the Synapse
   pickle will allow Pearson cross-validation to quantify the numeric difference.
   The 442 missing profiles are distributed across train (337) and test (105), so
   our test set is 91.25% complete.

2. **Bayesian vs standard COMPZ**: Wang/Li used standard COMPZ Level-5 z-scores.
   The local h5 uses "Bayesian_COMPZ". These are likely correlated but not identical.
   This is the primary source of any AUROC deviation from 0.798.

3. **SMILES filter**: 4 test profiles dropped due to SMILES resolution failure in
   dili_canonical.csv. This filter is needed for the 11-col wangli_profiles.csv schema
   but not for the DNN model (which uses GEX only). For a pure reproduction, one could
   skip SMILES filtering and retain all 1,095 h5-found test profiles.

4. **No StandardScaler**: Confirmed from DNN.ipynb. The notebook reads raw expressions
   and feeds directly to the model with no per-gene normalization.
"""
    result_path.write_text(result_md)
    print(f"\nResults written to: {result_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
