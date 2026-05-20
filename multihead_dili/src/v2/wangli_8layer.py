"""
v2 — Wang/Li 8-layer DNN implementation
========================================
Replicates the optimized DNN from Wang & Li 2020 (PMID 32681905).

Architecture:
  512→256→128→64→32→16→8→1(sigmoid), all hidden layers ELU activation.
  Optimizer: Adam, loss: binary_crossentropy, metric: AUC.
  Training: balanced class weights, custom monitor checkpoint (same as DNN.ipynb),
  max 100 epochs with EarlyStopping(patience=5) on val_loss.

Notes:
- input_dim is 978 for Run A (measured LINCS DE), 919 for Run B (predicted GEX).
- Seeds set deterministically via tf.random.set_seed + numpy seed.
- Model weights are saved to a temp path per (fold, seed) and deleted after use.
"""

import os
import sys
import tempfile
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# Note: TF 2.21 has a PTX version incompatibility on this box (CUDA_ERROR_UNSUPPORTED_PTX_VERSION)
# when Keras initializes Dense layers. The DNN is small (676K params) so CPU is adequate.
# GPU environment is controlled by the runner script (CUDA_VISIBLE_DEVICES set before import).
# If GPU works, TF auto-selects it; otherwise CPU is used.

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# Wang/Li custom monitor function (faithfully reproduced from DNN.ipynb)
# ---------------------------------------------------------------------------

def _eval_components(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    TP = K.sum(y_pos * y_pred_pos)
    TN = K.sum(y_neg * y_pred_neg)
    FP = K.sum(y_neg * y_pred_pos)
    FN = K.sum(y_pos * y_pred_neg)
    return TP, TN, FP, FN


def monitor_f(y_true, y_pred):
    """Wang/Li custom monitor metric — used for model checkpoint selection."""
    TP, TN, FP, FN = _eval_components(y_true, y_pred)
    part_a = 0.05 * (TN / (TN + FP + K.epsilon()))
    denom = (
        K.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) *
        (TN + FP + K.epsilon())
    )
    part_b = ((TP * TN * TN) - (FP * FN * TN)) / (denom + K.epsilon())
    return part_a + part_b


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def wangli_8layer(input_dim: int) -> tf.keras.Model:
    """
    Build the Wang/Li 8-layer DNN.
    input_dim: 978 (measured) or 919 (predicted).
    """
    m = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation="elu"),
        layers.Dense(256, activation="elu"),
        layers.Dense(128, activation="elu"),
        layers.Dense(64, activation="elu"),
        layers.Dense(32, activation="elu"),
        layers.Dense(16, activation="elu"),
        layers.Dense(8, activation="elu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(
        optimizer=optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["AUC", monitor_f],
    )
    return m


# ---------------------------------------------------------------------------
# Single-run training
# ---------------------------------------------------------------------------

def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_balanced_class_weights(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def train_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    max_epochs: int = 100,
    patience: int = 5,
    batch_size: int = 128,
    tmpdir: str = None,
) -> dict:
    """
    Train one instance of the 8-layer DNN and return test-set predictions + metrics.

    Returns:
        dict with keys: proba_test, y_test, auroc, auprc, mcc, balanced_acc
    """
    set_seeds(seed)
    input_dim = X_train.shape[1]
    model = wangli_8layer(input_dim)

    class_weights = compute_balanced_class_weights(y_train)

    # Checkpoint on best val_monitor_f (Wang/Li's criterion)
    with tempfile.NamedTemporaryFile(
        suffix=".weights.h5",
        dir=tmpdir,
        delete=False,
    ) as tf_handle:
        ckpt_path = tf_handle.name

    cb_checkpoint = callbacks.ModelCheckpoint(
        ckpt_path,
        monitor="val_monitor_f",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    )
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=False,
        verbose=0,
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[cb_checkpoint, cb_early_stop],
        verbose=0,
    )

    # Load best weights (by val_monitor_f)
    if Path(ckpt_path).exists():
        model.load_weights(ckpt_path)
        os.unlink(ckpt_path)

    proba_test = model.predict(X_test, verbose=0).ravel()
    pred_binary = (proba_test >= 0.5).astype(int)

    result = {
        "proba_test": proba_test,
        "y_test": y_test,
        "auroc": float(roc_auc_score(y_test, proba_test)) if len(np.unique(y_test)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_test, proba_test)) if len(np.unique(y_test)) > 1 else float("nan"),
        "mcc": float(matthews_corrcoef(y_test, pred_binary)),
        "balanced_acc": float(balanced_accuracy_score(y_test, pred_binary)),
    }
    tf.keras.backend.clear_session()
    return result


# ---------------------------------------------------------------------------
# Cross-validation runner (returns per-fold predictions)
# ---------------------------------------------------------------------------

def train_with_cv(
    X_all: np.ndarray,
    y_all: np.ndarray,
    pert_ids: np.ndarray,
    fold_indices: list,        # list of (train_idx, val_idx, test_idx) triples
    seeds: list,
    run_label: str,            # "A_measured" or "B_predicted"
    split_label: str,          # "scaffold" or "random"
    out_dir: Path,
    max_epochs: int = 100,
    patience: int = 5,
    batch_size: int = 128,
    tmpdir: str = None,
) -> list:
    """
    Run 5-fold × N-seed DNN experiment.

    fold_indices: list of 5 dicts with keys 'train', 'val', 'test' (integer indices into X_all).
    Returns list of result dicts (one per fold × seed).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    n_folds = len(fold_indices)

    for fold_idx, folds in enumerate(fold_indices):
        X_tr = X_all[folds["train"]].astype(np.float32)
        y_tr = y_all[folds["train"]].astype(np.float32)
        X_va = X_all[folds["val"]].astype(np.float32)
        y_va = y_all[folds["val"]].astype(np.float32)
        X_te = X_all[folds["test"]].astype(np.float32)
        y_te = y_all[folds["test"]].astype(np.float32)
        ids_te = pert_ids[folds["test"]]

        for seed in seeds:
            print(
                f"  Run {run_label} | split={split_label} fold={fold_idx} seed={seed} "
                f"| train={len(X_tr)} val={len(X_va)} test={len(X_te)}"
            )
            res = train_one(
                X_tr, y_tr, X_va, y_va, X_te, y_te,
                seed=seed,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                tmpdir=tmpdir,
            )

            # Save per-run predictions
            fname = f"run_{run_label}_split_{split_label}_fold{fold_idx}_seed{seed}.parquet"
            pred_df = pd.DataFrame({
                "pert_id": ids_te,
                "true_label": y_te.astype(int),
                "predicted_proba": res["proba_test"],
            })
            pred_df.to_parquet(out_dir / fname, index=False)

            all_results.append({
                "run": run_label,
                "split": split_label,
                "fold": fold_idx,
                "seed": seed,
                "auroc": res["auroc"],
                "auprc": res["auprc"],
                "mcc": res["mcc"],
                "balanced_acc": res["balanced_acc"],
                "n_test": len(y_te),
            })
            print(
                f"    => AUROC={res['auroc']:.4f} AUPRC={res['auprc']:.4f} "
                f"MCC={res['mcc']:.4f} BalAcc={res['balanced_acc']:.4f}"
            )

    return all_results
