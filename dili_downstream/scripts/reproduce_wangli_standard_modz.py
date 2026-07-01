#!/usr/bin/env python
"""Reproduce Li/Tong 2020 (fbioe.2020.562677) AUROC ~0.798 on THEIR split, with the
correct STANDARD MODZ data extracted from GSE92742 (the prior 0.51 was the Bayesian variant).

- weights: their published optimized_model.h5 (loaded via h5py; numpy ELU forward, no TF).
- data:    data/processed/wangli_6000_landmark.npz (standard MODZ, 6000 x 978, extracted here).
- split:   Usage == 'Test' (1200 profiles), profile-level, matching their protocol.
- preproc: raw MODZ (their DNN.ipynb feeds values directly). Also reports a StandardScaler
           variant as a diagnostic in case scaling matters.
Published: AUC 0.798, Sensitivity 0.839, Specificity 0.603, Accuracy 0.743.
"""
from __future__ import annotations
import numpy as np, h5py
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

REPO = Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/dili_downstream")
NPZ = REPO / "data/processed/wangli_6000_landmark.npz"
MODEL = REPO / "data/raw/L1000_DILI/optimized_model.h5"


def elu(x, a=1.0): return np.where(x >= 0, x, a * (np.exp(x) - 1.0))
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def load_weights(h5_path):
    layers = []
    with h5py.File(str(h5_path), "r") as f:
        mw = f["model_weights"]
        groups = sorted([k for k in mw.keys() if k.startswith("dense_")],
                        key=lambda k: int(k.split("_")[1]))
        for lg in groups:
            inner = mw[lg][lg]
            W = inner[[k for k in inner if "kernel" in k][0]][:]
            b = inner[[k for k in inner if "bias" in k][0]][:]
            layers.append((W.astype(np.float32), b.astype(np.float32)))
    print(f"loaded {len(groups)} dense layers: {[l[0].shape for l in layers]}")
    return layers


def forward(X, layers):
    h = X.astype(np.float32)
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        h = elu(h) if i < len(layers) - 1 else sigmoid(h)
    return h.flatten()


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    return dict(auroc=roc_auc_score(y, p), sens=sens, spec=spec, acc=accuracy_score(y, yhat))


def main():
    z = np.load(NPZ, allow_pickle=True)
    X, y, usage = z["X"].astype(np.float32), z["label"].astype(int), z["usage"]
    print(f"data {X.shape}  Test={int((usage=='Test').sum())} Train={int((usage=='Training').sum())}")
    layers = load_weights(MODEL)
    assert layers[0][0].shape[0] == X.shape[1], f"gene dim {X.shape[1]} != model input {layers[0][0].shape[0]}"

    te = usage == "Test"
    for tag, Xin in [("raw", X), ("scaled(fit on Train)", None)]:
        if Xin is None:
            sc = StandardScaler().fit(X[usage == "Training"]); Xin = sc.transform(X)
        p = forward(Xin, layers)
        m = metrics(y[te], p[te])
        print(f"\n[{tag}] TEST (n={int(te.sum())}): AUROC={m['auroc']:.4f}  "
              f"Sens={m['sens']:.3f} Spec={m['spec']:.3f} Acc={m['acc']:.3f}")
    print("\npublished: AUC 0.798  Sens 0.839  Spec 0.603  Acc 0.743")
    # verdict on the raw variant (their stated preprocessing)
    p_raw = forward(X, layers); a = roc_auc_score(y[te], p_raw[te])
    verdict = "PASS (in [0.78,0.82])" if 0.78 <= a <= 0.82 else f"OUT of [0.78,0.82] (got {a:.4f})"
    print(f"\nHALT-GATE-1 (raw, Test AUROC): {verdict}")


if __name__ == "__main__":
    main()
