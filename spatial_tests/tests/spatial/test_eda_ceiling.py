"""Unit tests for `src/spatial/eda/ceiling.py`.

In-memory fixtures only -- no real data files are read.

Behaviors covered:
  1. participation_ratio: identity-covariance (100, 50) matrix gives PR in (30, 50].
  2. per_gene_mi: returns shape (n_genes,) for an (n_drugs, n_genes) input.
  3. compute_ceiling returns 'ceiling_auroc' >= floor AUROC on separable synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.spatial.eda.ceiling import compute_ceiling, participation_ratio, per_gene_mi


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_participation_ratio() -> None:
    """Identity-covariance (100, 50) standard-normal matrix has PR in (30, 50]."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 50))
    pr = participation_ratio(X)

    # For iid standard normal, true PR -> n_dims = 50; finite-sample slack.
    assert pr > 30, f"Expected participation ratio > 30, got {pr:.2f}"
    assert pr <= 50, f"Expected participation ratio <= 50, got {pr:.2f}"


def test_per_gene_mi_shape() -> None:
    """per_gene_mi(X (40, 8), y binary) returns shape (8,)."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((40, 8))
    y = np.array([1] * 20 + [0] * 20, dtype=np.int32)

    mi = per_gene_mi(X, y, random_state=42)

    assert mi.shape == (8,), f"Expected shape (8,), got {mi.shape}"


def test_ceiling_ge_floor() -> None:
    """compute_ceiling AUROC >= floor AUROC when ceiling probs better track labels."""
    rng = np.random.default_rng(13)
    n = 40
    y = np.array([1] * 20 + [0] * 20, dtype=np.int32)

    # Ceiling DE: first 5 genes strongly correlated with label (good signal).
    de = np.zeros((n, 10), dtype=np.float32)
    de[:20, :5] = rng.standard_normal((20, 5)) + 2.0   # positive class elevated
    de[20:, :5] = rng.standard_normal((20, 5)) - 2.0   # negative class depressed
    de[:, 5:] = rng.standard_normal((n, 5))             # noise genes

    result = compute_ceiling(de, y)
    ceiling_auroc = result["ceiling_auroc"]

    # Simple floor: random probabilities (baseline ~0.5)
    floor_probs = rng.uniform(0, 1, size=n)
    floor_auroc = roc_auc_score(y, floor_probs)

    assert ceiling_auroc >= floor_auroc, (
        f"Expected ceiling AUROC ({ceiling_auroc:.3f}) >= floor AUROC ({floor_auroc:.3f})"
    )


def test_ceiling_no_drug_leakage() -> None:
    """Drug-grouped CV must stop per-drug memorization from inflating the ceiling.

    Each drug gets a unique constant identity feature dim (repeated across its
    profiles) and a label. Under profile-level CV the model memorizes the drug
    (AUROC -> ~1.0). Under leakage-free drug-grouped CV the held-out drug's
    identity dim is never seen with a label, so drug-level AUROC stays near
    chance. This locks the GroupKFold fix (the 784-profiles-per-drug leakage in
    the real Wang/Li set produced a meaningless 0.56 "ceiling").
    """
    n_drugs = 10
    profiles_per_drug = 8
    rng = np.random.default_rng(0)
    de_rows, labels, names = [], [], []
    for d in range(n_drugs):
        sig = np.zeros(n_drugs, dtype=np.float32)
        sig[d] = 5.0  # unique per-drug identity feature (memorizable only via leakage)
        lab = 1 if d < n_drugs // 2 else 0
        for _ in range(profiles_per_drug):
            de_rows.append(sig + rng.standard_normal(n_drugs).astype(np.float32) * 0.01)
            labels.append(lab)
            names.append(f"drug{d}")
    de = np.asarray(de_rows, dtype=np.float32)
    y = np.asarray(labels, dtype=int)
    profiles = pd.DataFrame({"compound_name": names})

    result = compute_ceiling(de, y, profiles=profiles, n_splits=5)

    assert result["ceiling_auroc"] < 0.75, (
        f"ceiling AUROC {result['ceiling_auroc']:.3f} too high -- drug leakage "
        "not prevented (CV is splitting a drug's profiles across train/test)"
    )
