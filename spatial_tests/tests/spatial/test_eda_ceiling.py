"""Unit tests for `src/spatial/eda/ceiling.py`.

In-memory fixtures only -- no real data files are read.

Behaviors covered:
  1. participation_ratio: identity-covariance (100, 50) matrix gives PR in (30, 50].
  2. per_gene_mi: returns shape (n_genes,) for an (n_drugs, n_genes) input.
  3. compute_ceiling returns 'ceiling_auroc' >= floor AUROC on separable synthetic data.
"""

from __future__ import annotations

import numpy as np
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
