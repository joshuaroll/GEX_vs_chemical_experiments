"""Unit tests for `src/spatial/eda/floor.py`.

In-memory fixtures only -- no real data files are read. Synthetic fingerprint
matrices with known separability are used.

Behaviors covered:
  1. Floor AUROC > 0.5 on a linearly separable synthetic fingerprint matrix.
"""

from __future__ import annotations

import numpy as np

from src.spatial.eda.floor import compute_floor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_separable_fps() -> tuple[np.ndarray, np.ndarray]:
    """Return a (20, 2048) uint8 fingerprint matrix with binary labels.

    First 10 rows: bit pattern A (bits 0..9 set) -> label 1.
    Last 10 rows: bit pattern B (bits 10..19 set) -> label 0.
    The two classes are perfectly linearly separable.
    """
    rng = np.random.default_rng(42)
    fps = np.zeros((20, 2048), dtype=np.uint8)
    # Class 1: first 10 drugs have bits 0..9 set
    fps[:10, :10] = 1
    # Class 0: last 10 drugs have bits 10..19 set
    fps[10:, 10:20] = 1
    y = np.array([1] * 10 + [0] * 10, dtype=np.int32)
    return fps, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_floor_auroc_above_chance() -> None:
    """Structure-only floor AUROC > 0.5 on a separable fingerprint matrix."""
    fps, y = _make_separable_fps()

    result = compute_floor(fps, y, seeds=(0, 1, 2))

    assert result.lr_auroc > 0.5, (
        f"Expected logistic regression AUROC > 0.5, got {result.lr_auroc:.3f}"
    )
    assert result.rf_auroc > 0.5, (
        f"Expected random forest AUROC > 0.5, got {result.rf_auroc:.3f}"
    )
