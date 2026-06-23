"""Unit tests for `src/spatial/eda/bootstrap.py`.

In-memory fixtures only -- no real data files are read. Synthetic arrays
are provided inline.

Behaviors covered:
  1. Paired bootstrap CI has nonzero width (ci_upper > ci_lower).
  2. When floor probs == ceiling probs, gap_observed == 0 and gate_fires == True.
"""

from __future__ import annotations

import numpy as np

from src.spatial.eda.bootstrap import paired_bootstrap_auroc_gap


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bootstrap_ci_nonzero_width() -> None:
    """Paired bootstrap CI is nonzero-width on meaningful synthetic data."""
    rng = np.random.default_rng(42)
    n = 50
    y = np.array([1] * 25 + [0] * 25, dtype=np.int32)

    # Ceiling probs: better separation
    ceil_probs = np.concatenate([
        rng.uniform(0.6, 1.0, size=25),
        rng.uniform(0.0, 0.4, size=25),
    ])
    # Floor probs: weaker separation
    floor_probs = np.concatenate([
        rng.uniform(0.4, 0.8, size=25),
        rng.uniform(0.2, 0.6, size=25),
    ])

    result = paired_bootstrap_auroc_gap(
        y, floor_probs, ceil_probs, n_resamples=1000, seed=42
    )

    assert result.ci_upper > result.ci_lower, (
        f"Expected CI nonzero width: ci_lower={result.ci_lower:.4f}, "
        f"ci_upper={result.ci_upper:.4f}"
    )


def test_bootstrap_identical_floor_ceil_fires_gate() -> None:
    """Identical floor == ceil -> gap_observed == 0 and gate_fires == True."""
    rng = np.random.default_rng(0)
    n = 50
    y = np.array([1] * 25 + [0] * 25, dtype=np.int32)

    # Both floor and ceiling are the SAME probabilities
    probs = rng.uniform(0.3, 0.7, size=n)
    floor_probs = probs.copy()
    ceil_probs = probs.copy()

    result = paired_bootstrap_auroc_gap(
        y, floor_probs, ceil_probs, n_resamples=200, seed=1
    )

    assert result.gap_observed == 0.0, (
        f"Identical probs should yield gap_observed=0, got {result.gap_observed}"
    )
    assert result.gate_fires is True or bool(result.gate_fires), (
        "gate_fires should be True when gap CI includes 0"
    )
