"""Pure library: paired bootstrap CI of the (ceiling - floor) AUROC gap.

Phase deliverable: EDA-02 Halt Gate 2 operationalization (D-02). Computes the
95% paired percentile CI of the AUROC gap (ceiling - floor) over the shared
drug set using 10,000 bootstrap resamples. Sets gate_fires = True when the CI
lower bound is <= 0 (CI includes 0), meaning the ceiling provides no
statistically distinguishable lift over the floor.

Policy (locked):
    - Default 10,000 resamples (D-02); floor 2,000 acceptable if too slow.
    - Degenerate resamples (all one class) are skipped (gaps[i] = nan; dropped
      before percentile computation); n_resamples_valid records how many
      non-degenerate resamples contributed.
    - CI = np.percentile(gaps, [2.5, 97.5]) (percentile CI, not BCa).
    - gate_fires = ci_lower <= 0. A positive ci_lower means the ceiling
      AUROC is robustly higher than the floor; gate_fires = False in that case.
    - Raises ValueError when y / floor_probs / ceil_probs differ in length.

Hard rules honored:
    - Pure library: NO hardcoded absolute paths, NO real-data paths.
    - No hand-rolled AUROC: sklearn.metrics.roc_auc_score used exclusively.
    - Caller (plan 06) is responsible for intersecting floor and ceiling to the
      SHARED drug set and aggregating ceiling to drug level BEFORE calling
      (Pitfall 8). This function treats its inputs as already drug-level and
      already aligned.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

__all__ = ["BootstrapResult", "paired_bootstrap_auroc_gap"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class BootstrapResult(NamedTuple):
    """Paired bootstrap CI result for the AUROC gap (ceiling - floor).

    Attributes
    ----------
    gap_observed : float
        Observed AUROC gap = AUROC(ceil) - AUROC(floor) on the full shared set.
    ci_lower : float
        2.5th percentile of the bootstrap gap distribution.
    ci_upper : float
        97.5th percentile of the bootstrap gap distribution.
    gate_fires : bool
        True when ci_lower <= 0 (the 95% CI includes 0, i.e., no significant
        lift). Halt Gate 2 fires when this is True.
    n_resamples_valid : int
        Number of non-degenerate bootstrap resamples that contributed to the CI
        (total - skipped due to single-class resamples).
    """

    gap_observed: float
    ci_lower: float
    ci_upper: float
    gate_fires: bool
    n_resamples_valid: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def paired_bootstrap_auroc_gap(
    y: np.ndarray,
    floor_probs: np.ndarray,
    ceil_probs: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> BootstrapResult:
    """Compute paired bootstrap 95% CI of the AUROC gap (ceiling - floor).

    All three arrays must be aligned over the SHARED drug set (intersection of
    drugs with floor predictions and drugs with ceiling predictions). The caller
    (plan 06) is responsible for this alignment and for aggregating the ceiling
    to drug level before calling (Pitfall 8).

    Per resample, ``n`` indices are drawn with replacement. Resamples where the
    resampled ``y`` is single-class (all 0 or all 1) are skipped (AUROC
    undefined). The gap on each valid resample is:

        gap = AUROC(ceil_probs[idx]) - AUROC(floor_probs[idx])

    The 95% CI is the 2.5th and 97.5th percentiles of the valid gap
    distribution. ``gate_fires = ci_lower <= 0``.

    When ``floor_probs == ceil_probs`` (identical arrays), ``gap_observed == 0``
    and the bootstrap CI is symmetric around 0, so ``gate_fires = True``.

    Parameters
    ----------
    y : np.ndarray
        Shape (n,). Binary DILI labels (0/1) aligned to the shared drug set.
    floor_probs : np.ndarray
        Shape (n,). Positive-class predicted probabilities for the floor
        classifier (structure-only baseline) on each drug.
    ceil_probs : np.ndarray
        Shape (n,). Positive-class predicted probabilities for the ceiling
        classifier (measured-signature baseline) on each drug.
    n_resamples : int
        Number of bootstrap resamples. Default 10,000 (D-02); use >= 2,000
        for stable CI estimates.
    seed : int
        Random seed for np.random.default_rng. Default 42.

    Returns
    -------
    BootstrapResult
        Named tuple with gap_observed, ci_lower, ci_upper, gate_fires,
        n_resamples_valid.

    Raises
    ------
    ValueError
        If y, floor_probs, and ceil_probs do not all have the same length.
    """
    y = np.asarray(y)
    floor_probs = np.asarray(floor_probs)
    ceil_probs = np.asarray(ceil_probs)

    if not (len(y) == len(floor_probs) == len(ceil_probs)):
        raise ValueError(
            f"paired_bootstrap_auroc_gap: y, floor_probs, ceil_probs must have "
            f"the same length. Got len(y)={len(y)!r}, "
            f"len(floor_probs)={len(floor_probs)!r}, "
            f"len(ceil_probs)={len(ceil_probs)!r}."
        )

    n = len(y)
    rng = np.random.default_rng(seed)

    # Observed gap on the full shared set
    obs_gap = float(
        roc_auc_score(y, ceil_probs) - roc_auc_score(y, floor_probs)
    )

    # Bootstrap resampling loop
    gaps = np.empty(n_resamples)
    gaps[:] = np.nan

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        # Skip degenerate resamples (single class)
        if y_b.sum() == 0 or y_b.sum() == n:
            continue
        gaps[i] = (
            roc_auc_score(y_b, ceil_probs[idx])
            - roc_auc_score(y_b, floor_probs[idx])
        )

    valid_gaps = gaps[~np.isnan(gaps)]
    n_valid = int(len(valid_gaps))

    ci_lo, ci_hi = float(np.percentile(valid_gaps, 2.5)), float(
        np.percentile(valid_gaps, 97.5)
    )
    gate = bool(ci_lo <= 0)

    log.info(
        "paired_bootstrap_auroc_gap: gap_observed=%.4f CI=[%.4f, %.4f] "
        "gate_fires=%s n_valid=%d / %d resamples",
        obs_gap,
        ci_lo,
        ci_hi,
        gate,
        n_valid,
        n_resamples,
    )

    return BootstrapResult(
        gap_observed=obs_gap,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        gate_fires=gate,
        n_resamples_valid=n_valid,
    )
