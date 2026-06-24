"""Pure library: structure-only floor classifier (LR + RF on ECFP4 fingerprints).

Phase deliverable: EDA-01 structure-only floor (D-03). Logistic regression and
random forest classifiers on ECFP4/Morgan fingerprints of SMILES-joined DILIrank
drugs. Reports label entropy, class balance, AUPRC base rate, and floor AUROC
across >=3 seeds. Exposes per-drug out-of-fold probabilities for the paired
bootstrap gap test (plan 06).

Policy (locked):
    - LR: LogisticRegression(max_iter=1000, class_weight='balanced').
    - RF: RandomForestClassifier(n_estimators=300, class_weight='balanced',
      random_state=seed).
    - StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) +
      cross_val_predict(method='predict_proba') for out-of-fold (OOF) probs.
    - AUROC: roc_auc_score(y, oof_prob[:,1]) per seed; mean reported (XC-06).
    - auprc_base_rate: float(y.mean()) == positive prevalence (PR no-skill baseline).
    - Input validation (T-01-05 threat): ValueError if fps.ndim != 2 or
      len(fps) != len(y).

Hard rules honored:
    - Pure library: NO hardcoded absolute paths, NO real-data filenames.
    - No mock or synthetic labels — the caller provides a real fingerprint matrix.
    - No hand-rolled AUROC: sklearn.metrics.roc_auc_score is used exclusively.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

__all__ = [
    "FloorResult",
    "compute_floor",
    "floor_probabilities",
    "floor_profile_disjoint_auroc",
    "floor_profile_disjoint_oof",
    "floor_profile_leaky_auroc",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class FloorResult(NamedTuple):
    """Floor classifier results for the EDA bracket.

    Attributes
    ----------
    lr_auroc : float
        Mean out-of-fold AUROC across seeds for logistic regression.
    rf_auroc : float
        Mean out-of-fold AUROC across seeds for random forest.
    label_entropy : float
        Binary Shannon entropy of the label vector (bits). 0.0 when
        all labels are identical; 1.0 at equal class balance.
    class_balance : float
        Positive fraction y.mean(); value in [0.0, 1.0].
    auprc_base_rate : float
        PR no-skill baseline == positive prevalence (float(y.mean())).
    n_drugs : int
        Total number of drugs passed (len(y)).
    n_drugs_with_smiles : int
        Number of drugs with a valid SMILES (caller passes the covered
        subset; this field records what the caller reported).
    """

    lr_auroc: float
    rf_auroc: float
    label_entropy: float
    class_balance: float
    auprc_base_rate: float
    n_drugs: int
    n_drugs_with_smiles: int


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _binary_entropy(p: float) -> float:
    """Binary Shannon entropy in bits. Returns 0.0 for p in {0, 1}."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def _validate_inputs(fps: np.ndarray, y: np.ndarray) -> None:
    """Raise ValueError if fps or y are malformed (T-01-05 mitigation)."""
    if fps.ndim != 2:
        raise ValueError(
            f"compute_floor: fps must be 2-D (n_drugs x n_bits), "
            f"got fps.ndim={fps.ndim!r}."
        )
    if len(fps) != len(y):
        raise ValueError(
            f"compute_floor: len(fps)={len(fps)!r} != len(y)={len(y)!r}. "
            "fps and y must have the same number of rows."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_floor(
    fps: np.ndarray,
    y: np.ndarray,
    seeds: Sequence[int] = (0, 1, 2),
    n_splits: int = 5,
    n_drugs_with_smiles: int | None = None,
) -> FloorResult:
    """Compute the structure-only AUROC floor via LR + RF on ECFP4 fingerprints.

    Runs StratifiedKFold cross-validation for each seed in `seeds` and averages
    out-of-fold AUROC across seeds (XC-06 multi-seed requirement). Logs the std
    deviation and completion summary.

    Parameters
    ----------
    fps : np.ndarray
        Shape (n_drugs, n_bits), dtype uint8. ECFP4 fingerprint matrix for the
        SMILES-covered DILIrank subset. Must be 2-D.
    y : np.ndarray
        Shape (n_drugs,), dtype int or bool. Binary DILI labels aligned to fps.
    seeds : Sequence[int]
        Random seeds for StratifiedKFold + RF. Minimum 3 required (XC-06).
        Default (0, 1, 2).
    n_splits : int
        Number of folds for StratifiedKFold. Default 5.
    n_drugs_with_smiles : int | None
        Number of drugs in the original label set that had valid SMILES.
        If None, defaults to len(y) (caller passed only covered drugs).

    Returns
    -------
    FloorResult
        NamedTuple with lr_auroc, rf_auroc, label_entropy, class_balance,
        auprc_base_rate, n_drugs, n_drugs_with_smiles.

    Raises
    ------
    ValueError
        If fps.ndim != 2 or len(fps) != len(y).
    """
    _validate_inputs(fps, y)

    if n_drugs_with_smiles is None:
        n_drugs_with_smiles = len(y)

    y = np.asarray(y, dtype=int)
    fps = np.asarray(fps, dtype=np.float32)

    # --- Label statistics ---
    p_pos = float(y.mean())
    label_entropy = _binary_entropy(p_pos)
    auprc_base_rate = p_pos  # PR no-skill baseline == positive prevalence

    # --- Per-seed cross-validation ---
    lr_aurocs: list[float] = []
    rf_aurocs: list[float] = []

    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        lr_oof = cross_val_predict(lr, fps, y, cv=cv, method="predict_proba")
        lr_aurocs.append(float(roc_auc_score(y, lr_oof[:, 1])))

        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        rf_oof = cross_val_predict(rf, fps, y, cv=cv, method="predict_proba")
        rf_aurocs.append(float(roc_auc_score(y, rf_oof[:, 1])))

    lr_mean = float(np.mean(lr_aurocs))
    rf_mean = float(np.mean(rf_aurocs))
    lr_std = float(np.std(lr_aurocs))
    rf_std = float(np.std(rf_aurocs))

    log.info(
        "compute_floor: n_drugs=%d n_drugs_with_smiles=%d "
        "lr_auroc=%.4f (std=%.4f) rf_auroc=%.4f (std=%.4f) "
        "entropy=%.4f balance=%.3f auprc_base=%.3f seeds=%s",
        len(y),
        n_drugs_with_smiles,
        lr_mean,
        lr_std,
        rf_mean,
        rf_std,
        label_entropy,
        p_pos,
        auprc_base_rate,
        list(seeds),
    )

    return FloorResult(
        lr_auroc=lr_mean,
        rf_auroc=rf_mean,
        label_entropy=label_entropy,
        class_balance=p_pos,
        auprc_base_rate=auprc_base_rate,
        n_drugs=len(y),
        n_drugs_with_smiles=n_drugs_with_smiles,
    )


def floor_probabilities(
    fps: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    n_splits: int = 5,
) -> np.ndarray:
    """Return per-drug out-of-fold logistic regression probabilities (positive class).

    Used by plan 06 to supply floor_probs to the paired-bootstrap gap test on the
    shared drug set (ceiling intersection). Output is aligned to the input order of
    fps / y.

    Parameters
    ----------
    fps : np.ndarray
        Shape (n_drugs, n_bits), dtype uint8. ECFP4 fingerprint matrix.
    y : np.ndarray
        Shape (n_drugs,), dtype int or bool. Binary DILI labels.
    seed : int
        Random seed for StratifiedKFold. Default 0.
    n_splits : int
        Number of folds for StratifiedKFold. Default 5.

    Returns
    -------
    np.ndarray
        Shape (n_drugs,), float64. Out-of-fold positive-class probability,
        aligned to input order.

    Raises
    ------
    ValueError
        If fps.ndim != 2 or len(fps) != len(y).
    """
    _validate_inputs(fps, y)

    y = np.asarray(y, dtype=int)
    fps = np.asarray(fps, dtype=np.float32)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    oof_proba = cross_val_predict(lr, fps, y, cv=cv, method="predict_proba")

    log.debug(
        "floor_probabilities: n_drugs=%d seed=%d oof_proba shape=%s",
        len(y),
        seed,
        oof_proba.shape,
    )

    return oof_proba[:, 1].astype(np.float64)


def floor_profile_disjoint_auroc(
    fps: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """Profile-level, drug-disjoint structure AUROC (head-to-head with the ceiling).

    Mirrors ``compute_ceiling``'s leakage-free path exactly so the structure floor
    and the measured ceiling sit on identical drug-disjoint folds at PROFILE
    granularity. Each PROFILE row carries ITS DRUG's ECFP4 fingerprint; folds are
    drawn with StratifiedGroupKFold grouped by the drug key, so a drug's profiles
    never straddle train/test. The AUROC is reported at profile granularity (no
    drug aggregation) -- this is the supporting, better-powered sensitivity view
    (D-06), the floor partner of the ceiling's profile_auroc_disjoint.

    Parameters
    ----------
    fps : np.ndarray
        Shape (n_profiles, n_bits). Each profile row holds its drug's ECFP4
        fingerprint. Must be 2-D.
    y : np.ndarray
        Shape (n_profiles,). Profile-level binary DILI labels aligned to fps.
    groups : np.ndarray
        Shape (n_profiles,). Drug grouping key (lowercased compound_name); a
        drug's profiles share one group so the split keeps them together.
    n_splits : int
        Requested StratifiedGroupKFold folds. Capped at the number of unique
        drugs. Default 5.
    seed : int
        Random seed for the StandardScaler+LR pipeline and the CV splitter.
        Default 42 (matches the ceiling's _leakage_decomposition).

    Returns
    -------
    float
        Profile-level drug-disjoint AUROC.

    Raises
    ------
    ValueError
        If fps.ndim != 2, len(fps) != len(y), or len(groups) != len(y).
    """
    # Single source of truth for the folds: delegate to floor_profile_disjoint_oof
    # and wrap in roc_auc_score so the scalar and the OOF vector can never drift.
    oof = floor_profile_disjoint_oof(fps, y, groups, n_splits=n_splits, seed=seed)
    y_int = np.asarray(y, dtype=int)
    auroc = float(roc_auc_score(y_int, oof))

    n_unique = len(set(np.asarray(groups).tolist()))
    log.info(
        "floor_profile_disjoint_auroc: n_profiles=%d n_drugs=%d auroc=%.4f seed=%d",
        len(y_int),
        n_unique,
        auroc,
        seed,
    )

    return auroc


def floor_profile_disjoint_oof(
    fps: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Per-profile drug-disjoint OOF probabilities for the structure floor.

    Identical folds, pipeline, and degenerate-fallback as
    ``floor_profile_disjoint_auroc`` (which now delegates here), but returns the
    out-of-fold positive-class probability VECTOR instead of the scalar AUROC.
    This lets the profile-level drug-disjoint gap feed a paired bootstrap on
    aligned floor/ceiling OOF vectors over identical rows + drug folds (D-09).

    Parameters
    ----------
    fps : np.ndarray
        Shape (n_profiles, n_bits). Each profile row holds its drug's ECFP4
        fingerprint. Must be 2-D.
    y : np.ndarray
        Shape (n_profiles,). Profile-level binary DILI labels aligned to fps.
    groups : np.ndarray
        Shape (n_profiles,). Drug grouping key (lowercased compound_name); a
        drug's profiles share one group so the split keeps them together.
    n_splits : int
        Requested StratifiedGroupKFold folds. Capped at the number of unique
        drugs. Default 5.
    seed : int
        Random seed for the StandardScaler+LR pipeline and the CV splitter.
        Default 42 (matches the ceiling's _leakage_decomposition).

    Returns
    -------
    np.ndarray
        Shape (n_profiles,), float64. Out-of-fold positive-class probability,
        aligned to input order.

    Raises
    ------
    ValueError
        If fps.ndim != 2, len(fps) != len(y), or len(groups) != len(y).
    """
    _validate_inputs(fps, y)
    if len(groups) != len(y):
        raise ValueError(
            f"floor_profile_disjoint_oof: len(groups)={len(groups)!r} != "
            f"len(y)={len(y)!r}. groups must align with y row-for-row."
        )

    fps = np.asarray(fps, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)

    # Same per-fold pipeline as compute_ceiling: StandardScaler fit per fold +
    # class-weighted LR. Keeps floor and ceiling unit-consistent on the fair view.
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )

    n_unique = len(set(groups.tolist()))
    eff_splits = min(n_splits, n_unique)
    if eff_splits >= 2:
        cv = StratifiedGroupKFold(n_splits=eff_splits, shuffle=True, random_state=seed)
        oof = cross_val_predict(
            clf, fps, y, cv=cv, groups=groups, method="predict_proba"
        )[:, 1]
    else:
        # Degenerate: <2 distinct drugs -- no grouped split possible (mirror ceiling).
        oof = cross_val_predict(
            clf,
            fps,
            y,
            cv=StratifiedKFold(
                n_splits=max(2, min(n_splits, len(y))),
                shuffle=True,
                random_state=seed,
            ),
            method="predict_proba",
        )[:, 1]

    return oof.astype(np.float64)


def floor_profile_leaky_auroc(
    fps: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """Profile-level structure AUROC under a LEAKY random split (NO drug groups).

    The missing floor-leaky cell of the floor x ceiling, leaky x disjoint 2x2
    (D-09). Mirrors ``_leakage_decomposition``'s leaky path EXACTLY: a random
    ``StratifiedKFold`` over profiles with NO ``groups`` argument, so a single
    drug's profiles can straddle train and test and the model can exploit drug
    identity. Reported at profile granularity, this is the floor partner of the
    ceiling's ``profile_auroc_leaky``: if floor-leaky is near ceiling-leaky
    (~0.912), the Wang/Li headline (~0.798) is largely drug-identity
    memorization that chemical structure reproduces.

    Parameters
    ----------
    fps : np.ndarray
        Shape (n_profiles, n_bits). Each profile row holds its drug's ECFP4
        fingerprint. Must be 2-D.
    y : np.ndarray
        Shape (n_profiles,). Profile-level binary DILI labels aligned to fps.
    n_splits : int
        Requested StratifiedKFold folds (floored at 2, capped at len(y)).
        Default 5.
    seed : int
        Random seed for the StandardScaler+LR pipeline and the CV splitter.
        Default 42 (matches the ceiling's _leakage_decomposition).

    Returns
    -------
    float
        Profile-level leaky (random-split, no-groups) AUROC.

    Raises
    ------
    ValueError
        If fps.ndim != 2 or len(fps) != len(y).
    """
    _validate_inputs(fps, y)

    fps = np.asarray(fps, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    # Same per-fold pipeline as the disjoint floor / ceiling (unit-consistent).
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )

    # Random profile split with NO groups -- the leaky setup that lets a drug's
    # profiles straddle train/test (mirror _leakage_decomposition's leaky path).
    cv = StratifiedKFold(
        n_splits=max(2, min(n_splits, len(y))), shuffle=True, random_state=seed
    )
    oof = cross_val_predict(clf, fps, y, cv=cv, method="predict_proba")[:, 1]

    auroc = float(roc_auc_score(y, oof))

    log.info(
        "floor_profile_leaky_auroc: n_profiles=%d auroc=%.4f seed=%d leaky (no groups)",
        len(y),
        auroc,
        seed,
    )

    return auroc
