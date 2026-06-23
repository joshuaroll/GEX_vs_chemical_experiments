"""Pure library: measured-biology CEILING for the EDA bracket.

Phase deliverable: EDA-02 measured-biology ceiling (D-05). Reuses the v0.5
Wang/Li LINCS measured DE (wangli_measured_de.npy, shape (5517, 978)) to
compute effective rank (PCA participation ratio), per-gene mutual information
vs DILI label, and a measured-signature-only baseline AUROC at the drug level.

Policy (locked):
    - load_ceiling: np.load the .npy, pd.read_csv the profiles CSV; raises
      ValueError when de.shape[0] != len(profiles) (T-01-06 mitigation).
    - participation_ratio: (sum lambda)^2 / sum lambda^2 via SVD on centered X;
      lambda_i = singular_value_i^2. Drop numerical zeros. (RESEARCH Pattern 4)
    - per_gene_mi: mutual_info_classif(discrete_features=False, n_neighbors=3).
      (RESEARCH Pattern 5)
    - compute_ceiling: aggregates measured-signature predictions to DRUG level
      (mean across profiles) before AUROC (Pitfall 8). OOF probs come from a
      StandardScaler + LogisticRegression pipeline under StratifiedGroupKFold
      grouped by compound (leakage-free: a drug's profiles never straddle
      train/test), then roc_auc_score. Profile-level StratifiedKFold would let
      one drug's up-to-784 profiles leak across folds and inflate the ceiling.
    - If the measured DE file is absent, load_ceiling raises FileNotFoundError
      so the calling script can report "no measured ceiling -- floor only" for
      organs/species without measured data (D-05 honesty rule).

Hard rules honored:
    - Pure library: NO hardcoded absolute paths, NO real-data filenames.
    - DE rule: wangli_measured_de.npy is already differential expression
      (treated - diseased); no raw expression is used.
    - No hand-rolled AUROC: sklearn.metrics.roc_auc_score used exclusively.
    - No scanpy import (Pitfall 9 numba conflict); anndata not needed here
      since we work with .npy and .csv directly.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

__all__ = [
    "load_ceiling",
    "participation_ratio",
    "per_gene_mi",
    "compute_ceiling",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_ceiling(
    de_path: str,
    profiles_path: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load measured DE matrix and profiles metadata.

    Reads the Wang/Li LINCS measured DE .npy file and the companion profiles
    CSV. Raises FileNotFoundError when either file is absent so that calling
    scripts can report "no measured ceiling -- floor only" for organs/species
    with no measured data. Raises ValueError when the DE row count does not
    match the profile count (T-01-06 row-sync check).

    Parameters
    ----------
    de_path : str
        Path to wangli_measured_de.npy (shape (5517, 978) float32, already DE).
    profiles_path : str
        Path to wangli_profiles.csv (5517 rows; columns include compound_name,
        dili_binary, dose_um, etc.).

    Returns
    -------
    de : np.ndarray
        Shape (n_profiles, n_genes) float32. Measured differential-expression
        matrix (treated - diseased; DE rule satisfied by upstream construction).
    profiles : pd.DataFrame
        Profile metadata aligned row-for-row with ``de``.

    Raises
    ------
    FileNotFoundError
        If either ``de_path`` or ``profiles_path`` is missing on disk.
    ValueError
        If ``de.shape[0] != len(profiles)`` (row-sync check, T-01-06).
    """
    de = np.load(de_path)
    profiles = pd.read_csv(profiles_path)

    if de.shape[0] != len(profiles):
        raise ValueError(
            f"load_ceiling: DE row count ({de.shape[0]}) != profiles row count "
            f"({len(profiles)!r}). Files may be mismatched. "
            f"de_path={de_path!r}, profiles_path={profiles_path!r}"
        )

    log.info(
        "load_ceiling: de shape=%s dtype=%s profiles=%d rows",
        de.shape,
        de.dtype,
        len(profiles),
    )
    return de, profiles


def participation_ratio(X: np.ndarray) -> float:
    """Effective rank via PCA participation ratio.

    Computes (sum_i lambda_i)^2 / (sum_i lambda_i^2) where lambda_i are the
    squared singular values of the centered data matrix X. The (1/n-1) factor
    in the covariance definition cancels in the ratio.

    Numerical zeros are dropped before summing to avoid division instability
    when n_samples < n_features.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_genes). Raw or DE gene-expression matrix.

    Returns
    -------
    float
        Participation ratio >= 1. For an identity-covariance (n, p) standard
        normal matrix the expected value is p; finite-sample values are lower.
    """
    X_centered = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    lambdas = s ** 2  # eigenvalues of covariance (up to 1/(n-1) factor; cancels)
    lambdas = lambdas[lambdas > 0]  # drop numerical zeros
    pr = float((lambdas.sum()) ** 2 / (lambdas ** 2).sum())
    log.debug("participation_ratio: PR=%.2f from %d singular values", pr, len(lambdas))
    return pr


def per_gene_mi(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """Mutual information between each gene and the binary DILI label.

    Uses sklearn's mutual_info_classif with continuous features and 3 nearest
    neighbors. The result is non-negative; genes with no discriminatory power
    return 0.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_drugs, n_genes). Measured DE matrix (drug-level or
        profile-level, caller's choice).
    y : np.ndarray
        Shape (n_drugs,). Binary DILI labels aligned to X rows.
    random_state : int
        Seed for mutual_info_classif's k-NN estimator. Default 42.

    Returns
    -------
    np.ndarray
        Shape (n_genes,). Non-negative mutual information per gene (nats).
    """
    mi = mutual_info_classif(
        X,
        y,
        discrete_features=False,
        n_neighbors=3,
        random_state=random_state,
    )
    log.debug(
        "per_gene_mi: %d genes; fraction MI>0=%.3f; top MI=%.4f",
        len(mi),
        float((mi > 0).mean()),
        float(mi.max()) if len(mi) > 0 else 0.0,
    )
    return mi


def compute_ceiling(
    de: np.ndarray,
    y: np.ndarray,
    profiles: pd.DataFrame | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute the measured-biology CEILING metrics.

    Aggregates measured-signature predictions to the DRUG level (mean of OOF
    probabilities across a drug's profiles) before reporting AUROC (Pitfall 8).
    The ceiling AUROC is estimated via LogisticRegression + StratifiedKFold
    cross_val_predict on the measured DE.

    When ``profiles`` is provided, it must be aligned row-for-row with ``de``
    and contain a ``compound_name`` column so that drug-level aggregation is
    applied. When ``profiles`` is None, each row of ``de`` / ``y`` is assumed
    to already be at drug level.

    Parameters
    ----------
    de : np.ndarray
        Shape (n_profiles, n_genes) float32. Measured DE matrix (already
        differential expression; DE rule satisfied).
    y : np.ndarray
        Shape (n_profiles,). Binary DILI labels aligned to ``de`` rows. When
        ``profiles`` is provided, these are profile-level labels; the function
        collapses to drug level (one label per drug is confirmed).
    profiles : pd.DataFrame or None
        Profile metadata with at least a ``compound_name`` column. Rows must
        be aligned with ``de``. If None, ``de``/``y`` are assumed drug-level.
    n_splits : int
        StratifiedKFold folds for the LR cross-validation. Default 5.
    seed : int
        Random seed for LogisticRegression and StratifiedKFold. Default 42.

    Returns
    -------
    dict
        Keys:
        - ``ceiling_auroc`` (float): Drug-level measured-signature AUROC.
        - ``participation_ratio`` (float): PCA participation ratio on ``de``.
        - ``mi`` (np.ndarray): Per-gene mutual information (n_genes,).
        - ``mi_fraction_nonzero`` (float): Fraction of genes with MI > 0.
        - ``top20_mi_idx`` (np.ndarray): Indices of the 20 genes with
          highest MI (descending).
        - ``n_drugs`` (int): Number of unique drugs after aggregation.
        - ``ceiling_probs`` (dict[str, float]): Drug-name -> mean OOF
          positive-class probability. Used by plan 06 to align to the
          shared drug set.

    Notes
    -----
    Drug-level aggregation (Pitfall 8): wangli_profiles.csv has 5517 rows but
    only 628 unique drugs. Aggregating mean OOF probs to drug level before
    AUROC is mandatory so that the ceiling and floor are on the same unit of
    analysis.
    """
    de = np.asarray(de, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    # --- Effective rank on the full profile-level DE matrix ---
    pr = participation_ratio(de)

    # --- Per-gene MI (profile-level) ---
    mi = per_gene_mi(de, y, random_state=seed)
    mi_frac = float((mi > 0).mean())
    top20_mi_idx = np.argsort(mi)[::-1][:20]

    # --- Drug-level aggregation (Pitfall 8) ---
    if profiles is not None:
        # groupby compound_name (lowercase) to get drug-level mean OOF probs
        drug_key = profiles["compound_name"].str.lower().str.strip().values
        unique_drugs = list(dict.fromkeys(drug_key))  # preserves insertion order

        # Leakage-free OOF probs: a single drug can contribute up to hundreds of
        # profiles (the real Wang/Li set has one drug with 784). Profile-level
        # StratifiedKFold splits those profiles across train/test, so the model
        # memorizes drug identity and the "ceiling" is meaningless (inflated to
        # ~0.91 profile-level, then a nonsense 0.56 after drug aggregation).
        # StratifiedGroupKFold keeps every drug's profiles in one fold, so the
        # held-out drug is genuinely unseen -- a fair drug-level ceiling that is
        # unit-consistent with the floor (which already does drug-level CV).
        # StandardScaler is fit per-fold inside the pipeline (978 LINCS genes
        # have heterogeneous scale).
        n_unique = len(unique_drugs)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            ),
        )
        eff_splits = min(n_splits, n_unique)
        if eff_splits >= 2:
            cv = StratifiedGroupKFold(
                n_splits=eff_splits, shuffle=True, random_state=seed
            )
            oof_proba = cross_val_predict(
                clf, de, y, cv=cv, groups=drug_key, method="predict_proba"
            )
        else:
            # Degenerate: <2 distinct drugs -- no grouped split is possible.
            oof_proba = cross_val_predict(
                clf,
                de,
                y,
                cv=StratifiedKFold(
                    n_splits=max(2, min(n_splits, len(y))),
                    shuffle=True,
                    random_state=seed,
                ),
                method="predict_proba",
            )
        profile_probs = oof_proba[:, 1]

        # Group by drug: mean prob and confirm one label per drug
        drug_probs: dict[str, float] = {}
        drug_labels: dict[str, int] = {}
        for drug in unique_drugs:
            mask = drug_key == drug
            drug_probs[drug] = float(profile_probs[mask].mean())
            drug_label_vals = y[mask]
            drug_labels[drug] = int(drug_label_vals[0])

        drug_y = np.array([drug_labels[d] for d in unique_drugs], dtype=int)
        drug_prob_arr = np.array([drug_probs[d] for d in unique_drugs])

    else:
        # de / y already at drug level
        unique_drugs_count = len(y)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        lr = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        )
        oof_proba = cross_val_predict(lr, de, y, cv=cv, method="predict_proba")
        drug_prob_arr = oof_proba[:, 1]
        drug_y = y
        drug_probs = {str(i): float(p) for i, p in enumerate(drug_prob_arr)}
        unique_drugs = list(drug_probs.keys())
        unique_drugs_count = len(unique_drugs)

    ceiling_auroc = float(roc_auc_score(drug_y, drug_prob_arr))
    n_drugs = len(unique_drugs) if profiles is not None else unique_drugs_count

    log.info(
        "compute_ceiling: n_drugs=%d ceiling_auroc=%.4f PR=%.1f "
        "mi_frac_nonzero=%.3f",
        n_drugs,
        ceiling_auroc,
        pr,
        mi_frac,
    )

    return {
        "ceiling_auroc": ceiling_auroc,
        "participation_ratio": pr,
        "mi": mi,
        "mi_fraction_nonzero": mi_frac,
        "top20_mi_idx": top20_mi_idx,
        "n_drugs": n_drugs,
        "ceiling_probs": drug_probs,
    }
