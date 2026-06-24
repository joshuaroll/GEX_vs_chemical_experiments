"""Unit tests for `src/spatial/eda/floor.py`.

In-memory fixtures only -- no real data files are read. Synthetic fingerprint
matrices with known separability are used.

Behaviors covered:
  1. Floor AUROC > 0.5 on a linearly separable synthetic fingerprint matrix.
"""

from __future__ import annotations

import numpy as np

from src.spatial.eda.floor import (
    compute_floor,
    floor_profile_disjoint_auroc,
    floor_profile_disjoint_oof,
    floor_profile_leaky_auroc,
)
from sklearn.metrics import roc_auc_score


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


def test_floor_profile_disjoint_no_drug_leakage() -> None:
    """Drug-grouped CV must stop per-drug memorization from inflating the floor.

    Mirrors test_ceiling_no_drug_leakage. Each drug gets a unique constant
    identity feature dim (repeated across its profiles) and a label. Under
    profile-level CV the model memorizes the drug (AUROC -> ~1.0). Under
    leakage-free drug-grouped CV the held-out drug's identity dim is never seen
    with a label, so the profile-level AUROC stays near chance. This locks the
    GroupKFold grouping for the new floor function (mirroring the ceiling fix).
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

    auroc = floor_profile_disjoint_auroc(de, y, np.asarray(names))

    assert auroc < 0.75, (
        f"floor profile-disjoint AUROC {auroc:.3f} too high -- drug leakage not "
        "prevented (CV is splitting a drug's profiles across train/test)"
    )


def _make_drug_identity_fixture() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-drug constant-identity fixture (mirror of the disjoint leakage test).

    n_drugs=10, profiles_per_drug=8. Each drug d gets a unique identity feature
    dim (sig[d]=5.0 repeated across its 8 profiles) plus tiny noise. Label = 1 if
    d < 5 else 0. groups = [f"drug{d}"]*8. The identity dim is memorizable ONLY
    via leakage (a drug's profiles straddling train/test).
    """
    n_drugs = 10
    profiles_per_drug = 8
    rng = np.random.default_rng(0)
    de_rows, labels, names = [], [], []
    for d in range(n_drugs):
        sig = np.zeros(n_drugs, dtype=np.float32)
        sig[d] = 5.0
        lab = 1 if d < n_drugs // 2 else 0
        for _ in range(profiles_per_drug):
            de_rows.append(sig + rng.standard_normal(n_drugs).astype(np.float32) * 0.01)
            labels.append(lab)
            names.append(f"drug{d}")
    fps = np.asarray(de_rows, dtype=np.float32)
    y = np.asarray(labels, dtype=int)
    return fps, y, names


def test_floor_profile_leaky_exploits_drug_identity() -> None:
    """Mirror-OPPOSITE of the disjoint test: the leaky split MUST exploit identity.

    floor_profile_leaky_auroc uses a random StratifiedKFold (NO groups), so a
    drug's identity dim IS seen with a label in training and the held-out
    profiles of the same drug are trivially classified. The leaky AUROC must
    therefore beat the drug-disjoint AUROC (which hides the identity), locking
    that floor_profile_leaky_auroc does NOT group by drug.
    """
    fps, y, names = _make_drug_identity_fixture()

    leaky = floor_profile_leaky_auroc(fps, y, seed=42)
    disjoint = floor_profile_disjoint_auroc(fps, y, np.asarray(names), seed=42)

    assert leaky > disjoint, (
        f"leaky AUROC {leaky:.3f} should exceed disjoint AUROC {disjoint:.3f} -- "
        "the leaky split is supposed to exploit per-drug identity"
    )
    assert leaky > 0.75, (
        f"leaky AUROC {leaky:.3f} too low -- the random split should memorize the "
        "per-drug identity feature"
    )


def test_floor_profile_disjoint_oof_matches_scalar() -> None:
    """The OOF vector's AUROC must equal the disjoint scalar exactly (same folds)."""
    fps, y, names = _make_drug_identity_fixture()
    groups = np.asarray(names)

    oof = floor_profile_disjoint_oof(fps, y, groups, seed=42)
    scalar = floor_profile_disjoint_auroc(fps, y, groups, seed=42)

    assert oof.ndim == 1 and len(oof) == len(y), (
        f"OOF vector must be 1-D of length {len(y)}, got shape {oof.shape}"
    )
    assert oof.dtype == np.float64, f"OOF vector must be float64, got {oof.dtype}"
    assert abs(roc_auc_score(y, oof) - scalar) < 1e-9, (
        "roc_auc_score(y, floor_profile_disjoint_oof(...)) must equal "
        "floor_profile_disjoint_auroc(...) within 1e-9 (same folds/pipeline)"
    )
