"""Unit tests for `src/spatial/eda/fingerprints.py`.

In-memory fixtures only -- no real data files are read. SMILES strings are
provided inline.

Behaviors covered:
  1. ECFP4 fingerprint shape: (n_smiles, 2048) uint8.
  2. valid_mask True for parseable SMILES, False for invalid.
  3. Known-valid SMILES (aspirin) produces nonzero fingerprint.
"""

from __future__ import annotations

import numpy as np

from src.spatial.eda.fingerprints import smiles_to_ecfp4


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ecfp4_shape() -> None:
    """smiles_to_ecfp4 returns fps shape (2, 2048) uint8 and correct valid_mask."""
    smiles_list = [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin: valid
        "not_a_smiles",             # invalid SMILES
    ]
    fps, valid_mask = smiles_to_ecfp4(smiles_list)

    assert fps.shape == (2, 2048), f"Expected shape (2, 2048), got {fps.shape}"
    assert fps.dtype == np.uint8, f"Expected uint8, got {fps.dtype}"
    assert valid_mask[0] is True or bool(valid_mask[0]), "Aspirin SMILES should be valid"
    assert not (valid_mask[1] is True or bool(valid_mask[1])), "Invalid SMILES should have valid_mask=False"


def test_ecfp4_valid_smiles_nonzero() -> None:
    """Known-valid SMILES produces at least one nonzero bit."""
    smiles_list = ["CC(=O)Oc1ccccc1C(=O)O"]  # aspirin
    fps, valid_mask = smiles_to_ecfp4(smiles_list)

    assert bool(valid_mask[0]), "Aspirin should parse successfully"
    assert fps[0].sum() > 0, "Aspirin fingerprint should have nonzero bits"


def test_ecfp4_invalid_smiles_zero_row() -> None:
    """Invalid SMILES produces a zero row (placeholder) and valid_mask=False."""
    smiles_list = ["not_a_smiles"]
    fps, valid_mask = smiles_to_ecfp4(smiles_list)

    assert not bool(valid_mask[0]), "Invalid SMILES should have valid_mask=False"
    assert fps[0].sum() == 0, "Invalid SMILES row should be all zeros"
