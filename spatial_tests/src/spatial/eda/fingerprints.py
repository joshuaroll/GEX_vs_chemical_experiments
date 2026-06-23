"""Pure library: compute ECFP4/Morgan fingerprints via RDKit with input validation.

Phase deliverable: SMILES-to-fingerprint utility consumed by the floor
pipeline (plan 03) for structure-only classification (D-03).

Policy (locked):
    - 2048-bit Morgan fingerprints at radius 2 (ECFP4 convention).
    - Input validation (V5 security domain, T-01-03 threat):
      Chem.MolFromSmiles returning None -> zero-vector + valid_mask=False.
      Never raise on a bad SMILES string.
    - Returns (fps, valid_mask) tuple: fps shape (n, 2048) uint8;
      valid_mask shape (n,) bool.

Hard rules honored:
    - Pure library: NO hardcoded absolute paths, NO real-data filenames.
    - No mock or synthetic labels — caller provides SMILES strings.
    - No import-time side effects; rdkit imported at module load but
      Chem.MolFromSmiles is only called inside smiles_to_ecfp4().
"""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

log = logging.getLogger(__name__)

__all__ = ["smiles_to_ecfp4"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def smiles_to_ecfp4(
    smiles_list: list[str],
    n_bits: int = 2048,
    radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert SMILES strings to ECFP4 Morgan fingerprints.

    For each SMILES, attempts RDKit parsing. If parsing fails (MolFromSmiles
    returns None), emits a zero vector and sets valid_mask=False (T-01-03
    mitigation; no exception raised).

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to featurize. May include invalid or empty strings.
    n_bits : int
        Fingerprint bit-vector length. Default 2048.
    radius : int
        Morgan radius. Default 2 (ECFP4 convention).

    Returns
    -------
    fps : np.ndarray
        Shape (len(smiles_list), n_bits), dtype uint8. Rows with invalid SMILES
        are zero-filled.
    valid_mask : np.ndarray
        Shape (len(smiles_list),), dtype bool. True where RDKit parsed
        successfully; False for unparseable SMILES.

    Notes
    -----
    GetMorganFingerprintAsBitVect returns a DataStructs.ExplicitBitVect.
    ToBitString() produces a '0'/'1' ASCII string; we decode via
    np.frombuffer(..., dtype='u1') - ord('0') to get uint8 bits.
    """
    fps: list[np.ndarray] = []
    valid_mask: list[bool] = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # V5 input validation: zero-vector placeholder, no crash
            fps.append(np.zeros(n_bits, dtype=np.uint8))
            valid_mask.append(False)
            log.debug(
                "smiles_to_ecfp4: unparseable SMILES %r -> zero-vector (valid_mask=False).",
                smi,
            )
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            bit_arr = np.frombuffer(fp.ToBitString().encode(), dtype="u1") - ord("0")
            fps.append(bit_arr.astype(np.uint8))
            valid_mask.append(True)

    fps_array = np.array(fps, dtype=np.uint8)
    mask_array = np.array(valid_mask, dtype=bool)

    n_valid = mask_array.sum()
    log.debug(
        "smiles_to_ecfp4: %d / %d SMILES parsed successfully "
        "(n_bits=%d, radius=%d).",
        n_valid,
        len(smiles_list),
        n_bits,
        radius,
    )

    return fps_array, mask_array
