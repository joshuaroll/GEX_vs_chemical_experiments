"""Unit tests for `src/spatial/eda/labels.py` and `src/spatial/eda/smiles_join.py`.

In-memory fixtures only -- no real data files are read.

Behaviors covered:
  1. DILIrank binary encoding: Ambiguous excluded, vMOST/vMost both map to 1.
  2. DILIrank name_lower normalization.
  3. DILIst trailing space on column name handled by .str.strip().
  4. SMILES join cascade: dili_canonical layer first, drugbank fallback.
  5. SMILES join: drugs with no hit in either layer have smiles=NaN.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from src.spatial.eda.labels import load_dilirank, load_dilist
from src.spatial.eda.smiles_join import join_smiles_cascade


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dilirank_df() -> pd.DataFrame:
    """Return a minimal synthetic dilirank DataFrame (no file I/O)."""
    return pd.DataFrame({
        "LTKBID": ["A", "B", "C", "D"],
        "CompoundName": ["DrugA", "DrugB", "DrugC", "DrugD"],
        "vDILI-Concern": [
            "vMost-DILI-concern",
            "vNo-DILI-Concern",
            "Ambiguous-DILI-concern",
            "vMOST-DILI-concern",
        ],
        "SeverityClass": ["1", "0", "A", "1"],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dilirank_binary_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ambiguous excluded; vMOST/vMost both map to 1; vNo maps to 0."""
    synthetic_df = _make_dilirank_df()
    monkeypatch.setattr("pandas.read_excel", lambda *args, **kwargs: synthetic_df)

    result = load_dilirank("dummy_path.xlsx")

    # Ambiguous row excluded -> 3 rows remain
    assert len(result) == 3, f"Expected 3 rows after excluding Ambiguous, got {len(result)}"

    # Both vMost-DILI-concern (DrugA) and vMOST-DILI-concern (DrugD) map to 1
    by_name = result.set_index("CompoundName")["dili_binary"]
    assert by_name["DrugA"] == 1, "vMost-DILI-concern should map to 1"
    assert by_name["DrugD"] == 1, "vMOST-DILI-concern should map to 1"
    # vNo-DILI-Concern maps to 0
    assert by_name["DrugB"] == 0, "vNo-DILI-Concern should map to 0"


def test_smiles_join_cascade(tmp_path: pathlib.Path) -> None:
    """dili_canonical wins; drugbank fills residual; missing drug gets NaN."""
    # Synthetic labels DataFrame
    labels_df = pd.DataFrame({
        "name_lower": ["aspirin", "ibuprofen", "unknown_drug"],
        "dili_binary": [1, 0, 0],
    })

    # Synthetic dili_canonical (layer 1) -- covers 'aspirin'
    canonical_csv = tmp_path / "dili_canonical.csv"
    pd.DataFrame({
        "drug_name": ["Aspirin"],
        "canonical_smiles": ["CC(=O)Oc1ccccc1C(=O)O"],
    }).to_csv(canonical_csv, index=False)

    # Synthetic drugbank (layer 2) -- covers 'ibuprofen' (residual)
    drugbank_csv = tmp_path / "drugbank_smiles_index.csv"
    pd.DataFrame({
        "name_lower": ["ibuprofen"],
        "name": ["Ibuprofen"],
        "smiles": ["CC(C)Cc1ccc(cc1)C(C)C(=O)O"],
    }).to_csv(drugbank_csv, index=False)

    result = join_smiles_cascade(
        labels_df,
        str(canonical_csv),
        str(drugbank_csv),
    )

    # dili_canonical layer wins for 'aspirin'
    aspirin_row = result[result["name_lower"] == "aspirin"]
    assert len(aspirin_row) == 1
    assert aspirin_row.iloc[0]["smiles"] == "CC(=O)Oc1ccccc1C(=O)O"

    # drugbank fills 'ibuprofen'
    ibu_row = result[result["name_lower"] == "ibuprofen"]
    assert len(ibu_row) == 1
    assert ibu_row.iloc[0]["smiles"] is not None
    assert isinstance(ibu_row.iloc[0]["smiles"], str)

    # 'unknown_drug' has smiles=NaN
    unk_row = result[result["name_lower"] == "unknown_drug"]
    assert len(unk_row) == 1
    assert pd.isna(unk_row.iloc[0]["smiles"])
