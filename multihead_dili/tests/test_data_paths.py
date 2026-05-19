"""Sanity tests for multihead_dili data dependencies. Verify external paths exist
and have expected shape/columns. NEVER tests model outputs (project rule)."""
from pathlib import Path

import pandas as pd
import pytest

EHILL_DIR = Path("/raid/home/joshua/data/MultiDCP/data/ehill_data")
EHILL_TRAIN = EHILL_DIR / "high_confident_data_train.csv"
EHILL_DEV = EHILL_DIR / "high_confident_data_dev.csv"
EHILL_TEST = EHILL_DIR / "high_confident_data_test.csv"

DILI_CANONICAL = Path(
    "/raid/home/joshua/projects/GEX_vs_chemical_experiments/"
    "dili_downstream/data/processed/dili_canonical.csv"
)


def test_ehill_train_exists():
    assert EHILL_TRAIN.exists(), f"E-Hill train CSV missing: {EHILL_TRAIN}"


def test_ehill_train_schema():
    df = pd.read_csv(EHILL_TRAIN, nrows=5)
    required = {"sig_id", "pert_id", "pert_type", "cell_id", "pert_idose", "ehill"}
    assert required.issubset(df.columns), \
        f"E-Hill train missing columns: {required - set(df.columns)}"


def test_ehill_train_nonempty():
    # cheap row count via wc-style read
    df = pd.read_csv(EHILL_TRAIN)
    assert len(df) > 50_000, f"E-Hill train suspiciously small: {len(df)} rows"


def test_dili_canonical_reachable():
    assert DILI_CANONICAL.exists(), f"v0.5 DILI canonical not found: {DILI_CANONICAL}"


def test_dili_canonical_schema():
    df = pd.read_csv(DILI_CANONICAL, nrows=5)
    required = {"pert_id", "drug_name", "smiles", "canonical_smiles", "scaffold", "dili_binary"}
    assert required.issubset(df.columns), \
        f"DILI canonical missing columns: {required - set(df.columns)}"
