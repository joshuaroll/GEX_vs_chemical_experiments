"""
CI regression tests for Stage-2 feature cache (Phase 3).

These tests verify that `data/processed/dili_features.parquet` was written
correctly by `src/stage2/cache_dili_features.py`.

Run with:
    conda run -n dili_v04_env pytest tests/test_stage2_features.py -v

Tests (EMBED-04 hard gate + structural sanity):
  1. File exists
  2. Row count == 1118
  3. Feature column count == 1688 (1 dose + 919 gex + 768 embed)
  4. Zero NaN in all feature columns (EMBED-04 hard gate)
  5. Class balance: {1: 685, 0: 433}
  6. feat_dose: finite, non-constant (mean ~55, std ~20)
  7. feat_gex: finite, non-constant (positive variance)
  8. feat_embed: finite, non-constant (MolFormer 768-dim, positive variance)
  9. MolFormer SHA matches pinned checkpoint
"""
import pytest
import os
from pathlib import Path

# ---- Locate parquet relative to repo root ----
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PARQUET = _REPO_ROOT / 'data' / 'processed' / 'dili_features.parquet'
_EMBED_SHA = '7b12d946c181a37f6012b9dc3b002275de070314'


@pytest.fixture(scope='module')
def df():
    """Load dili_features.parquet once for all tests in this module."""
    import pandas as pd
    if not _PARQUET.exists():
        pytest.skip(f'dili_features.parquet not found at {_PARQUET}')
    return pd.read_parquet(str(_PARQUET))


@pytest.fixture(scope='module')
def feat_cols(df):
    return [c for c in df.columns if c.startswith('feat_')]


@pytest.fixture(scope='module')
def gex_cols(feat_cols):
    return [c for c in feat_cols if 'gex' in c]


@pytest.fixture(scope='module')
def embed_cols(feat_cols):
    return [c for c in feat_cols if 'embed' in c]


# ---- Test 1: file exists ----
def test_parquet_exists():
    assert _PARQUET.exists(), (
        f'Stage-2 feature cache not found: {_PARQUET}\n'
        'Run: python src/stage2/cache_dili_features.py ...')


# ---- Test 2: row count ----
def test_row_count(df):
    assert len(df) == 1118, f'Expected 1118 drugs, got {len(df)}'


# ---- Test 3: feature column count ----
def test_feature_col_count(feat_cols, gex_cols, embed_cols):
    assert len(feat_cols) == 1688, (
        f'Expected 1688 feature cols (1+919+768), got {len(feat_cols)}')
    assert len(gex_cols) == 919, f'Expected 919 GEX cols, got {len(gex_cols)}'
    assert len(embed_cols) == 768, f'Expected 768 embed cols, got {len(embed_cols)}'
    assert 'feat_dose' in feat_cols


# ---- Test 4: EMBED-04 hard gate — no NaN in feature columns ----
def test_no_nan_in_feature_cols(df, feat_cols):
    """EMBED-04: zero NaN in feature columns (metadata cols like dili_severity can be NaN)."""
    nan_count = int(df[feat_cols].isnull().sum().sum())
    assert nan_count == 0, (
        f'EMBED-04 FAIL: {nan_count} NaN values found in feature columns.\n'
        'Expected 0 NaN — check cache_dili_features.py output.')


# ---- Test 5: class balance ----
def test_class_balance(df):
    counts = df['dili_binary'].value_counts().to_dict()
    assert counts.get(1) == 685, f'Expected 685 DILI=1, got {counts.get(1)}'
    assert counts.get(0) == 433, f'Expected 433 DILI=0, got {counts.get(0)}'


# ---- Test 6: feat_dose sanity ----
def test_feat_dose_sanity(df):
    dose = df['feat_dose']
    assert dose.notna().all(), 'feat_dose contains NaN'
    assert dose.std() > 1.0, (
        f'feat_dose is nearly constant (std={dose.std():.4f}); model may not be running')
    assert 20.0 < dose.mean() < 120.0, (
        f'feat_dose mean {dose.mean():.2f} out of expected range [20, 120]')


# ---- Test 7: feat_gex sanity ----
def test_feat_gex_sanity(df, gex_cols):
    gex = df[gex_cols]
    col_stds = gex.std()
    n_zero_std = int((col_stds == 0).sum())
    assert n_zero_std < 50, (
        f'{n_zero_std}/919 GEX cols have zero variance — model may not be running')
    gex_mean_abs = float(gex.values.__abs__().mean())
    assert gex_mean_abs > 0.01, (
        f'feat_gex near-zero mean ({gex_mean_abs:.4f}) — DE values are all zero?')


# ---- Test 8: feat_embed sanity (MolFormer) ----
def test_feat_embed_sanity(df, embed_cols):
    emb = df[embed_cols]
    assert emb.notna().all().all(), 'feat_embed contains NaN'
    col_stds = emb.std()
    n_zero_std = int((col_stds < 1e-8).sum())
    assert n_zero_std < 100, (
        f'{n_zero_std}/768 embed dims have near-zero variance — MolFormer issue?')


# ---- Test 9: MolFormer HF SHA pinned in wrapper ----
def test_molformer_sha_pinned():
    """MolFormer SHA must match the pinned commit in molformer_wrapper.py."""
    wrapper_path = _REPO_ROOT / 'src' / 'embed' / 'molformer_wrapper.py'
    if not wrapper_path.exists():
        pytest.skip(f'molformer_wrapper.py not found at {wrapper_path}')
    src = wrapper_path.read_text()
    assert _EMBED_SHA in src, (
        f'MolFormer SHA {_EMBED_SHA} not found in molformer_wrapper.py')
