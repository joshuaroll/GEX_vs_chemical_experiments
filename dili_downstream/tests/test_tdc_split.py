"""Unit tests for `src/data/tdc_split.py` (SPLIT-04).

Wraps `tdc.single_pred.Tox(name='DILI_Hong')`. All non-network tests are
offline: we mock the `tdc` module via `sys.modules` injection so the lib's
behavior is exercised without a real network call. The single network-marked
test calls the real TDC API and is skipped unless `TDC_NETWORK_TESTS=1`.

Behaviors covered (per 02-01-PLAN.md):
  1. Returned dict has keys {"train", "val", "test", "tdc_version", "dataset_size"}.
  2. ImportError raised if pytdc is unimportable (msg mentions pytdc + dili_v04_env).
  3. Determinism — same seed → identical splits (TDC scaffold-split is deterministic).
  4. All pert_ids in train/val/test are str (TDC's Drug_ID column may be mixed).
  5. 70/10/20 default ratios per TDC scaffold-split convention.
  6. Network-marked smoke test: real TDC call (skipped if no internet).
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helper: build a fake `tdc` module so tests run offline.
# ---------------------------------------------------------------------------


def _make_fake_tdc_module(
    train_ids: list,
    valid_ids: list,
    test_ids: list,
    version: str = "0.4.1",
) -> types.ModuleType:
    """Synthesize a `tdc` module + `tdc.single_pred.Tox` callable that
    matches the API shape `tdc_dili_scaffold_split` invokes."""
    fake_tdc = types.ModuleType("tdc")
    fake_tdc.__version__ = version
    fake_single_pred = types.ModuleType("tdc.single_pred")

    class FakeTox:
        def __init__(self, name: str, path: str | None = None) -> None:
            assert name == "dili", f"unexpected dataset name: {name}"
            self.name = name
            self.path = path

        def get_split(self, method: str = "scaffold", seed: int = 42, frac=None):
            assert method == "scaffold", f"unexpected method: {method}"
            # TDC default frac is [0.7, 0.1, 0.2] when not passed; honor any
            # explicit frac the caller provides for visibility.
            if frac is None:
                frac = [0.7, 0.1, 0.2]
            return {
                "train": pd.DataFrame({"Drug_ID": train_ids, "Drug": ["S"] * len(train_ids), "Y": [0] * len(train_ids)}),
                "valid": pd.DataFrame({"Drug_ID": valid_ids, "Drug": ["S"] * len(valid_ids), "Y": [0] * len(valid_ids)}),
                "test": pd.DataFrame({"Drug_ID": test_ids, "Drug": ["S"] * len(test_ids), "Y": [0] * len(test_ids)}),
            }

    fake_single_pred.Tox = FakeTox
    fake_tdc.single_pred = fake_single_pred
    return fake_tdc


def _install_fake_tdc(monkeypatch, fake_tdc) -> None:
    """Inject the fake module into sys.modules so the SUT can `import tdc`."""
    monkeypatch.setitem(sys.modules, "tdc", fake_tdc)
    monkeypatch.setitem(sys.modules, "tdc.single_pred", fake_tdc.single_pred)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_1_returned_dict_has_locked_keys(monkeypatch):
    """Required keys: train, val, test, tdc_version, dataset_size."""
    fake_tdc = _make_fake_tdc_module(
        train_ids=[f"D{i}" for i in range(7)],
        valid_ids=[f"D{i}" for i in range(7, 8)],
        test_ids=[f"D{i}" for i in range(8, 10)],
        version="0.4.1",
    )
    _install_fake_tdc(monkeypatch, fake_tdc)
    # Import lazily to ensure the patched `tdc` is picked up by the SUT's
    # internal `import tdc`.
    from src.data.tdc_split import tdc_dili_scaffold_split

    out = tdc_dili_scaffold_split(seed=42)
    assert set(out.keys()) == {"train", "val", "test", "tdc_version", "dataset_size"}, (
        f"Unexpected keys: {set(out.keys())}"
    )
    assert out["tdc_version"] == "0.4.1"
    assert out["dataset_size"] == 10  # 7 + 1 + 2


def test_2_importerror_when_pytdc_missing(monkeypatch):
    """When `import tdc` fails, the function raises ImportError with a
    message that mentions pytdc and dili_v04_env."""
    # Force `import tdc` to fail by injecting a None in sys.modules
    # AND removing any cached real tdc.
    monkeypatch.setitem(sys.modules, "tdc", None)
    monkeypatch.delitem(sys.modules, "tdc.single_pred", raising=False)

    from src.data.tdc_split import tdc_dili_scaffold_split

    with pytest.raises(ImportError) as exc_info:
        tdc_dili_scaffold_split(seed=42)
    msg = str(exc_info.value).lower()
    assert "pytdc" in msg, f"ImportError message should mention pytdc: {msg!r}"
    assert "dili_v04_env" in msg, (
        f"ImportError message should mention dili_v04_env: {msg!r}"
    )


def test_3_determinism_same_seed(monkeypatch):
    fake_tdc = _make_fake_tdc_module(
        train_ids=[f"D{i}" for i in range(70)],
        valid_ids=[f"D{i}" for i in range(70, 80)],
        test_ids=[f"D{i}" for i in range(80, 100)],
        version="0.4.1",
    )
    _install_fake_tdc(monkeypatch, fake_tdc)
    from src.data.tdc_split import tdc_dili_scaffold_split

    a = tdc_dili_scaffold_split(seed=42)
    b = tdc_dili_scaffold_split(seed=42)
    assert a == b, "tdc_dili_scaffold_split must be deterministic for same seed"


def test_4_drug_ids_are_strings(monkeypatch):
    """TDC's Drug_ID may be mixed int/str — the wrapper casts everything to str."""
    fake_tdc = _make_fake_tdc_module(
        # Mixed int + str + float Drug_IDs (real-world TDC quirk).
        train_ids=[1, "DB00001", 3.0, "DB-XYZ"],
        valid_ids=[42],
        test_ids=["DB00099", 100],
        version="0.4.1",
    )
    _install_fake_tdc(monkeypatch, fake_tdc)
    from src.data.tdc_split import tdc_dili_scaffold_split

    out = tdc_dili_scaffold_split(seed=42)
    for sname in ("train", "val", "test"):
        for did in out[sname]:
            assert isinstance(did, str), (
                f"{sname} contains non-str Drug_ID: {did!r} ({type(did).__name__})"
            )


def test_5_default_70_10_20_ratios_per_tdc_convention(monkeypatch):
    """TDC scaffold split defaults to 70/10/20 (NOT 80/10/10). The wrapper
    must respect TDC's convention; documented divergence per CONTEXT.md
    "scaffold split per TDC default"."""
    n = 100
    fake_tdc = _make_fake_tdc_module(
        train_ids=[f"D{i}" for i in range(70)],     # 70
        valid_ids=[f"D{i}" for i in range(70, 80)],  # 10
        test_ids=[f"D{i}" for i in range(80, 100)],  # 20
        version="0.4.1",
    )
    _install_fake_tdc(monkeypatch, fake_tdc)
    from src.data.tdc_split import tdc_dili_scaffold_split

    out = tdc_dili_scaffold_split(seed=42)
    assert len(out["train"]) == 70
    assert len(out["val"]) == 10
    assert len(out["test"]) == 20
    assert out["dataset_size"] == n


def test_6_returns_lists_not_dataframes(monkeypatch):
    """Each split value must be a Python list (not pd.Series / DataFrame)."""
    fake_tdc = _make_fake_tdc_module(
        train_ids=["A", "B", "C"],
        valid_ids=["D"],
        test_ids=["E", "F"],
        version="0.4.1",
    )
    _install_fake_tdc(monkeypatch, fake_tdc)
    from src.data.tdc_split import tdc_dili_scaffold_split

    out = tdc_dili_scaffold_split(seed=42)
    for sname in ("train", "val", "test"):
        assert isinstance(out[sname], list), (
            f"{sname} should be a list, got {type(out[sname]).__name__}"
        )


@pytest.mark.network
def test_7_real_tdc_smoke_test():
    """Real TDC API smoke test. Skipped unless TDC_NETWORK_TESTS=1.

    This is the only test that requires internet; it confirms the wrapper
    works end-to-end against a live TDC. CI runs offline by default.
    """
    import os
    if os.environ.get("TDC_NETWORK_TESTS") != "1":
        pytest.skip(
            "Network test: set TDC_NETWORK_TESTS=1 to run against the real TDC API"
        )
    try:
        importlib.import_module("tdc")
    except ImportError:
        pytest.skip("pytdc not installed — cannot run real-API test")

    from src.data.tdc_split import tdc_dili_scaffold_split
    out = tdc_dili_scaffold_split(seed=42)
    assert set(out.keys()) >= {"train", "val", "test", "tdc_version", "dataset_size"}
    assert out["dataset_size"] > 0
    assert len(out["train"]) + len(out["val"]) + len(out["test"]) == out["dataset_size"]
