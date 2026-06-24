"""Real-inference smoke for the frozen MultiDCP-CheMoE checkpoint (WIRE-01).

PURITY-GATE EXEMPTION
---------------------
This is the ONE Phase-2 test file that reads ``/raid`` paths and the real
gene-order file. It is an integration smoke, explicitly exempt from the
pure-test purity gate (no-/raid, no-real-data rule). Per Hard Rule 1 the frozen
model inference path CANNOT be mocked: a stubbed model output would silently
pass a fake contract. The whole module is therefore marked ``@pytest.mark.gpu``
via ``pytestmark`` so the pure run (``-m "not gpu"``) skips this file entirely;
it runs once per wave merge on a GPU box.

Behaviors covered (WIRE-01, real ckpt — D-04 row-17):
  1. Strict 0/0 state_dict load into ``MultiDCP_CheMoE_AE`` (row-17 checkpoint).
  2. One real forward -> finite ``[10716]`` vector in ~[0, 1] (normalization
     sanity, Pitfall 1).

RED until 02-02: ``RegionSignatureCacher.load_model`` raises NotImplementedError
today, so both tests fail when actually run under ``-m gpu``. Until then they
are collectible but excluded from the pure suite.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

# Whole-module gpu mark: pure run (`-m "not gpu"`) skips this file.
pytestmark = pytest.mark.gpu

# NOTE: the heavy / not-yet-existing imports (N_PDG, RegionSignatureCacher) are
# done INSIDE the fixtures/tests, not at module scope, so this gpu-marked file
# collects cleanly under `-m "not gpu"` (and is deselected) rather than erroring
# at import. They resolve once 02-02 adds N_PDG and fills load_model.

# Row-17 checkpoint (MANIFEST, D-04 amended). DO NOT use the stale
# dili_downstream/...best.pt path nor the collapsed row-18 kpgt checkpoint.
CHECKPOINT_PATH = "/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt"

# Real 10,716-symbol gene-order file (P0 output). Read only here (exempt file).
SYMBOLS_PATH = pathlib.Path("data/processed/spatial/multidcp_10716_symbols.txt")


def _load_gene_ids() -> tuple[str, ...]:
    """Read the real 10,716-symbol gene order (exempt: integration smoke)."""
    symbols = SYMBOLS_PATH.read_text().split()
    return tuple(symbols)


@pytest.fixture
def gene_ids() -> tuple[str, ...]:
    from src.spatial.region_signature import N_PDG  # RED until 02-02

    ids = _load_gene_ids()
    assert len(ids) == N_PDG, (
        f"gene-order file has {len(ids)} symbols, expected N_PDG={N_PDG}."
    )
    return ids


def test_strict_load_chemoe_ae(gene_ids):
    """load_model() strict-loads (0/0) the row-17 checkpoint, no raise."""
    from src.spatial.region_signature import RegionSignatureCacher

    cacher = RegionSignatureCacher(
        model_variant="multidcp_chemoe",
        gene_ids=gene_ids,
    )
    cacher.load_model(CHECKPOINT_PATH)
    assert cacher._model is not None, (
        "load_model must populate cacher._model with the frozen network."
    )


def test_forward_sane(gene_ids):
    """One real forward returns a finite [10716] vector in ~[0, 1] (Pitfall 1)."""
    from src.spatial.region_signature import N_PDG, RegionSignatureCacher

    cacher = RegionSignatureCacher(
        model_variant="multidcp_chemoe",
        gene_ids=gene_ids,
    )
    cacher.load_model(CHECKPOINT_PATH)

    # Region basal must be min-max normalized to the manifold [0, 1] before the
    # call (the silent-corruption bug guard). Use a benign normalized vector.
    region_basal = np.linspace(0.0, 1.0, N_PDG, dtype=np.float32)
    pred = cacher._call_model(smiles="CCO", region_basal=region_basal)

    pred = np.asarray(pred)
    assert pred.shape == (N_PDG,), (
        f"_call_model must return a [{N_PDG}] vector, got shape {pred.shape}."
    )
    assert np.all(np.isfinite(pred)), "Forward produced non-finite values."
    # Broad manifold-range sanity (not a hard clip): values near [0, 1].
    assert pred.min() > -0.5 and pred.max() < 1.5, (
        f"Forward values out of manifold range: min={pred.min()}, "
        f"max={pred.max()} (expected ~[0, 1], Pitfall 1 normalization sanity)."
    )
