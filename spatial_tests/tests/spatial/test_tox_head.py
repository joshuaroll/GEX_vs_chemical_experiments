"""Unit tests for ``src/spatial/tox_head.py`` (WIRE-02).

In-memory tensor fixtures — no real-data dependency, no file I/O, no GPU.
Analog: ``tests/spatial/test_region_combiner.py`` (nn.Module forward-shape
tests + NamedTuple-output assertions).

RED until 02-03: ``src.spatial.tox_head`` does not exist yet. The import below
raises ``ImportError`` so every test in this file errors (RED). 02-03 writes
``tox_head.py`` (ToxHead + ToxHeadOutput) and flips these GREEN by editing
source only.

Behaviors covered (WIRE-02):
  1. Zero GEX channel still yields a finite logit (condition A masking).
  2. Combiner feed: [B, n_regions, 10716] -> AttentionPoolCombiner.pooled
     [B, 10716] -> ToxHead -> finite logit.
  3. Dose-response channel optional (d_dr=0) this phase.
  4. ToxHeadOutput is a NamedTuple (logit, attn_weights).

Purity gate
-----------
- No ``/raid`` paths anywhere in this file.
- CPU-only, tiny batch sizes; no @pytest.mark.gpu (the head needs no GPU/ckpt).
"""

from __future__ import annotations

import pytest
import torch

from src.spatial.region_combiner import AttentionPoolCombiner

# RED until 02-03: module does not exist yet -> ImportError at collection.
from src.spatial.tox_head import ToxHead, ToxHeadOutput


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_GEX = 10716
D_CHEM = 2048
BATCH = 4
N_REGIONS = 3


# ---------------------------------------------------------------------------
# Test 1: zero GEX channel -> finite logit (condition A masking, WIRE-02)
# ---------------------------------------------------------------------------


def test_zero_gex_channel_finite_logit():
    """A zeroed GEX channel must still produce a finite logit (condition A)."""
    torch.manual_seed(0)
    head = ToxHead(d_gex=D_GEX, d_chem=D_CHEM, d_dr=0)
    gex_pooled = torch.zeros(BATCH, D_GEX)
    chem_emb = torch.randn(BATCH, D_CHEM)

    out = head(gex_pooled, chem_emb)

    assert isinstance(out, ToxHeadOutput)
    # logit shape (B,) or (B, 1)
    assert out.logit.shape in {(BATCH,), (BATCH, 1)}, (
        f"Expected logit shape ({BATCH},) or ({BATCH}, 1), got "
        f"{tuple(out.logit.shape)}"
    )
    assert torch.isfinite(out.logit).all(), (
        "Zero-GEX-channel forward produced a non-finite logit — masking broke."
    )


# ---------------------------------------------------------------------------
# Test 2: combiner feed [B, n_regions, 10716] -> [B, 10716] -> logit (WIRE-02)
# ---------------------------------------------------------------------------


def test_combiner_feed():
    """AttentionPoolCombiner.pooled feeds the head; chain yields a finite logit."""
    torch.manual_seed(1)
    combiner = AttentionPoolCombiner(d=D_GEX)
    head = ToxHead(d_gex=D_GEX, d_chem=D_CHEM, d_dr=0)

    x = torch.randn(BATCH, N_REGIONS, D_GEX)
    combined = combiner(x)
    assert combined.pooled.shape == (BATCH, D_GEX), (
        f"Combiner pooled shape: expected ({BATCH}, {D_GEX}), "
        f"got {tuple(combined.pooled.shape)}"
    )

    chem_emb = torch.randn(BATCH, D_CHEM)
    out = head(combined.pooled, chem_emb)

    assert out.logit.shape in {(BATCH,), (BATCH, 1)}
    assert torch.isfinite(out.logit).all(), (
        "[B, n_regions, 10716] -> [B, 10716] -> logit chain produced non-finite "
        "values."
    )


# ---------------------------------------------------------------------------
# Test 3: dose-response channel optional (d_dr=0) this phase
# ---------------------------------------------------------------------------


def test_dr_channel_optional():
    """ToxHead(d_dr=0) forwards without a dr_emb (dose-response zero this phase)."""
    torch.manual_seed(2)
    head = ToxHead(d_gex=D_GEX, d_chem=D_CHEM, d_dr=0)
    gex_pooled = torch.randn(BATCH, D_GEX)
    chem_emb = torch.randn(BATCH, D_CHEM)

    # dr_emb defaults to None when d_dr == 0.
    out = head(gex_pooled, chem_emb, dr_emb=None)
    assert torch.isfinite(out.logit).all()


# ---------------------------------------------------------------------------
# Test 4: ToxHeadOutput is a NamedTuple
# ---------------------------------------------------------------------------


def test_output_is_named_tuple():
    """ToxHeadOutput must be a NamedTuple with fields (logit, attn_weights)."""
    torch.manual_seed(3)
    head = ToxHead(d_gex=D_GEX, d_chem=D_CHEM, d_dr=0)
    out = head(torch.zeros(BATCH, D_GEX), torch.randn(BATCH, D_CHEM))

    assert isinstance(out, ToxHeadOutput)
    assert hasattr(out, "logit"), "ToxHeadOutput missing field 'logit'"
    assert hasattr(out, "attn_weights"), (
        "ToxHeadOutput missing field 'attn_weights'"
    )
