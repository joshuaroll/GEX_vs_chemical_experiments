"""Unit tests for `src/classifiers/region_combiner.py`.

In-memory tensor fixtures — no real-data dependency, no file I/O.

Behaviors covered:
  1.  AttentionPoolCombiner output shape: (batch, d).
  2.  Attention weights shape: (batch, n_regions).
  3.  Attention weights are non-negative.
  4.  Attention weights sum to 1 per sample (row-wise).
  5.  MeanPoolCombiner output shape: (batch, d).
  6.  MeanPoolCombiner equals plain torch.mean across region axis.
  7.  MeanPoolCombiner attn_weights is None.
  8.  ConcatCombiner output shape: (batch, d).
  9.  ConcatCombiner attn_weights is None.
  10. Unbatched input (n_regions, d) — single-sample, no batch dim.
  11. Higher-rank batch dim (batch1, batch2, n_regions, d).
  12. AttentionPoolCombiner is an nn.Module with a trainable parameter.
  13. MeanPoolCombiner has no learnable parameters.
  14. ConcatCombiner wrong n_regions raises ValueError.
  15. RegionCombinerOutput is a NamedTuple with fields (pooled, attn_weights).
  16. Gradient flows through AttentionPoolCombiner (autograd smoke test).
  17. Different temperature values change attention sharpness.
  18. AttentionPoolCombiner with n_regions=1 → weight = 1.0, pooled == x[..., 0, :].
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.spatial.region_combiner import (
    AttentionPoolCombiner,
    ConcatCombiner,
    MeanPoolCombiner,
    RegionCombinerOutput,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH = 8
N_REGIONS = 5
D = 64

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def x_batched() -> torch.Tensor:
    """Shape (BATCH, N_REGIONS, D) — standard batched input."""
    torch.manual_seed(0)
    return torch.randn(BATCH, N_REGIONS, D)


@pytest.fixture
def x_unbatched() -> torch.Tensor:
    """Shape (N_REGIONS, D) — single sample, no batch dimension."""
    torch.manual_seed(1)
    return torch.randn(N_REGIONS, D)


@pytest.fixture
def x_2d_batch() -> torch.Tensor:
    """Shape (3, 4, N_REGIONS, D) — higher-rank batch dimensions."""
    torch.manual_seed(2)
    return torch.randn(3, 4, N_REGIONS, D)


@pytest.fixture
def attn_combiner() -> AttentionPoolCombiner:
    torch.manual_seed(42)
    return AttentionPoolCombiner(d=D)


@pytest.fixture
def mean_combiner() -> MeanPoolCombiner:
    return MeanPoolCombiner(d=D)


@pytest.fixture
def concat_combiner() -> ConcatCombiner:
    torch.manual_seed(42)
    return ConcatCombiner(d=D, n_regions=N_REGIONS)


# ---------------------------------------------------------------------------
# Test 1-4: AttentionPoolCombiner output shape and weight properties
# ---------------------------------------------------------------------------


def test_1_attn_pooled_output_shape(attn_combiner, x_batched):
    """pooled must have shape (batch, d)."""
    out = attn_combiner(x_batched)
    assert out.pooled.shape == (BATCH, D), (
        f"Expected pooled shape {(BATCH, D)}, got {tuple(out.pooled.shape)}"
    )


def test_2_attn_weights_shape(attn_combiner, x_batched):
    """attn_weights must have shape (batch, n_regions)."""
    out = attn_combiner(x_batched)
    assert out.attn_weights is not None
    assert out.attn_weights.shape == (BATCH, N_REGIONS), (
        f"Expected attn_weights shape {(BATCH, N_REGIONS)}, "
        f"got {tuple(out.attn_weights.shape)}"
    )


def test_3_attn_weights_non_negative(attn_combiner, x_batched):
    """Softmax output is always non-negative."""
    out = attn_combiner(x_batched)
    assert out.attn_weights is not None
    assert (out.attn_weights >= 0).all(), (
        "Attention weights contain negative values — softmax invariant violated."
    )


def test_4_attn_weights_sum_to_one(attn_combiner, x_batched):
    """Each sample's attention weights must sum to 1 (within floating-point tol)."""
    out = attn_combiner(x_batched)
    assert out.attn_weights is not None
    row_sums = out.attn_weights.sum(dim=-1)  # (batch,)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"Attention weight row sums deviate from 1.0. "
        f"Max deviation: {(row_sums - 1).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 5-7: MeanPoolCombiner
# ---------------------------------------------------------------------------


def test_5_mean_pooled_output_shape(mean_combiner, x_batched):
    """pooled must have shape (batch, d)."""
    out = mean_combiner(x_batched)
    assert out.pooled.shape == (BATCH, D), (
        f"Expected pooled shape {(BATCH, D)}, got {tuple(out.pooled.shape)}"
    )


def test_6_mean_pool_equals_plain_mean(mean_combiner, x_batched):
    """MeanPoolCombiner.pooled must equal x.mean(dim=-2) exactly."""
    out = mean_combiner(x_batched)
    expected = x_batched.mean(dim=-2)
    assert torch.allclose(out.pooled, expected), (
        "MeanPoolCombiner output differs from x.mean(dim=-2). "
        f"Max abs diff: {(out.pooled - expected).abs().max().item():.2e}"
    )


def test_7_mean_combiner_attn_weights_is_none(mean_combiner, x_batched):
    """MeanPoolCombiner does not produce attention weights."""
    out = mean_combiner(x_batched)
    assert out.attn_weights is None, (
        f"Expected attn_weights=None for MeanPoolCombiner, got {out.attn_weights}"
    )


# ---------------------------------------------------------------------------
# Test 8-9: ConcatCombiner
# ---------------------------------------------------------------------------


def test_8_concat_pooled_output_shape(concat_combiner, x_batched):
    """pooled must have shape (batch, d)."""
    out = concat_combiner(x_batched)
    assert out.pooled.shape == (BATCH, D), (
        f"Expected pooled shape {(BATCH, D)}, got {tuple(out.pooled.shape)}"
    )


def test_9_concat_attn_weights_is_none(concat_combiner, x_batched):
    """ConcatCombiner does not produce attention weights."""
    out = concat_combiner(x_batched)
    assert out.attn_weights is None, (
        f"Expected attn_weights=None for ConcatCombiner, got {out.attn_weights}"
    )


# ---------------------------------------------------------------------------
# Test 10: Unbatched input (n_regions, d)
# ---------------------------------------------------------------------------


def test_10_unbatched_input(attn_combiner, mean_combiner, concat_combiner, x_unbatched):
    """All combiners accept (n_regions, d) input (no batch dim) and return (d,)."""
    # AttentionPool
    out_a = attn_combiner(x_unbatched)
    assert out_a.pooled.shape == (D,), (
        f"AttentionPoolCombiner unbatched pooled shape: expected ({D},), "
        f"got {tuple(out_a.pooled.shape)}"
    )
    assert out_a.attn_weights is not None
    assert out_a.attn_weights.shape == (N_REGIONS,), (
        f"AttentionPoolCombiner unbatched attn_weights shape: expected ({N_REGIONS},), "
        f"got {tuple(out_a.attn_weights.shape)}"
    )
    # MeanPool
    out_m = mean_combiner(x_unbatched)
    assert out_m.pooled.shape == (D,), (
        f"MeanPoolCombiner unbatched pooled shape: expected ({D},), "
        f"got {tuple(out_m.pooled.shape)}"
    )
    # Concat
    out_c = concat_combiner(x_unbatched)
    assert out_c.pooled.shape == (D,), (
        f"ConcatCombiner unbatched pooled shape: expected ({D},), "
        f"got {tuple(out_c.pooled.shape)}"
    )


# ---------------------------------------------------------------------------
# Test 11: Higher-rank batch dimensions
# ---------------------------------------------------------------------------


def test_11_higher_rank_batch(attn_combiner, mean_combiner, x_2d_batch):
    """(batch1, batch2, n_regions, d) inputs produce (batch1, batch2, d) output."""
    # AttentionPool
    out_a = attn_combiner(x_2d_batch)
    assert out_a.pooled.shape == (3, 4, D), (
        f"AttentionPoolCombiner 2D-batch pooled: expected (3, 4, {D}), "
        f"got {tuple(out_a.pooled.shape)}"
    )
    assert out_a.attn_weights is not None
    assert out_a.attn_weights.shape == (3, 4, N_REGIONS), (
        f"AttentionPoolCombiner 2D-batch attn_weights: expected (3, 4, {N_REGIONS}), "
        f"got {tuple(out_a.attn_weights.shape)}"
    )
    # attn_weights must still sum to 1 along the region axis
    row_sums = out_a.attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # MeanPool
    out_m = mean_combiner(x_2d_batch)
    assert out_m.pooled.shape == (3, 4, D), (
        f"MeanPoolCombiner 2D-batch pooled: expected (3, 4, {D}), "
        f"got {tuple(out_m.pooled.shape)}"
    )


# ---------------------------------------------------------------------------
# Test 12: AttentionPoolCombiner is an nn.Module with trainable parameter
# ---------------------------------------------------------------------------


def test_12_attn_combiner_is_module_with_params(attn_combiner):
    """AttentionPoolCombiner must be an nn.Module and expose learnable params."""
    assert isinstance(attn_combiner, nn.Module), (
        "AttentionPoolCombiner must subclass nn.Module"
    )
    params = list(attn_combiner.parameters())
    assert len(params) > 0, "AttentionPoolCombiner must have at least one learnable parameter"
    assert any(p.requires_grad for p in params), (
        "AttentionPoolCombiner has no parameter with requires_grad=True"
    )


# ---------------------------------------------------------------------------
# Test 13: MeanPoolCombiner has no learnable parameters
# ---------------------------------------------------------------------------


def test_13_mean_combiner_no_params(mean_combiner):
    """MeanPoolCombiner is non-parametric — parameter list must be empty."""
    params = list(mean_combiner.parameters())
    assert len(params) == 0, (
        f"MeanPoolCombiner should have 0 parameters, found {len(params)}"
    )


# ---------------------------------------------------------------------------
# Test 14: ConcatCombiner wrong n_regions raises ValueError
# ---------------------------------------------------------------------------


def test_14_concat_wrong_n_regions_raises():
    """Passing input with wrong n_regions to ConcatCombiner must raise ValueError."""
    combiner = ConcatCombiner(d=D, n_regions=N_REGIONS)
    wrong_n_regions = N_REGIONS + 2
    x_wrong = torch.randn(BATCH, wrong_n_regions, D)
    with pytest.raises(ValueError, match="n_regions"):
        combiner(x_wrong)


# ---------------------------------------------------------------------------
# Test 15: RegionCombinerOutput is a NamedTuple
# ---------------------------------------------------------------------------


def test_15_output_is_named_tuple(attn_combiner, x_batched):
    """RegionCombinerOutput must be a NamedTuple with fields pooled and attn_weights."""
    out = attn_combiner(x_batched)
    assert isinstance(out, RegionCombinerOutput), (
        f"Expected RegionCombinerOutput, got {type(out)}"
    )
    assert hasattr(out, "pooled"), "RegionCombinerOutput missing field 'pooled'"
    assert hasattr(out, "attn_weights"), "RegionCombinerOutput missing field 'attn_weights'"
    # NamedTuple supports index access
    assert out[0] is out.pooled
    assert out[1] is out.attn_weights


# ---------------------------------------------------------------------------
# Test 16: Gradient flows through AttentionPoolCombiner
# ---------------------------------------------------------------------------


def test_16_gradient_flows_through_attn_combiner(attn_combiner, x_batched):
    """Autograd smoke test: loss.backward() should populate query.grad."""
    x = x_batched.requires_grad_(True)
    out = attn_combiner(x)
    # Scalar loss
    loss = out.pooled.sum()
    loss.backward()
    # query param should have received gradient
    assert attn_combiner.query.grad is not None, (
        "query.grad is None after backward — gradient is not flowing."
    )
    assert not torch.isnan(attn_combiner.query.grad).any(), (
        "query.grad contains NaN — numerical issue in forward."
    )
    # Input should also receive gradient (x has requires_grad=True)
    assert x.grad is not None, (
        "Input x.grad is None — gradient is not flowing through the combiner."
    )


# ---------------------------------------------------------------------------
# Test 17: Temperature sharpens attention
# ---------------------------------------------------------------------------


def test_17_temperature_sharpens_attention(x_batched):
    """Lower temperature → sharper (more peaked) attention distribution.

    With temperature → 0 the distribution concentrates on the argmax.
    Measured by comparing entropy of weights at two temperatures.
    """
    torch.manual_seed(99)
    combiner_sharp = AttentionPoolCombiner(d=D, temperature=0.01)
    combiner_flat = AttentionPoolCombiner(d=D, temperature=100.0)
    # Share the same query for a fair comparison
    with torch.no_grad():
        combiner_flat.query.copy_(combiner_sharp.query)

    out_sharp = combiner_sharp(x_batched)
    out_flat = combiner_flat(x_batched)

    def entropy(w: torch.Tensor) -> torch.Tensor:
        """Shannon entropy along the last axis; safe log (clamp away from 0)."""
        w_safe = w.clamp(min=1e-9)
        return -(w_safe * w_safe.log()).sum(dim=-1)  # (batch,)

    h_sharp = entropy(out_sharp.attn_weights).mean().item()
    h_flat = entropy(out_flat.attn_weights).mean().item()
    assert h_sharp < h_flat, (
        f"Expected low-temperature combiner to have lower entropy "
        f"(sharper attention), but got h_sharp={h_sharp:.4f} >= h_flat={h_flat:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 18: Single-region edge case
# ---------------------------------------------------------------------------


def test_18_single_region_attention(x_batched):
    """With n_regions=1, attention weight should be 1.0 and pooled == x[..., 0, :]."""
    torch.manual_seed(7)
    combiner = AttentionPoolCombiner(d=D)
    x_one = x_batched[:, :1, :]  # (batch, 1, d)
    out = combiner(x_one)
    # Weight must be 1.0 for the single region
    assert torch.allclose(out.attn_weights, torch.ones(BATCH, 1), atol=1e-6), (
        "Single-region attention weight should be exactly 1.0. "
        f"Got: {out.attn_weights}"
    )
    # pooled == the single region embedding
    assert torch.allclose(out.pooled, x_one[:, 0, :], atol=1e-6), (
        "With n_regions=1, pooled should equal the only region embedding."
    )
