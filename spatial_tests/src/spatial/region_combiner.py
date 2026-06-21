"""Region-combiner modules for multi-region predicted-DE signatures.

Pure library. No file I/O; no hardcoded paths; no real-data filenames.

Context (v0.4, Phase 3+):
  Upstream models produce per-region predicted DE signatures
  (predicted_treated(drug, region_basal) - region_basal).  When multiple
  liver zones / kidney tubule segments / etc. are available, we need to
  combine their (n_regions, d) embedding matrix into a single (d,) feature
  vector before feeding the downstream DILI classifier.

  Three strategies are provided:

    AttentionPoolCombiner — learned single-query attention over regions
        (default, recommended).  Exposes `attention_weights` for
        regional-attribution interpretability.

    MeanPoolCombiner — plain mean across the region axis (non-parametric
        baseline).

    ConcatCombiner — flattens all regions into one vector, then projects
        back to `d` (parametric baseline; fixes `n_regions` at build time).

All modules follow the same interface contract:
  - `__init__(self, d: int, n_regions: int | None = None)`
  - `forward(x) -> RegionCombinerOutput`
  where `x` has shape `(*, n_regions, d)` — leading batch dimensions allowed.

The `RegionCombinerOutput` NamedTuple carries both the pooled feature and the
attention weights (None for non-attention variants) so callers never need to
introspect the module type to get interpretability data.

Gene-space constants (for reference — not enforced here):
  N_LANDMARK = 978
  N_PDG      = 10716

Usage example::

    combiner = AttentionPoolCombiner(d=978)
    # x: (batch, n_regions, d) per-region predicted-DE tensors
    out = combiner(x)
    pooled = out.pooled          # (batch, d)
    weights = out.attn_weights   # (batch, n_regions)  — sums to 1 per sample
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

__all__ = [
    "RegionCombinerOutput",
    "AttentionPoolCombiner",
    "MeanPoolCombiner",
    "ConcatCombiner",
]


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


class RegionCombinerOutput(NamedTuple):
    """Output of any region-combiner module.

    Attributes
    ----------
    pooled : torch.Tensor
        Pooled feature vector of shape `(*, d)`.  The leading dimensions
        mirror those of the input `(*, n_regions, d)`.
    attn_weights : torch.Tensor | None
        Per-region attention weights of shape `(*, n_regions)`, summing to 1
        along the last axis.  `None` for non-attention combiners
        (MeanPoolCombiner, ConcatCombiner).
    """

    pooled: torch.Tensor
    attn_weights: torch.Tensor | None


# ---------------------------------------------------------------------------
# AttentionPoolCombiner
# ---------------------------------------------------------------------------


class AttentionPoolCombiner(nn.Module):
    """Learned single-query attention pool over region embeddings.

    A trainable query vector is dot-producted against each region embedding,
    the resulting scores are scaled and softmax-normalised to produce
    attention weights, which then weight-sum the region embeddings into a
    single pooled feature.

    This is the recommended combiner for v0.4: the attention weights serve as
    regional-attribution scores (which liver zone / kidney segment drove the
    predicted DILI signal).

    Parameters
    ----------
    d : int
        Embedding dimension of each region vector.
    temperature : float, optional
        Softmax temperature τ.  Score = (q · hᵢ) / τ.  Default 1.0.

    Shapes
    ------
    Input  x: (*, n_regions, d)   — leading batch dims allowed
    Output pooled: (*, d)
    Output attn_weights: (*, n_regions)  — non-negative, sums to 1
    """

    def __init__(self, d: int, temperature: float = 1.0) -> None:
        super().__init__()
        if d <= 0:
            raise ValueError(f"AttentionPoolCombiner: d must be > 0, got {d}")
        if temperature <= 0.0:
            raise ValueError(
                f"AttentionPoolCombiner: temperature must be > 0, got {temperature}"
            )
        self.d = d
        self.temperature = temperature
        # Learned query vector — shape (d,).
        self.query = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.query, mean=0.0, std=d ** -0.5)

    def forward(self, x: torch.Tensor) -> RegionCombinerOutput:
        """Pool region embeddings via learned attention.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(*, n_regions, d)``.  At least 2-D required.

        Returns
        -------
        RegionCombinerOutput
            ``pooled`` has shape ``(*, d)``.
            ``attn_weights`` has shape ``(*, n_regions)``, non-negative,
            summing to 1 along the last axis.
        """
        if x.dim() < 2:
            raise ValueError(
                f"AttentionPoolCombiner.forward: input must be at least 2-D "
                f"(n_regions, d), got shape {tuple(x.shape)}"
            )
        # x: (*, n_regions, d)
        # scores = x @ q / τ  →  (*, n_regions)
        scores = torch.matmul(x, self.query) / self.temperature  # (*, n_regions)
        attn_weights = F.softmax(scores, dim=-1)                 # (*, n_regions)
        # pooled = weighted sum over regions
        # attn_weights.unsqueeze(-1): (*, n_regions, 1)
        pooled = (attn_weights.unsqueeze(-1) * x).sum(dim=-2)    # (*, d)
        return RegionCombinerOutput(pooled=pooled, attn_weights=attn_weights)


# ---------------------------------------------------------------------------
# MeanPoolCombiner
# ---------------------------------------------------------------------------


class MeanPoolCombiner(nn.Module):
    """Non-parametric mean pool over the region axis.

    All regions are given equal weight.  This is the simplest baseline and
    has no learnable parameters.

    Parameters
    ----------
    d : int
        Embedding dimension (stored for interface symmetry; not used in
        computation).

    Shapes
    ------
    Input  x: (*, n_regions, d)
    Output pooled: (*, d)
    Output attn_weights: None
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d

    def forward(self, x: torch.Tensor) -> RegionCombinerOutput:
        """Average region embeddings uniformly.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(*, n_regions, d)``.  At least 2-D required.

        Returns
        -------
        RegionCombinerOutput
            ``pooled`` equals ``x.mean(dim=-2)``.
            ``attn_weights`` is ``None``.
        """
        if x.dim() < 2:
            raise ValueError(
                f"MeanPoolCombiner.forward: input must be at least 2-D "
                f"(n_regions, d), got shape {tuple(x.shape)}"
            )
        pooled = x.mean(dim=-2)  # (*, d)
        return RegionCombinerOutput(pooled=pooled, attn_weights=None)


# ---------------------------------------------------------------------------
# ConcatCombiner
# ---------------------------------------------------------------------------


class ConcatCombiner(nn.Module):
    """Flatten-and-project combiner (fixes n_regions at construction time).

    Concatenates all region embeddings into a single vector of size
    `n_regions * d`, then projects back to `d` via a linear layer.  This
    is strictly more expressive than mean-pool but requires knowing
    `n_regions` up front and cannot handle variable region counts.

    Parameters
    ----------
    d : int
        Embedding dimension of each region vector.
    n_regions : int
        Number of regions (fixed; must be ≥ 1).
    bias : bool, optional
        Whether to include a bias in the projection linear. Default ``True``.

    Shapes
    ------
    Input  x: (*, n_regions, d)   — leading dims must broadcast to 1-D batch
    Output pooled: (*, d)
    Output attn_weights: None
    """

    def __init__(self, d: int, n_regions: int, bias: bool = True) -> None:
        super().__init__()
        if d <= 0:
            raise ValueError(f"ConcatCombiner: d must be > 0, got {d}")
        if n_regions <= 0:
            raise ValueError(f"ConcatCombiner: n_regions must be > 0, got {n_regions}")
        self.d = d
        self.n_regions = n_regions
        self.proj = nn.Linear(n_regions * d, d, bias=bias)

    def forward(self, x: torch.Tensor) -> RegionCombinerOutput:
        """Flatten regions and project to d.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(*, n_regions, d)``.  The `n_regions` dimension must match
            ``self.n_regions``.

        Returns
        -------
        RegionCombinerOutput
            ``pooled`` has shape ``(*, d)``.
            ``attn_weights`` is ``None``.
        """
        if x.dim() < 2:
            raise ValueError(
                f"ConcatCombiner.forward: input must be at least 2-D "
                f"(n_regions, d), got shape {tuple(x.shape)}"
            )
        if x.shape[-2] != self.n_regions:
            raise ValueError(
                f"ConcatCombiner.forward: expected n_regions={self.n_regions} "
                f"in dim -2, got {x.shape[-2]}. Shape: {tuple(x.shape)}"
            )
        # Flatten (*, n_regions, d) → (*, n_regions * d)
        leading = x.shape[:-2]
        flat = x.reshape(*leading, self.n_regions * self.d)  # (*, n_regions*d)
        pooled = self.proj(flat)                              # (*, d)
        return RegionCombinerOutput(pooled=pooled, attn_weights=None)
