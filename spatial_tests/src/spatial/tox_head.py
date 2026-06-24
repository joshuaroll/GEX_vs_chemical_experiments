"""Trainable concat-MLP organ-toxicity head (WIRE-02, CON-tox-head).

Pure library. No file I/O; no hardcoded paths; no real-data filenames.

Context (spatial milestone, Phase 2):
  The three-channel pipeline feeds this head:

    chem channel  -- a chemical-structure embedding (ECFP4 2048-d for the
                     condition-A smoke-train; the ChemBERTa/MolFormer/GIN/UniMol
                     headline encoder is a deferred P4 decision).
    GEX channel   -- region-pooled predicted DE (top-k of the 10,716 PDG genes),
                     produced by ``region_combiner.AttentionPoolCombiner(d=10716)``
                     over per-region predicted-DE signatures.
    dr channel    -- dose-response embedding. A first-class channel in the design,
                     but ZEROED this phase (d_dr=0); conditions G/H wire it later.

  Each active channel is linearly projected to a common ``d_proj`` width, the
  projections are concatenated, and a 3-layer MLP (GELU + dropout + batchnorm,
  CON-tox-head) maps the concat to a single organ-tox logit.

Per-condition zero-tensor channel masking (RESEARCH Pattern 3 / WIRE-02):
  An inactive channel is fed as a zero tensor of identical shape, NOT special-cased
  in code. Condition A (structure-only) feeds ``torch.zeros(B, d_gex)`` to the GEX
  channel and must still produce a finite logit -- a zeroed channel projects to the
  projection layer's bias, which is finite, so the masking is a no-op in the forward
  and never introduces a NaN. This is exactly what ``test_tox_head.py`` asserts.

Hard rules honored:
    - Pure library: NO hardcoded absolute paths (no ``/raid``), NO real-data
      filenames, NO torch checkpoints loaded here. The head is the only trainable
      module this phase; the backbone is frozen elsewhere.
    - DE rule: the GEX channel consumes region-pooled predicted DE (rule B), never
      raw expression. This module is agnostic to how the DE was produced; it only
      receives the already-pooled feature.
    - No mock / synthetic labels: this module is a pure ``nn.Module``; training data
      is supplied by the caller (the smoke-train driver uses real DILI labels).

Analog: ``src/spatial/region_combiner.py`` (``AttentionPoolCombiner`` -- same
nn.Module + NamedTuple-output style, ``__init__`` dimension validation, and the
``forward`` shape-guard idiom).

Usage example::

    from src.spatial.region_combiner import AttentionPoolCombiner
    from src.spatial.tox_head import ToxHead

    combiner = AttentionPoolCombiner(d=10716)
    head = ToxHead(d_gex=10716, d_chem=2048, d_dr=0)

    # x: (batch, n_regions, 10716) per-region predicted-DE tensors
    pooled = combiner(x)                       # RegionCombinerOutput
    out = head(pooled.pooled, chem_emb,        # ToxHeadOutput
               attn_weights=pooled.attn_weights)
    logit = out.logit                          # (batch,)
    attn = out.attn_weights                    # passthrough for P6 interpretability
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

__all__ = [
    "ToxHeadOutput",
    "ToxHead",
]


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


class ToxHeadOutput(NamedTuple):
    """Output of the toxicity head.

    Attributes
    ----------
    logit : torch.Tensor
        Organ-toxicity logit of shape ``(B,)`` (one logit per sample). Feed to
        ``BCEWithLogitsLoss`` for training; apply ``sigmoid`` for a probability.
    attn_weights : torch.Tensor | None
        Per-region attention weights passed through from the region combiner
        (shape ``(B, n_regions)``), kept for P6 regional-attribution
        interpretability. ``None`` when no combiner weights were supplied.
    """

    logit: torch.Tensor
    attn_weights: torch.Tensor | None


# ---------------------------------------------------------------------------
# ToxHead
# ---------------------------------------------------------------------------


class ToxHead(nn.Module):
    """Concat-MLP organ-toxicity head (CON-tox-head).

    Projects each active channel (GEX, chem, optional dose-response) to a common
    ``d_proj`` width, concatenates the projections, and runs a 3-layer MLP
    (GELU + dropout + batchnorm) to a single organ-tox logit.

    Per-condition zero-tensor masking: an inactive channel is fed as a zero
    tensor of identical shape (no special-casing). A zeroed channel projects to
    its layer bias (finite), so condition A (zero GEX channel) still yields a
    finite logit.

    Parameters
    ----------
    d_gex : int
        Width of the (region-pooled) GEX channel. Default 10716 (PDG space).
    d_chem : int
        Width of the chemical-structure channel. Default 2048 (ECFP4).
    d_dr : int
        Width of the dose-response channel. ``0`` disables the channel
        (no ``proj_dr``); the head then expects two channels. Default 0.
    d_proj : int
        Common projection width each channel is mapped to. Default 256.
    hidden : tuple[int, int]
        Hidden widths of the 3-layer MLP (concat -> hidden[0] -> hidden[1] -> 1).
        Default (256, 64).
    dropout : float
        Dropout probability in the MLP. Default 0.3.

    Shapes
    ------
    Input  gex_pooled : (B, d_gex)
    Input  chem_emb   : (B, d_chem)
    Input  dr_emb     : (B, d_dr) or None  (required only when d_dr > 0)
    Output logit      : (B,)
    """

    def __init__(
        self,
        d_gex: int = 10716,
        d_chem: int = 2048,
        d_dr: int = 0,
        d_proj: int = 256,
        hidden: tuple[int, int] = (256, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Dimension validation (mirror region_combiner.AttentionPoolCombiner.__init__).
        if d_gex <= 0:
            raise ValueError(f"ToxHead: d_gex must be > 0, got {d_gex}")
        if d_chem <= 0:
            raise ValueError(f"ToxHead: d_chem must be > 0, got {d_chem}")
        if d_dr < 0:
            raise ValueError(f"ToxHead: d_dr must be >= 0, got {d_dr}")
        if d_proj <= 0:
            raise ValueError(f"ToxHead: d_proj must be > 0, got {d_proj}")
        if len(hidden) != 2 or any(h <= 0 for h in hidden):
            raise ValueError(
                f"ToxHead: hidden must be two positive ints, got {hidden}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"ToxHead: dropout must be in [0, 1), got {dropout}")

        self.d_gex = d_gex
        self.d_chem = d_chem
        self.d_dr = d_dr
        self.d_proj = d_proj

        # Per-channel projections to the common width.
        self.proj_gex = nn.Linear(d_gex, d_proj)
        self.proj_chem = nn.Linear(d_chem, d_proj)
        self.proj_dr = nn.Linear(d_dr, d_proj) if d_dr > 0 else None

        # Concat width: 2 channels (gex + chem) or 3 when dose-response is active.
        n_channels = 2 if self.proj_dr is None else 3
        concat_dim = d_proj * n_channels

        # 3-layer MLP -> single logit (CON-tox-head: GELU + dropout + batchnorm).
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(
        self,
        gex_pooled: torch.Tensor,
        chem_emb: torch.Tensor,
        dr_emb: torch.Tensor | None = None,
        attn_weights: torch.Tensor | None = None,
    ) -> ToxHeadOutput:
        """Project channels, concat, MLP -> organ-tox logit.

        Parameters
        ----------
        gex_pooled : torch.Tensor
            Region-pooled GEX feature, shape ``(B, d_gex)``. For condition A this
            is ``torch.zeros(B, d_gex)`` (still projects to a finite value).
        chem_emb : torch.Tensor
            Chemical-structure embedding, shape ``(B, d_chem)``.
        dr_emb : torch.Tensor | None
            Dose-response embedding, shape ``(B, d_dr)``. Ignored when the head
            was built with ``d_dr == 0`` (``proj_dr is None``). Required when
            ``d_dr > 0``.
        attn_weights : torch.Tensor | None
            Per-region attention weights from the region combiner, passed through
            unchanged to ``ToxHeadOutput.attn_weights`` (P6 interpretability).

        Returns
        -------
        ToxHeadOutput
            ``logit`` has shape ``(B,)``; ``attn_weights`` is the passthrough.
        """
        # Shape-guard (mirror region_combiner.forward lines 148-152): require 2-D
        # inputs (batch, feature).
        if gex_pooled.dim() != 2:
            raise ValueError(
                f"ToxHead.forward: gex_pooled must be 2-D (B, d_gex), got shape "
                f"{tuple(gex_pooled.shape)}"
            )
        if chem_emb.dim() != 2:
            raise ValueError(
                f"ToxHead.forward: chem_emb must be 2-D (B, d_chem), got shape "
                f"{tuple(chem_emb.shape)}"
            )

        # Project each active channel. A zero tensor is a valid masked channel
        # (no special-casing) -- it projects to the layer bias, which is finite.
        proj = [self.proj_gex(gex_pooled), self.proj_chem(chem_emb)]

        if self.proj_dr is not None:
            if dr_emb is None:
                raise ValueError(
                    "ToxHead.forward: head was built with d_dr > 0 but dr_emb is "
                    "None. Pass a (B, d_dr) tensor (a zero tensor for a masked "
                    "dose-response channel)."
                )
            if dr_emb.dim() != 2:
                raise ValueError(
                    f"ToxHead.forward: dr_emb must be 2-D (B, d_dr), got shape "
                    f"{tuple(dr_emb.shape)}"
                )
            proj.append(self.proj_dr(dr_emb))
        # When proj_dr is None, dr_emb is ignored (dose-response zero this phase).

        concat = torch.cat(proj, dim=-1)        # (B, d_proj * n_channels)
        logit = self.mlp(concat).squeeze(-1)    # (B,)

        return ToxHeadOutput(logit=logit, attn_weights=attn_weights)
