"""
Phase 4 — DILI consumer classifier heads.

Three head architectures for the 7-way pathway ablation:
- LinearHead: input_dim -> 1 (raw logit, no hidden layers)
- MLP1Head:   input_dim -> 128 -> 1 (ReLU, Dropout 0.3)
- MLP2Head:   input_dim -> 256 -> 64 -> 1 (ReLU, BatchNorm1d, Dropout 0.3)

All heads output raw logits (not sigmoid). Use BCEWithLogitsLoss during training.
Hyperparameters are LOCKED across all 7 ablation variants per Phase 4 spec.
"""
import torch
import torch.nn as nn
from typing import Literal


class LinearHead(nn.Module):
    """Single linear layer: input_dim -> 1 (raw logit)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)  # [batch]


class MLP1Head(nn.Module):
    """Two-layer MLP: input_dim -> 128 -> 1 (ReLU, Dropout 0.3)."""

    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [batch]


class MLP2Head(nn.Module):
    """Three-layer MLP: input_dim -> 256 -> 64 -> 1 (ReLU, BatchNorm1d, Dropout 0.3)."""

    def __init__(self, input_dim: int, h1: int = 256, h2: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [batch]


HeadName = Literal["linear", "mlp1", "mlp2"]


def build_head(head_name: HeadName, input_dim: int) -> nn.Module:
    """Factory: return the requested head for a given input dim."""
    if head_name == "linear":
        return LinearHead(input_dim)
    elif head_name == "mlp1":
        return MLP1Head(input_dim)
    elif head_name == "mlp2":
        return MLP2Head(input_dim)
    else:
        raise ValueError(f"Unknown head: {head_name!r}. Choose from: linear, mlp1, mlp2")
