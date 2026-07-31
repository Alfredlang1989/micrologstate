from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class MicroLogClassifier(nn.Module):
    """Frozen text encoder output -> tiny shared adapter -> fixed categorical heads."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        label_sizes: Mapping[str, int],
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict(
            {name: nn.Linear(self.hidden_dim, int(size)) for name, size in label_sizes.items()}
        )

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.adapter(embeddings)
        return {name: head(hidden) for name, head in self.heads.items()}
