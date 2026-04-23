import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FeedForward(nn.Module):
    """
    Gated MLP used in Gemma/LLaMA-family blocks:
      y = down( gelu(gate(x)) * up(x) )
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        *,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.gate = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.up   = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.down = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, emb_dim]
        return self.down(F.gelu(self.gate(x)) * self.up(x))
