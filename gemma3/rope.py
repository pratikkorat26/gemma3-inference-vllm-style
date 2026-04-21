# Author: Pratik Korat
# Production-ready RoPE (Rotary Position Embeddings) for Gemma-style models.
#
# Goals:
# - Readable
# - Efficient (fp32 cached cos/sin, lazy growth)
# - Correct for training + KV-cache inference (offset support)
# - DDP/FSDP safe (buffers, not parameters)
#
# Usage:
#   rope = RotaryEmbedding(dim=head_dim, theta=1e4, max_position_embeddings=32768)
#   cos, sin = rope.get_cos_sin(seq_len=T, offset=past_len, device=x.device, dtype=x.dtype)
#   q = apply_rope_single(q, cos, sin)   # q: [B,H,T,D]
#   k = apply_rope_single(k, cos, sin)   # k: [B,G,T,D]

import torch
import torch.nn as nn
from typing import Tuple, Optional


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: [..., D]
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to a single tensor.

    Args:
      x:   [B, H, T, D] or [B, G, T, D] (any leading dims ok as long as last two are [T,D])
      cos: [T, D]
      sin: [T, D]

    Returns:
      x_rope: same shape as x
    """
    # Broadcast to [1,1,T,D] then rely on broadcasting for any leading dims
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,T,D]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1,1,T,D]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """
    Production-ready RoPE cache.

    - inv_freq stored as fp32 buffer
    - cos/sin cached as fp32 buffers
    - lazy cache growth (avoids frequent rebuilds)
    - offset-aware (KV cache / generation)
    - safe for DDP/FSDP (buffers)
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        cache_dtype: torch.dtype = torch.float32,
        growth_factor: float = 1.25,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even")

        self.dim = dim
        self.theta = float(theta)
        self.cache_dtype = cache_dtype
        self.growth_factor = float(growth_factor)

        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # fp32 caches on the module device
        self.register_buffer("cos_cached", torch.empty(0, dtype=self.cache_dtype), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0, dtype=self.cache_dtype), persistent=False)
        self.max_seq_len_cached: int = 0

        self._build_cache(max_position_embeddings, device=self.inv_freq.device)

    @torch.no_grad()
    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None) -> None:
        """Build/extend cache to at least seq_len on `device`."""
        if device is None:
            device = self.inv_freq.device

        # Ensure inv_freq is on the target device for outer()
        inv = self.inv_freq.to(device=device)

        positions = torch.arange(seq_len, device=device, dtype=torch.float32)   # [seq_len]
        freqs = torch.outer(positions, inv)                                    # [seq_len, dim/2]
        angles = torch.cat([freqs, freqs], dim=-1)                             # [seq_len, dim]

        self.cos_cached = angles.cos().to(dtype=self.cache_dtype)
        self.sin_cached = angles.sin().to(dtype=self.cache_dtype)
        self.max_seq_len_cached = seq_len

    def get_cos_sin(
        self,
        seq_len: int,
        offset: int = 0,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos/sin for absolute positions [offset, offset+seq_len).

        Shapes:
          cos, sin: [seq_len, dim] (returned on `device` and in `dtype`)
        """
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if offset < 0:
            raise ValueError("offset must be >= 0")

        needed = offset + seq_len

        # If module buffers are not on requested device (unusual), rebuild cache on device.
        if self.cos_cached.device != device:
            # Rebuild on target device at least to current cached len or needed len
            target_len = max(needed, self.max_seq_len_cached)
            self._build_cache(target_len, device=device)

        if needed > self.max_seq_len_cached:
            new_len = max(needed, int(self.max_seq_len_cached * self.growth_factor) + 1)
            self._build_cache(new_len, device=device)

        cos = self.cos_cached[offset : offset + seq_len]
        sin = self.sin_cached[offset : offset + seq_len]

        # Cast only at the end; keep cache in fp32
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)

        return cos, sin
