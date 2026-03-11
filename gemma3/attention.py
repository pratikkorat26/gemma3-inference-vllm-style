import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from gemma3.rope import apply_rope_single

class RMSNorm(nn.Module):
    """
    Gemma-style RMSNorm.

    - Computes variance in fp32
    - Uses zero-centered weights with (1 + scale)
    - Optional bias
    - Safe for bf16 / fp16
    """

    def __init__(self, emb_dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps

        # Gemma-style: zero-centered scale, applied as (1 + scale)
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # RMSNorm in fp32
        x_f = x.float()
        var = torch.mean(x_f * x_f, dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)

        # Apply scale (and optional bias) in fp32
        out = x_norm * (1.0 + self.scale.float())
        if self.shift is not None:
            out = out + self.shift.float()

        return out.to(orig_dtype)

class GroupedQueryAttention(nn.Module):
    """
    Unified GQA attention using PyTorch SDPA.
    - No KV expansion (no repeat_interleave)
    - Supports global + sliding window
    - RoPE compatible

    Notation:
      B = batch
      T = query length (new tokens in this forward)
      S = kv length (past + new)
      D = model dim
      H = num_heads
      G = num_kv_groups
      g = group_size = H // G
      Hd = head_dim
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int,
        rope=None,                     # RotaryEmbedding or None
        sliding_window: Optional[int] = None,
        qk_norm: bool = False,
        query_pre_attn_scalar: Optional[float] = None,
        dtype=None,
    ):
        super().__init__()
        if num_heads % num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups")
        if sliding_window is not None and sliding_window <= 0:
            raise ValueError("sliding_window must be a positive integer")

        self.num_heads = num_heads          # H
        self.num_kv_groups = num_kv_groups  # G
        self.group_size = num_heads // num_kv_groups  # g
        self.head_dim = head_dim            # Hd
        if query_pre_attn_scalar is not None and query_pre_attn_scalar <= 0:
            raise ValueError("query_pre_attn_scalar must be > 0")
        scale_base = float(query_pre_attn_scalar) if query_pre_attn_scalar is not None else float(head_dim)
        self.scale = scale_base ** -0.5
        self.sliding_window = sliding_window
        self.rope = rope

        # Projections
        # q_proj: [B,T,D] -> [B,T,H*Hd]
        # k_proj/v_proj: [B,T,D] -> [B,T,G*Hd]
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_model, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_model, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False, dtype=dtype)

        # Optional per-head RMSNorm on Q and K (Gemma-style)
        self.q_norm = RMSNorm(head_dim) if qk_norm else None
        self.k_norm = RMSNorm(head_dim) if qk_norm else None

    def _build_attn_mask(
        self,
        q_len: int,
        kv_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Returns additive attention mask [1, 1, q_len, kv_len].
        Mask values: 0 for allowed, -inf for blocked.

        This is offset-aware for KV cache decoding:
        query i maps to absolute position (kv_len - q_len + i).

        Shapes:
          mask: [q_len, kv_len] -> [1,1,q_len,kv_len]
        """
        # Absolute positions for queries and keys.
        q_pos = torch.arange(kv_len - q_len, kv_len, device=device).unsqueeze(1)  # [T,1]
        k_pos = torch.arange(kv_len, device=device).unsqueeze(0)                   # [1,S]

        mask = torch.zeros((q_len, kv_len), device=device, dtype=dtype)            # [T,S]

        # Causal mask: block keys strictly in the future of each query.
        mask = mask.masked_fill(k_pos > q_pos, float("-inf"))

        # sliding-window restriction (block attending too far in the past)
        if self.sliding_window is not None:
            min_k = q_pos - self.sliding_window + 1
            mask = mask.masked_fill(k_pos < min_k, float("-inf"))

        return mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,S]

    def _validate_past_kv(
        self,
        past_kv: tuple,
        *,
        batch_size: int,
    ) -> None:
        if not isinstance(past_kv, tuple) or len(past_kv) != 2:
            raise ValueError("past_kv must be a tuple of (past_k, past_v)")

        past_k, past_v = past_kv
        if past_k.ndim != 4 or past_v.ndim != 4:
            raise ValueError("past_kv tensors must be 4D: [B, G, S_past, Hd]")

        if past_k.shape != past_v.shape:
            raise ValueError("past_k and past_v must have the same shape")

        expected = (batch_size, self.num_kv_groups, self.head_dim)
        got = (past_k.shape[0], past_k.shape[1], past_k.shape[3])
        if got != expected:
            raise ValueError(
                f"past_kv shape mismatch. Expected [B={expected[0]}, G={expected[1]}, S, Hd={expected[2]}], "
                f"got {tuple(past_k.shape)}"
            )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple] = None,
        use_cache: bool = False,
    ):
        """
        Inputs:
          x:        [B, T, D]
          past_kv:  (past_k, past_v) or None
                   past_k: [B, G, S_past, Hd]
                   past_v: [B, G, S_past, Hd]

        Returns:
          y:        [B, T, D]
          next_kv:  (k, v) if use_cache else None
                   k: [B, G, S, Hd]
                   v: [B, G, S, Hd]
        """
        B, T, _ = x.shape
        device = x.device

        if past_kv is not None:
            self._validate_past_kv(past_kv, batch_size=B)

        # ---- Projections ----
        # q_lin: [B,T,H*Hd]
        # k_lin: [B,T,G*Hd]
        # v_lin: [B,T,G*Hd]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ---- Reshape to head/group format ----
        # q: [B,T,H,Hd] -> [B,H,T,Hd]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,T,Hd]
        # k/v: [B,T,G,Hd] -> [B,G,T,Hd]
        k = k.view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)  # [B,G,T,Hd]
        v = v.view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)  # [B,G,T,Hd]

        # ---- Optional QK Norm ----
        # preserves shapes: q [B,H,T,Hd], k [B,G,T,Hd]
        if self.q_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # ---- RoPE (offset-aware) ----
        # past_len: scalar S_past
        past_len = past_kv[0].shape[2] if past_kv is not None else 0

        # cos/sin: [T,Hd]
        if self.rope is not None:
            cos, sin = self.rope.get_cos_sin(
                seq_len=T,
                offset=past_len,
                device=device,
                dtype=x.dtype,
            )
            # apply_rope_single expects x: [B,*,T,Hd], cos/sin: [T,Hd]
            q = apply_rope_single(q, cos, sin)  # [B,H,T,Hd]
            k = apply_rope_single(k, cos, sin)  # [B,G,T,Hd]

        # ---- Append KV cache if provided ----
        # past_k/v: [B,G,S_past,Hd]
        # new k/v:  [B,G,T,Hd]
        # cat ->    [B,G,S,Hd] where S = S_past + T
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)  # [B,G,S,Hd]
            v = torch.cat([past_kv[1], v], dim=2)  # [B,G,S,Hd]

        S = k.shape[2]  # kv_len
        next_kv = (k, v) if use_cache else None  # k/v: [B,G,S,Hd]

        # ============================================================
        # GQA WITHOUT KV EXPANSION
        #
        # Goal: SDPA expects q,k,v to share same head dimension.
        # We'll reshape so that q has "G heads" but with batch multiplied by group_size (g).
        #
        # q: [B,H,T,Hd] -> [B,G,g,T,Hd] -> [B*g, G, T, Hd]
        # k: [B,G,S,Hd] -> [B,1,G,S,Hd] expand over g -> [B,g,G,S,Hd] -> [B*g, G, S, Hd]
        # v: same as k
        # ============================================================

        q = q.reshape(B, self.num_kv_groups, self.group_size, T, self.head_dim)  # [B,G,g,T,Hd]
        q = q.reshape(B * self.group_size, self.num_kv_groups, T, self.head_dim) # [B*g,G,T,Hd]

        k = k.unsqueeze(1)  # [B,1,G,S,Hd]
        v = v.unsqueeze(1)  # [B,1,G,S,Hd]

        k = k.expand(B, self.group_size, -1, -1, -1)  # [B,g,G,S,Hd] (view, no copy)
        v = v.expand(B, self.group_size, -1, -1, -1)  # [B,g,G,S,Hd]

        k = k.reshape(B * self.group_size, self.num_kv_groups, S, self.head_dim) # [B*g,G,S,Hd]
        v = v.reshape(B * self.group_size, self.num_kv_groups, S, self.head_dim) # [B*g,G,S,Hd]

        # Reuse SDPA causal fast path when the sequence shape allows it.
        use_causal = False
        attn_mask = None
        if self.sliding_window is None:
            if past_kv is None:
                use_causal = True
            elif T > 1:
                attn_mask = self._build_attn_mask(T, S, device=device, dtype=q.dtype)
        else:
            attn_mask = self._build_attn_mask(T, S, device=device, dtype=q.dtype)

        # ---- SDPA ----
        # out: [B*g, G, T, Hd]
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=use_causal,
            scale=self.scale,
        )

        # ---- Restore original head layout ----
        # out: [B*g,G,T,Hd] -> [B,g,G,T,Hd] -> [B,H,T,Hd] -> [B,T,H*Hd]
        out = out.reshape(B, self.group_size, self.num_kv_groups, T, self.head_dim) # [B,g,G,T,Hd]
        out = out.reshape(B, self.num_heads, T, self.head_dim)                   # [B,H,T,Hd]
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)  # [B,T,H*Hd]

        # final projection: [B,T,D]
        return self.out_proj(out), next_kv
