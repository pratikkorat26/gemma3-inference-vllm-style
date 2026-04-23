import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from gemma3.rope import apply_rope_single
from gemma3.paged_kv import PagedKVCache

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

    def _append_to_paged_cache(
        self,
        *,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        paged_kv_cache: PagedKVCache,
    ) -> None:
        _, _, q_len, _ = k_new.shape
        block_size = paged_kv_cache.block_size
        for batch_idx in range(k_new.shape[0]):
            start_pos = int(kv_lens[batch_idx].item())
            for token_offset in range(q_len):
                position = start_pos + token_offset
                block_slot = position // block_size
                block_offset = position % block_size
                block_id = int(block_tables[batch_idx, block_slot].item())
                if block_id < 0:
                    raise ValueError("paged KV block table is missing an assigned block")
                paged_kv_cache.k_blocks[block_id, :, block_offset, :] = k_new[batch_idx, :, token_offset, :]
                paged_kv_cache.v_blocks[block_id, :, block_offset, :] = v_new[batch_idx, :, token_offset, :]

    def _gather_sequence_kv(
        self,
        *,
        block_table: torch.Tensor,
        seq_len: int,
        paged_kv_cache: PagedKVCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = paged_kv_cache.block_size
        needed_blocks = (seq_len + block_size - 1) // block_size
        block_ids = block_table[:needed_blocks]
        valid_block_ids = block_ids[block_ids >= 0]
        if valid_block_ids.numel() != needed_blocks:
            raise ValueError("paged KV block table does not cover the active sequence")

        k_chunks = paged_kv_cache.k_blocks[valid_block_ids]
        v_chunks = paged_kv_cache.v_blocks[valid_block_ids]
        k_seq = k_chunks.permute(1, 0, 2, 3).reshape(
            self.num_kv_groups,
            needed_blocks * block_size,
            self.head_dim,
        )
        v_seq = v_chunks.permute(1, 0, 2, 3).reshape(
            self.num_kv_groups,
            needed_blocks * block_size,
            self.head_dim,
        )
        return k_seq[:, :seq_len, :], v_seq[:, :seq_len, :]

    def _forward_paged(
        self,
        x: torch.Tensor,
        *,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        paged_kv_cache: PagedKVCache,
    ) -> torch.Tensor:
        batch_size, q_len, _ = x.shape
        device = x.device

        q = self.q_proj(x).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, q_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, q_len, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope is not None:
            q_rope = []
            k_rope = []
            for batch_idx in range(batch_size):
                offset = int(kv_lens[batch_idx].item())
                cos, sin = self.rope.get_cos_sin(seq_len=q_len, offset=offset, device=device, dtype=x.dtype)
                q_rope.append(apply_rope_single(q[batch_idx : batch_idx + 1], cos, sin))
                k_rope.append(apply_rope_single(k[batch_idx : batch_idx + 1], cos, sin))
            q = torch.cat(q_rope, dim=0)
            k = torch.cat(k_rope, dim=0)

        # NOTE: paged KV writes are required for correctness (tokens must be
        # in the cache to attend to themselves), so they happen regardless of
        # use_cache. use_cache only controls whether next_kv is returned.
        self._append_to_paged_cache(
            k_new=k,
            v_new=v,
            block_tables=block_tables,
            kv_lens=kv_lens,
            paged_kv_cache=paged_kv_cache,
        )

        # NOTE: This loop processes batch items individually. It is readable
        # and correct, but not GPU-optimal. A production implementation would
        # fuse these operations across the batch dimension.
        outputs = []
        for batch_idx in range(batch_size):
            past_len = int(kv_lens[batch_idx].item())
            seq_len = past_len + q_len

            if past_len > 0:
                # Chunked prefill / decode with prior context: gather only the
                # past tokens from paged blocks and concat with freshly computed KV.
                k_past, v_past = self._gather_sequence_kv(
                    block_table=block_tables[batch_idx],
                    seq_len=past_len,
                    paged_kv_cache=paged_kv_cache,
                )
                k_seq = torch.cat([k_past, k[batch_idx]], dim=1)
                v_seq = torch.cat([v_past, v[batch_idx]], dim=1)
            else:
                # Pure prefill with no prior context: freshly computed KV is
                # already contiguous and correct. Skip the redundant gather.
                k_seq = k[batch_idx]
                v_seq = v[batch_idx]

            q_batch = q[batch_idx : batch_idx + 1]
            k_batch = k_seq.unsqueeze(0)
            v_batch = v_seq.unsqueeze(0)

            q_batch = q_batch.reshape(1, self.num_kv_groups, self.group_size, q_len, self.head_dim)
            q_batch = q_batch.reshape(self.group_size, self.num_kv_groups, q_len, self.head_dim)

            k_batch = k_batch.unsqueeze(1).expand(1, self.group_size, -1, -1, -1)
            v_batch = v_batch.unsqueeze(1).expand(1, self.group_size, -1, -1, -1)
            k_batch = k_batch.reshape(self.group_size, self.num_kv_groups, seq_len, self.head_dim)
            v_batch = v_batch.reshape(self.group_size, self.num_kv_groups, seq_len, self.head_dim)

            if past_len == 0 and self.sliding_window is None:
                # No prior context and no sliding window: use SDPA's highly
                # optimized causal fast path instead of a custom mask.
                attn_mask = None
                use_causal = True
            else:
                attn_mask = self._build_attn_mask(q_len, seq_len, device=device, dtype=q_batch.dtype)
                use_causal = False

            out_batch = F.scaled_dot_product_attention(
                q_batch,
                k_batch,
                v_batch,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=use_causal,
                scale=self.scale,
            )
            out_batch = out_batch.reshape(1, self.group_size, self.num_kv_groups, q_len, self.head_dim)
            out_batch = out_batch.reshape(1, self.num_heads, q_len, self.head_dim)
            out_batch = out_batch.transpose(1, 2).reshape(1, q_len, self.num_heads * self.head_dim)
            outputs.append(out_batch)

        out = torch.cat(outputs, dim=0)
        return self.out_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        block_tables: Optional[torch.Tensor] = None,
        kv_lens: Optional[torch.Tensor] = None,
        paged_kv_cache: Optional[PagedKVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
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
        if paged_kv_cache is not None:
            if block_tables is None or kv_lens is None:
                raise ValueError("block_tables and kv_lens are required for paged KV attention")
            return self._forward_paged(
                x,
                block_tables=block_tables,
                kv_lens=kv_lens,
                paged_kv_cache=paged_kv_cache,
            ), None

        B, T, _ = x.shape
        device = x.device

        if past_kv is not None:
            self._validate_past_kv(past_kv, batch_size=B)
            # Non-paged KV path assumes every sequence in the batch has the
            # same past length because past_kv is a single dense tensor.
            # Batched decoding with variable-length sequences requires the
            # paged KV path instead.
            if B > 1:
                raise NotImplementedError(
                    "Batched non-paged KV attention (batch_size > 1 with past_kv) "
                    "is not supported. Use paged_kv_cache for batched inference."
                )

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
        # When T==1 during decoding, the single query is always at the last
        # position, so causal masking is implicit and no mask is needed.
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
