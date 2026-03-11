from typing import Iterable, Optional, Sequence

import torch


def apply_repetition_penalty_(
    logits: torch.Tensor,
    all_token_ids: torch.Tensor | Sequence[Iterable[int]],
    penalty: Optional[float],
) -> torch.Tensor:
    if penalty is None or penalty <= 1.0:
        return logits

    if isinstance(all_token_ids, torch.Tensor):
        token_rows = [torch.unique(all_token_ids[batch_idx]).tolist() for batch_idx in range(logits.size(0))]
    else:
        token_rows = [list(token_ids) for token_ids in all_token_ids]

    for batch_idx, used_ids in enumerate(token_rows):
        if not used_ids:
            continue
        used_ids_tensor = torch.tensor(used_ids, device=logits.device, dtype=torch.long)
        used_logits = logits[batch_idx, used_ids_tensor]
        logits[batch_idx, used_ids_tensor] = torch.where(
            used_logits < 0,
            used_logits * penalty,
            used_logits / penalty,
        )
    return logits


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cdf = torch.cumsum(sorted_probs, dim=-1)

        to_remove = cdf > top_p
        to_remove[..., 1:] = to_remove[..., :-1].clone()
        to_remove[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
