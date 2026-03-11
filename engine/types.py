from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import torch

from .config import SamplingConfig


LayerCache = Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]


@dataclass
class RequestState:
    request_id: int
    prompt_token_ids: List[int]
    sampling: SamplingConfig
    max_new_tokens: int
    eos_token_id: Optional[int]
    generated_ids: List[int] = field(default_factory=list)
    all_token_ids: List[int] = field(default_factory=list)
    seen_token_ids: Set[int] = field(default_factory=set)
    past_kv: LayerCache = None
    current_input: Optional[torch.Tensor] = None
    text_chunks: List[str] = field(default_factory=list)
    status: str = "queued"
    stop_reason: Optional[str] = None
    error_message: Optional[str] = None
    created_at_s: float = 0.0
    first_scheduled_at_s: Optional[float] = None
    finished_at_s: Optional[float] = None
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    decode_steps: int = 0
    phase: str = "queued"

    @classmethod
    def from_prompt(
        cls,
        request_id: int,
        prompt_token_ids: List[int],
        sampling: SamplingConfig,
        max_new_tokens: int,
        eos_token_id: Optional[int],
        created_at_s: float,
    ) -> "RequestState":
        return cls(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            created_at_s=created_at_s,
        )

    def should_stop(self) -> bool:
        if len(self.generated_ids) >= self.max_new_tokens:
            return True
        if self.eos_token_id is None:
            return False
        return bool(self.generated_ids and self.generated_ids[-1] == self.eos_token_id)


@dataclass
class GenerationResult:
    request_id: int
    text: str
    token_ids: List[int]
    stop_reason: str
    error_message: Optional[str] = None
    queue_wait_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    total_latency_s: float = 0.0
    model_tokens_per_s: float = 0.0
    prefill_steps: int = 0
    decode_steps: int = 0


@dataclass(frozen=True)
class StreamEvent:
    kind: str
    text: str = ""
    stop_reason: Optional[str] = None
    error_message: Optional[str] = None
