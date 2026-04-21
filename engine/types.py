from dataclasses import dataclass, field
from typing import List, Optional, Set

import torch

from .config import SamplingConfig


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
    current_input: Optional[torch.Tensor] = None
    text_chunks: List[str] = field(default_factory=list)
    block_table: List[int] = field(default_factory=list)
    prompt_cursor: int = 0
    num_computed_tokens: int = 0
    live_kv_tokens: int = 0
    status: str = "queued"
    stop_reason: Optional[str] = None
    error_message: Optional[str] = None
    created_at_s: float = 0.0
    first_scheduled_at_s: Optional[float] = None
    finished_at_s: Optional[float] = None
    prefill_time_s: float = 0.0
    prefill_steps: int = 0
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
