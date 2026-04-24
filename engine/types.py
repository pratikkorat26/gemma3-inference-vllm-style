from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set

import torch

from .config import SamplingConfig


class RequestStatus(str, Enum):
    QUEUED = "queued"
    ACTIVE = "active"
    FINISHED = "finished"
    ERROR = "error"


class RequestPhase(str, Enum):
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    ERROR = "error"


class StopReason(str, Enum):
    EOS = "eos"
    MAX_NEW_TOKENS = "max_new_tokens"
    CONTEXT_LIMIT = "context_limit"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    ERROR = "error"


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
    live_kv_tokens: int = 0
    status: RequestStatus = RequestStatus.QUEUED
    stop_reason: Optional[StopReason] = None
    error_message: Optional[str] = None
    created_at_s: float = 0.0
    first_scheduled_at_s: Optional[float] = None
    finished_at_s: Optional[float] = None
    prefill_time_s: float = 0.0
    prefill_steps: int = 0
    decode_time_s: float = 0.0
    decode_steps: int = 0
    phase: RequestPhase = RequestPhase.QUEUED

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


    def reset_for_generation(self, current_input: Optional[torch.Tensor]) -> None:
        if self.status != RequestStatus.QUEUED:
            raise RuntimeError(f"cannot start request from status={self.status.value}")
        self.status = RequestStatus.ACTIVE
        self.phase = RequestPhase.PREFILL
        self.generated_ids = []
        self.all_token_ids = list(self.prompt_token_ids)
        self.seen_token_ids = set(self.prompt_token_ids)
        self.text_chunks = []
        self.block_table = []
        self.prompt_cursor = 0
        self.live_kv_tokens = 0
        self.stop_reason = None
        self.error_message = None
        self.prefill_time_s = 0.0
        self.prefill_steps = 0
        self.decode_time_s = 0.0
        self.decode_steps = 0
        self.current_input = current_input

    def move_to_prefill(self) -> None:
        if self.status != RequestStatus.ACTIVE:
            raise RuntimeError(f"cannot move request to prefill from status={self.status.value}")
        if self.phase != RequestPhase.PREFILL:
            raise RuntimeError(f"cannot move request to prefill from phase={self.phase.value}")
        self.phase = RequestPhase.PREFILL

    def move_to_decode(self) -> None:
        if self.status != RequestStatus.ACTIVE:
            raise RuntimeError(f"cannot move request to decode from status={self.status.value}")
        if self.phase not in (RequestPhase.PREFILL, RequestPhase.DECODE):
            raise RuntimeError(f"cannot move request to decode from phase={self.phase.value}")
        self.phase = RequestPhase.DECODE

    def finish(self, reason: StopReason, *, error_message: Optional[str] = None, finished_at_s: float) -> None:
        if self.is_done():
            raise RuntimeError(f"request is already done with status={self.status.value}")
        self.status = RequestStatus.ERROR if reason == StopReason.ERROR else RequestStatus.FINISHED
        self.phase = RequestPhase.ERROR if self.status == RequestStatus.ERROR else RequestPhase.FINISHED
        self.stop_reason = reason
        self.error_message = error_message
        self.finished_at_s = finished_at_s

    def is_done(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.ERROR)


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
