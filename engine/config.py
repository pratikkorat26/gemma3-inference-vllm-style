import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be a finite value >= 0")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be a finite value in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be a finite value > 0")


@dataclass(frozen=True)
class EngineConfig:
    choose_model: str = "270m"
    use_instruct_model: bool = True
    max_new_tokens: int = 180
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    # Multi-worker support is reserved for future multi-device backends.
    # The current single-device scheduler requires both to be 1.
    num_prefill_workers: int = 1
    num_decode_workers: int = 1
    max_decode_batch_size: int = 4
    decode_selection_window: int = 8
    max_kv_cache_tokens: int = 32_768
    kv_block_size: int = 16
    num_kv_blocks: Optional[int] = None
    prefill_chunk_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if self.num_prefill_workers != 1 or self.num_decode_workers != 1:
            raise ValueError("single-device mode requires num_prefill_workers=1 and num_decode_workers=1")
        if self.max_decode_batch_size <= 0:
            raise ValueError("max_decode_batch_size must be > 0")
        if self.decode_selection_window <= 0:
            raise ValueError("decode_selection_window must be > 0")
        if self.max_kv_cache_tokens <= 0:
            raise ValueError("max_kv_cache_tokens must be > 0")
        if self.kv_block_size <= 0:
            raise ValueError("kv_block_size must be > 0")
        if self.num_kv_blocks is not None:
            if self.num_kv_blocks <= 0:
                raise ValueError("num_kv_blocks must be > 0")
            if self.num_kv_blocks * self.kv_block_size > self.max_kv_cache_tokens:
                raise ValueError("num_kv_blocks * kv_block_size cannot exceed max_kv_cache_tokens")
        if self.prefill_chunk_size is not None and self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be > 0 when set")
