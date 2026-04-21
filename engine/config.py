from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


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
