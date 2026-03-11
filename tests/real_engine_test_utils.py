import os
import sys
import importlib.util
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import EngineConfig, GemmaRuntime, LLMEngine, SamplingConfig, get_device


def should_run_real_engine_tests() -> bool:
    return os.environ.get("RUN_REAL_ENGINE_TESTS", "0") == "1"


def should_run_real_engine_load_tests() -> bool:
    return (
        should_run_real_engine_tests()
        and os.environ.get("RUN_REAL_ENGINE_LOAD_TESTS", "0") == "1"
    )


def has_local_model_assets() -> bool:
    model_dir = PROJECT_ROOT / "gemma-3-270m-it"
    weights_path = model_dir / "model.safetensors"
    tokenizer_path = model_dir / "tokenizer.json"
    return model_dir.exists() and weights_path.exists() and tokenizer_path.exists()


def missing_runtime_dependencies() -> list[str]:
    required = ["torch", "safetensors", "tokenizers"]
    return [name for name in required if importlib.util.find_spec(name) is None]


def real_engine_skip_reason(load_test: bool = False) -> str:
    if not should_run_real_engine_tests():
        return "real-engine tests are disabled (set RUN_REAL_ENGINE_TESTS=1)"
    if load_test and not should_run_real_engine_load_tests():
        return (
            "real-engine load tests are disabled "
            "(set RUN_REAL_ENGINE_TESTS=1 RUN_REAL_ENGINE_LOAD_TESTS=1)"
        )
    if not has_local_model_assets():
        return "missing local model files in gemma-3-270m-it/"
    missing = missing_runtime_dependencies()
    if missing:
        return f"missing runtime dependencies: {', '.join(missing)}"
    return ""


@lru_cache(maxsize=1)
def get_shared_runtime() -> GemmaRuntime:
    return GemmaRuntime(
        choose_model="270m",
        use_instruct_model=True,
        device=get_device(),
    )


def build_engine(
    *,
    max_new_tokens: int = 32,
    max_decode_batch_size: int = 4,
) -> LLMEngine:
    runtime = get_shared_runtime()
    return LLMEngine(
        runtime=runtime,
        config=EngineConfig(
            choose_model="270m",
            use_instruct_model=False,
            max_new_tokens=max_new_tokens,
            max_decode_batch_size=max_decode_batch_size,
            sampling=SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            ),
        ),
    )
