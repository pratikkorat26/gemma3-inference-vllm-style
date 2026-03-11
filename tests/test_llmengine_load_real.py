import json
import os
import statistics
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from engine.config import SamplingConfig
from real_engine_test_utils import build_engine, real_engine_skip_reason


BASELINE_PATH = PROJECT_ROOT / "tests" / "baselines" / "llmengine_load_thresholds.json"


def _sampling():
    return SamplingConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
    )


def _percentile(values, p):
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values = sorted(values)
    idx = int((len(values) - 1) * p)
    return float(values[idx])


def _load_thresholds():
    with open(BASELINE_PATH, "r", encoding="utf-8") as file_handle:
        thresholds = json.load(file_handle)

    if os.environ.get("LOAD_MAX_ERROR_RATE") is not None:
        thresholds["max_error_rate"] = float(os.environ["LOAD_MAX_ERROR_RATE"])
    if os.environ.get("LOAD_MAX_P95_TOTAL_S") is not None:
        thresholds["max_p95_total_s"] = float(os.environ["LOAD_MAX_P95_TOTAL_S"])
    if os.environ.get("LOAD_MIN_THROUGHPUT_TPS") is not None:
        thresholds["min_throughput_tps"] = float(os.environ["LOAD_MIN_THROUGHPUT_TPS"])
    return thresholds


def _prompt_bank():
    return [
        "Write one concise sentence about KV cache reuse.",
        "Give one sentence on why deterministic sampling helps testing.",
        "Explain round-robin decode fairness in one sentence.",
        "Describe the benefit of batching decode tokens briefly.",
        "Summarize context windows in plain language, one line.",
        "Write one line about stop reasons in generation.",
        "Explain throughput vs latency in one sentence.",
        "Give one sentence on why regression tests matter.",
        "Share one short tip for writing robust APIs.",
        "Describe p95 latency in one sentence for engineers.",
    ]


@unittest.skipIf(bool(real_engine_skip_reason(load_test=True)), real_engine_skip_reason(load_test=True))
class LLMEngineRealLoadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = build_engine(max_new_tokens=24, max_decode_batch_size=4)
        cls.thresholds = _load_thresholds()

    def test_concurrency_sweep_stability(self):
        concurrency_levels = [4 * 2, 8 * 2, 16 * 2]
        bank = _prompt_bank()
        summaries = []

        for level in concurrency_levels:
            prompts = [f"{bank[idx % len(bank)]} [{idx}]" for idx in range(level)]
            started = time.perf_counter()
            results = self.engine.generate_many(
                prompts,
                sampling=_sampling(),
                max_new_tokens=24,
            )
            elapsed_s = time.perf_counter() - started

            errors = [result for result in results if result.error_message is not None]
            total_tokens = sum(len(result.token_ids) for result in results)
            throughpt_tps = 0.0 if elapsed_s <= 0 else total_tokens / elapsed_s
            total_latencies = [result.total_latency_s for result in results]
            p95_total = _percentile(total_latencies, 0.95)
            avg_total = statistics.mean(total_latencies) if total_latencies else 0.0
            error_rate = 0.0 if not results else len(errors) / len(results)

            summaries.append(
                {
                    "concurrency": level,
                    "requests": len(results),
                    "errors": len(errors),
                    "error_rate": round(error_rate, 4),
                    "total_tokens": total_tokens,
                    "elapsed_s": round(elapsed_s, 4),
                    "throughput_tps": round(throughpt_tps, 2),
                    "avg_total_latency_s": round(avg_total, 4),
                    "p95_total_latency_s": round(p95_total, 4),
                }
            )

            self.assertEqual(len(results), level)
            self.assertLessEqual(error_rate, self.thresholds["max_error_rate"])
            self.assertLessEqual(p95_total, self.thresholds["max_p95_total_s"])
            self.assertGreaterEqual(throughpt_tps, self.thresholds["min_throughput_tps"])

        print("llmengine_load_summary", json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
