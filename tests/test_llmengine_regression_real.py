import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from engine.config import SamplingConfig
from real_engine_test_utils import build_engine, real_engine_skip_reason


def _sampling():
    return SamplingConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
    )


def _prompt_exceeding_context(engine) -> str:
    context_limit = int(engine.runtime.model.context_length)
    chunk = "context-limit-check "
    prompt = chunk
    encoded = engine.runtime.tokenizer.encode(prompt)
    while len(encoded) <= context_limit:
        prompt += chunk
        encoded = engine.runtime.tokenizer.encode(prompt)
    return prompt


@unittest.skipIf(bool(real_engine_skip_reason()), real_engine_skip_reason())
class LLMEngineRealRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine_batch = build_engine(max_new_tokens=16, max_decode_batch_size=4)
        cls.engine_single = build_engine(max_new_tokens=16, max_decode_batch_size=1)
        cls.prompts = [
            "Give one short sentence about caching in LLM inference.",
            "Explain batching in one concise sentence.",
            "Write one line about prompt templating.",
            "Describe top-k sampling in plain words.",
        ]

    def test_batch_and_single_request_consistency(self):
        batch_results = self.engine_batch.generate_many(self.prompts, sampling=_sampling(), max_new_tokens=12)
        single_results = [
            self.engine_batch.generate_many([prompt], sampling=_sampling(), max_new_tokens=12)[0]
            for prompt in self.prompts
        ]

        for batch_result, single_result in zip(batch_results, single_results):
            self.assertEqual(batch_result.token_ids, single_result.token_ids)
            self.assertEqual(batch_result.text, single_result.text)
            self.assertEqual(batch_result.stop_reason, single_result.stop_reason)
            self.assertIsNone(batch_result.error_message)

    def test_decode_batch_size_invariance(self):
        batch4 = self.engine_batch.generate_many(self.prompts, sampling=_sampling(), max_new_tokens=10)
        batch1 = self.engine_single.generate_many(self.prompts, sampling=_sampling(), max_new_tokens=10)

        for result4, result1 in zip(batch4, batch1):
            self.assertEqual(result4.token_ids, result1.token_ids)
            self.assertEqual(result4.text, result1.text)
            self.assertEqual(result4.stop_reason, result1.stop_reason)
            self.assertIsNone(result4.error_message)
            self.assertIsNone(result1.error_message)

    def test_stream_matches_non_stream(self):
        prompt = "Share one practical testing tip for inference engines."
        non_stream = self.engine_batch.generate_many([prompt], sampling=_sampling(), max_new_tokens=12)[0]
        streamed = "".join(
            chunk
            for chunk in self.engine_batch.generate_stream(
                prompt=prompt,
                sampling=_sampling(),
                max_new_tokens=12,
            )
        )
        self.assertEqual(non_stream.text, streamed)
        self.assertIsNone(non_stream.error_message)

    def test_zero_max_new_tokens(self):
        result = self.engine_batch.generate_many(
            ["Return nothing."],
            sampling=_sampling(),
            max_new_tokens=0,
        )[0]
        self.assertEqual(result.stop_reason, "max_new_tokens")
        self.assertEqual(result.token_ids, [])
        self.assertEqual(result.text, "")
        self.assertIsNone(result.error_message)

    def test_context_limit_stop_reason(self):
        prompt = _prompt_exceeding_context(self.engine_batch)
        result = self.engine_batch.generate_many(
            [prompt],
            sampling=_sampling(),
            max_new_tokens=8,
        )[0]
        self.assertEqual(result.stop_reason, "context_limit")
        self.assertEqual(result.token_ids, [])
        self.assertEqual(result.decode_steps, 0)
        self.assertIsNone(result.error_message)

    def test_result_metrics_sanity(self):
        results = self.engine_batch.generate_many(self.prompts, sampling=_sampling(), max_new_tokens=8)
        for result in results:
            self.assertIsNone(result.error_message)
            self.assertGreaterEqual(result.queue_wait_s, 0.0)
            self.assertGreaterEqual(result.prefill_s, 0.0)
            self.assertGreaterEqual(result.decode_s, 0.0)
            self.assertGreaterEqual(result.total_latency_s, 0.0)
            self.assertEqual(result.decode_steps, len(result.token_ids))
            self.assertGreaterEqual(result.model_tokens_per_s, 0.0)


if __name__ == "__main__":
    unittest.main()
