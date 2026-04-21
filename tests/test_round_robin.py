import unittest
from typing import Optional

import torch
import sys
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.config import EngineConfig, SamplingConfig
from engine.scheduler import LLMEngine
from engine.types import RequestState
from gemma3.paged_kv import PagedKVCache


class FakeTokenizer:
    eos_token_id = 999

    def encode(self, text):
        return [int(text)]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{token_id}>" for token_id in ids)


class FakeTokenizerSequence:
    eos_token_id = 999

    def encode(self, text):
        return [int(part) for part in text.split()]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{token_id}>" for token_id in ids)


class FakePagedModel:
    def __init__(self, fail_on_token=None, *, strict_cache_check=True):
        self.context_length = 1024
        self.fail_on_token = fail_on_token
        self.strict_cache_check = strict_cache_check
        self.call_last_tokens = []
        self.call_seq_lens = []

    def init_paged_kv_caches(self, *, num_blocks, block_size, device):
        return [
            PagedKVCache.empty(
                num_blocks=num_blocks,
                num_kv_groups=1,
                block_size=block_size,
                head_dim=1,
                device=device,
                dtype=torch.float32,
            )
        ]

    def __call__(
        self,
        input_ids,
        *,
        block_tables=None,
        kv_lens=None,
        paged_kv_caches=None,
        past_kv=None,
        use_cache=True,
    ):
        if block_tables is None or kv_lens is None or paged_kv_caches is None:
            raise RuntimeError("paged KV inputs are required")

        cache = paged_kv_caches[0]
        batch_size, seq_len = input_ids.shape
        last_tokens = [int(input_ids[idx, -1].item()) for idx in range(batch_size)]
        self.call_last_tokens.extend(last_tokens)
        self.call_seq_lens.extend([seq_len] * batch_size)
        for last_token in last_tokens:
            if self.fail_on_token is not None and last_token == self.fail_on_token:
                raise RuntimeError(f"forced failure on token {last_token}")

        vocab_size = 4096
        logits = torch.full((batch_size, seq_len, vocab_size), -1e9)
        for batch_idx in range(batch_size):
            current_token = int(input_ids[batch_idx, -1].item())
            kv_len = int(kv_lens[batch_idx].item())
            if self.strict_cache_check and kv_len > 0:
                prev_pos = kv_len - 1
                prev_block = int(block_tables[batch_idx, prev_pos // cache.block_size].item())
                prev_offset = prev_pos % cache.block_size
                prev_token = int(cache.k_blocks[prev_block, 0, prev_offset, 0].item())
                expected_prev_token = current_token - 10
                if prev_token != expected_prev_token:
                    raise RuntimeError(
                        f"KV cache leakage detected: cache={prev_token}, expected={expected_prev_token}"
                    )

            for token_offset in range(seq_len):
                token = int(input_ids[batch_idx, token_offset].item())
                pos = kv_len + token_offset
                block_id = int(block_tables[batch_idx, pos // cache.block_size].item())
                if block_id < 0:
                    raise RuntimeError("missing paged KV block assignment")
                block_offset = pos % cache.block_size
                cache.k_blocks[block_id, 0, block_offset, 0] = float(token)
                cache.v_blocks[block_id, 0, block_offset, 0] = float(token)

            next_token = current_token + 10
            logits[batch_idx, -1, next_token] = 0.0
        return logits


class FakeModel:
    def __init__(self, fail_on_token=None):
        self.context_length = 1024
        self.fail_on_token = fail_on_token
        self.call_last_tokens = []
        self.call_input_lengths = []
        self.call_seq_lens = []

    def init_paged_kv_caches(self, *, num_blocks, block_size, device):
        return []

    def __call__(
        self,
        input_ids,
        *,
        block_tables=None,
        kv_lens=None,
        paged_kv_caches=None,
        past_kv=None,
        use_cache=True,
    ):
        batch_size, seq_len = input_ids.shape
        self.call_input_lengths.append(int(seq_len))
        last_tokens = [int(input_ids[idx, -1].item()) for idx in range(batch_size)]
        self.call_last_tokens.extend(last_tokens)
        self.call_seq_lens.extend([seq_len] * batch_size)
        for last_token in last_tokens:
            if self.fail_on_token is not None and last_token == self.fail_on_token:
                raise RuntimeError(f"forced failure on token {last_token}")

        vocab_size = 4096
        logits = torch.full((batch_size, seq_len, vocab_size), -1e9)
        for batch_idx in range(batch_size):
            current_token = int(input_ids[batch_idx, -1].item())
            next_token = current_token + 10
            logits[batch_idx, -1, next_token] = 0.0
        return logits


class FakeRuntime:
    def __init__(self, fail_on_token=None, *, strict_cache_check=True):
        self.device = torch.device("cpu")
        self.tokenizer = FakeTokenizer()
        self.model = FakePagedModel(
            fail_on_token=fail_on_token,
            strict_cache_check=strict_cache_check,
        )


class FakeRuntimeSequence:
    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = FakeTokenizerSequence()
        self.model = FakeModel()


class RoundRobinSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.config = EngineConfig(
            choose_model="270m",
            use_instruct_model=False,
            max_new_tokens=2,
            num_prefill_workers=1,
            num_decode_workers=1,
            kv_block_size=1,
            sampling=SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            ),
        )

    def _make_decode_request(
        self,
        request_id: int,
        *,
        token_length: int,
        sampling: Optional[SamplingConfig] = None,
    ) -> RequestState:
        request = RequestState.from_prompt(
            request_id=request_id,
            prompt_token_ids=[request_id],
            sampling=sampling or self.config.sampling,
            max_new_tokens=2,
            eos_token_id=FakeTokenizer.eos_token_id,
            created_at_s=float(request_id),
        )
        request.all_token_ids = [request_id] * token_length
        return request

    def _make_prompt_request(
        self,
        request_id: int,
        *,
        prompt_token_ids: list[int],
        max_new_tokens: int = 2,
        sampling: Optional[SamplingConfig] = None,
    ) -> RequestState:
        return RequestState.from_prompt(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling=sampling or self.config.sampling,
            max_new_tokens=max_new_tokens,
            eos_token_id=FakeTokenizer.eos_token_id,
            created_at_s=float(request_id),
        )

    def test_round_robin_token_order(self):
        runtime = FakeRuntime(strict_cache_check=False)
        engine = LLMEngine(runtime=runtime, config=self.config)

        results = engine.generate_many(["1", "2", "3"])

        self.assertEqual(len(runtime.model.call_last_tokens), 6)
        self.assertEqual(sorted(runtime.model.call_last_tokens), [1, 2, 3, 11, 12, 13])
        self.assertEqual([result.token_ids for result in results], [[11, 21], [12, 22], [13, 23]])
        self.assertEqual([result.stop_reason for result in results], ["max_new_tokens"] * 3)
        self.assertEqual([result.prefill_steps for result in results], [1, 1, 1])

    def test_error_isolation(self):
        runtime = FakeRuntime(fail_on_token=2)
        engine = LLMEngine(runtime=runtime, config=self.config)

        results = engine.generate_many(["1", "2", "3"])

        self.assertEqual(results[1].stop_reason, "error")
        self.assertIn("forced failure", results[1].error_message or "")
        self.assertEqual(results[0].stop_reason, "max_new_tokens")
        self.assertEqual(results[2].stop_reason, "max_new_tokens")

    def test_kv_cache_isolation_between_requests(self):
        runtime = FakeRuntime(strict_cache_check=False)
        engine = LLMEngine(runtime=runtime, config=self.config)

        results = engine.generate_many(["100", "200", "300"])

        for result in results:
            self.assertEqual(result.stop_reason, "max_new_tokens")
            self.assertIsNone(result.error_message)

    def test_zero_max_new_tokens_respected(self):
        runtime = FakeRuntime()
        engine = LLMEngine(runtime=runtime, config=self.config)

        result = engine.generate_many(["1"], max_new_tokens=0)[0]

        self.assertEqual(result.stop_reason, "max_new_tokens")
        self.assertEqual(result.token_ids, [])
        self.assertEqual(result.text, "")

    def test_capacity_exceeded_rejects_request(self):
        runtime = FakeRuntime()
        config = EngineConfig(
            choose_model="270m",
            use_instruct_model=False,
            max_new_tokens=2,
            max_kv_cache_tokens=1,
            kv_block_size=1,
            num_kv_blocks=1,
            num_prefill_workers=1,
            num_decode_workers=1,
            sampling=SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            ),
        )
        engine = LLMEngine(runtime=runtime, config=config)

        result = engine.generate_many(["1"], max_new_tokens=2)[0]

        self.assertEqual(result.stop_reason, "capacity_exceeded")
        self.assertIn("KV cache capacity exceeded", result.error_message or "")
        self.assertEqual(result.token_ids, [11])

    def test_capacity_released_after_finished_request(self):
        runtime = FakeRuntime()
        config = EngineConfig(
            choose_model="270m",
            use_instruct_model=False,
            max_new_tokens=2,
            max_kv_cache_tokens=3,
            kv_block_size=1,
            num_prefill_workers=1,
            num_decode_workers=1,
            max_decode_batch_size=1,
            sampling=SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            ),
        )
        engine = LLMEngine(runtime=runtime, config=config)

        first = engine.generate_many(["1"], max_new_tokens=2)[0]
        second = engine.generate_many(["2"], max_new_tokens=2)[0]

        self.assertEqual(first.stop_reason, "max_new_tokens")
        self.assertEqual(second.stop_reason, "max_new_tokens")
        self.assertEqual(engine.capacity.reserved_tokens, 0)

    def test_capacity_defers_later_request_until_budget_frees(self):
        runtime = FakeRuntime()
        config = EngineConfig(
            choose_model="270m",
            use_instruct_model=False,
            max_new_tokens=2,
            max_kv_cache_tokens=3,
            kv_block_size=1,
            num_prefill_workers=1,
            num_decode_workers=1,
            max_decode_batch_size=4,
            sampling=SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            ),
        )
        engine = LLMEngine(runtime=runtime, config=config)

        results = engine.generate_many(["1", "2"], max_new_tokens=2)

        self.assertEqual([result.stop_reason for result in results], ["max_new_tokens", "max_new_tokens"])
        self.assertEqual([result.token_ids for result in results], [[11, 21], [12, 22]])
        self.assertEqual(engine.capacity.reserved_tokens, 0)

    def test_chunked_prefill_splits_prompt_into_multiple_steps(self):
        runtime = FakeRuntime(strict_cache_check=False)
        config = EngineConfig(
            choose_model="270m",
            use_instruct_model=False,
            max_new_tokens=2,
            kv_block_size=1,
            prefill_chunk_size=2,
            sampling=SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            ),
        )
        engine = LLMEngine(runtime=runtime, config=config)

        request = self._make_prompt_request(request_id=0, prompt_token_ids=[1, 2, 3, 4])
        engine._init_request(request)
        final_logits = None
        while not engine._prefill_complete(request):
            final_logits = engine._run_prefill_chunk(request)

        self.assertIsNotNone(final_logits)
        next_token = engine._sample_next_tokens(final_logits, [request])
        engine._record_next_token(request, next_token, emit_text=False)

        self.assertEqual(request.prefill_steps, 2)
        self.assertEqual(request.generated_ids, [14])
        self.assertEqual(runtime.model.call_seq_lens[:2], [2, 2])

    def test_chunked_prefill_allows_decode_between_prompt_chunks(self):
        runtime = FakeRuntime(strict_cache_check=False)
        config = EngineConfig(
            choose_model="270m",
            use_instruct_model=False,
            max_new_tokens=2,
            kv_block_size=1,
            prefill_chunk_size=1,
            sampling=SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            ),
        )
        engine = LLMEngine(runtime=runtime, config=config)
        long_request = self._make_prompt_request(request_id=0, prompt_token_ids=[1, 2, 3], max_new_tokens=2)
        short_request = self._make_prompt_request(request_id=1, prompt_token_ids=[7], max_new_tokens=2)

        long_request.first_scheduled_at_s = 0.0
        short_request.first_scheduled_at_s = 0.0
        engine._init_request(long_request)
        engine._init_request(short_request)

        logits = engine._run_prefill_chunk(long_request)
        self.assertIsNone(logits)

        final_short_logits = engine._run_prefill_chunk(short_request)
        self.assertIsNotNone(final_short_logits)
        short_next = engine._sample_next_tokens(final_short_logits, [short_request])
        engine._record_next_token(short_request, short_next, emit_text=False)

        second_long_logits = engine._run_prefill_chunk(long_request)
        self.assertIsNone(second_long_logits)

        engine._run_one_step(short_request, phase="decode")

        final_long_logits = engine._run_prefill_chunk(long_request)
        self.assertIsNotNone(final_long_logits)

        self.assertEqual(long_request.prefill_steps, 3)
        self.assertEqual(short_request.prefill_steps, 1)
        self.assertEqual(runtime.model.call_last_tokens[:5], [1, 7, 2, 17, 3])

    def test_select_decode_batch_prefers_fuller_compatible_cohort(self):
        runtime = FakeRuntime()
        config = EngineConfig(
            choose_model="270m",
            use_instruct_model=False,
            kv_block_size=1,
            max_decode_batch_size=2,
            decode_selection_window=4,
        )
        engine = LLMEngine(runtime=runtime, config=config)
        decode_queue = deque(
            [
                self._make_decode_request(1, token_length=5),
                self._make_decode_request(2, token_length=7),
                self._make_decode_request(3, token_length=7),
                self._make_decode_request(4, token_length=9),
            ]
        )

        batch = engine._select_decode_batch(decode_queue)

        self.assertEqual([request.request_id for request in batch], [2, 3])
        self.assertEqual([request.request_id for request in decode_queue], [1, 4])

    def test_select_decode_batch_tie_breaks_to_older_group(self):
        runtime = FakeRuntime()
        config = EngineConfig(
            choose_model="270m",
            use_instruct_model=False,
            kv_block_size=1,
            max_decode_batch_size=2,
            decode_selection_window=4,
        )
        engine = LLMEngine(runtime=runtime, config=config)
        decode_queue = deque(
            [
                self._make_decode_request(1, token_length=5),
                self._make_decode_request(2, token_length=5),
                self._make_decode_request(3, token_length=7),
                self._make_decode_request(4, token_length=7),
            ]
        )

        batch = engine._select_decode_batch(decode_queue)

        self.assertEqual([request.request_id for request in batch], [1, 2])
        self.assertEqual([request.request_id for request in decode_queue], [3, 4])

    def test_select_decode_batch_respects_window_limit(self):
        runtime = FakeRuntime()
        config = EngineConfig(
            choose_model="270m",
            use_instruct_model=False,
            kv_block_size=1,
            max_decode_batch_size=2,
            decode_selection_window=4,
        )
        engine = LLMEngine(runtime=runtime, config=config)
        decode_queue = deque(
            [
                self._make_decode_request(1, token_length=5),
                self._make_decode_request(2, token_length=7),
                self._make_decode_request(3, token_length=7),
                self._make_decode_request(4, token_length=9),
                self._make_decode_request(5, token_length=9),
                self._make_decode_request(6, token_length=9),
            ]
        )

        batch = engine._select_decode_batch(decode_queue)

        self.assertEqual([request.request_id for request in batch], [2, 3])
        self.assertEqual([request.request_id for request in decode_queue], [1, 4, 5, 6])

if __name__ == "__main__":
    unittest.main()
