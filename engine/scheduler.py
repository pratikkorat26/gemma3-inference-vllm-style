from collections import deque
import time
from typing import Deque, Dict, Iterator, List, Optional

import torch

from .config import EngineConfig, SamplingConfig
from .runtime import GemmaRuntime, apply_chat_template
from .sampling import apply_repetition_penalty_, sample_next_token
from .types import GenerationResult, RequestState, StreamEvent


class LLMEngine:
    """
    Single-device scheduler with explicit prefill/decode phases.

    - Prefill phase: one full-prompt step per request, then move request to decode queue.
    - Decode phase: token-level round-robin over active requests.
    """

    def __init__(self, runtime: GemmaRuntime, config: EngineConfig):
        self.runtime = runtime
        self.config = config

    def _encode_prompt(self, prompt: str) -> List[int]:
        if self.config.use_instruct_model:
            prompt = apply_chat_template(prompt)
        return self.runtime.tokenizer.encode(prompt)

    def _build_request(
        self,
        request_id: int,
        prompt: str,
        sampling: Optional[SamplingConfig],
        max_new_tokens: Optional[int],
        created_at_s: float,
    ) -> RequestState:
        return RequestState.from_prompt(
            request_id=request_id,
            prompt_token_ids=self._encode_prompt(prompt),
            sampling=sampling or self.config.sampling,
            max_new_tokens=(
                max_new_tokens
                if max_new_tokens is not None
                else self.config.max_new_tokens
            ),
            eos_token_id=self.runtime.tokenizer.eos_token_id,
            created_at_s=created_at_s,
        )

    def _context_limit(self) -> int:
        return int(self.runtime.model.context_length)

    def _finish_request(self, request: RequestState, reason: str, error_message: Optional[str] = None) -> None:
        request.status = "finished" if reason != "error" else "error"
        request.phase = request.status
        request.stop_reason = reason
        request.error_message = error_message
        request.finished_at_s = time.perf_counter()

    def _init_request(self, request: RequestState) -> None:
        request.status = "active"
        request.phase = "prefill"
        request.generated_ids = []
        request.all_token_ids = list(request.prompt_token_ids)
        request.seen_token_ids = set(request.prompt_token_ids)
        request.text_chunks = []
        request.past_kv = None
        request.stop_reason = None
        request.error_message = None
        request.prefill_time_s = 0.0
        request.decode_time_s = 0.0
        request.decode_steps = 0

        prompt_ids = torch.tensor(request.prompt_token_ids, device=self.runtime.device).unsqueeze(0)
        request.current_input = prompt_ids

    def _make_result(self, request: RequestState) -> GenerationResult:
        first_scheduled = request.first_scheduled_at_s if request.first_scheduled_at_s is not None else request.created_at_s
        finished = request.finished_at_s if request.finished_at_s is not None else time.perf_counter()
        queue_wait_s = max(0.0, first_scheduled - request.created_at_s)
        total_latency_s = max(0.0, finished - request.created_at_s)
        model_time_s = request.prefill_time_s + request.decode_time_s
        model_tokens_per_s = 0.0 if model_time_s <= 0 else len(request.generated_ids) / model_time_s
        text_ids = list(request.generated_ids)
        if (
            request.stop_reason == "eos"
            and request.eos_token_id is not None
            and text_ids
            and text_ids[-1] == request.eos_token_id
        ):
            text_ids = text_ids[:-1]

        return GenerationResult(
            request_id=request.request_id,
            text=self.runtime.tokenizer.decode(text_ids) if text_ids else "",
            token_ids=list(request.generated_ids),
            stop_reason=request.stop_reason or "error",
            error_message=request.error_message,
            queue_wait_s=queue_wait_s,
            prefill_s=request.prefill_time_s,
            decode_s=request.decode_time_s,
            total_latency_s=total_latency_s,
            model_tokens_per_s=model_tokens_per_s,
            prefill_steps=1 if request.prefill_time_s > 0 else 0,
            decode_steps=request.decode_steps,
        )

    def _decode_step(self, request: RequestState) -> torch.Tensor:
        assert request.current_input is not None
        with torch.inference_mode():
            logits, request.past_kv = self.runtime.model(
                request.current_input,
                past_kv=request.past_kv,
                use_cache=True,
            )
            next_logits = logits[:, -1, :]
            next_logits = apply_repetition_penalty_(
                next_logits,
                [request.seen_token_ids],
                penalty=request.sampling.repetition_penalty,
            )
            return sample_next_token(
                next_logits,
                temperature=request.sampling.temperature,
                top_p=request.sampling.top_p,
                top_k=request.sampling.top_k,
            )

    def _record_next_token(self, request: RequestState, next_token: torch.Tensor, *, emit_text: bool) -> None:
        next_id = int(next_token.item())
        request.generated_ids.append(next_id)

        if request.eos_token_id is not None and next_id == request.eos_token_id:
            self._finish_request(request, reason="eos")
            return

        request.all_token_ids.append(next_id)
        request.seen_token_ids.add(next_id)
        request.current_input = next_token
        if emit_text:
            request.text_chunks.append(self.runtime.tokenizer.decode([next_id]))

        if len(request.generated_ids) >= request.max_new_tokens:
            self._finish_request(request, reason="max_new_tokens")
            return

        request.phase = "decode"

    def _run_one_step(self, request: RequestState, phase: str, *, emit_text: bool = False) -> None:
        if len(request.all_token_ids) >= self._context_limit():
            self._finish_request(request, reason="context_limit")
            return

        started = time.perf_counter()
        next_token = self._decode_step(request)
        elapsed = time.perf_counter() - started

        if phase == "prefill":
            request.prefill_time_s += elapsed
            request.decode_steps += 1
        else:
            request.decode_time_s += elapsed
            request.decode_steps += 1

        self._record_next_token(request, next_token, emit_text=emit_text)

    def _sampling_key(self, request: RequestState):
        return (
            request.sampling.temperature,
            request.sampling.top_p,
            request.sampling.top_k,
            request.sampling.repetition_penalty,
        )

    def _select_decode_batch(self, decode_queue: Deque[RequestState]) -> List[RequestState]:
        first = decode_queue.popleft()
        batch = [first]
        max_batch = max(1, int(self.config.max_decode_batch_size))

        target_len = len(first.all_token_ids)
        target_sampling = self._sampling_key(first)
        remaining = len(decode_queue)
        for _ in range(remaining):
            candidate = decode_queue.popleft()
            candidate_len = len(candidate.all_token_ids)
            candidate_sampling = self._sampling_key(candidate)
            if len(batch) < max_batch and candidate_len == target_len and candidate_sampling == target_sampling:
                batch.append(candidate)
            else:
                decode_queue.append(candidate)
        return batch

    def _run_decode_batch(self, batch: List[RequestState]) -> None:
        eligible: List[RequestState] = []
        for request in batch:
            if request.current_input is None:
                self._finish_request(request, reason="error", error_message="request tensors are not initialized")
                continue
            if len(request.all_token_ids) >= self._context_limit():
                self._finish_request(request, reason="context_limit")
                continue
            eligible.append(request)

        if not eligible:
            return

        current_input = torch.cat([request.current_input for request in eligible], dim=0)

        layer_count = len(eligible[0].past_kv) if eligible[0].past_kv is not None else 0
        if layer_count == 0:
            # Decode batching is used only after prefill populated the cache.
            for request in eligible:
                self._run_one_step(request, phase="decode")
            return

        merged_past = []
        for layer_idx in range(layer_count):
            merged_k = torch.cat([request.past_kv[layer_idx][0] for request in eligible], dim=0)
            merged_v = torch.cat([request.past_kv[layer_idx][1] for request in eligible], dim=0)
            merged_past.append((merged_k, merged_v))

        started = time.perf_counter()
        with torch.inference_mode():
            logits, merged_next = self.runtime.model(current_input, past_kv=merged_past, use_cache=True)
            next_logits = logits[:, -1, :]
            next_logits = apply_repetition_penalty_(
                next_logits,
                [request.seen_token_ids for request in eligible],
                penalty=eligible[0].sampling.repetition_penalty,
            )
            next_tokens = sample_next_token(
                next_logits,
                temperature=eligible[0].sampling.temperature,
                top_p=eligible[0].sampling.top_p,
                top_k=eligible[0].sampling.top_k,
            )
        elapsed = time.perf_counter() - started
        per_request_elapsed = elapsed / len(eligible)

        for batch_idx, request in enumerate(eligible):
            request.past_kv = [
                (
                    layer_k[batch_idx : batch_idx + 1],
                    layer_v[batch_idx : batch_idx + 1],
                )
                for (layer_k, layer_v) in merged_next
            ]
            request.decode_time_s += per_request_elapsed
            request.decode_steps += 1

            next_token = next_tokens[batch_idx : batch_idx + 1]
            self._record_next_token(request, next_token, emit_text=False)

    def generate_many(
        self,
        prompts: List[str],
        *,
        sampling: Optional[SamplingConfig] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[GenerationResult]:
        if not prompts:
            return []

        if self.config.num_prefill_workers != 1 or self.config.num_decode_workers != 1:
            raise ValueError("Single-device mode supports only num_prefill_workers=1 and num_decode_workers=1")

        prefill_queue: Deque[RequestState] = deque()
        decode_queue: Deque[RequestState] = deque()
        results: Dict[int, GenerationResult] = {}
        ordered_ids: List[int] = []

        for request_id, prompt in enumerate(prompts):
            request = self._build_request(
                request_id=request_id,
                prompt=prompt,
                sampling=sampling,
                max_new_tokens=max_new_tokens,
                created_at_s=time.perf_counter(),
            )
            ordered_ids.append(request_id)
            prefill_queue.append(request)

        while prefill_queue or decode_queue:
            if prefill_queue and len(decode_queue) < max(1, int(self.config.max_decode_batch_size)):
                request = prefill_queue.popleft()
                try:
                    if request.first_scheduled_at_s is None:
                        request.first_scheduled_at_s = time.perf_counter()
                        self._init_request(request)

                    if request.max_new_tokens <= 0:
                        self._finish_request(request, reason="max_new_tokens")
                    elif len(request.prompt_token_ids) >= self._context_limit():
                        self._finish_request(request, reason="context_limit")
                    else:
                        request.phase = "prefill"
                        self._run_one_step(request, phase="prefill")
                except Exception as exc:
                    self._finish_request(request, reason="error", error_message=str(exc))

                if request.status in ("finished", "error"):
                    results[request.request_id] = self._make_result(request)
                else:
                    decode_queue.append(request)
                continue

            if decode_queue:
                batch = self._select_decode_batch(decode_queue)
                try:
                    self._run_decode_batch(batch)
                except Exception as exc:
                    for request in batch:
                        if request.status in ("finished", "error"):
                            continue
                        try:
                            request.phase = "decode"
                            self._run_one_step(request, phase="decode")
                        except Exception as single_exc:
                            self._finish_request(request, reason="error", error_message=str(single_exc))

                for request in batch:
                    if request.status in ("finished", "error"):
                        results[request.request_id] = self._make_result(request)
                    else:
                        decode_queue.append(request)

        return [results[request_id] for request_id in ordered_ids]

    def generate_stream_events(
        self,
        prompt: str,
        *,
        sampling: Optional[SamplingConfig] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[StreamEvent]:
        request = self._build_request(
            request_id=0,
            prompt=prompt,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            created_at_s=time.perf_counter(),
        )
        request.first_scheduled_at_s = time.perf_counter()
        self._init_request(request)

        if request.max_new_tokens <= 0:
            self._finish_request(request, reason="max_new_tokens")
            yield StreamEvent(kind="done", stop_reason=request.stop_reason)
            return

        if len(request.prompt_token_ids) >= self._context_limit():
            self._finish_request(request, reason="context_limit")
            yield StreamEvent(kind="done", stop_reason=request.stop_reason)
            return

        try:
            self._run_one_step(request, phase="prefill", emit_text=True)
            if request.text_chunks:
                yield StreamEvent(kind="text", text=request.text_chunks[-1])

            while request.status not in ("finished", "error"):
                previous_chunks = len(request.text_chunks)
                self._run_one_step(request, phase="decode", emit_text=True)
                if len(request.text_chunks) > previous_chunks:
                    yield StreamEvent(kind="text", text=request.text_chunks[-1])
        except Exception as exc:
            self._finish_request(request, reason="error", error_message=str(exc))

        yield StreamEvent(
            kind="done",
            stop_reason=request.stop_reason,
            error_message=request.error_message,
        )

    def generate_stream(
        self,
        prompt: str,
        *,
        sampling: Optional[SamplingConfig] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        for event in self.generate_stream_events(
            prompt=prompt,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
        ):
            if event.kind == "text" and event.text:
                yield event.text

    def generate(
        self,
        prompt: str,
        *,
        sampling: Optional[SamplingConfig] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        result = self.generate_many(
            [prompt],
            sampling=sampling,
            max_new_tokens=max_new_tokens,
        )[0]
        if result.error_message is not None:
            raise RuntimeError(result.error_message)
        return result.text
