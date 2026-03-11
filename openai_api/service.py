import json
import time
import uuid
from typing import Iterator

from engine import EngineConfig, GemmaRuntime, LLMEngine, SamplingConfig, get_device

from .prompting import messages_to_gemma_prompt
from .schemas import ChatCompletionRequest, SUPPORTED_MODEL, Usage


def _now_unix() -> int:
    return int(time.time())


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _finish_reason(stop_reason: str) -> str:
    if stop_reason == "eos":
        return "stop"
    if stop_reason in ("max_new_tokens", "context_limit"):
        return "length"
    return "stop"


class ChatCompletionService:
    def __init__(self) -> None:
        self.model_name = SUPPORTED_MODEL
        self.runtime = GemmaRuntime(
            choose_model="270m",
            use_instruct_model=True,
            device=get_device(),
        )
        self.engine = LLMEngine(
            runtime=self.runtime,
            config=EngineConfig(
                choose_model="270m",
                use_instruct_model=False,
                max_new_tokens=128,
                sampling=SamplingConfig(
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                ),
            ),
        )

    def _usage(self, prompt: str, completion_tokens: int) -> Usage:
        prompt_tokens = len(self.runtime.tokenizer.encode(prompt))
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def create_chat_completion(self, request: ChatCompletionRequest) -> dict:
        prompt = messages_to_gemma_prompt(request.messages)
        sampling = SamplingConfig(
            temperature=float(request.temperature),
            top_p=float(request.top_p),
            top_k=50,
            repetition_penalty=1.1,
        )
        result = self.engine.generate_many(
            prompts=[prompt],
            sampling=sampling,
            max_new_tokens=request.max_tokens,
        )[0]
        if result.error_message is not None:
            raise RuntimeError(result.error_message)

        usage = self._usage(prompt, completion_tokens=len(result.token_ids))
        return {
            "id": _completion_id(),
            "object": "chat.completion",
            "created": _now_unix(),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": _finish_reason(result.stop_reason),
                }
            ],
            "usage": usage.dict(),
        }

    def stream_chat_completion(self, request: ChatCompletionRequest) -> Iterator[str]:
        prompt = messages_to_gemma_prompt(request.messages)
        completion_id = _completion_id()
        created = _now_unix()
        sampling = SamplingConfig(
            temperature=float(request.temperature),
            top_p=float(request.top_p),
            top_k=50,
            repetition_penalty=1.1,
        )

        first = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first)}\n\n"

        final_stop_reason = "eos"
        for event in self.engine.generate_stream_events(
            prompt=prompt,
            sampling=sampling,
            max_new_tokens=request.max_tokens,
        ):
            if event.kind == "text":
                if not event.text:
                    continue
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_name,
                    "choices": [{"index": 0, "delta": {"content": event.text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"
                continue

            final_stop_reason = event.stop_reason or "eos"
            if event.error_message is not None:
                error_payload = {"error": {"message": event.error_message, "type": "server_error"}}
                yield f"data: {json.dumps(error_payload)}\n\n"
                break

        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": _finish_reason(final_stop_reason)}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"
