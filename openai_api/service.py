import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Iterator, Optional

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


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ServiceOverloadedError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatServiceConfig:
    choose_model: str = "270m"
    use_instruct_model: bool = True
    max_new_tokens: int = 128
    top_k: int = 50
    repetition_penalty: float = 1.1
    default_temperature: float = 0.8
    default_top_p: float = 0.9
    max_active_requests: int = 1
    max_queued_requests: int = 8

    def __post_init__(self) -> None:
        if self.max_active_requests <= 0:
            raise ValueError("max_active_requests must be > 0")
        if self.max_queued_requests < 0:
            raise ValueError("max_queued_requests must be >= 0")

    @classmethod
    def from_env(cls) -> "ChatServiceConfig":
        return cls(
            choose_model=os.environ.get("GEMMA_MODEL_SIZE", cls.choose_model),
            use_instruct_model=_env_bool("GEMMA_USE_INSTRUCT_MODEL", cls.use_instruct_model),
            max_new_tokens=_env_int("GEMMA_MAX_NEW_TOKENS", cls.max_new_tokens),
            top_k=_env_int("GEMMA_TOP_K", cls.top_k),
            repetition_penalty=_env_float("GEMMA_REPETITION_PENALTY", cls.repetition_penalty),
            default_temperature=_env_float("GEMMA_TEMPERATURE", cls.default_temperature),
            default_top_p=_env_float("GEMMA_TOP_P", cls.default_top_p),
            max_active_requests=_env_int("GEMMA_MAX_ACTIVE_REQUESTS", cls.max_active_requests),
            max_queued_requests=_env_int("GEMMA_MAX_QUEUED_REQUESTS", cls.max_queued_requests),
        )


class _AdmissionTicket:
    def __init__(self, service: "ChatCompletionService") -> None:
        self._service = service
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self, *, completed: bool = False, cancelled: bool = False, error: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        self._service._release_admission(completed=completed, cancelled=cancelled, error=error)


class ChatCompletionService:
    def __init__(self, config: Optional[ChatServiceConfig] = None) -> None:
        self.config = config or ChatServiceConfig.from_env()
        self.model_name = SUPPORTED_MODEL
        self._engine_lock = threading.RLock()
        self._engine_slots = threading.BoundedSemaphore(self.config.max_active_requests)
        self._admission_lock = threading.Lock()
        self._active_requests = 0
        self._queued_requests = 0
        self._started_requests = 0
        self._completed_requests = 0
        self._rejected_requests = 0
        self._cancelled_requests = 0
        self._error_requests = 0
        self.runtime = GemmaRuntime(
            choose_model=self.config.choose_model,
            use_instruct_model=self.config.use_instruct_model,
            device=get_device(),
        )
        self.engine = LLMEngine(
            runtime=self.runtime,
            config=EngineConfig(
                choose_model=self.config.choose_model,
                use_instruct_model=False,
                max_new_tokens=self.config.max_new_tokens,
                sampling=SamplingConfig(
                    temperature=self.config.default_temperature,
                    top_p=self.config.default_top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                ),
            ),
        )

    def _acquire_admission(self) -> _AdmissionTicket:
        if not self._engine_slots.acquire(blocking=False):
            with self._admission_lock:
                if self._queued_requests >= self.config.max_queued_requests:
                    self._rejected_requests += 1
                    raise ServiceOverloadedError("service overloaded: request queue is full")
                self._queued_requests += 1
            try:
                self._engine_slots.acquire()
            finally:
                with self._admission_lock:
                    self._queued_requests = max(0, self._queued_requests - 1)

        with self._admission_lock:
            self._active_requests += 1
            self._started_requests += 1
        return _AdmissionTicket(self)

    def _release_admission(self, *, completed: bool, cancelled: bool, error: bool) -> None:
        with self._admission_lock:
            self._active_requests = max(0, self._active_requests - 1)
            if completed:
                self._completed_requests += 1
            if cancelled:
                self._cancelled_requests += 1
            if error:
                self._error_requests += 1
        self._engine_slots.release()

    def metrics_snapshot(self) -> dict:
        with self._admission_lock:
            snapshot = {
                "active_requests": self._active_requests,
                "queued_requests": self._queued_requests,
                "started_requests": self._started_requests,
                "completed_requests": self._completed_requests,
                "rejected_requests": self._rejected_requests,
                "cancelled_requests": self._cancelled_requests,
                "error_requests": self._error_requests,
                "max_active_requests": self.config.max_active_requests,
                "max_queued_requests": self.config.max_queued_requests,
            }
        capacity = getattr(self.engine, "capacity", None)
        if capacity is not None:
            snapshot["kv_blocks_total"] = capacity.num_blocks
            snapshot["kv_blocks_used"] = capacity._allocated_blocks
            snapshot["kv_blocks_free"] = len(capacity._free_block_ids)
            snapshot["kv_tokens_reserved"] = capacity.reserved_tokens
        return snapshot

    def _usage(self, prompt: str, completion_tokens: int) -> Usage:
        prompt_tokens = len(self.runtime.tokenizer.encode(prompt))
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _trace_event(self, trace_events: Optional[list[dict]], name: str, **data) -> None:
        if trace_events is None:
            return
        event = {
            "name": name,
            "ts_unix_s": time.time(),
        }
        if data:
            event["data"] = data
        trace_events.append(event)

    def _create_chat_completion_impl(
        self,
        request: ChatCompletionRequest,
        *,
        trace_id: Optional[str] = None,
        trace_events: Optional[list[dict]] = None,
    ) -> dict:
        self._trace_event(
            trace_events,
            "service.request_received",
            trace_id=trace_id,
            stream=bool(request.stream),
        )
        prompt = messages_to_gemma_prompt(request.messages)
        self._trace_event(
            trace_events,
            "service.prompt_built",
            prompt_chars=len(prompt),
            message_count=len(request.messages),
        )
        temperature = request.temperature if request.temperature is not None else self.config.default_temperature
        top_p = request.top_p if request.top_p is not None else self.config.default_top_p
        sampling = SamplingConfig(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
        )
        self._trace_event(
            trace_events,
            "service.engine_start",
            max_tokens=request.max_tokens,
            temperature=float(temperature),
            top_p=float(top_p),
        )
        ticket = self._acquire_admission()
        try:
            with self._engine_lock:
                result = self.engine.generate_many(
                    prompts=[prompt],
                    sampling=sampling,
                    max_new_tokens=request.max_tokens,
                )[0]
            self._trace_event(
                trace_events,
                "service.engine_done",
                stop_reason=result.stop_reason,
                completion_tokens=len(result.token_ids),
            )
            if result.error_message is not None:
                self._trace_event(
                    trace_events,
                    "service.error",
                    error_message=result.error_message,
                )
                ticket.close(error=True)
                raise RuntimeError(result.error_message)

            usage = self._usage(prompt, completion_tokens=len(result.token_ids))
            response = {
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
                "usage": usage.model_dump(),
            }
            self._trace_event(
                trace_events,
                "service.response_built",
                response_finish_reason=response["choices"][0]["finish_reason"],
            )
            ticket.close(completed=True)
            return response
        except BaseException:
            if not ticket.closed:
                ticket.close(error=True)
            raise

    def create_chat_completion(self, request: ChatCompletionRequest) -> dict:
        return self._create_chat_completion_impl(request)

    def create_chat_completion_traced(self, request: ChatCompletionRequest, trace_id: str) -> tuple[dict, dict]:
        trace_events: list[dict] = []
        response = self._create_chat_completion_impl(
            request,
            trace_id=trace_id,
            trace_events=trace_events,
        )
        return response, {
            "trace_id": trace_id,
            "component": "server",
            "events": trace_events,
        }

    def stream_chat_completion(self, request: ChatCompletionRequest) -> Iterator[str]:
        prompt = messages_to_gemma_prompt(request.messages)
        completion_id = _completion_id()
        created = _now_unix()
        temperature = request.temperature if request.temperature is not None else self.config.default_temperature
        top_p = request.top_p if request.top_p is not None else self.config.default_top_p
        sampling = SamplingConfig(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
        )
        ticket = self._acquire_admission()

        return self._stream_chat_completion_admitted(
            prompt=prompt,
            request=request,
            sampling=sampling,
            completion_id=completion_id,
            created=created,
            ticket=ticket,
        )

    def _stream_chat_completion_admitted(
        self,
        *,
        prompt: str,
        request: ChatCompletionRequest,
        sampling: SamplingConfig,
        completion_id: str,
        created: int,
        ticket: _AdmissionTicket,
    ) -> Iterator[str]:
        first = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        completed = False
        errored = False
        try:
            yield f"data: {json.dumps(first)}\n\n"

            final_stop_reason = "eos"
            with self._engine_lock:
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
                        errored = True
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
            completed = not errored
        except GeneratorExit:
            ticket.close(cancelled=True)
            raise
        except BaseException:
            ticket.close(error=True)
            raise
        finally:
            if not ticket.closed:
                ticket.close(completed=completed, error=errored)
