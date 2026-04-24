import unittest
import threading

try:
    from fastapi.testclient import TestClient
    from openai_api.app import create_app
    from openai_api.prompting import messages_to_gemma_prompt
    from openai_api.schemas import ChatCompletionRequest, SUPPORTED_MODEL
    from openai_api.service import ChatCompletionService, ChatServiceConfig, ServiceOverloadedError
    from engine.types import GenerationResult, StreamEvent
    FASTAPI_AVAILABLE = True
except ModuleNotFoundError:
    FASTAPI_AVAILABLE = False


class FakeChatService:
    def create_chat_completion(self, request):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": SUPPORTED_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }

    def stream_chat_completion(self, request):
        yield 'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        yield 'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
        yield 'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    def create_chat_completion_traced(self, request, trace_id):
        response = self.create_chat_completion(request)
        return response, {
            "trace_id": trace_id,
            "component": "server",
            "events": [{"name": "service.request_received", "ts_unix_s": 0.0}],
        }

    def metrics_snapshot(self):
        return {"active_requests": 0, "queued_requests": 0}


class OverloadedChatService(FakeChatService):
    def create_chat_completion(self, request):
        raise ServiceOverloadedError("service overloaded: request queue is full")

    def stream_chat_completion(self, request):
        raise ServiceOverloadedError("service overloaded: request queue is full")


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class OpenAIChatAPITests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(service_factory=FakeChatService)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_non_stream_response_shape(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["role"], "assistant")
        self.assertIn("usage", payload)

    def test_stream_response_has_done(self):
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        ) as response:
            self.assertEqual(response.status_code, 200)
            body = "".join(part for part in response.iter_text())
            self.assertIn("data: [DONE]", body)

    def test_rejects_unknown_model(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_readyz_reports_ready_after_startup(self):
        response = self.client.get("/readyz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ready"})

    def test_metrics_returns_service_snapshot(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["active_requests"], 0)

    def test_non_stream_trace_header_is_returned_when_trace_id_is_present(self):
        response = self.client.post(
            "/v1/chat/completions",
            headers={"X-Trace-Id": "trace_test"},
            json={
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("X-Trace-Data", response.headers)


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class OpenAIChatAPIStartupFailureTests(unittest.TestCase):
    def setUp(self):
        def failing_service_factory():
            raise RuntimeError("model init failed")

        self.app = create_app(service_factory=failing_service_factory)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_readyz_reports_startup_failure(self):
        response = self.client.get("/readyz")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "service unavailable: model init failed")

    def test_chat_completion_fails_fast_when_service_unavailable(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "service unavailable: model init failed")


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class OpenAIChatAPIOverloadTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(service_factory=OverloadedChatService)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_non_stream_overload_returns_429(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(response.status_code, 429)
        self.assertIn("service overloaded", response.json()["detail"])

    def test_stream_overload_returns_429(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        self.assertEqual(response.status_code, 429)
        self.assertIn("service overloaded", response.json()["detail"])


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class OpenAIChatInternalsTests(unittest.TestCase):
    def _make_service(self, text: str) -> ChatCompletionService:
        class FakeTokenizer:
            def encode(self, prompt):
                return list(range(len(prompt)))

        class FakeRuntime:
            tokenizer = FakeTokenizer()

        class FakeEngine:
            def generate_many(self, prompts, sampling, max_new_tokens):
                return [
                    GenerationResult(
                        request_id=0,
                        text=text,
                        token_ids=[1, 2, 3],
                        stop_reason="eos",
                    )
                ]

        service = ChatCompletionService.__new__(ChatCompletionService)
        service.config = ChatServiceConfig()
        service.model_name = SUPPORTED_MODEL
        service._engine_lock = threading.RLock()
        service._engine_slots = threading.BoundedSemaphore(service.config.max_active_requests)
        service._admission_lock = threading.Lock()
        service._active_requests = 0
        service._queued_requests = 0
        service._started_requests = 0
        service._completed_requests = 0
        service._rejected_requests = 0
        service._cancelled_requests = 0
        service._error_requests = 0
        service.runtime = FakeRuntime()
        service.engine = FakeEngine()
        return service

    def test_prompt_renders_messages(self):
        request = ChatCompletionRequest.model_validate(
            {
                "model": SUPPORTED_MODEL,
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hello"},
                ],
            }
        )
        prompt = messages_to_gemma_prompt(request.messages)
        self.assertIn("Be concise.", prompt)
        self.assertIn("Hello", prompt)

    def test_service_returns_plain_text_response(self):
        request = ChatCompletionRequest.model_validate(
            {
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )
        service = self._make_service("plain text answer")
        response = service.create_chat_completion(request)
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["choices"][0]["message"]["content"], "plain text answer")

    def test_stream_close_counts_as_cancellation_and_releases_slot(self):
        class FakeTokenizer:
            def encode(self, prompt):
                return list(range(len(prompt)))

        class FakeRuntime:
            tokenizer = FakeTokenizer()

        class FakeEngine:
            capacity = None

            def generate_stream_events(self, prompt, sampling, max_new_tokens):
                yield StreamEvent(kind="text", text="a")
                yield StreamEvent(kind="text", text="b")
                yield StreamEvent(kind="done", stop_reason="eos")

        service = ChatCompletionService.__new__(ChatCompletionService)
        service.config = ChatServiceConfig()
        service.model_name = SUPPORTED_MODEL
        service._engine_lock = threading.RLock()
        service._engine_slots = threading.BoundedSemaphore(service.config.max_active_requests)
        service._admission_lock = threading.Lock()
        service._active_requests = 0
        service._queued_requests = 0
        service._started_requests = 0
        service._completed_requests = 0
        service._rejected_requests = 0
        service._cancelled_requests = 0
        service._error_requests = 0
        service.runtime = FakeRuntime()
        service.engine = FakeEngine()
        request = ChatCompletionRequest.model_validate(
            {
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }
        )

        stream = service.stream_chat_completion(request)
        next(stream)
        stream.close()
        metrics = service.metrics_snapshot()

        self.assertEqual(metrics["active_requests"], 0)
        self.assertEqual(metrics["cancelled_requests"], 1)


if __name__ == "__main__":
    unittest.main()
