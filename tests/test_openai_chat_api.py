import unittest

try:
    from fastapi.testclient import TestClient
    from openai_api.app import create_app
    from openai_api.schemas import SUPPORTED_MODEL
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


if __name__ == "__main__":
    unittest.main()
