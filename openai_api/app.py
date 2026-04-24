from contextlib import asynccontextmanager
import json
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .schemas import ChatCompletionRequest
from .service import ChatCompletionService, ServiceOverloadedError

ServiceFactory = Callable[[], ChatCompletionService]


def _service_unavailable_message(fastapi_app: FastAPI) -> str:
    startup_error = getattr(fastapi_app.state, "startup_error", None)
    if startup_error:
        return f"service unavailable: {startup_error}"
    return "service unavailable: startup incomplete"


def get_service(fastapi_app: FastAPI) -> ChatCompletionService:
    service = getattr(fastapi_app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail=_service_unavailable_message(fastapi_app))
    return service


def create_app(service_factory: ServiceFactory = ChatCompletionService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        if not hasattr(fastapi_app.state, "service"):
            fastapi_app.state.service = None
        fastapi_app.state.startup_error = None
        if fastapi_app.state.service is None:
            try:
                fastapi_app.state.service = service_factory()
            except Exception as exc:
                fastapi_app.state.startup_error = str(exc) or exc.__class__.__name__
        yield

    fastapi_app = FastAPI(
        title="Gemma OpenAI-like API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @fastapi_app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @fastapi_app.get("/readyz")
    async def readyz() -> dict:
        service = getattr(fastapi_app.state, "service", None)
        if service is None:
            raise HTTPException(status_code=503, detail=_service_unavailable_message(fastapi_app))
        return {"status": "ready"}

    @fastapi_app.get("/metrics")
    async def metrics(request: Request) -> dict:
        service = get_service(request.app)
        if hasattr(service, "metrics_snapshot"):
            return service.metrics_snapshot()
        return {"status": "metrics_unavailable"}

    @fastapi_app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request):
        service = get_service(request.app)
        trace_id = request.headers.get("X-Trace-Id")
        try:
            if payload.stream:
                return StreamingResponse(
                    service.stream_chat_completion(payload),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            if trace_id and hasattr(service, "create_chat_completion_traced"):
                response, trace_data = await run_in_threadpool(service.create_chat_completion_traced, payload, trace_id)
                return JSONResponse(
                    response,
                    headers={"X-Trace-Data": json.dumps(trace_data, separators=(",", ":"))},
                )
            response = await run_in_threadpool(service.create_chat_completion, payload)
            return JSONResponse(response)
        except ServiceOverloadedError as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return fastapi_app


app = create_app()
