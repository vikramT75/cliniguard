"""FastAPI adapter: OpenAI-compatible routes on top of :class:`GuardrailService`."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST

from guardrail_proxy import __version__
from guardrail_proxy.audit import AuditLog
from guardrail_proxy.config import Policy, Settings, get_settings
from guardrail_proxy.schemas import ChatCompletionRequest, ScanRequest
from guardrail_proxy.service import Components, ServiceResult, build_service, warmup


def create_app(
    settings: Settings | None = None,
    policy: Policy | None = None,
    components: Components | None = None,
    audit: AuditLog | None = None,
    warm: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    audit = audit or AuditLog(settings.audit_db_path, settings.audit_hmac_key)
    service = build_service(settings, policy, components, audit)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if warm:
            await warmup(service)  # load DeBERTa / spaCy before the first request, not during it
        yield

    app = FastAPI(
        title="Clinical Guardrail Proxy",
        version=__version__,
        lifespan=lifespan,
        description="OpenAI-compatible proxy that screens clinical LLM traffic.",
    )
    app.state.service = service

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Response:
        result = await service.chat(req)
        if req.stream and result.status_code == 200:
            return StreamingResponse(
                _as_sse(result), media_type="text/event-stream", headers=result.headers
            )
        return JSONResponse(result.body, status_code=result.status_code, headers=result.headers)

    @app.get("/v1/models")
    async def models() -> dict:
        ids = [settings.upstream_model] + (
            [settings.fallback_model] if settings.fallback_model else []
        )
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "created": 0, "owned_by": "guardrail-proxy"}
                for m in ids
            ],
        }

    @app.post("/v1/guardrails/scan")
    async def scan(req: ScanRequest) -> dict:
        return await service.scan(req.text, req.context)

    @app.get("/v1/guardrails/audit")
    async def audit_log(limit: int = Query(50, ge=1, le=1000)) -> dict:
        return {"data": audit.recent(limit)}

    @app.get("/v1/guardrails/policy")
    async def current_policy() -> dict:
        return service.policy.model_dump(mode="json")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "policy": service.policy.name,
            "guards": [g.name for g in service.pipeline.guards],
            "upstream": service.provider.name,
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(service.metrics.render(), media_type=CONTENT_TYPE_LATEST)

    return app


async def _as_sse(result: ServiceResult) -> AsyncIterator[str]:
    """Buffered streaming: output guards need the whole answer, so it is checked first and then
    replayed as OpenAI-style chunks. Clients using ``stream=True`` keep working unchanged."""
    body = result.body
    choice = body["choices"][0]
    base = {
        "id": body["id"],
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": body["model"],
    }
    content = choice["message"]["content"]
    first = {
        **base,
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(first)}\n\n"
    for i in range(0, len(content), 64):
        chunk = {
            **base,
            "choices": [
                {"index": 0, "delta": {"content": content[i : i + 64]}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    last = {
        **base,
        "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
        "guardrails": body["guardrails"],
    }
    yield f"data: {json.dumps(last)}\n\n"
    yield "data: [DONE]\n\n"
