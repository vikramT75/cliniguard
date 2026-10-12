"""LLM provider abstraction: any OpenAI-compatible endpoint, a fallback chain, and a fake for tests."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    text: str
    model: str
    provider: str
    finish_reason: str = "stop"
    usage: dict[str, int] = Field(default_factory=dict)


class ProviderError(RuntimeError):
    pass


class LLMProvider(ABC):
    name: str
    default_model: str

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> LLMResponse:
        """Return a single (non-streamed) chat completion."""


class OpenAICompatibleProvider(LLMProvider):
    """Ollama, vLLM, llama.cpp server, OpenAI, Azure, Together... anything speaking /v1/chat/completions."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        default_model: str,
        timeout_s: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.name = name
        self.default_model = default_model
        # The SDK handles retries with exponential backoff on connection errors, 429s and 5xx.
        self.client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=max_retries
        )

    async def complete(self, messages, *, model=None, temperature=None, max_tokens=None, **extra):
        kwargs: dict[str, Any] = {"model": model or self.default_model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs.update(extra)
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # normalise SDK errors for the fallback chain
            raise ProviderError(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        choice = resp.choices[0]
        usage = resp.usage.model_dump() if resp.usage else {}
        return LLMResponse(
            text=choice.message.content or "",
            model=resp.model,
            provider=self.name,
            finish_reason=choice.finish_reason or "stop",
            usage={k: v for k, v in usage.items() if isinstance(v, int)},
        )


class FallbackProvider(LLMProvider):
    """Try providers in order; the first success wins."""

    def __init__(self, providers: list[LLMProvider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self.providers = providers
        self.name = "fallback(" + ",".join(p.name for p in providers) + ")"
        self.default_model = providers[0].default_model

    async def complete(self, messages, *, model=None, temperature=None, max_tokens=None, **extra):
        errors: list[str] = []
        for i, provider in enumerate(self.providers):
            # A model name requested by the client only makes sense for the primary provider.
            chosen = model if i == 0 else None
            try:
                return await provider.complete(
                    messages, model=chosen, temperature=temperature, max_tokens=max_tokens, **extra
                )
            except ProviderError as exc:
                log.warning("provider %s failed, trying next: %s", provider.name, exc)
                errors.append(str(exc))
        raise ProviderError("all providers failed: " + " | ".join(errors))


class FakeLLM(LLMProvider):
    """Deterministic provider for tests and latency benchmarks.

    ``responder`` receives the message list and returns the reply text. The default echoes the
    last user message, which makes pseudonymization easy to assert on.
    """

    def __init__(
        self,
        responder: Callable[[list[dict[str, str]]], str] | None = None,
        delay_s: float = 0.0,
        name: str = "fake",
    ) -> None:
        self.name = name
        self.default_model = "fake-model"
        self.responder = responder or (lambda msgs: f"ECHO: {msgs[-1]['content']}")
        self.delay_s = delay_s
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages, *, model=None, temperature=None, max_tokens=None, **extra):
        self.calls.append([dict(m) for m in messages])
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        text = self.responder(messages)
        return LLMResponse(
            text=text,
            model=model or self.default_model,
            provider=self.name,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
