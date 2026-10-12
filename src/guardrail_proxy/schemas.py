"""The subset of the OpenAI Chat Completions request schema the proxy understands."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_CONTEXT_TAG_RE = re.compile(r"<context>(.*?)</context>", re.DOTALL | re.IGNORECASE)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None

    def text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        # Multi-part content: keep the text parts, drop images etc.
        return "\n".join(p.get("text", "") for p in self.content if p.get("type") == "text")


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "default"
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    # Extension: grounding documents for the hallucination judge. With the OpenAI SDK, send them
    # via ``extra_body={"context": [...]}``. Alternatively wrap them in <context>...</context>
    # inside a system message and the proxy will find them there.
    context: list[str] | None = None

    def plain_messages(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.text()} for m in self.messages]

    def grounding(self) -> tuple[list[str], bool]:
        """Return (context documents, whether they came from the extension field)."""
        if self.context:
            return [c for c in self.context if c and c.strip()], True
        docs = [
            m.strip()
            for msg in self.messages
            if msg.role == "system"
            for m in _CONTEXT_TAG_RE.findall(msg.text())
        ]
        return docs, False


class ScanRequest(BaseModel):
    text: str
    context: list[str] = Field(default_factory=list)
