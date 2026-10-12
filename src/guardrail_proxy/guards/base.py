"""Guard abstraction shared by every input and output check."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from guardrail_proxy.config import Action, GuardPolicy
from guardrail_proxy.vault import PseudonymVault


class Stage(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class GuardContext:
    """Everything a guard may look at for one request.

    Input guards read ``messages`` / ``context_docs`` exactly as the client sent them.
    Output guards read ``response_text`` plus the *sanitized* prompt that went upstream.
    """

    request_id: str
    messages: list[dict[str, str]]
    context_docs: list[str] = field(default_factory=list)
    vault: PseudonymVault = field(default_factory=PseudonymVault)
    response_text: str | None = None


class GuardVerdict(BaseModel):
    """What a guard found. The pipeline turns ``score`` into an action using the policy."""

    score: float = Field(ge=0.0, le=1.0, description="Risk in [0, 1]; higher is riskier.")
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    # Lets a guard soften the policy action for weaker evidence (e.g. block -> flag).
    action: Action | None = None
    # Optional rewrites, applied only when the resulting action is REDACT.
    messages: list[dict[str, str]] | None = None
    context_docs: list[str] | None = None
    text: str | None = None


class GuardResult(BaseModel):
    """A verdict after the policy has been applied. Safe to log: never contains raw PHI."""

    guard: str
    stage: Stage
    action: Action
    score: float | None = None
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


class Guard(ABC):
    """Base class for guards. Subclasses implement :meth:`check` and set ``name``/``stage``."""

    name: str
    stage: Stage

    def __init__(self, policy: GuardPolicy) -> None:
        self.policy = policy

    @abstractmethod
    async def check(self, ctx: GuardContext) -> GuardVerdict:
        """Inspect the request/response and return a risk score (and optional rewrite)."""

    async def warmup(self) -> None:  # noqa: B027 - optional hook, default is a no-op
        """Load heavy resources ahead of the first request."""

    def decide(self, verdict: GuardVerdict) -> Action:
        if verdict.score < self.policy.threshold:
            return Action.ALLOW
        action = verdict.action or self.policy.action
        has_rewrite = verdict.messages is not None or verdict.text is not None
        if action is Action.REDACT and not has_rewrite:
            return Action.FLAG
        return action
