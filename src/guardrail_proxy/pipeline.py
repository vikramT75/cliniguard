"""Runs guards with timeouts, fail-open/fail-closed semantics and short-circuiting."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import perf_counter

from guardrail_proxy.config import Action, FailMode
from guardrail_proxy.guards.base import Guard, GuardContext, GuardResult, GuardVerdict


@dataclass
class StageOutcome:
    results: list[GuardResult] = field(default_factory=list)
    blocked_by: GuardResult | None = None
    messages: list[dict[str, str]] | None = None
    context_docs: list[str] | None = None
    text: str | None = None


class GuardPipeline:
    def __init__(self, input_guards: list[Guard], output_guards: list[Guard]) -> None:
        self.input_guards = input_guards
        self.output_guards = output_guards

    @property
    def guards(self) -> list[Guard]:
        return [*self.input_guards, *self.output_guards]

    async def warmup(self) -> None:
        await asyncio.gather(*(g.warmup() for g in self.guards))

    @staticmethod
    async def _execute(guard: Guard, ctx: GuardContext) -> tuple[GuardResult, GuardVerdict | None]:
        start = perf_counter()
        try:
            verdict = await asyncio.wait_for(guard.check(ctx), timeout=guard.policy.timeout_s)
        except Exception as exc:
            error = (
                "timeout"
                if isinstance(exc, TimeoutError)
                else f"{type(exc).__name__}: {str(exc)[:200]}"
            )
            closed = guard.policy.fail_mode is FailMode.CLOSED
            return GuardResult(
                guard=guard.name,
                stage=guard.stage,
                action=Action.BLOCK if closed else Action.FLAG,
                reason=f"guard unavailable ({'fail-closed' if closed else 'fail-open: unverified'})",
                latency_ms=round((perf_counter() - start) * 1000, 2),
                error=error,
            ), None
        return GuardResult(
            guard=guard.name,
            stage=guard.stage,
            action=guard.decide(verdict),
            score=verdict.score,
            reason=verdict.reason,
            details=verdict.details,
            latency_ms=round((perf_counter() - start) * 1000, 2),
        ), verdict

    async def run_input(self, ctx: GuardContext) -> StageOutcome:
        """Run input guards concurrently; the first BLOCK cancels the rest."""
        outcome = StageOutcome()
        if not self.input_guards:
            return outcome
        tasks = {asyncio.create_task(self._execute(g, ctx)): g for g in self.input_guards}
        finished: dict[str, tuple[GuardResult, GuardVerdict | None]] = {}
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                result, verdict = task.result()
                finished[result.guard] = (result, verdict)
                if result.action is Action.BLOCK and outcome.blocked_by is None:
                    outcome.blocked_by = result
            if outcome.blocked_by is not None:
                for task in pending:
                    task.cancel()
                for task in pending:
                    g = tasks[task]
                    finished[g.name] = (
                        GuardResult(
                            guard=g.name,
                            stage=g.stage,
                            action=Action.ALLOW,
                            reason="skipped: request already blocked",
                        ),
                        None,
                    )
                break

        # Report in policy order and apply rewrites deterministically.
        for guard in self.input_guards:
            result, verdict = finished[guard.name]
            outcome.results.append(result)
            if (
                outcome.blocked_by is None
                and verdict is not None
                and result.action is Action.REDACT
            ):
                if verdict.messages is not None:
                    outcome.messages = verdict.messages
                if verdict.context_docs is not None:
                    outcome.context_docs = verdict.context_docs
        return outcome

    async def run_output(self, ctx: GuardContext) -> StageOutcome:
        """Run output guards in order so a redaction is visible to the guards after it."""
        outcome = StageOutcome(text=ctx.response_text)
        for guard in self.output_guards:
            result, verdict = await self._execute(guard, ctx)
            outcome.results.append(result)
            if result.action is Action.BLOCK:
                outcome.blocked_by = result
                break
            if verdict is not None and result.action is Action.REDACT and verdict.text is not None:
                outcome.text = ctx.response_text = verdict.text
        return outcome
