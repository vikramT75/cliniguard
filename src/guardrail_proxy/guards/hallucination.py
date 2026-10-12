from __future__ import annotations

from guardrail_proxy.config import GuardPolicy
from guardrail_proxy.guards.base import Guard, GuardContext, GuardVerdict, Stage
from guardrail_proxy.judge import Judge


class HallucinationGuard(Guard):
    """LLM-as-judge (or NLI) faithfulness check of the answer against the grounding context.

    Without grounding context there is nothing to check against, so the guard reports
    ``skipped`` rather than guessing; see the README for why self-consistency was not used.
    """

    name = "hallucination"
    stage = Stage.OUTPUT

    def __init__(self, policy: GuardPolicy, judge: Judge) -> None:
        super().__init__(policy)
        self.judge = judge

    async def check(self, ctx: GuardContext) -> GuardVerdict:
        answer = (ctx.response_text or "").strip()
        if not ctx.context_docs:
            return GuardVerdict(
                score=0.0,
                reason="no grounding context supplied; skipped",
                details={"skipped": True},
            )
        if not answer:
            return GuardVerdict(score=0.0, reason="empty answer")
        result = await self.judge.judge(answer, "\n\n".join(ctx.context_docs))
        unsupported = result.unsupported
        reason = (
            f"{len(unsupported)}/{len(result.claims)} claim(s) not supported by context"
            if unsupported
            else f"all {len(result.claims)} claim(s) supported"
        )
        return GuardVerdict(
            score=round(result.risk, 4),
            reason=reason,
            details={
                "backend": result.backend,
                "faithfulness": round(1 - result.risk, 4),
                "claims": [c.model_dump() for c in result.claims],
            },
        )
