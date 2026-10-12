from __future__ import annotations

import asyncio
from collections import Counter

from guardrail_proxy.config import GuardPolicy
from guardrail_proxy.guards.base import Guard, GuardContext, GuardVerdict, Stage
from guardrail_proxy.phi.detector import PHIDetector


class PHIGuard(Guard):
    """Pseudonymizes PHI before anything leaves the proxy.

    Every message is processed, including earlier assistant turns: those were re-hydrated with
    real values on the way back to the client, so sending them upstream unmodified would leak.
    All messages and context documents share one vault, so "John Smith" is <PERSON_1> everywhere.
    """

    name = "phi"
    stage = Stage.INPUT

    def __init__(self, policy: GuardPolicy, detector: PHIDetector) -> None:
        super().__init__(policy)
        self.detector = detector
        self.scan_context = bool(policy.params.get("scan_context", True))

    async def warmup(self) -> None:
        await asyncio.to_thread(self.detector.warmup)

    def _run(self, ctx: GuardContext) -> GuardVerdict:
        threshold = self.policy.threshold
        counts: Counter[str] = Counter()
        max_score = 0.0
        messages = []
        for m in ctx.messages:
            text, spans = self.detector.pseudonymize(m.get("content") or "", ctx.vault, threshold)
            messages.append({**m, "content": text})
            counts.update(s.entity for s in spans)
            max_score = max([max_score, *(s.score for s in spans)])
        docs = list(ctx.context_docs)
        if self.scan_context:
            docs = []
            for doc in ctx.context_docs:
                text, spans = self.detector.pseudonymize(doc, ctx.vault, threshold)
                docs.append(text)
                counts.update(s.entity for s in spans)
                max_score = max([max_score, *(s.score for s in spans)])
        total = sum(counts.values())
        reason = (
            (
                f"pseudonymized {total} PHI span(s): "
                + ", ".join(f"{k}x{v}" for k, v in sorted(counts.items()))
            )
            if total
            else "no PHI"
        )
        # Only entity *types and counts* are reported; values never leave the vault.
        return GuardVerdict(
            score=round(max_score, 4),
            reason=reason,
            details={"entities": dict(counts)},
            messages=messages,
            context_docs=docs,
        )

    async def check(self, ctx: GuardContext) -> GuardVerdict:
        return await asyncio.to_thread(self._run, ctx)


class PHILeakGuard(Guard):
    """Masks PHI that appears in the model output but was never supplied by the user.

    Because the model only saw placeholders, any real identifier in its answer is either
    memorised training data or a fabrication. Either way it must not reach the user.
    """

    name = "phi_leak"
    stage = Stage.OUTPUT

    def __init__(self, policy: GuardPolicy, detector: PHIDetector) -> None:
        super().__init__(policy)
        self.detector = detector

    def _run(self, ctx: GuardContext) -> GuardVerdict:
        text = ctx.response_text or ""
        redacted, spans = self.detector.redact(text, self.policy.threshold, keep=ctx.vault)
        if not spans:
            return GuardVerdict(score=0.0, reason="no PHI in output")
        counts = Counter(s.entity for s in spans)
        return GuardVerdict(
            score=round(max(s.score for s in spans), 4),
            reason=f"redacted {len(spans)} PHI span(s) not present in the request",
            details={"entities": dict(counts)},
            text=redacted,
        )

    async def check(self, ctx: GuardContext) -> GuardVerdict:
        return await asyncio.to_thread(self._run, ctx)
