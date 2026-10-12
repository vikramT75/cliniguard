from __future__ import annotations

import asyncio

from guardrail_proxy.config import Action, GuardPolicy
from guardrail_proxy.guards.base import Guard, GuardContext, GuardVerdict, Stage
from guardrail_proxy.injection import ClassifierDetector, HeuristicDetector


class PromptInjectionGuard(Guard):
    """Scores user turns and (optionally) RAG context documents for prompt injection.

    Context documents matter as much as the user turn: in a clinical copilot the most realistic
    attack is *indirect*, an instruction hidden in a referral letter or patient note that gets
    pasted into the prompt. System messages are trusted (they come from the application).
    """

    name = "prompt_injection"
    stage = Stage.INPUT

    def __init__(
        self,
        policy: GuardPolicy,
        heuristic: HeuristicDetector,
        classifier: ClassifierDetector | None,
    ) -> None:
        super().__init__(policy)
        detectors = policy.params.get("detectors", ["heuristic", "classifier"])
        self.heuristic = heuristic if "heuristic" in detectors else None
        self.classifier = classifier if "classifier" in detectors else None
        if self.heuristic is None and self.classifier is None:
            raise ValueError("prompt_injection guard needs at least one detector")
        self.ensemble = policy.params.get("ensemble", "max")
        self.scan_context = bool(policy.params.get("scan_context", True))

    async def warmup(self) -> None:
        if self.classifier is not None:
            await asyncio.to_thread(self.classifier.score, "warmup")

    def _segments(self, ctx: GuardContext) -> list[tuple[str, str]]:
        segs = [
            (f"{m['role']}[{i}]", m["content"])
            for i, m in enumerate(ctx.messages)
            if m["role"] == "user" and m.get("content")
        ]
        if self.scan_context:
            segs += [(f"context[{i}]", doc) for i, doc in enumerate(ctx.context_docs) if doc]
        return segs

    def _combine(self, parts: dict[str, float]) -> tuple[float, Action | None]:
        values = list(parts.values())
        if self.ensemble == "mean":
            return sum(values) / len(values), None
        score = max(values)
        if self.ensemble == "tiered" and parts.get("heuristic", 0.0) < self.policy.threshold:
            # Only the classifier fired. It over-triggers on clinical shorthand ("disregard the
            # earlier potassium"), so weaker evidence gets FLAG + spotlighting, not a hard block.
            return score, Action.FLAG
        return score, None

    def _score_sync(self, segments: list[tuple[str, str]]) -> GuardVerdict:
        texts = [t for _, t in segments]
        heur = [self.heuristic.score(t) for t in texts] if self.heuristic else None
        clf = self.classifier.score_many(texts) if self.classifier else None
        # Pick the most incriminating segment: hard (blocking) evidence first, then raw score.
        best_key: tuple[bool, float] | None = None
        score, idx, parts, hits, action = 0.0, 0, {}, [], None
        for i in range(len(texts)):
            seg_parts: dict[str, float] = {}
            seg_hits: list[str] = []
            if heur is not None:
                seg_parts["heuristic"] = heur[i].score
                seg_hits += heur[i].hits
            if clf is not None:
                seg_parts["classifier"] = clf[i].score
                seg_hits += clf[i].hits
            seg_score, seg_action = self._combine(seg_parts)
            key = (seg_action is None and seg_score >= self.policy.threshold, seg_score)
            if best_key is None or key > best_key:
                best_key = key
                score, idx, parts, hits, action = seg_score, i, seg_parts, seg_hits, seg_action
        where = segments[idx][0]
        reason = (
            f"injection signals in {where}: {', '.join(hits)}" if hits else "no injection signal"
        )
        return GuardVerdict(
            score=round(score, 4),
            reason=reason,
            action=action,
            details={
                "segment": where,
                "detectors": {k: round(v, 4) for k, v in parts.items()},
                "signals": hits,
                "segments_scanned": len(segments),
            },
        )

    async def check(self, ctx: GuardContext) -> GuardVerdict:
        segments = self._segments(ctx)
        if not segments:
            return GuardVerdict(score=0.0, reason="nothing to scan")
        return await asyncio.to_thread(self._score_sync, segments)
