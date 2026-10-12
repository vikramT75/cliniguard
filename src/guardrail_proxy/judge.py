"""Faithfulness judges: does the answer stay within the grounding context?

Two interchangeable backends:

* ``LLMJudge``: a second LLM decomposes the answer into atomic claims and labels each one
  supported / contradicted / not_found against the context, using JSON-schema constrained output.
* ``NLIJudge``: a small NLI cross-encoder scores entailment of each answer sentence against the
  context. Much faster, weaker on paraphrase and multi-sentence reasoning.

Both return a ``Judgement`` whose ``risk`` is the share of claims not supported by the context.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from functools import cached_property
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from guardrail_proxy.providers import LLMProvider

Verdict = Literal["supported", "contradicted", "not_found"]


class Claim(BaseModel):
    claim: str
    verdict: Verdict
    evidence: str = ""


class Judgement(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    backend: str = ""

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.claims if c.verdict != "supported"]

    @property
    def risk(self) -> float:
        return len(self.unsupported) / len(self.claims) if self.claims else 0.0


class Judge(Protocol):
    name: str

    async def judge(self, answer: str, context: str) -> Judgement: ...


class JudgeError(RuntimeError):
    pass


_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "contradicted", "not_found"],
                    },
                    "evidence": {"type": "string"},
                },
                "required": ["claim", "verdict", "evidence"],
            },
        }
    },
    "required": ["claims"],
}

_SYSTEM = (
    "You are a meticulous clinical fact-checker. You verify an ANSWER strictly against a CONTEXT "
    "excerpt (for example an FDA drug label). You never use outside medical knowledge: if the "
    "CONTEXT does not say it, it is not supported. Placeholders such as <PERSON_1> are "
    "de-identified values; treat them as ordinary names."
)

_INSTRUCTIONS = """\
Check the ANSWER against the CONTEXT.

1. Split the ANSWER into short atomic factual claims. One claim = one fact (one dose, one
   frequency, one population, one warning). Skip greetings, hedges and "consult your doctor".
2. Label every claim:
   - "supported": the CONTEXT states it. Paraphrase is fine, but numbers, units, frequencies,
     drug names and polarity (recommended vs contraindicated, should vs should not) must match.
   - "contradicted": the CONTEXT states something incompatible (different number, unit,
     frequency or drug, or the opposite polarity).
   - "not_found": the CONTEXT does not mention it.
3. "evidence": copy the CONTEXT sentence you relied on, or "" if none.

Example
CONTEXT: The recommended starting dose is 500 mg orally twice daily with meals. Do not use in
patients with eGFR below 30 mL/min/1.73 m2.
ANSWER: Start at 500 mg twice daily with meals. It can be used in patients with an eGFR below 30.
It also lowers the risk of stroke.
OUTPUT: {"claims": [
 {"claim": "Starting dose is 500 mg twice daily with meals", "verdict": "supported",
  "evidence": "The recommended starting dose is 500 mg orally twice daily with meals."},
 {"claim": "It can be used in patients with eGFR below 30", "verdict": "contradicted",
  "evidence": "Do not use in patients with eGFR below 30 mL/min/1.73 m2."},
 {"claim": "It lowers the risk of stroke", "verdict": "not_found", "evidence": ""}]}

CONTEXT:
<<<
{context}
>>>

ANSWER:
<<<
{answer}
>>>

Return only the JSON object."""


class LLMJudge:
    def __init__(
        self,
        provider: LLMProvider,
        model: str | None = None,
        max_context_chars: int = 6000,
        extra: dict | None = None,
    ) -> None:
        self.provider = provider
        self.model = model or provider.default_model
        self.max_context_chars = max_context_chars
        self.extra = extra or {}
        self.name = f"llm:{self.model}"

    async def judge(self, answer: str, context: str) -> Judgement:
        prompt = _INSTRUCTIONS.replace("{context}", context[: self.max_context_chars]).replace(
            "{answer}", answer
        )
        messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
        last_error = ""
        for _ in range(2):  # one retry on malformed JSON
            resp = await self.provider.complete(
                messages,
                model=self.model,
                temperature=0.0,
                max_tokens=1500,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "judgement", "schema": _SCHEMA},
                },
                **self.extra,
            )
            try:
                data = json.loads(_strip_fences(resp.text))
                return Judgement(
                    claims=[Claim.model_validate(c) for c in data.get("claims", [])],
                    backend=self.name,
                )
            except (json.JSONDecodeError, ValidationError, AttributeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        raise JudgeError(f"judge returned unparseable output: {last_error}")


def _strip_fences(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(<\"])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if len(s.strip()) > 3]


class NLIJudge:
    """Sentence-level entailment with a cross-encoder (labels: contradiction, entailment, neutral)."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        entail_threshold: float = 0.5,
        window_sentences: int = 3,
    ) -> None:
        self.model_name = model_name
        self.entail_threshold = entail_threshold
        self.window_sentences = window_sentences
        self.name = f"nli:{model_name.split('/')[-1]}"
        self._lock = threading.Lock()

    @cached_property
    def model(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_name, device="cpu")

    def _premises(self, context: str) -> list[str]:
        sents = split_sentences(context) or [context]
        w = self.window_sentences
        return [" ".join(sents[i : i + w]) for i in range(0, max(1, len(sents) - w + 1))]

    def _judge_sync(self, answer: str, context: str) -> Judgement:
        hypotheses = split_sentences(answer)
        premises = self._premises(context)
        if not hypotheses:
            return Judgement(backend=self.name)
        pairs = [(p, h) for h in hypotheses for p in premises]
        with self._lock:
            probs = self.model.predict(pairs, apply_softmax=True, batch_size=32)
        claims = []
        n = len(premises)
        for i, hyp in enumerate(hypotheses):
            block = probs[i * n : (i + 1) * n]  # rows: [contradiction, entailment, neutral]
            best = int(block[:, 1].argmax())
            entail, contra = float(block[best, 1]), float(block[:, 0].max())
            if entail >= self.entail_threshold:
                verdict: Verdict = "supported"
            elif contra >= 0.5:
                verdict = "contradicted"
            else:
                verdict = "not_found"
            claims.append(Claim(claim=hyp, verdict=verdict, evidence=premises[best]))
        return Judgement(claims=claims, backend=self.name)

    async def judge(self, answer: str, context: str) -> Judgement:
        return await asyncio.to_thread(self._judge_sync, answer, context)
