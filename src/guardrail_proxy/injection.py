"""Prompt-injection detectors: fast regex heuristics and a DeBERTa classifier."""

from __future__ import annotations

import base64
import binascii
import re
import threading
from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol

_I = re.IGNORECASE | re.DOTALL


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    weight: float
    pattern: re.Pattern[str]


# Rules aim for high precision. Each one requires an *instruction-like* object ("ignore the
# previous instructions"), so clinical phrasing such as "ignore the previous dose" or "override
# the default heparin protocol" does not trip them. The classifier supplies recall.
_OVERRIDE_OBJ = (
    r"(?:instructions?|prompts?|rules?|directions?|guidelines?|guardrails?|"
    r"context|messages?|commands?|constraints?|polic(?:y|ies)|programming|system\s+prompt)"
)
RULES: list[Rule] = [
    Rule(
        "ignore_previous",
        "override",
        0.9,
        re.compile(
            r"\b(?:ignore|disregard|forget|skip|bypass|override|drop)\b[\w\s,'-]{0,40}?"
            r"\b(?:previous|prior|above|earlier|preceding|all|any|your|these|those|the|initial|original|"
            r"system|safety)\b[\w\s,'-]{0,25}?\b" + _OVERRIDE_OBJ + r"\b",
            _I,
        ),
    ),
    Rule(
        "new_instructions",
        "override",
        0.75,
        re.compile(
            r"\b(?:new|updated|real|actual|revised)\s+(?:instructions?|task|rules|system\s+prompt)\s*[:\-]",
            _I,
        ),
    ),
    Rule(
        "persona_jailbreak",
        "jailbreak",
        0.9,
        re.compile(
            r"\b(?:you\s+are\s+now|from\s+now\s+on,?\s+you\s+(?:are|will)|act\s+as|pretend\s+(?:to\s+be|you\s+are)"
            r"|roleplay\s+as)\b[^.\n]{0,60}\b(?:DAN|unfiltered|unrestricted|uncensored|jailbroken|evil|"
            r"without\s+(?:any\s+)?(?:restrictions|limits|filters|rules|ethics)|no\s+(?:restrictions|limits|"
            r"filters|rules))",
            _I,
        ),
    ),
    Rule(
        "dan_mode",
        "jailbreak",
        0.95,
        re.compile(
            r"\b(?:DAN\s+mode|do\s+anything\s+now|developer\s+mode\s+(?:enabled|on|output)|jailbreak(?:ed)?\s+mode"
            r"|god\s+mode|STAN|DUDE\s+mode|AIM\s+mode)\b",
            _I,
        ),
    ),
    Rule(
        "no_restrictions",
        "jailbreak",
        0.7,
        re.compile(
            r"\b(?:without|free\s+(?:of|from)|no\s+longer\s+(?:bound|restricted)\s+by)\s+(?:any\s+)?"
            r"(?:ethical|moral|safety|content)\s+(?:guidelines|restrictions|filters|constraints|policies)",
            _I,
        ),
    ),
    Rule(
        "prompt_leak",
        "exfiltration",
        0.85,
        re.compile(
            r"\b(?:reveal|show|print|repeat|output|display|tell\s+me|give\s+me|leak|dump|what\s+(?:is|are))\b"
            r"[\w\s,'-]{0,30}?\b(?:your|the)\s+(?:system\s+prompt|initial\s+prompt|hidden\s+prompt|"
            r"(?:original|initial|hidden|secret|system)\s+instructions|instructions\s+above|prompt\s+above)",
            _I,
        ),
    ),
    Rule(
        "data_exfil",
        "exfiltration",
        0.8,
        re.compile(
            r"\b(?:send|post|forward|upload|email|exfiltrate|transmit)\b[^.\n]{0,60}\b(?:to|at)\s+"
            r"(?:https?://|www\.|[\w.-]+@[\w-]+\.)",
            _I,
        ),
    ),
    Rule(
        "markdown_image_exfil",
        "exfiltration",
        0.8,
        re.compile(r"!\[[^\]]*\]\(\s*https?://[^)]*(?:\?|&)[\w-]+=", _I),
    ),
    Rule(
        "dump_records",
        "exfiltration",
        0.7,
        re.compile(
            r"\b(?:list|output|print|dump|reveal|include)\b[^.\n]{0,40}\b(?:all|every|full|complete|other)\b"
            r"[^.\n]{0,30}\b(?:patients?|records?|SSNs?|social\s+security|PHI|database)\b",
            _I,
        ),
    ),
    Rule(
        "chat_template_tokens",
        "delimiter",
        0.85,
        re.compile(
            r"<\|?(?:im_start|im_end|system|endoftext|eot_id|start_header_id)\|?>|\[/?INST\]|<<SYS>>",
            _I,
        ),
    ),
    Rule(
        "fake_role_header",
        "delimiter",
        0.6,
        re.compile(
            r"(?:^|\n)\s*(?:#{2,}\s*)?(?:system|assistant|developer|admin)\s*(?:message|prompt|override)?\s*:"
            r"\s*(?:you|ignore|from\s+now|new|disregard|reveal|the\s+assistant)",
            _I,
        ),
    ),
    Rule(
        "hidden_ai_note",
        "indirect",
        0.8,
        re.compile(
            r"(?:note|message|instruction)s?\s+(?:to|for)\s+(?:the\s+)?(?:AI|assistant|LLM|model|chatbot|GPT)"
            r"\b[^.\n]{0,80}\b(?:ignore|disregard|must|instead|do\s+not|always|never|reveal|include|output)",
            _I,
        ),
    ),
    Rule(
        "privilege_escalation",
        "override",
        0.65,
        re.compile(
            r"\b(?:admin|root|sudo|maintenance|debug|god)\s+(?:mode|access|override|privileges?)\b"
            r"[^.\n]{0,40}\b(?:enabled|activated|granted|on)\b",
            _I,
        ),
    ),
    Rule(
        "addressed_to_ai",
        "indirect",
        0.85,
        re.compile(
            r"\b(?:instructions?|note|message|p\.?s\.?)\s+(?:for|to)\s+(?:the\s+|any\s+|whatever\s+)?"
            r"(?:AI|assistant|LLM|model|summari[sz]er|chatbot|bot)\b\s*(?::|reading\s+this|that\s+reads"
            r"|which\s+reads|reads\s+this)"
            r"|\b(?:AI|assistant|LLM|model|summari[sz]er|chatbot)\s+(?:reading|processing)\s+this\b",
            _I,
        ),
    ),
    Rule(
        "third_party_override",
        "override",
        0.9,
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,30}\bthe\s+(?:user|clinician|doctor|human)'?s?"
            r"\s+(?:question|request|instructions?|message|query)",
            _I,
        ),
    ),
    Rule(
        "ignore_above",
        "override",
        0.8,
        re.compile(
            r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:of\s+)?(?:the\s+|everything\s+)?above"
            r"(?=\s*(?:[.,;:!]|and\b|then\b|$))",
            _I,
        ),
    ),
    Rule(
        "config_leak",
        "exfiltration",
        0.8,
        re.compile(
            r"\b(?:contents?|text|wording|copy)\s+of\s+(?:your|the)\s+(?:configuration|config|"
            r"instructions|system\s+message|initial\s+message|message\s+(?:that\s+was\s+)?given\s+to\s+you)",
            _I,
        ),
    ),
    Rule(
        "no_policy_persona",
        "jailbreak",
        0.8,
        re.compile(
            r"\b(?:AI|assistant|model|chatbot)\s+(?:with\s+no|without\s+(?:a|any))\s+(?:content\s+)?"
            r"(?:policy|policies|filters?|restrictions|rules|guidelines)",
            _I,
        ),
    ),
]

_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060-\u2064\ufeff]")


@dataclass
class Detection:
    score: float
    hits: list[str] = field(default_factory=list)


class Detector(Protocol):
    name: str

    def score(self, text: str) -> Detection: ...


class HeuristicDetector:
    """Weighted regex rules plus de-obfuscation (zero-width characters, base64 payloads)."""

    name = "heuristic"

    def score(self, text: str) -> Detection:
        candidates = [_ZERO_WIDTH_RE.sub("", text)]
        for blob in _BASE64_RE.findall(text):
            try:
                decoded = base64.b64decode(blob, validate=True).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError, ValueError):
                continue
            if decoded.isprintable():
                candidates.append(decoded)

        weights: dict[str, float] = {}
        hits: set[str] = set()
        for i, candidate in enumerate(candidates):
            for rule in RULES:
                if rule.pattern.search(candidate):
                    weights[rule.id] = max(weights.get(rule.id, 0.0), rule.weight)
                    hits.add(rule.id if i == 0 else f"{rule.id}[base64]")
        if not weights:
            return Detection(0.0)
        categories = {r.category for r in RULES if r.id in weights}
        # Independent signals compound: strongest rule + 0.1 per extra category.
        score = min(1.0, max(weights.values()) + 0.1 * (len(categories) - 1))
        return Detection(score, sorted(hits))


class ClassifierDetector:
    """Fine-tuned DeBERTa-v3 injection classifier (ProtectAI v2), run on CPU.

    The model sees at most 512 tokens, so long inputs (patient notes, retrieved documents) are
    scored in overlapping character windows and the maximum window score is used. An injection
    buried at the end of a long note is therefore still seen.
    """

    name = "classifier"

    def __init__(
        self,
        model_name: str,
        window_chars: int = 1500,
        overlap_chars: int = 250,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.window_chars = window_chars
        self.overlap_chars = overlap_chars
        self.batch_size = batch_size
        self._lock = threading.Lock()

    @cached_property
    def pipeline(self):
        import torch
        from transformers import pipeline

        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
        return pipeline(
            "text-classification", model=self.model_name, truncation=True, max_length=512, device=-1
        )

    def _windows(self, text: str) -> list[str]:
        if len(text) <= self.window_chars:
            return [text]
        step = self.window_chars - self.overlap_chars
        return [
            text[i : i + self.window_chars] for i in range(0, len(text) - self.overlap_chars, step)
        ]

    def score_many(self, texts: list[str]) -> list[Detection]:
        windows: list[str] = []
        owners: list[int] = []
        for idx, text in enumerate(texts):
            for w in self._windows(text or " "):
                windows.append(w)
                owners.append(idx)
        with self._lock:
            preds = self.pipeline(windows, batch_size=self.batch_size, top_k=None)
        best = [0.0] * len(texts)
        for owner, pred in zip(owners, preds, strict=True):
            labels = {p["label"].upper(): p["score"] for p in pred}
            best[owner] = max(best[owner], float(labels.get("INJECTION", 0.0)))
        return [Detection(s, ["deberta"] if s >= 0.5 else []) for s in best]

    def score(self, text: str) -> Detection:
        return self.score_many([text])[0]


class EnsembleDetector:
    """Combine detectors. ``max`` favours recall (any detector can fire); ``mean`` favours precision."""

    def __init__(self, detectors: list[Detector], strategy: str = "max") -> None:
        if strategy not in {"max", "mean"}:
            raise ValueError(f"unknown ensemble strategy: {strategy}")
        self.detectors = detectors
        self.strategy = strategy
        self.name = "+".join(d.name for d in detectors)

    def score(self, text: str) -> tuple[Detection, dict[str, float]]:
        parts = {d.name: d.score(text) for d in self.detectors}
        scores = [p.score for p in parts.values()]
        combined = max(scores) if self.strategy == "max" else sum(scores) / len(scores)
        hits = sorted({h for p in parts.values() for h in p.hits})
        return Detection(combined, hits), {k: round(v.score, 4) for k, v in parts.items()}
