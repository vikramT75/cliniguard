"""PHI detection and pseudonymization on top of Microsoft Presidio."""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from functools import cached_property

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

from guardrail_proxy.phi.recognizers import build_clinical_recognizers
from guardrail_proxy.vault import _TOKEN_RE, PseudonymVault

# Presidio warns once per unmapped spaCy label (CARDINAL, PRODUCT...) on every call.
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

# Entities we treat as PHI (HIPAA Safe Harbor identifiers that show up in clinical chat).
# spaCy's generic DATE_TIME is excluded on purpose: it fires on "twice daily" and "for 3 days",
# and redacting dosing schedules destroys the clinical answer. Calendar dates are covered by
# CLINICAL_DATE / DATE_OF_BIRTH instead.
CLINICAL_ENTITIES = [
    "PERSON",
    "LOCATION",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "US_DRIVER_LICENSE",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "MRN",
    "NPI",
    "DEA_NUMBER",
    "MEMBER_ID",
    "DATE_OF_BIRTH",
    "CLINICAL_DATE",
    "STREET_ADDRESS",
    "ZIP_CODE",
]

# NER models tag capitalised clinical words as PERSON/LOCATION ("Lipitor", "Warfarin", "Tylenol").
_NER_ENTITIES = {"PERSON", "LOCATION"}
_CLINICAL_STOPWORDS = {
    "patient",
    "pt",
    "dr",
    "doctor",
    "nurse",
    "assessment",
    "plan",
    "history",
    "allergies",
    "medications",
    "vitals",
    "labs",
    "impression",
    "diagnosis",
    "follow-up",
    "er",
    "ed",
    "icu",
    "bid",
    "tid",
    "qid",
    "prn",
    "qd",
    "qhs",
    "po",
    "iv",
    "im",
    "sq",
    "mg",
    "mcg",
    "hba1c",
    "a1c",
    "inr",
    "bmp",
    "cbc",
    "ekg",
    "ecg",
    "mri",
    "ct",
    "copd",
    "chf",
    "ckd",
    "htn",
    "dm",
    "afib",
    # Payers are organisations, not individuals.
    "aetna",
    "cigna",
    "humana",
    "anthem",
    "bluecross",
    "blueshield",
    "unitedhealthcare",
    "kaiser",
    "medicare",
    "medicaid",
    "tricare",
}
# Safe Harbor permits geography at state level, so states are not PHI.
_US_STATES = {
    "al",
    "ak",
    "az",
    "ar",
    "ca",
    "co",
    "ct",
    "de",
    "dc",
    "fl",
    "ga",
    "hi",
    "id",
    "il",
    "in",
    "ia",
    "ks",
    "ky",
    "la",
    "me",
    "md",
    "ma",
    "mi",
    "mn",
    "ms",
    "mo",
    "mt",
    "ne",
    "nv",
    "nh",
    "nj",
    "nm",
    "ny",
    "nc",
    "nd",
    "oh",
    "ok",
    "or",
    "pa",
    "ri",
    "sc",
    "sd",
    "tn",
    "tx",
    "ut",
    "vt",
    "va",
    "wa",
    "wv",
    "wi",
    "wy",
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
    "usa",
    "us",
    "united states",
}


@dataclass(frozen=True)
class PHISpan:
    entity: str
    start: int
    end: int
    score: float
    text: str


class PHIDetector:
    """Thread-safe wrapper around a Presidio ``AnalyzerEngine``.

    ``mode="clinical"`` = Presidio + the healthcare recognizers + clinical entity list + drug-name
    allow-list. ``mode="presidio_default"`` = Presidio out of the box (used as the eval baseline).
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_md",
        mode: str = "clinical",
        allow_terms: set[str] | None = None,
    ) -> None:
        if mode not in {"clinical", "presidio_default"}:
            raise ValueError(f"unknown PHI detector mode: {mode}")
        self.spacy_model = spacy_model
        self.mode = mode
        self.allow_terms = {t.lower() for t in (allow_terms or set())} | _CLINICAL_STOPWORDS
        self._lock = threading.Lock()

    @cached_property
    def analyzer(self) -> AnalyzerEngine:
        nlp_engine = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": self.spacy_model}],
            }
        ).create_engine()
        registry = RecognizerRegistry(supported_languages=["en"])
        registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en"])
        if self.mode == "clinical":
            for recognizer in build_clinical_recognizers():
                registry.add_recognizer(recognizer)
        return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=["en"])

    def warmup(self) -> None:
        self.detect(
            "Warmup: John Smith, MRN 1234567, john@example.com, https://example.org, 617-555-0100."
        )

    def detect(self, text: str, threshold: float = 0.5) -> list[PHISpan]:
        if not text.strip():
            return []
        entities = CLINICAL_ENTITIES if self.mode == "clinical" else None
        with self._lock:  # spaCy pipelines are not guaranteed thread-safe
            raw = self.analyzer.analyze(
                text=text, language="en", entities=entities, score_threshold=threshold
            )
        placeholders = [m.span() for m in _TOKEN_RE.finditer(text)]
        spans = []
        for r in raw:
            value = text[r.start : r.end]
            if self.mode == "clinical" and not self._keep(
                r.entity_type, value, r.start, r.end, placeholders
            ):
                continue
            spans.append(PHISpan(r.entity_type, r.start, r.end, float(r.score), value))
        return _merge_adjacent_names(_resolve_overlaps(spans), text)

    def _keep(
        self, entity: str, value: str, start: int, end: int, placeholders: list[tuple[int, int]]
    ) -> bool:
        if any(start < pe and ps < end for ps, pe in placeholders):
            return False  # already-pseudonymized token such as <PERSON_1>
        if entity in _NER_ENTITIES:
            if any(ch.isdigit() for ch in value) or len(value.strip()) < 2:
                return False  # names and places don't contain digits ("atorvastatin 25 mg")
            words = re.findall(r"[a-z-]+", value.lower())
            if not words or any(w in self.allow_terms for w in words if w not in {"dr", "patient"}):
                return False  # contains a drug, payer or clinical abbreviation
            if entity == "LOCATION" and value.strip(" .,").lower() in _US_STATES:
                return False
        return True

    def pseudonymize(
        self, text: str, vault: PseudonymVault, threshold: float = 0.5
    ) -> tuple[str, list[PHISpan]]:
        spans = self.detect(text, threshold)
        tokens = [vault.tokenize(s.entity, s.text) for s in spans]  # number in reading order
        out = text
        for span, token in sorted(zip(spans, tokens, strict=True), key=lambda p: -p[0].start):
            out = out[: span.start] + token + out[span.end :]
        return out, spans

    def redact(
        self, text: str, threshold: float = 0.5, keep: PseudonymVault | None = None
    ) -> tuple[str, list[PHISpan]]:
        """Mask PHI with ``[REDACTED_<TYPE>]``. Values the vault already knows are left alone."""
        spans = [
            s for s in self.detect(text, threshold) if keep is None or not keep.knows_value(s.text)
        ]
        out = text
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            out = out[: span.start] + f"[REDACTED_{span.entity}]" + out[span.end :]
        return out, spans


def _merge_adjacent_names(spans: list[PHISpan], text: str) -> list[PHISpan]:
    """Join PERSON spans separated only by a comma/space ("Hernandez", "Mario")."""
    merged: list[PHISpan] = []
    for span in spans:
        prev = merged[-1] if merged else None
        if (
            prev
            and prev.entity == span.entity == "PERSON"
            and re.fullmatch(r",?\s{1,2}", text[prev.end : span.start])
        ):
            merged[-1] = PHISpan(
                "PERSON",
                prev.start,
                span.end,
                max(prev.score, span.score),
                text[prev.start : span.end],
            )
        else:
            merged.append(span)
    return merged


def _resolve_overlaps(spans: list[PHISpan]) -> list[PHISpan]:
    """Keep the highest-confidence (then longest) span wherever detections overlap."""
    kept: list[PHISpan] = []
    for span in sorted(spans, key=lambda s: (-s.score, -(s.end - s.start), s.start)):
        if all(span.end <= k.start or span.start >= k.end for k in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)
