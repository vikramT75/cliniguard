"""Per-request reversible pseudonymization store.

The upstream LLM only ever sees placeholders such as ``<PERSON_1>``; the vault maps them back
to the original values when the response returns to the caller. A vault lives for exactly one
request and is never persisted.
"""

from __future__ import annotations

import re

# Presidio entity type -> placeholder label shown to the LLM.
LABELS: dict[str, str] = {
    "PERSON": "PERSON",
    "DATE_OF_BIRTH": "DOB",
    "CLINICAL_DATE": "DATE",
    "DATE_TIME": "DATE",
    "MRN": "MRN",
    "NPI": "NPI",
    "DEA_NUMBER": "DEA",
    "MEDICAL_LICENSE": "DEA",
    "MEMBER_ID": "MEMBER_ID",
    "US_SSN": "SSN",
    "PHONE_NUMBER": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "STREET_ADDRESS": "ADDRESS",
    "ZIP_CODE": "ZIP",
    "LOCATION": "LOCATION",
    "US_DRIVER_LICENSE": "LICENSE",
    "CREDIT_CARD": "CARD",
    "IP_ADDRESS": "IP",
    "URL": "URL",
}

# Small models sometimes rewrite "<PERSON_1>" as "[PERSON_1]", "PERSON_1" or "<Person_1>".
_TOKEN_RE = re.compile(
    r"(?:[<\[]\s*)?\b("
    + "|".join(sorted(set(LABELS.values()), key=len, reverse=True))
    + r")_(\d+)\b(?:\s*[>\]])?",
    re.IGNORECASE,
)


class PseudonymVault:
    def __init__(self) -> None:
        self._token_by_value: dict[tuple[str, str], str] = {}
        self._value_by_token: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._value_by_token)

    def tokenize(self, entity_type: str, value: str) -> str:
        label = LABELS.get(entity_type, "PHI")
        key = (label, value.strip().lower())
        if key in self._token_by_value:
            return self._token_by_value[key]
        self._counters[label] = self._counters.get(label, 0) + 1
        token = f"<{label}_{self._counters[label]}>"
        self._token_by_value[key] = token
        self._value_by_token[token] = value
        return token

    def knows_value(self, value: str) -> bool:
        needle = value.strip().lower()
        return any(v.strip().lower() == needle for v in self._value_by_token.values())

    def rehydrate(self, text: str) -> str:
        if not self._value_by_token:
            return text

        def swap(match: re.Match[str]) -> str:
            token = f"<{match.group(1).upper()}_{match.group(2)}>"
            return self._value_by_token.get(token, match.group(0))

        return _TOKEN_RE.sub(swap, text)

    def entity_counts(self) -> dict[str, int]:
        return dict(self._counters)
