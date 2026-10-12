"""Healthcare-specific Presidio recognizers.

Presidio ships good generic recognizers (names via spaCy, phone, email, SSN), but none for the
identifiers that show up in clinical text: medical record numbers, NPIs, DEA numbers, insurance
member IDs and dates of birth. These recognizers use a capture group so the reported span is the
identifier itself (``12345678``), not the label in front of it (``MRN: ``), and they validate
checksums where the identifier has one, which keeps false positives on random digit runs low.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from presidio_analyzer import AnalysisExplanation, EntityRecognizer, RecognizerResult


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def npi_valid(npi: str) -> bool:
    """NPI check digit = Luhn over the card-issuer prefix 80840 + the 10 NPI digits."""
    return len(npi) == 10 and npi.isdigit() and npi[0] in "12" and luhn_valid("80840" + npi)


def dea_valid(dea: str) -> bool:
    """DEA: 2 letters + 7 digits; (d1+d3+d5) + 2*(d2+d4+d6) must end in d7."""
    if not re.fullmatch(r"[A-Z][A-Z9]\d{7}", dea):
        return False
    d = [int(c) for c in dea[2:]]
    return (d[0] + d[2] + d[4] + 2 * (d[1] + d[3] + d[5])) % 10 == d[6]


@dataclass(frozen=True)
class GroupPattern:
    regex: re.Pattern[str]
    score: float
    # Returns a replacement score (or None to drop the match) given the captured value.
    validate: Callable[[str, float], float | None] | None = None


class GroupPatternRecognizer(EntityRecognizer):
    """Regex recognizer that reports the span of the ``value`` named group."""

    def __init__(self, entity: str, patterns: list[GroupPattern]) -> None:
        self.patterns = patterns
        super().__init__(supported_entities=[entity], name=f"{entity.title()}Recognizer")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: list[str], nlp_artifacts=None) -> list[RecognizerResult]:
        entity = self.supported_entities[0]
        results: list[RecognizerResult] = []
        for pattern in self.patterns:
            for m in pattern.regex.finditer(text):
                value = m.group("value")
                score: float | None = pattern.score
                if pattern.validate is not None:
                    score = pattern.validate(value, pattern.score)
                if score is None:
                    continue
                start, end = m.span("value")
                explanation = AnalysisExplanation(
                    recognizer=self.name,
                    original_score=score,
                    pattern_name=entity,
                    pattern=pattern.regex.pattern,
                )
                results.append(RecognizerResult(entity, start, end, score, explanation))
        return results


_I = re.IGNORECASE


def _npi_score(value: str, base: float) -> float | None:
    if npi_valid(value):
        return base
    return 0.6 if base >= 0.9 else None  # labelled-but-invalid still counts; bare-invalid dropped


def _dea_score(value: str, base: float) -> float | None:
    return base if dea_valid(value.upper()) else None


def _has_digit(value: str, base: float) -> float | None:
    return base if any(c.isdigit() for c in value) else None


_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_DATE = (
    r"(?:\d{1,2}[/-]\d{1,2}[/-](?:19|20)?\d{2}"
    r"|(?:19|20)\d{2}-\d{2}-\d{2}"
    rf"|{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}"
    rf"|\d{{1,2}}\s+{_MONTH},?\s+(?:19|20)\d{{2}})"
)

# Common USPS Publication 28 street suffixes (full forms).
_USPS_SUFFIXES = (
    "Street|Avenue|Road|Boulevard|Lane|Drive|Court|Way|Place|Circle|Terrace|Parkway|Highway|Square"
    "|Trail|Loop|Run|Path|Pike|Alley|Bend|Bluff|Bluffs|Branch|Bridge|Brook|Brooks|Burg|Bypass|Camp"
    "|Canyon|Cape|Causeway|Center|Centers|Cliff|Cliffs|Club|Common|Corner|Corners|Course|Courts"
    "|Cove|Coves|Creek|Crescent|Crest|Crossing|Crossroad|Curve|Dale|Dam|Divide|Drives|Estate"
    "|Estates|Expressway|Extension|Falls|Ferry|Field|Fields|Flat|Flats|Ford|Fords|Forest|Forge"
    "|Forges|Fork|Forks|Fort|Freeway|Garden|Gardens|Gateway|Glen|Glens|Green|Greens|Grove|Groves"
    "|Harbor|Harbors|Haven|Heights|Hill|Hills|Hollow|Inlet|Island|Islands|Isle|Junction|Junctions"
    "|Key|Keys|Knoll|Knolls|Lake|Lakes|Landing|Light|Lights|Lock|Locks|Lodge|Manor|Manors|Meadow"
    "|Meadows|Mews|Mill|Mills|Mission|Motorway|Mount|Mountain|Mountains|Neck|Orchard|Oval"
    "|Overpass|Park|Parks|Pass|Passage|Pine|Pines|Plain|Plains|Plaza|Point|Points|Port|Ports"
    "|Prairie|Radial|Ramp|Ranch|Rapid|Rapids|Rest|Ridge|Ridges|River|Roads|Route|Row|Rue|Shoal"
    "|Shoals|Shore|Shores|Skyway|Spring|Springs|Spur|Spurs|Squares|Station|Stravenue|Stream"
    "|Streets|Summit|Throughway|Trace|Track|Trafficway|Tunnel|Turnpike|Underpass|Union|Unions"
    "|Valley|Valleys|Via|Viaduct|View|Views|Village|Villages|Ville|Vista|Walk|Walks|Wall|Ways"
    "|Well|Wells|Burgs|Circles|Extensions|Parkways|Lanes|Avenues|Plazas|Trails|Crescents"
)
_NAME = r"[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?"
_STATES = (
    "AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV"
    "|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|PR|GU|VI|AS|MP|FM|MH|PW"
    "|AA|AE|AP"
)


def build_clinical_recognizers() -> list[EntityRecognizer]:
    return [
        GroupPatternRecognizer(
            "MRN",
            [
                GroupPattern(
                    re.compile(
                        r"\b(?:MRN|medical\s+record(?:\s+(?:number|no\.?|num|#))?|chart\s*(?:number|no\.?|#))"
                        r"\s*[:#]?\s*(?P<value>[A-Z]{0,3}-?\d{5,10})\b",
                        _I,
                    ),
                    0.95,
                ),
            ],
        ),
        GroupPatternRecognizer(
            "NPI",
            [
                GroupPattern(
                    re.compile(r"\bNPI\s*(?:number|no\.?|#)?\s*[:#]?\s*(?P<value>\d{10})\b", _I),
                    0.95,
                    _npi_score,
                ),
                GroupPattern(
                    re.compile(r"(?<![\d-])(?P<value>[12]\d{9})(?![\d-])"), 0.6, _npi_score
                ),
            ],
        ),
        GroupPatternRecognizer(
            "DEA_NUMBER",
            [
                GroupPattern(
                    re.compile(r"\b(?P<value>[ABCDEFGHJKLMPRSTUX][A-Z9]\d{7})\b"), 0.9, _dea_score
                ),
            ],
        ),
        GroupPatternRecognizer(
            "MEMBER_ID",
            [
                GroupPattern(
                    re.compile(
                        r"\b(?:member|subscriber|policy|insurance|plan)\s*(?:id|#|no\.?|number)\s*[:#]?\s*"
                        r"(?P<value>[A-Z0-9]{2,5}-?[A-Z0-9]{4,12})\b",
                        _I,
                    ),
                    0.9,
                    _has_digit,
                ),
            ],
        ),
        GroupPatternRecognizer(
            "DATE_OF_BIRTH",
            [
                GroupPattern(
                    re.compile(
                        rf"\b(?:DOB|D\.O\.B\.?|date\s+of\s+birth|born(?:\s+on)?|birth\s*date)\s*[:\-]?\s*"
                        rf"(?P<value>{_DATE})",
                        _I,
                    ),
                    0.95,
                ),
            ],
        ),
        # HIPAA Safe Harbor treats every date tied to an individual (admission, discharge, visit)
        # as PHI. Durations such as "for 3 days" are not matched, so dosing context survives.
        GroupPatternRecognizer(
            "CLINICAL_DATE",
            [
                GroupPattern(re.compile(rf"(?<![\w/-])(?P<value>{_DATE})(?![\w/-])", _I), 0.6),
            ],
        ),
        GroupPatternRecognizer(
            "STREET_ADDRESS",
            [
                GroupPattern(
                    re.compile(
                        r"\b(?P<value>\d{1,6}\s+(?:[A-Z][a-zA-Z]+\.?\s+){1,4}"
                        r"(?:(?:" + _USPS_SUFFIXES + r")\b"
                        r"|(?:St|Ave|Rd|Blvd|Ln|Dr|Ct|Pl|Cir|Ter|Pkwy|Hwy|Sq|Trl)\b\.?)"
                        r"(?:,?\s*(?:Apt\.?|Suite|Ste\.?|Unit|#)\s*[\w-]+)?)"
                    ),
                    0.85,
                ),
            ],
        ),
        # Safe Harbor: ZIP codes are PHI. A bare 5-digit number is too ambiguous, so require the
        # "City, ST 02139" shape.
        GroupPatternRecognizer(
            "ZIP_CODE",
            [
                GroupPattern(
                    re.compile(r"\b(?:" + _STATES + r")\.?,?\s+(?P<value>\d{5}(?:-\d{4})?)\b"), 0.85
                ),
                GroupPattern(
                    re.compile(
                        r"\b(?:zip|postal)\s*(?:code)?\s*[:#]?\s*(?P<value>\d{5}(?:-\d{4})?)\b", _I
                    ),
                    0.85,
                ),
            ],
        ),
        # Names right after a clinical label ("Patient: Doe, Jane", "Rx for Jane Doe", "Dr. Patel").
        # NER alone misses "Last, First" ordering and surname-only references.
        GroupPatternRecognizer(
            "PERSON",
            [
                GroupPattern(
                    re.compile(
                        r"\b(?:Patient|Pt|Name|Rx\s+for|referring|Prescriber|Signed|Attending|PCP|from)\s*:?\s*"
                        r"(?:Dr\.?\s+)?(?P<value>" + _NAME + r"(?:,\s*|\s+)" + _NAME + r")\b"
                    ),
                    0.9,
                ),
                GroupPattern(
                    re.compile(r"\bDr\.?\s+(?P<value>" + _NAME + r"(?:\s+" + _NAME + r")?)\b"), 0.8
                ),
            ],
        ),
        GroupPatternRecognizer(
            "LOCATION",
            [
                GroupPattern(
                    re.compile(
                        r"(?P<value>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?),\s+(?:"
                        + _STATES
                        + r")\s+\d{5}\b"
                    ),
                    0.85,
                ),
            ],
        ),
        # The phonenumbers-based recognizer rejects numbers with unassigned area codes; in a
        # privacy filter a false negative is worse, so US-formatted numbers are matched directly.
        GroupPatternRecognizer(
            "PHONE_NUMBER",
            [
                GroupPattern(
                    re.compile(
                        r"(?<![\w-])(?P<value>(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])"
                        r"\d{3}[\s.-]\d{4})(?:\s*(?:x|ext\.?)\s*\d{1,5})?(?![\w-])"
                    ),
                    0.7,
                ),
            ],
        ),
        GroupPatternRecognizer(
            "US_SSN",
            [
                GroupPattern(
                    re.compile(
                        r"\b(?:SSN|social\s+security(?:\s+(?:number|no\.?|#))?)\s*[:#]?\s*"
                        r"(?P<value>\d{3}-?\d{2}-?\d{4})\b",
                        _I,
                    ),
                    0.95,
                ),
            ],
        ),
    ]
