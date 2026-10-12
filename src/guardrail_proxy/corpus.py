"""openFDA drug-label corpus: real, public-domain clinical text used as grounding context.

Fetched once from api.fda.gov and cached in data/openfda_labels.json so the project (and its
evals) work offline afterwards.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import httpx

from guardrail_proxy.config import DATA_DIR

CORPUS_PATH = DATA_DIR / "openfda_labels.json"
OPENFDA_URL = "https://api.fda.gov/drug/label.json"

DRUGS = [
    "metformin",
    "lisinopril",
    "atorvastatin",
    "amlodipine",
    "levothyroxine",
    "metoprolol",
    "omeprazole",
    "losartan",
    "simvastatin",
    "gabapentin",
    "hydrochlorothiazide",
    "sertraline",
    "warfarin",
    "apixaban",
    "prednisone",
    "amoxicillin",
    "azithromycin",
    "clopidogrel",
    "furosemide",
    "escitalopram",
    "montelukast",
    "rosuvastatin",
    "pantoprazole",
    "tramadol",
    "glipizide",
    "spironolactone",
    "carvedilol",
    "allopurinol",
    "doxycycline",
    "fluoxetine",
]
SECTIONS = [
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "warnings_and_cautions",
    "warnings",
    "adverse_reactions",
    "use_in_specific_populations",
]
MAX_SECTION_CHARS = 4000


def _clean(text: str) -> str:
    text = re.sub(r"\(\s*\d+(\.\d+)*(\s*,\s*\d+(\.\d+)*)*\s*\)", "", text)  # "( 2.1, 5.3 )" refs
    text = re.sub(r"\s+", " ", text).strip()
    # Drop leading section numbers like "2 DOSAGE AND ADMINISTRATION 2.1 Adult Dosage".
    return re.sub(r"^\d+(\.\d+)*\s+[A-Z][A-Z &,/-]{3,}\s+", "", text)


def fetch_corpus(path: Path = CORPUS_PATH, drugs: list[str] = DRUGS) -> list[dict]:
    labels = []
    with httpx.Client(timeout=60) as client:
        for drug in drugs:
            resp = client.get(
                OPENFDA_URL,
                params={
                    "search": f'openfda.generic_name:"{drug}" AND _exists_:dosage_and_administration',
                    "limit": 25,
                },
            )
            if resp.status_code != 200:
                print(f"  ! {drug}: HTTP {resp.status_code}")
                continue

            # Prefer single-ingredient labels, then the most complete one.
            def rank(r: dict, drug: str = drug) -> tuple[bool, int]:
                generic = " ".join(r.get("openfda", {}).get("generic_name", [])).lower()
                single = drug in generic and not any(
                    sep in generic for sep in (" and ", ",", "-", "/")
                )
                return single, sum(bool(r.get(s)) for s in SECTIONS)

            best = max(resp.json()["results"], key=rank)
            sections = {
                s: _clean(" ".join(best[s]))[:MAX_SECTION_CHARS] for s in SECTIONS if best.get(s)
            }
            fda = best.get("openfda", {})
            labels.append(
                {
                    "drug": drug,
                    "generic_name": (fda.get("generic_name") or [drug])[0].lower(),
                    "brand_names": sorted({b.title() for b in fda.get("brand_name", [])})[:5],
                    "set_id": best.get("set_id"),
                    "effective_time": best.get("effective_time"),
                    "sections": sections,
                }
            )
            print(f"  + {drug}: {len(sections)} sections")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return labels


@lru_cache
def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def drug_terms() -> set[str]:
    terms = set(DRUGS)
    for label in load_corpus():
        terms.add(label["drug"])
        terms.update(w for w in re.findall(r"[a-z]+", label["generic_name"]) if len(w) > 3)
        terms.update(b.lower() for b in label["brand_names"])
    return terms
