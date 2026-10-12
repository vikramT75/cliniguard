"""PHI detection evaluation on 300 synthetic clinical notes with gold spans."""

from __future__ import annotations

import time
from collections import defaultdict

from guardrail_proxy.config import Settings
from guardrail_proxy.corpus import drug_terms
from guardrail_proxy.evals.common import fmt, latency_summary, md_table, save
from guardrail_proxy.evals.datasets import EVAL_DIR, read_jsonl
from guardrail_proxy.phi.detector import PHIDetector


def _coverage(gold: dict, predicted: list[tuple[int, int]]) -> tuple[bool, bool]:
    """(any overlap, every alphanumeric character of the gold value covered)."""
    chars = {
        i for i in range(gold["start"], gold["end"]) if gold["text"][i - gold["start"]].isalnum()
    }
    covered = {i for s, e in predicted for i in range(s, e)}
    hit = chars & covered
    return bool(hit), chars <= covered


def evaluate(detector: PHIDetector, notes: list[dict], threshold: float) -> dict:
    per_label: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # total, any, full
    pred_total = pred_true = leaked_notes = 0
    latency: list[float] = []
    misses: list[dict] = []
    false_pos: list[dict] = []
    for note in notes:
        t0 = time.perf_counter()
        spans = detector.detect(note["text"], threshold)
        latency.append((time.perf_counter() - t0) * 1000)
        predicted = [(s.start, s.end) for s in spans]
        leaked = False
        for gold in note["spans"]:
            any_hit, full = _coverage(gold, predicted)
            stats = per_label[gold["label"]]
            stats[0] += 1
            stats[1] += any_hit
            stats[2] += full
            if not full:
                leaked = True
                if len(misses) < 60:
                    misses.append(
                        {"note": note["id"], "label": gold["label"], "text": gold["text"]}
                    )
        leaked_notes += leaked
        for s in spans:
            pred_total += 1
            if any(s.start < g["end"] and g["start"] < s.end for g in note["spans"]):
                pred_true += 1
            elif len(false_pos) < 60:
                false_pos.append({"note": note["id"], "entity": s.entity, "text": s.text})
    total = sum(v[0] for v in per_label.values())
    return {
        "entities": total,
        "recall_any_overlap": round(sum(v[1] for v in per_label.values()) / total, 4),
        "recall_full": round(sum(v[2] for v in per_label.values()) / total, 4),
        "precision": round(pred_true / pred_total, 4) if pred_total else None,
        "note_leak_rate": round(leaked_notes / len(notes), 4),
        "per_label_recall_full": {k: round(v[2] / v[0], 4) for k, v in sorted(per_label.items())},
        "latency": latency_summary(latency),
        "sample_misses": misses,
        "sample_false_positives": false_pos,
    }


def run(settings: Settings) -> dict:
    notes = read_jsonl(EVAL_DIR / "phi_notes.jsonl")
    default = PHIDetector(settings.spacy_model, mode="presidio_default")
    clinical = PHIDetector(settings.spacy_model, mode="clinical", allow_terms=drug_terms())
    # Presidio's own default threshold is 0 (keep everything); the proxy policy uses 0.5.
    configs = {
        "Presidio default (threshold 0)": (default, 0.0),
        "Presidio default (threshold 0.5)": (default, 0.5),
        "Clinical recognizers (ours, 0.5)": (clinical, 0.5),
    }
    results = {}
    for name, (detector, threshold) in configs.items():
        detector.warmup()
        results[name] = evaluate(detector, notes, threshold)
    data = {"notes": len(notes), "configs": results}
    save("phi", data, _markdown(data))
    return data


def _markdown(data: dict) -> str:
    cfgs = data["configs"]
    out = [
        "## PHI detection",
        "",
        f"{data['notes']} synthetic notes, {next(iter(cfgs.values()))['entities']} gold PHI spans. "
        "*Full recall* counts an identifier as protected only if every character is masked; "
        "*note leak rate* is the share of notes where at least one identifier survives.",
        "",
    ]
    out.append(
        md_table(
            [
                "Config",
                "Recall (full)",
                "Recall (any overlap)",
                "Precision",
                "Note leak rate",
                "p50 latency",
            ],
            [
                [
                    n,
                    fmt(c["recall_full"]),
                    fmt(c["recall_any_overlap"]),
                    fmt(c["precision"]),
                    fmt(c["note_leak_rate"]),
                    f"{c['latency']['p50_ms']} ms",
                ]
                for n, c in cfgs.items()
            ],
        )
    )
    labels = sorted({k for c in cfgs.values() for k in c["per_label_recall_full"]})
    out += ["", "Recall by identifier type:", ""]
    out.append(
        md_table(
            ["Type", *cfgs],
            [
                [lab, *(fmt(c["per_label_recall_full"].get(lab)) for c in cfgs.values())]
                for lab in labels
            ],
        )
    )
    return "\n".join(out)
