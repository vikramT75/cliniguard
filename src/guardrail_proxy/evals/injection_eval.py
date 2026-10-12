"""Prompt-injection detector evaluation: heuristic vs DeBERTa vs ensemble."""

from __future__ import annotations

import time
from collections import defaultdict

from guardrail_proxy.config import Settings
from guardrail_proxy.evals.common import Binary, fmt, latency_summary, md_table, save
from guardrail_proxy.evals.datasets import EVAL_DIR, read_jsonl
from guardrail_proxy.injection import ClassifierDetector, HeuristicDetector

THRESHOLD = 0.5


def _groups(row: dict) -> list[str]:
    if row["source"] != "clinical":
        return [row["source"]]
    if row["subset"].startswith("heldout"):
        return ["clinical held-out", f"clinical held-out: {row['subset']}"]
    return ["clinical (dev)", f"clinical: {row['subset']}"]


def _tiered(rows: list[dict], heur: list[float], clf: list[float]) -> dict[str, dict]:
    """What the production policy does: regex hit -> block, classifier-only hit -> flag."""
    out: dict[str, dict] = {}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[_groups(row)[0]].append(i)
    for group, idxs in groups.items():
        pos = [i for i in idxs if rows[i]["label"]]
        neg = [i for i in idxs if not rows[i]["label"]]
        blocked = {i for i in idxs if heur[i] >= THRESHOLD}
        caught = {i for i in idxs if max(heur[i], clf[i]) >= THRESHOLD}

        def rate(subset: list[int], hit: set[int]) -> float | None:
            return round(sum(i in hit for i in subset) / len(subset), 4) if subset else None

        out[group] = {
            "attacks": len(pos),
            "benign": len(neg),
            "attacks_blocked": rate(pos, blocked),
            "attacks_blocked_or_flagged": rate(pos, caught),
            "benign_blocked": rate(neg, blocked),
            "benign_flagged": rate(neg, caught - blocked),
        }
    return out


def run(settings: Settings, use_classifier: bool = True) -> dict:
    rows = read_jsonl(EVAL_DIR / "injection_public.jsonl") + read_jsonl(
        EVAL_DIR / "injection_clinical.jsonl"
    )
    heuristic = HeuristicDetector()
    scores: dict[str, list[float]] = {"heuristic": []}
    latency: dict[str, list[float]] = {"heuristic": []}
    for row in rows:
        t0 = time.perf_counter()
        scores["heuristic"].append(heuristic.score(row["text"]).score)
        latency["heuristic"].append((time.perf_counter() - t0) * 1000)

    if use_classifier:
        clf = ClassifierDetector(settings.injection_model)
        clf.score("warmup")
        scores["classifier"], latency["classifier"] = [], []
        for row in rows:  # one at a time so latency reflects a single request
            t0 = time.perf_counter()
            scores["classifier"].append(clf.score(row["text"]).score)
            latency["classifier"].append((time.perf_counter() - t0) * 1000)
        scores["ensemble"] = [
            max(h, c) for h, c in zip(scores["heuristic"], scores["classifier"], strict=True)
        ]
        latency["ensemble"] = [
            h + c for h, c in zip(latency["heuristic"], latency["classifier"], strict=True)
        ]

    metrics: dict[str, dict[str, dict]] = {}
    errors: dict[str, list[dict]] = defaultdict(list)
    for detector, values in scores.items():
        per_group: dict[str, Binary] = defaultdict(Binary)
        for row, score in zip(rows, values, strict=True):
            predicted, actual = score >= THRESHOLD, bool(row["label"])
            for group in _groups(row):
                per_group[group].add(predicted, actual)
            if predicted != actual and row["source"] == "clinical":
                errors[detector].append(
                    {
                        "id": row["id"],
                        "subset": row["subset"],
                        "score": round(score, 3),
                        "text": row["text"][:300],
                    }
                )
        metrics[detector] = {g: b.summary() for g, b in sorted(per_group.items())}

    data = {
        "threshold": THRESHOLD,
        "n": len(rows),
        "metrics": metrics,
        "latency": {d: latency_summary(v) for d, v in latency.items()},
        "clinical_errors": dict(errors),
    }
    save("injection", data, _markdown(data))
    return data


def _markdown(data: dict) -> str:
    detectors = list(data["metrics"])
    groups = list(next(iter(data["metrics"].values())))
    out = [
        "## Prompt-injection detection",
        "",
        f"Threshold {data['threshold']}, {data['n']} prompts. P = precision, R = recall, "
        "FPR = false-positive rate on benign prompts.",
        "",
    ]
    rows = []
    for g in groups:
        for d in detectors:
            m = data["metrics"][d][g]
            rows.append(
                [g, d, m["n"], fmt(m["precision"]), fmt(m["recall"]), fmt(m["f1"]), fmt(m["fpr"])]
            )
    out.append(md_table(["Dataset", "Detector", "N", "P", "R", "F1", "FPR"], rows))
    if data.get("tiered_policy"):
        out += [
            "",
            "Production policy (`ensemble: tiered`): a regex hit blocks; a DeBERTa-only hit is "
            "flagged and the request is answered under a spotlighting system prompt.",
            "",
        ]
        out.append(
            md_table(
                [
                    "Dataset",
                    "Attacks blocked",
                    "Attacks blocked or flagged",
                    "Benign blocked",
                    "Benign flagged",
                ],
                [
                    [
                        g,
                        fmt(t["attacks_blocked"]),
                        fmt(t["attacks_blocked_or_flagged"]),
                        fmt(t["benign_blocked"]),
                        fmt(t["benign_flagged"]),
                    ]
                    for g, t in data["tiered_policy"].items()
                ],
            )
        )
    out += ["", "Latency per prompt (CPU):", ""]
    out.append(
        md_table(
            ["Detector", "p50", "p95"],
            [[d, f"{v['p50_ms']} ms", f"{v['p95_ms']} ms"] for d, v in data["latency"].items()],
        )
    )
    return "\n".join(out)
