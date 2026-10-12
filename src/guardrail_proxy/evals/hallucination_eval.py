"""Hallucination judge evaluation on openFDA passages with labelled perturbations."""

from __future__ import annotations

import asyncio
import random
import time
from collections import defaultdict

from guardrail_proxy.config import Settings
from guardrail_proxy.evals.common import Binary, fmt, latency_summary, md_table, save
from guardrail_proxy.evals.datasets import EVAL_DIR, read_jsonl
from guardrail_proxy.judge import Judge, LLMJudge, NLIJudge
from guardrail_proxy.providers import OpenAICompatibleProvider

THRESHOLD = 0.2  # matches config/policy.yaml: flag when >20% of claims are unsupported


async def evaluate(judge: Judge, rows: list[dict]) -> dict:
    overall = Binary()
    per_type: dict[str, Binary] = defaultdict(Binary)
    latency: list[float] = []
    failures = 0
    errors: list[dict] = []
    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        try:
            result = await judge.judge(row["answer"], row["context"])
            risk = result.risk
        except Exception as exc:  # count it; a broken judge must not silently look accurate
            failures += 1
            risk = 0.0
            result = None
            errors.append({"id": row["id"], "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
        latency.append((time.perf_counter() - t0) * 1000)
        predicted, actual = risk >= THRESHOLD, bool(row["label"])
        overall.add(predicted, actual)
        per_type[row["perturbation"]].add(predicted, actual)
        if predicted != actual and result is not None and len(errors) < 80:
            errors.append(
                {
                    "id": row["id"],
                    "perturbation": row["perturbation"],
                    "risk": round(risk, 3),
                    "unsupported": [c.claim for c in result.unsupported][:3],
                }
            )
        if (i + 1) % 25 == 0:
            print(f"  {judge.name}: {i + 1}/{len(rows)}", flush=True)
    by_type = {}
    for kind, b in per_type.items():
        s = b.summary()
        # "none" rows are all negatives: report how many stayed unflagged; others: detection rate.
        by_type[kind] = {
            "n": s["n"],
            "rate": (round(s["tn"] / s["n"], 4) if kind == "none" else s["recall"]),
        }
    return {
        "metrics": overall.summary(),
        "by_perturbation": by_type,
        "failures": failures,
        "latency": latency_summary(latency),
        "errors": errors,
    }


def build_judges(settings: Settings, models: list[str], with_nli: bool) -> list[Judge]:
    judges: list[Judge] = []
    for model in models:
        provider = OpenAICompatibleProvider(
            "judge",
            settings.judge_base_url,
            settings.judge_api_key,
            model,
            settings.judge_timeout_s,
            1,
        )
        judges.append(LLMJudge(provider, model, extra=settings.judge_extra_params()))
    if with_nli:
        judges.append(NLIJudge(settings.nli_model))
    return judges


def run(
    settings: Settings, models: list[str], with_nli: bool = True, limit: int | None = None
) -> dict:
    rows = read_jsonl(EVAL_DIR / "hallucination.jsonl")
    if limit and limit < len(rows):  # stratified by perturbation type, seeded
        rng = random.Random(0)
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[r["perturbation"]].append(r)
        rows = [
            r
            for g in groups.values()
            for r in rng.sample(g, max(1, round(limit * len(g) / len(rows))))
        ]
    results = {}
    for judge in build_judges(settings, models, with_nli):
        print(f"judging with {judge.name} ({len(rows)} items)...", flush=True)
        results[judge.name] = asyncio.run(evaluate(judge, rows))
    data = {
        "threshold": THRESHOLD,
        "n": len(rows),
        "positives": sum(r["label"] for r in rows),
        "judges": results,
    }
    save("hallucination", data, _markdown(data))
    return data


def _markdown(data: dict) -> str:
    judges = data["judges"]
    out = [
        "## Hallucination judge",
        "",
        f"{data['n']} answers grounded on openFDA label passages ({data['positives']} with an "
        f"injected error). Flag when ≥{int(data['threshold'] * 100)}% of claims are unsupported.",
        "",
    ]
    out.append(
        md_table(
            ["Judge", "Precision", "Recall", "F1", "Faithful kept", "Parse failures", "p50", "p95"],
            [
                [
                    name,
                    fmt(j["metrics"]["precision"]),
                    fmt(j["metrics"]["recall"]),
                    fmt(j["metrics"]["f1"]),
                    fmt(j["by_perturbation"].get("none", {}).get("rate")),
                    j["failures"],
                    f"{j['latency']['p50_ms']} ms",
                    f"{j['latency']['p95_ms']} ms",
                ]
                for name, j in judges.items()
            ],
        )
    )
    kinds = [
        k
        for k in ["dose_number", "frequency", "drug_swap", "negation", "fabricated_claim"]
        if any(k in j["by_perturbation"] for j in judges.values())
    ]
    out += ["", "Detection rate by error type:", ""]
    out.append(
        md_table(
            ["Error type", *judges],
            [
                [k, *(fmt(j["by_perturbation"].get(k, {}).get("rate")) for j in judges.values())]
                for k in kinds
            ],
        )
    )
    return "\n".join(out)
