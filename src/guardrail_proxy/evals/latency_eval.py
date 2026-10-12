"""Proxy overhead benchmark.

The upstream is a zero-latency FakeLLM, so every millisecond measured is the proxy's own cost
(guards, serialization, ASGI). Requests are fired concurrently through the real FastAPI app.
"""

from __future__ import annotations

import asyncio
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import httpx

from guardrail_proxy.app import create_app
from guardrail_proxy.audit import AuditLog
from guardrail_proxy.config import Policy, Settings
from guardrail_proxy.evals.common import latency_summary, md_table, save
from guardrail_proxy.evals.datasets import EVAL_DIR, read_jsonl
from guardrail_proxy.injection import ClassifierDetector
from guardrail_proxy.phi.detector import PHIDetector
from guardrail_proxy.providers import FakeLLM
from guardrail_proxy.service import Components, load_drug_terms

PHI_LEAK = {"phi_leak": {"action": "redact"}}


def _policy(name: str, inputs: dict, outputs: dict) -> Policy:
    return Policy.model_validate({"name": name, "input_guards": inputs, "output_guards": outputs})


CONFIGS = {
    "no guards (passthrough)": _policy("none", {}, {}),
    "injection: heuristic": _policy(
        "h", {"prompt_injection": {"params": {"detectors": ["heuristic"]}}}, {}
    ),
    "injection: heuristic + DeBERTa": _policy("hc", {"prompt_injection": {}}, {}),
    "PHI pseudonymization": _policy("phi", {"phi": {"action": "redact"}}, {}),
    "full input + output PHI": _policy(
        "full", {"prompt_injection": {}, "phi": {"action": "redact"}}, PHI_LEAK
    ),
}


async def _bench(app, prompts: list[str], concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    e2e: list[float] = []
    per_guard: dict[str, list[float]] = defaultdict(list)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://bench", timeout=60
    ) as client:

        async def one(text: str) -> None:
            async with sem:
                t0 = time.perf_counter()
                r = await client.post(
                    "/v1/chat/completions", json={"messages": [{"role": "user", "content": text}]}
                )
                e2e.append((time.perf_counter() - t0) * 1000)
                body = r.json()
                for g in body.get("guardrails", {}).get("results", []):
                    per_guard[g["guard"]].append(g["latency_ms"])

        await asyncio.gather(*(one(p) for p in prompts[:concurrency]))  # warm caches
        e2e.clear()
        per_guard.clear()
        t_start = time.perf_counter()
        await asyncio.gather(*(one(p) for p in prompts))
        wall = time.perf_counter() - t_start
    return {
        "e2e": latency_summary(e2e),
        "throughput_rps": round(len(prompts) / wall, 1),
        "per_guard": {g: latency_summary(v) for g, v in per_guard.items()},
    }


def run(settings: Settings, n: int = 200, concurrency: int = 8) -> dict:
    rng = random.Random(0)
    notes = [r["text"] for r in read_jsonl(EVAL_DIR / "phi_notes.jsonl")]
    questions = [
        r["text"] for r in read_jsonl(EVAL_DIR / "injection_clinical.jsonl") if r["label"] == 0
    ]
    prompts = [rng.choice(notes + questions) for _ in range(n)]
    comp = Components(
        provider=FakeLLM(lambda m: "Acknowledged. Continue the current plan."),
        phi_detector=PHIDetector(settings.spacy_model, allow_terms=load_drug_terms()),
        classifier=ClassifierDetector(settings.injection_model),
    )
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, policy in CONFIGS.items():
            audit = AuditLog(Path(tmp) / f"{policy.name}.db")
            app = create_app(settings, policy, comp, audit, warm=False)
            asyncio.run(app.state.service.pipeline.warmup())
            print(f"benchmarking: {name}", flush=True)
            results[name] = asyncio.run(_bench(app, prompts, concurrency))
    data = {"requests": n, "concurrency": concurrency, "configs": results}
    save("latency", data, _markdown(data))
    return data


def _markdown(data: dict) -> str:
    out = [
        "## Proxy overhead",
        "",
        f"{data['requests']} requests at concurrency {data['concurrency']} against a zero-latency "
        "fake upstream (so this is pure proxy cost), CPU only. Prompts are a mix of clinical "
        "notes and questions.",
        "",
    ]
    out.append(
        md_table(
            ["Configuration", "p50", "p95", "Throughput"],
            [
                [
                    name,
                    f"{c['e2e']['p50_ms']} ms",
                    f"{c['e2e']['p95_ms']} ms",
                    f"{c['throughput_rps']} req/s",
                ]
                for name, c in data["configs"].items()
            ],
        )
    )
    full = data["configs"]["full input + output PHI"]["per_guard"]
    out += [
        "",
        "Per-guard latency inside the full configuration (guards run concurrently, so these "
        "overlap rather than add up):",
        "",
    ]
    out.append(
        md_table(
            ["Guard", "p50", "p95"],
            [[g, f"{v['p50_ms']} ms", f"{v['p95_ms']} ms"] for g, v in full.items()],
        )
    )
    return "\n".join(out)
