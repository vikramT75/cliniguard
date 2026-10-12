"""Prometheus metrics. Each app instance owns its registry so tests can build many apps."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from guardrail_proxy.guards.base import GuardResult

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "guardrail_requests_total",
            "Requests by final outcome",
            ["outcome"],
            registry=self.registry,
        )
        self.decisions = Counter(
            "guardrail_guard_decisions_total",
            "Guard decisions",
            ["guard", "action"],
            registry=self.registry,
        )
        self.guard_errors = Counter(
            "guardrail_guard_errors_total",
            "Guard failures/timeouts",
            ["guard"],
            registry=self.registry,
        )
        self.guard_latency = Histogram(
            "guardrail_guard_latency_seconds",
            "Per-guard latency",
            ["guard"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.upstream_latency = Histogram(
            "guardrail_upstream_latency_seconds",
            "Upstream LLM latency",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "guardrail_request_latency_seconds",
            "End-to-end latency",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )

    def observe_guards(self, results: list[GuardResult]) -> None:
        for r in results:
            self.decisions.labels(r.guard, r.action.value).inc()
            self.guard_latency.labels(r.guard).observe(r.latency_ms / 1000)
            if r.error:
                self.guard_errors.labels(r.guard).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)
