from __future__ import annotations

import json

import pytest

from guardrail_proxy.audit import AuditLog
from guardrail_proxy.config import Policy, Settings
from guardrail_proxy.judge import Claim, Judgement
from guardrail_proxy.phi.detector import PHIDetector
from guardrail_proxy.providers import FakeLLM
from guardrail_proxy.service import Components


class FakeJudge:
    """Marks any claim containing a marker word as unsupported."""

    name = "fake-judge"

    def __init__(self, marker: str = "UNSUPPORTED") -> None:
        self.marker = marker
        self.calls: list[tuple[str, str]] = []

    async def judge(self, answer: str, context: str) -> Judgement:
        self.calls.append((answer, context))
        claims = [
            Claim(claim=s, verdict="not_found" if self.marker in s else "supported")
            for s in answer.split(".")
            if s.strip()
        ]
        return Judgement(claims=claims, backend=self.name)


@pytest.fixture(scope="session")
def phi_detector() -> PHIDetector:
    detector = PHIDetector()
    detector.warmup()
    return detector


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        audit_db_path=tmp_path / "audit.db", audit_hmac_key="test-key", upstream_model="fake-model"
    )


def make_policy(**overrides) -> Policy:
    data = {
        "name": "test",
        "on_block": "error",
        "input_guards": {
            "prompt_injection": {
                "action": "block",
                "threshold": 0.5,
                "params": {"detectors": ["heuristic"], "scan_context": True},
            },
            "phi": {"action": "redact", "threshold": 0.5},
        },
        "output_guards": {
            "phi_leak": {"action": "redact", "threshold": 0.5},
            "hallucination": {"action": "flag", "threshold": 0.2, "fail_mode": "open"},
        },
    }
    data.update(overrides)
    return Policy.model_validate(json.loads(json.dumps(data)))


@pytest.fixture
def policy() -> Policy:
    return make_policy()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_judge() -> FakeJudge:
    return FakeJudge()


@pytest.fixture
def components(fake_llm, fake_judge, phi_detector) -> Components:
    return Components(provider=fake_llm, judge=fake_judge, phi_detector=phi_detector)


@pytest.fixture
def audit(settings) -> AuditLog:
    return AuditLog(settings.audit_db_path, settings.audit_hmac_key)
