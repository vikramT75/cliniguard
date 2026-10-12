"""Runtime settings (env / .env) and the YAML guard policy."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


class Settings(BaseSettings):
    """Process-level configuration. Every field can be overridden with a GUARDRAIL_* env var."""

    model_config = SettingsConfigDict(env_prefix="GUARDRAIL_", env_file=".env", extra="ignore")

    # Upstream LLM the application actually wants to talk to.
    upstream_base_url: str = "http://localhost:11434/v1"
    upstream_api_key: str = "ollama"
    upstream_model: str = "llama3.2:3b"
    upstream_timeout_s: float = 120.0
    upstream_max_retries: int = 2

    # Optional second upstream used when the primary fails (e.g. a hosted endpoint).
    fallback_base_url: str | None = None
    fallback_api_key: str = "ollama"
    fallback_model: str | None = None

    # Judge LLM for the hallucination check. Deliberately a different model family.
    judge_base_url: str = "http://localhost:11434/v1"
    judge_api_key: str = "ollama"
    judge_model: str = "qwen3.5:4b"
    judge_timeout_s: float = 90.0
    # Hybrid reasoning models (Qwen3.x) think by default; a judge needs a fast, direct verdict.
    judge_disable_thinking: bool = True

    # Local ML models used by the guards.
    injection_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    spacy_model: str = "en_core_web_md"

    policy_path: Path = PROJECT_ROOT / "config" / "policy.yaml"
    audit_db_path: Path = DATA_DIR / "audit.db"

    # Key for the audit log's HMAC digests. Set it in production so digests are stable across
    # restarts; when unset a random per-process key is used.
    audit_hmac_key: str | None = None

    host: str = "127.0.0.1"
    port: int = 8000

    def judge_extra_params(self) -> dict[str, Any]:
        return {"reasoning_effort": "none"} if self.judge_disable_thinking else {}


class Action(StrEnum):
    """What a guard wants done. Ordered by severity (see ``severity``)."""

    ALLOW = "allow"
    FLAG = "flag"
    REDACT = "redact"
    BLOCK = "block"

    @property
    def severity(self) -> int:
        return {"allow": 0, "redact": 1, "flag": 2, "block": 3}[self.value]


class FailMode(StrEnum):
    OPEN = "open"  # guard error/timeout -> let the request through, but flag it
    CLOSED = "closed"  # guard error/timeout -> block


class GuardPolicy(BaseModel):
    enabled: bool = True
    action: Action = Action.BLOCK  # action taken when the guard's risk score >= threshold
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    timeout_s: float = Field(5.0, gt=0)
    fail_mode: FailMode = FailMode.CLOSED
    params: dict[str, Any] = Field(default_factory=dict)


class Policy(BaseModel):
    name: str = "default"
    on_block: str = Field("error", pattern="^(error|refusal)$")
    refusal_message: str = "I can't help with that request."
    input_guards: dict[str, GuardPolicy] = Field(default_factory=dict)
    output_guards: dict[str, GuardPolicy] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> Policy:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@lru_cache
def get_settings() -> Settings:
    return Settings()
