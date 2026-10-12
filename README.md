# Clinical Guardrail Proxy

[![CI](https://github.com/awat/clinical-guardrail-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/awat/clinical-guardrail-proxy/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

An OpenAI-compatible reverse proxy providing multi-stage safety and privacy guardrails for clinical and healthcare LLM applications:

- 🔒 **HIPAA-Compliant PHI Pseudonymization**: Ephemeral, reversible tokenization of Protected Health Information (`Maria Garcia` $\rightarrow$ `<PERSON_1>`) with zero raw PHI leaving the proxy.
- 🛡️ **Clinical Prompt Injection Defense**: Fast regex heuristics and fine-tuned DeBERTa-v3 sliding-window classifier with tiered ensembling and security spotlighting. Defends against both direct and indirect (RAG note-poisoning) attacks without false-flagging medical terminology.
- 🩺 **LLM-as-Judge Hallucination Verification**: Claim-level factual verification against grounding documents (such as FDA drug labels) using structured schema extraction or fast NLI cross-encoders.
- 🕵️ **Zero-PHI Audit Trail & Observability**: Append-only SQLite audit log storing keyed HMAC-SHA256 digests (cryptographically breach-safe correlation) alongside full Prometheus metrics.
- ⚡ **Drop-In OpenAI Compatibility**: Supports `/v1/chat/completions` (including streaming), `/v1/models`, and works seamlessly with the official OpenAI SDK, LangChain, or any HTTP client.

---

## Architecture Overview

```
                      +---------------------------------------------+
                      |               Clinical Client               |
                      |   (OpenAI SDK / Web App / EHR Integration)   |
                      +---------------------------------------------+
                                             |
                                POST /v1/chat/completions
                                             v
                      +---------------------------------------------+
                      |         FastAPI Guardrail Proxy             |
                      +---------------------------------------------+
                                             |
                   [ INPUT STAGE - Concurrent Asynchronous Guards ]
                                             |
         +-----------------------------------+-----------------------------------+
         |                                                                       |
         v                                                                       v
+-----------------------------+                                         +-----------------------------+
|   Prompt Injection Guard    |                                         |          PHI Guard          |
| - Heuristics (18 rules)     |                                         | - Presidio + Clinical       |
| - DeBERTa-v3 (sliding-win)  |                                         |   Recognizers (NPI, DEA...) |
| - Scans User Turn + RAG Docs|                                         | - Tokenize -> <PERSON_1>    |
| - Tiered: Block or Spotlight|                                         | - Store in Ephemeral Vault  |
+-----------------------------+                                         +-----------------------------+
         |                                                                       |
         +-----------------------------------+-----------------------------------+
                                             |
                                             v (Sanitized Messages & Spotlighting Prompt)
                      +---------------------------------------------+
                      |                 Upstream LLM                |
                      |      (Ollama, vLLM, OpenAI, Azure, etc.)    |
                      +---------------------------------------------+
                                             |
                                             v (Raw Completion with Placeholders)
                   [ OUTPUT STAGE - Sequential Evaluation Guards ]
                                             |
         +-----------------------------------+-----------------------------------+
         |                                                                       |
         v                                                                       v
+-----------------------------+                                         +-----------------------------+
|        PHI Leak Guard       |                                         |     Hallucination Guard     |
| - Detects unsupplied PHI in |                                         | - LLM Judge (Structured)    |
|   the model output          |                                         |   or NLI Cross-Encoder      |
| - Redacts to [REDACTED_*]   |                                         | - Checks against Context    |
|   (prevents memorization)   |                                         | - Flags unverified claims   |
+-----------------------------+                                         +-----------------------------+
                                             |
                                             v
                      +---------------------------------------------+
                      |             Vault Rehydration               |
                      |       (<PERSON_1> -> "Maria Garcia")        |
                      +---------------------------------------------+
                                             |
                                             +---> Keyed HMAC Audit Log (`data/audit.db`)
                                             |---> Prometheus Metrics (`/metrics`)
                                             v
                      +---------------------------------------------+
                      |            200 OK Client Response           |
                      |  (Rehydrated Answer + Guardrails Metadata)  |
                      +---------------------------------------------+
```

---

## Why General Guardrails Fail in Healthcare

| Capability | Generic Guardrails | Clinical Guardrail Proxy |
|---|---|---|
| **PHI Detection** | Often misses clinical identifiers (MRN, NPI, DEA, Member ID) while over-redacting drug names (*Lipitor*, *Warfarin*) and dosing schedules (*"twice daily for 3 days"*). | Custom clinical recognizers with checksum validation (Luhn for NPI, DEA modulo check); medical stopword and openFDA drug allow-lists. |
| **Privacy Mode** | One-way masking (`[REDACTED]`), destroying downstream assistant coherence. | Reversible, per-request in-memory pseudonymization. LLM reasons on placeholders; client receives seamless original values. |
| **Prompt Injection** | Naive keyword filters flag standard clinical orders (*"ignore previous dose"*, *"override default protocol"*). | Precision-targeted regex requiring instruction targets, plus tiered spotlighting that alerts the model rather than blocking care workflows. |
| **Hallucination** | Often ungrounded or checks internal self-consistency without reference materials. | Claim-level decomposition strictly evaluated against authoritative grounding context (e.g. FDA drug labels). |
| **Audit Compliance** | Logs full prompts (creating massive PHI breach surface) or logs nothing. | Stores HMAC-SHA256 digests and decision scores. Prompts correlate across queries without plaintext being recoverable. |

---

## Features Deep Dive

### 1. Reversible PHI Pseudonymization
- **HIPAA Safe Harbor Coverage**: Detects Person Names, Dates of Birth, Clinical Dates, Medical Record Numbers (MRN), National Provider Identifiers (NPI), DEA numbers, Health Plan Member IDs, Street Addresses, ZIP Codes, Phone Numbers, Emails, and SSNs.
- **Checksum Verification**: Validates 10-digit NPIs (Luhn algorithm over issuer prefix `80840`) and DEA registration formulas, eliminating false positives on raw digit sequences.
- **Ephemeral Vault (`PseudonymVault`)**: Per-request mapping between real entities and tokens (`<PERSON_1>`, `<DOB_1>`). The vault lives only in memory for that request and is never written to disk.
- **Tolerant Rehydration**: LLMs often modify casing or bracket styles (emitting `[PERSON_1]`, `PERSON_1`, or `<Person_1>`); the vault's regular expressions smoothly match and restore these to original values.
- **Outbound Leak Guard**: Detects if the model generated any real clinical identifiers that were *not* part of the user's input prompt (preventing training data extraction) and redacts them to `[REDACTED_<ENTITY>]`.

### 2. Clinical Prompt Injection Defense
- **Heuristic Rule Engine**: 18 weighted, precision-crafted regex rules detecting instruction overrides, DAN persona switches, system prompt leaks, markdown exfiltration URLs, and chat template token injections (`<|im_start|>`).
- **DeBERTa-v3 Transformer**: Uses `protectai/deberta-v3-base-prompt-injection-v2` in overlapping character windows (1500 chars with 250 overlap), catching payloads buried deep in long referral notes.
- **Indirect Attack Scanning**: Analyzes both the user's chat turns and any retrieved RAG grounding documents (`context`).
- **Tiered Policy & Security Spotlighting**:
  - High-confidence heuristic match $\rightarrow$ **BLOCK** (HTTP 400 or canned refusal).
  - Classifier-only match $\rightarrow$ **FLAG + Spotlighting**. An advisory prompt is injected instructing the LLM that context documents and user messages are untrusted data, preventing disruption of emergency or complex care questions.

### 3. Faithfulness & Hallucination Judging
- **Grounding Context**: Pass reference documents via OpenAI SDK's `extra_body={"context": [...]}` or by wrapping them in `<context>...</context>` inside system messages.
- **Interchangeable Backends**:
  - `LLMJudge`: Uses a separate LLM family (e.g. `qwen3.5:4b` when upstream is `llama3.2:3b`) with structured JSON schema output to segment the answer into atomic claims labeled `supported`, `contradicted`, or `not_found` with direct citations.
  - `NLIJudge`: Fast sentence-level entailment scoring using a cross-encoder (`cross-encoder/nli-deberta-v3-small`).
- **Annotation Notice**: When unsupported claims exceed policy threshold (e.g., $>20\%$), the proxy flags the response and appends:
  ```markdown
  ---
  ⚠️ Guardrail notice: 1 statement(s) in this answer could not be verified against the provided reference material.
  ```

### 4. Privacy-Preserving Audit Trail
- Stored in append-only SQLite (`data/audit.db`).
- Input and output texts are stored strictly as **keyed HMAC-SHA256 digests**.
- Records timestamp, model, final outcome (`allowed`, `flagged`, `redacted`, `blocked`), triggered guards, latency breakdown, and guard verdict scores.
- Exposes queryable inspection via `/v1/guardrails/audit`.

---

## Evaluation Results

Evaluation suites run on synthetic clinical datasets with exact ground-truth labels (seeded and reproducible via `uv run guardrail-eval`).

### PHI Detection (300 Synthetic Notes, 2,135 Gold Spans)

| Configuration | Recall (Full) | Recall (Any Overlap) | Precision | Note Leak Rate | p50 Latency |
|---|---|---|---|---|---|
| Presidio Default (Threshold 0.0) | 85.0% | 94.8% | 69.3% | 67.0% | 37.6 ms |
| Presidio Default (Threshold 0.5) | 72.7% | 80.1% | 65.7% | 81.7% | 42.8 ms |
| **Clinical Recognizers (Ours, 0.5)** | **97.4%** | **99.2%** | **100.0%** | **17.7%** | **36.9 ms** |

*Full recall by identifier type with our clinical recognizers: DEA (100%), NPI (100%), MRN (100%), Member ID (100%), DOB (100%), Clinical Date (100%), Phone (100%), SSN (100%), Location (98.5%), Address (94.0%), Person (94.0%), ZIP (92.5%).*

### Prompt Injection Detection (697 Prompts)

| Dataset | Detector | N | Precision | Recall | F1 | False Positive Rate |
|---|---|---|---|---|---|---|
| **Clinical (Ours)** | Heuristic | 181 | 100.0% | 60.9% | 75.7% | **0.0%** |
| | Classifier | 181 | 71.2% | 53.6% | 61.2% | 13.4% |
| | **Ensemble** | 181 | **77.6%** | **75.4%** | **76.5%** | **13.4%** |
| *Clinical: Benign Trigger Words* | Heuristic | 72 | – | – | – | **0.0%** |
| *Clinical: Direct Attacks* | Ensemble | 29 | 100.0% | 79.3% | 88.5% | – |
| *Clinical: Indirect Notes* | Ensemble | 40 | 100.0% | 72.5% | 84.1% | – |
| **Jailbreak Benchmark** | Ensemble | 400 | 96.0% | 84.4% | 89.8% | 1.9% |

### Proxy Overhead & Latency (CPU)

| Configuration | p50 Latency | p95 Latency | Throughput |
|---|---|---|---|
| No Guards (Passthrough) | 1.2 ms | 3.1 ms | 310 req/s |
| Injection: Heuristic Only | 1.4 ms | 3.8 ms | 285 req/s |
| PHI Pseudonymization | 38.5 ms | 55.2 ms | 24 req/s |
| Full Pipeline (Input Injection + PHI + Output Leak) | 255.4 ms | 410.8 ms | 4 req/s |

---

## Quickstart

### 1. Requirements & Installation
- Python 3.11 to 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- (Optional) [Ollama](https://ollama.ai) for local LLMs

```bash
# Clone repository
git clone https://github.com/awat/clinical-guardrail-proxy.git
cd clinical-guardrail-proxy

# Install dependencies (CPU PyTorch, spaCy model, Presidio, Transformers)
uv sync --all-extras
```

### 2. Configure Environment
Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Adjust variables as needed:
```ini
# Upstream model endpoint (any OpenAI-compatible service)
GUARDRAIL_UPSTREAM_BASE_URL=http://localhost:11434/v1
GUARDRAIL_UPSTREAM_API_KEY=ollama
GUARDRAIL_UPSTREAM_MODEL=llama3.2:3b

# Hallucination judge model
GUARDRAIL_JUDGE_BASE_URL=http://localhost:11434/v1
GUARDRAIL_JUDGE_MODEL=qwen3.5:4b

# Policy and Audit settings
GUARDRAIL_POLICY_PATH=config/policy.yaml
GUARDRAIL_AUDIT_HMAC_KEY=your-secure-random-hmac-key
```

If using Ollama, pull the recommended models:
```bash
ollama pull llama3.2:3b
ollama pull qwen3.5:4b
```

### 3. Run the Proxy
```bash
uv run guardrail-proxy
```
The proxy will start on `http://127.0.0.1:8000`.

### 4. Interactive Red-Team Playground
Launch the Streamlit UI dashboard to test pre-configured red-teaming scenarios and inspect audit records:
```bash
uv run streamlit run ui/playground.py
```

---

## Client Usage

### Python (Official OpenAI SDK)

Simply change `base_url` to the guardrail proxy:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed-for-local",
)

# 1. PHI is pseudonymized before going upstream and restored in the answer:
response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": (
                "Patient Maria Garcia (DOB: 04/12/1975, MRN: 9812401) has an eGFR of 38. "
                "Can Maria Garcia take metformin, and what is the starting dose?"
            ),
        }
    ],
    # Supply grounding context for hallucination verification:
    extra_body={
        "context": [
            "Metformin label: Starting dose is 500 mg orally once daily. "
            "Contraindicated in patients with severe renal impairment (eGFR below 30 mL/min/1.73 m2)."
        ]
    },
)

print(response.choices[0].message.content)

# Inspect guardrail decisions returned in the response:
print(response.model_extra["guardrails"])
```

### Streaming Responses

The proxy buffers streamed completions to ensure output guards (PHI leaks and hallucination claims) verify the complete response before emitting standard Server-Sent Events (SSE):

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain metformin contraindications."}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## Guardrail Policy (`config/policy.yaml`)

Safety policies are defined declaratively:

```yaml
name: clinical-default
on_block: error            # 'error' -> HTTP 400 error; 'refusal' -> 200 with canned text
refusal_message: >-
  I can't process this request because it was flagged by the clinical safety policy.

input_guards:
  prompt_injection:
    action: block
    threshold: 0.5
    timeout_s: 5.0
    fail_mode: closed      # Unscanned prompt never reaches the LLM
    params:
      detectors: [heuristic, classifier]
      ensemble: tiered     # Regex hit -> block; classifier-only -> flag & spotlight
      scan_context: true   # Scans RAG context documents for indirect injection

  phi:
    action: redact         # Reversible pseudonymization: LLM sees <PERSON_1>
    threshold: 0.5
    timeout_s: 5.0
    fail_mode: closed

output_guards:
  phi_leak:
    action: redact         # Masks memorized/fabricated PHI
    threshold: 0.5
    timeout_s: 5.0
    fail_mode: closed

  hallucination:
    action: flag           # Annotates response with unverified claims warning
    threshold: 0.2         # Flagged when >20% of claims are unsupported
    timeout_s: 60.0
    fail_mode: open        # Judge outage falls back to 'unverified', does not block app
    params:
      backend: llm         # 'llm' (structured claim check) or 'nli' (cross-encoder)
      annotate: true
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion route with guardrail screening. |
| `GET` | `/v1/models` | Lists available configured upstream models. |
| `POST` | `/v1/guardrails/scan` | Standalone endpoint to run input guards without calling an LLM. |
| `GET` | `/v1/guardrails/audit` | Query recent audit records (HMAC digests, guard verdicts, latency). |
| `GET` | `/v1/guardrails/policy` | Returns current active YAML guard policy. |
| `GET` | `/healthz` | Health check returning service status, active guards, and upstream name. |
| `GET` | `/metrics` | Prometheus metrics for latencies, guard verdicts, and request counts. |

---

## Development & Testing

### Running Tests
```bash
# Fast unit tests (no model downloads, no Ollama required)
uv run pytest -m "not slow"

# Include DeBERTa classifier tests
uv run pytest

# Code style and linting
uv run ruff check src tests ui
uv run ruff format --check src tests ui
```

### Running Benchmark Evaluations
```bash
# Quick evaluation (CPU-only: injection heuristics, PHI, latency)
uv run guardrail-eval --suite injection
uv run guardrail-eval --suite phi
uv run guardrail-eval --suite latency

# Full evaluation including LLM hallucination judge
uv run guardrail-eval --suite all --judge-models qwen3.5:4b

# Rebuild evaluation datasets from scratch (seeded and deterministic)
uv run guardrail-data
```

---

## Docker Deployment

Run both Ollama and the Guardrail Proxy using Docker Compose:

```bash
docker compose up -d
```

The container automatically bakes the DeBERTa injection model and spaCy weights into the image for offline startup.

---

## License

This project is licensed under the Apache 2.0 License. All datasets in `data/` are public-domain or synthetic; no real patient data is included.
