FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1 \
    GUARDRAIL_HOST=0.0.0.0 GUARDRAIL_PORT=8000
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY config ./config
COPY data/openfda_labels.json ./data/openfda_labels.json
RUN uv sync --frozen --no-dev
# Bake the injection classifier into the image so the container starts offline.
RUN uv run python -c "from transformers import pipeline; pipeline('text-classification', model='protectai/deberta-v3-base-prompt-injection-v2')"

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uv", "run", "--no-sync", "guardrail-proxy"]
