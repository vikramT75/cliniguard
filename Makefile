# Convenience targets. Every command also works without make: copy the line after the tab.
.PHONY: install models data proxy ui test test-all lint eval eval-quick

install:          ## install dependencies (CPU-only PyTorch)
	uv sync --all-extras

models:           ## pull the local models used by the proxy and the judge
	ollama pull llama3.2:3b
	ollama pull qwen3.5:4b

data:             ## rebuild evaluation datasets (openFDA corpus is cached in data/)
	uv run guardrail-data

proxy:            ## run the proxy on http://127.0.0.1:8000
	uv run guardrail-proxy

ui:               ## red-team playground + audit dashboard (needs the proxy running)
	uv run streamlit run ui/playground.py

test:             ## fast tests (no model downloads, no Ollama)
	uv run pytest -m "not slow"

test-all:         ## include DeBERTa tests
	uv run pytest

lint:
	uv run ruff check src tests ui
	uv run ruff format --check src tests ui

eval:             ## every evaluation suite; writes results/
	uv run guardrail-eval --suite all --judge-models qwen3.5:4b,llama3.2:3b

eval-quick:       ## CPU-only suites (no Ollama needed)
	uv run guardrail-eval --suite injection
	uv run guardrail-eval --suite phi
	uv run guardrail-eval --suite latency
