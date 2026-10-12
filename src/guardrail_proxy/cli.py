"""`guardrail-proxy` entry point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from guardrail_proxy.config import get_settings


def serve() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the clinical guardrail proxy.")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument(
        "--policy",
        type=Path,
        default=settings.policy_path,
        help="YAML guard policy (default: config/policy.yaml)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings.policy_path = args.policy

    from guardrail_proxy.app import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port)
