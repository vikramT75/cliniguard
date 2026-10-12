from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from guardrail_proxy.config import RESULTS_DIR


@dataclass
class Binary:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, predicted: bool, actual: bool) -> None:
        if predicted and actual:
            self.tp += 1
        elif predicted:
            self.fp += 1
        elif actual:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def summary(self) -> dict[str, Any]:
        def div(a: float, b: float) -> float | None:
            return round(a / b, 4) if b else None

        precision = div(self.tp, self.tp + self.fp)
        recall = div(self.tp, self.tp + self.fn)
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if precision and recall
            else (0.0 if precision is not None and recall is not None else None)
        )
        return {
            **asdict(self),
            "n": self.n,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": div(self.fp, self.fp + self.tn),
            "accuracy": div(self.tp + self.tn, self.n),
        }


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 2)
    return round(statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1], 2)


def latency_summary(ms: list[float]) -> dict[str, float | None]:
    return {
        "p50_ms": pct(ms, 50),
        "p95_ms": pct(ms, 95),
        "mean_ms": round(statistics.fmean(ms), 2) if ms else None,
    }


def fmt(value: Any, percent: bool = True) -> str:
    if value is None:
        return "–"
    if isinstance(value, float) and percent:
        return f"{value * 100:.1f}%"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def save(name: str, data: dict[str, Any], markdown: str, out_dir: Path = RESULTS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (out_dir / f"{name}.md").write_text(markdown.strip() + "\n", encoding="utf-8")
    print(markdown)
