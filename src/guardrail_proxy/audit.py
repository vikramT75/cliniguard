"""Append-only SQLite audit trail.

Stores what a compliance reviewer needs (who decided what, with which score, how fast) and
nothing a breach could expose: prompts and responses are kept only as keyed HMAC-SHA256 digests,
so identical prompts can be correlated without the text being recoverable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    request_id   TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    model        TEXT,
    action       TEXT NOT NULL,
    triggered    TEXT NOT NULL,
    input_hmac   TEXT NOT NULL,
    output_hmac  TEXT,
    upstream_ms  REAL,
    total_ms     REAL,
    guards_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_ts ON audit(ts);
"""


class AuditLog:
    def __init__(self, path: Path, hmac_key: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._key = (hmac_key or secrets.token_hex(32)).encode()
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def digest(self, text: str | None) -> str | None:
        if text is None:
            return None
        return hmac.new(self._key, text.encode("utf-8"), hashlib.sha256).hexdigest()

    def write(
        self,
        *,
        request_id: str,
        model: str | None,
        action: str,
        triggered: list[str],
        input_text: str,
        output_text: str | None,
        upstream_ms: float | None,
        total_ms: float,
        guards: list[dict[str, Any]],
    ) -> None:
        row = (
            request_id,
            time.time(),
            model,
            action,
            ",".join(triggered),
            self.digest(input_text),
            self.digest(output_text),
            upstream_ms,
            total_ms,
            json.dumps(guards, default=str),
        )
        with self._lock, self._connect() as conn:
            conn.execute("INSERT INTO audit VALUES (?,?,?,?,?,?,?,?,?,?)", row)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM audit ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["guards"] = json.loads(item.pop("guards_json"))
            out.append(item)
        return out
