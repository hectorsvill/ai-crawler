"""
llm/usage.py — Persistent per-call LLM token usage (stdlib sqlite3).

Table ``llm_usage`` lives in the same SQLite file as the LLM crawler
(``storage.db_path``, default crawl_data.db) so ``run_id`` joins to
``crawl_sessions.id``.
"""

from __future__ import annotations

import contextvars
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,             -- UTC ISO-8601
    run_id            TEXT,                      -- crawl session id (or a label, e.g. 'llm-ping')
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    purpose           TEXT,                      -- e.g. NavigatorDecision, ExtractionResult
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    estimated         INTEGER NOT NULL DEFAULT 0, -- 1 = tiktoken estimate (API gave no usage)
    cost_usd          REAL,                      -- NULL when model has no price in config
    latency_ms        INTEGER,
    ok                INTEGER NOT NULL DEFAULT 1,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS ix_llm_usage_run ON llm_usage(run_id);
CREATE INDEX IF NOT EXISTS ix_llm_usage_provider_ts ON llm_usage(provider, ts);
"""

_current_run: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_run_id", default=None)


def set_current_run(run_id: str | None) -> None:
    """Tag subsequent LLM calls (in this context) with *run_id*."""
    _current_run.set(run_id)


def current_run() -> str | None:
    return _current_run.get()


def estimate_cost(prices: dict[str, Any] | None, model: str, prompt_tokens: int,
                  completion_tokens: int) -> float | None:
    """USD cost from a price table {model: {input: $/1M, output: $/1M}}; None if unknown."""
    if not prices:
        return None
    p = prices.get(model)
    if p is None:  # allow prefix match, e.g. "gemini-3.5-flash-lite" for "gemini-3.5-flash-lite-001"
        for k, v in prices.items():
            if model.startswith(k):
                p = v
                break
    if not p:
        return None
    try:
        return round(prompt_tokens / 1e6 * float(p["input"])
                     + completion_tokens / 1e6 * float(p["output"]), 6)
    except (KeyError, TypeError, ValueError):
        return None


class UsageStore:
    def __init__(self, db_path: str | Path) -> None:
        self.path = str(db_path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def record(self, *, provider: str, model: str, prompt_tokens: int, completion_tokens: int,
               total_tokens: int | None = None, estimated: bool = False, purpose: str | None = None,
               run_id: str | None = None, cost_usd: float | None = None,
               latency_ms: int | None = None, ok: bool = True, error: str | None = None) -> int:
        row = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            run_id if run_id is not None else current_run(), provider, model, purpose,
            int(prompt_tokens or 0), int(completion_tokens or 0),
            int(total_tokens if total_tokens is not None else (prompt_tokens or 0) + (completion_tokens or 0)),
            1 if estimated else 0, cost_usd, latency_ms, 1 if ok else 0, (error or None) and error[:500],
        )
        with self._lock, self._conn() as c:
            cur = c.execute(
                """INSERT INTO llm_usage (ts, run_id, provider, model, purpose, prompt_tokens,
                       completion_tokens, total_tokens, estimated, cost_usd, latency_ms, ok, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
            return int(cur.lastrowid)

    def _where(self, provider=None, model=None, since=None, until=None, run_id=None):
        where, args = [], []
        for col, val in (("provider", provider), ("model", model), ("run_id", run_id)):
            if val:
                where.append(f"{col} = ?")
                args.append(val)
        if since:
            where.append("ts >= ?")
            args.append(since)
        if until:
            where.append("ts <= ?")
            args.append(until + "T23:59:59+00:00" if len(until) == 10 else until)
        return (" WHERE " + " AND ".join(where)) if where else "", args

    def summary(self, provider=None, model=None, since=None, until=None, run_id=None,
                group_by: str | None = "model") -> dict[str, Any]:
        w, args = self._where(provider, model, since, until, run_id)
        agg = ("COUNT(*) AS calls, COALESCE(SUM(ok=0),0) AS errors, COALESCE(SUM(estimated),0) AS estimated_calls, "
               "COALESCE(SUM(prompt_tokens),0) AS prompt_tokens, "
               "COALESCE(SUM(completion_tokens),0) AS completion_tokens, "
               "COALESCE(SUM(total_tokens),0) AS total_tokens, "
               "ROUND(SUM(cost_usd), 6) AS cost_usd, COALESCE(SUM(cost_usd IS NULL AND ok=1),0) AS unpriced_calls")
        with self._conn() as c:
            totals = dict(c.execute(f"SELECT {agg} FROM llm_usage{w}", args).fetchone())
            groups: list[dict[str, Any]] = []
            cols = {"provider": "provider", "model": "provider, model", "run": "run_id",
                    "day": "substr(ts, 1, 10) AS day", "purpose": "purpose"}
            if group_by:
                if group_by not in cols:
                    raise ValueError(f"group_by must be one of {', '.join(cols)}")
                sel = cols[group_by]
                key = sel.split(" AS ")[-1]
                groups = [dict(r) for r in c.execute(
                    f"SELECT {sel}, {agg} FROM llm_usage{w} GROUP BY {key} "
                    f"ORDER BY total_tokens DESC", args).fetchall()]
        return {"db_path": self.path,
                "filters": {"provider": provider, "model": model, "since": since,
                            "until": until, "run_id": run_id},
                "totals": totals, "group_by": group_by, "groups": groups}

    def rows(self, provider=None, model=None, since=None, until=None, run_id=None,
             limit: int = 50) -> list[dict[str, Any]]:
        w, args = self._where(provider, model, since, until, run_id)
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                f"SELECT * FROM llm_usage{w} ORDER BY id DESC LIMIT ?", args + [limit]).fetchall()]
