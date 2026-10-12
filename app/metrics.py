"""Request-level metrics in SQLite: cost, latency, cache hits, routing decisions.

Everything the dashboard shows is aggregated from this table — the point of the
project is being able to prove cost/latency impact per request.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    query TEXT NOT NULL,
    route_tier TEXT,
    model TEXT,
    difficulty REAL,
    reasons TEXT,
    forced TEXT,
    cache_hit INTEGER,
    cache_similarity REAL,
    cache_lookup_ms REAL,
    retrieval_ms REAL,
    generation_ms REAL,
    total_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    escalated INTEGER,
    context_docs TEXT,
    answer_excerpt TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
"""


class MetricsDB:
    def __init__(self, path: Path = config.DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(self, **fields) -> int:
        fields = dict(fields)
        fields.setdefault("ts", time.time())
        for key in ("reasons", "context_docs"):
            if isinstance(fields.get(key), (list, dict)):
                fields[key] = json.dumps(fields[key])
        columns = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        with self._lock:
            cur = self._conn.execute(
                f"INSERT INTO requests ({columns}) VALUES ({placeholders})",
                tuple(fields.values()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM requests ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode(dict(r)) for r in rows]

    def stats(self) -> dict:
        now = time.time()
        day_ago = now - 86400
        with self._lock:
            totals = self._conn.execute(
                """
                SELECT COUNT(*) AS requests,
                       SUM(cache_hit) AS cache_hits,
                       SUM(cost_usd) AS cost,
                       AVG(total_ms) AS avg_ms,
                       SUM(CASE WHEN escalated = 1 THEN 1 ELSE 0 END) AS escalations
                FROM requests
                """
            ).fetchone()
            day = self._conn.execute(
                "SELECT COUNT(*) AS requests, SUM(cost_usd) AS cost FROM requests WHERE ts >= ?",
                (day_ago,),
            ).fetchone()
            p95 = self._conn.execute(
                """
                SELECT total_ms FROM requests WHERE total_ms IS NOT NULL
                ORDER BY total_ms ASC
                LIMIT 1 OFFSET (SELECT COUNT(*) * 95 / 100 FROM requests WHERE total_ms IS NOT NULL)
                """
            ).fetchone()
            by_tier = self._conn.execute(
                """
                SELECT route_tier, COUNT(*) AS count,
                       AVG(total_ms) AS avg_ms, SUM(cost_usd) AS cost
                FROM requests WHERE route_tier IS NOT NULL GROUP BY route_tier
                """
            ).fetchall()
            by_model = self._conn.execute(
                """
                SELECT model, COUNT(*) AS count, SUM(cost_usd) AS cost,
                       AVG(prompt_tokens) AS avg_prompt, AVG(completion_tokens) AS avg_completion
                FROM requests WHERE model IS NOT NULL GROUP BY model
                """
            ).fetchall()
            series = self._conn.execute(
                """
                SELECT strftime('%Y-%m-%d %H:%M', ts, 'unixepoch') AS bucket,
                       COUNT(*) AS requests,
                       SUM(cost_usd) AS cost,
                       AVG(total_ms) AS avg_ms,
                       SUM(cache_hit) AS cache_hits
                FROM requests
                WHERE ts >= ? GROUP BY bucket ORDER BY bucket ASC
                """,
                (now - 6 * 3600,),
            ).fetchall()

        total_requests = totals["requests"] or 0
        cache_hits = totals["cache_hits"] or 0
        # Savings = what the cached requests would have cost as uncached
        # (retrieval + small-model generation) at small-model pricing.
        avg_uncached_cost = (totals["cost"] or 0) / max(1, total_requests - cache_hits) if total_requests > cache_hits else 0.0
        savings = avg_uncached_cost * cache_hits

        return {
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / total_requests, 4) if total_requests else 0.0,
            "total_cost_usd": round(totals["cost"] or 0, 6),
            "cost_saved_usd": round(savings, 6),
            "cost_last_24h_usd": round(day["cost"] or 0, 6),
            "requests_last_24h": day["requests"] or 0,
            "avg_latency_ms": round(totals["avg_ms"] or 0, 1),
            "p95_latency_ms": round(p95["total_ms"] or 0, 1) if p95 else 0.0,
            "escalations": totals["escalations"] or 0,
            "by_tier": [dict(r) for r in by_tier],
            "by_model": [dict(r) for r in by_model],
            "series": [dict(r) for r in series],
        }

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM requests")
            self._conn.commit()

    @staticmethod
    def _decode(row: dict) -> dict:
        for key in ("reasons", "context_docs"):
            if isinstance(row.get(key), str) and row[key].startswith(("[", "{")):
                try:
                    row[key] = json.loads(row[key])
                except json.JSONDecodeError:
                    pass
        return row
