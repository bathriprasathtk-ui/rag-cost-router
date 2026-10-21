"""Request-level metrics: an in-memory log with pure-Python aggregation.

Every request appends one record to a bounded list; the dashboard numbers
(totals, hit rate, p95, per-tier/per-model rollups, the 6h time series) are
computed with plain Python — grouping, percentile and time buckets, no SQL and
no database engine to install, run or keep in sync with the app.

Durability: records are appended as JSON lines (``METRICS_PATH``, default
``data/requests.jsonl``) and reloaded on boot, so restarts keep the history.
File writes are best-effort — on a read-only filesystem (serverless) the log
degrades to memory-only instead of failing the request.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from . import config

MAX_REQUESTS = 20_000        # in-memory cap; older records are dropped first
_COMPACT_EVERY = 500         # rewrite the file after this many drops
_SERIES_WINDOW_S = 6 * 3600  # dashboard "last 6h" chart
_JSON_PREFIXES = ("[", "{")


def _json_default(obj: Any):
    """Make numpy scalars (embeddings/router math) and odd values JSON-safe."""
    item = getattr(obj, "item", None)
    if callable(item):
        return item()
    return str(obj)


def _decode(value):
    """Decode JSON-encoded strings so API responses match what was logged."""
    if isinstance(value, str) and value.startswith(_JSON_PREFIXES):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _row_id(row: dict) -> int:
    try:
        return int(row.get("id") or 0)
    except (TypeError, ValueError):
        return 0


def _normalize(fields: dict) -> dict:
    """Produce a JSON-safe record: timestamped, decoded, native types."""
    row = dict(fields)
    row.setdefault("ts", time.time())
    for key in ("reasons", "context_docs"):
        if key in row:
            row[key] = _decode(row[key])
    # Round-trip so the stored record, the file line and the API response are
    # all the same JSON-native value (float, int, str, list, None).
    return json.loads(json.dumps(row, default=_json_default))


class MetricsLog:
    """Bounded, thread-safe request history + dashboard aggregates."""

    def __init__(self, path: Path = config.METRICS_PATH, maxlen: int = MAX_REQUESTS):
        self.path = Path(path)
        self.maxlen = maxlen
        self._lock = threading.Lock()
        self._rows: list[dict] = []
        self._fh = None            # append handle, opened lazily
        self._persist = True        # flipped off after the first write error
        self._dropped_since_compaction = 0
        self._seq = 0
        self._load()

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        """Read the JSONL log, oldest first, keeping the newest `maxlen`."""
        if self.path.exists():
            try:
                lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError as exc:
                print(f"[metrics] cannot read {self.path} ({exc}); starting empty")
                lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn trailing line from an interrupted write
                if isinstance(row, dict) and "ts" in row:
                    self._rows.append(_normalize(row))
        if not self._rows:
            return
        self._rows.sort(key=lambda r: r.get("ts") or 0.0)
        if len(self._rows) > self.maxlen:
            del self._rows[: -self.maxlen]
            self._rewrite()  # bound the file too, not just memory
        self._seq = max((_row_id(r) for r in self._rows), default=0)

    def _append_line(self, row: dict) -> None:
        """Append one JSON line; on failure keep serving from memory."""
        if not self._persist:
            return
        try:
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(self.path, "a", encoding="utf-8")
            self._fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            self._fh.flush()
        except (OSError, TypeError, ValueError) as exc:
            print(f"[metrics] file log disabled ({exc}); continuing in memory")
            self._persist = False
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None

    def _rewrite(self, rows: Iterable[dict] | None = None) -> None:
        """Rewrite the whole log from memory (boot trim + periodic compaction)."""
        rows = list(self._rows if rows is None else rows)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            tmp.replace(self.path)  # atomic: never leaves a half-written log
        except (OSError, TypeError, ValueError) as exc:
            print(f"[metrics] could not rewrite {self.path} ({exc}); append-only from here")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #
    def record(self, **fields) -> int:
        """Log one request. Never raises: metrics must not break a query."""
        row = _normalize(fields)
        with self._lock:
            self._seq += 1
            row.setdefault("id", self._seq)
            self._rows.append(row)
            while len(self._rows) > self.maxlen:
                self._rows.pop(0)
                self._dropped_since_compaction += 1
            if self._dropped_since_compaction >= _COMPACT_EVERY:
                self._dropped_since_compaction = 0
                snapshot = list(self._rows)
            else:
                snapshot = None
            self._append_line(row)
            if snapshot is not None:
                self._rewrite(snapshot)
            return int(row["id"])

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()
            self._seq = 0
            self._dropped_since_compaction = 0
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None
            try:
                if self.path.exists():
                    self.path.write_text("", encoding="utf-8")
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    # reads (pure Python — no SQL anywhere below)
    # ------------------------------------------------------------------ #
    def recent(self, limit: int = 50) -> list[dict]:
        """Newest-first slice of the log, matching the /api/requests shape."""
        with self._lock:
            ordered = sorted(self._rows, key=lambda r: r.get("ts") or 0.0, reverse=True)
        return [dict(r) for r in ordered[: max(limit, 0)]]

    def stats(self, now: float | None = None) -> dict:
        """All dashboard aggregates in one pass over the in-memory log."""
        now = time.time() if now is None else now
        with self._lock:
            rows = list(self._rows)

        total_requests = len(rows)
        cache_hits = sum(int(r.get("cache_hit") or 0) for r in rows)
        cost = sum(float(r.get("cost_usd") or 0.0) for r in rows)
        latencies = [float(r["total_ms"]) for r in rows if r.get("total_ms") is not None]
        escalations = sum(1 for r in rows if int(r.get("escalated") or 0) == 1)

        day_rows = [r for r in rows if float(r.get("ts") or 0.0) >= now - 86400]
        day_cost = sum(float(r.get("cost_usd") or 0.0) for r in day_rows)

        # p95: same index rule as the old SQL OFFSET n*95/100 (floor division).
        p95 = sorted(latencies)[(len(latencies) * 95) // 100] if latencies else 0.0

        # Savings = what cached requests would have cost uncached, priced at
        # the observed average uncached cost.
        avg_uncached = (cost / max(1, total_requests - cache_hits)) if total_requests > cache_hits else 0.0
        savings = avg_uncached * cache_hits

        by_tier: dict[str, dict] = {}
        by_model: dict[str, dict] = {}
        for r in rows:
            tier = r.get("route_tier")
            if tier is not None:
                g = by_tier.setdefault(tier, {"route_tier": tier, "count": 0, "_ms": [], "cost": 0.0})
                g["count"] += 1
                if r.get("total_ms") is not None:
                    g["_ms"].append(float(r["total_ms"]))
                g["cost"] += float(r.get("cost_usd") or 0.0)
            model = r.get("model")
            if model is not None:
                g = by_model.setdefault(
                    model,
                    {"model": model, "count": 0, "cost": 0.0, "_prompt": [], "_completion": []},
                )
                g["count"] += 1
                g["cost"] += float(r.get("cost_usd") or 0.0)
                if r.get("prompt_tokens") is not None:
                    g["_prompt"].append(float(r["prompt_tokens"]))
                if r.get("completion_tokens") is not None:
                    g["_completion"].append(float(r["completion_tokens"]))

        # 6h series: one bucket per UTC minute, only non-empty ones (what the
        # chart has always received), ordered oldest -> newest.
        cutoff = now - _SERIES_WINDOW_S
        buckets: dict[str, dict] = {}
        for r in rows:
            ts = float(r.get("ts") or 0.0)
            if ts < cutoff:
                continue
            key = time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts))
            b = buckets.setdefault(
                key,
                {"bucket": key, "requests": 0, "cost": 0.0, "_ms": [], "cache_hits": 0},
            )
            b["requests"] += 1
            b["cost"] += float(r.get("cost_usd") or 0.0)
            b["cache_hits"] += int(r.get("cache_hit") or 0)
            if r.get("total_ms") is not None:
                b["_ms"].append(float(r["total_ms"]))
        series = []
        for key in sorted(buckets):
            b = buckets[key]
            ms = b.pop("_ms")
            b["avg_ms"] = (sum(ms) / len(ms)) if ms else None
            series.append(b)

        return {
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / total_requests, 4) if total_requests else 0.0,
            "total_cost_usd": round(cost, 6),
            "cost_saved_usd": round(savings, 6),
            "cost_last_24h_usd": round(day_cost, 6),
            "requests_last_24h": len(day_rows),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "p95_latency_ms": round(p95, 1),
            "escalations": escalations,
            "by_tier": self._tier_rows(by_tier),
            "by_model": self._model_rows(by_model),
            "series": series,
        }

    @staticmethod
    def _tier_rows(groups: dict[str, dict]) -> list[dict]:
        return [
            {
                "route_tier": g["route_tier"],
                "count": g["count"],
                "avg_ms": (sum(g["_ms"]) / len(g["_ms"])) if g["_ms"] else None,
                "cost": g["cost"],
            }
            for g in groups.values()
        ]

    @staticmethod
    def _model_rows(groups: dict[str, dict]) -> list[dict]:
        return [
            {
                "model": g["model"],
                "count": g["count"],
                "cost": g["cost"],
                "avg_prompt": (sum(g["_prompt"]) / len(g["_prompt"])) if g["_prompt"] else None,
                "avg_completion": (sum(g["_completion"]) / len(g["_completion"])) if g["_completion"] else None,
            }
            for g in groups.values()
        ]
