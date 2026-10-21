"""In-memory metrics log: pure-Python aggregation + JSONL persistence."""
import json
import time

from app.metrics import MetricsLog

NOW = 1_700_000_000.0  # fixed clock so window math is deterministic


def test_recent_is_newest_first_and_decodes_reasons(tmp_path):
    log = MetricsLog(path=tmp_path / "requests.jsonl")
    log.record(query="q1", ts=10.0, route_tier="small", model="m", cache_hit=0,
               cost_usd=0.001, total_ms=42.0, reasons=["route a"])
    log.record(query="q2", ts=20.0, route_tier="cache", model="m", cache_hit=1,
               cost_usd=0.0, total_ms=5.0, reasons='["semantic cache hit"]')

    rows = log.recent(10)
    assert [r["query"] for r in rows] == ["q2", "q1"]  # newest first
    assert [r["id"] for r in rows] == [2, 1]           # stable ids
    assert rows[0]["reasons"] == ["semantic cache hit"]  # JSON string decoded
    assert rows[1]["reasons"] == ["route a"]             # list kept as-is
    assert all(isinstance(r["ts"], float) for r in rows)


def test_stats_aggregates_match_hand_computed(tmp_path):
    log = MetricsLog(path=tmp_path / "requests.jsonl")
    # oldest row falls outside both the 24h and the 6h windows
    log.record(query="old", ts=NOW - 7 * 86400, route_tier="small", model="small-model",
               cache_hit=0, cost_usd=0.5, total_ms=500.0, prompt_tokens=30,
               completion_tokens=10, escalated=0)
    log.record(query="hit", ts=NOW - 300, route_tier="cache", model="small-model",
               cache_hit=1, cost_usd=0.0, total_ms=100.0, prompt_tokens=0,
               completion_tokens=0, escalated=0)
    log.record(query="miss", ts=NOW - 200, route_tier="small", model="small-model",
               cache_hit=0, cost_usd=0.001, total_ms=200.0, prompt_tokens=10,
               completion_tokens=5, escalated=0)
    log.record(query="big", ts=NOW - 100, route_tier="large", model="large-model",
               cache_hit=0, cost_usd=0.002, total_ms=3000.0, prompt_tokens=20,
               completion_tokens=40, escalated=1)

    s = log.stats(now=NOW)
    assert s["total_requests"] == 4
    assert s["cache_hits"] == 1
    assert s["cache_hit_rate"] == 0.25
    assert s["total_cost_usd"] == 0.503
    # savings = observed average uncached cost (0.503 / 3) x 1 cached request
    assert s["cost_saved_usd"] == round(0.503 / 3, 6)
    assert s["requests_last_24h"] == 3 and s["cost_last_24h_usd"] == 0.003
    # latencies 100/200/3000/500 -> p95 index = 4*95//100 = 3 -> 3000
    assert s["p95_latency_ms"] == 3000.0
    assert s["avg_latency_ms"] == 950.0
    assert s["escalations"] == 1

    small = next(g for g in s["by_tier"] if g["route_tier"] == "small")
    assert small["count"] == 2
    assert small["avg_ms"] == 350.0
    assert abs(small["cost"] - 0.501) < 1e-12
    assert {g["route_tier"] for g in s["by_tier"]} == {"small", "cache", "large"}

    large_model = next(g for g in s["by_model"] if g["model"] == "large-model")
    assert large_model == {"model": "large-model", "count": 1, "cost": 0.002,
                           "avg_prompt": 20.0, "avg_completion": 40.0}
    small_model = next(g for g in s["by_model"] if g["model"] == "small-model")
    assert small_model["count"] == 3
    assert abs(small_model["avg_prompt"] - 40 / 3) < 1e-9
    assert small_model["avg_completion"] == 5.0

    # 6h series: only the three recent rows, one UTC-minute bucket each, ascending
    expected = [time.strftime("%Y-%m-%d %H:%M", time.gmtime(NOW - off)) for off in (300, 200, 100)]
    assert [b["bucket"] for b in s["series"]] == expected
    assert all(b["requests"] == 1 for b in s["series"])
    assert sum(b["cache_hits"] for b in s["series"]) == 1
    assert s["series"][0]["avg_ms"] == 100.0


def test_empty_log_reports_zeroes(tmp_path):
    s = MetricsLog(path=tmp_path / "fresh.jsonl").stats()
    assert s["total_requests"] == 0
    assert s["cache_hit_rate"] == 0.0
    assert s["total_cost_usd"] == 0.0
    assert s["p95_latency_ms"] == 0.0
    assert s["by_tier"] == [] and s["by_model"] == [] and s["series"] == []


def test_jsonl_persistence_roundtrip_and_clear(tmp_path):
    path = tmp_path / "requests.jsonl"
    first = MetricsLog(path=path)
    first.record(query="q ünïcode", ts=1.0, route_tier="small", model="m",
                 cache_hit=0, cost_usd=0.001, total_ms=42.0, reasons=["r"],
                 context_docs=["doc.md"])
    first.record(query="q2", ts=2.0, route_tier="cache", model="m", cache_hit=1,
                 cost_usd=0.0, total_ms=5.0)

    reloaded = MetricsLog(path=path)  # new process, same file
    assert reloaded.stats(now=3600.0) == first.stats(now=3600.0)
    assert path.read_text(encoding="utf-8").count("\n") == 2
    assert reloaded.recent(1)[0]["query"] == "q2"

    reloaded.clear()
    assert reloaded.stats()["total_requests"] == 0
    assert path.stat().st_size == 0  # clear wipes the log, not just memory
    assert MetricsLog(path=path).stats()["total_requests"] == 0


def test_ring_buffer_bounds_memory_and_file(tmp_path):
    path = tmp_path / "requests.jsonl"
    log = MetricsLog(path=path, maxlen=5)
    for i in range(8):
        log.record(query=f"q{i}", ts=float(i + 1), route_tier="small", model="m",
                   cache_hit=0, cost_usd=0.0, total_ms=1.0)

    assert [r["query"] for r in log.recent(50)] == ["q7", "q6", "q5", "q4", "q3"]

    # a fresh load keeps the newest `maxlen` and rewrites the file to match
    reloaded = MetricsLog(path=path, maxlen=5)
    assert [r["query"] for r in reloaded.recent(50)] == ["q7", "q6", "q5", "q4", "q3"]
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 5
    assert json.loads(lines[-1])["query"] == "q7"


def test_record_never_raises_on_odd_values(tmp_path):
    """Metrics must not be able to fail a request."""
    log = MetricsLog(path=tmp_path / "odd.jsonl")
    row_id = log.record(query="odd", ts=10.0, weird=object())
    assert isinstance(row_id, int)
    assert isinstance(log.recent(1)[0]["weird"], str)  # coerced, not dropped


def test_write_failure_degrades_to_memory(tmp_path):
    """Unwritable log path (read-only filesystem) -> still records in memory."""
    blocker = tmp_path / "not-a-file"
    blocker.mkdir()  # reading/opening a directory always fails on every OS
    log = MetricsLog(path=blocker)
    log.record(query="serverless", ts=10.0, route_tier="small", model="m",
               cache_hit=0, cost_usd=0.001, total_ms=7.0)
    s = log.stats()
    assert s["total_requests"] == 1
    assert log.recent(1)[0]["query"] == "serverless"
