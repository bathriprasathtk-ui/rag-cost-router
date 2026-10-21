"""End-to-end pipeline tests: cache behavior, routing, and metrics recording."""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
from app.main import app, pipeline  # noqa: E402
from app.router import score_difficulty  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # startup() runs via the ASGI lifespan -> ensure_started()
    pipeline.cache.clear()
    pipeline.db.clear()
    with TestClient(app) as c:
        yield c


def test_stats_endpoint(client):
    res = client.get("/api/stats")
    assert res.status_code == 200
    body = res.json()
    assert "stats" in body and "cache" in body and "routing" in body
    assert body["routing"]["small_model"] == config.SMALL_MODEL


def test_shallow_query_routes_small(client):
    res = client.post("/api/query", json={"query": "What is semantic caching?"})
    assert res.status_code == 200
    body = res.json()
    assert body["cache"]["hit"] is False
    assert body["route"]["tier"] in {"small", "large"}
    assert body["usage"]["cost_usd"] > 0
    assert body["answer"]


def test_repeat_query_is_cache_hit_and_free(client):
    q = "What is semantic caching?"
    res = client.post("/api/query", json={"query": q})
    body = res.json()
    assert body["cache"]["hit"] is True
    assert body["route"]["tier"] == "cache"
    assert body["usage"]["cost_usd"] == 0.0
    assert "no LLM call" in " ".join(body["route"]["reasons"])


def test_ingest_and_retrieval_returns_context(client):
    res = client.post("/api/query", json={"query": "How do HNSW indexes work in vector databases?"})
    body = res.json()
    assert body["cache"]["hit"] is False
    assert isinstance(body["context"], list)
    assert len(body["context"]) > 0
    assert body["context"][0]["score"] > 0


def test_metrics_recorded(client):
    res = client.get("/api/requests?limit=20")
    rows = res.json()["requests"]
    assert len(rows) >= 2
    assert any(r["cache_hit"] == 1 for r in rows)
    assert any(r["cache_hit"] == 0 and r["cost_usd"] > 0 for r in rows)


def test_stats_aggregates(client):
    stats = client.get("/api/stats").json()["stats"]
    assert stats["total_requests"] >= 2
    assert 0.0 <= stats["cache_hit_rate"] <= 1.0
    assert stats["p95_latency_ms"] >= 0
    assert stats["by_tier"], "routing tiers should be aggregated"


def test_empty_query_rejected(client):
    assert client.post("/api/query", json={"query": ""}).status_code == 422


def test_mock_small_punts_below_router_threshold():
    """Offline escalation must be reachable: the mock small model punts on
    mid-difficulty queries that the router still optimistically sends to it.
    """
    from app.llm import MockLLM

    q = (
        "Why do embedding models normalize vectors, and how does that change "
        "similarity scores?"
    )
    difficulty = score_difficulty(q).difficulty
    assert config.ROUTE_THRESHOLD - 0.2 <= difficulty < config.ROUTE_THRESHOLD
    assert MockLLM._is_hard(q), "mock small model should refuse this"


def test_static_assets_and_index_revalidate(client):
    """Assets must send Cache-Control: no-cache — heuristic caching is what
    left browsers running stale dashboard code after UI updates."""
    for path in ("/", "/static/app.js", "/static/style.css"):
        res = client.get(path)
        assert res.status_code == 200, path
        assert res.headers.get("cache-control") == "no-cache", path


def test_path_ingest_can_be_disabled(client, monkeypatch):
    """ALLOW_PATH_INGEST=false must block filesystem ingestion (403) while
    inline text ingestion keeps working."""
    from app import config as cfg

    monkeypatch.setattr(cfg, "ALLOW_PATH_INGEST", False)
    res = client.post("/api/ingest", json={"path": "/etc"})
    assert res.status_code == 403
    assert "disabled" in res.json()["detail"]

    res = client.post("/api/ingest", json={"text": "ingest gating check", "doc": "t"})
    assert res.status_code == 200


def test_small_model_refusal_escalates_and_is_flagged(client, monkeypatch):
    from app.llm import MockLLM

    # Force the small tier to refuse; the query itself is shallow, so it
    # still routes small and the escalation branch is what promotes it.
    monkeypatch.setattr(MockLLM, "_is_hard", staticmethod(lambda question: True))

    q = "What is a bloom filter in vector database terms?"
    res = client.post("/api/query", json={"query": q})
    assert res.status_code == 200
    body = res.json()
    assert body["cache"]["hit"] is False
    assert body["route"]["tier"] == "large"
    assert body["route"]["forced"] == "escalation"
    assert body["route"]["model"] == config.LARGE_MODEL
    assert body["usage"]["escalated"] is True
    assert body["usage"]["cost_usd"] > 0
    assert any("escalated" in r for r in body["route"]["reasons"])

    rows = client.get("/api/requests?limit=20").json()["requests"]
    assert any(r["query"] == q and r["escalated"] == 1 for r in rows)
