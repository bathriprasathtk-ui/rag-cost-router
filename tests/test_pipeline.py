"""End-to-end pipeline tests: cache behavior, routing, and metrics recording."""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
from app.main import app, pipeline  # noqa: E402


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
