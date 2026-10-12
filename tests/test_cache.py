import numpy as np
import pytest

from app.cache import SemanticCache
from app.embeddings import HashingEmbedder


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=512)


def test_identical_query_hits(embedder):
    cache = SemanticCache(threshold=0.94)
    q = "What is retrieval augmented generation?"
    vec = embedder.embed([q])[0]
    cache.store(q, vec, "answer-1", "small")

    hit = cache.lookup(embedder.embed([q])[0])
    assert hit.hit
    assert hit.similarity == pytest.approx(1.0, abs=1e-5)
    assert hit.entry.answer == "answer-1"
    assert cache.hits == 1


def test_paraphrase_hits_at_moderate_threshold(embedder):
    cache = SemanticCache(threshold=0.75)
    vec = embedder.embed(["what is semantic caching"])[0]
    cache.store("what is semantic caching", vec, "cached", "small")

    hit = cache.lookup(embedder.embed(["what is a semantic cache"])[0])
    assert hit.hit, "paraphrases sharing wording should be cache hits"
    assert hit.entry.answer == "cached"


def test_unrelated_query_misses(embedder):
    cache = SemanticCache(threshold=0.94)
    vec = embedder.embed(["what is semantic caching"])[0]
    cache.store("what is semantic caching", vec, "cached", "small")

    hit = cache.lookup(embedder.embed(["how do I configure nginx TLS certificates"])[0])
    assert not hit.hit
    assert cache.misses == 1


def test_ttl_expiry(embedder):
    cache = SemanticCache(threshold=0.5, ttl_seconds=0)
    vec = embedder.embed(["hello"])[0]
    cache.store("hello", vec, "answer", "small")

    import time

    time.sleep(0.01)
    hit = cache.lookup(embedder.embed(["hello"])[0])
    assert not hit.hit


def test_lru_overflow_bound(embedder):
    cache = SemanticCache(threshold=0.99, max_entries=3)
    for i in range(10):
        vec = embedder.embed([f"query number {i}"])[0]
        cache.store(f"query number {i}", vec, f"a{i}", "small")
    assert len(cache) <= 3


def test_hit_rate(embedder):
    cache = SemanticCache(threshold=0.9)
    qv = embedder.embed(["what is RAG"])[0]
    cache.store("what is RAG", qv, "a", "small")
    cache.lookup(qv)
    cache.lookup(embedder.embed(["completely different topic here"])[0])
    assert cache.stats()["hit_rate"] == 0.5
