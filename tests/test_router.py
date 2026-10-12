import pytest

from app.router import CostAwareRouter, score_difficulty


def test_shallow_query_is_easy():
    d = score_difficulty("What is RAG?")
    assert d.difficulty < 0.4
    assert d.tier == "small"


def test_deep_query_scores_higher_than_shallow():
    shallow = score_difficulty("What is semantic caching?").difficulty
    deep = score_difficulty(
        "Why does plain RAG fail on multi-hop questions, and how would you design "
        "a system that fixes it while trading off latency and cost?"
    ).difficulty
    assert deep > shallow
    assert deep > 0.55


def test_multi_part_question_adds_signal():
    d = score_difficulty("What is RAG? How does chunking work? When should I rerank?")
    assert any("multi-part" in r for r in d.reasons)


def test_difficulty_is_bounded():
    d = score_difficulty("why " * 200 + " compare and analyze and design and plan " * 50)
    assert 0.0 <= d.difficulty <= 1.0


def test_router_routes_small():
    router = CostAwareRouter(small_model="small-x", large_model="large-y", threshold=0.55)
    decision = router.route("What is Redis?")
    assert decision.tier == "small"
    assert decision.model == "small-x"


def test_router_routes_large():
    router = CostAwareRouter(small_model="small-x", large_model="large-y", threshold=0.55)
    decision = router.route(
        "Compare HNSW and IVF-PQ trade-offs and explain how you would design a "
        "hybrid retriever that handles both latency and recall constraints?"
    )
    assert decision.tier == "large"
    assert decision.model == "large-y"
    assert decision.difficulty >= 0.55


def test_budget_forces_downshift():
    router = CostAwareRouter(small_model="small-x", large_model="large-y", threshold=0.55, budget_usd_per_day=0.01)
    router.add_spend(0.05)
    assert router.budget_exhausted()
    decision = router.route("Explain and compare and design a full architecture plan?")
    assert decision.tier == "small"
    assert decision.forced == "budget"


def test_escalation_promotes_small_to_large():
    router = CostAwareRouter(small_model="small-x", large_model="large-y", threshold=0.99)
    decision = router.route("what is a cache")
    assert decision.tier == "small"
    escalated = router.escalate(decision)
    assert escalated is not None
    assert decision.model == "large-y"
    assert decision.forced == "escalation"


def test_escalation_noop_on_large():
    router = CostAwareRouter(small_model="small-x", large_model="large-y", threshold=0.0)
    decision = router.route("compare and design and analyze everything")
    assert decision.tier == "large"
    assert router.escalate(decision) is None
