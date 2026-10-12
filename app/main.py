"""FastAPI app: the RAG service with semantic caching + cost-aware routing.

Request lifecycle for POST /api/query:
  embed query ─► semantic cache lookup ─┬─ hit  ─► return cached answer (0 LLM calls)
                                        └─ miss ─► retrieve chunks ─► route (small|large)
                                                              ─► generate ─► cache ─► log
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .cache import SemanticCache
from .embeddings import build_embedder
from .llm import build_llm, compute_cost, estimate_tokens
from .metrics import MetricsDB
from .rag import build_store
from .router import CostAwareRouter

SYSTEM_PROMPT = (
    "You are a precise assistant for a documentation corpus. Answer using only "
    "the provided context. If the context is insufficient, say so plainly."
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)


class IngestRequest(BaseModel):
    text: str | None = None
    doc: str = "inline"
    path: str | None = None


class Pipeline:
    """Owns every component of the service; created once at startup."""

    def __init__(self):
        self.embedder = build_embedder()
        self.llm = build_llm(self.embedder)
        self.cache = SemanticCache()
        self.router = CostAwareRouter()
        self.store = build_store(self.embedder)
        self.db = MetricsDB()
        self.warmup_ms = 0.0

    def startup(self) -> None:
        started = time.perf_counter()
        if self.store.backend == "memory" or self.store.stats().get("chunks", 0) == 0:
            # Memory store is always empty on boot; a new Qdrant collection too.
            added = self.store.add_directory(config.DOCS_DIR)
            if added:
                print(f"[startup] ingested {added} chunks from {config.DOCS_DIR}")
        else:
            print(f"[startup] vector store already has {self.store.stats()['chunks']} chunks persisted")
        self.warmup_ms = (time.perf_counter() - started) * 1000
        llm_desc = (
            f"{config.PROVIDER} ({config.SMALL_MODEL} / {config.LARGE_MODEL})"
            if self.llm.__class__.__name__ == "LLMClient"
            else "mock (offline demo — set GROQ_API_KEY for real calls)"
        )
        print(f"[startup] embedder={self.embedder.name} llm={llm_desc}")

    async def answer(self, query: str) -> dict:
        t0 = time.perf_counter()

        # 1. Embed the query once; reuse for cache lookup and retrieval.
        t_embed = time.perf_counter()
        query_vec = self.embedder.embed([query])[0]
        embed_ms = (time.perf_counter() - t_embed) * 1000

        # 2. Semantic cache lookup — a hit skips retrieval and generation.
        t_cache = time.perf_counter()
        lookup = self.cache.lookup(query_vec)
        cache_lookup_ms = (time.perf_counter() - t_cache) * 1000

        if lookup.hit and lookup.entry is not None:
            total_ms = (time.perf_counter() - t0) * 1000
            entry = lookup.entry
            self.db.record(
                query=query,
                route_tier="cache",
                model=entry.model,
                difficulty=None,
                reasons='["semantic cache hit"]',
                forced=None,
                cache_hit=1,
                cache_similarity=round(lookup.similarity, 4),
                cache_lookup_ms=round(cache_lookup_ms, 3),
                retrieval_ms=0.0,
                generation_ms=0.0,
                total_ms=round(total_ms, 2),
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                escalated=0,
                context_docs=[],
                answer_excerpt=entry.answer[:280],
            )
            return {
                "answer": entry.answer,
                "cache": {"hit": True, "similarity": round(lookup.similarity, 4),
                          "cached_query": entry.query},
                "route": {"tier": "cache", "model": entry.model, "difficulty": None,
                          "reasons": ["semantic cache hit — no LLM call made"]},
                "usage": {"cost_usd": 0.0, "total_ms": round(total_ms, 2),
                          "embed_ms": round(embed_ms, 2),
                          "cache_lookup_ms": round(cache_lookup_ms, 2)},
                "context": [],
            }

        # 3. Retrieval (on miss only).
        t_ret = time.perf_counter()
        hits = self.store.search(query_vec, k=config.TOP_K)
        retrieval_ms = (time.perf_counter() - t_ret) * 1000
        context_chunks = [{"doc": c.doc, "text": c.text[:500], "score": round(c.score, 4)} for c in hits]

        # 4. Cost-aware routing.
        decision = self.router.route(query, retrieved_scores=[c.score for c in hits])

        # 5. Generation.
        context_block = "\n\n".join(f"[{c['doc']}] {c['text']}" for c in context_chunks)
        user_prompt = f"Context:\n{context_block or '(no documents ingested)'}\n\nQuestion: {query}"
        result = await self.llm.complete(decision.model, SYSTEM_PROMPT, user_prompt)

        escalated = False
        if (
            config.ESCALATE_ON_LOW_CONFIDENCE
            and decision.tier == "small"
            and result.refusal
            and self.router.escalate(decision) is not None
        ):
            result = await self.llm.complete(decision.model, SYSTEM_PROMPT, user_prompt)
            escalated = True

        self.router.add_spend(result.cost_usd)
        total_ms = (time.perf_counter() - t0) * 1000

        # 6. Populate cache + persist metrics.
        self.cache.store(query, query_vec, result.text, decision.model, tier=decision.tier)
        self.db.record(
            query=query,
            route_tier=decision.tier,
            model=result.model,
            difficulty=decision.difficulty,
            reasons=decision.reasons,
            forced=decision.forced,
            cache_hit=0,
            cache_similarity=round(float(lookup.similarity), 4) if lookup.similarity else None,
            cache_lookup_ms=round(cache_lookup_ms, 3),
            retrieval_ms=round(retrieval_ms, 3),
            generation_ms=round(result.latency_ms, 2),
            total_ms=round(total_ms, 2),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=round(result.cost_usd, 8),
            escalated=int(escalated),
            context_docs=[c["doc"] for c in context_chunks],
            answer_excerpt=result.text[:280],
        )

        return {
            "answer": result.text,
            "cache": {"hit": False, "similarity": round(float(lookup.similarity), 4) if lookup.similarity else 0.0},
            "route": decision.to_dict() | {"model": result.model},
            "usage": {
                "cost_usd": round(result.cost_usd, 8),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_ms": round(total_ms, 2),
                "embed_ms": round(embed_ms, 2),
                "cache_lookup_ms": round(cache_lookup_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(result.latency_ms, 2),
                "escalated": escalated,
                "budget_remaining_usd": (
                    round(config.BUDGET_USD_PER_DAY - self.router.spend_today, 4)
                    if config.BUDGET_USD_PER_DAY > 0 else None
                ),
            },
            "context": context_chunks,
        }


pipeline = Pipeline()

# Serverless (Vercel) may not run the ASGI lifespan, so startup is also
# triggered lazily from the request handlers. Idempotent + thread-safe.
_started = False
_start_lock = __import__("threading").Lock()


def ensure_started() -> None:
    global _started
    if _started:
        return
    with _start_lock:
        if not _started:
            pipeline.startup()
            _started = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_started()
    yield


app = FastAPI(title="Semantic Cache + Cost-Aware RAG Router", lifespan=lifespan)


@app.post("/api/query")
async def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    ensure_started()
    try:
        return await pipeline.answer(req.query.strip())
    except Exception as exc:  # surface provider errors as a clean 502
        raise HTTPException(status_code=502, detail=f"generation failed: {exc}") from exc


@app.post("/api/ingest")
def ingest(req: IngestRequest):
    ensure_started()
    if req.path:
        added = pipeline.store.add_directory(Path(req.path))
        if added == 0:
            raise HTTPException(status_code=404, detail=f"no documents found under {req.path}")
    elif req.text:
        added = pipeline.store.add_text(req.text, doc=req.doc)
    else:
        raise HTTPException(status_code=422, detail="provide text or path")
    return {"chunks_added": added, "store": pipeline.store.stats()}


@app.get("/api/stats")
def stats():
    ensure_started()
    return {
        "stats": pipeline.db.stats(),
        "cache": pipeline.cache.stats(),
        "retrieval": pipeline.store.stats(),
        "routing": {
            "small_model": config.SMALL_MODEL,
            "large_model": config.LARGE_MODEL,
            "threshold": config.ROUTE_THRESHOLD,
            "spend_today_usd": round(pipeline.router.spend_today, 6),
            "budget_usd_per_day": config.BUDGET_USD_PER_DAY or None,
            "budget_exhausted": pipeline.router.budget_exhausted(),
        },
        "provider": {
            "llm": f"{config.PROVIDER}:{config.LARGE_MODEL}" if pipeline.llm.__class__.__name__ == "LLMClient" else "mock (offline demo)",
            "embeddings": pipeline.embedder.name,
            "vector_store": pipeline.store.backend,
            "api_key_configured": bool(config.LLM_API_KEY),
        },
    }


@app.get("/api/requests")
def requests(limit: int = 50):
    ensure_started()
    return {"requests": pipeline.db.recent(limit=min(max(limit, 1), 200))}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
