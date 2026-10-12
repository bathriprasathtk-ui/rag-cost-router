# Semantic Caching + Cost-Aware RAG Router

A local-first RAG service that decides, **before** spending money on an LLM,
whether it needs to call a model at all — and if so, which model.

```
query ─► embed ─► semantic cache ─┬─ hit  ─► cached answer (0 LLM calls, $0)
                                  └─ miss ─► retrieve top-k ─► difficulty score
                                                └─ small model ◄─► large model
                                                        └─ answer ─► cache ─► metrics
```

Built as a resume project from the "three RAG projects that stand out" list:
**semantic caching + cost-aware routing with a cost/latency dashboard.**

## Stack

| Component | Implementation |
|---|---|
| **LLM (free tier)** | Groq API — `llama-3.1-8b-instant` (small) vs `llama-3.3-70b-versatile` (large). Any OpenAI-compatible endpoint works (OpenAI, Together, vLLM...). |
| **Embeddings** | Local `sentence-transformers/all-MiniLM-L6-v2` — true semantic similarity, free, offline after first download. |
| **Vector store** | Qdrant in Docker (persistent across restarts), automatic in-memory fallback when Docker isn't running. |
| **Semantic cache** | Cosine similarity over query embeddings with tunable threshold, TTL expiry, and LRU cap. Hits skip retrieval *and* generation. |
| **Cost-aware router** | Scores query difficulty (0–1) from lexical depth, structure, entity count, and retrieval spread; routes at a configurable threshold. |
| **Escalation** | Small-model refusal → automatic retry on the large model, flagged in metrics — the safety net behind optimistic routing. |
| **Budget guard** | Optional daily spend cap downshifts all traffic to the small model when reached. |
| **Metrics** | Every request logged to SQLite: tier, model, difficulty, reasons, cache hit + similarity, per-stage latency, tokens, cost. |
| **Dashboard** | Live hit rate, spend, p95 latency, cost/latency time series, routing split, per-request log with routing *reasons*. |

## Quickstart (local, full stack)

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
pip install -r requirements-local.txt   # adds sentence-transformers + qdrant-client

# 1. Get a FREE Groq key (no card): https://console.groq.com/keys
#    then in .env:  GROQ_API_KEY=gsk_...

# 2. Start the vector store — pick ONE:
.\start-qdrant.ps1       # native Qdrant binary (auto-downloads to tools\qdrant, no Docker needed)
# or, with Docker/WSL available:  docker compose up -d qdrant

# 3. Start the app:
.\start-app.ps1          # or: uvicorn app.main:app --port 8000
```

## Deploy to Vercel

The repo is Vercel-ready — `vercel.json` rewrites every route to the
serverless ASGI function in `api/index.py`, with these cloud defaults:

| Env | Value | Why |
|---|---|---|
| `VECTOR_STORE` | `memory` | no Qdrant server in serverless; corpus re-ingests on cold start |
| `EMBEDDING_PROVIDER` | `hashing` | PyTorch exceeds Vercel's bundle limit; hashing needs zero deps |
| `DB_PATH` | `/tmp/metrics.db` | the repo dir is read-only in functions; `/tmp` is writable |

`GROQ_API_KEY` is read from the environment at runtime — add it in the
Vercel dashboard (Settings → Environment Variables). Without it the app
deploys fine and answers in mock mode.

Steps:

1. Push this repo to GitHub, then import it at https://vercel.com/new
2. Add `GROQ_API_KEY` under Environment Variables (Production + Preview + Development)
3. Deploy — you get a URL like `rag-router-xyz.vercel.app`

Notes for the deployed instance:

- **Embeddings are lexical on Vercel** (`hashing`): the cache still catches
  wording-level repeats, but true paraphrase matching needs the local setup
  (or set `EMBEDDING_PROVIDER=openai` + `OPENAI_API_KEY` — a plain HTTP call,
  fine in serverless).
- Metrics/cache live per warm instance and reset on cold starts — expected
  for a serverless demo; the numbers on the dashboard are still real.
- `docker-compose.yml`, `start-*.ps1`, and Qdrant are only used locally.

Open <http://127.0.0.1:8000>.

Without a key the app still runs (offline mock LLM, hashing embedder, in-memory
store) — everything works for demos and tests; only generation is simulated.

### Provider presets (`LLM_PROVIDER`)

| Value | Small model | Large model | Cost |
|---|---|---|---|
| `groq` (default with `GROQ_API_KEY`) | llama-3.1-8b-instant | llama-3.3-70b-versatile | free tier |
| `openai` (with `OPENAI_API_KEY`) | gpt-4o-mini | gpt-4o | usage-based |
| `mock` | — | — | $0, no key |

Override with `SMALL_MODEL` / `LARGE_MODEL`; point at any OpenAI-compatible
endpoint with `LLM_BASE_URL`.

### Try the routing live

| Prompt | Expected behavior |
|---|---|
| `What is semantic caching?` | difficulty < threshold → **small model** |
| Ask it again (or hit the chip) | **cache hit**, $0, no LLM call |
| `Why does plain RAG fail on multi-hop questions, and how would you design a system to fix that?` | difficulty ≥ threshold → **large model** |

## API

```
POST /api/query     {query} -> {answer, route, cache, usage, context}
POST /api/ingest    {text, doc} | {path}
GET  /api/stats     aggregates: cost, hit rate, p95, tiers, models, time series
GET  /api/requests  per-request log with routing reasons
GET  /api/health
```

Example response:

```json
{
  "answer": "...",
  "route": {"tier": "large", "model": "llama-3.3-70b-versatile", "difficulty": 0.71,
            "reasons": ["depth keywords: why, how", "multi-part question", "context spread across several chunks"]},
  "cache": {"hit": false, "similarity": 0.42},
  "usage": {"cost_usd": 0.00031, "total_ms": 640.2, "retrieval_ms": 3.1,
            "generation_ms": 520.4, "escalated": false}
}
```

## Tests

```bash
pytest -q
```

Covers cache hit/miss/TTL/LRU behavior, difficulty scoring and routing rules,
budget downshift, escalation, and end-to-end pipeline + metrics aggregation.

## Configuration

All via environment variables — see `.env.example`. Key knobs:

- `CACHE_THRESHOLD` (0.94) — raise for precision, lower for hit rate
- `ROUTE_THRESHOLD` (0.55) — the small/large difficulty cutoff
- `BUDGET_USD_PER_DAY` (off) — enables budget downshifting
- `ESCALATE_ON_LOW_CONFIDENCE` (true) — small→large fallback on refusals
- `VECTOR_STORE` (`auto`) — `qdrant` to force, `memory` to disable

## Project layout

```
app/
  config.py      provider presets, env-driven settings + pricing table
  embeddings.py  pluggable embedders (sentence-transformers | openai | hashing)
  rag.py         chunking + Qdrant/memory vector stores
  router.py      difficulty scoring + cost-aware routing + escalation
  cache.py       semantic response cache (threshold, TTL, LRU)
  llm.py         OpenAI-compatible client, mock provider, cost math
  metrics.py     SQLite request log + dashboard aggregates
  main.py        FastAPI wiring / request lifecycle
api/index.py     Vercel serverless entry (ASGI export)
vercel.json      rewrites + cloud env defaults
static/          dashboard (Chart.js)
sample_data/     demo corpus (auto-ingested at startup)
tests/           unit + end-to-end tests (offline: mock LLM, memory store)
tools/qdrant/    native Qdrant binary (fetched by start-qdrant.ps1)
docker-compose.yml  Qdrant, for Docker/WSL setups
start-qdrant.ps1    launch native Qdrant with local storage
start-app.ps1       launch the API + dashboard
requirements.txt       core deps (Vercel-safe, no PyTorch)
requirements-local.txt  + sentence-transformers, qdrant-client, pytest
```

## Talking about it in interviews

- **Hit rate vs. threshold**: a precision/recall trade-off measured on answers,
  not just embeddings — per-task thresholds are the answer.
- **Why route before calling?** Difficulty scoring costs ~0ms and $0; every
  wrong routing decision costs a full generation.
- **Escalation rate** is the router's feedback loop: too high → lower the
  threshold, too low → raise it and re-measure quality.
- **Savings are computed against a baseline** (all-frontier-model, no cache),
  which is what makes the number defensible.

## Resume line

> Added semantic caching and model routing to a RAG service, reducing LLM cost
> and p95 latency while maintaining answer quality.
