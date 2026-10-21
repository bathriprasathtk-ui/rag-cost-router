"""Central configuration, loaded from environment variables (optionally .env).

Provider presets keep .env tiny:
    LLM_PROVIDER=groq     -> free tier, openai/gpt-oss-20b + openai/gpt-oss-120b
    LLM_PROVIDER=openai   -> gpt-4o-mini + gpt-4o
    LLM_PROVIDER=mock     -> offline demo, no key required
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional dependency-free .env loading
    from dotenv import load_dotenv  # type: ignore

    # Always prefer the project-root .env, no matter which directory the
    # server was launched from — otherwise `uvicorn` started elsewhere misses
    # the key and the app silently falls back to mock mode.
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
    else:
        load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


BASE_DIR = Path(__file__).resolve().parent.parent

# --- LLM provider -----------------------------------------------------------
# auto: groq if GROQ_API_KEY, openai if OPENAI_API_KEY, else mock
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

# Model defaults per provider; explicit SMALL_MODEL/LARGE_MODEL always win.
PROVIDER_PRESETS: dict[str, dict] = {
    "groq": {
        # Groq retired the llama-3.x fleet; the gpt-oss pair is what its
        # free tier serves now (verified via /openai/v1/models, Oct 2026).
        "small": "openai/gpt-oss-20b",
        "large": "openai/gpt-oss-120b",
        "base_url": GROQ_BASE_URL,
        "api_key": GROQ_API_KEY,
    },
    "openai": {
        "small": "gpt-4o-mini",
        "large": "gpt-4o",
        "base_url": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
    },
    "mock": {
        "small": "mock-small",
        "large": "mock-large",
        "base_url": "",
        "api_key": "",
    },
}


def resolve_provider() -> str:
    """Pick the active provider: explicit setting, else first with a key, else mock."""
    if LLM_PROVIDER in PROVIDER_PRESETS and LLM_PROVIDER != "auto":
        return LLM_PROVIDER
    if LLM_PROVIDER == "auto" or not LLM_PROVIDER:
        if GROQ_API_KEY:
            return "groq"
        if OPENAI_API_KEY:
            return "openai"
        return "mock"
    return "mock"


PROVIDER = resolve_provider()
_PRESET = PROVIDER_PRESETS.get(PROVIDER, PROVIDER_PRESETS["mock"])

LLM_API_KEY = _PRESET["api_key"]
LLM_BASE_URL = _PRESET["base_url"]
SMALL_MODEL = os.getenv("SMALL_MODEL", _PRESET["small"])
LARGE_MODEL = os.getenv("LARGE_MODEL", _PRESET["large"])

# --- Embeddings -------------------------------------------------------------
# auto: sentence-transformers when installed, else hashing. "openai" opt-in.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))  # used by the hashing embedder

# --- Vector store -----------------------------------------------------------
# auto: qdrant when reachable at QDRANT_URL, else in-memory
VECTOR_STORE = os.getenv("VECTOR_STORE", "auto").strip().lower()
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")

# --- Routing ----------------------------------------------------------------
ROUTE_THRESHOLD = _float("ROUTE_THRESHOLD", 0.55)
ESCALATE_ON_LOW_CONFIDENCE = _bool("ESCALATE_ON_LOW_CONFIDENCE", True)
BUDGET_USD_PER_DAY = _float("BUDGET_USD_PER_DAY", 0.0)  # 0 disables budget downshifting

# --- Semantic cache ---------------------------------------------------------
CACHE_THRESHOLD = _float("CACHE_THRESHOLD", 0.94)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "2000"))

# --- Retrieval ---------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))

# POST /api/ingest accepts a filesystem `path`; that lets any caller on a
# deployed instance pull server files into the corpus. Keep it enabled for
# local convenience, disable it wherever the API is public (vercel.json sets
# ALLOW_PATH_INGEST=false).
ALLOW_PATH_INGEST = _bool("ALLOW_PATH_INGEST", True)

# --- App --------------------------------------------------------------------
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
# Append-only JSONL request log — no database engine (see app/metrics.py).
METRICS_PATH = Path(os.getenv("METRICS_PATH", str(BASE_DIR / "data" / "requests.jsonl")))
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(BASE_DIR / "sample_data")))

# --- Pricing (USD per 1M tokens). Extend as models ship. ---------------------
PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Groq
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "gpt-oss-20b": {"input": 0.075, "output": 0.30},
    # Google
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    # Demo providers
    "mock-small": {"input": 0.15, "output": 0.60},
    "mock-large": {"input": 2.50, "output": 10.00},
}


def price_for(model: str) -> dict[str, float]:
    """Return per-1M-token prices for a model, best-effort matching the prefix.

    Provider-prefixed ids ("openai/gpt-oss-20b") are matched on their basename
    too, so a slash in the model id never silently falls back to mid-market.
    """
    base = model.rsplit("/", 1)[-1]
    if model in PRICING:
        return PRICING[model]
    if base in PRICING:
        return PRICING[base]
    for known, price in PRICING.items():
        if model.startswith(known) or base.startswith(known):
            return price
    # Unknown model: assume mid-market pricing rather than free.
    return {"input": 1.00, "output": 3.00}
