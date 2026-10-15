"""Test isolation: force offline, dependency-free settings before app imports.

Tests must not hit live services (Qdrant, Groq) or load heavy models
(sentence-transformers), so the app modules pick up these env defaults.
"""
import os
import tempfile

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "hashing")
os.environ.setdefault("VECTOR_STORE", "memory")
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("OPENAI_API_KEY", "")
# The suite clears tables on purpose — point it at a scratch DB so running
# `pytest -q` never wipes a locally running dashboard's history.
os.environ.setdefault(
    "DB_PATH",
    os.path.join(tempfile.mkdtemp(prefix="rag-router-tests-"), "metrics.db"),
)
