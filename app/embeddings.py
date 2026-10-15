"""Pluggable embedding backends.

`auto` prefers sentence-transformers (local, free, true semantic similarity)
and falls back to the hashing embedder so the app always runs even without
the optional heavy dependency.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np

from . import config

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Dependency-free lexical embedder: hashed word + character-trigram features.

    Not a substitute for a real semantic model, but it captures paraphrase-level
    overlap well enough for demos and keeps the app runnable offline.
    """

    name = "hashing"
    dim: int

    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.dim = dim

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = _WORD_RE.findall(text.lower())
        for word in words:
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[idx] += 1.5  # words weigh more than characters
            padded = f"^{word}$"
            for i in range(len(padded) - 2):
                tri = padded[i : i + 3]
                tidx = int(hashlib.md5(tri.encode()).hexdigest(), 16) % self.dim
                vec[tidx] += 1.0
        return vec

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        matrix = np.stack([self._vectorize(t) for t in texts])
        return _normalize(matrix.astype(np.float32))


class OpenAIEmbedder:
    def __init__(self, model: str = config.EMBEDDING_MODEL):
        self.name = f"openai:{model}"
        self.model = model
        # Use OpenAI credentials specifically — not the active LLM provider's.
        # Otherwise a Groq-key setup (the default when both keys exist) would
        # send its key to the wrong /embeddings endpoint.
        self.api_key = config.OPENAI_API_KEY
        self.base_url = config.OPENAI_BASE_URL

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1), np.float32)
        try:
            import httpx

            resp = httpx.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts},
                timeout=30,
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            matrix = np.array([row["embedding"] for row in data], dtype=np.float32)
            return _normalize(matrix)
        except Exception as exc:  # network/key problems fall back so the demo never dies
            print(f"[embeddings] provider API failed ({exc}); falling back to hashing embedder")
            return HashingEmbedder().embed(texts)


class SentenceTransformerEmbedder:
    """Local semantic embeddings (free, offline after first model download)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.name = f"sentence-transformers:{model}"
        self.model_name = model
        self._model = SentenceTransformer(model)
        try:
            self.dim = int(self._model.get_embedding_dimension())  # sentence-transformers >= 4
        except AttributeError:
            self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1), np.float32)
        matrix = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(matrix, dtype=np.float32)


def build_embedder():
    provider = config.EMBEDDING_PROVIDER.lower()
    if provider == "auto":
        try:
            import sentence_transformers  # noqa: F401

            provider = "sentence-transformers"
        except Exception:
            provider = "hashing"
    if provider == "openai":
        return OpenAIEmbedder()
    if provider in {"sentence-transformers", "st"}:
        try:
            return SentenceTransformerEmbedder(config.EMBEDDING_MODEL)
        except Exception as exc:
            print(f"[embeddings] sentence-transformers unavailable ({exc}); using hashing")
            return HashingEmbedder()
    return HashingEmbedder()
