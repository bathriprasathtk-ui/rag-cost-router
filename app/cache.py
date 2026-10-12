"""Semantic (embedding-similarity) response cache.

A query is a hit when cosine similarity to a stored query exceeds the
threshold. Hits skip both retrieval and the LLM call, which is where the
cost/latency savings come from.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    answer: str
    model: str
    created_at: float
    last_used: float
    hits: int = 1
    meta: dict = field(default_factory=dict)


@dataclass
class CacheLookup:
    hit: bool
    similarity: float = 0.0
    entry: CacheEntry | None = None


class SemanticCache:
    def __init__(
        self,
        threshold: float = config.CACHE_THRESHOLD,
        ttl_seconds: int = config.CACHE_TTL_SECONDS,
        max_entries: int = config.CACHE_MAX_ENTRIES,
    ):
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: list[CacheEntry] = []
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    # -- helpers --------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def _evict_expired(self) -> None:
        now = time.time()
        self._entries = [e for e in self._entries if now - e.created_at < self.ttl_seconds]

    def _evict_overflow(self) -> None:
        if len(self._entries) <= self.max_entries:
            return
        # Drop least-recently used entries until we are back under the cap.
        self._entries.sort(key=lambda e: e.last_used)
        overflow = len(self._entries) - self.max_entries
        del self._entries[:overflow]

    # -- public API -----------------------------------------------------
    def lookup(self, query_embedding: np.ndarray) -> CacheLookup:
        with self._lock:
            self._evict_expired()
            if not self._entries:
                self.misses += 1
                return CacheLookup(hit=False)

            matrix = np.stack([e.embedding for e in self._entries])
            sims = matrix @ query_embedding.astype(np.float32)
            best = int(np.argmax(sims))
            similarity = float(sims[best])
            if similarity >= self.threshold:
                entry = self._entries[best]
                entry.hits += 1
                entry.last_used = time.time()
                self.hits += 1
                return CacheLookup(hit=True, similarity=similarity, entry=entry)
            self.misses += 1
            return CacheLookup(hit=False, similarity=similarity)

    def store(self, query: str, query_embedding: np.ndarray, answer: str, model: str, **meta) -> CacheEntry:
        with self._lock:
            self._evict_expired()
            entry = CacheEntry(
                query=query,
                embedding=np.asarray(query_embedding, dtype=np.float32).copy(),
                answer=answer,
                model=model,
                created_at=time.time(),
                last_used=time.time(),
                meta=meta,
            )
            self._entries.append(entry)
            self._evict_overflow()
            return entry

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hit_rate, 4),
                "threshold": self.threshold,
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = 0
