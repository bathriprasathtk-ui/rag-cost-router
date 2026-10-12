"""Vector stores: chunk documents, embed them, retrieve top-k.

Two backends with the same interface:
  * QdrantStore   - persistent, runs in Docker, survives restarts
  * MemoryStore   - in-process fallback so the app works without Docker

`build_store()` picks Qdrant when reachable, else memory.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


@dataclass
class Chunk:
    doc: str
    text: str
    index: int
    embedding: np.ndarray | None = None
    score: float = 0.0
    id: str = ""


def chunk_text(text: str, doc: str, size: int = 700, overlap: int = 120) -> list[Chunk]:
    """Paragraph-aware sliding window chunker."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)

    # Add overlap between adjacent chunks for boundary-spanning answers.
    stitched: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        prefix = ""
        if i > 0:
            prefix = chunks[i - 1][-overlap:] + "\n...\n"
        stitched.append(Chunk(doc=doc, text=prefix + chunk, index=i))
    return stitched


class MemoryStore:
    """In-process store: chunks live in RAM (and are persisted to disk by the app)."""

    backend = "memory"

    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []

    def add_text(self, text: str, doc: str = "inline") -> int:
        new = chunk_text(text, doc)
        if not new:
            return 0
        vectors = self.embedder.embed([c.text for c in new])
        for chunk, vec in zip(new, vectors):
            chunk.embedding = vec
            chunk.id = str(uuid.uuid4())
        self.chunks.extend(new)
        return len(new)

    def add_directory(self, path: Path) -> int:
        added = 0
        if not path.exists():
            return 0
        for file in sorted(path.rglob("*")):
            if file.suffix.lower() in {".md", ".txt", ".rst"} and file.is_file():
                added += self.add_text(file.read_text(encoding="utf-8", errors="ignore"), doc=file.name)
        return added

    def search(self, query_embedding: np.ndarray, k: int = config.TOP_K) -> list[Chunk]:
        if not self.chunks:
            return []
        matrix = np.stack([c.embedding for c in self.chunks])
        sims = matrix @ query_embedding.astype(np.float32)
        order = np.argsort(sims)[::-1][:k]
        return [
            Chunk(doc=self.chunks[i].doc, text=self.chunks[i].text,
                  index=self.chunks[i].index, embedding=None,
                  score=float(sims[i]), id=self.chunks[i].id)
            for i in order
        ]

    def stats(self) -> dict:
        docs = {c.doc for c in self.chunks}
        return {"chunks": len(self.chunks), "documents": len(docs), "backend": self.backend}

    def clear(self) -> None:
        self.chunks.clear()


class QdrantStore:
    """Persistent store backed by a Qdrant server (Docker)."""

    backend = "qdrant"

    def __init__(self, embedder, url: str = config.QDRANT_URL,
                 collection: str = config.QDRANT_COLLECTION,
                 api_key: str = config.QDRANT_API_KEY):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.embedder = embedder
        self.url = url
        self.collection = collection
        self._client = QdrantClient(url=url, api_key=api_key or None, timeout=10)

        dim = int(getattr(embedder, "dim", config.EMBEDDING_DIM))
        existing = {c.name for c in self._client.get_collections().collections}
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        # Payload index on `doc` enables faceting (distinct-document counts).
        try:
            from qdrant_client.models import PayloadSchemaType

            self._client.create_payload_index(
                collection_name=collection,
                field_name="doc",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # index may already exist

    @staticmethod
    def ping(url: str = config.QDRANT_URL, api_key: str = config.QDRANT_API_KEY) -> bool:
        try:
            import httpx

            resp = httpx.get(f"{url}/collections", headers=(
                {"api-key": api_key} if api_key else {}), timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def add_text(self, text: str, doc: str = "inline") -> int:
        from qdrant_client.models import PointStruct

        new = chunk_text(text, doc)
        if not new:
            return 0
        vectors = self.embedder.embed([c.text for c in new])
        points = []
        for chunk, vec in zip(new, vectors):
            chunk.id = str(uuid.uuid4())
            points.append(PointStruct(
                id=chunk.id,
                vector=vec.tolist(),
                payload={"doc": chunk.doc, "text": chunk.text, "index": chunk.index},
            ))
        self._client.upsert(collection_name=self.collection, points=points)
        return len(new)

    def add_directory(self, path: Path) -> int:
        added = 0
        if not path.exists():
            return 0
        for file in sorted(path.rglob("*")):
            if file.suffix.lower() in {".md", ".txt", ".rst"} and file.is_file():
                added += self.add_text(file.read_text(encoding="utf-8", errors="ignore"), doc=file.name)
        return added

    def search(self, query_embedding: np.ndarray, k: int = config.TOP_K) -> list[Chunk]:
        from qdrant_client.models import Distance, ScoredPoint  # noqa: F401

        try:
            hits = self._client.query_points(
                collection_name=self.collection,
                query=query_embedding.astype(np.float32).tolist(),
                limit=k,
                with_payload=True,
            ).points
        except Exception as exc:
            print(f"[qdrant] search failed ({exc}); returning no chunks")
            return []
        return [
            Chunk(doc=p.payload.get("doc", "?"), text=p.payload.get("text", ""),
                  index=int(p.payload.get("index", 0)), score=float(p.score), id=str(p.id))
            for p in hits
        ]

    def stats(self) -> dict:
        try:
            info = self._client.get_collection(self.collection)
            count = info.points_count or 0
            docs: set[str] = set()
            try:  # facet API (qdrant >= 1.15): count distinct doc values
                for f in self._client.facet(collection_name=self.collection, key="doc", limit=100).hits:
                    docs.add(f.value)
            except Exception:
                pass
            return {"chunks": count, "documents": len(docs) or None, "backend": self.backend,
                    "collection": self.collection}
        except Exception:
            return {"chunks": 0, "documents": None, "backend": self.backend}

    def clear(self) -> None:
        self._client.delete_collection(self.collection)


def build_store(embedder):
    """Pick Qdrant when it is reachable, otherwise fall back to memory."""
    mode = config.VECTOR_STORE
    if mode in {"auto", "qdrant"} and QdrantStore.ping(config.QDRANT_URL, config.QDRANT_API_KEY):
        try:
            store = QdrantStore(embedder)
            print(f"[store] using Qdrant at {config.QDRANT_URL} (collection={config.QDRANT_COLLECTION})")
            return store
        except Exception as exc:
            print(f"[store] Qdrant init failed ({exc}); falling back to in-memory")
    elif mode == "qdrant":
        print(f"[store] Qdrant requested but not reachable at {config.QDRANT_URL}; "
              "run 'docker compose up -d qdrant'")
    print("[store] using in-memory vector store")
    return MemoryStore(embedder)
