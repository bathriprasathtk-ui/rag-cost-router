# Vector Databases and Similarity Search

Vector stores index embeddings so nearest-neighbor search returns the chunks
most similar to a query in milliseconds.

## Index types

- **Flat (brute force)**: exact, simple, fine up to ~1M vectors.
- **HNSW**: graph-based approximate NN; the default choice for low latency.
- **IVF / IVF-PQ**: cluster-then-search; memory-efficient at billions of vectors.

## Distance metrics

Cosine similarity measures angle and ignores magnitude - the usual choice for
normalized text embeddings. Dot product is equivalent for normalized vectors
and faster. Euclidean distance matters when magnitude carries meaning.

## Hybrid retrieval

Pure vector search underperforms on exact identifiers: error codes, SKUs, and
legal citations. Combining BM25 keyword search with vector search (RRF fusion)
recovers those while keeping semantic recall.

## Evaluation

Retrieval quality is measured before generation: context precision, context
recall, and nDCG against labeled query-document pairs. Chasing answer quality
without measuring retrieval hides the real failure.
