# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation pairs a search/retrieval step with a language
model. Instead of relying on parametric memory, the system fetches relevant
document chunks and puts them in the prompt context.

## Why RAG instead of fine-tuning?

- **Freshness**: documents can be updated without retraining.
- **Citations**: answers can point at source chunks.
- **Cost**: ingesting 10k documents costs far less than fine-tuning on them.
- **Access control**: permissions can be enforced at retrieval time.

## The standard pipeline

1. **Ingest** documents from connectors (PDF, Confluence, S3, Git).
2. **Chunk** them into ~300-800 token windows with overlap.
3. **Embed** each chunk into a vector space.
4. **Index** vectors in a store such as Qdrant, pgvector, or FAISS.
5. **Retrieve** top-k chunks per query (optionally with hybrid BM25 + vector).
6. **Rerank** with a cross-encoder when precision matters more than latency.
7. **Generate** an answer grounded in the retrieved context.
8. **Evaluate** with faithfulness, context precision, and answer relevance.

## Failure modes worth knowing

- **Stale index**: documents deleted upstream still retrievable.
- **Chunk boundary loss**: the answer straddles two chunks.
- **Lost-in-the-middle**: models weight the start and end of long context.
- **No answer found**: the model hallucinates instead of declining.
- **Permission leakage**: retrieval ignores who is asking.

## Multi-hop questions

A question whose answer spans several documents ("compare X's SLA with Y's
penalty clause") often fails with plain top-k retrieval: the first hop retrieves
one document, but the second hop's query can only be formed after reading it.
Graph-based retrieval or iterative search closes that gap.
