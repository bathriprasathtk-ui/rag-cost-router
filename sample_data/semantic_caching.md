# Semantic Caching for LLM Services

A semantic cache answers a repeated *paraphrase* without calling the model. A
string-equality cache misses on "what is RAG?" vs "what does retrieval
augmented generation mean?"; an embedding-similarity cache catches both.

## How it works

1. Embed the incoming query.
2. Compare it against cached query embeddings with cosine similarity.
3. Above a threshold (typically 0.92-0.97), return the stored answer.
4. Below the threshold, run the normal pipeline and store the new pair.

## Threshold selection

Too low and users receive answers to *similar but different* questions - the
quality failure mode. Too high and the cache never hits. Measure the
similarity distribution of queries that share an answer, then pick a
threshold with high precision. Per-tenant or per-task thresholds are common:
support bots can cache aggressively, medical advice cannot.

## Operational details

- **TTL**: answers go stale when source documents change; invalidate on ingest.
- **Key on the full request**: include model, prompt version, and user locale
  in the cache key, or you will serve answers generated with a different system
  prompt.
- **Negative caching**: cache "I don't know" answers too, but with a short TTL.
- **Warming**: pre-populate from historical query logs to see savings on day one.

## Measurable impact

The two metrics that matter are hit rate (production RAG traffic is often
30-60% semantically redundant) and the dollars and p95 latency removed per hit.
A cache hit costs one embedding call instead of retrieval plus generation.
