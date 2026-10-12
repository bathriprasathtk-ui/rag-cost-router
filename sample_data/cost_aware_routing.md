# Cost-Aware Model Routing

Not every query needs the frontier model. Routing splits traffic by task
difficulty so the cheap model handles the bulk and the expensive model is
reserved for questions that actually need it.

## Signals used for routing

- **Lexical depth**: "what is X" is shallow; "why", "compare", "design a
  system", "trade-offs" are deep.
- **Structure**: multi-part questions and constraint-heavy prompts.
- **Entity count**: questions spanning many entities need more synthesis.
- **Retrieval spread**: when relevant context is smeared across several chunks
  with near-equal scores, the answer must be stitched together.
- **User tier**: enterprise plans can pin a larger default model.

## Escalation as a safety net

Route optimistically to the small model, then escalate when the answer looks
like a refusal, is too short, or fails a cheap verifier. Escalation rates are
a direct measure of how well the router's threshold is tuned: above ~15%
lower the threshold, near zero and you can raise it.

## Budget guards

A daily spend cap downshifts traffic once reached. This turns cost from an
outcome into a constraint: the system degrades to a cheaper model instead of
surprising finance with an overage.

## What to put on a dashboard

Per-request cost, cost by model tier, cache hit rate, p95 latency by tier,
escalation rate, and savings versus an all-frontier-model baseline. If you
cannot show the delta, the optimization is unproven.
