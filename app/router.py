"""Cost-aware routing: score query difficulty, pick the cheapest model that can
handle it, and downshift under budget pressure.

Routing signals (all cheap, computed before any LLM call):
  * lexical    - shallow "what is X" vs. deep "why/how/compare/design" phrasing
  * structure  - multi-part questions, enumeration, constraint clauses
  * entities   - number of distinct nouns/entities a question spans
  * retrieval  - whether the answer looks distributed across several documents
  * budget     - daily spend cap forces a downshift to the small model
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from . import config

DEEP_PATTERNS = [
    r"\bwhy\b", r"\bhow\b", r"\bexplain\b", r"\bcompare\b", r"\bcontrast\b",
    r"\btrade[- ]?offs?\b", r"\bpros and cons\b", r"\bdesign\b", r"\barchitect",
    r"\bimplement", r"\boptimi[sz]e\b", r"\banaly[sz]e\b", r"\bjustify\b",
    r"\bstep[- ]by[- ]step\b", r"\bwrite (?:a|the) (?:essay|report|plan)\b",
    r"\bdebug\b", r"\brefactor\b", r"\bstrateg", r"\bplan\b", r"\bderive\b",
    r"\bbetween .* and\b", r"\bif .* then\b",
]
SHALLOW_PATTERNS = [
    r"^\s*what (?:is|are)\b", r"^\s*who (?:is|are)\b", r"^\s*when (?:is|are|did)\b",
    r"^\s*where (?:is|are)\b", r"^\s*define\b", r"^\s*list\b", r"^\s*give me\b",
    r"^\s*translate\b", r"^\s*capital of\b", r"^\s*say (?:hello|hi)\b",
]
_DEEP_RE = [re.compile(p, re.IGNORECASE) for p in DEEP_PATTERNS]
_SHALLOW_RE = [re.compile(p, re.IGNORECASE) for p in SHALLOW_PATTERNS]

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")


@dataclass
class RouteDecision:
    model: str
    tier: str                     # "small" | "large"
    difficulty: float              # 0..1
    reasons: list[str] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    forced: str | None = None      # set when budget/escalation overrode the score

    def to_dict(self) -> dict:
        return asdict(self)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + pow(2.718281828, -x))


def score_difficulty(query: str, retrieved_scores: list[float] | None = None) -> RouteDecision:
    """Return a 0..1 difficulty score plus the human-readable reasons behind it."""
    text = query.strip()
    lower = text.lower()
    reasons: list[str] = []
    score = 0.0

    deep_hits = [m.group(0).lower() for rx in _DEEP_RE if (m := rx.search(text))]
    shallow_hits = [m.group(0).lower().strip() for rx in _SHALLOW_RE if (m := rx.search(text))]

    if deep_hits:
        score += 0.16 * min(len(deep_hits), 3)
        reasons.append(f"depth keywords: {', '.join(sorted(set(deep_hits))[:4])}")
    if shallow_hits:
        score -= 0.25
        reasons.append(f"shallow template: '{shallow_hits[0]}'")

    # Structure: multi-part / enumerated questions need synthesis.
    q_marks = text.count("?") + max(0, text.count(",") - 2)
    if q_marks >= 2:
        score += 0.18
        reasons.append("multi-part question")
    if re.search(r"\b(?:and|also|additionally|moreover)\b", lower) and len(text.split()) > 12:
        score += 0.10
        reasons.append("compound clauses")
    if re.search(r"\b(?:for example|such as|e\.g\.)\b", lower):
        score += 0.05
        reasons.append("asks for examples")

    # Length: long prompts usually carry more constraints.
    words = len(text.split())
    length_score = min(words / 60.0, 1.0) * 0.20
    score += length_score
    if words > 40:
        reasons.append(f"long query ({words} words)")

    # Entities: distinct capitalized tokens / numbers / quoted terms.
    entities = {w for w in _WORD_RE.findall(text) if w[0].isupper() or w.isdigit()}
    ent_score = min(len(entities) / 8.0, 1.0) * 0.15
    score += ent_score
    if len(entities) >= 4:
        reasons.append(f"{len(entities)} entities referenced")

    # Retrieval spread: if relevant context is smeared across many chunks with
    # near-equal scores, the answer must be stitched together -> harder.
    if retrieved_scores:
        top = sorted(retrieved_scores, reverse=True)[: config.TOP_K]
        if len(top) >= 3:
            spread = (top[0] - top[-1]) / max(top[0], 1e-6)
            if spread < 0.35:
                score += 0.15
                reasons.append("context spread across several chunks")

    difficulty = round(max(0.0, min(1.0, score)), 4)
    if not reasons:
        reasons.append("no strong depth signals")
    return RouteDecision(
        model=config.SMALL_MODEL,
        tier="small",
        difficulty=difficulty,
        reasons=reasons,
        features={"words": words, "entities": len(entities), "deep_hits": len(deep_hits)},
    )


class CostAwareRouter:
    """Turns a difficulty score + budget state into a concrete model choice."""

    def __init__(
        self,
        small_model: str = config.SMALL_MODEL,
        large_model: str = config.LARGE_MODEL,
        threshold: float = config.ROUTE_THRESHOLD,
        budget_usd_per_day: float = config.BUDGET_USD_PER_DAY,
    ):
        self.small_model = small_model
        self.large_model = large_model
        self.threshold = threshold
        self.budget_usd_per_day = budget_usd_per_day
        self._spend_today = 0.0

    # -- budget ---------------------------------------------------------
    def add_spend(self, usd: float) -> None:
        self._spend_today += usd

    @property
    def spend_today(self) -> float:
        return self._spend_today

    def budget_exhausted(self) -> bool:
        return self.budget_usd_per_day > 0 and self._spend_today >= self.budget_usd_per_day

    # -- routing --------------------------------------------------------
    def route(self, query: str, retrieved_scores: list[float] | None = None) -> RouteDecision:
        decision = score_difficulty(query, retrieved_scores)

        if self.budget_exhausted():
            decision.model = self.small_model
            decision.tier = "small"
            decision.forced = "budget"
            decision.reasons.append(
                f"daily budget ${self.budget_usd_per_day:.2f} reached -> downshift"
            )
            return decision

        if decision.difficulty >= self.threshold:
            decision.model = self.large_model
            decision.tier = "large"
        else:
            decision.model = self.small_model
            decision.tier = "small"
        return decision

    def escalate(self, decision: RouteDecision) -> RouteDecision | None:
        """Promote a small-tier decision to the large model (post-hoc escalation)."""
        if decision.tier == "large" or self.budget_exhausted():
            return None
        decision.tier = "large"
        decision.model = self.large_model
        decision.forced = "escalation"
        decision.reasons.append("small model answer rejected -> escalated to large model")
        return decision
