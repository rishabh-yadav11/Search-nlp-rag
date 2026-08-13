TOP_WEAK_THRESHOLD = 0.3


def weak_results(query: str, scores: list[float], limit: int = 3) -> bool:
    """True when fewer than `limit` of the top reranked scores exceed
    TOP_WEAK_THRESHOLD, i.e. retrieval is too weak to answer the query.
    An empty scores list counts as weak; if there are fewer than `limit`
    results total, only what exists is evaluated."""
    strong = sum(1 for s in scores if s > TOP_WEAK_THRESHOLD)
    return strong < limit


def fallback_answer(query: str, n_weak: int) -> str:
    """Honest fallback for /ask: never fabricates facts, mentions the query."""
    if n_weak <= 0:
        return (
            f"I couldn't find strong matches in the VCCircle corpus for '{query}'. "
            "Try rephrasing, or ask about a specific company/sector."
        )
    return (
        f"I couldn't find strong matches in the VCCircle corpus for '{query}'. "
        f"The {n_weak} closest articles are only weakly related, so I won't guess. "
        "Try rephrasing, or ask about a specific company/sector."
    )


def search_note(scores: list[float]) -> str | None:
    """Short annotation for /search when results are weak, else None."""
    if weak_results("", scores):
        return "Top results are weakly related to this query — consider rephrasing."
    return None
