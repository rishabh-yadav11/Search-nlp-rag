import calendar
import re

TOP_WEAK_THRESHOLD = 0.3

_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


def date_label(from_date: str | None, to_date: str | None) -> str | None:
    """Human-readable label for an effective date window used by the note, or
    None when the window isn't a plain month or year range. E.g. ('2025-01-01',
    '2025-01-31') -> 'January 2025'; ('2025-01-01','2025-12-31') -> '2025'."""
    if not from_date or not to_date:
        return None
    m1 = re.match(r"^(\d{4})-(\d{2})-01$", from_date)
    if not m1:
        return None
    year, month = int(m1.group(1)), int(m1.group(2))
    last = calendar.monthrange(year, month)[1]
    if to_date == f"{year}-{month:02d}-{last:02d}":
        return f"{_MONTH_NAMES[month]} {year}"
    if from_date == f"{year}-01-01" and to_date == f"{year}-12-31":
        return str(year)
    return None


def results_are_weak(scores: list[float], limit: int = 3) -> bool:
    """True when fewer than `limit` of the top reranked scores exceed
    TOP_WEAK_THRESHOLD, i.e. retrieval is too weak to answer the query.

    ``limit`` is capped at the number of available scores (min 1): a topic with
    only 1-2 strong matches in the corpus is NOT treated as weak — refusing to
    answer then would suppress every narrow/niche question. An empty scores
    list still counts as weak."""
    if not scores:
        return True
    strong = sum(1 for s in scores if s > TOP_WEAK_THRESHOLD)
    return strong < min(limit, len(scores))


def fallback_answer(query: str, n_weak: int, label: str | None = None) -> str:
    """Honest fallback for chat: never fabricates facts, mentions the query.
    With a date label, frames the result as a best-effort for that period."""
    if label:
        return (
            f"I found only a few articles matching '{query}' for {label}. "
            f"Here are the closest {n_weak if n_weak > 0 else 1} matches, "
            "but none is a strong fit — check the sources below."
        )
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


def weak_results_note(scores: list[float], label: str | None = None) -> str | None:
    """Short annotation for /search when results are weak, else None. With a
    date label the note is framed as a best-effort for that period."""
    if not results_are_weak(scores):
        return None
    if label:
        return f"Showing the closest {label} matches — only a few articles cover this exact topic."
    return "Top results are weakly related to this query — consider rephrasing."
