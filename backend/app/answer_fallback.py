import calendar
import re

from app.config import config

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
    try:
        last = calendar.monthrange(year, month)[1]
    except ValueError:
        # Only the month is range-checked: calendar.monthrange raises
        # IllegalMonthError (a ValueError) for a month outside 1-12, while an
        # out-of-range year is normalized rather than rejected, so this handler
        # exists for the month case and any other ValueError monthrange raises.
        return None
    if to_date == f"{year}-{month:02d}-{last:02d}":
        return f"{_MONTH_NAMES[month]} {year}"
    if from_date == f"{year}-01-01" and to_date == f"{year}-12-31":
        return str(year)
    return None


def results_are_weak(scores: list[float], limit: int | None = None) -> bool:
    """True when fewer than `limit` of the top reranked scores exceed the
    WEAK_RESULT_SCORE knob, i.e. retrieval is too weak to answer the query.

    ``limit`` defaults to the WEAK_RESULT_MIN_STRONG knob, is capped at the
    number of available scores and never falls below 1: a topic with only 1-2
    strong corpus matches is NOT weak, or every narrow question would go
    unanswered, and the floor also stops a nonsensical (<= 0) knob from opening
    the gate on everything. An empty scores list still counts as weak.

    Knob names are spelled without their ``config.`` prefix on purpose:
    tests/test_config_knobs.py counts a ``config.KNOB`` mention anywhere in the
    source as a reader, so dotted names here would vouch for a deleted read."""
    if not scores:
        return True
    strong = sum(1 for s in scores if s > config.WEAK_RESULT_SCORE)
    needed = max(1, min(config.WEAK_RESULT_MIN_STRONG if limit is None else limit, len(scores)))
    return strong < needed


def fallback_answer(query: str, n_weak: int, label: str | None = None) -> str:
    """Honest fallback for chat: never fabricates facts, mentions the query.
    With a date label, frames the result as a best-effort for that period."""
    if label:
        # `n_weak` counts the sources actually retrieved, so no wording below may
        # advertise a count or a source that does not exist.
        # Defensive: the only production caller (chat.py) returns early on an
        # empty source list, but this function is public and any caller may pass 0.
        if n_weak <= 0:
            return (
                f"I couldn't find any articles matching '{query}' for {label}. "
                "Try rephrasing, or ask about a specific company/sector."
            )
        if n_weak == 1:
            return (
                f"I found only one article matching '{query}' for {label}. "
                "Here is the closest match, but it isn't a strong fit — "
                "check the source below."
            )
        return (
            f"I found only a few articles matching '{query}' for {label}. "
            f"Here are the closest {n_weak} matches, "
            "but none is a strong fit — check the sources below."
        )
    if n_weak <= 0:
        return (
            f"I couldn't find strong matches in the VCCircle corpus for '{query}'. "
            "Try rephrasing, or ask about a specific company/sector."
        )
    if n_weak == 1:
        return (
            f"I couldn't find strong matches in the VCCircle corpus for '{query}'. "
            "The closest article is only weakly related, so I won't guess. "
            "Try rephrasing, or ask about a specific company/sector."
        )
    return (
        f"I couldn't find strong matches in the VCCircle corpus for '{query}'. "
        f"The {n_weak} closest articles are only weakly related, so I won't guess. "
        "Try rephrasing, or ask about a specific company/sector."
    )


def weak_results_note(scores: list[float], label: str | None = None) -> str | None:
    """Short annotation for /search when results are weak, else None. With a
    date label the note is framed as a best-effort for that period.

    ``results_are_weak`` still reports an empty score list as weak -- chat relies
    on that to refuse to answer rather than guess -- but the note must not claim
    to show matches that do not exist, so the empty case gets its own wording."""
    if not results_are_weak(scores):
        return None
    if not scores:
        if label:
            return f"No articles matched this query for {label}. Try a different period or rephrase."
        return "No articles matched this query. Try rephrasing, or ask about a specific company/sector."
    if label:
        return f"Showing the closest {label} matches — only a few articles cover this exact topic."
    return "Top results are weakly related to this query — consider rephrasing."
