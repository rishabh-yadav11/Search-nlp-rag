"""Shared test helpers: the coroutine runner, article factory and cache stand-in.

Not named ``_common``: that module is ``backend/scripts/_common.py``, imported as
a top-level module by several tests, and ``backend/tests`` is on ``sys.path``.
"""

import asyncio
from typing import Any

from app.main import SourceArticle

__all__ = [
    "DERIVE",
    "MISSING",
    "OMIT",
    "FakeCache",
    "make_article",
    "run_sync",
]

#: "No value configured" -- distinct from ``None``, which is a legitimate value.
MISSING = object()

#: "Build this field from the article id" -- ``make_article``'s default.
DERIVE = object()

#: "Leave this field out of the constructor" so the model's own default applies.
OMIT = object()


def run_sync(coro):
    """Run ``coro`` to completion on a fresh event loop and return its result."""
    return asyncio.run(coro)


def make_article(
    id_: int,
    score: float,
    published_date: str | None = None,
    *,
    title: Any = DERIVE,
    url: Any = DERIVE,
    summary: Any = DERIVE,
    **extra: Any,
) -> SourceArticle:
    """Build a ``SourceArticle`` for a test.

    ``title``, ``url`` and ``summary`` derive from ``id_`` at the default
    :data:`DERIVE`; pass :data:`OMIT` to keep the field out of the constructor
    so the model default applies, or a value to use it verbatim. Every other
    keyword is passed straight through to ``SourceArticle``.
    """
    if title is DERIVE:
        title = f"Title {id_}"
    if url is DERIVE:
        url = f"https://example.com/{id_}"
    if summary is DERIVE:
        summary = f"summary {id_}"

    data: dict[str, Any] = {"id": id_, "score": score, "published_date": published_date}
    for field, value in (("title", title), ("url", url), ("summary", summary)):
        if value is not OMIT:
            data[field] = value
    data.update(extra)
    return SourceArticle(**data)


class FakeCache:
    """In-memory stand-in for the ``HybridCache`` the pipeline retrieves through.

    ``get()`` raises ``get_error`` when set, else returns ``get_result`` when
    configured, else the value in ``store``. Reads append to ``gets``, writes to
    ``sets`` as a ``(key, value, ttl)`` triple.
    """

    def __init__(self, get_result: Any = MISSING, get_error: Exception | None = None) -> None:
        self.get_result = get_result
        self.get_error = get_error
        self.store: dict = {}
        self.sets: list = []
        self.gets: list = []

    async def get(self, key):
        self.gets.append(key)
        if self.get_error is not None:
            raise self.get_error
        if self.get_result is not MISSING:
            return self.get_result
        return self.store.get(key)

    async def get_many(self, keys):
        """As ``HybridCache.get_many``, reading through ``get()`` so ``gets`` and
        ``get_result`` apply to every key in the batch."""
        return [await self.get(key) for key in keys]

    async def set(self, key, value, ttl=None):
        self.store[key] = value
        self.sets.append((key, value, ttl))
