"""In-memory stand-in for the Redis commands the rate limiter uses.

The production limiter establishes its window with a single ``SET key 0 NX EX
window`` and then ``INCR``s the key, and both properties are invisible to a
double that accepts ``**kwargs``: ``EX`` is what makes the counter reclaimable,
and ``NX`` is what stops the window being re-armed on every hit (which would
pin the count at 1 and turn the limit into no limit). So this fake models them,
and RECORDS a contract violation when a counter key is written without a TTL.

Recording rather than raising is deliberate: ``_consume_counter`` wraps the
whole Redis exchange in ``except Exception``, so a fake that raised on an
unexpected call would push the request onto the in-process fallback and leave
the suite green while proving nothing.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

__all__ = ["CountersView", "RateLimitRedisFake"]


class RateLimitRedisFake:
    """Fake Redis supporting the ``SET NX EX`` / ``INCR`` / ``EXPIRE`` subset.

    What this fake does enforce, modelling real Redis:

    * ``EX`` sets an absolute expiry, and an expired key is indistinguishable
      from an absent one to every operation here.
    * ``NX`` returns ``None`` (Redis's not-set reply) and leaves an existing,
      unexpired key untouched.
    * ``INCR`` counts up from the stored value -- or from 0 when the key is
      absent or expired -- and, like real ``INCR``, does NOT clear or extend an
      existing expiry.
    * ``EXPIRE`` sets or refreshes an expiry.
    * A ``SET`` of a counter with no positive ``EX`` is recorded in
      ``violations``. The write still happens, so the test sees the
      consequence rather than a swallowed error.

    Not provided, and no test should assume: connection, RESP, pipelining or
    transactions, eviction, persistence, other commands (``ttl()``/``expiry()``
    /``counters`` are inspection helpers, not Redis replies), error injection
    (patch ``auth._rate_client`` with something that raises), or concurrency --
    access is not locked, so this is only meaningful from a single-threaded
    test.

    Time is a plain callable, defaulting to ``time.monotonic``, plus an offset
    moved by :meth:`advance`, so a test can cross a window boundary without
    sleeping.
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._offset: float = 0.0
        self._values: dict[str, int] = {}
        self._expiries: dict[str, float] = {}
        # Ordered call log, so tests can assert on the arguments production
        # actually passed.
        self.calls: list[tuple[Any, ...]] = []
        self.violations: list[str] = []

    # --- clock -----------------------------------------------------------

    def now(self) -> float:
        """Current fake time: the injected clock plus any :meth:`advance` offset."""
        return self._clock() + self._offset

    def advance(self, seconds: float) -> float:
        """Move fake time forward by ``seconds`` and return the new :meth:`now`."""
        self._offset += float(seconds)
        return self.now()

    # --- expiry bookkeeping ----------------------------------------------

    def _prune(self) -> None:
        now = self.now()
        for key in [k for k, e in self._expiries.items() if e <= now]:
            del self._expiries[key]
            self._values.pop(key, None)

    def _is_ttl(self, ex: Any) -> bool:
        return isinstance(ex, (int, float)) and not isinstance(ex, bool) and math.isfinite(ex) and ex > 0

    def expiry(self, key: str) -> float | None:
        """Absolute expiry timestamp for ``key``, or ``None`` if it has none."""
        self._prune()
        return self._expiries.get(key)

    def ttl(self, key: str) -> float | None:
        """Seconds left before ``key`` expires, or ``None`` if absent/no TTL."""
        expiry = self.expiry(key)
        if expiry is None:
            return None
        return expiry - self.now()

    @property
    def counters(self) -> CountersView:
        """Live ``key -> count`` view of the unexpired counters.

        A VIEW, not a copy: returning one would hand tests a permanently empty
        dict. Reading prunes, so ``len``/``in``/iteration see an expired key as
        absent, exactly as Redis does.
        """
        return CountersView(self)

    # --- commands ---------------------------------------------------------

    async def set(self, key: str, value: Any, nx: Any = False, ex: Any = None) -> str | None:
        self.calls.append(("set", key, value, nx, ex))
        if not self._is_ttl(ex):
            self.violations.append(
                f"counter key {key!r} written with no TTL (ex={ex!r}); a counter with no expiry "
                "locks its subject out forever"
            )
        self._prune()
        if nx and key in self._values:
            # Real NX: do not overwrite, and do not touch the existing expiry.
            return None
        self._values[key] = int(value)
        if self._is_ttl(ex):
            self._expiries[key] = self.now() + float(ex)
        else:
            self._expiries.pop(key, None)
        return "OK"

    async def incr(self, key: str) -> int:
        self.calls.append(("incr", key))
        self._prune()
        # Absent or expired starts from 0; an existing expiry is deliberately
        # kept, because real INCR does not extend a TTL.
        value = self._values.get(key, 0) + 1
        self._values[key] = value
        return value

    async def expire(self, key: str, ttl: float) -> bool:
        self.calls.append(("expire", key, ttl))
        self._prune()
        if key not in self._values:
            return False
        self._expiries[key] = self.now() + float(ttl)
        return True


class CountersView(Mapping):
    """Read-through ``Mapping`` of a fake's unexpired ``key -> count`` pairs.

    Implements only the ``Mapping`` protocol; every read prunes first, so an
    expired key is invisible to ``len``, ``in``, and iteration exactly as it is
    to Redis. Deliberately not a ``dict`` subclass, so a test cannot mutate the
    store through it and then convince itself it exercised the limiter.
    """

    def __init__(self, fake: RateLimitRedisFake) -> None:
        self._fake = fake

    def _snapshot(self) -> dict[str, int]:
        self._fake._prune()
        return dict(self._fake._values)

    def __getitem__(self, key: str) -> int:
        return self._snapshot()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._snapshot())

    def __len__(self) -> int:
        return len(self._snapshot())

    def __repr__(self) -> str:
        return f"CountersView({self._snapshot()!r})"
