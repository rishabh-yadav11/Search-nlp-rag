"""Transition-based reporting for a dependency that is failing.

Every caller here keeps running when a backing store disappears, so the only signal an
operator gets is a log line. A plain "warn once" flag gets that wrong in both directions: it
silences every later outage in the process forever, and -- if you fix that by logging on every
failure -- it turns one dependency outage into one line per request.

:class:`DegradedLatch` reports transitions, but rate-limits them against the clock, because
transitions alone are not a volume bound: without a time bound a dependency that fails every
other call makes every request its own transition and the policy degenerates into two lines per
request. A real example is one permanently corrupt cache key alternating with good ones.

The rules are:

* a failure is logged when it falls outside the window, and the latch then *claims* the outage,
  so a later success has something to close;
* a failure inside the window is silent and claims nothing -- the log already says the
  dependency is down, and pairing every flap is the spam again;
* a success logs the recovery only for a claimed outage, and releases the claim so the next
  outage is announced rather than swallowed.

The window is measured from the last *outage* line and nothing else. That detail is what
separates a rate limit from a warning owed and never paid: letting a recovery restart the
window hands a dependency that fails again shortly after recovering a fresh full window, and
the suppressed outage never claims either, so the eventual recovery logs nothing and the log's
last word stays "recovered" while the dependency is in fact down.

Both lines are WARNING, not INFO. Neither gunicorn nor uvicorn attaches a handler to the root
logger and the deployed command passes no ``--log-config``, so root keeps Python's default
WARNING level and an INFO record from an application logger is dropped before it reaches PM2.
A recovery signal the log never shows cannot bound anything, so it is emitted at the same level
as the outage it closes.
"""

import logging
import time
from collections.abc import Callable

#: How long a latch stays quiet after it has emitted a line. Long enough that a flapping
#: dependency cannot turn a hot path into one line per request, short enough that an ongoing
#: outage is re-announced while it lasts.
REANNOUNCE_SECONDS = 300.0


class DegradedLatch:
    """Announce one dependency's outages, without one line per request.

    ``name`` labels the thing that can go down and appears in the recovery line (e.g. "analytics
    Redis"); it is normally the dependency, qualified by the operation when one module watches
    several ("user profile Redis (get_user_interactions)"). The outage line is whatever the
    caller passes to :meth:`warn_degraded`, because only the caller knows the consequence
    ("recording paused", "using in-process cache").

    ``now`` is the clock, injectable so tests can control the window instead of sleeping.
    """

    def __init__(
        self,
        logger: logging.Logger,
        name: str,
        reannounce_after: float = REANNOUNCE_SECONDS,
        *,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._logger = logger
        self._name = name
        self._interval = reannounce_after
        self._now = now
        # Whether the log currently claims an outage, i.e. whether a success would close one.
        self._announced = False
        # When the last *outage* line was emitted, which is what the window is measured from.
        # Recovery lines never write here.
        self._last = None

    def warn_degraded(self, message: str, *args: object) -> None:
        """Log ``message`` on a failure that falls due, and claim the outage.

        ``message`` is a :mod:`logging` format string and ``args`` its arguments; they are only
        interpolated when a line is actually emitted.
        """
        now = self._now()
        if self._within_window(now):
            return
        self._announced = True
        self._last = now
        self._logger.warning(message, *args)

    def log_recovered(self) -> None:
        """Log the recovery for a claimed outage and release the claim.

        Releasing the claim is what makes the latch once *per outage* rather than once per
        process: a dependency that comes back and goes down again is two incidents, and only
        the first one may be free.

        Deliberately does not touch ``_last``. The window rate-limits announcements of failures,
        and a recovery is not one: letting it restart the window would hand the NEXT failure a
        fresh full window, delaying a warning that is already overdue and leaving the log's
        last word as "recovered" while the dependency is down (see the module docstring).

        A no-op while healthy, and a no-op for an outage that was never announced, so a success
        on a hot path costs a boolean check and cannot manufacture a recovery line.
        """
        if not self._announced:
            return
        self._announced = False
        self._logger.warning("%s recovered", self._name)

    def _within_window(self, now: float) -> bool:
        return self._last is not None and now - self._last < self._interval
