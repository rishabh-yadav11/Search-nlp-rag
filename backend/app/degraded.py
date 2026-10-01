"""Transition-based reporting for a dependency that is failing.

Every caller in this service keeps running when a backing store disappears, so
the only signal an operator gets is a log line. A plain "warn once" flag gets
that wrong in both directions: it silences every later outage in the process
forever, and logging on every failure instead turns one outage into one line per
request. :class:`DegradedLatch` reports transitions but rate-limits them against
the clock, because transitions alone are not a volume bound: a dependency that
fails every other call would otherwise make every request its own transition.

The rules are:

* a failure outside the window is logged and *claims* the outage, so a later
  success has something to close;
* a failure inside the window is silent and claims nothing -- the log already
  says the dependency is down, and pairing every flap is the spam again;
* a success logs the recovery only for a claimed outage, and releases the claim
  so the next outage is announced rather than swallowed.

The window is measured from the last *outage* line and from nothing else. That
detail separates a rate limit from a warning owed and never paid: letting a
recovery restart the window gives a dependency that fails again shortly after a
fresh full window, so an outage announced at t=0, recovered at t=299 and down
again at t=301 falls silent for another :data:`REANNOUNCE_SECONDS`. Worse, that
suppressed outage never claims, so the recovery that eventually comes logs
nothing either and the log's last word stays "recovered" while the dependency is
down. A flapping dependency still costs one line per window.

Both lines are WARNING, not INFO: neither gunicorn nor uvicorn attaches a handler
to the root logger and the deployed command passes no ``--log-config``, so root
keeps WARNING and an INFO record is dropped before it reaches PM2. A recovery
signal the log never shows cannot bound anything.
"""

import logging
import time
from collections.abc import Callable

#: How long a latch stays quiet after a line: long enough that a flapping
#: dependency cannot become one line per request, short enough to re-announce.
REANNOUNCE_SECONDS = 300.0


class DegradedLatch:
    """Announce one dependency's outages, without one line per request.

    ``name`` labels what can go down and appears in the recovery line (e.g.
    "analytics Redis"), normally qualified by the operation when one module
    watches several. The outage line is whatever the caller passes to
    :meth:`warn_degraded`, because only the caller knows the consequence
    ("recording paused"). ``now`` is injectable so tests need not sleep.
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
        # Whether the log currently claims an outage, i.e. whether a success has
        # anything to close.
        self._announced = False
        # When the last *outage* line was emitted; recovery lines never write here.
        self._last = None

    def warn_degraded(self, message: str, *args: object) -> None:
        """Log ``message`` on a failure that falls due, and claim the outage.

        ``message`` is a :mod:`logging` format string, interpolated only when a
        line is actually emitted.
        """
        now = self._now()
        if self._within_window(now):
            return
        self._announced = True
        self._last = now
        self._logger.warning(message, *args)

    def log_recovered(self) -> None:
        """Log the recovery for a claimed outage and release the claim.

        Releasing the claim makes the latch once *per outage* rather than once
        per process: a dependency that comes back and goes down again is two
        incidents, and only the first may be free.

        Deliberately does not touch ``_last``: the window rate-limits failure
        announcements, and a recovery is not one (see the module docstring).

        A no-op while healthy or for an outage never announced, so a success on
        a hot path costs a boolean check and cannot invent a recovery line.
        """
        if not self._announced:
            return
        self._announced = False
        self._logger.warning("%s recovered", self._name)

    def _within_window(self, now: float) -> bool:
        return self._last is not None and now - self._last < self._interval
