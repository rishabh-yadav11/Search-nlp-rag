

import logging
import time
from collections.abc import Callable

REANNOUNCE_SECONDS = 300.0


class DegradedLatch:


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


        self._announced = False

        self._last = None

    def warn_degraded(self, message: str, *args: object) -> None:

        now = self._now()
        if self._within_window(now):
            return
        self._announced = True
        self._last = now
        self._logger.warning(message, *args)

    def log_recovered(self) -> None:

        if not self._announced:
            return
        self._announced = False
        self._logger.warning("%s recovered", self._name)

    def _within_window(self, now: float) -> bool:
        return self._last is not None and now - self._last < self._interval
