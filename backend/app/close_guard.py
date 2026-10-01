"""One bounded, logged close primitive shared by every teardown path.

Closing an external client is cleanup, never business logic: a raise or a hang
in one ``close()`` must not strand the sockets owned by every resource released
after it. This module holds that discipline once, so the readiness probe
(``app.health``) and the application lifespan (``app.main``) cannot drift into
two divergent variants of "best-effort release".
"""

import asyncio
import logging
from collections.abc import Callable

# Same bound the readiness probe gives its own client release: a close that has
# not finished in this long is stuck on a connection that is not coming back.
DEFAULT_CLOSE_TIMEOUT = 2.0

#: Teardown failures a caller considers "expected" and wants dropped without
#: masking a defect. ``OSError`` and ``TimeoutError`` are the network-side
#: ones; a broken close implementation is not in here, so it still propagates
#: unless a caller deliberately widens the tuple.
EXPECTED_CLOSE_ERRORS: tuple[type[BaseException], ...] = (OSError, TimeoutError)

logger = logging.getLogger("close_guard")


def resolve_close(resource: object) -> Callable[[], object] | None:
    """Return the object's close callable, preferring the modern ``aclose``.

    ``None`` means the object has no close at all (a plain stub, or something
    that manages no socket). Sync and async callables are both returned as-is;
    the caller decides how to await the result.
    """
    close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
    return close if callable(close) else None


async def await_close(close: Callable[[], object], timeout: float) -> None:
    """Invoke ``close`` and bound the wait.

    A coroutine-function close is awaited under ``asyncio.wait_for``, so a merely
    SLOW close -- the ordinary case, a socket with no answer -- is cancelled and
    abandoned and the caller regains control after ``timeout``.

    The bound holds only for a close that cooperates with cancellation:
    ``wait_for`` waits for the cancellation to be delivered, so a close that
    swallows ``CancelledError`` keeps the caller blocked indefinitely. Nothing
    here defends against that -- a caller whose teardown step may behave that way
    must structure the step itself around it, the way
    ``app.main._cancel_and_wait`` does with ``asyncio.wait``.

    A synchronous close is invoked directly and its return value ignored, so a
    sync close that blocks the event loop is not made bounded here either.
    """
    outcome = close()
    if asyncio.iscoroutine(outcome) or isinstance(outcome, asyncio.Future):
        await asyncio.wait_for(outcome, timeout=timeout)


async def close_quietly(
    resource_name: str,
    resource: object | Callable[[], object],
    *,
    timeout: float = DEFAULT_CLOSE_TIMEOUT,
    suppress: tuple[type[BaseException], ...] = (),
    log: logging.Logger | None = None,
) -> None:
    """Release one resource, best effort, and never raise ``suppress``.

    ``resource`` may be the object itself (its ``aclose``/``close`` is resolved)
    or a zero-argument callable performing the close, for a teardown step that is
    not a single method call.

    A close that merely runs past ``timeout`` is cancelled and abandoned -- a slow
    teardown step is expected to be reported, not to abort the rest of shutdown
    -- and every failure in ``suppress`` is logged and swallowed. That bound
    assumes the close cooperates with cancellation (see ``await_close``).
    Exceptions outside ``suppress`` propagate, and ``asyncio.CancelledError``
    (a ``BaseException``, never in an expected tuple) always does, so a cancelled
    shutdown still unwinds.

    One call releases one resource; continuing with the rest after a failure is
    the caller's job.
    """
    log = log or logger
    close = resource if callable(resource) else resolve_close(resource)
    if close is None:
        return
    try:
        await await_close(close, timeout)
    except TimeoutError:
        log.warning("Timed out after %.1fs closing %s; abandoning it", timeout, resource_name)
    except suppress:
        log.warning("Error closing %s; continuing with the rest of shutdown", resource_name, exc_info=True)
