

import asyncio
import logging
from collections.abc import Callable

DEFAULT_CLOSE_TIMEOUT = 2.0





EXPECTED_CLOSE_ERRORS: tuple[type[BaseException], ...] = (OSError, TimeoutError)

logger = logging.getLogger("close_guard")


def resolve_close(resource: object) -> Callable[[], object] | None:

    close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
    return close if callable(close) else None


async def await_close(close: Callable[[], object], timeout: float) -> None:

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
