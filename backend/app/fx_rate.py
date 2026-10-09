"""Live USD→INR exchange rate, refreshed twice a day from a public feed.

Cost accounting is canonical in USD; INR is only a display unit, so accuracy
matters little and a free, keyless feed is the right source. The rate lives in
one module global so the two readers (``llm.LLMResult.cost`` and
``cost_budget.to_usd``) cannot drift; before any fetch succeeds, and on network
failure, the configured ``config.INR_PER_USD`` fallback is served. Never raises.
"""

import asyncio
import json
import logging
import urllib.request

from app.config import config

logger = logging.getLogger("fx_rate")

# The current live rate. Initialised to the configured fallback so the app
# behaves as before until the first successful fetch replaces it.
_rate: float = config.INR_PER_USD
_retry_seconds: float = config.FX_RATE_REFRESH_SECONDS


def rate_usd_inr() -> float:
    """The current USD→INR rate (live when fetched, configured fallback before)."""
    return _rate


def _fetch() -> float | None:
    """Synchronous one-shot fetch of USD→INR; the rate, or None on failure.

    Kept sync so the async loop can run it via ``asyncio.to_thread`` (urllib
    blocks) while the worker stays responsive.
    """
    try:
        with urllib.request.urlopen(config.FX_RATE_API_URL, timeout=10) as resp:
            payload = json.loads(resp.read())
        value = payload["rates"]["INR"]
        rate = float(value)
        if rate <= 0:
            logger.warning("fx feed returned a non-positive INR rate %r; ignoring", value)
            return None
        return rate
    except Exception:
        logger.warning("fx feed fetch failed; keeping the previous rate", exc_info=True)
        return None


async def refresh() -> None:
    """Fetch the rate once and publish it. Never raises."""
    global _rate
    fetched = await asyncio.to_thread(_fetch)
    if fetched is not None:
        _rate = fetched
        logger.info("USD→INR rate updated to %.2f", fetched)


async def fx_rate_loop() -> None:
    """Background task: refresh USD→INR every ``FX_RATE_REFRESH_SECONDS``. Never raises.

    Disabled when the interval is 0 (e.g. tests); the initial fetch still runs
    so the fallback rate is replaced the moment the first refresh succeeds.
    """
    await refresh()
    while _retry_seconds > 0:
        await asyncio.sleep(_retry_seconds)
        await refresh()
