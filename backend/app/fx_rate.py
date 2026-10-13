

import asyncio
import json
import logging
import urllib.request

from app.config import config

logger = logging.getLogger("fx_rate")



_rate: float = config.INR_PER_USD
_retry_seconds: float = config.FX_RATE_REFRESH_SECONDS


def rate_usd_inr() -> float:

    return _rate


def _fetch() -> float | None:

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

    global _rate
    fetched = await asyncio.to_thread(_fetch)
    if fetched is not None:
        _rate = fetched
        logger.info("USD→INR rate updated to %.2f", fetched)


async def fx_rate_loop() -> None:

    await refresh()
    while _retry_seconds > 0:
        await asyncio.sleep(_retry_seconds)
        await refresh()
