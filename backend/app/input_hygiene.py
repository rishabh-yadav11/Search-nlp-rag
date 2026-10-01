"""Contain untrusted query and facet input before it reaches a cache key, log line, Qdrant filter or error message.

Stripping is fail-closed: every control character is mapped or removed, never passed through.
"""

import hashlib
import re
import unicodedata

from fastapi import HTTPException

# Facet lists feed both a Qdrant MatchAny and a cache key, so value count and per-value length are
# bounded; the caps sit far above the real single-digit-per-field vocabularies.
MAX_FACET_VALUES = 10
MAX_FACET_VALUE_LEN = 200

# Above this, a key is replaced by its SHA-256 digest: normal pairs stay readable in Redis while a
# pathological one is bounded instead of becoming an arbitrarily long key.
MAX_KEY_LEN = 256

# Word separators, not noise: dropping the tab in "Ola\tIPO" would fuse two *different* questions onto
# one spelling.
_WHITESPACE_CONTROLS = re.compile(r"[\t\n\v\f\r]")

# The remaining C0 controls, plus DEL: NUL truncates a C string at the first byte, and none of the
# rest separate words.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

_WHITESPACE_RUN = re.compile(r"\s+")

_DIGEST_PREFIX = "sha256:"


def normalize_text(value: str) -> str:
    """Canonical spelling for keys, filters and logs; case is preserved because it reaches the embedder."""
    text = unicodedata.normalize("NFKC", value)
    text = _WHITESPACE_CONTROLS.sub(" ", text)
    text = _CONTROL_CHARS.sub("", text)
    return _WHITESPACE_RUN.sub(" ", text).strip()


def split_facet_values(field: str, raw: str | None) -> list[str]:
    """A rejected value is never echoed back -- only ``field``, the caller's own name, reaches the error message."""
    if not raw:
        return []
    # Reject the raw string before normalising it, rather than splitting a megabyte just to count it.
    if len(raw) > MAX_FACET_VALUES * (MAX_FACET_VALUE_LEN + 1):
        raise HTTPException(
            status_code=400,
            detail=f"{field} facet too long "
                   f"(maximum {MAX_FACET_VALUES} values of {MAX_FACET_VALUE_LEN} characters)",
        )
    values = [v for v in (normalize_text(part) for part in raw.split(",")) if v]
    if len(values) > MAX_FACET_VALUES:
        raise HTTPException(
            status_code=400,
            detail=f"too many {field} values (maximum {MAX_FACET_VALUES})",
        )
    for value in values:
        if len(value) > MAX_FACET_VALUE_LEN:
            raise HTTPException(
                status_code=400,
                detail=f"{field} value too long (maximum {MAX_FACET_VALUE_LEN} characters)",
            )
    return values


def build_cache_key(*parts: object, namespace: str = "") -> str:
    """Injective by construction: parts are length-prefixed, unlike ``"|".join``, which lets two filters collide."""
    prefix = f"{namespace}:" if namespace else ""
    encoded = "".join(f"{len(part)}:{part}" for part in (str(p) for p in parts))
    if len(prefix) + len(encoded) > MAX_KEY_LEN:
        return prefix + _DIGEST_PREFIX + hashlib.sha256(encoded.encode()).hexdigest()
    return prefix + encoded
