

import hashlib
import re
import unicodedata

from fastapi import HTTPException

MAX_FACET_VALUES = 10
MAX_FACET_VALUE_LEN = 200



MAX_KEY_LEN = 256





_WHITESPACE_CONTROLS = re.compile(r"[\t\n\v\f\r]")



_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")



_WHITESPACE_RUN = re.compile(r"\s+")

_DIGEST_PREFIX = "sha256:"


def normalize_text(value: str) -> str:

    text = unicodedata.normalize("NFKC", value)


    text = _WHITESPACE_CONTROLS.sub(" ", text)
    text = _CONTROL_CHARS.sub("", text)
    return _WHITESPACE_RUN.sub(" ", text).strip()


def split_facet_values(field: str, raw: str | None) -> list[str]:

    if not raw:
        return []



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

    prefix = f"{namespace}:" if namespace else ""
    encoded = "".join(f"{len(part)}:{part}" for part in (str(p) for p in parts))
    if len(prefix) + len(encoded) > MAX_KEY_LEN:
        return prefix + _DIGEST_PREFIX + hashlib.sha256(encoded.encode()).hexdigest()
    return prefix + encoded
