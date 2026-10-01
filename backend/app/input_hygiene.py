"""Input normalisation and bounds for untrusted strings on their way into a
cache key, a log line, a Qdrant filter or an error message.

Three problems are solved here, at the point where input is first accepted
rather than at each sink, so a new call site is safe by default:

* **Canonicalisation.** A cache key built from the raw query fragments on
  byte-different spellings of the same question. NFKC-folding, dropping control
  characters and collapsing whitespace make full-width, plain and control-byte
  spellings one spelling, so the same query stops producing a fresh cache entry
  (and a fresh embedding) per variation, and no NUL or CRLF can reach a cache key
  or a log line from here. Whitespace-bearing controls (TAB, LF, CR) become
  spaces rather than being deleted, so two questions differing only in line
  structure stay two questions instead of fusing into one never asked.
* **Unbounded growth.** Facet params are comma-separated lists with no limit on
  count or value length, and both feed a Qdrant ``MatchAny`` *and* a cache key,
  so one request could build a multi-kilobyte filter and key.
* **Ambiguity.** Delimiter-joining stops being injective once a component may
  itself contain the delimiter -- ``industry='a', dealtype='b|c'`` and
  ``industry='a|b', dealtype='c'`` joined with ``|`` are one string for two
  filters, so one request is served the other's cached results. Keys use a
  length-prefixed encoding, unambiguous by construction, and are digested when
  they would otherwise be unreasonably long.

Normalisation is deliberately *not* case-folding: NFKC folds presentation
variants only, while case reaches the embedder, so folding it would change what
a query retrieves. See ``normalize_text``.
"""

import hashlib
import re
import unicodedata

from fastapi import HTTPException

# Facet params build a Qdrant MatchAny and a cache-key component, so count and
# value size must both be bounded. Ten values of 200 characters is far more than
# any UI selection needs (the widest real tag value is 112 characters), so the cap
# sits above every real filter while bounding abuse by orders of magnitude.
MAX_FACET_VALUES = 10
MAX_FACET_VALUE_LEN = 200

# Above this many characters a key is replaced by its SHA-256 digest: normal keys
# stay readable in Redis, a pathological one is bounded instead of unbounded.
MAX_KEY_LEN = 256

# The whitespace-bearing C0 controls are word separators, not noise: deleting
# "Ola\tIPO"'s tab would fuse it into "OlaIPO", canonicalising two DIFFERENT
# questions onto one spelling. Mapped to a space, then collapsed below, so no
# control character reaches a key or a log line.
_WHITESPACE_CONTROLS = re.compile(r"[\t\n\v\f\r]")

# Every other C0 control, plus DEL: NUL truncates a C string at the first byte,
# and none of the rest separates words in a query or a facet name.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

# Whitespace runs, including the separators NFKC leaves behind (e.g. the
# ideographic space U+3000, which NFKC maps to a plain space anyway).
_WHITESPACE_RUN = re.compile(r"\s+")

_DIGEST_PREFIX = "sha256:"


def normalize_text(value: str) -> str:
    """Return ``value`` in the canonical spelling used for keys, filters and logs.

    NFKC, whitespace-bearing controls to spaces, remaining controls dropped,
    whitespace runs collapsed, ends stripped. Case is preserved on purpose:
    folding it would change the text handed to the embedder, and therefore what
    a query retrieves.
    """
    text = unicodedata.normalize("NFKC", value)
    # Space first, strip second: dropping TAB/LF/CR outright would merge the words
    # around them into a spelling no caller sent.
    text = _WHITESPACE_CONTROLS.sub(" ", text)
    text = _CONTROL_CHARS.sub("", text)
    return _WHITESPACE_RUN.sub(" ", text).strip()


def split_facet_values(field: str, raw: str | None) -> list[str]:
    """Split one comma-separated facet param into normalised, bounded values.

    ``field`` is the caller's own field name and is the only thing that appears
    in the error message -- a rejected value is never echoed back, so the error
    cannot reflect input into the response.
    """
    if not raw:
        return []
    # Cheap bound first: a raw string longer than the largest in-bounds facet
    # cannot pass the per-value checks, so rejecting here avoids normalising and
    # splitting a megabyte of caller input just to count it.
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
    """Build an unambiguous cache key from ``parts``.

    Each part is length-prefixed, so the encoding is injective -- what a plain
    ``"|".join`` is not (see the module docstring). ``namespace`` stays a plain
    leading segment so the key remains greppable and ``SCAN MATCH``-able in
    Redis; a body longer than :data:`MAX_KEY_LEN` becomes a SHA-256 digest.
    """
    prefix = f"{namespace}:" if namespace else ""
    encoded = "".join(f"{len(part)}:{part}" for part in (str(p) for p in parts))
    if len(prefix) + len(encoded) > MAX_KEY_LEN:
        return prefix + _DIGEST_PREFIX + hashlib.sha256(encoded.encode()).hexdigest()
    return prefix + encoded
