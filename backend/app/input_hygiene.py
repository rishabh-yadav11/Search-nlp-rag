"""Input normalisation and bounds for untrusted strings on their way into a
cache key, a log line, a Qdrant filter or an error message.

Three problems are solved here, all at the point where input is first accepted
rather than at each sink, so a new call site is safe by default:

* **Canonicalisation.** A cache key built from the raw query fragments on
  byte-different spellings of the same question. NFKC-folding, dropping control
  characters and collapsing whitespace make ``ＴＥＳＴ``, ``test`` and
  ``test\\x00\\r\\n"`` one spelling, so the same query stops producing a fresh
  cache entry (and a fresh embedding) per variation. It also means no NUL or
  CRLF can reach a cache key or a log line from here.
  The whitespace-bearing controls (TAB, LF, CR, ...) are turned into spaces
  rather than deleted, so two questions that differ only in their line
  structure stay two questions instead of fusing into one that was never
  asked.
* **Unbounded growth.** Facet params are comma-separated lists with no limit on
  how many values, or how long each value, a caller may send. Both feed a
  Qdrant ``MatchAny`` *and* a cache key, so one request can be made to build a
  multi-kilobyte filter and a multi-kilobyte key.
* **Ambiguity.** Joining key components with a delimiter stops being injective
  as soon as a component may itself contain that delimiter -- ``industry='a',
  dealtype='b|c'`` and ``industry='a|b', dealtype='c'`` joined with ``|`` are
  the same string for two different filters, so one request is served the other
  one's cached results. Keys are built from a length-prefixed encoding, which is
  unambiguous by construction, and digested when they would otherwise be
  unreasonably long.

Normalisation is deliberately *not* case-folding. NFKC is Unicode compatibility
normalisation -- full-width Latin, ligatures and decomposed accents are the same
character in a different presentation, and folding them leaves the meaning
intact. Case is not a presentation variant: it reaches the embedder, so folding
it would change what a query retrieves. See ``normalize_text``.
"""

import hashlib
import re
import unicodedata

from fastapi import HTTPException

# Facet params build a Qdrant MatchAny and a cache-key component, so both the
# number of values and the size of each value have to be bounded. Ten values of
# 200 characters is still far more than any single UI selection needs: the
# industry/dealtype vocabularies are single-digit per field, and the widest
# real value the tag vocabulary contains is 112 characters, so the cap sits
# above every real filter while still bounding the abuse (10 x 200 plus
# separators) by orders of magnitude.
MAX_FACET_VALUES = 10
MAX_FACET_VALUE_LEN = 200

# Above this many characters a key is replaced by its SHA-256 digest. Normal
# (query, filter) pairs stay readable in Redis for debugging; a pathological one
# is bounded instead of becoming an arbitrarily long key.
MAX_KEY_LEN = 256

# The whitespace-bearing C0 controls. These are word separators, not noise:
# deleting ``"Ola\tIPO"``'s tab would fuse it into ``"OlaIPO"`` and canonicalise
# two *different* questions onto one spelling, which is the collision class this
# module exists to remove (and it would silently answer the fused question).
# They are mapped to a space first and collapsed below, so no control character
# still reaches a key or a log line.
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

    Applies NFKC, maps the whitespace-bearing control characters to spaces,
    drops the remaining control characters, collapses whitespace runs to a
    single space and strips the ends. Case is preserved on purpose: folding it
    would change the text handed to the embedder, and therefore what a query
    retrieves, which is a search-semantics change rather than a hygiene one.
    """
    text = unicodedata.normalize("NFKC", value)
    # Space first, strip second: dropping TAB/LF/CR outright would merge the
    # words around them into a spelling no caller sent.
    text = _WHITESPACE_CONTROLS.sub(" ", text)
    text = _CONTROL_CHARS.sub("", text)
    return _WHITESPACE_RUN.sub(" ", text).strip()


def split_facet_values(field: str, raw: str | None) -> list[str]:
    """Split one comma-separated facet param into normalised, bounded values.

    ``field`` is the caller's own field name and is the only thing that appears
    in the error message -- a rejected value is never echoed back, so the error
    cannot be used to reflect input into the response.
    """
    if not raw:
        return []
    # Cheap bound first. A raw string longer than the largest in-bounds facet
    # (MAX_FACET_VALUES values of MAX_FACET_VALUE_LEN plus separators) cannot
    # pass the per-value checks below, so rejecting it here avoids normalising
    # and splitting a megabyte of caller input just to count it.
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

    Each part is length-prefixed, so the encoding is injective: no two different
    part sequences can produce the same key, whatever the parts contain. (This
    is what a plain ``"|".join`` is not -- see the module docstring.) ``namespace``
    is kept as a plain leading segment so the key stays greppable and scannable
    in Redis (``SCAN MATCH search:*``) and a normal key stays readable. Keys
    longer than :data:`MAX_KEY_LEN` have their body replaced by a SHA-256
    digest, so one pathological request cannot create an arbitrarily long key.
    """
    prefix = f"{namespace}:" if namespace else ""
    encoded = "".join(f"{len(part)}:{part}" for part in (str(p) for p in parts))
    if len(prefix) + len(encoded) > MAX_KEY_LEN:
        return prefix + _DIGEST_PREFIX + hashlib.sha256(encoded.encode()).hexdigest()
    return prefix + encoded
