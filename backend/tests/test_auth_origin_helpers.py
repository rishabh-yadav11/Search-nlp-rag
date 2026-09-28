"""Unit tests for the CSRF guard's two load-bearing string helpers.

``_host_only`` and ``_origin_host`` carry the security claims written in their
docstrings — IPv6 literals reduce correctly, a port is never a security
boundary, and ``Origin: null`` names no host — and the guard's entire decision
rests on them. Tested here against the real functions rather than through
requests, because a request-level test cannot tell *which* half of the
comparison produced a 403: a helper that mishandles a bracket is
indistinguishable from a guard that is simply strict.

The calls are the real ones from ``app.auth``; nothing here reimplements them.
"""

import pytest

from app.auth import _host_only, _origin_host

# --- _host_only: lowercasing, and dropping the port without eating IPv6 ---


@pytest.mark.parametrize(
    ("authority", "expected"),
    [
        # A port is not a security boundary: the dev stack really does serve
        # the frontend on :3000 and the API on :8001 under one host.
        ("example.com", "example.com"),
        ("example.com:8001", "example.com"),
        ("example.com:443", "example.com"),
        # Surrounding whitespace, which arrives with any proxy that joins
        # header values with a comma.
        ("  example.com:8001  ", "example.com"),
        # DNS is case-insensitive, so a browser may send either case and they
        # are the same host.
        ("Example.COM", "example.com"),
        ("EXAMPLE.com:443", "example.com"),
        # The load-bearing IPv6 cases. A split on the FIRST colon would turn
        # "[::1]:8001" into "[" and let any bracketed address match any other,
        # so brackets are peeled before the port is dropped.
        ("[::1]", "::1"),
        ("[::1]:8001", "::1"),
        ("[2001:db8::1]", "2001:db8::1"),
        ("[2001:db8::1]:8001", "2001:db8::1"),
        ("[::1]:80", "::1"),
        # An uppercase IPv6 literal lowercases like any other host.
        ("[2001:DB8::A]", "2001:db8::a"),
    ],
)
def test_host_only_lowercases_and_drops_the_port(authority, expected):
    assert _host_only(authority) == expected


@pytest.mark.parametrize(
    "authority",
    [
        # An unbalanced bracket: no "]" to stop at, so the bracket is kept
        # verbatim. It can therefore never equal a well-formed host, which is
        # the safe direction to fail in — a malformed authority is refused,
        # not normalised into something matchable.
        "[::1",
        "[",
        "example.com:8001:9000",
        # A bare (unbracketed) IPv6 literal is deliberately NOT in this list. It
        # is not a malformed authority: `_host_only` is called on BOTH a raw
        # authority (the Host header, always bracketed for IPv6) and on
        # `urlsplit(...).hostname`, which returns the address with its brackets
        # already stripped. So "::1" is a well-formed input on that path, and
        # reducing it to "" would make the two sides of an IPv6 comparison
        # disagree. The IPv6 round trip is pinned by
        # test_a_genuinely_same_origin_pair_reduces_to_the_same_host.
        # A trailing dot is a legitimate absolute-FQDN form and is NOT stripped
        # here. That is deliberate in effect if not in intent: it makes the
        # comparison stricter, never looser, so "example.com." and
        # "example.com" do not silently match.
        "example.com.",
    ],
)
def test_host_only_does_not_normalize_malformed_authorities_into_a_matchable_host(authority):
    """A malformed authority must not reduce to a value an Origin could match.

    The docstrings promise that a well-formed authority reduces to its host.
    They say nothing about malformed ones, and the security requirement is
    one-directional: whatever these return, it must not be a host an attacker
    can name in ``Origin`` and thereby be compared equal to the real one.
    """
    # The hosts a forged `Origin` could name and thereby be compared equal to.
    # "::1" is absent on purpose: reducing a bare IPv6 literal to itself is the
    # correct behaviour, not a forgery, so it cannot be a forbidden output here.
    reduced = _host_only(authority)
    assert reduced not in {
        "example.com",
        "2001:db8::1",
        "2001:db8::a",
    }, f"{authority!r} reduced to a real host, so a forged Origin could match it"


def test_host_only_does_not_confuse_two_different_bracketed_addresses():
    """The specific failure the bracket handling exists to prevent.

    would reduce to "[" and a request to one IPv6 host would be accepted as
    though addressed to another.
    """
    assert _host_only("[::1]:8001") != _host_only("[2001:db8::1]:8001")
    assert _host_only("[::1]") != _host_only("[::2]")


# --- _origin_host: what an Origin names, or None when it names nothing ---


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("https://example.com", "example.com"),
        # The port and the scheme are both discarded: the app sits behind TLS
        # termination and cannot trust the scheme it sees.
        ("https://example.com:8443", "example.com"),
        ("http://example.com:3000", "example.com"),
        # Case-insensitive host, again.
        ("https://Example.COM", "example.com"),
        ("HTTPS://EXAMPLE.com", "example.com"),
        # IPv6, with and without a port. urlsplit's .hostname is IPv6-safe and
        # returns the address without its brackets.
        ("https://[::1]", "::1"),
        ("https://[::1]:8443", "::1"),
        ("http://[2001:db8::1]:3000", "2001:db8::1"),
    ],
)
def test_origin_host_extracts_the_host_it_names(origin, expected):
    assert _origin_host(origin) == expected


@pytest.mark.parametrize("origin", ["null", "NULL", " null "])
def test_origin_host_is_none_for_the_literal_null(origin):
    """``Origin: null`` names no host at all.

    A sandboxed iframe or a privacy browser sends it. It must come back as
    None so the caller REFUSES the request: treating "no host" as "same host"
    would hand every sandboxed cross-origin POST a free pass.
    """
    assert _origin_host(origin) is None


def test_origin_host_is_none_for_something_that_is_not_a_url():
    """Unparseable input is None, not a raise and not a guess."""
    assert _origin_host("not a url") is None


# --- the two together: what the guard actually compares ---


@pytest.mark.parametrize(
    ("origin", "authority"),
    [
        # Same site, different port: the dev topology, and it must match.
        ("http://localhost:3000", "localhost:8001"),
        # Scheme differs (TLS termination) and must not matter.
        ("https://example.com", "example.com:443"),
        # Case and whitespace in the Host header, as a proxy list would send.
        ("https://example.com", "  Example.com:8001  "),
        # IPv6, same address either way round, with and without ports.
        ("https://[::1]:8443", "[::1]:8001"),
        ("https://[::1]", "[::1]"),
    ],
)
def test_a_genuinely_same_origin_pair_reduces_to_the_same_host(origin, authority):
    assert _host_only(_origin_host(origin)) == _host_only(authority)


@pytest.mark.parametrize(
    ("origin", "authority"),
    [
        # The cross-site cases the guard exists to refuse.
        ("https://evil.example", "example.com"),
        # A registrable-domain sibling: not a suffix/substring match.
        ("https://evil.example.com", "example.com"),
        # Prefix/suffix traps in the other direction.
        ("https://example.com.evil.example", "example.com"),
        ("https://notexample.com", "example.com"),
        # A different port is not a different site, but a different ADDRESS is.
        ("https://[::2]", "[::1]"),
        # "null" can never be shown to be our own host.
        ("null", "example.com"),
    ],
)
def test_a_cross_site_pair_reduces_to_a_different_host(origin, authority):
    origin_host = _origin_host(origin)
    assert not origin_host or _host_only(origin_host) != _host_only(authority)
