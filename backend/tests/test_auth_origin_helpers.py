"""Unit tests for the CSRF guard's two load-bearing string helpers.

The guard's entire decision rests on them: IPv6 literals must reduce correctly,
a port is never a security boundary, and ``Origin: null`` names no host. They
are tested directly because a request-level test cannot tell *which* half of
the comparison produced a 403.
"""

import pytest

from app.auth import _host_only, _origin_host

# --- _host_only: lowercasing, and dropping the port without eating IPv6 ---


@pytest.mark.parametrize(
    ("authority", "expected"),
    [
        # A port is not a security boundary: dev serves frontend and API under one host.
        ("example.com", "example.com"),
        ("example.com:8001", "example.com"),
        ("example.com:443", "example.com"),
        # Whitespace arrives with any proxy that joins header values with a comma.
        ("  example.com:8001  ", "example.com"),
        # DNS is case-insensitive; either case is the same host.
        ("Example.COM", "example.com"),
        ("EXAMPLE.com:443", "example.com"),
        # Load-bearing: splitting on the FIRST colon would turn "[::1]:8001"
        # into "[", letting any bracketed address match any other.
        ("[::1]", "::1"),
        ("[::1]:8001", "::1"),
        ("[2001:db8::1]", "2001:db8::1"),
        ("[2001:db8::1]:8001", "2001:db8::1"),
        ("[::1]:80", "::1"),
        ("[2001:DB8::A]", "2001:db8::a"),
    ],
)
def test_host_only_lowercases_and_drops_the_port(authority, expected):
    assert _host_only(authority) == expected


@pytest.mark.parametrize(
    "authority",
    [
        # An unbalanced bracket has no "]" to stop at, so it is kept verbatim
        # and can never equal a well-formed host -- the safe direction to fail.
        "[::1",
        "[",
        "example.com:8001:9000",
        # A bare IPv6 literal is deliberately NOT in this list: `_host_only`
        # also runs on `urlsplit(...).hostname`, which arrives unbracketed, and
        # reducing it to "" would make the two sides of a comparison disagree.
        # A trailing dot is a legitimate absolute-FQDN form and is NOT stripped,
        # which makes the comparison stricter, never looser.
        "example.com.",
    ],
)
def test_host_only_does_not_normalize_malformed_authorities_into_a_matchable_host(authority):
    """Whatever a malformed authority reduces to, it must not be a host an
    attacker can name in ``Origin`` and be compared equal to the real one."""
    # "::1" is absent on purpose: reducing a bare IPv6 literal to itself is
    # correct behaviour, not a forgery.
    reduced = _host_only(authority)
    assert reduced not in {
        "example.com",
        "2001:db8::1",
        "2001:db8::a",
    }, f"{authority!r} reduced to a real host, so a forged Origin could match it"


def test_host_only_does_not_confuse_two_different_bracketed_addresses():
    """The failure the bracket handling exists to prevent: both would reduce
    to "[" and one host would authenticate as another."""
    assert _host_only("[::1]:8001") != _host_only("[2001:db8::1]:8001")
    assert _host_only("[::1]") != _host_only("[::2]")


# --- _origin_host: what an Origin names, or None when it names nothing ---


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("https://example.com", "example.com"),
        # Port and scheme are discarded: TLS termination means the scheme cannot be trusted.
        ("https://example.com:8443", "example.com"),
        ("http://example.com:3000", "example.com"),
        ("https://Example.COM", "example.com"),
        ("HTTPS://EXAMPLE.com", "example.com"),
        # urlsplit's .hostname is IPv6-safe and strips the brackets.
        ("https://[::1]", "::1"),
        ("https://[::1]:8443", "::1"),
        ("http://[2001:db8::1]:3000", "2001:db8::1"),
    ],
)
def test_origin_host_extracts_the_host_it_names(origin, expected):
    assert _origin_host(origin) == expected


@pytest.mark.parametrize("origin", ["null", "NULL", " null "])
def test_origin_host_is_none_for_the_literal_null(origin):
    """``Origin: null`` names no host, so it must come back as None: treating
    "no host" as "same host" would give every sandboxed cross-origin POST a
    free pass."""
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
