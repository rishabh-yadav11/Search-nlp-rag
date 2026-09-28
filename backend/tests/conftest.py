import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BACKEND_DIR, "scripts")
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

for _path in (BACKEND_DIR, SCRIPTS_DIR, TESTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)


@pytest.fixture
def fake_cache():
    """Factory for the in-memory ``HybridCache`` stand-in.

    Yields the shared ``FakeCache`` class rather than an instance: a test calls
    ``fake_cache()`` for a bare cache, or ``fake_cache(get_result=...)`` /
    ``fake_cache(get_error=...)`` when ``get()`` has to return or raise
    something specific. Every call builds a fresh store, so no state is shared
    between tests.
    """
    from _support import FakeCache

    return FakeCache


@pytest.fixture(autouse=True)
def _reset_auth_rate_limits():
    """Clear the auth rate limiter's in-process counters around every test.

    When the limiter's Redis is unreachable the limiter falls back to a
    bounded in-process counter, and no Redis runs in the test environment --
    so that fallback is the live limiter here. Its state lives on the module
    and the whole session shares one process, without this reset the logins of
    one test would 429 an unrelated test later in the run.
    """
    from app import auth

    auth.reset_local_rate_limits()
    yield
    auth.reset_local_rate_limits()
