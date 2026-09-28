import logging

import pytest

from app.degraded import REANNOUNCE_SECONDS, DegradedLatch

# A dedicated logger name keeps these assertions off every other module's
# records, so a test can only see the latch under test.
LOGGER_NAME = "test_degraded_latch"

OUTAGE = "widget Redis unavailable (%s)"


class _FakeClock:
    """Callable stand-in for ``time.monotonic`` that moves only when told.

    The latch rate-limits its lines against the clock, so a test driving
    several outages in a row has to decide whether they are minutes or
    microseconds apart. Injecting the clock makes that decision explicit and
    keeps the tests from sleeping.
    """

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _messages(records):
    """Rendered messages, which is what identifies each line.

    Outage and recovery lines are both WARNING, so the level alone cannot say
    which is which; the message carries the ordering proof.
    """
    return [record.getMessage() for record in records]


@pytest.fixture
def clock():
    return _FakeClock()


@pytest.fixture
def latch_logs(caplog):
    """Yield the ``caplog`` object, not ``caplog.records``.

    pytest clears the handler's record list at each phase boundary, so a list
    handed over during setup is already empty by the time the test body runs.
    No level override is needed: every line the latch emits -- outage and
    recovery alike -- is WARNING, which caplog captures by default.
    """
    return caplog


@pytest.fixture
def latch(clock):
    return DegradedLatch(logging.getLogger(LOGGER_NAME), "widget Redis", now=clock)


def test_outage_then_recovery_then_outage_announces_both_outages(
    latch, latch_logs, clock
):
    latch.warn_degraded(OUTAGE, ConnectionError("down"))
    latch.log_recovered()
    # Two incidents minutes apart are two incidents, not one: the window has to
    # pass before the second outage is announced again. Without the jump the
    # third event would still be inside the window and correctly suppressed.
    clock.advance(REANNOUNCE_SECONDS + 1)
    latch.warn_degraded(OUTAGE, ConnectionError("down"))

    assert [r.levelno for r in latch_logs.records] == [logging.WARNING] * 3
    assert _messages(latch_logs.records) == [
        "widget Redis unavailable (down)",
        "widget Redis recovered",
        "widget Redis unavailable (down)",
    ]


def test_sustained_outage_logs_once_regardless_of_how_many_failures(latch, latch_logs):
    # The clock never moves, so no re-announce window can open: 50 failures are
    # one line, not 50.
    for _ in range(50):
        latch.warn_degraded(OUTAGE, ConnectionError("down"))

    assert len(latch_logs.records) == 1


def test_success_without_a_preceding_outage_logs_nothing(latch, latch_logs):
    for _ in range(5):
        latch.log_recovered()

    assert latch_logs.records == []


def test_repeated_recovery_logs_once(latch, latch_logs):
    latch.warn_degraded(OUTAGE, ConnectionError("down"))
    for _ in range(5):
        latch.log_recovered()

    assert [r.levelno for r in latch_logs.records] == [logging.WARNING] * 2
    assert _messages(latch_logs.records) == [
        "widget Redis unavailable (down)",
        "widget Redis recovered",
    ]


def test_recovery_line_names_the_dependency(latch, latch_logs):
    latch.warn_degraded(OUTAGE, ConnectionError("down"))
    latch.log_recovered()

    recovery = latch_logs.records[-1]
    assert "widget Redis" in recovery.getMessage()
    # The recovery signal is worthless below WARNING: the root logger keeps
    # Python's default level under gunicorn/uvicorn, so a quieter record never
    # reaches PM2 and cannot bound the incident.
    assert recovery.levelno == logging.WARNING


def test_outage_message_interpolates_the_caller_s_arguments(latch, latch_logs):
    latch.warn_degraded(OUTAGE, ConnectionError("refused"))

    assert latch_logs.records[0].levelno == logging.WARNING
    assert latch_logs.records[0].getMessage() == "widget Redis unavailable (refused)"


# --- the volume bound: transitions rate-limited against the clock ---


@pytest.mark.parametrize("flaps", [10, 100, 1000])
def test_fast_flapping_dependency_does_not_log_a_line_per_request(
    latch, latch_logs, clock, flaps
):
    """fail, succeed, fail, succeed... stays at two lines, whatever the rate.

    A dependency that fails every other call makes every request its own
    transition, so a policy that logs on every transition degenerates into two
    lines per request -- the spam the latch exists to prevent. The bound here
    is a time bound: every flap size below is asserted to produce the same two
    lines, so the count cannot be growing with the number of requests.
    """
    for _ in range(flaps):
        latch.warn_degraded(OUTAGE, ConnectionError("down"))
        latch.log_recovered()
        clock.advance(0.001)  # ~1ms per flap: 1000 flaps is barely one second

    assert _messages(latch_logs.records) == [
        "widget Redis unavailable (down)",
        "widget Redis recovered",
    ]
    assert len(latch_logs.records) == 2, f"{2 * flaps} lines for {flaps} flaps"


def test_sustained_outage_with_a_frozen_clock_logs_exactly_one_line(latch, latch_logs):
    """The anti-spam property, at a scale that leaves no room for a pair.

    Same policy as the 50-failure test above, larger N: an outage that never
    ends and a clock that never moves can only ever cost its first line.
    """
    for _ in range(200):
        latch.warn_degraded(OUTAGE, ConnectionError("down"))

    assert _messages(latch_logs.records) == ["widget Redis unavailable (down)"]


def test_sustained_outage_is_reannounced_once_per_elapsed_window(
    latch, latch_logs, clock
):
    """A long outage stays visible: one extra line per window, not per request.

    Silence after the first line is the other half of the anti-spam failure --
    an incident that outlives its first warning. The count has to follow the
    elapsed windows, so 100 failing requests spread over 5 windows cost 5
    lines and not 100.
    """
    windows, requests_per_window = 5, 20

    for _ in range(windows):
        for _ in range(requests_per_window):
            latch.warn_degraded(OUTAGE, ConnectionError("down"))
        clock.advance(REANNOUNCE_SECONDS + 1)

    records = latch_logs.records
    assert _messages(records) == ["widget Redis unavailable (down)"] * windows
    assert len(records) == windows, (
        f"{windows * requests_per_window} failures must not mean "
        f"{windows * requests_per_window} lines"
    )


def test_outage_arriving_inside_the_window_produces_no_recovery_line(
    latch, latch_logs, clock
):
    """An outage dropped by the window must not open a line to close.

    The first incident is reported and closed, then a second outage starts
    while that recovery line is still inside the window. Nothing is logged for
    it, so nothing may be logged for its recovery either: the log already says
    the dependency is down, and pairing every flap is the spam again. If the
    suppressed outage claimed a line that was never emitted, the following
    success would close an outage the log never opened.
    """
    latch.warn_degraded(OUTAGE, ConnectionError("down"))
    latch.log_recovered()  # the first incident is over, and is the recent line
    clock.advance(REANNOUNCE_SECONDS / 2)  # still inside the window
    latch.warn_degraded(OUTAGE, ConnectionError("down"))  # suppressed
    latch.log_recovered()  # closes the suppressed outage

    # Only the first incident's pair; the suppressed outage added nothing.
    assert _messages(latch_logs.records) == [
        "widget Redis unavailable (down)",
        "widget Redis recovered",
    ]


def test_a_dependency_that_fails_again_long_after_the_first_outage_is_announced(
    latch, latch_logs, clock
):
    """Idle then down again: no success ever arrived, the clock re-announces.

    A helper that is not called for hours and then fails has nobody to call
    ``log_recovered()`` in between, so only elapsed time can distinguish the
    two episodes. Without the jump the second failure would be swallowed as a
    flap of an outage the log has not been told about since the first line.
    """
    latch.warn_degraded(OUTAGE, ConnectionError("down"))
    clock.advance(REANNOUNCE_SECONDS * 10)  # ten re-announce windows of silence
    latch.warn_degraded(OUTAGE, ConnectionError("down"))

    assert _messages(latch_logs.records) == [
        "widget Redis unavailable (down)",
        "widget Redis unavailable (down)",
    ]
 
 
def test_a_recovery_does_not_extend_the_silence_of_the_outage_before_it(
    latch, latch_logs, clock
):
    """The second outage is owed a line because the *outage* line aged out.

    Outage announced at t=0, real recovery at t=299, dependency down again at
    t=301. Measured from the outage that is 301s of silence, so the failure is
    a new incident and is announced. A window that restarted on the recovery
    would instead measure 2s and swallow it, handing a dependency that flaps on
    a short cycle a silence it keeps renewing itself.
    """
    latch.warn_degraded(OUTAGE, ConnectionError("down"))
    clock.advance(REANNOUNCE_SECONDS - 1)
    latch.log_recovered()  # the dependency really did come back
    clock.advance(2)
    latch.warn_degraded(OUTAGE, ConnectionError("down"))

    assert _messages(latch_logs.records) == [
        "widget Redis unavailable (down)",
        "widget Redis recovered",
        "widget Redis unavailable (down)",
    ]


def test_a_flapping_dependency_cannot_renew_its_own_silence_indefinitely(
    latch, latch_logs, clock
):
    """A flapping dependency is bounded by elapsed windows, not by flaps.

    Flapping with a real recovery on every cycle: a latch that simply reset on
    recovery would emit a pair per cycle, and a hot path that flaps per request
    would emit a pair per request. Here the run covers three windows, so the
    cost is three pairs however many cycles the operator sees -- the recovery
    lines never move the window, which is what stops the latch renewing its
    own silence indefinitely.
    """
    cycles, windows = 12, 3

    for _ in range(cycles):
        latch.warn_degraded(OUTAGE, ConnectionError("down"))
        latch.log_recovered()
        clock.advance(REANNOUNCE_SECONDS / 4)  # 12 * 75s == 3 windows

    assert _messages(latch_logs.records) == [
        "widget Redis unavailable (down)",
        "widget Redis recovered",
    ] * windows
    assert len(latch_logs.records) == 2 * windows, (
        f"{len(latch_logs.records)} lines for {cycles} flaps; the latch is "
        f"logging per transition rather than per window"
    )


def test_an_outage_that_outlives_its_window_is_announced_again_despite_flapping(
    latch, latch_logs, clock
):
    """The re-announce still happens once a flap's outage really does age out.

    Flapping slower than the window is not the spam case -- each outage has
    gone unmentioned for a full window, so each is announced. The bound is one
    line per elapsed window, never one per flap.
    """
    for _ in range(3):
        latch.warn_degraded(OUTAGE, ConnectionError("down"))
        latch.log_recovered()
        clock.advance(REANNOUNCE_SECONDS + 1)

    assert _messages(latch_logs.records) == [
        "widget Redis unavailable (down)",
        "widget Redis recovered",
        "widget Redis unavailable (down)",
        "widget Redis recovered",
        "widget Redis unavailable (down)",
        "widget Redis recovered",
    ]
