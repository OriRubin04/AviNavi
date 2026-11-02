"""Injecting the visual position as a synthetic GPS fix (``GPS_INPUT``).

Every assertion here is on bytes that went through the real pymavlink codec and
came back out, so a passing test means a genuine wire-format claim.

The six ``AP_GPS_MAV::handle_msg`` details each get a test of their own, for one
reason: getting any of them wrong produces **silence**, not an error.  The
message is accepted, no ``STATUSTEXT`` appears, the EKF simply never uses the
position, and on the bench that is indistinguishable from a bad camera mount or
a thin map.  A unit test is the only cheap place to catch them.
"""

from __future__ import annotations

import math

import pytest

from avinavi.geo import LocalOrigin
from avinavi.mavlink.gps_input import (
    GPS_EPOCH_UNIX,
    GPS_FIX_TYPE_3D,
    GPS_FIX_TYPE_NO_FIX,
    GPS_LEAP_SECONDS,
    IGNORE_VEL_HORIZ,
    IGNORE_VEL_VERT,
    SECONDS_PER_WEEK,
    GpsInputPublisher,
    unix_to_gps_week_and_ms,
)
from avinavi.types import Attitude, NEDPosition, NavigationSolution

from tests.fakes import FakeConnection


@pytest.fixture
def connection() -> FakeConnection:
    return FakeConnection()


@pytest.fixture
def publisher(connection, origin) -> GpsInputPublisher:
    return GpsInputPublisher(
        connection=connection, origin=origin, send_to_autopilot=True
    )


def solution(**overrides) -> NavigationSolution:
    defaults = dict(
        timestamp_monotonic=100.0,
        timestamp_unix=1_760_000_000.0,
        position_ned=NEDPosition(120.0, -45.0, -80.0),
        attitude=Attitude(1.0, 0.0, 0.0, 0.0),
        velocity_body_frd=(22.0, 0.0, 0.0),
        position_sigma_m=3.0,
        velocity_sigma_mps=1.0,
        confidence=0.8,
    )
    defaults.update(overrides)
    return NavigationSolution(**defaults)


# ---------------------------------------------------------------------------
# GPS time
# ---------------------------------------------------------------------------
def test_the_gps_week_and_time_of_week_round_trip():
    unix = 1_760_000_000.0
    week, week_ms = unix_to_gps_week_and_ms(unix)
    recovered = GPS_EPOCH_UNIX - GPS_LEAP_SECONDS + week * SECONDS_PER_WEEK + week_ms / 1000.0
    assert recovered == pytest.approx(unix, abs=0.001)


def test_the_time_of_week_stays_inside_the_week():
    for unix in (GPS_EPOCH_UNIX + 1.0, 1_700_000_000.0, 1_900_000_000.0):
        _, week_ms = unix_to_gps_week_and_ms(unix)
        assert 0 <= week_ms < SECONDS_PER_WEEK * 1000


def test_a_timestamp_before_the_gps_epoch_degrades_to_zero():
    """An unset clock must not produce a wild week number."""
    assert unix_to_gps_week_and_ms(0.0) == (0, 0)


def test_the_week_is_never_zero_for_a_real_timestamp(publisher, connection):
    """Gotcha 4: jitter correction needs ``fix_type >= 3 && time_week > 0``.

    A zero week costs the corrected timestamp and the time-of-week, degrading
    the EKF's time alignment — with nothing reported.
    """
    publisher.publish(solution())
    message = connection.last("GPS_INPUT")
    assert message.time_week > 0
    assert message.fix_type >= 3


# ---------------------------------------------------------------------------
# The wire format
# ---------------------------------------------------------------------------
def test_the_gps_id_is_the_instance_index_not_the_instance_number(connection, origin):
    """Gotcha 1: ``if (state.instance != packet.gps_id) return;``.

    ``GPS1_TYPE = 14`` is instance **0**.  Sending 1 means every message is
    dropped by the first line of the handler, in total silence.
    """
    publisher = GpsInputPublisher(
        connection=connection, origin=origin, send_to_autopilot=True
    )
    assert publisher.gps_id == 0
    publisher.publish(solution())
    assert connection.last("GPS_INPUT").gps_id == 0


def test_sending_velocity_clears_both_velocity_ignore_flags(publisher, connection):
    """Gotchas 2 and 3: the flags are inverted, and ``vd`` nests inside ``vn/ve``.

    A *set* flag means the field is absent, and setting the horizontal flag
    drops the vertical component too regardless of its own flag — so the three
    components travel together or not at all.
    """
    publisher.publish(solution())
    flags = connection.last("GPS_INPUT").ignore_flags
    assert flags & IGNORE_VEL_HORIZ == 0
    assert flags & IGNORE_VEL_VERT == 0


def test_omitting_velocity_sets_both_velocity_ignore_flags(connection, origin):
    publisher = GpsInputPublisher(
        connection=connection, origin=origin, send_to_autopilot=True,
        send_velocity=False,
    )
    publisher.publish(solution())
    flags = connection.last("GPS_INPUT").ignore_flags
    assert flags & IGNORE_VEL_HORIZ
    assert flags & IGNORE_VEL_VERT


def test_the_satellite_count_satisfies_the_ekf_check(publisher, connection):
    """Gotcha 5: ``EK3_GPS_CHECK`` defaults to 31 and wants 6 satellites.

    Fabricated, and necessarily so — but a fix that reports a constellation the
    EKF rejects is a fix that does nothing at all.
    """
    publisher.publish(solution())
    assert connection.last("GPS_INPUT").satellites_visible >= 6


def test_a_low_satellite_count_is_warned_about(connection, origin, caplog):
    with caplog.at_level("WARNING"):
        GpsInputPublisher(
            connection=connection, origin=origin, send_to_autopilot=True,
            satellites_visible=4,
        )
    assert "needs 6" in caplog.text or "requires 6" in caplog.text


def test_yaw_is_centidegrees_with_zero_meaning_unavailable(publisher, connection):
    """Gotcha 6: true north is 36000, and 0 is the 'no yaw' sentinel.

    With ``send_yaw`` off — the default — the field must be the sentinel rather
    than an accidental claim of due north.
    """
    publisher.publish(solution())
    assert connection.last("GPS_INPUT").yaw == 0


def test_a_northward_heading_is_sent_as_36000_not_zero(connection, origin):
    publisher = GpsInputPublisher(
        connection=connection, origin=origin, send_to_autopilot=True, send_yaw=True
    )
    publisher.publish(solution(attitude=Attitude(1.0, 0.0, 0.0, 0.0)))
    assert connection.last("GPS_INPUT").yaw == 36000


def test_an_eastward_heading_is_sent_as_nine_thousand(connection, origin):
    publisher = GpsInputPublisher(
        connection=connection, origin=origin, send_to_autopilot=True, send_yaw=True
    )
    half = math.radians(90.0) / 2.0
    publisher.publish(
        solution(attitude=Attitude(math.cos(half), 0.0, 0.0, math.sin(half)))
    )
    assert connection.last("GPS_INPUT").yaw == pytest.approx(9000, abs=2)


# ---------------------------------------------------------------------------
# The position and velocity themselves
# ---------------------------------------------------------------------------
def test_the_local_position_is_converted_back_to_degrees(publisher, connection, origin):
    publisher.publish(solution(position_ned=NEDPosition(120.0, -45.0, -80.0)))
    message = connection.last("GPS_INPUT")
    expected = origin.to_geo(NEDPosition(120.0, -45.0, -80.0))
    assert message.lat / 1e7 == pytest.approx(expected.lat_deg, abs=1e-7)
    assert message.lon / 1e7 == pytest.approx(expected.lon_deg, abs=1e-7)


def test_body_frame_velocity_is_rotated_into_ned(publisher, connection):
    """``GPS_INPUT`` is earth-fixed; the propagator stores body FRD for ODOMETRY.

    Heading east at 22 m/s forward must arrive as ve=22, not vn=22 — the exact
    frame confusion that makes an EKF fight the vision solution.
    """
    half = math.radians(90.0) / 2.0
    publisher.publish(
        solution(
            attitude=Attitude(math.cos(half), 0.0, 0.0, math.sin(half)),
            velocity_body_frd=(22.0, 0.0, 0.0),
        )
    )
    message = connection.last("GPS_INPUT")
    assert message.vn == pytest.approx(0.0, abs=1e-4)
    assert message.ve == pytest.approx(22.0, abs=1e-4)
    assert message.vd == pytest.approx(0.0, abs=1e-4)


def test_the_confidence_sigma_becomes_the_horizontal_accuracy(publisher, connection):
    """The one field ArduPilot actually weights the fix by.

    If the confidence model does not reach ``horiz_accuracy`` it has no effect
    on the filter at all, however carefully it was computed.
    """
    publisher.publish(solution(position_sigma_m=7.5))
    assert connection.last("GPS_INPUT").horiz_accuracy == pytest.approx(7.5, abs=1e-4)


def test_an_implausibly_small_sigma_is_floored(publisher, connection):
    """Both ArduPilot and the EKF behave badly given a near-zero accuracy."""
    publisher.publish(solution(position_sigma_m=0.001))
    assert connection.last("GPS_INPUT").horiz_accuracy >= 0.5


def test_the_vertical_accuracy_is_looser_than_the_horizontal(publisher, connection):
    """Honest: the vertical channel comes from image scale and is the weaker one."""
    publisher.publish(solution(position_sigma_m=4.0))
    message = connection.last("GPS_INPUT")
    assert message.vert_accuracy > message.horiz_accuracy


# ---------------------------------------------------------------------------
# Safety behaviour
# ---------------------------------------------------------------------------
def test_nothing_is_sent_while_transmission_is_disabled(connection, origin):
    """The safety progression depends on this being the default."""
    publisher = GpsInputPublisher(
        connection=connection, origin=origin, send_to_autopilot=False
    )
    assert publisher.enabled is False
    assert publisher.publish(solution()) is False
    assert connection.of_type("GPS_INPUT") == []
    assert publisher.suppressed_count == 1


def test_nothing_is_sent_without_an_origin(connection):
    """Without an origin a local NED position cannot become a lat/lon at all."""
    publisher = GpsInputPublisher(
        connection=connection, origin=None, send_to_autopilot=True
    )
    assert publisher.publish(solution()) is False
    assert connection.of_type("GPS_INPUT") == []
    assert "origin" in publisher.last_error


def test_the_origin_can_be_supplied_after_construction(connection, origin):
    publisher = GpsInputPublisher(
        connection=connection, origin=None, send_to_autopilot=True
    )
    publisher.set_origin(origin)
    assert publisher.publish(solution()) is True


def test_nothing_is_sent_while_the_link_is_down(publisher, connection):
    connection.set_connected(False)
    assert publisher.publish(solution()) is False
    assert connection.of_type("GPS_INPUT") == []
    assert "link" in publisher.last_error


def test_a_missing_solution_is_not_an_error(publisher, connection):
    """A gap between fixes is normal; it must not be reported as a failure."""
    assert publisher.publish(None) is False
    assert connection.of_type("GPS_INPUT") == []


def test_losing_the_solution_declares_no_fix_rather_than_going_quiet(
    publisher, connection
):
    """Better than a stream that simply stops.

    Declaring ``GPS_FIX_TYPE_NO_FIX`` puts ArduPilot through its normal
    GPS-loss logic immediately, instead of leaving it to time out on silence.
    """
    publisher.publish(solution())
    assert publisher.publish_failure() is True
    message = connection.last("GPS_INPUT")
    assert message.fix_type == GPS_FIX_TYPE_NO_FIX
    assert message.satellites_visible == 0


def test_a_failure_report_is_suppressed_while_transmission_is_disabled(
    connection, origin
):
    publisher = GpsInputPublisher(
        connection=connection, origin=origin, send_to_autopilot=False
    )
    assert publisher.publish_failure() is False
    assert connection.of_type("GPS_INPUT") == []


def test_the_sent_counter_tracks_successful_sends(publisher, connection):
    for _ in range(3):
        publisher.publish(solution())
    assert publisher.sent_count == 3
    assert len(connection.of_type("GPS_INPUT")) == 3
    assert connection.sent_counts["GPS_INPUT"] == 3


def test_a_good_fix_reports_a_three_dimensional_fix(publisher, connection):
    """ArduPilot's arming checks require at least a 3D fix."""
    publisher.publish(solution())
    assert connection.last("GPS_INPUT").fix_type == GPS_FIX_TYPE_3D


def test_the_publisher_names_its_message_for_the_diagnostics_view(publisher):
    assert publisher.message == "gps_input"
