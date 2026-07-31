"""MAVLink external navigation output.

These tests encode real messages with pymavlink and decode them back, so every
assertion is about bytes that would genuinely reach ArduPilot.  The frame IDs
and covariance indices checked here are the ones ArduPilot's
``GCS_MAVLINK::handle_odometry`` actually reads; getting them wrong makes the
message silently disappear, which is exactly the failure a test should catch.
"""

from __future__ import annotations

import math

import pytest

from avinavi.frames import quaternion_from_euler
from avinavi.mavlink.external_navigation import (
    MAV_ESTIMATOR_TYPE_VISION,
    MAV_FRAME_BODY_FRD,
    MAV_FRAME_LOCAL_FRD,
    ExternalNavigationPublisher,
    pose_covariance_from_sigmas,
    velocity_covariance_from_sigma,
)
from avinavi.types import (
    Attitude,
    NavigationSolution,
    NEDPosition,
    covariance_upper_triangle,
)
from tests.fakes import FakeConnection


def make_solution(**overrides) -> NavigationSolution:
    defaults = dict(
        timestamp_monotonic=100.0,
        timestamp_unix=1700000000.0,
        position_ned=NEDPosition(12.5, -3.25, -40.0),
        attitude=Attitude(*quaternion_from_euler(0.0, 0.0, math.pi / 4)),
        velocity_body_frd=(2.0, 0.5, -0.1),
        position_sigma_m=1.5,
        velocity_sigma_mps=0.8,
        yaw_sigma_rad=0.1,
        confidence=0.85,
    )
    defaults.update(overrides)
    return NavigationSolution(**defaults)


@pytest.fixture
def publisher():
    connection = FakeConnection(connected=True)
    return ExternalNavigationPublisher(
        connection, message="odometry", send_to_autopilot=True
    ), connection


# ---------------------------------------------------------------------------
# The frame requirement
# ---------------------------------------------------------------------------
def test_odometry_uses_the_frames_ardupilot_requires(publisher):
    """ArduPilot drops ODOMETRY unless frame is LOCAL_FRD and child is BODY_FRD."""
    publisher, connection = publisher
    assert publisher.publish(make_solution())

    message = connection.last("ODOMETRY")
    assert message is not None
    assert message.frame_id == MAV_FRAME_LOCAL_FRD == 20
    assert message.child_frame_id == MAV_FRAME_BODY_FRD == 12


def test_odometry_carries_ned_position_and_the_vision_estimator_type(publisher):
    publisher, connection = publisher
    publisher.publish(make_solution())

    message = connection.last("ODOMETRY")
    assert message.x == pytest.approx(12.5)
    assert message.y == pytest.approx(-3.25)
    assert message.z == pytest.approx(-40.0)
    assert message.estimator_type == MAV_ESTIMATOR_TYPE_VISION


def test_odometry_quaternion_is_w_first(publisher):
    publisher, connection = publisher
    yaw = math.pi / 4
    publisher.publish(
        make_solution(attitude=Attitude(*quaternion_from_euler(0.0, 0.0, yaw)))
    )

    message = connection.last("ODOMETRY")
    expected = quaternion_from_euler(0.0, 0.0, yaw)
    assert list(message.q) == pytest.approx(list(expected), abs=1e-6)
    # A yaw-only rotation has zero x and y components; catches an xyzw mix-up.
    assert message.q[1] == pytest.approx(0.0, abs=1e-6)
    assert message.q[2] == pytest.approx(0.0, abs=1e-6)


def test_odometry_velocity_is_sent_in_the_body_frame(publisher):
    publisher, connection = publisher
    publisher.publish(make_solution(velocity_body_frd=(3.0, -1.0, 0.25)))

    message = connection.last("ODOMETRY")
    assert (message.vx, message.vy, message.vz) == pytest.approx((3.0, -1.0, 0.25))


def test_velocity_is_zeroed_when_disabled():
    connection = FakeConnection()
    publisher = ExternalNavigationPublisher(
        connection, message="odometry", send_to_autopilot=True, send_velocity=False
    )
    publisher.publish(make_solution(velocity_body_frd=(9.0, 9.0, 9.0)))

    message = connection.last("ODOMETRY")
    assert (message.vx, message.vy, message.vz) == pytest.approx((0.0, 0.0, 0.0))
    # Unknown velocity covariance is signalled with NaN in element 0.
    assert math.isnan(message.velocity_covariance[0])


# ---------------------------------------------------------------------------
# Covariance packing
# ---------------------------------------------------------------------------
def test_covariance_diagonal_lands_where_ardupilot_reads_it():
    """Indices 0, 6, 11, 15, 18, 20 are the diagonal of the packed triangle."""
    packed = covariance_upper_triangle([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert len(packed) == 21
    assert [packed[i] for i in (0, 6, 11, 15, 18, 20)] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert all(packed[i] == 0.0 for i in range(21) if i not in (0, 6, 11, 15, 18, 20))


def test_pose_covariance_reproduces_ardupilots_poserr():
    """ArduPilot computes posErr = sqrt(c[0] + c[6] + c[11]); check it is sane."""
    sigma = 2.0
    covariance = pose_covariance_from_sigmas(sigma, yaw_sigma_rad=0.1)
    position_error = math.sqrt(covariance[0] + covariance[6] + covariance[11])
    # Horizontal sigma twice plus a deliberately looser vertical channel.
    assert position_error == pytest.approx(math.sqrt(2 * sigma**2 + (2 * sigma) ** 2))


def test_pose_covariance_reproduces_ardupilots_angerr():
    """ArduPilot computes angErr = sqrt(c[15] + c[18] + c[20])."""
    covariance = pose_covariance_from_sigmas(1.0, yaw_sigma_rad=0.1, attitude_sigma_rad=0.05)
    angle_error = math.sqrt(covariance[15] + covariance[18] + covariance[20])
    assert angle_error == pytest.approx(math.sqrt(2 * 0.05**2 + 0.1**2))


def test_roll_and_pitch_get_a_large_variance_by_default():
    """We do not estimate roll/pitch visually, so the EKF must not trust them."""
    covariance = pose_covariance_from_sigmas(1.0, yaw_sigma_rad=0.05)
    assert covariance[15] > covariance[20]
    assert covariance[18] > covariance[20]


def test_covariance_length_is_enforced(publisher):
    publisher, _ = publisher
    with pytest.raises(ValueError, match="21 elements"):
        publisher.send_visual_odometry(
            timestamp=0,
            position=(0, 0, 0),
            orientation=(1, 0, 0, 0),
            pose_covariance=[0.0] * 20,
        )


def test_negative_variance_is_rejected(publisher):
    publisher, _ = publisher
    bad = covariance_upper_triangle([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="negative variance"):
        publisher.send_visual_odometry(
            timestamp=0,
            position=(0, 0, 0),
            orientation=(1, 0, 0, 0),
            pose_covariance=bad,
        )


def test_velocity_covariance_is_isotropic():
    covariance = velocity_covariance_from_sigma(0.5)
    assert covariance[0] == covariance[6] == covariance[11] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Quality and reset counter
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "confidence,expected", [(1.0, 100), (0.85, 85), (0.5, 50), (0.0, 1)]
)
def test_quality_maps_confidence_to_percent(confidence, expected):
    """0 means 'unknown' in MAVLink, so a real estimate never reports 0."""
    assert make_solution(confidence=confidence).quality_percent == expected


def test_quality_is_transmitted(publisher):
    publisher, connection = publisher
    publisher.publish(make_solution(confidence=0.73))
    assert connection.last("ODOMETRY").quality == 73


def test_quality_can_be_suppressed():
    connection = FakeConnection()
    publisher = ExternalNavigationPublisher(
        connection, message="odometry", send_to_autopilot=True, send_quality=False
    )
    publisher.publish(make_solution(confidence=0.73))
    # 0 tells ArduPilot the quality is unknown rather than bad.
    assert connection.last("ODOMETRY").quality == 0


def test_failure_notice_uses_quality_minus_one():
    connection = FakeConnection()
    publisher = ExternalNavigationPublisher(
        connection, message="odometry", send_to_autopilot=True
    )
    assert publisher.publish_failure()
    assert connection.last("ODOMETRY").quality == -1


def test_reset_counter_is_transmitted(publisher):
    publisher, connection = publisher
    publisher.publish(make_solution(reset_counter=7))
    assert connection.last("ODOMETRY").reset_counter == 7


# ---------------------------------------------------------------------------
# Suppression rules
# ---------------------------------------------------------------------------
def test_nothing_is_sent_when_transmission_is_disabled():
    connection = FakeConnection()
    publisher = ExternalNavigationPublisher(
        connection, message="odometry", send_to_autopilot=False
    )
    assert publisher.publish(make_solution()) is False
    assert connection.of_type("ODOMETRY") == []
    assert publisher.suppressed_count == 1


def test_nothing_is_sent_when_the_link_is_down():
    connection = FakeConnection(connected=False)
    publisher = ExternalNavigationPublisher(
        connection, message="odometry", send_to_autopilot=True
    )
    assert publisher.publish(make_solution()) is False
    assert connection.of_type("ODOMETRY") == []
    assert "link is down" in publisher.last_error


def test_a_none_solution_sends_nothing_rather_than_repeating(publisher):
    """No valid solution must mean silence, never a resent stale pose."""
    publisher, connection = publisher
    publisher.publish(make_solution())
    assert publisher.publish(None) is False
    assert len(connection.of_type("ODOMETRY")) == 1


def test_message_none_disables_output():
    connection = FakeConnection()
    publisher = ExternalNavigationPublisher(
        connection, message="none", send_to_autopilot=True
    )
    assert publisher.enabled is False
    assert publisher.publish(make_solution()) is False


def test_unknown_message_type_is_rejected():
    with pytest.raises(ValueError):
        ExternalNavigationPublisher(FakeConnection(), message="telepathy")


# ---------------------------------------------------------------------------
# Only one authoritative source
# ---------------------------------------------------------------------------
def test_odometry_mode_never_emits_vision_position_estimate(publisher):
    publisher, connection = publisher
    for _ in range(5):
        publisher.publish(make_solution())
    assert len(connection.of_type("ODOMETRY")) == 5
    assert connection.of_type("VISION_POSITION_ESTIMATE") == []


def test_vision_position_estimate_mode_never_emits_odometry():
    connection = FakeConnection()
    publisher = ExternalNavigationPublisher(
        connection, message="vision_position_estimate", send_to_autopilot=True
    )
    for _ in range(3):
        publisher.publish(make_solution())
    assert len(connection.of_type("VISION_POSITION_ESTIMATE")) == 3
    assert connection.of_type("ODOMETRY") == []


def test_vision_position_estimate_converts_the_quaternion_to_euler():
    connection = FakeConnection()
    publisher = ExternalNavigationPublisher(
        connection, message="vision_position_estimate", send_to_autopilot=True
    )
    roll, pitch, yaw = 0.1, -0.2, 1.3
    publisher.publish(
        make_solution(attitude=Attitude(*quaternion_from_euler(roll, pitch, yaw)))
    )

    message = connection.last("VISION_POSITION_ESTIMATE")
    assert message.roll == pytest.approx(roll, abs=1e-6)
    assert message.pitch == pytest.approx(pitch, abs=1e-6)
    assert message.yaw == pytest.approx(yaw, abs=1e-6)


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
def test_unix_timestamps_are_microseconds_since_the_epoch(publisher):
    publisher, connection = publisher
    publisher.publish(make_solution(timestamp_unix=1700000000.5))
    assert connection.last("ODOMETRY").time_usec == 1700000000500000


def test_boot_timestamps_are_small():
    """ArduPilot distinguishes the two conventions by magnitude."""
    connection = FakeConnection()
    publisher = ExternalNavigationPublisher(
        connection, message="odometry", send_to_autopilot=True, timestamp_source="boot"
    )
    publisher._boot_time_unix = 1700000000.0
    publisher.publish(make_solution(timestamp_unix=1700000012.0))
    assert connection.last("ODOMETRY").time_usec == 12000000
