"""Rotations, quaternions and the camera -> body -> NED chain."""

from __future__ import annotations

import math

import numpy as np
import pytest

from avinavi.frames import (
    CameraExtrinsics,
    euler_to_matrix,
    matrix_to_euler,
    matrix_to_quaternion,
    ned_velocity_to_body_frd,
    normalize_quaternion,
    quaternion_from_euler,
    quaternion_multiply,
    quaternion_to_euler,
    quaternion_to_matrix,
    rotate_vector,
    wrap_pi,
    yaw_from_camera_axis_bearing,
)

ANGLE_SETS = [
    (0.0, 0.0, 0.0),
    (0.1, -0.2, 0.3),
    (0.0, 0.0, math.pi / 2),
    (-0.4, 0.35, -2.5),
    (0.5, 0.5, 3.0),
]


@pytest.mark.parametrize("rpy", ANGLE_SETS)
def test_euler_matrix_round_trip(rpy):
    recovered = matrix_to_euler(euler_to_matrix(*rpy))
    for actual, expected in zip(recovered, rpy):
        assert wrap_pi(actual - expected) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("rpy", ANGLE_SETS)
def test_quaternion_matrix_agreement(rpy):
    matrix = euler_to_matrix(*rpy)
    assert np.allclose(quaternion_to_matrix(quaternion_from_euler(*rpy)), matrix, atol=1e-9)
    assert np.allclose(quaternion_to_matrix(matrix_to_quaternion(matrix)), matrix, atol=1e-9)


@pytest.mark.parametrize("rpy", ANGLE_SETS)
def test_quaternion_euler_round_trip(rpy):
    recovered = quaternion_to_euler(quaternion_from_euler(*rpy))
    for actual, expected in zip(recovered, rpy):
        assert wrap_pi(actual - expected) == pytest.approx(0.0, abs=1e-9)


def test_matrix_to_quaternion_handles_180_degree_rotations():
    """The largest-component branch must stay stable where the trace is negative."""
    for axis in range(3):
        rpy = [0.0, 0.0, 0.0]
        rpy[axis] = math.pi
        matrix = euler_to_matrix(*rpy)
        q = matrix_to_quaternion(matrix)
        assert np.allclose(quaternion_to_matrix(q), matrix, atol=1e-8)


def test_yaw_rotates_north_towards_east():
    """A 90 degree yaw must send body forward (north) to east."""
    matrix = euler_to_matrix(0.0, 0.0, math.pi / 2)
    forward_in_ned = matrix @ np.array([1.0, 0.0, 0.0])
    assert forward_in_ned == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_quaternion_multiply_composes_rotations():
    a = quaternion_from_euler(0.0, 0.0, math.pi / 4)
    b = quaternion_from_euler(0.0, 0.0, math.pi / 4)
    combined = quaternion_multiply(a, b)
    assert quaternion_to_euler(combined)[2] == pytest.approx(math.pi / 2, abs=1e-9)


def test_normalize_quaternion_handles_a_zero_input():
    assert normalize_quaternion((0.0, 0.0, 0.0, 0.0)) == (1.0, 0.0, 0.0, 0.0)


def test_identity_extrinsics_put_the_optical_axis_down():
    """The default mount is nadir: camera +Z must be body +Z (down)."""
    extrinsics = CameraExtrinsics()
    assert extrinsics.camera_to_body([0.0, 0.0, 1.0]) == pytest.approx([0.0, 0.0, 1.0])


def test_identity_extrinsics_put_image_x_along_the_nose():
    """Pins the physical meaning of the identity mount.

    Identity makes the camera axes coincide with the body axes, so image +X
    (right across the picture) points along body +X (the nose) and image +Y
    (down the picture) along body +Y (the right wing).  Because image +Y is
    *down* the picture, the nose appears at the right edge of the frame.

    This test exists because the documentation originally claimed the opposite,
    and a 90-degree error here shows up as a yaw bias the EKF fights rather
    than rejects.
    """
    extrinsics = CameraExtrinsics()
    assert extrinsics.camera_to_body([1.0, 0.0, 0.0]) == pytest.approx([1.0, 0.0, 0.0])
    assert extrinsics.camera_to_body([0.0, 1.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])


@pytest.mark.parametrize(
    "yaw_deg,image_x_in_body",
    [
        (0.0, [1.0, 0.0, 0.0]),     # nose at the right edge
        (90.0, [0.0, 1.0, 0.0]),    # nose at the top edge
        (180.0, [-1.0, 0.0, 0.0]),  # nose at the left edge
        (-90.0, [0.0, -1.0, 0.0]),  # nose at the bottom edge
    ],
)
def test_mount_yaw_table_matches_the_documentation(yaw_deg, image_x_in_body):
    """The four mountings tabulated in CameraExtrinsics' docstring."""
    extrinsics = CameraExtrinsics(yaw_deg=yaw_deg)
    assert extrinsics.camera_to_body([1.0, 0.0, 0.0]) == pytest.approx(
        image_x_in_body, abs=1e-9
    )


@pytest.mark.parametrize("mount_yaw_deg", [0.0, 37.0, 90.0, -125.0])
@pytest.mark.parametrize("vehicle_yaw", [0.0, 0.7, -2.5, 3.0])
def test_yaw_recovery_inverts_the_forward_projection(mount_yaw_deg, vehicle_yaw):
    """Whatever bearing the image +X axis has, we must recover the true yaw."""
    extrinsics = CameraExtrinsics(yaw_deg=mount_yaw_deg)
    R_ned_from_body = euler_to_matrix(0.0, 0.0, vehicle_yaw)
    axis_in_ned = R_ned_from_body @ extrinsics.R_body_from_camera @ np.array([1.0, 0.0, 0.0])
    bearing = math.atan2(axis_in_ned[1], axis_in_ned[0])

    recovered = yaw_from_camera_axis_bearing(bearing, extrinsics)
    assert wrap_pi(recovered - vehicle_yaw) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("roll,pitch", [(0.0, 0.0), (0.15, -0.1), (-0.2, 0.25)])
def test_yaw_recovery_accounts_for_tilt(roll, pitch):
    extrinsics = CameraExtrinsics(yaw_deg=45.0)
    vehicle_yaw = 1.1
    R_ned_from_body = euler_to_matrix(roll, pitch, vehicle_yaw)
    axis_in_ned = R_ned_from_body @ extrinsics.R_body_from_camera @ np.array([1.0, 0.0, 0.0])
    bearing = math.atan2(axis_in_ned[1], axis_in_ned[0])

    recovered = yaw_from_camera_axis_bearing(
        bearing, extrinsics, roll_rad=roll, pitch_rad=pitch
    )
    assert wrap_pi(recovered - vehicle_yaw) == pytest.approx(0.0, abs=1e-9)


def test_yaw_is_unobservable_when_the_camera_x_axis_is_vertical():
    """A camera whose +X points straight down cannot fix yaw; say so loudly."""
    extrinsics = CameraExtrinsics(pitch_deg=90.0)
    with pytest.raises(ValueError, match="unobservable"):
        yaw_from_camera_axis_bearing(0.0, extrinsics)


def test_body_origin_offset_uses_the_vehicle_attitude():
    """The camera lever arm must rotate with the aircraft, not stay in NED."""
    extrinsics = CameraExtrinsics(translation_m=(1.0, 0.0, 0.0))
    # Nose east: the camera 1 m forward sits 1 m east of the body origin.
    R = euler_to_matrix(0.0, 0.0, math.pi / 2)
    body = extrinsics.body_origin_from_camera_position([0.0, 10.0, -50.0], R)
    assert body == pytest.approx([0.0, 9.0, -50.0], abs=1e-9)


def test_ned_velocity_to_body_frd_is_the_inverse_rotation():
    """ArduPilot rotates ODOMETRY velocity by q, so we must un-rotate first."""
    yaw = math.pi / 2
    q = quaternion_from_euler(0.0, 0.0, yaw)
    # Flying north at 5 m/s while pointing east: that is 5 m/s to the *right*.
    body = ned_velocity_to_body_frd((5.0, 0.0, 0.0), q)
    assert body == pytest.approx((0.0, -5.0, 0.0), abs=1e-9)
    # And rotating back by q recovers the NED velocity, exactly as ArduPilot does.
    assert rotate_vector(q, body) == pytest.approx((5.0, 0.0, 0.0), abs=1e-9)


def test_wrap_pi_bounds():
    assert wrap_pi(3.5 * math.pi) == pytest.approx(-0.5 * math.pi, abs=1e-9)
    assert -math.pi <= wrap_pi(100.0) < math.pi
