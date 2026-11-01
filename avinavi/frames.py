"""Rotation utilities and the camera -> body -> navigation frame chain.

Frames used in AviNavi (see ``docs/coordinate-frames.md`` for diagrams):

``camera optical``
    X right across the image, Y down the image, Z forward along the optical
    axis.  This is the OpenCV convention and the frame SuperPoint keypoints
    live in.

``body FRD``
    X forward (nose), Y right (starboard wing), Z down.  ArduPilot's body
    frame, and MAVLink's ``MAV_FRAME_BODY_FRD``.

``local NED``
    X north, Y east, Z down, anchored at the EKF origin.  MAVLink calls the
    variant ArduPilot expects in ``ODOMETRY`` ``MAV_FRAME_LOCAL_FRD``.

The extrinsic ``R_body_from_camera`` is a *fixed* rotation configured in
``camera_extrinsics``.  Identity means the camera axes coincide with the body
axes: optical axis (+Z) straight down, image +X along the nose, image +Y along
the right wing.  Note what that implies for the picture: image +Y is *down the
image*, so with identity extrinsics the nose appears at the **right edge** of
the frame.  The more usual physical mount - camera rotated so the nose is at the
**top** of the frame - is ``yaw_deg: 90``.  See ``docs/coordinate-frames.md``
for the full table.

Quaternions are stored ``(w, x, y, z)`` - MAVLink's ordering - and always
represent a rotation *from* the frame named second *to* the frame named first,
e.g. ``q_ned_from_body`` rotates a body-frame vector into NED.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Elementary rotations
# ---------------------------------------------------------------------------
def rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_to_matrix(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Aerospace 3-2-1 (yaw-pitch-roll) Euler angles -> rotation matrix.

    The returned matrix takes a vector in the rotated (body) frame into the
    reference (NED) frame: ``v_ned = R @ v_body``.  This is the same convention
    ArduPilot uses for roll/pitch/yaw in ``ATTITUDE``.
    """
    return rot_z(yaw_rad) @ rot_y(pitch_rad) @ rot_x(roll_rad)


def matrix_to_euler(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Rotation matrix -> ``(roll, pitch, yaw)`` in radians, 3-2-1 convention."""
    m = np.asarray(matrix, dtype=float)
    # Clamp guards against a slightly non-orthonormal matrix pushing asin out of range.
    sin_pitch = -float(np.clip(m[2, 0], -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    if abs(m[2, 0]) > 1.0 - 1e-9:
        # Gimbal lock: roll and yaw are not separable; attribute all to yaw.
        roll = 0.0
        yaw = math.atan2(-m[0, 1], m[1, 1])
    else:
        roll = math.atan2(m[2, 1], m[2, 2])
        yaw = math.atan2(m[1, 0], m[0, 0])
    return (roll, pitch, yaw)


# ---------------------------------------------------------------------------
# Quaternions, MAVLink (w, x, y, z) ordering
# ---------------------------------------------------------------------------
def quaternion_from_euler(
    roll_rad: float, pitch_rad: float, yaw_rad: float
) -> Quaternion:
    """3-2-1 Euler angles -> unit quaternion ``(w, x, y, z)``."""
    cr, sr = math.cos(roll_rad * 0.5), math.sin(roll_rad * 0.5)
    cp, sp = math.cos(pitch_rad * 0.5), math.sin(pitch_rad * 0.5)
    cy, sy = math.cos(yaw_rad * 0.5), math.sin(yaw_rad * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def quaternion_to_euler(q: Sequence[float]) -> Tuple[float, float, float]:
    """Unit quaternion ``(w, x, y, z)`` -> ``(roll, pitch, yaw)`` radians."""
    w, x, y, z = normalize_quaternion(q)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return (roll, pitch, yaw)


def quaternion_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Unit quaternion ``(w, x, y, z)`` -> rotation matrix."""
    w, x, y, z = normalize_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quaternion(matrix: np.ndarray) -> Quaternion:
    """Rotation matrix -> unit quaternion ``(w, x, y, z)``.

    Uses Shepperd's largest-component branch so the result stays well
    conditioned for any input rotation, including 180-degree turns.
    """
    m = np.asarray(matrix, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize_quaternion((w, x, y, z))


def normalize_quaternion(q: Sequence[float]) -> Quaternion:
    w, x, y, z = (float(v) for v in q)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / norm, x / norm, y / norm, z / norm)


def quaternion_multiply(a: Sequence[float], b: Sequence[float]) -> Quaternion:
    """Hamilton product ``a * b`` (apply ``b`` first, then ``a``)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quaternion_conjugate(q: Sequence[float]) -> Quaternion:
    w, x, y, z = q
    return (w, -x, -y, -z)


def rotate_vector(q: Sequence[float], v: Sequence[float]) -> Vector3:
    """Rotate ``v`` by quaternion ``q``."""
    result = quaternion_to_matrix(q) @ np.asarray(v, dtype=float)
    return (float(result[0]), float(result[1]), float(result[2]))


def wrap_pi(angle_rad: float) -> float:
    """Wrap an angle into [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Camera extrinsics
# ---------------------------------------------------------------------------
class CameraExtrinsics:
    """The fixed camera-to-body transform, built once from configuration.

    ``rotation`` in the config is applied as ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``
    and, by definition, maps a vector expressed in the **camera optical frame**
    into the **body FRD frame**::

        v_body = R_body_from_camera @ v_camera

    The identity rotation therefore makes the camera axes coincide with the body
    axes: optical axis down (a nadir mount), image +X along the nose, image +Y
    along the right wing.  Because image +Y points *down the image*, identity
    puts the nose at the right edge of the picture.

    The common mountings, for a downward-facing camera:

    ======================  ==========  =========================
    Nose appears at         ``yaw_deg``  image +X points along
    ======================  ==========  =========================
    right edge              0            body +X (forward)
    top edge                90           body +Y (right)
    left edge               180          body -X (aft)
    bottom edge             270 / -90    body -Y (left)
    ======================  ==========  =========================

    Get this wrong and the recovered yaw is off by a multiple of 90 degrees,
    which the EKF will fight rather than reject.  The cheapest check is the
    bench test in ``docs/safety-and-testing.md``: carry the aircraft a measured
    distance and confirm the reported position moves that way.
    """

    def __init__(
        self,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
        translation_m: Vector3 = (0.0, 0.0, 0.0),
    ) -> None:
        self.roll_rad = math.radians(roll_deg)
        self.pitch_rad = math.radians(pitch_deg)
        self.yaw_rad = math.radians(yaw_deg)
        self.R_body_from_camera = euler_to_matrix(
            self.roll_rad, self.pitch_rad, self.yaw_rad
        )
        self.R_camera_from_body = self.R_body_from_camera.T
        self.translation_body = np.asarray(translation_m, dtype=float)

    @classmethod
    def from_config(cls, config) -> "CameraExtrinsics":
        return cls(
            roll_deg=config.rotation.roll_deg,
            pitch_deg=config.rotation.pitch_deg,
            yaw_deg=config.rotation.yaw_deg,
            translation_m=(
                config.translation.x_m,
                config.translation.y_m,
                config.translation.z_m,
            ),
        )

    def camera_to_body(self, v_camera: Sequence[float]) -> np.ndarray:
        return self.R_body_from_camera @ np.asarray(v_camera, dtype=float)

    def body_to_camera(self, v_body: Sequence[float]) -> np.ndarray:
        return self.R_camera_from_body @ np.asarray(v_body, dtype=float)

    def R_ned_from_camera(self, R_ned_from_body: np.ndarray) -> np.ndarray:
        """Compose the vehicle attitude with the fixed mount rotation."""
        return np.asarray(R_ned_from_body, dtype=float) @ self.R_body_from_camera

    def body_origin_from_camera_position(
        self, camera_position_ned: Sequence[float], R_ned_from_body: np.ndarray
    ) -> np.ndarray:
        """Shift a camera-centre position to the body origin.

        The camera sits at ``translation_body`` in the body frame, so::

            p_body_origin_ned = p_camera_ned - R_ned_from_body @ t_camera_body
        """
        offset = np.asarray(R_ned_from_body, dtype=float) @ self.translation_body
        return np.asarray(camera_position_ned, dtype=float) - offset


def yaw_from_camera_axis_bearing(
    axis_bearing_rad: float,
    extrinsics: CameraExtrinsics,
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
) -> float:
    """Recover vehicle yaw from the measured ground bearing of the camera +X axis.

    The homography against a north-up map tile tells us where the camera image
    +X axis points in the world (its bearing clockwise from north).  Writing the
    vehicle attitude as ``R_ned_from_body = Rz(yaw) @ R_level`` with
    ``R_level = Ry(pitch) @ Rx(roll)``, the camera +X axis in NED is::

        v_ned = Rz(yaw) @ R_level @ R_body_from_camera @ [1, 0, 0]

    Let ``w = R_level @ R_body_from_camera @ [1, 0, 0]``.  Because ``Rz`` only
    rotates the horizontal components, the measured bearing is
    ``atan2(w_y, w_x) + yaw``, hence the subtraction below.  Roll and pitch come
    from the autopilot's ``ATTITUDE`` message when available and default to
    level flight otherwise.
    """
    R_level = rot_y(pitch_rad) @ rot_x(roll_rad)
    w = R_level @ extrinsics.R_body_from_camera @ np.array([1.0, 0.0, 0.0])
    if abs(w[0]) < 1e-9 and abs(w[1]) < 1e-9:
        raise ValueError(
            "camera +X axis is vertical in the body frame; yaw is unobservable "
            "with this extrinsic calibration"
        )
    return wrap_pi(axis_bearing_rad - math.atan2(w[1], w[0]))


def ned_velocity_to_body_frd(
    velocity_ned: Sequence[float], q_ned_from_body: Sequence[float]
) -> Vector3:
    """Rotate an NED velocity into the body FRD frame.

    MAVLink ``ODOMETRY`` carries ``vx/vy/vz`` in ``child_frame_id``, which
    ArduPilot requires to be ``MAV_FRAME_BODY_FRD``: its ``handle_odometry``
    rotates the vector by the message quaternion to recover NED, so we must
    apply the inverse here.
    """
    R = quaternion_to_matrix(q_ned_from_body)
    body = R.T @ np.asarray(velocity_ned, dtype=float)
    return (float(body[0]), float(body[1]), float(body[2]))
