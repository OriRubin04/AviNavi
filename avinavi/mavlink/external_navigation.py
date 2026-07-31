"""Publishing the visual navigation solution to ArduPilot.

Primary interface: MAVLink ``ODOMETRY`` (message 331).  ``VISION_POSITION_ESTIMATE``
(102) is available as an optional prototype fallback.  Only one is ever active -
see :meth:`ExternalNavigationPublisher.__init__` - because two external
navigation sources feeding the same EKF fuse against each other.

Field semantics below are taken from the MAVLink common message definitions and
from ArduPilot's own receiving code (``GCS_MAVLINK::handle_odometry`` in
``libraries/GCS_MAVLink/GCS_Common.cpp``).  Three details there are easy to get
wrong and silently fatal:

1. **Frames are checked, and a mismatch is dropped without any error.**
   ArduPilot accepts ``ODOMETRY`` only when ``frame_id == MAV_FRAME_LOCAL_FRD``
   and ``child_frame_id == MAV_FRAME_BODY_FRD``.  Sending ``MAV_FRAME_LOCAL_NED``
   - which reads more naturally for a north/east/down position - makes the
   message vanish with no warning at either end.  ArduPilot treats the position
   in that "local FRD" frame as ordinary north/east/down relative to the EKF
   origin, so the numbers we put in are NED numbers.

2. **Velocity is in the child frame, i.e. body FRD.**  ArduPilot rotates
   ``(vx, vy, vz)`` by the message quaternion to recover an NED velocity, so an
   NED velocity placed directly in those fields is wrong by the vehicle's
   attitude.  :func:`avinavi.frames.ned_velocity_to_body_frd` does the inverse
   rotation for us.

3. **Covariance is the packed upper triangle.**  ArduPilot reads the position
   variances from indices 0, 6 and 11 and the angular variances from 15, 18 and
   20, then takes ``posErr = sqrt(c[0]+c[6]+c[11])`` and
   ``angErr = sqrt(c[15]+c[18]+c[20])``.  Anything else in the array is ignored,
   but those six entries directly set how much the EKF trusts the fix.

The quaternion is ``(w, x, y, z)`` and rotates body FRD into local NED.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional, Sequence

from ..types import NavigationSolution, covariance_upper_triangle
from .connection import MavlinkConnection, MavlinkConnectionError

LOG = logging.getLogger(__name__)

#: MAV_FRAME values ArduPilot requires in ODOMETRY.  Hard requirement, not a
#: preference - see the module docstring.
MAV_FRAME_LOCAL_FRD = 20
MAV_FRAME_BODY_FRD = 12

#: MAV_ESTIMATOR_TYPE_VISION.  Use VIO (3) if IMU fusion is added later.
MAV_ESTIMATOR_TYPE_VISION = 2

#: ODOMETRY.quality: -1 means the odometry has failed.
QUALITY_FAILED = -1


class ExternalNavigationPublisher:
    """Sends exactly one kind of external-navigation message.

    ``message`` selects the interface:

    ``"odometry"``                   the production path
    ``"vision_position_estimate"``   prototype/compatibility path
    ``"none"``                       compute and log, transmit nothing

    ``send_to_autopilot`` is a separate master switch so an operator can run the
    whole stack, watch the diagnostics, and still be certain the flight
    controller is receiving nothing.
    """

    def __init__(
        self,
        connection: MavlinkConnection,
        message: str = "odometry",
        send_to_autopilot: bool = False,
        send_velocity: bool = True,
        send_quality: bool = True,
        timestamp_source: str = "unix",
    ) -> None:
        if message not in ("odometry", "vision_position_estimate", "none"):
            raise ValueError("unknown external navigation message %r" % message)
        self.connection = connection
        self.message = message
        self.send_to_autopilot = bool(send_to_autopilot)
        self.send_velocity = bool(send_velocity)
        self.send_quality = bool(send_quality)
        self.timestamp_source = timestamp_source

        self._boot_time_unix = time.time()
        self._sent = 0
        self._suppressed = 0
        self._last_send_monotonic: Optional[float] = None
        self._last_error: Optional[str] = None

    @classmethod
    def from_config(cls, connection: MavlinkConnection, config) -> "ExternalNavigationPublisher":
        return cls(
            connection=connection,
            message=config.message,
            send_to_autopilot=config.send_to_autopilot,
            send_velocity=config.send_velocity,
            send_quality=config.send_quality,
            timestamp_source=config.timestamp_source,
        )

    # -- statistics ---------------------------------------------------------
    @property
    def sent_count(self) -> int:
        return self._sent

    @property
    def suppressed_count(self) -> int:
        """Solutions we deliberately did not transmit (disarmed output, no link)."""
        return self._suppressed

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def enabled(self) -> bool:
        return self.send_to_autopilot and self.message != "none"

    # -- the public entry point --------------------------------------------
    def publish(self, solution: Optional[NavigationSolution]) -> bool:
        """Publish a navigation solution.  Returns True if a message went out.

        ``None`` means "we have nothing valid right now" and results in silence,
        which is the correct behaviour: ArduPilot's own external-navigation
        timeout then handles the gap.  We never resend a stale pose to keep the
        rate up.
        """
        if solution is None:
            return False
        if not self.enabled:
            self._suppressed += 1
            return False
        if not self.connection.connected:
            self._suppressed += 1
            self._last_error = "MAVLink link is down; external navigation suppressed"
            return False

        timestamp_us = self._timestamp_us(solution)
        try:
            if self.message == "odometry":
                self.send_visual_odometry(
                    timestamp=timestamp_us,
                    position=solution.position_ned.as_tuple(),
                    orientation=solution.attitude.as_tuple(),
                    velocity=solution.velocity_body_frd if self.send_velocity else None,
                    angular_velocity=solution.angular_velocity_body,
                    pose_covariance=pose_covariance_from_sigmas(
                        solution.position_sigma_m, solution.yaw_sigma_rad
                    ),
                    velocity_covariance=velocity_covariance_from_sigma(
                        solution.velocity_sigma_mps
                    )
                    if self.send_velocity
                    else None,
                    reset_counter=solution.reset_counter,
                    quality=solution.quality_percent if self.send_quality else 0,
                )
            else:
                self.send_vision_position_estimate(
                    timestamp=timestamp_us,
                    position=solution.position_ned.as_tuple(),
                    orientation=solution.attitude.as_tuple(),
                    pose_covariance=pose_covariance_from_sigmas(
                        solution.position_sigma_m, solution.yaw_sigma_rad
                    ),
                    reset_counter=solution.reset_counter,
                )
        except MavlinkConnectionError as exc:
            self._last_error = str(exc)
            LOG.error("external navigation send failed: %s", exc)
            return False

        self._sent += 1
        self._last_send_monotonic = time.monotonic()
        self._last_error = None
        return True

    def publish_failure(self) -> bool:
        """Tell ArduPilot the odometry source has failed, once, explicitly.

        ``ODOMETRY.quality = -1`` is the specification's "odometry has failed"
        value, and ArduPilot's ``VISO_QUAL_MIN`` gate rejects the sample rather
        than fusing it.  This is strictly better than going silent when we know
        the solution is bad rather than merely late - but it is only sent when
        we actually have a last-known pose to attach it to.
        """
        if not self.enabled or self.message != "odometry":
            return False
        if not self.connection.connected:
            return False
        try:
            self.send_visual_odometry(
                timestamp=self._now_us(),
                position=(0.0, 0.0, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                velocity=None,
                pose_covariance=None,
                velocity_covariance=None,
                reset_counter=0,
                quality=QUALITY_FAILED,
            )
        except MavlinkConnectionError as exc:
            self._last_error = str(exc)
            return False
        return True

    # -- the message builders ----------------------------------------------
    def send_visual_odometry(
        self,
        timestamp: int,
        position: Sequence[float],
        orientation: Sequence[float],
        velocity: Optional[Sequence[float]] = None,
        angular_velocity: Optional[Sequence[float]] = None,
        pose_covariance: Optional[Sequence[float]] = None,
        velocity_covariance: Optional[Sequence[float]] = None,
        reset_counter: int = 0,
        quality: int = 0,
    ) -> None:
        """Populate and send a MAVLink ``ODOMETRY`` message.

        :param timestamp: microseconds, UNIX epoch or time since boot
        :param position: ``(north, east, down)`` metres, relative to the EKF origin
        :param orientation: quaternion ``(w, x, y, z)``, body FRD -> local NED
        :param velocity: ``(vx, vy, vz)`` m/s in **body FRD**, or None
        :param angular_velocity: ``(roll, pitch, yaw)`` rates rad/s in body FRD
        :param pose_covariance: 21-element packed upper triangle, or None for unknown
        :param velocity_covariance: 21-element packed upper triangle, or None
        :param reset_counter: incremented on estimate discontinuities
        :param quality: -1 failed, 0 unknown, 1 worst .. 100 best
        """
        connection = self.connection
        mavlink = connection.mavutil.mavlink

        unknown_covariance = [float("nan")] + [0.0] * 20
        pose_cov = list(pose_covariance) if pose_covariance is not None else unknown_covariance
        velocity_cov = (
            list(velocity_covariance)
            if velocity_covariance is not None
            else list(unknown_covariance)
        )
        _check_covariance(pose_cov, "pose_covariance")
        _check_covariance(velocity_cov, "velocity_covariance")

        vx, vy, vz = velocity if velocity is not None else (0.0, 0.0, 0.0)
        rollspeed, pitchspeed, yawspeed = (
            angular_velocity if angular_velocity is not None else (0.0, 0.0, 0.0)
        )
        w, qx, qy, qz = orientation

        with connection.lock:
            connection.master.mav.odometry_send(
                int(timestamp),
                # ArduPilot rejects anything but these two frames.
                MAV_FRAME_LOCAL_FRD,
                MAV_FRAME_BODY_FRD,
                float(position[0]),
                float(position[1]),
                float(position[2]),
                [float(w), float(qx), float(qy), float(qz)],
                float(vx),
                float(vy),
                float(vz),
                float(rollspeed),
                float(pitchspeed),
                float(yawspeed),
                pose_cov,
                velocity_cov,
                int(reset_counter) & 0xFF,
                MAV_ESTIMATOR_TYPE_VISION,
                int(max(-1, min(100, quality))),
            )
        connection.note_sent("ODOMETRY")

    def send_vision_position_estimate(
        self,
        timestamp: int,
        position: Sequence[float],
        orientation: Sequence[float],
        pose_covariance: Optional[Sequence[float]] = None,
        reset_counter: int = 0,
    ) -> None:
        """Send ``VISION_POSITION_ESTIMATE`` - the optional prototype interface.

        Carries roll/pitch/yaw as Euler angles rather than a quaternion and has
        no velocity fields, so it is strictly less informative than ``ODOMETRY``.
        Useful for bring-up on an ArduPilot build where ``ODOMETRY`` handling is
        not compiled in.  Position is in local NED metres relative to the EKF
        origin, same as ``ODOMETRY``.
        """
        from ..frames import quaternion_to_euler  # local import avoids a cycle

        connection = self.connection
        roll, pitch, yaw = quaternion_to_euler(orientation)

        covariance = (
            list(pose_covariance)
            if pose_covariance is not None
            else [float("nan")] + [0.0] * 20
        )
        _check_covariance(covariance, "covariance")

        with connection.lock:
            connection.master.mav.vision_position_estimate_send(
                int(timestamp),
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(roll),
                float(pitch),
                float(yaw),
                covariance,
                int(reset_counter) & 0xFF,
            )
        connection.note_sent("VISION_POSITION_ESTIMATE")

    # -- timestamps ---------------------------------------------------------
    def _timestamp_us(self, solution: NavigationSolution) -> int:
        """Microsecond timestamp for the message.

        ArduPilot's ``correct_offboard_timestamp_usec_to_ms`` copes with either
        a UNIX epoch timestamp or a time-since-boot one, distinguishing them by
        magnitude, and estimates the offset to its own clock.  UNIX epoch is the
        default here because it survives a companion-computer restart without
        the offset estimator having to relearn.
        """
        if self.timestamp_source == "boot":
            return int(max(0.0, solution.timestamp_unix - self._boot_time_unix) * 1e6)
        return int(solution.timestamp_unix * 1e6)

    def _now_us(self) -> int:
        if self.timestamp_source == "boot":
            return int(max(0.0, time.time() - self._boot_time_unix) * 1e6)
        return int(time.time() * 1e6)


def pose_covariance_from_sigmas(
    position_sigma_m: float,
    yaw_sigma_rad: float,
    attitude_sigma_rad: Optional[float] = None,
) -> list:
    """Build a ``pose_covariance`` array from position and angle sigmas.

    States are ``(x, y, z, roll, pitch, yaw)``.  Vertical position gets a
    slightly larger variance than horizontal because it comes from the image
    scale (or an external height source), which is the weaker of the two
    channels.  Roll and pitch are not estimated visually, so unless an explicit
    ``attitude_sigma_rad`` is given they are marked with a large variance to
    tell the EKF not to lean on them.
    """
    sigma = max(float(position_sigma_m), 1e-3)
    yaw_sigma = max(float(yaw_sigma_rad), 1e-3)
    attitude_sigma = (
        max(float(attitude_sigma_rad), 1e-3) if attitude_sigma_rad is not None else 1.0
    )
    horizontal_variance = sigma * sigma
    # The vertical channel is the least well constrained; 2x sigma = 4x variance.
    vertical_variance = (2.0 * sigma) ** 2
    return covariance_upper_triangle(
        [
            horizontal_variance,
            horizontal_variance,
            vertical_variance,
            attitude_sigma * attitude_sigma,
            attitude_sigma * attitude_sigma,
            yaw_sigma * yaw_sigma,
        ]
    )


def velocity_covariance_from_sigma(
    velocity_sigma_mps: float, angular_rate_sigma: float = 1.0
) -> list:
    """Build a ``velocity_covariance`` array from a single speed sigma."""
    sigma = max(float(velocity_sigma_mps), 1e-3)
    variance = sigma * sigma
    rate_variance = max(float(angular_rate_sigma), 1e-3) ** 2
    return covariance_upper_triangle(
        [variance, variance, variance, rate_variance, rate_variance, rate_variance]
    )


def _check_covariance(values: Sequence[float], name: str) -> None:
    if len(values) != 21:
        raise ValueError(
            "%s must have 21 elements (packed 6x6 upper triangle), got %d"
            % (name, len(values))
        )
    if not math.isnan(values[0]) and any(
        values[index] < 0.0 for index in (0, 6, 11, 15, 18, 20)
    ):
        raise ValueError("%s has a negative variance on the diagonal" % name)
