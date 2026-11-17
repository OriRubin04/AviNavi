"""Injecting the visual position as a synthetic GPS fix (``GPS_INPUT``).

This is the primary output path for 1 MB flight controllers such as the
MATEKSYS F405-WING, where `ODOMETRY` handling is not compiled into the firmware
at all.  See ``docs/flight-controller-compatibility.md`` — that page is not
optional reading, because on a stock F405 build **none** of the external
navigation interfaces exist and the failures are silent.

``GPS_INPUT`` (message 232) hands ArduPilot a complete GNSS solution:
latitude, longitude, MSL altitude, NED velocity, accuracies and a satellite
count.  ArduPilot's `AP_GPS_MAV` backend (`GPS1_TYPE = 14`) presents it to the
EKF exactly as if it came from a receiver, so the whole normal GPS → EKF3 path
is used with `EK3_SRC1_*` left at its defaults.

Six details from `AP_GPS_MAV::handle_msg` that are easy to get wrong, each of
which produces silence rather than an error:

1. **``gps_id`` must equal the GPS *instance index*.** The handler's first
   statement is ``if (state.instance != packet.gps_id) { return; }``.  For
   `GPS1_TYPE = 14` that index is **0**, not 1.  This is the single most common
   integration bug.

2. **``ignore_flags`` is inverted from how it reads.** A flag that is *set*
   means the field is *absent*.  ArduPilot computes
   ``have_x = ((ignore_flags & FLAG) == 0)``.

3. **The vertical velocity is nested inside the horizontal block.** If you set
   `IGNORE_FLAG_VEL_HORIZ`, `vd` is dropped too, regardless of
   `IGNORE_FLAG_VEL_VERT`. Send all three or none.

4. **``time_week`` must be non-zero.** Jitter correction only runs when
   ``fix_type >= 3 && time_week > 0``; leave the week at zero and ArduPilot gets
   no corrected timestamp and no time-of-week, which degrades EKF time
   alignment.

5. **EKF3 wants 6 satellites and a passing HDOP** before it will align.
   `EK3_GPS_CHECK` defaults to 31, and `AP_NavEKF3_VehicleStatus` reports
   `"GPS numsats %u (needs 6)"`.  A synthetic fix must report a plausible
   constellation or the EKF will never use it.

6. **``yaw`` is centidegrees with 0 meaning "not available".**  True north is
   therefore `36000`, not `0`.

The accuracies are where the confidence model earns its keep: rather than
claiming a fixed quality, the horizontal accuracy sent to ArduPilot is derived
from the match statistics, so a marginal visual fix arrives looking marginal and
the EKF weights it accordingly.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional, Tuple

from ..geo import LocalOrigin
from ..types import GeoPoint, NavigationSolution
from .connection import MavlinkConnection, MavlinkConnectionError

LOG = logging.getLogger(__name__)

# GPS_INPUT_IGNORE_FLAGS. A set bit means "this field is NOT present".
IGNORE_ALT = 1
IGNORE_HDOP = 2
IGNORE_VDOP = 4
IGNORE_VEL_HORIZ = 8
IGNORE_VEL_VERT = 16
IGNORE_SPEED_ACCURACY = 32
IGNORE_HORIZONTAL_ACCURACY = 64
IGNORE_VERTICAL_ACCURACY = 128

#: GPS_FIX_TYPE_3D_FIX. ArduPilot's arming checks require at least a 3D fix.
GPS_FIX_TYPE_3D = 3
#: GPS_FIX_TYPE_NO_FIX — used to declare the visual solution unusable.
GPS_FIX_TYPE_NO_FIX = 1

#: GPS epoch (1980-01-06 00:00:00 UTC) as a UNIX timestamp, minus leap seconds.
#: Only the week number and millisecond-of-week need to be self-consistent for
#: ArduPilot's jitter correction, so the 18 s of accumulated leap seconds are
#: included for correctness rather than because anything depends on them.
GPS_EPOCH_UNIX = 315964800
GPS_LEAP_SECONDS = 18
SECONDS_PER_WEEK = 604800


def unix_to_gps_week_and_ms(unix_time: float) -> Tuple[int, int]:
    """Convert a UNIX timestamp to ``(gps_week, milliseconds_of_week)``."""
    gps_seconds = unix_time - GPS_EPOCH_UNIX + GPS_LEAP_SECONDS
    if gps_seconds < 0:
        return (0, 0)
    week = int(gps_seconds // SECONDS_PER_WEEK)
    into_week = gps_seconds - week * SECONDS_PER_WEEK
    return (week, int(round(into_week * 1000.0)))


class GpsInputPublisher:
    """Publishes the visual solution as MAVLink ``GPS_INPUT``.

    ``satellites`` and the DOP values are fabricated — they have to be, there is
    no constellation — but they are fabricated *consistently with the real
    uncertainty*: the satellite count is held at a plausible fixed value so the
    EKF's ``NSats`` check passes, while the accuracy fields carry the actual
    confidence-derived error estimate.  That keeps the honest signal in the
    field ArduPilot actually weights by (`horiz_accuracy`) rather than hiding it.
    """

    def __init__(
        self,
        connection: MavlinkConnection,
        origin: Optional[LocalOrigin] = None,
        send_to_autopilot: bool = False,
        gps_id: int = 0,
        satellites_visible: int = 12,
        hdop: float = 0.9,
        vdop: float = 1.3,
        send_velocity: bool = True,
        send_yaw: bool = False,
        #: Floor on the reported horizontal accuracy, metres.  ArduPilot and the
        #: EKF both behave badly when handed an implausibly small number.
        min_horizontal_accuracy_m: float = 0.5,
    ) -> None:
        self.connection = connection
        self.origin = origin
        self.send_to_autopilot = bool(send_to_autopilot)
        self.gps_id = int(gps_id)
        self.satellites_visible = int(satellites_visible)
        self.hdop = float(hdop)
        self.vdop = float(vdop)
        self.send_velocity = bool(send_velocity)
        self.send_yaw = bool(send_yaw)
        self.min_horizontal_accuracy_m = float(min_horizontal_accuracy_m)

        self._sent = 0
        self._suppressed = 0
        self._last_error: Optional[str] = None

        if self.satellites_visible < 6:
            LOG.warning(
                "satellites_visible is %d; EK3_GPS_CHECK requires 6 by default and "
                "the EKF will refuse to align",
                self.satellites_visible,
            )

    @classmethod
    def from_config(
        cls, connection: MavlinkConnection, config, origin: Optional[LocalOrigin] = None
    ) -> "GpsInputPublisher":
        return cls(
            connection=connection,
            origin=origin,
            send_to_autopilot=config.send_to_autopilot,
            gps_id=config.gps_input_id,
            satellites_visible=config.gps_input_satellites,
            hdop=config.gps_input_hdop,
            vdop=config.gps_input_vdop,
            send_velocity=config.send_velocity,
            send_yaw=config.gps_input_send_yaw,
        )

    # -- statistics ---------------------------------------------------------
    @property
    def message(self) -> str:
        return "gps_input"

    @property
    def enabled(self) -> bool:
        return self.send_to_autopilot

    @property
    def sent_count(self) -> int:
        return self._sent

    @property
    def suppressed_count(self) -> int:
        return self._suppressed

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def set_origin(self, origin: LocalOrigin) -> None:
        """Set the origin the local NED solution is converted against."""
        self.origin = origin

    # -- publishing ---------------------------------------------------------
    def publish(self, solution: Optional[NavigationSolution]) -> bool:
        """Publish a navigation solution as a synthetic GPS fix."""
        if solution is None:
            return False
        if not self.enabled:
            self._suppressed += 1
            return False
        if self.origin is None:
            self._suppressed += 1
            self._last_error = "no local origin; cannot convert NED to lat/lon"
            return False
        if not self.connection.connected:
            self._suppressed += 1
            self._last_error = "MAVLink link is down; GPS_INPUT suppressed"
            return False

        geo = self.origin.to_geo(solution.position_ned)
        # The EKF weights the fix by horiz_accuracy, so this is where the
        # confidence has to land to have any effect.
        horizontal_accuracy = max(
            self.min_horizontal_accuracy_m, float(solution.position_sigma_m)
        )
        # The vertical channel comes from image scale and is the weaker one.
        vertical_accuracy = 2.0 * horizontal_accuracy
        speed_accuracy = max(0.2, float(solution.velocity_sigma_mps))

        velocity_ned = None
        if self.send_velocity:
            velocity_ned = self._velocity_ned(solution)

        yaw_cdeg = 0
        if self.send_yaw:
            yaw_cdeg = self._yaw_cdeg(solution)

        try:
            self.send_gps_input(
                timestamp_unix=solution.timestamp_unix,
                geo=geo,
                velocity_ned=velocity_ned,
                horizontal_accuracy_m=horizontal_accuracy,
                vertical_accuracy_m=vertical_accuracy,
                speed_accuracy_mps=speed_accuracy,
                yaw_cdeg=yaw_cdeg,
            )
        except MavlinkConnectionError as exc:
            self._last_error = str(exc)
            LOG.error("GPS_INPUT send failed: %s", exc)
            return False

        self._sent += 1
        self._last_error = None
        return True

    def publish_failure(self) -> bool:
        """Report loss of the visual solution as a GPS with no fix.

        Honest and useful: ArduPilot sees a GPS that has dropped to
        ``GPS_FIX_TYPE_NO_FIX`` and reacts through its normal GPS-loss logic,
        rather than being left to time out on a stream that simply stopped.
        """
        if not self.enabled or self.origin is None:
            return False
        if not self.connection.connected:
            return False
        try:
            self.send_gps_input(
                timestamp_unix=time.time(),
                geo=self.origin.geo,
                velocity_ned=None,
                horizontal_accuracy_m=99.0,
                vertical_accuracy_m=99.0,
                speed_accuracy_mps=9.0,
                yaw_cdeg=0,
                fix_type=GPS_FIX_TYPE_NO_FIX,
                satellites_visible=0,
            )
        except MavlinkConnectionError as exc:
            self._last_error = str(exc)
            return False
        return True

    def send_gps_input(
        self,
        timestamp_unix: float,
        geo: GeoPoint,
        velocity_ned: Optional[Tuple[float, float, float]] = None,
        horizontal_accuracy_m: float = 1.0,
        vertical_accuracy_m: float = 2.0,
        speed_accuracy_mps: float = 0.5,
        yaw_cdeg: int = 0,
        fix_type: int = GPS_FIX_TYPE_3D,
        satellites_visible: Optional[int] = None,
    ) -> None:
        """Populate and send one ``GPS_INPUT`` message."""
        connection = self.connection
        week, week_ms = unix_to_gps_week_and_ms(timestamp_unix)

        ignore_flags = 0
        if velocity_ned is None:
            # Both velocity components must go together: ArduPilot nests the
            # vertical block inside the horizontal one.
            ignore_flags |= IGNORE_VEL_HORIZ | IGNORE_VEL_VERT
            vn = ve = vd = 0.0
        else:
            vn, ve, vd = (float(v) for v in velocity_ned)

        satellites = (
            self.satellites_visible if satellites_visible is None else satellites_visible
        )

        with connection.lock:
            connection.master.mav.gps_input_send(
                int(timestamp_unix * 1e6),
                # Must match the AP GPS *instance index*: 0 for GPS1_TYPE=14.
                self.gps_id,
                ignore_flags,
                week_ms,
                week,
                int(fix_type),
                int(round(geo.lat_deg * 1e7)),
                int(round(geo.lon_deg * 1e7)),
                float(geo.alt_m),
                float(self.hdop),
                float(self.vdop),
                float(vn),
                float(ve),
                float(vd),
                float(speed_accuracy_mps),
                float(horizontal_accuracy_m),
                float(vertical_accuracy_m),
                int(max(0, min(255, satellites))),
                # Extension field: centidegrees, 0 = unavailable, 36000 = north.
                int(max(0, min(36000, yaw_cdeg))),
            )
        connection.note_sent("GPS_INPUT")

    # -- helpers ------------------------------------------------------------
    def _velocity_ned(self, solution: NavigationSolution) -> Tuple[float, float, float]:
        """Rotate the solution's body-frame velocity back into NED.

        The propagator stores velocity in body FRD because that is what
        ``ODOMETRY`` needs; ``GPS_INPUT`` wants earth-fixed NED, so it has to
        come back out.
        """
        from ..frames import quaternion_to_matrix  # local import avoids a cycle

        R = quaternion_to_matrix(solution.attitude.as_tuple())
        ned = R @ list(solution.velocity_body_frd)
        return (float(ned[0]), float(ned[1]), float(ned[2]))

    def _yaw_cdeg(self, solution: NavigationSolution) -> int:
        """Vehicle yaw in centidegrees, with 0 reserved for 'unavailable'.

        A visually derived yaw is genuinely useful on an airframe with a poor
        compass, but note it is *heading*, not ground track — and ArduPilot will
        use it for yaw alignment, so an error here is not harmless.  Off by
        default for that reason.
        """
        from ..frames import quaternion_to_euler  # local import avoids a cycle

        _, _, yaw = quaternion_to_euler(solution.attitude.as_tuple())
        degrees = math.degrees(yaw) % 360.0
        centidegrees = int(round(degrees * 100.0))
        # 0 means "no yaw", so exact north must be sent as a full turn.
        return 36000 if centidegrees == 0 else centidegrees
