"""Getting the operator-selected starting position onto the Raspberry Pi.

The operator picks one point on the Mission Planner map before the mission.
That point does two jobs:

1. it seeds the map-tile search window, so the first localization attempt looks
   in the right few hundred metres of map;
2. it anchors the local NED frame that ``ODOMETRY`` positions are expressed in.

Job 2 is the reason for the mechanism chosen here.  ArduPilot's EKF has its own
notion of where local ``(0, 0, 0)`` is - the *EKF origin* - and every external
navigation position we send is interpreted relative to it.  If the Pi picked its
own origin, every fix would be offset by the difference between the two, and the
EKF would reject or badly fuse the data.  So rather than inventing an origin, we
read the autopilot's.

Chosen mechanism
----------------
**Listen for ``GPS_GLOBAL_ORIGIN`` (message 49).**

In Mission Planner the operator right-clicks the map and chooses
*Set Home Here -> Set EKF Origin Here*.  Mission Planner sends the origin to the
autopilot (historically as ``SET_GPS_GLOBAL_ORIGIN``, message 48, now superseded
by ``MAV_CMD_DO_SET_GLOBAL_ORIGIN``, command 611), and per the message
specification the vehicle "should emit ``GPS_GLOBAL_ORIGIN`` irrespective of
whether the origin is changed".  ArduPilot does exactly that.  The Pi simply
subscribes to that broadcast.

Why this and not the alternatives:

* it is a **standard MAVLink message**, no custom protocol and no new dialect;
* it is the *same* number ArduPilot's EKF is using, so the frames cannot drift
  apart - which a separately-transmitted "start position" could not guarantee;
* the operator interaction already exists in stock Mission Planner, with no
  plugin or Lua script;
* it works identically in GPS-denied flight, because the origin is set by hand
  rather than derived from a fix.

Because ``GPS_GLOBAL_ORIGIN`` is emitted on change (and on request) rather than
streamed, the provider also actively asks for it at startup with
``MAV_CMD_REQUEST_MESSAGE``, so a system that booted before we did still answers.

Fallbacks
---------
``initial_position.source: manual`` takes latitude/longitude/altitude straight
from the configuration file, for bench work with no autopilot attached.
``auto`` tries MAVLink first and falls back to the configured values on timeout.
``push_origin_to_autopilot`` covers the reverse case - the Pi knows the origin
and the autopilot does not - by sending ``MAV_CMD_DO_SET_GLOBAL_ORIGIN``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from ..geo import LocalOrigin
from ..types import GeoPoint
from .connection import MavlinkConnection, MavlinkConnectionError

LOG = logging.getLogger(__name__)

#: MAV_CMD_REQUEST_MESSAGE - ask for a one-off send of a given message ID.
MAV_CMD_REQUEST_MESSAGE = 512
#: MAV_CMD_DO_SET_GLOBAL_ORIGIN - the current way to set the vehicle's local
#: origin; supersedes SET_GPS_GLOBAL_ORIGIN (message 48).
MAV_CMD_DO_SET_GLOBAL_ORIGIN = 611
#: GPS_GLOBAL_ORIGIN message ID, used with MAV_CMD_REQUEST_MESSAGE.
MAVLINK_MSG_ID_GPS_GLOBAL_ORIGIN = 49

#: How often to have the autopilot re-send the origin, in microseconds, via
#: MAV_CMD_SET_MESSAGE_INTERVAL.  Slow on purpose: this is a safety net against
#: a dropped packet, not a data stream.
ORIGIN_REFRESH_INTERVAL_US = 5_000_000.0


class InitialPositionError(RuntimeError):
    """No usable starting position could be obtained."""


@dataclass
class InitialPosition:
    """The operator-selected start, and where it came from."""

    geo: GeoPoint
    source: str
    received_monotonic: float

    @property
    def origin(self) -> LocalOrigin:
        return LocalOrigin.from_geo(self.geo)


class InitialPositionProvider:
    """Acquires the initial position over MAVLink, or from configuration."""

    def __init__(
        self,
        connection: Optional[MavlinkConnection] = None,
        source: str = "mavlink_origin",
        wait_timeout_s: float = 0.0,
        request_origin_on_start: bool = True,
        manual_lat_deg: Optional[float] = None,
        manual_lon_deg: Optional[float] = None,
        manual_alt_m: Optional[float] = None,
        push_origin_to_autopilot: bool = False,
    ) -> None:
        self.connection = connection
        self.source = source
        self.wait_timeout_s = float(wait_timeout_s)
        self.request_origin_on_start = bool(request_origin_on_start)
        self.manual_lat_deg = manual_lat_deg
        self.manual_lon_deg = manual_lon_deg
        self.manual_alt_m = manual_alt_m
        self.push_origin_to_autopilot = bool(push_origin_to_autopilot)

        self._received = threading.Event()
        self._position: Optional[InitialPosition] = None
        self._lock = threading.Lock()

    @classmethod
    def from_config(
        cls, config, connection: Optional[MavlinkConnection] = None
    ) -> "InitialPositionProvider":
        return cls(
            connection=connection,
            source=config.source,
            wait_timeout_s=config.wait_timeout_s,
            request_origin_on_start=config.request_origin_on_start,
            manual_lat_deg=config.manual_lat_deg,
            manual_lon_deg=config.manual_lon_deg,
            manual_alt_m=config.manual_alt_m,
            push_origin_to_autopilot=config.push_origin_to_autopilot,
        )

    # -- state --------------------------------------------------------------
    @property
    def position(self) -> Optional[InitialPosition]:
        with self._lock:
            return self._position

    @property
    def available(self) -> bool:
        return self._received.is_set()

    # -- acquisition --------------------------------------------------------
    def start_listening(self) -> None:
        """Subscribe to ``GPS_GLOBAL_ORIGIN`` and request it once."""
        if self.connection is None:
            return
        self.connection.add_handler("GPS_GLOBAL_ORIGIN", self._on_origin)
        if self.request_origin_on_start:
            self.request_origin()
            # GPS_GLOBAL_ORIGIN is in no ArduPilot stream group: it is sent once
            # on change, with no retry, so one dropped packet on a lossy link
            # would otherwise mean we never learn the origin at all.  A 5 s
            # refresh costs nothing and removes that single point of failure.
            try:
                self.connection.request_message_interval(
                    MAVLINK_MSG_ID_GPS_GLOBAL_ORIGIN, ORIGIN_REFRESH_INTERVAL_US
                )
            except MavlinkConnectionError as exc:
                LOG.warning("could not set the origin refresh interval: %s", exc)

    def request_origin(self) -> None:
        """Ask the autopilot to emit its current origin now."""
        if self.connection is None:
            return
        try:
            self.connection.send_command_long(
                MAV_CMD_REQUEST_MESSAGE, param1=MAVLINK_MSG_ID_GPS_GLOBAL_ORIGIN
            )
            LOG.info("requested GPS_GLOBAL_ORIGIN from the autopilot")
        except MavlinkConnectionError as exc:
            LOG.warning("could not request the origin: %s", exc)

    def _on_origin(self, message) -> None:
        """Handle an inbound ``GPS_GLOBAL_ORIGIN``.

        Fields are ``latitude``/``longitude`` in degE7 and ``altitude`` in
        millimetres MSL, per the MAVLink definition.
        """
        geo = GeoPoint(
            lat_deg=message.latitude / 1e7,
            lon_deg=message.longitude / 1e7,
            alt_m=message.altitude / 1000.0,
        )
        if geo.lat_deg == 0.0 and geo.lon_deg == 0.0:
            LOG.warning("ignoring GPS_GLOBAL_ORIGIN at null island (0, 0)")
            return
        with self._lock:
            previous = self._position
            self._position = InitialPosition(
                geo=geo, source="mavlink_origin", received_monotonic=time.monotonic()
            )
        self._received.set()
        if previous is None:
            LOG.info(
                "received EKF origin from the autopilot: %.7f, %.7f, %.1f m",
                geo.lat_deg,
                geo.lon_deg,
                geo.alt_m,
            )
        else:
            LOG.warning(
                "the autopilot changed its EKF origin to %.7f, %.7f, %.1f m; "
                "the local NED frame has moved",
                geo.lat_deg,
                geo.lon_deg,
                geo.alt_m,
            )

    def acquire(self, timeout_s: Optional[float] = None) -> InitialPosition:
        """Block until an initial position is available.

        Raises :class:`InitialPositionError` if none can be obtained.
        """
        timeout_s = self.wait_timeout_s if timeout_s is None else timeout_s

        if self.source == "manual":
            return self._manual_position()

        if self.connection is None:
            if self.source == "auto":
                return self._manual_position()
            raise InitialPositionError(
                "initial_position.source is %r but there is no MAVLink connection; "
                "use 'manual' and set manual_lat_deg/manual_lon_deg" % self.source
            )

        self.start_listening()
        LOG.info(
            "waiting for the operator to set the EKF origin in Mission Planner "
            "(right-click the map -> Set Home Here -> Set EKF Origin Here)"
        )

        deadline = None if timeout_s <= 0 else time.monotonic() + timeout_s
        last_request = time.monotonic()
        while True:
            if self._received.wait(timeout=1.0):
                position = self.position
                assert position is not None
                return position
            now = time.monotonic()
            # Re-request periodically: the autopilot may have booted after us.
            if now - last_request >= 5.0:
                self.request_origin()
                last_request = now
            if deadline is not None and now >= deadline:
                break

        if self.source == "auto" and self._has_manual():
            LOG.warning(
                "no EKF origin after %.0f s; falling back to the configured "
                "manual initial position",
                timeout_s,
            )
            return self._manual_position()

        raise InitialPositionError(
            "no GPS_GLOBAL_ORIGIN received within %.0f s. Set the EKF origin in "
            "Mission Planner, or configure initial_position.source: manual" % timeout_s
        )

    def _has_manual(self) -> bool:
        return self.manual_lat_deg is not None and self.manual_lon_deg is not None

    def _manual_position(self) -> InitialPosition:
        if not self._has_manual():
            raise InitialPositionError(
                "manual initial position requested but initial_position.manual_lat_deg "
                "/ manual_lon_deg are not set"
            )
        geo = GeoPoint(
            lat_deg=float(self.manual_lat_deg),
            lon_deg=float(self.manual_lon_deg),
            alt_m=float(self.manual_alt_m or 0.0),
        )
        LOG.info(
            "using the manually configured initial position: %.7f, %.7f, %.1f m",
            geo.lat_deg,
            geo.lon_deg,
            geo.alt_m,
        )
        position = InitialPosition(
            geo=geo, source="manual", received_monotonic=time.monotonic()
        )
        with self._lock:
            self._position = position
        self._received.set()

        if self.push_origin_to_autopilot and self.connection is not None:
            self.set_autopilot_origin(geo)
        return position

    def set_autopilot_origin(self, geo: GeoPoint) -> None:
        """Push an origin to the autopilot with ``MAV_CMD_DO_SET_GLOBAL_ORIGIN``.

        The reverse of the normal flow, for the case where the Pi is configured
        with the start position and the autopilot has no origin yet.  Parameters
        5, 6 and 7 of the command are latitude, longitude and altitude in metres.

        Only use this when you are sure: overwriting a live EKF origin mid-flight
        shifts every local coordinate the autopilot is using.
        """
        if self.connection is None:
            raise InitialPositionError("no MAVLink connection to set the origin on")
        LOG.warning(
            "setting the autopilot EKF origin to %.7f, %.7f, %.1f m",
            geo.lat_deg,
            geo.lon_deg,
            geo.alt_m,
        )
        self.connection.send_command_long(
            MAV_CMD_DO_SET_GLOBAL_ORIGIN,
            param5=geo.lat_deg,
            param6=geo.lon_deg,
            param7=geo.alt_m,
        )
