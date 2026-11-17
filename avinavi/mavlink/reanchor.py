"""Manual re-anchoring: the operator clicks the map, the Pi searches there.

When the visual solution is lost — flown over water, over terrain the basemap
does not cover, through heavy cloud shadow, or simply drifted further than the
search window can grow — the system needs a way back in.  Growing the radius
handles small drift; it cannot handle "the aircraft is nowhere near where the
search thinks it is".

So the operator gets a manual override: click a point on the map, and the Pi
re-centres its tile search there.  This changes **only where the Pi looks**.  It
does not move the EKF origin, it does not touch the navigation solution, and it
is not a position fix — the next camera frame still has to earn a match.

Which channel actually works depends on the wiring, and the answer is not
obvious, so it was traced through ArduPilot's router and Mission Planner's
source rather than guessed.  Two topologies:

**(A) Pi on one flight-controller UART, Mission Planner on a telemetry radio on
another.**  The Pi only sees what the flight controller forwards.
**(B) Pi is the telemetry hub** — Mission Planner connects to the Pi, which
forwards to the flight controller.  The Pi sees all GCS traffic.

The decisive fact for (A) is ArduPilot's forwarding rule
(``MAVLink_routing::forward``): a message addressed to ``target_system = 1``,
``target_component = 1`` hits an early return and is **never forwarded** to
other links.  Every Mission Planner map action is addressed to
``target_component = 1``.  So in topology (A), sniffing "Point Camera Here" or
"Fly To Here" cannot work — the packets never leave the autopilot.

The channels, in the order this module prefers them
---------------------------------------------------

``COMMAND_INT`` / ``MAV_CMD_SPATIAL_USER_1`` (31005) — the right answer
    Addressed to *this* component, so there is no ambiguity about who the point
    was for, and we can answer with ``COMMAND_ACK`` so the operator sees the
    click land.  ``SPATIAL_USER_1`` is the slot the MAVLink specification
    reserves for a user-defined point on the ground (its own description gives
    "ROI item" as the example), and it is absent from ArduPilot's
    ``command_long_stores_location()``, so even if the autopilot receives a copy
    it answers ``MAV_RESULT_UNSUPPORTED`` and does nothing.  Zero side effects
    by construction.

    Stock Mission Planner has no UI for it, so ``tools/mission_planner_plugin/``
    in this repository contains a small plugin that adds **"AviNavi: Search
    Here"** to the right-click map menu.

    In topology (A) this relies on the autopilot having *learned* a route to us,
    which is why :class:`~avinavi.mavlink.connection.MavlinkConnection` sends a
    continuous heartbeat.  Without it the router's rule 1e applies and the
    autopilot processes the command itself — harmless for ``SPATIAL_USER_1``,
    which is the other reason not to repurpose ``DO_SET_HOME`` for this.

``SET_GPS_GLOBAL_ORIGIN`` (48) — the zero-code option for topology (A)
    Mission Planner: **right-click → Set Home Here → Set EKF Origin Here**.

    This is the *only* stock map action that reaches a companion on a different
    UART, and for a precise structural reason: the message has a
    ``target_system`` field but **no ``target_component`` field**, so the
    router treats the component as a broadcast, skips the early return, and
    forwards it to every link holding a route for that system.  **The Pi must
    therefore transmit with the same system ID as the autopilot** (the
    ``mavlink.system_id`` default of 1).

    Verified safe to use mid-flight: ArduPilot *rejects* an origin change once
    the origin is set, in two places — ``GCS_MAVLINK::set_ekf_origin`` returns
    ``MAV_RESULT_FAILED`` if ``ahrs.get_origin()`` succeeds, and
    ``NavEKF3::setOriginLLH`` refuses with ``"EKF3: origin already set"`` to
    avoid "causing upsets from a shifting origin".  So the autopilot's EKF frame
    cannot be moved this way; the message is effectively a harmless no-op that
    we happen to be able to overhear.  The operator does get that warning in the
    HUD on every click, which is cosmetic but worth knowing about.

``DO_SET_ROI`` (201) — topology (B) only
    Mission Planner: **right-click → Point Camera Here**.  Note it sends the
    *legacy* command 201, not ``DO_SET_ROI_LOCATION`` (195); both are accepted
    here.  On a gimbal-less plane it is a navigation no-op
    (``AP_Mount::set_roi_target`` returns early with no backend).  Because it is
    addressed to the autopilot, we are deliberately eavesdropping and must
    **not** send a ``COMMAND_ACK`` — that would collide with the autopilot's.

``HOME_POSITION`` (242) — reliable, but it moves the RTL point
    "Set Home Here" works, and the resulting ``HOME_POSITION`` broadcast reaches
    every link directly from ``AP_AHRS::set_home()`` without going through the
    router at all.  But it relocates the return point on a fixed-wing aircraft
    and re-references every relative altitude, so it is **off by default**.

A UDP control socket
    For a script or a custom button.  No MAVLink involved, so no ambiguity::

        echo '{"lat": 32.0853, "lon": 34.7818, "radius_m": 2000}' \\
            | nc -u -w1 <pi-address> 14555

    Off by default and localhost-only unless configured otherwise.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ..types import GeoPoint
from .connection import MavlinkConnection

LOG = logging.getLogger(__name__)

#: The MAVLink-reserved "user defined spatial item" command.  Its own
#: specification text gives "ROI item" as the example use, and ArduPilot does
#: not treat it as a location, so the autopilot cannot act on it.
MAV_CMD_SPATIAL_USER_1 = 31005
#: Legacy ROI command - what Mission Planner's "Point Camera Here" sends.
MAV_CMD_DO_SET_ROI = 201
#: Current ROI command, accepted in case another GCS sends it.
MAV_CMD_DO_SET_ROI_LOCATION = 195

MAV_RESULT_ACCEPTED = 0
MAV_RESULT_DENIED = 2

#: Below this magnitude, a field that should hold degE7 is really float degrees
#: packed into an integer by a buggy sender.  Reject rather than act on a point
#: a few millimetres from null island.
DEGE7_SANITY_FLOOR = 1000


@dataclass
class ReanchorRequest:
    """An operator instruction to resume searching at a given point."""

    geo: GeoPoint
    source: str
    received_monotonic: float
    #: Search radius for the first attempt after re-anchoring.  Wider than
    #: normal: a map click is a rougher estimate than a visual fix.
    radius_m: Optional[float] = None


class ReanchorListener:
    """Collects manual re-anchor requests from every enabled channel.

    The latest request wins, and :meth:`take` hands it over exactly once, so the
    engine applies it on the next cycle rather than repeatedly.
    """

    def __init__(
        self,
        connection: Optional[MavlinkConnection] = None,
        accept_spatial_user: bool = True,
        accept_set_origin: bool = True,
        accept_roi: bool = True,
        accept_home_position: bool = False,
        default_radius_m: float = 1500.0,
        min_move_m: float = 25.0,
        control_socket_host: Optional[str] = None,
        control_socket_port: int = 14555,
        on_request: Optional[Callable[[ReanchorRequest], None]] = None,
    ) -> None:
        self.connection = connection
        self.accept_spatial_user = bool(accept_spatial_user)
        self.accept_set_origin = bool(accept_set_origin)
        self.accept_roi = bool(accept_roi)
        self.accept_home_position = bool(accept_home_position)
        self.default_radius_m = float(default_radius_m)
        self.min_move_m = float(min_move_m)
        self.control_socket_host = control_socket_host
        self.control_socket_port = int(control_socket_port)
        self.on_request = on_request

        self._pending: Optional[ReanchorRequest] = None
        self._last_home: Optional[GeoPoint] = None
        self._last_origin_sniff: Optional[GeoPoint] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._socket_thread: Optional[threading.Thread] = None
        self._socket: Optional[socket.socket] = None
        self._total = 0

    @classmethod
    def from_config(
        cls, config, connection: Optional[MavlinkConnection] = None
    ) -> "ReanchorListener":
        return cls(
            connection=connection,
            accept_spatial_user=config.accept_spatial_user,
            accept_set_origin=config.accept_set_origin,
            accept_roi=config.accept_roi,
            accept_home_position=config.accept_home_position,
            default_radius_m=config.radius_m,
            min_move_m=config.min_move_m,
            control_socket_host=config.control_socket_host or None,
            control_socket_port=config.control_socket_port,
        )

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        """Subscribe to the enabled MAVLink channels and open the socket."""
        if self.connection is not None:
            if self.accept_spatial_user or self.accept_roi:
                self.connection.add_handler("COMMAND_INT", self._on_command_int)
                self.connection.add_handler("COMMAND_LONG", self._on_command_long)
            if self.accept_set_origin:
                self.connection.add_handler(
                    "SET_GPS_GLOBAL_ORIGIN", self._on_set_gps_global_origin
                )
            if self.accept_home_position:
                self.connection.add_handler("HOME_POSITION", self._on_home_position)
            self._log_available_channels()
        if self.control_socket_host:
            self._start_control_socket()

    def _log_available_channels(self) -> None:
        options = []
        if self.accept_spatial_user:
            options.append(
                "the 'AviNavi: Search Here' map menu item (install the Mission "
                "Planner plugin from tools/mission_planner_plugin/)"
            )
        if self.accept_set_origin:
            options.append(
                "right-click the map -> Set Home Here -> Set EKF Origin Here "
                "(the autopilot will warn 'origin already set'; that is expected "
                "and harmless)"
            )
        if self.accept_roi:
            options.append(
                "right-click the map -> Point Camera Here (only reaches the Pi "
                "when the Pi is the telemetry hub)"
            )
        if self.accept_home_position:
            options.append(
                "right-click the map -> Set Home Here (WARNING: this also moves "
                "the RTL point)"
            )
        if options:
            LOG.info("manual re-anchor armed. To re-anchor, use any of:")
            for index, option in enumerate(options, start=1):
                LOG.info("  %d. %s", index, option)

    def stop(self) -> None:
        self._stop.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:  # pragma: no cover - best-effort teardown
                pass
            self._socket = None
        if self._socket_thread is not None and self._socket_thread.is_alive():
            self._socket_thread.join(timeout=2.0)

    # -- requests -----------------------------------------------------------
    def take(self) -> Optional[ReanchorRequest]:
        """Return and clear the pending request, if any."""
        with self._lock:
            request = self._pending
            self._pending = None
        return request

    @property
    def pending(self) -> Optional[ReanchorRequest]:
        with self._lock:
            return self._pending

    @property
    def total_requests(self) -> int:
        return self._total

    def submit(
        self, geo: GeoPoint, source: str, radius_m: Optional[float] = None
    ) -> ReanchorRequest:
        """Record a re-anchor request.  Also the entry point for tests."""
        request = ReanchorRequest(
            geo=geo,
            source=source,
            received_monotonic=time.monotonic(),
            radius_m=radius_m if radius_m is not None else self.default_radius_m,
        )
        with self._lock:
            self._pending = request
            self._total += 1
        LOG.warning(
            "manual re-anchor from %s: tile search will resume at %.7f, %.7f "
            "with a %.0f m radius",
            source,
            geo.lat_deg,
            geo.lon_deg,
            request.radius_m,
        )
        if self.on_request is not None:
            try:
                self.on_request(request)
            except Exception:  # pragma: no cover - a callback must not break us
                LOG.exception("re-anchor callback failed")
        return request

    # -- COMMAND_INT / COMMAND_LONG ----------------------------------------
    def _addressed_to_us(self, message) -> bool:
        """Is this command for this component, or a broadcast?"""
        if self.connection is None:
            return True
        target_system = getattr(message, "target_system", 0)
        target_component = getattr(message, "target_component", 0)
        system_ok = target_system in (0, self.connection.system_id)
        component_ok = target_component in (0, self.connection.component_id)
        return system_ok and component_ok

    def _on_command_int(self, message) -> None:
        """``COMMAND_INT``: ``x``/``y`` are **degE7 integers**, ``z`` is metres."""
        command = getattr(message, "command", 0)

        if command == MAV_CMD_SPATIAL_USER_1:
            if not self.accept_spatial_user:
                return
            if not self._addressed_to_us(message):
                return
            geo = self._decode_dege7(message.x, message.y, float(message.z), "SPATIAL_USER_1")
            if geo is None:
                self._acknowledge(message, MAV_RESULT_DENIED)
                return
            # param1 carries an optional search radius, by our own convention.
            radius = float(message.param1) if message.param1 and message.param1 > 0 else None
            self.submit(geo, source="spatial_user_1", radius_m=radius)
            # The ACK is what turns a click into visible confirmation for the
            # operator; none of the sniffing channels can offer this.
            self._acknowledge(message, MAV_RESULT_ACCEPTED)
            return

        if command in (MAV_CMD_DO_SET_ROI, MAV_CMD_DO_SET_ROI_LOCATION):
            if not self.accept_roi:
                return
            # Deliberately eavesdropping on traffic addressed to the autopilot,
            # so no ACK: it would collide with the autopilot's own reply.
            geo = self._decode_dege7(message.x, message.y, float(message.z), "DO_SET_ROI")
            if geo is not None:
                self.submit(geo, source="roi_command_int")

    def _on_command_long(self, message) -> None:
        """``COMMAND_LONG``: ``param5``/``param6`` are **float degrees**.

        The trap in the pair: the same command carries degE7 integers in
        ``COMMAND_INT`` and plain float degrees in ``COMMAND_LONG``.  Reading one
        as the other is a factor of ten million — the point does not land
        slightly wrong, it vanishes.
        """
        command = getattr(message, "command", 0)
        if command == MAV_CMD_SPATIAL_USER_1:
            if not self.accept_spatial_user or not self._addressed_to_us(message):
                return
            source = "spatial_user_1_long"
        elif command in (MAV_CMD_DO_SET_ROI, MAV_CMD_DO_SET_ROI_LOCATION):
            if not self.accept_roi:
                return
            source = "roi_command_long"
        else:
            return

        latitude = float(message.param5)
        longitude = float(message.param6)
        if latitude == 0.0 and longitude == 0.0:
            return
        if abs(latitude) > 90.0 or abs(longitude) > 180.0:
            LOG.warning(
                "ignoring %s with out-of-range coordinates (%.3f, %.3f); a sender "
                "packing degE7 into COMMAND_LONG floats would look exactly like this",
                source,
                latitude,
                longitude,
            )
            return
        self.submit(
            GeoPoint(latitude, longitude, float(message.param7)), source=source
        )

    def _acknowledge(self, message, result: int) -> None:
        """Reply with ``COMMAND_ACK`` so the operator's GCS confirms the click."""
        if self.connection is None:
            return
        try:
            self.connection.send_command_ack(
                command=int(message.command),
                result=result,
                target_system=message.get_srcSystem(),
                target_component=message.get_srcComponent(),
            )
        except Exception:  # pragma: no cover - acknowledgement is best-effort
            LOG.debug("could not send COMMAND_ACK", exc_info=True)

    # -- SET_GPS_GLOBAL_ORIGIN ---------------------------------------------
    def _on_set_gps_global_origin(self, message) -> None:
        """"Set EKF Origin Here" — the zero-code path for a Pi on its own UART.

        ``latitude``/``longitude`` are degE7 and ``altitude`` is **millimetres**
        (not centimetres, which is what ArduPilot uses internally and an easy
        thing to carry over by mistake).

        We only *overhear* this; the autopilot rejects it outright once its
        origin is set, so nothing on the aircraft moves.
        """
        geo = self._decode_dege7(
            message.latitude,
            message.longitude,
            message.altitude / 1000.0,
            "SET_GPS_GLOBAL_ORIGIN",
        )
        if geo is None:
            return
        # Mission Planner sends this once per click with no retry. Debounce
        # identical consecutive values, but treat a genuine repeat as a
        # deliberate second re-anchor rather than swallowing it.
        if self._last_origin_sniff is not None:
            from ..geo import haversine_distance_m  # local import avoids a cycle

            if haversine_distance_m(self._last_origin_sniff, geo) < 1.0:
                LOG.debug("ignoring duplicate Set EKF Origin Here at the same point")
                return
        self._last_origin_sniff = geo
        self.submit(geo, source="set_gps_global_origin")

    # -- HOME_POSITION ------------------------------------------------------
    def _on_home_position(self, message) -> None:
        """``HOME_POSITION``: degE7 and **millimetres**.

        Reaches every link directly from ``AP_AHRS::set_home()``, bypassing the
        router entirely, so it is the most reliable channel — and the most
        intrusive, because it means the operator has moved the RTL point.
        """
        geo = self._decode_dege7(
            message.latitude, message.longitude, message.altitude / 1000.0, "HOME_POSITION"
        )
        if geo is None:
            return

        from ..geo import haversine_distance_m  # local import avoids a cycle

        if self._last_home is None:
            # The first HOME_POSITION is just where home already was. Record it
            # as a baseline rather than reading it as operator intent.
            self._last_home = geo
            LOG.info(
                "baseline home recorded at %.7f, %.7f; moving it will re-anchor "
                "the tile search",
                geo.lat_deg,
                geo.lon_deg,
            )
            return

        moved = haversine_distance_m(self._last_home, geo)
        if moved < self.min_move_m:
            # ArduPilot re-broadcasts home on arming and on origin changes;
            # that is not an operator asking for anything.
            LOG.debug("ignoring HOME_POSITION that moved only %.1f m", moved)
            return

        self._last_home = geo
        self.submit(geo, source="home_position")

    # -- decoding -----------------------------------------------------------
    @staticmethod
    def _decode_dege7(
        latitude_e7, longitude_e7, altitude_m: float, what: str
    ) -> Optional[GeoPoint]:
        """Decode a degE7 pair, rejecting the things that are not coordinates."""
        if latitude_e7 == 0 and longitude_e7 == 0:
            # ArduPilot's own "use current location" sentinel. Not a point in
            # the Gulf of Guinea.
            LOG.debug("ignoring %s with the (0, 0) 'use current' sentinel", what)
            return None
        if abs(latitude_e7) < DEGE7_SANITY_FLOOR and abs(longitude_e7) < DEGE7_SANITY_FLOOR:
            LOG.warning(
                "ignoring %s whose coordinates (%d, %d) are too small to be degE7; "
                "the sender has probably packed float degrees into an integer field",
                what,
                latitude_e7,
                longitude_e7,
            )
            return None
        latitude = latitude_e7 / 1e7
        longitude = longitude_e7 / 1e7
        if abs(latitude) > 90.0 or abs(longitude) > 180.0:
            LOG.warning(
                "ignoring %s with out-of-range coordinates (%.6f, %.6f)",
                what,
                latitude,
                longitude,
            )
            return None
        return GeoPoint(latitude, longitude, altitude_m)

    # -- control socket -----------------------------------------------------
    def _start_control_socket(self) -> None:
        host = self.control_socket_host or "127.0.0.1"
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, self.control_socket_port))
            sock.settimeout(0.5)
        except OSError as exc:
            LOG.error("could not open the re-anchor control socket: %s", exc)
            return
        self._socket = sock
        self._socket_thread = threading.Thread(
            target=self._control_loop, name="reanchor-socket", daemon=True
        )
        self._socket_thread.start()
        if host not in ("127.0.0.1", "localhost"):
            LOG.warning(
                "re-anchor control socket is listening on %s:%d, reachable from the "
                "network; anything that can reach it can redirect the navigation "
                "search",
                host,
                self.control_socket_port,
            )
        else:
            LOG.info("re-anchor control socket on %s:%d", host, self.control_socket_port)

    def _control_loop(self) -> None:
        while not self._stop.is_set():
            sock = self._socket
            if sock is None:
                return
            try:
                payload, address = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                data = json.loads(payload.decode("utf-8"))
                geo = GeoPoint(
                    lat_deg=float(data["lat"]),
                    lon_deg=float(data["lon"]),
                    alt_m=float(data.get("alt", 0.0)),
                )
                radius = data.get("radius_m")
            except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
                LOG.warning(
                    "ignoring malformed re-anchor datagram from %s: %s", address, exc
                )
                continue
            if abs(geo.lat_deg) > 90.0 or abs(geo.lon_deg) > 180.0:
                LOG.warning("ignoring out-of-range re-anchor from %s", address)
                continue
            self.submit(
                geo,
                source="control_socket",
                radius_m=float(radius) if radius is not None else None,
            )
