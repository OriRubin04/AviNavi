"""MAVLink transport and inbound message handling.

Direction of travel, per the project specification:

**ArduPilot -> Raspberry Pi**
    ``HEARTBEAT`` (required - it is how we know the link is alive),
    ``ATTITUDE`` (optional, improves the yaw solve and the tilt correction),
    ``VFR_HUD`` / ``LOCAL_POSITION_NED`` (optional, supply a height hint),
    ``GPS_GLOBAL_ORIGIN`` (the operator-selected origin, see
    :mod:`avinavi.mavlink.initial_position`),
    ``GLOBAL_POSITION_INT`` (**testing only**, ground-truth comparison).

**Raspberry Pi -> ArduPilot**
    ``HEARTBEAT`` and exactly one external-navigation message - see
    :mod:`avinavi.mavlink.external_navigation`.

Nothing here feeds GPS into the localization loop.  ``GLOBAL_POSITION_INT`` is
only decoded when ``mavlink.request_gps_for_ground_truth`` is explicitly
enabled, and even then it is written to the log and nowhere else.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

LOG = logging.getLogger(__name__)


class MavlinkConnectionError(RuntimeError):
    """The MAVLink link could not be established or has been lost."""


@dataclass
class VehicleState:
    """Latest autopilot-reported state we care about.

    Every field is optional and timestamped: the localization loop degrades
    gracefully when the autopilot is not sending a given stream, rather than
    blocking on it.
    """

    roll_rad: Optional[float] = None
    pitch_rad: Optional[float] = None
    yaw_rad: Optional[float] = None
    attitude_time: float = 0.0

    #: Height above the EKF origin in metres (positive up), from LOCAL_POSITION_NED.
    height_above_origin_m: Optional[float] = None
    height_time: float = 0.0

    #: Ground-truth only; never consumed by the localizer.
    ground_truth_lat_deg: Optional[float] = None
    ground_truth_lon_deg: Optional[float] = None
    ground_truth_alt_m: Optional[float] = None
    ground_truth_time: float = 0.0

    last_heartbeat: float = 0.0
    autopilot_boot_time_ms: Optional[int] = None
    armed: bool = False
    custom_mode: Optional[int] = None

    def attitude_age(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.monotonic()
        return now - self.attitude_time if self.attitude_time else float("inf")

    def height_age(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.monotonic()
        return now - self.height_time if self.height_time else float("inf")


class MavlinkConnection:
    """A pymavlink connection with a background receive thread.

    Reception runs on its own thread so a slow localization cycle never stalls
    heartbeat tracking, and so inbound origin messages are picked up the moment
    the operator sets them in Mission Planner.
    """

    def __init__(
        self,
        connection_type: str = "serial",
        device: str = "/dev/ttyAMA0",
        baudrate: int = 921600,
        system_id: int = 1,
        component_id: int = 197,
        target_system: int = 1,
        target_component: int = 1,
        heartbeat_timeout_s: float = 3.0,
        send_heartbeat: bool = True,
        heartbeat_hz: float = 1.0,
        request_attitude_hz: float = 10.0,
        request_gps_for_ground_truth: bool = False,
    ) -> None:
        self.connection_type = connection_type
        self.device = device
        self.baudrate = int(baudrate)
        self.system_id = int(system_id)
        self.component_id = int(component_id)
        self.target_system = int(target_system)
        self.target_component = int(target_component)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.send_heartbeat = bool(send_heartbeat)
        self.heartbeat_hz = float(heartbeat_hz)
        self.request_attitude_hz = float(request_attitude_hz)
        self.request_gps_for_ground_truth = bool(request_gps_for_ground_truth)

        self.state = VehicleState()
        self._master = None
        self._mavutil = None
        self._rx_thread: Optional[threading.Thread] = None
        self._hb_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._handlers: Dict[str, List[Callable]] = {}
        self._sent_counts: Dict[str, int] = {}
        self._received_counts: Dict[str, int] = {}
        self._start_monotonic = time.monotonic()

    @classmethod
    def from_config(cls, config) -> "MavlinkConnection":
        return cls(
            connection_type=config.connection_type,
            device=config.device,
            baudrate=config.baudrate,
            system_id=config.system_id,
            component_id=config.component_id,
            target_system=config.target_system,
            target_component=config.target_component,
            heartbeat_timeout_s=config.heartbeat_timeout_s,
            send_heartbeat=config.send_heartbeat,
            heartbeat_hz=config.heartbeat_hz,
            request_attitude_hz=config.request_attitude_hz,
            request_gps_for_ground_truth=config.request_gps_for_ground_truth,
        )

    # -- lifecycle ----------------------------------------------------------
    def connect(self, wait_for_heartbeat: bool = True, timeout_s: float = 30.0) -> None:
        """Open the link and (optionally) block until the autopilot is heard."""
        try:
            from pymavlink import mavutil  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise MavlinkConnectionError(
                "pymavlink is required for MAVLink operation: pip install pymavlink"
            ) from exc
        self._mavutil = mavutil

        descriptor = self._connection_string()
        LOG.info("opening MAVLink connection: %s", descriptor)
        try:
            if self.connection_type == "serial":
                master = mavutil.mavlink_connection(
                    descriptor,
                    baud=self.baudrate,
                    source_system=self.system_id,
                    source_component=self.component_id,
                )
            else:
                master = mavutil.mavlink_connection(
                    descriptor,
                    source_system=self.system_id,
                    source_component=self.component_id,
                )
        except Exception as exc:  # pymavlink raises a variety of OS/serial errors
            raise MavlinkConnectionError(
                "could not open %s: %s" % (descriptor, exc)
            ) from exc

        self._master = master
        self._stop.clear()

        self._rx_thread = threading.Thread(
            target=self._receive_loop, name="mavlink-rx", daemon=True
        )
        self._rx_thread.start()
        if self.send_heartbeat:
            self._hb_thread = threading.Thread(
                target=self._heartbeat_loop, name="mavlink-hb", daemon=True
            )
            self._hb_thread.start()

        if wait_for_heartbeat and not self.wait_for_heartbeat(timeout_s):
            raise MavlinkConnectionError(
                "no HEARTBEAT from system %d within %.0f s on %s"
                % (self.target_system, timeout_s, descriptor)
            )

        self.request_data_streams()

    def _connection_string(self) -> str:
        if self.connection_type == "serial":
            return self.device
        # udp/tcp devices are already full pymavlink descriptors such as
        # "udpout:127.0.0.1:14550"; accept a bare host:port too.
        if ":" in self.device and not self.device.startswith(("udp", "tcp")):
            return "%s:%s" % (self.connection_type, self.device)
        return self.device

    def close(self) -> None:
        self._stop.set()
        for thread in (self._rx_thread, self._hb_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
        with self._lock:
            if self._master is not None:
                try:
                    self._master.close()
                except Exception:  # pragma: no cover - best-effort teardown
                    LOG.debug("error closing MAVLink connection", exc_info=True)
                self._master = None
        LOG.info("MAVLink connection closed")

    def __enter__(self) -> "MavlinkConnection":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- link health --------------------------------------------------------
    @property
    def connected(self) -> bool:
        """True while the link is open *and* heartbeats are arriving."""
        if self._master is None:
            return False
        if not self.state.last_heartbeat:
            return False
        return (time.monotonic() - self.state.last_heartbeat) <= self.heartbeat_timeout_s

    def wait_for_heartbeat(self, timeout_s: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.state.last_heartbeat:
                return True
            time.sleep(0.05)
        return False

    @property
    def heartbeat_age_s(self) -> float:
        if not self.state.last_heartbeat:
            return float("inf")
        return time.monotonic() - self.state.last_heartbeat

    # -- streams ------------------------------------------------------------
    def request_data_streams(self) -> None:
        """Ask the autopilot for the inbound messages we can make use of.

        Uses ``MAV_CMD_SET_MESSAGE_INTERVAL``, the per-message mechanism that
        replaced the deprecated ``REQUEST_DATA_STREAM``.
        """
        if self._master is None or self._mavutil is None:
            return
        mavlink = self._mavutil.mavlink

        wanted = []
        if self.request_attitude_hz > 0:
            wanted.append((mavlink.MAVLINK_MSG_ID_ATTITUDE, self.request_attitude_hz))
            wanted.append(
                (mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, self.request_attitude_hz)
            )
        if self.request_gps_for_ground_truth:
            LOG.warning(
                "GLOBAL_POSITION_INT requested: this is ground-truth logging only "
                "and must be disabled for GPS-denied operation"
            )
            wanted.append((mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 5.0))

        for message_id, rate_hz in wanted:
            interval_us = 1e6 / rate_hz if rate_hz > 0 else -1.0
            self.send_command_long(
                mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, param1=message_id, param2=interval_us
            )

    def send_command_long(self, command: int, **params) -> None:
        """Send a ``COMMAND_LONG`` to the autopilot."""
        with self._lock:
            if self._master is None:
                raise MavlinkConnectionError("MAVLink connection is not open")
            self._master.mav.command_long_send(
                self.target_system,
                self.target_component,
                command,
                0,
                float(params.get("param1", 0.0)),
                float(params.get("param2", 0.0)),
                float(params.get("param3", 0.0)),
                float(params.get("param4", 0.0)),
                float(params.get("param5", 0.0)),
                float(params.get("param6", 0.0)),
                float(params.get("param7", 0.0)),
            )
            self._sent_counts["COMMAND_LONG"] = self._sent_counts.get("COMMAND_LONG", 0) + 1

    # -- handlers -----------------------------------------------------------
    def add_handler(self, message_type: str, handler: Callable) -> None:
        """Register a callback for a MAVLink message type (or ``"*"`` for all)."""
        self._handlers.setdefault(message_type, []).append(handler)

    def _dispatch(self, message) -> None:
        message_type = message.get_type()
        for handler in self._handlers.get(message_type, ()):
            try:
                handler(message)
            except Exception:  # a bad handler must not kill the rx thread
                LOG.exception("error in %s handler", message_type)
        for handler in self._handlers.get("*", ()):
            try:
                handler(message)
            except Exception:
                LOG.exception("error in wildcard handler")

    # -- threads ------------------------------------------------------------
    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            master = self._master
            if master is None:
                break
            try:
                message = master.recv_match(blocking=True, timeout=0.5)
            except Exception:  # pragma: no cover - transient serial errors
                LOG.exception("MAVLink receive error")
                time.sleep(0.1)
                continue
            if message is None:
                continue
            message_type = message.get_type()
            if message_type == "BAD_DATA":
                continue
            self._received_counts[message_type] = (
                self._received_counts.get(message_type, 0) + 1
            )
            self._update_state(message_type, message)
            self._dispatch(message)

    def _update_state(self, message_type: str, message) -> None:
        now = time.monotonic()
        if message_type == "HEARTBEAT":
            if message.get_srcSystem() != self.target_system:
                return
            self.state.last_heartbeat = now
            self.state.custom_mode = getattr(message, "custom_mode", None)
            base_mode = getattr(message, "base_mode", 0)
            if self._mavutil is not None:
                armed_flag = self._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                self.state.armed = bool(base_mode & armed_flag)
        elif message_type == "ATTITUDE":
            self.state.roll_rad = float(message.roll)
            self.state.pitch_rad = float(message.pitch)
            self.state.yaw_rad = float(message.yaw)
            self.state.attitude_time = now
            self.state.autopilot_boot_time_ms = int(message.time_boot_ms)
        elif message_type == "LOCAL_POSITION_NED":
            # z is down-positive relative to the EKF origin.
            self.state.height_above_origin_m = -float(message.z)
            self.state.height_time = now
        elif message_type == "GLOBAL_POSITION_INT":
            # Ground truth only. Never read by the localization loop.
            self.state.ground_truth_lat_deg = message.lat / 1e7
            self.state.ground_truth_lon_deg = message.lon / 1e7
            self.state.ground_truth_alt_m = message.alt / 1000.0
            self.state.ground_truth_time = now

    def _heartbeat_loop(self) -> None:
        interval = 1.0 / max(self.heartbeat_hz, 0.1)
        while not self._stop.wait(interval):
            try:
                self.send_our_heartbeat()
            except MavlinkConnectionError:
                break
            except Exception:  # pragma: no cover - transient serial errors
                LOG.exception("failed to send HEARTBEAT")

    def send_our_heartbeat(self) -> None:
        """Announce the companion computer on the link."""
        with self._lock:
            if self._master is None or self._mavutil is None:
                raise MavlinkConnectionError("MAVLink connection is not open")
            mavlink = self._mavutil.mavlink
            self._master.mav.heartbeat_send(
                mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                mavlink.MAV_STATE_ACTIVE,
            )
            self._sent_counts["HEARTBEAT"] = self._sent_counts.get("HEARTBEAT", 0) + 1

    def send_statustext(self, text: str, severity: Optional[int] = None) -> None:
        """Send a ``STATUSTEXT`` so messages appear in the Mission Planner HUD."""
        with self._lock:
            if self._master is None or self._mavutil is None:
                return
            mavlink = self._mavutil.mavlink
            if severity is None:
                severity = mavlink.MAV_SEVERITY_INFO
            # STATUSTEXT carries at most 50 bytes of text.
            payload = text.encode("utf-8")[:50]
            self._master.mav.statustext_send(severity, payload)

    # -- raw access ---------------------------------------------------------
    @property
    def master(self):
        """The underlying pymavlink connection (for the message senders)."""
        if self._master is None:
            raise MavlinkConnectionError("MAVLink connection is not open")
        return self._master

    @property
    def mavutil(self):
        if self._mavutil is None:
            raise MavlinkConnectionError("MAVLink connection is not open")
        return self._mavutil

    @property
    def lock(self) -> threading.RLock:
        """Held around every transmit so threads cannot interleave writes."""
        return self._lock

    def note_sent(self, message_type: str) -> None:
        self._sent_counts[message_type] = self._sent_counts.get(message_type, 0) + 1

    @property
    def sent_counts(self) -> Dict[str, int]:
        return dict(self._sent_counts)

    @property
    def received_counts(self) -> Dict[str, int]:
        return dict(self._received_counts)

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self._start_monotonic
