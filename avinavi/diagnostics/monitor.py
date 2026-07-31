"""Live rate and health monitoring.

Produces the diagnostic view the specification asks for::

    MAVLink connection: OK
    Heartbeat: OK
    ODOMETRY transmission:
        Rate: 19.8 Hz
    Visual localization:
        Rate: 4.7 Hz
    ...

Rates are measured over a sliding window rather than assumed from the
configuration, because the whole point of watching them is to catch the case
where the Pi cannot keep up with what it was asked to do.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Optional

from ..types import InvalidReason, NavigationSolution, VisualFix


class RateTracker:
    """Sliding-window rate estimator for one event stream."""

    def __init__(self, window_s: float = 5.0, name: str = "") -> None:
        self.window_s = float(window_s)
        self.name = name
        self._events: Deque[float] = deque()
        self._total = 0
        self._lock = threading.Lock()

    def tick(self, timestamp: Optional[float] = None) -> None:
        now = timestamp if timestamp is not None else time.monotonic()
        with self._lock:
            self._events.append(now)
            self._total += 1
            self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    @property
    def rate_hz(self) -> float:
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            if len(self._events) < 2:
                return 0.0
            span = self._events[-1] - self._events[0]
            if span <= 0:
                return 0.0
            return (len(self._events) - 1) / span

    @property
    def total(self) -> int:
        return self._total

    @property
    def age_s(self) -> float:
        """Seconds since the most recent event; infinite if there has been none."""
        with self._lock:
            if not self._events:
                return float("inf")
            return time.monotonic() - self._events[-1]


class DiagnosticsMonitor:
    """Aggregates everything the operator and the log need to see."""

    def __init__(self, window_s: float = 5.0) -> None:
        self.frames = RateTracker(window_s, "camera")
        self.localizations = RateTracker(window_s, "visual_localization")
        self.valid_fixes = RateTracker(window_s, "valid_fix")
        self.odometry = RateTracker(window_s, "odometry_tx")

        self._lock = threading.Lock()
        self._last_fix: Optional[VisualFix] = None
        self._last_solution: Optional[NavigationSolution] = None
        self._failure_counts: Dict[str, int] = {}
        self._latencies: Deque[float] = deque(maxlen=200)
        self._started = time.monotonic()
        self._status: str = "starting"
        self._status_detail: str = ""

    # -- inputs -------------------------------------------------------------
    def record_frame(self) -> None:
        self.frames.tick()

    def record_fix(self, fix: VisualFix) -> None:
        self.localizations.tick()
        with self._lock:
            self._last_fix = fix
            self._latencies.append(fix.processing_time_s)
            if not fix.valid:
                key = fix.reason.value if isinstance(fix.reason, InvalidReason) else str(fix.reason)
                self._failure_counts[key] = self._failure_counts.get(key, 0) + 1
        if fix.valid:
            self.valid_fixes.tick()

    def record_odometry(self, solution: NavigationSolution) -> None:
        self.odometry.tick()
        with self._lock:
            self._last_solution = solution

    def set_status(self, status: str, detail: str = "") -> None:
        with self._lock:
            self._status = status
            self._status_detail = detail

    # -- outputs ------------------------------------------------------------
    @property
    def last_fix(self) -> Optional[VisualFix]:
        with self._lock:
            return self._last_fix

    @property
    def mean_latency_s(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)

    @property
    def max_latency_s(self) -> float:
        with self._lock:
            return max(self._latencies) if self._latencies else 0.0

    def snapshot(self, connection=None, publisher=None, engine=None) -> Dict:
        """A JSON-serialisable view of the current state."""
        with self._lock:
            fix = self._last_fix
            solution = self._last_solution
            failures = dict(self._failure_counts)
            status = self._status
            status_detail = self._status_detail

        data: Dict = {
            "uptime_s": round(time.monotonic() - self._started, 1),
            "status": status,
            "status_detail": status_detail,
            "rates_hz": {
                "camera": round(self.frames.rate_hz, 2),
                "visual_localization": round(self.localizations.rate_hz, 2),
                "valid_fixes": round(self.valid_fixes.rate_hz, 2),
                "odometry_tx": round(self.odometry.rate_hz, 2),
            },
            "latency_s": {
                "mean": round(self.mean_latency_s, 4),
                "max": round(self.max_latency_s, 4),
            },
            "failures": failures,
        }

        if fix is not None:
            data["visual_localization"] = {
                "valid": fix.valid,
                "reason": fix.reason.value if isinstance(fix.reason, InvalidReason) else str(fix.reason),
                "tile_id": fix.tile_id,
                "candidate_tiles": fix.candidate_tiles,
                "age_s": round(time.monotonic() - fix.timestamp_monotonic, 3),
                **fix.quality.as_dict(),
            }
            if fix.geo is not None:
                data["visual_localization"]["geo"] = {
                    "lat": round(fix.geo.lat_deg, 7),
                    "lon": round(fix.geo.lon_deg, 7),
                    "alt_m": round(fix.geo.alt_m, 2),
                }
            if fix.position_ned is not None:
                data["visual_localization"]["ned"] = {
                    "north_m": round(fix.position_ned.north_m, 2),
                    "east_m": round(fix.position_ned.east_m, 2),
                    "down_m": round(fix.position_ned.down_m, 2),
                }

        if solution is not None:
            data["last_solution"] = {
                "north_m": round(solution.position_ned.north_m, 2),
                "east_m": round(solution.position_ned.east_m, 2),
                "down_m": round(solution.position_ned.down_m, 2),
                "confidence": round(solution.confidence, 3),
                "propagated": solution.propagated,
                "age_s": round(solution.age_s, 3),
                "position_sigma_m": round(solution.position_sigma_m, 2),
                "reset_counter": solution.reset_counter,
            }

        if connection is not None:
            data["mavlink"] = {
                "connected": connection.connected,
                "heartbeat_age_s": (
                    round(connection.heartbeat_age_s, 2)
                    if connection.heartbeat_age_s != float("inf")
                    else None
                ),
                "sent": connection.sent_counts,
                "received": connection.received_counts,
            }

        if publisher is not None:
            data["external_navigation"] = {
                "message": publisher.message,
                "transmitting": publisher.enabled,
                "sent": publisher.sent_count,
                "suppressed": publisher.suppressed_count,
                "last_error": publisher.last_error,
            }

        if engine is not None:
            centre = engine.search_center
            data["tile_search"] = {
                "radius_m": round(engine.search_radius_m, 1),
                "consecutive_failures": engine.consecutive_failures,
                "center": (
                    {"lat": round(centre.lat_deg, 7), "lon": round(centre.lon_deg, 7)}
                    if centre
                    else None
                ),
            }

        return data

    def render(self, connection=None, publisher=None, engine=None) -> str:
        """The human-readable console view."""
        data = self.snapshot(connection, publisher, engine)
        lines = []

        mavlink = data.get("mavlink")
        if mavlink is not None:
            lines.append(
                "MAVLink connection: %s" % ("OK" if mavlink["connected"] else "DOWN")
            )
            age = mavlink["heartbeat_age_s"]
            lines.append(
                "Heartbeat: %s"
                % ("OK (%.1f s ago)" % age if age is not None else "NOT RECEIVED")
            )
        else:
            lines.append("MAVLink connection: not in use (offline mode)")

        external = data.get("external_navigation")
        if external is not None:
            lines.append("")
            lines.append("%s transmission:" % external["message"].upper())
            lines.append("    Enabled: %s" % ("yes" if external["transmitting"] else "NO (log only)"))
            lines.append("    Rate: %.1f Hz" % data["rates_hz"]["odometry_tx"])
            lines.append("    Sent: %d   Suppressed: %d" % (external["sent"], external["suppressed"]))
            if external["last_error"]:
                lines.append("    Last error: %s" % external["last_error"])

        lines.append("")
        lines.append("Visual localization:")
        lines.append("    Rate: %.1f Hz (valid %.1f Hz)" % (
            data["rates_hz"]["visual_localization"],
            data["rates_hz"]["valid_fixes"],
        ))
        lines.append("    Camera: %.1f Hz" % data["rates_hz"]["camera"])
        lines.append(
            "    Latency: mean %.0f ms, max %.0f ms"
            % (data["latency_s"]["mean"] * 1000.0, data["latency_s"]["max"] * 1000.0)
        )

        visual = data.get("visual_localization")
        if visual is not None:
            lines.append("    Valid: %s%s" % (
                visual["valid"],
                "" if visual["valid"] else " (%s)" % visual["reason"],
            ))
            if "geo" in visual:
                lines.append(
                    "    Position: %.7f, %.7f, %.1f m"
                    % (visual["geo"]["lat"], visual["geo"]["lon"], visual["geo"]["alt_m"])
                )
            if "ned" in visual:
                lines.append(
                    "    Local NED: N %.1f  E %.1f  D %.1f m"
                    % (visual["ned"]["north_m"], visual["ned"]["east_m"], visual["ned"]["down_m"])
                )
            lines.append("")
            lines.append("Confidence:")
            lines.append("    %.2f" % visual["localization_confidence"])
            lines.append("Matches:")
            lines.append("    %d" % visual["number_of_matches"])
            lines.append("RANSAC inliers:")
            lines.append(
                "    %d (ratio %.3f)" % (visual["number_of_inliers"], visual["inlier_ratio"])
            )
            if visual["reprojection_error_px"] is not None:
                lines.append(
                    "Reprojection error: %.2f px" % visual["reprojection_error_px"]
                )
            lines.append(
                "Tile: %s (of %d candidates)"
                % (visual["tile_id"] or "-", visual["candidate_tiles"])
            )

        search = data.get("tile_search")
        if search is not None:
            lines.append(
                "Tile search: r=%.0f m, %d consecutive failures"
                % (search["radius_m"], search["consecutive_failures"])
            )

        if data["failures"]:
            lines.append("")
            lines.append("Failure tally:")
            for reason, count in sorted(
                data["failures"].items(), key=lambda item: -item[1]
            ):
                lines.append("    %-32s %d" % (reason, count))

        return "\n".join(lines)
