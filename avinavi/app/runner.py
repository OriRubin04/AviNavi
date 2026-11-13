"""The main loop: three modes, two rates, one authoritative output.

Threads
-------
``localization``  captures frames and runs the visual localization cycle at
                  ``rates.visual_localization_hz``
``output``        publishes external navigation at ``rates.mavlink_odometry_hz``,
                  propagating the last fix forward between localization cycles
``diagnostics``   prints and logs the health view at ``rates.diagnostics_hz``

Decoupling the two working rates is what lets a 3 Hz localizer feed a 20 Hz EKF
input without either of them waiting on the other, and it is why the propagator
exists.  MAVLink receive runs on its own thread inside
:class:`~avinavi.mavlink.connection.MavlinkConnection`.

Modes
-----
``offline``  no flight controller.  Recorded frames plus the map; the whole
             pipeline runs and logs, nothing is transmitted.
``bench``    hardware-in-the-loop.  The aircraft is stationary; MAVLink,
             coordinate frames and EKF acceptance are verified for real.
``flight``   the real thing.  GPS may be absent; the loop runs on the operator's
             initial position plus visual localization alone.

The mode changes which components are constructed and how loudly the safety
gate complains - it does not change the localization maths, so what is
validated offline is what flies.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import replace
from typing import Optional, Union

from ..camera import CameraError, create_camera
from ..config import Config
from ..diagnostics.logging_setup import SolutionLogger
from ..diagnostics.monitor import DiagnosticsMonitor
from ..geo import LocalOrigin, haversine_distance_m
from ..frames import CameraExtrinsics
from ..localization.engine import LocalizationEngine
from ..localization.propagator import PosePropagator
from ..localization.visual_odometry import DeadReckoner, FrameToFrameOdometry
from ..mapdb.database import ExtractorMismatchError, MapCoverageError, MapDatabase
from ..mavlink.connection import MavlinkConnection, MavlinkConnectionError
from ..mavlink.external_navigation import ExternalNavigationPublisher
from ..mavlink.gps_input import GpsInputPublisher
from ..mavlink.initial_position import InitialPositionError, InitialPositionProvider
from ..mavlink.reanchor import ReanchorListener, ReanchorRequest
from ..types import GeoPoint, InvalidReason, NEDPosition, VisualFix
from .safety import SafetyGate

LOG = logging.getLogger(__name__)


#: The two output paths present the same interface to the runner: ``publish``,
#: ``publish_failure``, ``enabled``, ``message`` and the three counters.  Which
#: one is built depends on ``output.message``; see :meth:`Runner._build_publisher`.
Publisher = Union[ExternalNavigationPublisher, GpsInputPublisher]


class RunnerError(RuntimeError):
    """The system could not start, or could not continue."""


class Runner:
    """Wires everything together and runs it."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.mode = config.mode

        self.monitor = DiagnosticsMonitor()
        self.solution_logger = SolutionLogger.from_config(config.logging)
        self.propagator = PosePropagator.from_config(config.propagation)
        self.safety = SafetyGate.from_config(config.safety)

        self.database: Optional[MapDatabase] = None
        self.engine: Optional[LocalizationEngine] = None
        self.camera = None
        self.connection: Optional[MavlinkConnection] = None
        self.publisher: Optional[Publisher] = None
        self.initial_position: Optional[InitialPositionProvider] = None
        self.reanchor_listener: Optional[ReanchorListener] = None

        extrinsics = CameraExtrinsics.from_config(config.camera_extrinsics)
        self.odometry = FrameToFrameOdometry.from_config(
            config.camera, extrinsics, config.visual_odometry
        )
        self.reckoner = DeadReckoner.from_config(config.visual_odometry)
        #: The last fix that came from actually matching the map.  Dead
        #: reckoning is always expressed as an offset from this.
        self._last_map_fix: Optional[VisualFix] = None
        #: The odometry thread accumulates into the reckoner while the
        #: localization thread starts and reads it, so it needs a lock.
        self._reckoner_lock = threading.Lock()
        self._reckoning_just_expired = False
        #: True when odometry runs on its own thread.  False for a replay
        #: source, where the localization loop must do the tracking instead -
        #: see :meth:`_odometry_loop`.
        self._odometry_threaded = False

        self._stop = threading.Event()
        self._threads = []
        self._frames_exhausted = threading.Event()
        self._last_frame_sequence = 0
        self._camera_failed = False

    # -- construction -------------------------------------------------------
    @property
    def uses_mavlink(self) -> bool:
        """Offline mode never opens a link; the other two always do."""
        return self.mode in ("bench", "flight")

    def setup(self) -> None:
        """Build every component and acquire the initial position."""
        LOG.info("AviNavi starting in %s mode", self.mode)

        try:
            self.database = MapDatabase.load(
                self.config.map.database_path,
                feature_cache_size=self.config.map.feature_cache_size,
            )
        except MapCoverageError as exc:
            raise RunnerError(
                "map database failure: %s. Build one with 'avinavi build-map'." % exc
            ) from exc

        self.engine = LocalizationEngine(self.config, self.database)
        self._verify_extractor()

        if self.uses_mavlink:
            self._setup_mavlink()

        origin, start = self._acquire_initial_position()
        self.engine.initialize(origin, start)
        # GPS_INPUT carries absolute lat/lon, so that publisher needs the origin
        # to convert the local NED solution back out.  ODOMETRY does not: it is
        # already expressed in the local frame the autopilot holds.
        if isinstance(self.publisher, GpsInputPublisher):
            self.publisher.set_origin(origin)

        self.camera = create_camera(self.config.camera)
        try:
            self.camera.open()
        except CameraError as exc:
            raise RunnerError("camera failure: %s" % exc) from exc

        LOG.info("loading SuperPoint and LightGlue weights")
        try:
            self.engine.warmup()
        except RuntimeError as exc:
            raise RunnerError("could not load the vision models: %s" % exc) from exc

        self.solution_logger.open()
        self._log_startup_summary(origin)

    def _setup_mavlink(self) -> None:
        self.connection = MavlinkConnection.from_config(self.config.mavlink)
        try:
            self.connection.connect(wait_for_heartbeat=True, timeout_s=30.0)
        except MavlinkConnectionError as exc:
            raise RunnerError(str(exc)) from exc

        self.publisher = self._build_publisher()
        if self.publisher.enabled:
            LOG.warning(
                "external navigation TRANSMISSION IS ENABLED (%s). Confirm the "
                "bench checks in docs/safety-and-testing.md have passed.",
                self.publisher.message,
            )
        else:
            LOG.info(
                "external navigation transmission is disabled; computing and "
                "logging only (output.send_to_autopilot = false)"
            )

        if self.config.reanchor.enabled:
            self.reanchor_listener = ReanchorListener.from_config(
                self.config.reanchor, self.connection
            )
            self.reanchor_listener.start()

    def _build_publisher(self) -> Publisher:
        """Pick the output path named by ``output.message``.

        ``GPS_INPUT`` and the two external-navigation messages are handled by
        different ArduPilot subsystems with different compile-time gates, so
        this is not a cosmetic choice: on a 1 MB board such as the
        MatekF405-Wing, a stock ArduPlane build has *all three* compiled out and
        every one of them fails silently.  See
        ``docs/flight-controller-compatibility.md``.
        """
        if self.config.output.message == "gps_input":
            return GpsInputPublisher.from_config(self.connection, self.config.output)
        return ExternalNavigationPublisher.from_config(
            self.connection, self.config.output
        )

    def _verify_extractor(self) -> None:
        """Check the tiles were built by something this run can match against.

        Done here, before MAVLink, before the camera and before the operator is
        asked for a position, because it needs none of them and because the
        failure it catches is otherwise invisible: a database built with one
        detector and flown with another produces no matches at all, with a clean
        log and a healthy-looking video stream.

        The comparison uses the engine's **actual** extractor rather than a
        second reading of the configuration, so there is no mirror of
        ``_build_extractor`` here to fall out of step with it.
        """
        assert self.database is not None and self.engine is not None
        fingerprint = getattr(self.engine.extractor, "fingerprint", None)
        if fingerprint is None:
            LOG.warning(
                "%s exposes no fingerprint; skipping the map compatibility check",
                type(self.engine.extractor).__name__,
            )
            return
        try:
            self.database.verify_extractor(
                fingerprint, strict=self.config.map.verify_extractor
            )
        except ExtractorMismatchError as exc:
            raise RunnerError(str(exc)) from exc

    def _acquire_initial_position(self):
        """Get the origin, and the tile-search seed, before anything else runs."""
        self.initial_position = InitialPositionProvider.from_config(
            self.config.initial_position, self.connection
        )
        try:
            position = self.initial_position.acquire()
        except InitialPositionError as exc:
            raise RunnerError("initial position failure: %s" % exc) from exc

        origin = position.origin
        LOG.info(
            "initial position from %s: %.7f, %.7f, %.1f m",
            position.source,
            position.geo.lat_deg,
            position.geo.lon_deg,
            position.geo.alt_m,
        )
        self._check_map_covers(position.geo)
        return origin, position.geo

    def _check_map_covers(self, point: GeoPoint) -> None:
        """Fail fast when the operator's start position is off the map."""
        assert self.database is not None
        coverage = self.database.coverage
        if coverage is None:
            raise RunnerError("the map database is empty")
        if coverage.contains(point):
            return
        distance = coverage.distance_to_m(point)
        radius = self.config.tile_search.max_search_radius_m
        message = (
            "the initial position (%.6f, %.6f) is %.0f m outside the map coverage "
            "%s" % (point.lat_deg, point.lon_deg, distance, coverage.as_tuple())
        )
        if distance > radius:
            raise RunnerError(message + " - build tiles covering the flight area")
        LOG.warning("%s, but within the maximum search radius", message)

    def _log_startup_summary(self, origin: LocalOrigin) -> None:
        assert self.database is not None
        coverage = self.database.coverage
        LOG.info(
            "map: %d tiles, coverage %s",
            len(self.database),
            coverage.as_tuple() if coverage else "unknown",
        )
        LOG.info(
            "rates: camera %.1f Hz, localization %.1f Hz, external nav %.1f Hz",
            self.config.rates.camera_fps,
            self.config.rates.visual_localization_hz,
            self.config.rates.mavlink_odometry_hz,
        )
        LOG.info(
            "local NED origin: %.7f, %.7f at %.1f m; ODOMETRY positions are "
            "relative to this point",
            origin.lat_deg,
            origin.lon_deg,
            origin.alt_m,
        )

    # -- running ------------------------------------------------------------
    def run(self, max_cycles: Optional[int] = None, duration_s: Optional[float] = None) -> int:
        """Run until stopped, the frames run out, or a limit is reached.

        Returns a process exit code: 0 for a clean finish.
        """
        self.setup()
        self.monitor.set_status("running")

        self._threads = [
            threading.Thread(
                target=self._localization_loop,
                name="localization",
                args=(max_cycles,),
                daemon=True,
            )
        ]
        if self._should_thread_odometry():
            self._odometry_threaded = True
            self._threads.append(
                threading.Thread(
                    target=self._odometry_loop, name="odometry", daemon=True
                )
            )
        if self.uses_mavlink:
            self._threads.append(
                threading.Thread(target=self._output_loop, name="output", daemon=True)
            )
        if self.config.logging.console_diagnostics:
            self._threads.append(
                threading.Thread(
                    target=self._diagnostics_loop, name="diagnostics", daemon=True
                )
            )

        for thread in self._threads:
            thread.start()

        deadline = time.monotonic() + duration_s if duration_s else None
        try:
            while not self._stop.is_set():
                if self._frames_exhausted.is_set():
                    LOG.info("frame source exhausted; shutting down")
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    LOG.info("run duration reached; shutting down")
                    break
                if self._camera_failed:
                    LOG.error("camera failure; shutting down")
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            LOG.info("interrupted")
        finally:
            self.stop()

        return 1 if self._camera_failed else 0

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self.monitor.set_status("stopping")
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=3.0)
        self.shutdown()

    def shutdown(self) -> None:
        """Release everything, in the reverse of the order it was acquired."""
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:  # pragma: no cover - best-effort teardown
                LOG.debug("error closing the camera", exc_info=True)
            self.camera = None
        if self.reanchor_listener is not None:
            self.reanchor_listener.stop()
            self.reanchor_listener = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.database is not None:
            self.database.release_all()
        self.solution_logger.close()
        LOG.info(
            "shutdown complete: %d localization cycles, %d valid, %d messages sent",
            self.monitor.localizations.total,
            self.monitor.valid_fixes.total,
            self.publisher.sent_count if self.publisher else 0,
        )

    # -- the localization thread -------------------------------------------
    def _localization_loop(self, max_cycles: Optional[int]) -> None:
        interval = 1.0 / self.config.rates.visual_localization_hz
        cycles = 0
        next_cycle = time.monotonic()

        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_cycle:
                self._stop.wait(min(next_cycle - now, 0.05))
                continue
            # Schedule from the previous target, not from now, so a slow cycle
            # does not permanently shift the phase - but never build up a backlog.
            next_cycle = max(next_cycle + interval, now + interval * 0.25)

            self._apply_pending_reanchor()

            try:
                frame = self.camera.read()
            except CameraError as exc:
                LOG.error("camera failure: %s", exc)
                self._on_camera_failure()
                return
            if frame is None:
                self._frames_exhausted.set()
                return

            self.monitor.record_frame()
            self._last_frame_sequence = frame.sequence

            fix = self._localize(frame)
            # Only a real map match counts toward the lock rate; a
            # dead-reckoned position would otherwise flatter the statistics
            # with its own predecessor's success.
            self.monitor.record_fix(fix)
            fix = self._apply_dead_reckoning(fix, frame)

            accepted, reason = self.safety.observe(fix)
            if accepted:
                if self.propagator.last_fix is None:
                    # First fix after a dropout: tell the EKF the track restarted.
                    self.propagator.note_reset()
                self.propagator.update(fix)
            elif not fix.valid:
                self._handle_invalid_fix(fix)
            else:
                LOG.warning("fix rejected by the safety gate: %s", reason)
                self.propagator.clear()

            self._write_row(fix)

            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                LOG.info("reached the %d cycle limit", max_cycles)
                self._frames_exhausted.set()
                return

    def _should_thread_odometry(self) -> bool:
        """Does the odometry get its own thread?

        Not for a ``directory`` replay source. Two threads pulling from one
        replay would consume images twice and desynchronise the run, and it is
        unnecessary there anyway: consecutive images in a directory *are*
        consecutive frames, so tracking inside the localization loop is both
        safe and correct.
        """
        if not self.config.visual_odometry.enabled or not self.odometry.available:
            return False
        if (self.config.camera.source or "").lower() == "directory":
            return False
        return self.config.rates.visual_odometry_hz > 0

    def _odometry_loop(self) -> None:
        """Track the ground between map fixes, on its own schedule.

        This runs far faster than the localization loop, and that is the whole
        point — see :meth:`track_frame`. It reads the camera concurrently with
        the localization loop, which is safe for a live source because the
        capture thread always hands over the newest frame; both consumers
        occasionally seeing the same frame is harmless, and a repeated frame
        yields a zero delta rather than a wrong one.
        """
        interval = 1.0 / self.config.rates.visual_odometry_hz
        next_cycle = time.monotonic()
        failures = 0

        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_cycle:
                self._stop.wait(min(next_cycle - now, 0.02))
                continue
            next_cycle = max(next_cycle + interval, now + interval * 0.25)

            if self.camera is None:
                return
            try:
                frame = self.camera.read()
            except CameraError:
                # The localization loop reports camera faults and shuts the run
                # down; this thread just stops contributing.
                return
            if frame is None:
                continue

            if not self.track_frame(frame):
                failures += 1
                if failures == 40:
                    LOG.warning(
                        "frame-to-frame odometry has produced no estimate for "
                        "%d consecutive frames. Dead reckoning cannot cover a "
                        "map dropout in this state - check the height source, "
                        "and that rates.visual_odometry_hz is high enough that "
                        "consecutive frames still overlap.",
                        failures,
                    )
            else:
                failures = 0

    def track_frame(self, frame) -> bool:
        """Feed one frame to the frame-to-frame odometry.

        Called from :meth:`_odometry_loop` at ``rates.visual_odometry_hz``,
        which is **deliberately not** the localization rate. Lucas-Kanade needs
        consecutive frames to overlap heavily, and it was measured to succeed at
        a 0.5 s spacing (11 m at 22 m/s, 92% frame overlap) and to fail outright
        at 1.0 s (22 m, 84% overlap). Since map matching runs at well under 1 Hz
        with the learned backend, tracking between *localization* frames would
        produce nothing at all — the feature would be present and inert exactly
        when it is needed.

        Returns True when a delta was accumulated.
        """
        if not self.config.visual_odometry.enabled or not self.odometry.available:
            return False

        roll, pitch, yaw = self._attitude()
        try:
            delta = self.odometry.update(
                frame.image,
                timestamp_monotonic=frame.timestamp_monotonic,
                height_m=self._height_hint(),
                roll_rad=roll,
                pitch_rad=pitch,
                yaw_rad=yaw or 0.0,
            )
        except Exception:
            LOG.exception("visual odometry raised; continuing without it")
            return False
        if delta is None:
            return False

        with self._reckoner_lock:
            if not self.reckoner.active:
                # Nothing to reckon from yet, or it has already expired.
                return False
            if not self.reckoner.accumulate(delta):
                self._reckoning_just_expired = True
                return False
        return True

    def _apply_dead_reckoning(self, fix: VisualFix, frame=None) -> VisualFix:
        """Carry the solution across a map-matching dropout.

        Consumes whatever :meth:`track_frame` has accumulated since the last map
        fix; it does not run the odometry itself.

        A dead-reckoned fix is returned as ``valid`` because it is a genuine
        estimate, but it carries ``dead_reckoned=True`` and a sigma grown by the
        distance flown, so everything downstream can tell the difference. It is
        bounded twice: by ``max_dead_reckoning_s``, after which this produces
        nothing at all, and by the safety gate, which still applies.
        """
        if not self.config.visual_odometry.enabled or not self.odometry.available:
            return fix

        # In replay mode there is no separate odometry thread - see
        # _odometry_loop - so the localization loop does the tracking, which is
        # correct there because consecutive images in a directory *are*
        # consecutive frames.
        if frame is not None and not self._odometry_threaded:
            self.track_frame(frame)

        if fix.valid and fix.position_ned is not None:
            # A map fix resets the drift: dead reckoning is always relative to
            # the most recent real match.
            self._last_map_fix = fix
            sigma = fix.position_sigma_m
            with self._reckoner_lock:
                self.reckoner.start(0.0 if math.isnan(sigma) else sigma)
                self._reckoning_just_expired = False
            return fix

        if self._reckoning_just_expired:
            # Say why once, so the log distinguishes "could not match" from
            # "stopped guessing"; afterwards the honest reason is whatever the
            # map matching reported.
            self._reckoning_just_expired = False
            self._notify(
                "AviNavi: dead reckoning expired after %.0f m; no position"
                % self.reckoner.distance_m,
                warning=True,
            )
            return replace(fix, reason=InvalidReason.DEAD_RECKONING_EXPIRED)

        with self._reckoner_lock:
            if self._last_map_fix is None or not self.reckoner.active:
                return fix
            if self.reckoner.distance_m <= 0.0:
                # Reckoning is live but the odometry has not produced anything
                # yet: no information, which is not the same as no movement.
                return fix
        return self._dead_reckoned_fix(fix)

    def _dead_reckoned_fix(self, failed: VisualFix) -> VisualFix:
        """Build a position from the last map fix plus the accumulated offset."""
        base = self._last_map_fix
        origin = self.engine.origin
        if base is None or base.position_ned is None or origin is None:
            return failed

        north_offset, east_offset = self.reckoner.offset_ned_m
        position = NEDPosition(
            base.position_ned.north_m + north_offset,
            base.position_ned.east_m + east_offset,
            base.position_ned.down_m,
        )
        sigma = self.reckoner.sigma_m
        return VisualFix(
            timestamp_monotonic=failed.timestamp_monotonic,
            timestamp_unix=failed.timestamp_unix,
            valid=True,
            reason=InvalidReason.NONE,
            geo=origin.to_geo(position),
            position_ned=position,
            velocity_ned=base.velocity_ned,
            attitude=base.attitude,
            # Yaw is not observed by dead reckoning; carrying the old one
            # forward would misrepresent it as fresh.
            yaw_rad=None,
            quality=base.quality,
            tile_id=base.tile_id,
            candidate_tiles=0,
            ground_resolution_mpp=base.ground_resolution_mpp,
            processing_time_s=failed.processing_time_s,
            position_sigma_m=sigma,
            yaw_sigma_rad=base.yaw_sigma_rad,
            dead_reckoned=True,
            dead_reckoning_s=self.reckoner.elapsed_s,
        )

    def _attitude(self):
        """``(roll, pitch, yaw)`` from the autopilot, or level flight."""
        if self.connection is None:
            return (0.0, 0.0, None)
        state = self.connection.state
        if state.attitude_age() < 0.5 and state.roll_rad is not None:
            return (state.roll_rad, state.pitch_rad, state.yaw_rad)
        return (0.0, 0.0, None)

    def _height_hint(self) -> Optional[float]:
        """Height above the mapped terrain, from the autopilot where fresh."""
        if self.connection is None:
            return None
        state = self.connection.state
        if state.height_age() >= 1.0 or state.height_above_origin_m is None:
            return None
        origin = self.engine.origin
        if origin is None:
            return None
        return origin.alt_m + state.height_above_origin_m - self.config.map.terrain_alt_m

    def _apply_pending_reanchor(self) -> None:
        """Move the search window if the operator clicked the map.

        This only changes *where the next frame looks*.  It is deliberately not
        a position fix: the frame still has to earn its match, and until it
        does, nothing is published.  A click is therefore always safe, including
        mid-flight with transmission enabled.
        """
        if self.reanchor_listener is None:
            return
        request: Optional[ReanchorRequest] = self.reanchor_listener.take()
        if request is None:
            return

        previous = self.engine.search_center
        moved = (
            haversine_distance_m(previous, request.geo) if previous is not None else None
        )
        self.engine.reanchor(request.geo, request.radius_m)
        # Any propagated pose still in flight was derived from the old area, so
        # it must not survive the move.
        self.propagator.clear()
        self.safety.reset()

        self._notify(
            "re-anchored to %.6f, %.6f (r=%.0f m, via %s%s)"
            % (
                request.geo.lat_deg,
                request.geo.lon_deg,
                self.engine.search_radius_m,
                request.source,
                "" if moved is None else ", %.0f m from the last search centre" % moved,
            )
        )

    def _localize(self, frame) -> VisualFix:
        """Run one cycle, feeding in autopilot attitude/height when available.

        Only attitude that is actually fresh is used; stale angles are worse
        than assuming level flight. Yaw is passed separately from roll and
        pitch because it steers the north-up rectification and seeds the
        yaw-bias estimator, and unlike the other two there is no safe default,
        so ``None`` means "do not assume".
        """
        roll, pitch, yaw = self._attitude()
        height_hint = self._height_hint()

        try:
            return self.engine.localize(
                frame.image,
                roll_rad=roll,
                pitch_rad=pitch,
                yaw_rad=yaw,
                height_hint_m=height_hint,
                timestamp_unix=frame.timestamp_unix,
            )
        except Exception:
            LOG.exception("localization cycle raised; treating it as a failed fix")
            return VisualFix(
                timestamp_monotonic=frame.timestamp_monotonic,
                timestamp_unix=frame.timestamp_unix,
                valid=False,
                reason=InvalidReason.GEOMETRIC_VERIFICATION_FAILED,
            )

    def _handle_invalid_fix(self, fix: VisualFix) -> None:
        """React to a failed localization according to *why* it failed."""
        if fix.reason == InvalidReason.NO_CANDIDATE_TILES:
            if self.engine.search_exhausted:
                LOG.error(
                    "map coverage failure: no tiles within %.0f m of the last "
                    "visual position",
                    self.engine.search_radius_m,
                )
                self._notify("AviNavi: no map coverage", warning=True)
        elif fix.reason == InvalidReason.LOW_CONFIDENCE:
            LOG.info(
                "visual solution invalid: confidence %.2f below the %.2f threshold",
                fix.confidence,
                self.config.confidence.min_confidence,
            )

    def _on_camera_failure(self) -> None:
        """Camera fault: invalidate navigation immediately and stop the run."""
        self._camera_failed = True
        self.propagator.clear()
        self.safety.reset()
        self.monitor.set_status("failed", "camera failure")
        self._notify("AviNavi: camera failure", warning=True)
        if self.publisher is not None:
            self.publisher.publish_failure()

    # -- the output thread --------------------------------------------------
    def _output_loop(self) -> None:
        interval = 1.0 / self.config.rates.mavlink_odometry_hz
        next_send = time.monotonic()
        link_was_up = True

        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_send:
                self._stop.wait(min(next_send - now, 0.02))
                continue
            next_send = max(next_send + interval, now + interval * 0.25)

            link_ok = self.connection is not None and self.connection.connected
            if link_was_up and not link_ok:
                # MAVLink loss: stop transmitting and say so, rather than
                # queueing messages nobody is receiving.
                LOG.error("MAVLink connection lost; external navigation stopped")
                self.monitor.set_status("degraded", "MAVLink disconnected")
            elif link_ok and not link_was_up:
                LOG.info("MAVLink connection restored")
                self.monitor.set_status("running")
            link_was_up = link_ok
            if not link_ok:
                continue

            solution = self.propagator.solution_at(now)
            if solution is None:
                # Nothing valid to send. Deliberate silence - see the propagator.
                continue

            permitted, reason = self.safety.may_transmit(link_ok)
            if not permitted:
                LOG.debug("not transmitting: %s", reason)
                continue

            if self.publisher is not None and self.publisher.publish(solution):
                self.monitor.record_odometry(solution)

    # -- the diagnostics thread ---------------------------------------------
    def _diagnostics_loop(self) -> None:
        interval = 1.0 / max(self.config.rates.diagnostics_hz, 0.05)
        while not self._stop.wait(interval):
            try:
                text = self.monitor.render(
                    connection=self.connection,
                    publisher=self.publisher,
                    engine=self.engine,
                )
            except Exception:  # pragma: no cover - never let diagnostics kill a run
                LOG.exception("diagnostics rendering failed")
                continue
            print("\n" + "=" * 60)
            print(text)
            print("=" * 60, flush=True)

    # -- helpers ------------------------------------------------------------
    def _write_row(self, fix: VisualFix) -> None:
        solution = self.propagator.solution_at(time.monotonic()) if fix.valid else None
        ground_truth = None
        error_m = None
        if self.connection is not None and self.config.mavlink.request_gps_for_ground_truth:
            state = self.connection.state
            if state.ground_truth_lat_deg is not None:
                ground_truth = GeoPoint(
                    state.ground_truth_lat_deg,
                    state.ground_truth_lon_deg,
                    state.ground_truth_alt_m or 0.0,
                )
                if fix.geo is not None:
                    error_m = haversine_distance_m(ground_truth, fix.geo)

        self.solution_logger.write(
            fix=fix,
            solution=solution,
            transmitted=bool(
                solution is not None and self.publisher is not None and self.publisher.enabled
            ),
            message=self.config.output.message,
            frame_sequence=self._last_frame_sequence,
            ground_truth=ground_truth,
            ground_truth_error_m=error_m,
        )

    def _notify(self, text: str, warning: bool = False) -> None:
        """Surface an event in the Mission Planner HUD when a link exists."""
        if self.connection is None:
            return
        try:
            severity = None
            if warning:
                severity = self.connection.mavutil.mavlink.MAV_SEVERITY_WARNING
            self.connection.send_statustext(text, severity)
        except Exception:  # pragma: no cover - notification is best-effort
            LOG.debug("could not send STATUSTEXT", exc_info=True)

    def snapshot_json(self) -> str:
        return json.dumps(
            self.monitor.snapshot(self.connection, self.publisher, self.engine), indent=2
        )
