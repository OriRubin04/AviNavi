"""Frame-to-frame odometry: staying alive between map fixes.

Map matching does not succeed every cycle, and it does not have to.  What it
must not do is go silent: at 22 m/s a fixed wing covers 22 metres a second, and
on a Pi 5 the real SuperPoint + LightGlue stack takes seconds per cycle, so the
gap between two consecutive map fixes is routinely tens of metres of flight.
Something has to carry the solution across it.

Tracking the ground between consecutive *live* frames does that without a map at
all.  Two frames taken a tenth of a second apart overlap almost completely and
differ by a small translation, which is the easiest problem in this whole
system — no appearance gap, no seasonal change, no scale mismatch, no learned
matcher.  Sparse Lucas-Kanade flow handles it in a few milliseconds.

How the pixels become metres
----------------------------
The same projection the rectifier uses: a pixel defines a ray, the ray meets the
ground plane at a known height, and that gives the feature's position in metres
*relative to the point directly under the camera*.  For a static ground feature
seen in two frames::

    ground = camera₁ + p₁ = camera₂ + p₂

    so   camera₂ − camera₁ = p₁ − p₂

Each frame is projected with **its own** attitude and height, so a roll or pitch
change between the two does not leak into the translation.  That matters on a
fixed wing, where the aircraft is banking for much of the flight.

What this cannot do
-------------------
It drifts.  Every estimate is relative to the previous frame, so errors
accumulate without bound, and the dominant term is heading: a constant 1° yaw
error turns 100 m of flight into 1.7 m of cross-track error that never cancels.

So drift is **reported rather than hidden**.  The position sigma grows with
distance travelled at ``drift_fraction``, dead reckoning is abandoned after
``max_dead_reckoning_s``, and after that the system goes silent rather than
publishing a position it cannot justify.  A growing uncertainty the EKF can
weight is useful; a confident wrong position is the one failure mode this
project exists to avoid.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..frames import CameraExtrinsics, euler_to_matrix

LOG = logging.getLogger(__name__)


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("visual odometry needs OpenCV") from exc
    return cv2


@dataclass(frozen=True)
class OdometryDelta:
    """Displacement between two consecutive frames, in local NED metres."""

    north_m: float
    east_m: float
    dt_s: float
    #: Points that survived tracking and the robust rejection.
    tracked_points: int
    #: Median absolute deviation of the per-point estimates, metres.  The
    #: honest spread: a large value means the points disagree, which happens
    #: over water, under cloud shadow, or when something in the scene moved.
    residual_m: float
    #: Rotation between the two north-up projections, radians.  Both frames are
    #: levelled with the autopilot's own yaw, so this is a *diagnostic* for yaw
    #: error between the frames, not the vehicle's turn rate - and it is
    #: deliberately not integrated into anything.
    rotation_rad: float = 0.0

    @property
    def distance_m(self) -> float:
        return math.hypot(self.north_m, self.east_m)

    @property
    def speed_mps(self) -> float:
        return self.distance_m / self.dt_s if self.dt_s > 0 else 0.0


class FrameToFrameOdometry:
    """Sparse optical flow between consecutive live frames.

    Needs calibrated intrinsics and a height above terrain, exactly as the
    rectifier does: without them a pixel displacement cannot be turned into a
    distance.  :attr:`available` says whether it can run at all.
    """

    def __init__(
        self,
        extrinsics: CameraExtrinsics,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        max_points: int = 400,
        min_points: int = 60,
        resize_max_side: int = 480,
        #: Reject a point whose estimate is this many MADs from the median.
        outlier_mads: float = 3.0,
        #: Refuse an estimate implying a speed above this; a fixed wing does not
        #: teleport, and a flow field that says it did is a tracking failure.
        max_speed_mps: float = 60.0,
        #: How far tracked points may disagree with each other, in PIXELS of
        #: ground sample distance, before the estimate is refused.  **This gate
        #: is what stops the module producing a confident wrong answer.**
        #:
        #: It is expressed in pixels and converted to metres with the current
        #: height, so it is absolute: it does not scale with the displacement
        #: being tested.  The earlier form, max(1.0 m, 0.25 * distance), did,
        #: and that made it forgiving in exact proportion to how wrong the
        #: answer was - a 63 m displacement against a 2.2 m truth passed it.
        #:
        #: 4 px is roughly 0.4 m at 120 m AGL and 0.9 m at 240 m, against a
        #: measured residual of 0.07 m when the tracker is working.
        max_residual_pixels: float = 4.0,
        #: Floor, so that a very low pass does not make the gate impossible.
        min_residual_m: float = 0.25,
        #: Lucas-Kanade pyramid depth and search window.  See the config for
        #: the measurement behind the defaults.
        lk_pyramid_levels: int = 1,
        lk_window_px: int = 31,
        #: Ceiling on the adaptive depth; 3 bridges about 8 windows.
        max_pyramid_levels: int = 3,
    ) -> None:
        self.extrinsics = extrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.max_points = int(max_points)
        self.min_points = int(min_points)
        self.resize_max_side = int(resize_max_side)
        self.outlier_mads = float(outlier_mads)
        self.max_speed_mps = float(max_speed_mps)
        self.max_residual_pixels = float(max_residual_pixels)
        self.min_residual_m = float(min_residual_m)
        self.lk_pyramid_levels = max(0, int(lk_pyramid_levels))
        self.lk_window_px = max(5, int(lk_window_px))
        self.max_pyramid_levels = max(0, int(max_pyramid_levels))
        #: Ground speed from the last successful estimate, for predicting how
        #: far the next frame will have moved.
        self._last_speed_mps: Optional[float] = None
        self._last_track_rad: Optional[float] = None

        self._previous_gray = None
        self._previous_points = None
        self._previous_state = None   # (R_ned_from_camera, height_m, timestamp)
        self._scale = 1.0

    @classmethod
    def from_config(cls, camera_config, extrinsics: CameraExtrinsics, config=None):
        kwargs = {}
        if config is not None:
            kwargs = dict(
                max_points=config.max_points,
                min_points=config.min_points,
                resize_max_side=config.resize_max_side,
                max_residual_pixels=getattr(config, "max_residual_pixels", 4.0),
                min_residual_m=getattr(config, "min_residual_m", 0.25),
                lk_pyramid_levels=getattr(config, "lk_pyramid_levels", 1),
                lk_window_px=getattr(config, "lk_window_px", 31),
            )
        return cls(
            extrinsics=extrinsics,
            fx=camera_config.fx,
            fy=camera_config.fy,
            cx=camera_config.cx,
            cy=camera_config.cy,
            **kwargs,
        )

    @property
    def available(self) -> bool:
        return bool(self.fx and self.fy and self.fx > 0 and self.fy > 0)

    def reset(self) -> None:
        """Forget the previous frame.  Call after a map fix or a re-anchor."""
        self._previous_gray = None
        self._previous_points = None
        self._previous_state = None
        self._last_speed_mps = None
        self._last_track_rad = None

    # -- the cycle ----------------------------------------------------------
    def update(
        self,
        frame: np.ndarray,
        timestamp_monotonic: float,
        height_m: Optional[float],
        roll_rad: Optional[float] = 0.0,
        pitch_rad: Optional[float] = 0.0,
        yaw_rad: Optional[float] = 0.0,
    ) -> Optional[OdometryDelta]:
        """Track from the previous frame to this one.

        Returns ``None`` on the first frame, or whenever an honest estimate
        cannot be made - too few points survived, they disagree too much, or the
        implied speed is impossible.  ``None`` means "no information", which the
        caller must treat differently from a zero displacement.
        """
        cv2 = _require_cv2()
        if not self.available:
            self.reset()
            return None

        # Cache this frame before any of the checks below, and never reset on
        # them. A missing height or a stale attitude means this *one* estimate
        # cannot be made; throwing the frame away would also destroy the next
        # one, because there would be nothing to track from. One bad cycle
        # should cost one update, not two.
        gray, scale = self._prepare(cv2, frame)
        previous_gray = self._previous_gray
        previous_points = self._previous_points
        previous_state = self._previous_state
        self._previous_gray = gray
        self._scale = scale
        self._previous_points = self._detect(cv2, gray)

        unknown = (
            height_m is None
            or height_m <= 0.0
            or roll_rad is None
            or pitch_rad is None
            or yaw_rad is None
        )
        if unknown:
            # No information, which is not the same as no movement. Projecting
            # with assumed angles would produce a displacement that every
            # tracked point agrees with and that is still wrong, because both
            # frames share the same wrong assumption.
            self._previous_state = None
            return None

        R = self.extrinsics.R_ned_from_camera(
            euler_to_matrix(roll_rad, pitch_rad, yaw_rad)
        )
        state = (R, float(height_m), float(timestamp_monotonic))
        self._previous_state = state

        if previous_gray is None or previous_points is None or previous_state is None:
            return None
        if previous_points.shape[0] < self.min_points:
            return None

        dt = state[2] - previous_state[2]
        if dt <= 0.0:
            return None

        tracked, previous_kept = self._track(
            cv2, previous_gray, gray, previous_points,
            expected_px=self._expected_displacement_px(previous_state, state, scale),
            predicted=self._predict_pixels(
                previous_points, previous_state, state, scale
            ),
        )
        if tracked is None or tracked.shape[0] < self.min_points:
            LOG.debug(
                "visual odometry: only %d points survived tracking (need %d)",
                0 if tracked is None else tracked.shape[0],
                self.min_points,
            )
            return None

        return self._solve(
            previous_pixels=previous_kept / scale,
            current_pixels=tracked / scale,
            previous_state=previous_state,
            current_state=state,
            dt_s=dt,
        )

    # -- stages -------------------------------------------------------------
    def _prepare(self, cv2, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Grayscale, and downscale: flow does not need full resolution."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        longest = max(gray.shape[:2])
        if self.resize_max_side and longest > self.resize_max_side:
            scale = self.resize_max_side / float(longest)
            gray = cv2.resize(
                gray, (int(round(gray.shape[1] * scale)), int(round(gray.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            scale = 1.0
        return gray, scale

    def _detect(self, cv2, gray: np.ndarray) -> Optional[np.ndarray]:
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_points,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )
        if points is None:
            return None
        return points.reshape(-1, 2).astype(np.float32)

    def _expected_displacement_px(self, previous_state, state, scale: float):
        """How far the ground will have moved between these two frames, in the
        tracker's working pixels.

        From the last measured ground speed and the actual time between the
        frames, converted through the ground sample distance. ``None`` before
        the first successful estimate, when there is nothing to predict from.
        """
        if previous_state is None:
            return None
        # With no prior estimate - the first pair, or the first after a map fix
        # resets the tracker - assume the airframe's limit rather than nothing.
        # A pyramid that is too deep costs accuracy on a small displacement; a
        # pyramid that is too shallow loses the frame entirely on a large one,
        # and "we do not know yet" is exactly when a large one is possible.
        speed = self._last_speed_mps
        if speed is None:
            speed = self.max_speed_mps
        dt = state[2] - previous_state[2]
        if dt <= 0.0:
            return None
        fx = float(self.fx or 0.0)
        height = float(state[1])
        if fx <= 0.0 or height <= 0.0:
            return None
        gsd = height / fx
        return (speed * dt / gsd) * float(scale)

    def _predict_pixels(self, pixels, previous_state, state, scale: float):
        """Where each tracked point should land, from the attitude change alone.

        Lucas-Kanade models the motion of a window as pure translation. Over a
        banked turn that is false: the aircraft yaws, so the image rotates, and
        the ground plane's perspective shifts with the roll. At 30 degrees of
        bank the turn rate is about 15 deg/s, so 0.2 s between frames rotates
        the image by ~3 degrees - which moves a point 300 px from the centre by
        15 px *tangentially*, on top of the translation, in a direction the
        tracker's model cannot represent.

        Measured consequence before this existed, at 240 m on a racetrack:

            bank  0-9 deg   29% of cycles accepted
            bank 10-19 deg   0%
            bank 20-29 deg   0%

        Every banked frame failed, and they are the frames a turning aircraft
        spends much of its time in.

        But the attitude of both frames is known, so this motion is predictable
        rather than something the tracker has to discover. Each point is
        projected to the ground with the previous pose and re-projected with
        the current one; the translation between the two cameras is left for
        Lucas-Kanade to find, which is the part it is good at. Handing it the
        prediction as a starting point turns a 15 px rotational displacement
        into a sub-pixel residual.

        Returns ``None`` when the geometry is unusable, in which case the
        caller tracks without a prediction exactly as before.
        """
        from .prewarp import FramePrewarper  # noqa: PLC0415 - avoids a cycle

        if previous_state is None or state is None or scale <= 0.0:
            return None
        R_previous, height_previous, _ = previous_state
        R_current, height_current, _ = state
        if height_previous <= 0.0 or height_current <= 0.0:
            return None

        K = self._intrinsics()
        full = pixels.reshape(-1, 2) / scale
        ground = FramePrewarper._project_to_ground(full, K, R_previous, height_previous)
        if ground is None:
            return None

        # Ground points as rays from the current camera, including a predicted
        # translation from the last measured speed and heading.
        north_step = east_step = 0.0
        if self._last_speed_mps:
            dt = state[2] - previous_state[2]
            if dt > 0.0:
                step = self._last_speed_mps * dt
                north_step = step * math.cos(self._last_track_rad or 0.0)
                east_step = step * math.sin(self._last_track_rad or 0.0)
        rays_ned = np.column_stack(
            [
                ground[:, 0] - north_step,
                ground[:, 1] - east_step,
                np.full(ground.shape[0], height_current),
            ]
        )
        rays_camera = (R_current.T @ rays_ned.T).T
        depth = rays_camera[:, 2]
        if np.any(depth <= 1e-6) or not np.all(np.isfinite(depth)):
            return None
        projected = (K @ (rays_camera / depth[:, None]).T).T[:, :2]
        if not np.all(np.isfinite(projected)):
            return None
        return np.ascontiguousarray(
            (projected * scale).reshape(pixels.shape).astype(np.float32)
        )

    def _pyramid_levels_for(self, expected_px: Optional[float]) -> int:
        """How deep a pyramid this displacement needs - no deeper.

        A pyramid exists to bridge a displacement larger than the search
        window: each level halves the apparent motion. Levels beyond that are
        not free, they are harmful, because the coarse levels are then asked to
        measure a motion they cannot resolve and their answer initialises every
        level below.

        Measured at 240 m AGL, where the displacement is ~14 px against a 31 px
        window and no pyramid is needed at all:

            maxLevel 3 ->  4 points surviving   (below the 60-point minimum)
            maxLevel 1 -> 72 points surviving

        But a fixed shallow pyramid is wrong in the other direction. After a
        dropped frame the gap doubles, and at 120 m a 0.5 s gap is 102 px -
        which level 1 cannot bridge, and which measured as a total failure.

        So the depth follows the displacement, which is the quantity the
        pyramid is for. ``expected_px`` of ``None`` means the caller has no
        estimate yet, and the configured depth is used as-is.
        """
        configured = self.lk_pyramid_levels
        if not expected_px or expected_px <= 0.0:
            return configured
        needed = math.ceil(math.log2(max(expected_px, 1.0) / float(self.lk_window_px)))
        return int(min(self.max_pyramid_levels, max(configured, needed)))

    def _track(self, cv2, previous_gray, gray, previous_points, expected_px=None,
               predicted=None):
        """Forward-backward Lucas-Kanade.

        The backward check is what makes this usable over repetitive ground -
        ploughed fields, orchards, water. A point that tracks forward to a
        plausible place but does not track back to where it started was matched
        to a different, similar-looking feature, and it would otherwise
        contribute a confident wrong displacement.
        """
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        window = (self.lk_window_px, self.lk_window_px)
        levels = self._pyramid_levels_for(expected_px)
        if predicted is not None and predicted.shape == previous_points.shape:
            forward, status, _err = cv2.calcOpticalFlowPyrLK(
                previous_gray, gray, previous_points, predicted.copy(),
                winSize=window, maxLevel=levels, criteria=criteria,
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            )
        else:
            forward, status, _err = cv2.calcOpticalFlowPyrLK(
                previous_gray, gray, previous_points, None,
                winSize=window, maxLevel=levels, criteria=criteria,
            )
        if forward is None or status is None:
            return (None, None)
        ok = status.reshape(-1).astype(bool)
        if not ok.any():
            return (None, None)

        backward, back_status, _err = cv2.calcOpticalFlowPyrLK(
            gray, previous_gray, forward, None,
            winSize=window, maxLevel=levels, criteria=criteria,
        )
        if backward is not None and back_status is not None:
            round_trip = np.linalg.norm(backward - previous_points, axis=1)
            ok &= back_status.reshape(-1).astype(bool)
            ok &= round_trip < 1.0

        if ok.sum() == 0:
            return (None, None)
        return (forward[ok], previous_points[ok])

    def _solve(
        self,
        previous_pixels: np.ndarray,
        current_pixels: np.ndarray,
        previous_state,
        current_state,
        dt_s: float,
    ) -> Optional[OdometryDelta]:
        """Turn matched pixel pairs into a displacement in metres."""
        from .prewarp import FramePrewarper

        R_previous, height_previous, _ = previous_state
        R_current, height_current, _ = current_state
        K = self._intrinsics()

        ground_previous = FramePrewarper._project_to_ground(
            previous_pixels, K, R_previous, height_previous
        )
        ground_current = FramePrewarper._project_to_ground(
            current_pixels, K, R_current, height_current
        )
        if ground_previous is None or ground_current is None:
            # Some ray missed the ground plane: too steep to be usable.
            return None

        # camera2 - camera1 = p1 - p2, per the module docstring.
        per_point = ground_previous - ground_current

        median = np.median(per_point, axis=0)
        deviation = np.linalg.norm(per_point - median, axis=1)
        mad = float(np.median(deviation))
        if mad <= 1e-9:
            keep = np.ones(per_point.shape[0], dtype=bool)
        else:
            keep = deviation <= self.outlier_mads * mad
        if keep.sum() < self.min_points:
            LOG.debug(
                "visual odometry: %d of %d points agreed (need %d); the scene is "
                "probably featureless or something in it moved",
                int(keep.sum()),
                per_point.shape[0],
                self.min_points,
            )
            return None

        kept = per_point[keep]
        displacement = kept.mean(axis=0)
        residual = float(np.median(np.linalg.norm(kept - displacement, axis=1)))

        delta = OdometryDelta(
            north_m=float(displacement[0]),
            east_m=float(displacement[1]),
            dt_s=float(dt_s),
            tracked_points=int(keep.sum()),
            residual_m=residual,
            rotation_rad=self._rotation(ground_previous[keep], ground_current[keep]),
        )
        self._last_speed_mps = delta.speed_mps
        self._last_track_rad = math.atan2(delta.east_m, delta.north_m)

        if delta.speed_mps > self.max_speed_mps:
            LOG.warning(
                "visual odometry implied %.0f m/s over %.0f ms; rejecting it as a "
                "tracking failure rather than believing it",
                delta.speed_mps,
                dt_s * 1000.0,
            )
            return None

        # Do the surviving points actually agree with one another? Enough of
        # them can pass the count check while disagreeing wildly, and the mean
        # of a disagreeing set is not an estimate - it is a number. This is the
        # gate that turns "tracking failed" from a confident wrong displacement
        # into no displacement.
        #
        # The tolerance is ABSOLUTE, derived from the imaging geometry. It was
        # once max(1.0 m, 0.25 * distance), which scaled with the very quantity
        # under test: the more wrong the answer, the more forgiving the gate.
        # Measured consequence, with a stale attitude projecting a banked frame
        # as level at a 60 degree field of view:
        #
        #     mismatch  implied   truth   residual  old limit  verdict
        #        5 deg    11.8 m   2.2 m     0.9 m      2.9 m  ACCEPTED
        #       25 deg    63.1 m   2.2 m     9.0 m     15.8 m  ACCEPTED
        #
        # A 63 m displacement against a 2.2 m truth passed, because the limit
        # grew with it. An absolute limit cannot be inflated that way.
        #
        # The scale comes from the ground sample distance: a point mismatched
        # by a few pixels is wrong by a few pixels' worth of ground, and that
        # is height/focal_length per pixel. So the gate tightens as the
        # aircraft descends and loosens as it climbs, which is how the real
        # error behaves - unlike the displacement, which does not govern it at
        # all.
        tolerance = self._residual_tolerance_m(height_current)
        if residual > tolerance:
            LOG.warning(
                "visual odometry rejected: %d points claim %.1f m but disagree "
                "among themselves by %.2f m (limit %.2f m at %.0f m AGL). Either "
                "the tracker lost its footing, or the attitude used to project "
                "the two frames was inconsistent between them.",
                delta.tracked_points,
                delta.distance_m,
                residual,
                tolerance,
                height_current,
            )
            return None
        return delta

    def _residual_tolerance_m(self, height_m: float) -> float:
        """How far tracked points may disagree before the estimate is refused.

        Absolute, and a function of the imaging geometry rather than of the
        answer: ``residual_pixels`` worth of ground sample distance, with a
        floor so that a very low pass does not make the gate impossibly tight.
        """
        fx = float(self.fx or 0.0)
        if fx <= 0.0 or height_m <= 0.0:
            return self.min_residual_m
        gsd = height_m / fx
        return max(self.min_residual_m, self.max_residual_pixels * gsd)

    @staticmethod
    def _rotation(ground_previous: np.ndarray, ground_current: np.ndarray) -> float:
        """Residual rotation between the two north-up point sets.

        Diagnostic only.  Both sets are levelled with the autopilot's own yaw,
        so a non-zero value means the yaw used for the two frames disagreed -
        not that the aircraft turned.
        """
        a = ground_previous - ground_previous.mean(axis=0)
        b = ground_current - ground_current.mean(axis=0)
        if a.shape[0] < 3:
            return 0.0
        # Kabsch in 2D: the angle that best rotates a onto b.
        numerator = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
        denominator = float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
        if abs(numerator) < 1e-12 and abs(denominator) < 1e-12:
            return 0.0
        return float(math.atan2(numerator, denominator))

    def _intrinsics(self) -> np.ndarray:
        cx = self.cx if self.cx is not None else 0.0
        cy = self.cy if self.cy is not None else 0.0
        return np.array(
            [[self.fx, 0.0, cx], [0.0, self.fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )


class DeadReckoner:
    """Accumulates odometry deltas onto the last map fix, and gives up in time.

    The giving up is the important part.  Dead reckoning is useful for the few
    seconds it takes to get another map fix and actively harmful after that, so
    the growing uncertainty is reported honestly and there is a hard limit past
    which this produces nothing at all.
    """

    def __init__(
        self,
        drift_fraction: float = 0.05,
        max_dead_reckoning_s: float = 10.0,
        max_search_guidance_s: float = 60.0,
        max_position_sigma_m: float = 50.0,
        unobserved_speed_mps: float = 40.0,
        clock=None,
    ) -> None:
        self.drift_fraction = float(drift_fraction)
        self.max_dead_reckoning_s = float(max_dead_reckoning_s)
        self.max_search_guidance_s = float(max_search_guidance_s)
        self.max_position_sigma_m = float(max_position_sigma_m)
        self.unobserved_speed_mps = float(unobserved_speed_mps)
        #: Injectable so the staleness logic can be tested without sleeping.
        self._clock = clock or time.monotonic
        self._north_m = 0.0
        self._east_m = 0.0
        self._distance_m = 0.0
        self._elapsed_s = 0.0
        self._base_sigma_m = 0.0
        self._active = False
        self._expired = False
        #: Wall-clock times, not sums of odometry dt.  Summing dt was a bug:
        #: when optical flow failed it stopped being called, so the elapsed
        #: time froze, the limit never tripped, and a stale position was
        #: published as current with an unchanged sigma - forever.  See
        #: :meth:`unobserved_s`.
        self._started_at: Optional[float] = None
        self._last_delta_at: Optional[float] = None

    @classmethod
    def from_config(cls, config, clock=None) -> "DeadReckoner":
        return cls(
            drift_fraction=config.drift_fraction,
            max_dead_reckoning_s=config.max_dead_reckoning_s,
            max_search_guidance_s=getattr(config, "max_search_guidance_s", 60.0),
            max_position_sigma_m=getattr(config, "max_position_sigma_m", 50.0),
            unobserved_speed_mps=getattr(config, "unobserved_speed_mps", 40.0),
            clock=clock,
        )

    def start(self, base_sigma_m: float) -> None:
        """Begin reckoning from a map fix with the given position sigma."""
        self._north_m = 0.0
        self._east_m = 0.0
        self._distance_m = 0.0
        self._elapsed_s = 0.0
        self._base_sigma_m = max(0.0, float(base_sigma_m))
        self._active = True
        self._expired = False
        now = self._clock()
        self._started_at = now
        self._last_delta_at = now

    def clear(self) -> None:
        self._active = False
        self._expired = False
        self._started_at = None
        self._last_delta_at = None

    @property
    def unobserved_s(self) -> float:
        """Wall-clock seconds since optical flow last produced a displacement.

        This is the "has optical flow failed?" signal, and it has to come from
        a clock rather than from the odometry's own timestamps: when flow fails
        it produces nothing at all, so anything derived from its output stops
        advancing and every limit built on it silently stops applying.
        """
        if not self._active or self._last_delta_at is None:
            return 0.0
        return max(0.0, self._clock() - self._last_delta_at)

    @property
    def wall_elapsed_s(self) -> float:
        """Wall-clock seconds since the map fix this reckoning started from."""
        if not self._active or self._started_at is None:
            return 0.0
        return max(0.0, self._clock() - self._started_at)

    @property
    def active(self) -> bool:
        """Has a map fix to reckon from, whether or not it may be published."""
        return self._active

    @property
    def expired(self) -> bool:
        """Past the publishing budget."""
        return self._active and self._expired

    @property
    def publishable(self) -> bool:
        """May the accumulated position be sent to the autopilot?

        Governed by **uncertainty in metres**, not by a stopwatch. Optical-flow
        navigation is a legitimate mode, not a few seconds' grace: as long as
        the drift is reported honestly, ArduPilot's EKF weights the measurement
        by the accuracy it is given, so a 20 m-sigma position is useful and a
        300 m one is correctly ignored. The thing that was never safe was
        *misreporting* the drift.

        So the limit is a tolerance the mission sets in metres, and the time
        limit is only a backstop. Expressing it this way also makes a failure
        of flow itself self-limiting: the unobserved term in :attr:`sigma_m`
        grows at the airframe's speed, so a second of silence costs tens of
        metres and the gate closes on its own.
        """
        if not self._active or self._expired:
            return False
        if self.max_position_sigma_m > 0.0 and self.sigma_m > self.max_position_sigma_m:
            return False
        return True

    @property
    def flow_lost(self) -> bool:
        """Has optical flow stopped producing displacements altogether?

        The hand-off point to the third tier. Reported separately from
        :attr:`publishable` because the two mean different things to an
        operator: "drifted too far to trust" is the system working as designed,
        while "the camera can no longer see the ground move" is a failure, and
        it is the one that means nothing but the autopilot's own IMU remains.
        """
        if not self._active:
            return False
        return self.unobserved_s > self._flow_timeout_s

    @property
    def _flow_timeout_s(self) -> float:
        """How long without a displacement counts as flow having failed.

        Derived rather than configured: the moment the unmeasured term alone
        exceeds the mission's tolerance, the position is worthless whatever
        else is true, so there is nothing to gain by waiting longer.
        """
        if self.max_position_sigma_m <= 0.0 or self.unobserved_speed_mps <= 0.0:
            return float("inf")
        return self.max_position_sigma_m / self.unobserved_speed_mps

    @property
    def guides_search(self) -> bool:
        """May the accumulated displacement still steer the tile search?

        This is the distinction that lets the system recover on its own. The
        displacement stops being good enough to *navigate* on long before it
        stops being the best available guess at *where to look*, and the two
        were previously governed by one limit: the accumulator froze at the
        publishing budget, so the search window stayed at the last fix while
        the aircraft flew on at 22 m/s. The radius grew, around a point the
        aircraft had left.
        """
        return (
            self._active
            and self.max_search_guidance_s > 0.0
            and self.wall_elapsed_s <= self.max_search_guidance_s
        )

    @property
    def offset_ned_m(self) -> Tuple[float, float]:
        """Displacement accumulated since the fix this started from."""
        return (self._north_m, self._east_m)

    @property
    def elapsed_s(self) -> float:
        return self._elapsed_s

    @property
    def distance_m(self) -> float:
        return self._distance_m

    @property
    def sigma_m(self) -> float:
        """Position uncertainty: the fix's own, grown by two separate things.

        **Distance flown**, measured: optical flow tracked it, and the drift is
        a fraction of it.

        **Time not observed**, unmeasured: seconds during which flow produced
        nothing, so the displacement over them is unknown and could be anything
        up to the airframe's speed. This term is why a frozen position cannot
        masquerade as a confident one. Without it, flow failing meant the
        position stopped moving *and the uncertainty stopped growing*, so the
        system published "you are here, 3 m sure, as of now" indefinitely while
        the aircraft flew away at 22 m/s.

        All three are added in quadrature because they are independent: the map
        fix's error does not become more wrong because the aircraft then flew
        somewhere, and neither does the tracked drift because the camera later
        lost its footing.
        """
        drift = self.drift_fraction * self._distance_m
        unobserved = self.unobserved_speed_mps * self.unobserved_s
        return math.sqrt(self._base_sigma_m ** 2 + drift ** 2 + unobserved ** 2)

    def accumulate(self, delta: OdometryDelta) -> bool:
        """Add one delta.  Returns False once the publishing budget is passed.

        Accumulation continues past that point, because the displacement is
        still the best guess at where the aircraft is for the purpose of
        aiming the tile search - see :attr:`guides_search`. What stops at the
        budget is *publishing*, which is what the return value reports.
        """
        if not self._active:
            return False
        self._last_delta_at = self._clock()
        self._elapsed_s += delta.dt_s
        self._north_m += delta.north_m
        self._east_m += delta.east_m
        self._distance_m += delta.distance_m
        if not self._expired and self.wall_elapsed_s > self.max_dead_reckoning_s:
            LOG.warning(
                "dead reckoning has run %.1f s (limit %.1f s) and %.0f m without a "
                "map fix; no longer publishing a position, but still steering the "
                "tile search with it for up to %.0f s",
                self.wall_elapsed_s,
                self.max_dead_reckoning_s,
                self._distance_m,
                self.max_search_guidance_s,
            )
            self._expired = True
        return not self._expired
