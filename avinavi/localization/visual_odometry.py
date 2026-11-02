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

    # -- the cycle ----------------------------------------------------------
    def update(
        self,
        frame: np.ndarray,
        timestamp_monotonic: float,
        height_m: Optional[float],
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        yaw_rad: float = 0.0,
    ) -> Optional[OdometryDelta]:
        """Track from the previous frame to this one.

        Returns ``None`` on the first frame, or whenever an honest estimate
        cannot be made - too few points survived, they disagree too much, or the
        implied speed is impossible.  ``None`` means "no information", which the
        caller must treat differently from a zero displacement.
        """
        cv2 = _require_cv2()
        if not self.available or height_m is None or height_m <= 0.0:
            self.reset()
            return None

        gray, scale = self._prepare(cv2, frame)
        R = self.extrinsics.R_ned_from_camera(
            euler_to_matrix(roll_rad, pitch_rad, yaw_rad)
        )
        state = (R, float(height_m), float(timestamp_monotonic))

        previous_gray = self._previous_gray
        previous_points = self._previous_points
        previous_state = self._previous_state

        # Always leave this frame as the basis for the next cycle, whatever
        # happens below: a failed estimate must not also lose the frame.
        self._previous_gray = gray
        self._scale = scale
        self._previous_state = state
        self._previous_points = self._detect(cv2, gray)

        if previous_gray is None or previous_points is None or previous_state is None:
            return None
        if previous_points.shape[0] < self.min_points:
            return None

        dt = state[2] - previous_state[2]
        if dt <= 0.0:
            return None

        tracked, previous_kept = self._track(cv2, previous_gray, gray, previous_points)
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

    def _track(self, cv2, previous_gray, gray, previous_points):
        """Forward-backward Lucas-Kanade.

        The backward check is what makes this usable over repetitive ground -
        ploughed fields, orchards, water. A point that tracks forward to a
        plausible place but does not track back to where it started was matched
        to a different, similar-looking feature, and it would otherwise
        contribute a confident wrong displacement.
        """
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        forward, status, _err = cv2.calcOpticalFlowPyrLK(
            previous_gray, gray, previous_points, None,
            winSize=(21, 21), maxLevel=3, criteria=criteria,
        )
        if forward is None or status is None:
            return (None, None)
        ok = status.reshape(-1).astype(bool)
        if not ok.any():
            return (None, None)

        backward, back_status, _err = cv2.calcOpticalFlowPyrLK(
            gray, previous_gray, forward, None,
            winSize=(21, 21), maxLevel=3, criteria=criteria,
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

        if delta.speed_mps > self.max_speed_mps:
            LOG.warning(
                "visual odometry implied %.0f m/s over %.0f ms; rejecting it as a "
                "tracking failure rather than believing it",
                delta.speed_mps,
                dt_s * 1000.0,
            )
            return None
        return delta

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
    ) -> None:
        self.drift_fraction = float(drift_fraction)
        self.max_dead_reckoning_s = float(max_dead_reckoning_s)
        self._north_m = 0.0
        self._east_m = 0.0
        self._distance_m = 0.0
        self._elapsed_s = 0.0
        self._base_sigma_m = 0.0
        self._active = False

    @classmethod
    def from_config(cls, config) -> "DeadReckoner":
        return cls(
            drift_fraction=config.drift_fraction,
            max_dead_reckoning_s=config.max_dead_reckoning_s,
        )

    def start(self, base_sigma_m: float) -> None:
        """Begin reckoning from a map fix with the given position sigma."""
        self._north_m = 0.0
        self._east_m = 0.0
        self._distance_m = 0.0
        self._elapsed_s = 0.0
        self._base_sigma_m = max(0.0, float(base_sigma_m))
        self._active = True

    def clear(self) -> None:
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def expired(self) -> bool:
        return self._active and self._elapsed_s > self.max_dead_reckoning_s

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
        """Position uncertainty: the fix's own, grown by the distance flown.

        Added in quadrature rather than summed, because the two are
        independent: the map fix's error does not become more wrong because the
        aircraft then flew somewhere.
        """
        drift = self.drift_fraction * self._distance_m
        return math.hypot(self._base_sigma_m, drift)

    def accumulate(self, delta: OdometryDelta) -> bool:
        """Add one delta.  Returns False once the limit is passed."""
        if not self._active:
            return False
        self._elapsed_s += delta.dt_s
        if self._elapsed_s > self.max_dead_reckoning_s:
            LOG.warning(
                "dead reckoning has run %.1f s (limit %.1f s) and %.0f m without a "
                "map fix; stopping rather than publishing an unjustifiable position",
                self._elapsed_s,
                self.max_dead_reckoning_s,
                self._distance_m,
            )
            self._active = False
            return False
        self._north_m += delta.north_m
        self._east_m += delta.east_m
        self._distance_m += delta.distance_m
        return True
