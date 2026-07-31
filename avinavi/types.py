"""Core value types shared across the pipeline.

Everything here is a plain dataclass with no heavyweight imports so that the
MAVLink layer, the localization engine and the tests can all agree on the same
vocabulary without pulling in torch or OpenCV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence, Tuple


class InvalidReason(str, Enum):
    """Why a visual navigation solution is not usable.

    Anything other than :data:`NONE` means the system must *not* publish an
    external navigation solution to ArduPilot.
    """

    NONE = "none"
    NO_CANDIDATE_TILES = "no_candidate_tiles"
    NO_FRAME = "no_frame"
    CAMERA_FAILURE = "camera_failure"
    TOO_FEW_KEYPOINTS = "too_few_keypoints"
    TOO_FEW_MATCHES = "too_few_matches"
    GEOMETRIC_VERIFICATION_FAILED = "geometric_verification_failed"
    LOW_CONFIDENCE = "low_confidence"
    STALE = "stale"
    NOT_INITIALIZED = "not_initialized"
    MAVLINK_DISCONNECTED = "mavlink_disconnected"


@dataclass(frozen=True)
class GeoPoint:
    """A WGS-84 geographic position.

    ``alt_m`` is metres above the WGS-84 ellipsoid-referenced MSL datum used by
    the map product; the whole system only ever uses *differences* in altitude
    against :class:`~avinavi.geo.LocalOrigin`, so the absolute datum only has to
    be self-consistent.
    """

    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0


@dataclass(frozen=True)
class NEDPosition:
    """Position in the local navigation frame: X=North, Y=East, Z=Down (metres)."""

    north_m: float
    east_m: float
    down_m: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.north_m, self.east_m, self.down_m)


@dataclass(frozen=True)
class NEDVelocity:
    """Velocity in the local navigation frame (m/s), same axes as :class:`NEDPosition`."""

    north_mps: float
    east_mps: float
    down_mps: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.north_mps, self.east_mps, self.down_mps)


@dataclass(frozen=True)
class Attitude:
    """Body attitude relative to the local NED frame, as a quaternion.

    Stored in MAVLink/ArduPilot ordering ``(w, x, y, z)`` and describing the
    rotation that takes a vector expressed in the **body FRD** frame into the
    **local NED** frame.
    """

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.w, self.x, self.y, self.z)


@dataclass
class MatchQuality:
    """Match statistics for one camera frame against one map tile.

    Every field required by the project specification is materialised here so
    that the confidence model, the diagnostics view and the CSV log all read
    from the same record.
    """

    number_of_matches: int = 0
    number_of_inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error_px: float = float("nan")
    matching_confidence: float = 0.0
    spatial_spread: float = 0.0
    localization_confidence: float = 0.0

    def as_dict(self) -> dict:
        return {
            "number_of_matches": self.number_of_matches,
            "number_of_inliers": self.number_of_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "reprojection_error_px": (
                None
                if math.isnan(self.reprojection_error_px)
                else round(self.reprojection_error_px, 3)
            ),
            "matching_confidence": round(self.matching_confidence, 4),
            "spatial_spread": round(self.spatial_spread, 4),
            "localization_confidence": round(self.localization_confidence, 4),
        }


@dataclass
class VisualFix:
    """One visual localization result.

    A fix with ``valid == False`` still carries diagnostics (which tile was
    tried, how many matches were found) so failures are debuggable, but it must
    never be converted into an external navigation message.
    """

    timestamp_monotonic: float
    timestamp_unix: float
    valid: bool
    reason: InvalidReason = InvalidReason.NONE
    geo: Optional[GeoPoint] = None
    position_ned: Optional[NEDPosition] = None
    velocity_ned: Optional[NEDVelocity] = None
    attitude: Optional[Attitude] = None
    yaw_rad: Optional[float] = None
    quality: MatchQuality = field(default_factory=MatchQuality)
    tile_id: Optional[str] = None
    candidate_tiles: int = 0
    ground_resolution_mpp: Optional[float] = None
    processing_time_s: float = 0.0
    # Position standard deviations (metres) and yaw standard deviation (rad)
    # derived from the match quality; consumed when building covariances.
    position_sigma_m: float = float("nan")
    yaw_sigma_rad: float = float("nan")

    @property
    def confidence(self) -> float:
        return self.quality.localization_confidence


@dataclass
class NavigationSolution:
    """A pose ready to be published as MAVLink ``ODOMETRY``.

    This is what the output stage consumes.  It may be a direct visual fix or a
    short forward-propagation of one (see
    :class:`~avinavi.localization.propagator.PosePropagator`); ``age_s``
    records how far it has been propagated so the output stage can refuse to
    send stale data.
    """

    timestamp_monotonic: float
    timestamp_unix: float
    position_ned: NEDPosition
    attitude: Attitude
    velocity_body_frd: Tuple[float, float, float]
    angular_velocity_body: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_sigma_m: float = 1.0
    velocity_sigma_mps: float = 1.0
    yaw_sigma_rad: float = 0.2
    confidence: float = 0.0
    age_s: float = 0.0
    propagated: bool = False
    reset_counter: int = 0

    @property
    def quality_percent(self) -> int:
        """ODOMETRY ``quality`` field: -1 failed, 0 unknown, 1 worst, 100 best."""
        pct = int(round(max(0.0, min(1.0, self.confidence)) * 100.0))
        return max(1, pct)


def covariance_upper_triangle(
    diagonal: Sequence[float],
) -> list:
    """Build the 21-element row-major upper triangle of a diagonal 6x6 matrix.

    MAVLink packs a 6x6 covariance as the upper-right triangle, row major:
    row 0 contributes 6 entries (indices 0-5), row 1 contributes 5 (6-10),
    row 2 contributes 4 (11-14), row 3 contributes 3 (15-17), row 4 contributes
    2 (18-19) and row 5 contributes 1 (index 20).  The diagonal therefore lands
    at indices ``0, 6, 11, 15, 18, 20`` - which is exactly where ArduPilot's
    ``GCS_MAVLINK::handle_odometry`` reads the position and angle variances
    from when computing ``posErr`` and ``angErr``.
    """
    if len(diagonal) != 6:
        raise ValueError("expected 6 diagonal variances, got %d" % len(diagonal))
    out = [0.0] * 21
    for row, value in enumerate(diagonal):
        out[_UPPER_TRIANGLE_DIAGONAL_INDEX[row]] = float(value)
    return out


#: Index of each diagonal element inside the 21-element packed upper triangle.
_UPPER_TRIANGLE_DIAGONAL_INDEX = (0, 6, 11, 15, 18, 20)
