"""Turning a verified tile match into a georeferenced pose.

The chain implemented here is exactly the one the project specification calls
for::

    camera image  ->  map correspondences  ->  geometric transform
                  ->  map pixel location   ->  georeferenced position
                  ->  local NED position   ->  MAVLink ODOMETRY

Geometric model
---------------
The camera looks down at terrain that is locally planar, so a single homography
relates the tile raster to the live image.  Inverting it and evaluating it at
the camera's principal point gives the tile pixel the optical axis pierces -
the ground point directly under the camera for a nadir mount.  Georeferencing
that pixel gives latitude/longitude; the local NED conversion is anchored at
the EKF origin.

Two further quantities fall out of the same homography:

**Scale**  The local Jacobian gives metres of ground per live-image pixel.  With
calibrated intrinsics this yields the range along the optical axis, since for a
pinhole camera ``ground_metres_per_pixel = depth / focal_length``.  That is what
makes the vertical channel observable from imagery alone.

**Bearing**  The same Jacobian says which way the image +X axis points on the
ground.  Combined with the fixed camera-to-body rotation (and roll/pitch, when
the autopilot supplies them) this recovers vehicle yaw.

Assumptions, stated plainly: the terrain under the frame is flat at
``terrain_alt_m``, and the depth-from-scale relation is exact for a plane
perpendicular to the optical axis and first-order accurate for the small tilts a
stabilised multirotor holds.  Where a rangefinder or barometric height is
available, feed it in as ``height_hint_m`` - it is more trustworthy than
scale-derived height and is used in preference to it when
``prefer_height_hint`` is set.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..frames import (
    CameraExtrinsics,
    euler_to_matrix,
    matrix_to_quaternion,
    yaw_from_camera_axis_bearing,
)
from ..geo import LocalOrigin
from ..mapdb.tile import MapTile
from ..types import Attitude, GeoPoint, NEDPosition
from ..features.geometric import GeometricResult, _apply, local_jacobian

LOG = logging.getLogger(__name__)


class PoseError(RuntimeError):
    """The verified match could not be converted into a usable pose."""


@dataclass
class PoseSolution:
    """A georeferenced pose recovered from one tile match."""

    geo: GeoPoint
    position_ned: NEDPosition
    attitude: Attitude
    yaw_rad: float
    roll_rad: float
    pitch_rad: float
    #: Range from the camera to the ground along the optical axis, metres.
    range_m: float
    #: Height of the camera above the mapped terrain, metres.
    height_agl_m: float
    #: Ground sample distance of the live frame at the principal point, m/px.
    frame_gsd_mpp: float
    #: Tile pixel the optical axis pierces.
    tile_pixel: Tuple[float, float]
    height_source: str


class PoseSolver:
    """Converts a :class:`GeometricResult` into a :class:`PoseSolution`."""

    def __init__(
        self,
        extrinsics: CameraExtrinsics,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        prefer_height_hint: bool = False,
    ) -> None:
        self.extrinsics = extrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.prefer_height_hint = prefer_height_hint

    @classmethod
    def from_config(cls, camera_config, extrinsics: CameraExtrinsics) -> "PoseSolver":
        return cls(
            extrinsics=extrinsics,
            fx=camera_config.fx,
            fy=camera_config.fy,
            cx=camera_config.cx,
            cy=camera_config.cy,
        )

    @property
    def has_intrinsics(self) -> bool:
        return self.fx is not None and self.fy is not None and self.fx > 0 and self.fy > 0

    def principal_point(self, frame_size: Tuple[int, int]) -> Tuple[float, float]:
        """Image point the optical axis passes through, in live-frame pixels."""
        cx = self.cx if self.cx is not None else frame_size[0] * 0.5
        cy = self.cy if self.cy is not None else frame_size[1] * 0.5
        return (float(cx), float(cy))

    def solve(
        self,
        geometry: GeometricResult,
        tile: MapTile,
        origin: LocalOrigin,
        frame_size: Tuple[int, int],
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        terrain_alt_m: Optional[float] = None,
        height_hint_m: Optional[float] = None,
    ) -> PoseSolution:
        """Solve for the vehicle pose.  Raises :class:`PoseError` on failure."""
        if not geometry.ok or geometry.tile_from_frame is None:
            raise PoseError("geometric verification did not produce a usable transform")

        tile_from_frame = np.asarray(geometry.tile_from_frame, dtype=np.float64)
        principal = self.principal_point(frame_size)

        # -- 1. map pixel under the optical axis ----------------------------
        pierced = _apply(tile_from_frame, np.array([principal]))[0]
        if not np.all(np.isfinite(pierced)):
            raise PoseError("optical axis does not intersect the tile plane")
        col, row = float(pierced[0]), float(pierced[1])
        margin = 0.05 * max(tile.metadata.width_px, tile.metadata.height_px)
        if not (
            -margin <= col <= tile.metadata.width_px + margin
            and -margin <= row <= tile.metadata.height_px + margin
        ):
            raise PoseError(
                "optical axis pierces the tile at (%.1f, %.1f), outside its extent" % (col, row)
            )

        # -- 2. that pixel in geographic and local NED coordinates ----------
        terrain_alt = tile.terrain_alt_m if terrain_alt_m is None else terrain_alt_m
        ground_geo = tile.pixel_to_geo(col, row)
        ground_geo = GeoPoint(ground_geo.lat_deg, ground_geo.lon_deg, terrain_alt)
        ground_ned = origin.to_ned(ground_geo)

        # -- 3. metric scale and ground bearing of the image axes ----------
        jacobian = local_jacobian(tile_from_frame, np.array(principal))
        if not np.all(np.isfinite(jacobian)):
            raise PoseError("transform Jacobian is not finite at the principal point")
        metres_per_col, metres_per_row = tile.georeference.local_scale(ground_geo)

        # Columns of the Jacobian: displacement in tile pixels produced by a
        # one-pixel step along the live image's +X and +Y axes.
        d_x = jacobian[:, 0]
        d_y = jacobian[:, 1]
        gsd_x = math.hypot(d_x[0] * metres_per_col, d_x[1] * metres_per_row)
        gsd_y = math.hypot(d_y[0] * metres_per_col, d_y[1] * metres_per_row)
        frame_gsd = math.sqrt(max(gsd_x * gsd_y, 0.0))
        if not math.isfinite(frame_gsd) or frame_gsd <= 0.0:
            raise PoseError("degenerate image scale")

        # Tile rasters are north-up: +column is east, +row is south.
        east_component = d_x[0] * metres_per_col
        north_component = -d_x[1] * metres_per_row
        axis_bearing = math.atan2(east_component, north_component)

        # -- 4. yaw ---------------------------------------------------------
        yaw = yaw_from_camera_axis_bearing(
            axis_bearing, self.extrinsics, roll_rad=roll_rad, pitch_rad=pitch_rad
        )
        R_ned_from_body = euler_to_matrix(roll_rad, pitch_rad, yaw)
        R_ned_from_camera = self.extrinsics.R_ned_from_camera(R_ned_from_body)

        # -- 5. range along the optical axis --------------------------------
        optical_axis_ned = R_ned_from_camera @ np.array([0.0, 0.0, 1.0])
        down_component = float(optical_axis_ned[2])

        range_m, height_agl, height_source = self._resolve_range(
            frame_gsd, height_hint_m, down_component
        )

        # -- 6. camera centre, then body origin -----------------------------
        ground_vector = np.array(ground_ned.as_tuple(), dtype=float)
        camera_ned = ground_vector - range_m * optical_axis_ned
        body_ned = self.extrinsics.body_origin_from_camera_position(
            camera_ned, R_ned_from_body
        )

        position_ned = NEDPosition(float(body_ned[0]), float(body_ned[1]), float(body_ned[2]))
        geo = origin.to_geo(position_ned)
        attitude = Attitude(*matrix_to_quaternion(R_ned_from_body))

        return PoseSolution(
            geo=geo,
            position_ned=position_ned,
            attitude=attitude,
            yaw_rad=yaw,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            range_m=range_m,
            height_agl_m=height_agl,
            frame_gsd_mpp=frame_gsd,
            tile_pixel=(col, row),
            height_source=height_source,
        )

    def _resolve_range(
        self,
        frame_gsd: float,
        height_hint_m: Optional[float],
        optical_axis_down_component: float,
    ) -> Tuple[float, float, str]:
        """Work out how far the ground is along the optical axis.

        Two independent sources, in preference order controlled by
        ``prefer_height_hint``:

        ``scale``  ``range = focal_length_px * metres_per_pixel``.  Needs
                   calibrated intrinsics; degrades gracefully with tilt.
        ``hint``   A vertical height above terrain from a rangefinder or the
                   barometer, converted to a slant range by dividing by the
                   downward component of the optical axis.
        """
        hint_usable = (
            height_hint_m is not None
            and math.isfinite(height_hint_m)
            and height_hint_m > 0.1
            and optical_axis_down_component > 0.1
        )
        scale_usable = self.has_intrinsics

        use_hint = hint_usable and (self.prefer_height_hint or not scale_usable)
        if use_hint:
            range_m = height_hint_m / optical_axis_down_component
            return (range_m, float(height_hint_m), "hint")

        if scale_usable:
            focal_px = math.sqrt(self.fx * self.fy)
            range_m = focal_px * frame_gsd
            height_agl = range_m * max(optical_axis_down_component, 0.0)
            return (range_m, height_agl, "scale")

        raise PoseError(
            "no height source: set camera.fx/fy (calibrated intrinsics) or supply "
            "a rangefinder/barometric height to make the vertical channel observable"
        )
