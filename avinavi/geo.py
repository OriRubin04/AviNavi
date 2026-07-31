"""Geodetic, local-NED and map-pixel conversions.

Three coordinate systems meet in this module; ``docs/coordinate-frames.md``
describes them in prose.

**Geographic (WGS-84)**
    Latitude/longitude in degrees, altitude in metres.  This is what the map
    tiles are georeferenced against and what the operator picks in Mission
    Planner.

**Local NED**
    A flat tangent plane anchored at :class:`LocalOrigin`.  X points north,
    Y east, Z down, all in metres.  This is the frame the MAVLink ``ODOMETRY``
    position fields are expressed in, and it must be anchored at *exactly* the
    same origin that ArduPilot's EKF uses - which is why the origin is taken
    from the autopilot's own ``GPS_GLOBAL_ORIGIN`` message rather than invented
    locally.

**Map pixel**
    Column/row inside a tile image.  Tiles are stored north-up, so +column is
    east and +row is south.

The conversion chain used by the localizer runs:

    map pixel  ->  geographic  ->  local NED

and the tile search runs the other way to decide which tiles to load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from .types import GeoPoint, NEDPosition

#: WGS-84 ellipsoid.
WGS84_SEMI_MAJOR_M = 6378137.0
WGS84_FLATTENING = 1.0 / 298.257223563
WGS84_ECC_SQ = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)


def meridional_radius_m(lat_deg: float) -> float:
    """Radius of curvature in the meridian (north-south) at ``lat_deg``."""
    s = math.sin(math.radians(lat_deg))
    denom = 1.0 - WGS84_ECC_SQ * s * s
    return WGS84_SEMI_MAJOR_M * (1.0 - WGS84_ECC_SQ) / (denom ** 1.5)


def prime_vertical_radius_m(lat_deg: float) -> float:
    """Radius of curvature in the prime vertical (east-west) at ``lat_deg``."""
    s = math.sin(math.radians(lat_deg))
    return WGS84_SEMI_MAJOR_M / math.sqrt(1.0 - WGS84_ECC_SQ * s * s)


@dataclass(frozen=True)
class LocalOrigin:
    """Anchor of the local NED frame.

    Conversions use the local tangent-plane ("flat earth") approximation with
    the WGS-84 radii of curvature evaluated at the origin latitude.  Over the
    few kilometres a tile database realistically covers the error is well under
    a metre, and it matches how ArduPilot itself relates its EKF origin to
    latitude/longitude.
    """

    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat_deg <= 90.0:
            raise ValueError("origin latitude out of range: %r" % self.lat_deg)
        if not -180.0 <= self.lon_deg <= 180.0:
            raise ValueError("origin longitude out of range: %r" % self.lon_deg)

    @classmethod
    def from_geo(cls, point: GeoPoint) -> "LocalOrigin":
        return cls(point.lat_deg, point.lon_deg, point.alt_m)

    @property
    def geo(self) -> GeoPoint:
        return GeoPoint(self.lat_deg, self.lon_deg, self.alt_m)

    @property
    def metres_per_deg_lat(self) -> float:
        return meridional_radius_m(self.lat_deg) * math.pi / 180.0

    @property
    def metres_per_deg_lon(self) -> float:
        return (
            prime_vertical_radius_m(self.lat_deg)
            * math.cos(math.radians(self.lat_deg))
            * math.pi
            / 180.0
        )

    def to_ned(self, point: GeoPoint) -> NEDPosition:
        """Geographic -> local NED (metres).  Down is positive."""
        north = (point.lat_deg - self.lat_deg) * self.metres_per_deg_lat
        east = _wrap_deg(point.lon_deg - self.lon_deg) * self.metres_per_deg_lon
        down = self.alt_m - point.alt_m
        return NEDPosition(north, east, down)

    def to_geo(self, ned: NEDPosition) -> GeoPoint:
        """Local NED (metres) -> geographic."""
        lat = self.lat_deg + ned.north_m / self.metres_per_deg_lat
        lon = self.lon_deg + ned.east_m / self.metres_per_deg_lon
        alt = self.alt_m - ned.down_m
        return GeoPoint(lat, _wrap_deg(lon), alt)

    def offset_geo(self, point: GeoPoint, north_m: float, east_m: float) -> GeoPoint:
        """Move ``point`` by a north/east offset in metres."""
        ned = self.to_ned(point)
        return self.to_geo(
            NEDPosition(ned.north_m + north_m, ned.east_m + east_m, ned.down_m)
        )


def _wrap_deg(delta_deg: float) -> float:
    """Wrap a longitude difference into [-180, 180)."""
    return (delta_deg + 180.0) % 360.0 - 180.0


def haversine_distance_m(a: GeoPoint, b: GeoPoint) -> float:
    """Great-circle distance between two geographic points, metres."""
    lat1 = math.radians(a.lat_deg)
    lat2 = math.radians(b.lat_deg)
    dlat = lat2 - lat1
    dlon = math.radians(_wrap_deg(b.lon_deg - a.lon_deg))
    h = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    mean_radius = 0.5 * (
        meridional_radius_m(a.lat_deg) + prime_vertical_radius_m(a.lat_deg)
    )
    return 2.0 * mean_radius * math.asin(min(1.0, math.sqrt(h)))


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned geographic bounding box in WGS-84 degrees."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if self.east <= self.west or self.north <= self.south:
            raise ValueError(
                "degenerate bounding box: %r" % ((self.west, self.south, self.east, self.north),)
            )

    @property
    def center(self) -> GeoPoint:
        return GeoPoint(
            0.5 * (self.south + self.north), 0.5 * (self.west + self.east)
        )

    def contains(self, point: GeoPoint) -> bool:
        return (
            self.west <= point.lon_deg <= self.east
            and self.south <= point.lat_deg <= self.north
        )

    def expanded(self, degrees_lat: float, degrees_lon: float) -> "BoundingBox":
        return BoundingBox(
            self.west - degrees_lon,
            self.south - degrees_lat,
            self.east + degrees_lon,
            self.north + degrees_lat,
        )

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def distance_to_m(self, point: GeoPoint) -> float:
        """Metres from ``point`` to the nearest edge of the box (0 if inside).

        Used by the tile search to decide whether a tile is within the search
        radius without requiring the aircraft to be inside the tile.
        """
        lat = min(max(point.lat_deg, self.south), self.north)
        lon = min(max(point.lon_deg, self.west), self.east)
        if lat == point.lat_deg and lon == point.lon_deg:
            return 0.0
        m_per_deg_lat = meridional_radius_m(point.lat_deg) * math.pi / 180.0
        m_per_deg_lon = (
            prime_vertical_radius_m(point.lat_deg)
            * math.cos(math.radians(point.lat_deg))
            * math.pi
            / 180.0
        )
        dn = (lat - point.lat_deg) * m_per_deg_lat
        de = _wrap_deg(lon - point.lon_deg) * m_per_deg_lon
        return math.hypot(dn, de)


class Georeference:
    """Maps tile pixel coordinates to geographic coordinates.

    Two forms are supported:

    ``transform``
        A GDAL-style affine 6-tuple ``(a, b, c, d, e, f)`` such that::

            x = a * col + b * row + c
            y = d * col + e * row + f

        with ``(x, y)`` in the tile's CRS.  For a CRS other than EPSG:4326 the
        result is reprojected with pyproj, which is imported lazily so the
        runtime does not need it when tiles are already in EPSG:4326.

    bounds only
        When no transform is stored, a north-up transform is derived from the
        WGS-84 bounding box and the pixel dimensions.  This is the common case
        for tiles cut from a web-mercator/lat-lon mosaic.

    Pixel convention: ``(col, row)`` are continuous pixel coordinates with
    ``(0, 0)`` at the *corner* of the top-left pixel, matching GDAL.
    """

    def __init__(
        self,
        width_px: int,
        height_px: int,
        bounds: BoundingBox,
        crs: str = "EPSG:4326",
        transform: Optional[Sequence[float]] = None,
    ) -> None:
        if width_px <= 0 or height_px <= 0:
            raise ValueError("tile pixel dimensions must be positive")
        self.width_px = int(width_px)
        self.height_px = int(height_px)
        self.bounds = bounds
        self.crs = crs
        self._transform = tuple(float(v) for v in transform) if transform else None
        if self._transform is not None and len(self._transform) != 6:
            raise ValueError("transform must have 6 elements")
        self._to_wgs84 = None  # lazily built pyproj Transformer

    # -- pixel -> geographic ------------------------------------------------
    def pixel_to_geo(self, col: float, row: float) -> GeoPoint:
        """Tile pixel -> WGS-84 geographic position (altitude left at 0)."""
        if self._transform is None:
            lon = self.bounds.west + (col / self.width_px) * (
                self.bounds.east - self.bounds.west
            )
            # Row 0 is the *north* edge of a north-up tile.
            lat = self.bounds.north - (row / self.height_px) * (
                self.bounds.north - self.bounds.south
            )
            return GeoPoint(lat, lon)

        a, b, c, d, e, f = self._transform
        x = a * col + b * row + c
        y = d * col + e * row + f
        if self._is_wgs84():
            return GeoPoint(y, x)
        lon, lat = self._transformer().transform(x, y)
        return GeoPoint(lat, lon)

    def geo_to_pixel(self, point: GeoPoint) -> Tuple[float, float]:
        """WGS-84 geographic position -> tile pixel ``(col, row)``."""
        if self._transform is None:
            col = (
                (point.lon_deg - self.bounds.west)
                / (self.bounds.east - self.bounds.west)
                * self.width_px
            )
            row = (
                (self.bounds.north - point.lat_deg)
                / (self.bounds.north - self.bounds.south)
                * self.height_px
            )
            return (col, row)

        a, b, c, d, e, f = self._transform
        if self._is_wgs84():
            x, y = point.lon_deg, point.lat_deg
        else:
            x, y = self._transformer(inverse=True).transform(point.lon_deg, point.lat_deg)
        det = a * e - b * d
        if abs(det) < 1e-18:
            raise ValueError("tile transform is not invertible")
        dx = x - c
        dy = y - f
        col = (e * dx - b * dy) / det
        row = (-d * dx + a * dy) / det
        return (col, row)

    # -- scale --------------------------------------------------------------
    @property
    def ground_resolution_mpp(self) -> float:
        """Average ground sample distance in metres per pixel.

        Computed from the geographic extent so it stays correct regardless of
        which of the two georeferencing forms is in use.
        """
        center = self.bounds.center
        m_per_deg_lat = meridional_radius_m(center.lat_deg) * math.pi / 180.0
        m_per_deg_lon = (
            prime_vertical_radius_m(center.lat_deg)
            * math.cos(math.radians(center.lat_deg))
            * math.pi
            / 180.0
        )
        width_m = (self.bounds.east - self.bounds.west) * m_per_deg_lon
        height_m = (self.bounds.north - self.bounds.south) * m_per_deg_lat
        return 0.5 * (width_m / self.width_px + height_m / self.height_px)

    def local_scale(self, point: GeoPoint) -> Tuple[float, float]:
        """Metres per pixel along the column and row axes near ``point``."""
        col, row = self.geo_to_pixel(point)
        origin = LocalOrigin.from_geo(self.pixel_to_geo(col, row))
        east_step = origin.to_ned(self.pixel_to_geo(col + 1.0, row))
        south_step = origin.to_ned(self.pixel_to_geo(col, row + 1.0))
        return (
            math.hypot(east_step.north_m, east_step.east_m),
            math.hypot(south_step.north_m, south_step.east_m),
        )

    # -- internals ----------------------------------------------------------
    def _is_wgs84(self) -> bool:
        return self.crs.strip().upper() in ("EPSG:4326", "WGS84", "CRS:84", "OGC:CRS84")

    def _transformer(self, inverse: bool = False):
        try:
            from pyproj import Transformer  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "tile CRS %s requires pyproj; install the 'maps' extra or "
                "re-cut the tiles in EPSG:4326" % self.crs
            ) from exc
        if inverse:
            return Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        if self._to_wgs84 is None:
            self._to_wgs84 = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        return self._to_wgs84


def bounding_box_around(point: GeoPoint, radius_m: float) -> BoundingBox:
    """Smallest lat/lon box containing every point within ``radius_m``."""
    if radius_m <= 0:
        raise ValueError("radius must be positive")
    m_per_deg_lat = meridional_radius_m(point.lat_deg) * math.pi / 180.0
    cos_lat = math.cos(math.radians(point.lat_deg))
    m_per_deg_lon = prime_vertical_radius_m(point.lat_deg) * cos_lat * math.pi / 180.0
    dlat = radius_m / m_per_deg_lat
    # Near the poles the longitude span degenerates; clamp to a full circle.
    dlon = 180.0 if m_per_deg_lon < 1e-6 else min(180.0, radius_m / m_per_deg_lon)
    return BoundingBox(
        max(-180.0, point.lon_deg - dlon),
        max(-90.0, point.lat_deg - dlat),
        min(180.0, point.lon_deg + dlon),
        min(90.0, point.lat_deg + dlat),
    )


def mean_geo(points: Iterable[GeoPoint]) -> Optional[GeoPoint]:
    """Arithmetic mean of geographic points (adequate for nearby tiles)."""
    points = list(points)
    if not points:
        return None
    n = float(len(points))
    return GeoPoint(
        sum(p.lat_deg for p in points) / n,
        sum(p.lon_deg for p in points) / n,
        sum(p.alt_m for p in points) / n,
    )
