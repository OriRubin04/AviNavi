"""Geodesy: local NED conversions, bounding boxes and tile georeferencing."""

from __future__ import annotations

import math

import pytest

from avinavi.geo import (
    BoundingBox,
    Georeference,
    LocalOrigin,
    bounding_box_around,
    haversine_distance_m,
)
from avinavi.types import GeoPoint, NEDPosition


def test_ned_round_trip(origin):
    for north, east, down in [(0, 0, 0), (1000, -500, -20), (-2500, 3000, 120)]:
        point = origin.to_geo(NEDPosition(north, east, down))
        back = origin.to_ned(point)
        assert back.north_m == pytest.approx(north, abs=1e-3)
        assert back.east_m == pytest.approx(east, abs=1e-3)
        assert back.down_m == pytest.approx(down, abs=1e-6)


def test_down_is_positive_downwards(origin):
    """Flying 100 m above the origin must give a negative Down."""
    above = GeoPoint(origin.lat_deg, origin.lon_deg, origin.alt_m + 100.0)
    assert origin.to_ned(above).down_m == pytest.approx(-100.0)


def test_north_east_axes_point_the_right_way(origin):
    north = origin.to_geo(NEDPosition(100.0, 0.0, 0.0))
    east = origin.to_geo(NEDPosition(0.0, 100.0, 0.0))
    assert north.lat_deg > origin.lat_deg
    assert north.lon_deg == pytest.approx(origin.lon_deg)
    assert east.lon_deg > origin.lon_deg
    assert east.lat_deg == pytest.approx(origin.lat_deg)


def test_ned_distance_matches_haversine(origin):
    """The flat-earth approximation must stay tight over map-sized distances."""
    point = origin.to_geo(NEDPosition(3000.0, 4000.0, 0.0))
    great_circle = haversine_distance_m(origin.geo, point)
    assert great_circle == pytest.approx(5000.0, rel=1e-3)


def test_longitude_scale_shrinks_towards_the_pole():
    equator = LocalOrigin(0.0, 0.0)
    high = LocalOrigin(60.0, 0.0)
    # cos(60 deg) = 0.5, so a degree of longitude is about half as long.
    assert high.metres_per_deg_lon == pytest.approx(
        equator.metres_per_deg_lon * 0.5, rel=0.01
    )


def test_longitude_wrap_across_the_antimeridian():
    origin = LocalOrigin(0.0, 179.999)
    ned = origin.to_ned(GeoPoint(0.0, -179.999))
    # 0.002 degrees east, not 359.998 degrees west.
    assert 0 < ned.east_m < 300


def test_bounding_box_rejects_degenerate_input():
    with pytest.raises(ValueError):
        BoundingBox(1.0, 1.0, 1.0, 2.0)


def test_bounding_box_distance_is_zero_inside(tile_bounds):
    assert tile_bounds.distance_to_m(tile_bounds.center) == 0.0


def test_bounding_box_distance_outside(tile_bounds):
    origin = LocalOrigin.from_geo(tile_bounds.center)
    # 2 km north of the box centre; the box is ~1 km tall, so ~1.5 km to its edge.
    far = origin.to_geo(NEDPosition(2000.0, 0.0, 0.0))
    distance = tile_bounds.distance_to_m(far)
    assert 1000.0 < distance < 2000.0


def test_bounding_box_around_covers_the_radius(origin):
    box = bounding_box_around(origin.geo, 500.0)
    for bearing in range(0, 360, 45):
        angle = math.radians(bearing)
        point = origin.to_geo(
            NEDPosition(499.0 * math.cos(angle), 499.0 * math.sin(angle), 0.0)
        )
        assert box.contains(point), "radius not covered at bearing %d" % bearing


def test_georeference_pixel_round_trip(tile_bounds):
    georeference = Georeference(1024, 1024, tile_bounds)
    for col, row in [(0.0, 0.0), (512.0, 512.0), (1023.0, 7.0)]:
        point = georeference.pixel_to_geo(col, row)
        back_col, back_row = georeference.geo_to_pixel(point)
        assert back_col == pytest.approx(col, abs=1e-6)
        assert back_row == pytest.approx(row, abs=1e-6)


def test_georeference_row_zero_is_north(tile_bounds):
    """Tiles are north-up: increasing row must go south."""
    georeference = Georeference(512, 512, tile_bounds)
    top = georeference.pixel_to_geo(256, 0)
    bottom = georeference.pixel_to_geo(256, 511)
    assert top.lat_deg > bottom.lat_deg


def test_georeference_column_zero_is_west(tile_bounds):
    georeference = Georeference(512, 512, tile_bounds)
    left = georeference.pixel_to_geo(0, 256)
    right = georeference.pixel_to_geo(511, 256)
    assert left.lon_deg < right.lon_deg


def test_georeference_ground_resolution(tile_bounds):
    georeference = Georeference(1000, 1000, tile_bounds)
    # ~1 km across 1000 px.
    assert 0.5 < georeference.ground_resolution_mpp < 1.5


def test_georeference_affine_transform_matches_bounds(tile_bounds):
    """An explicit north-up affine transform must agree with the bounds form."""
    width = height = 256
    a = (tile_bounds.east - tile_bounds.west) / width
    e = -(tile_bounds.north - tile_bounds.south) / height
    transform = (a, 0.0, tile_bounds.west, 0.0, e, tile_bounds.north)

    from_bounds = Georeference(width, height, tile_bounds)
    from_transform = Georeference(width, height, tile_bounds, transform=transform)
    for col, row in [(0, 0), (128, 64), (255, 255)]:
        expected = from_bounds.pixel_to_geo(col, row)
        actual = from_transform.pixel_to_geo(col, row)
        assert actual.lat_deg == pytest.approx(expected.lat_deg, abs=1e-9)
        assert actual.lon_deg == pytest.approx(expected.lon_deg, abs=1e-9)


def test_local_scale_is_metres_per_pixel(tile_bounds):
    georeference = Georeference(1000, 1000, tile_bounds)
    col_scale, row_scale = georeference.local_scale(tile_bounds.center)
    assert 0.5 < col_scale < 1.5
    assert 0.5 < row_scale < 1.5
