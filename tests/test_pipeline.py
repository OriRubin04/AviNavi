"""End-to-end: synthetic camera frame -> geographic position -> ODOMETRY.

This is the test that would catch a sign error in the middle of the chain.  A
known camera pose is chosen, the correspondences a nadir camera at that pose
would produce are synthesised exactly, and the pipeline is asked to recover the
pose.  Because the ground truth is constructed rather than measured, any
discrepancy is a bug in the code under test.

SuperPoint and LightGlue are stubbed (see :mod:`tests.fakes`); everything from
RANSAC onwards - geometric verification, the pose solve, the confidence gate,
the tile search, the propagator and the MAVLink encoding - is the real thing.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from avinavi.config import Config
from avinavi.features.geometric import GeometricVerifier
from avinavi.frames import CameraExtrinsics, quaternion_to_euler, wrap_pi
from avinavi.geo import BoundingBox, LocalOrigin
from avinavi.localization.engine import LocalizationEngine
from avinavi.localization.pose import PoseError, PoseSolver
from avinavi.localization.propagator import PosePropagator
from avinavi.mapdb.database import MapDatabase
from avinavi.mapdb.tile import MapTile, TileFeatures, TileMetadata
from avinavi.mavlink.external_navigation import ExternalNavigationPublisher
from avinavi.types import GeoPoint, InvalidReason, NEDPosition
from tests.fakes import FakeConnection, NoMatchMatcher, ScriptedExtractor, ScriptedMatcher

# A 1024 px tile covering 512 m => 0.5 m/px, a realistic satellite basemap GSD.
TILE_PX = 1024
TILE_SPAN_M = 512.0
TILE_GSD = TILE_SPAN_M / TILE_PX

# Live camera: 640x480 with a 500 px focal length.
FRAME_W, FRAME_H = 640, 480
FOCAL_PX = 500.0


@pytest.fixture
def origin() -> LocalOrigin:
    return LocalOrigin(32.0853, 34.7818, 40.0)


@pytest.fixture
def tile(origin) -> MapTile:
    """A single north-up tile centred on the origin."""
    half = TILE_SPAN_M / 2.0
    south_west = origin.to_geo(NEDPosition(-half, -half, 0.0))
    north_east = origin.to_geo(NEDPosition(half, half, 0.0))
    metadata = TileMetadata(
        tile_id="tile_center",
        directory="/tmp/tile_center",
        width_px=TILE_PX,
        height_px=TILE_PX,
        bounds=BoundingBox(
            south_west.lon_deg, south_west.lat_deg, north_east.lon_deg, north_east.lat_deg
        ),
        terrain_alt_m=0.0,
    )
    return MapTile(metadata)


def synthesise_view(
    tile: MapTile,
    origin: LocalOrigin,
    camera_north: float,
    camera_east: float,
    height_agl: float,
    yaw_rad: float,
    extrinsics: CameraExtrinsics,
    num_points: int = 220,
    noise_px: float = 0.3,
    seed: int = 7,
):
    """Correspondences a nadir camera at the given pose would actually see.

    Points are sampled on the ground plane, projected into the live image with
    an exact pinhole model, and paired with their true tile pixels.  Points that
    fall outside the image are dropped, so the geometry the verifier sees is
    physically consistent rather than merely algebraically consistent.
    """
    rng = np.random.default_rng(seed)
    R_ned_from_body = np.array(
        [
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R_ned_from_camera = R_ned_from_body @ extrinsics.R_body_from_camera
    R_camera_from_ned = R_ned_from_camera.T

    camera_ned = np.array([camera_north, camera_east, -height_agl])
    # Sample ground points within the footprint the camera can actually see.
    footprint = height_agl / FOCAL_PX * max(FRAME_W, FRAME_H) * 0.6
    ground_north = camera_north + rng.uniform(-footprint, footprint, num_points)
    ground_east = camera_east + rng.uniform(-footprint, footprint, num_points)

    frame_points = []
    tile_points = []
    for north, east in zip(ground_north, ground_east):
        world = np.array([north, east, 0.0])
        in_camera = R_camera_from_ned @ (world - camera_ned)
        if in_camera[2] <= 1e-6:
            continue
        u = FOCAL_PX * in_camera[0] / in_camera[2] + FRAME_W / 2.0
        v = FOCAL_PX * in_camera[1] / in_camera[2] + FRAME_H / 2.0
        if not (0.0 <= u < FRAME_W and 0.0 <= v < FRAME_H):
            continue
        geo = origin.to_geo(NEDPosition(north, east, 0.0))
        col, row = tile.geo_to_pixel(geo)
        frame_points.append((u, v))
        tile_points.append((col, row))

    frame = np.asarray(frame_points, dtype=np.float32)
    tiles = np.asarray(tile_points, dtype=np.float32)
    if noise_px > 0 and len(frame):
        frame = frame + rng.normal(0.0, noise_px, size=frame.shape).astype(np.float32)
    return frame, tiles


def expected_down(origin: LocalOrigin, height_agl_m: float, terrain_alt_m: float = 0.0) -> float:
    """Down of a camera ``height_agl_m`` above terrain, relative to the origin.

    The tiles are georeferenced against terrain at ``terrain_alt_m`` while the
    NED origin sits at ``origin.alt_m``, so the two are only the same number
    when the origin happens to be on the ground.  Keeping them distinct in the
    test is deliberate: it is exactly the offset that would go unnoticed if the
    fixture used a zero-altitude origin.
    """
    return origin.alt_m - (terrain_alt_m + height_agl_m)


@pytest.fixture
def extrinsics() -> CameraExtrinsics:
    return CameraExtrinsics()


@pytest.fixture
def solver(extrinsics) -> PoseSolver:
    return PoseSolver(
        extrinsics=extrinsics,
        fx=FOCAL_PX,
        fy=FOCAL_PX,
        cx=FRAME_W / 2.0,
        cy=FRAME_H / 2.0,
    )


# ---------------------------------------------------------------------------
# The pose solve
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "north,east,height,yaw_deg",
    [
        (0.0, 0.0, 100.0, 0.0),
        (50.0, -30.0, 120.0, 45.0),
        (-80.0, 60.0, 80.0, -110.0),
        (120.0, 120.0, 150.0, 175.0),
    ],
)
def test_pose_solve_recovers_the_true_camera_pose(
    tile, origin, solver, extrinsics, north, east, height, yaw_deg
):
    yaw = math.radians(yaw_deg)
    frame_points, tile_points = synthesise_view(
        tile, origin, north, east, height, yaw, extrinsics
    )
    assert len(frame_points) > 50, "synthetic view produced too few points"

    geometry = GeometricVerifier().verify(
        frame_points, tile_points, frame_size=(FRAME_W, FRAME_H)
    )
    assert geometry.ok, geometry.reason

    pose = solver.solve(geometry, tile, origin, frame_size=(FRAME_W, FRAME_H))

    assert pose.position_ned.north_m == pytest.approx(north, abs=2.0)
    assert pose.position_ned.east_m == pytest.approx(east, abs=2.0)
    # Down is measured from the *origin*, which sits origin.alt_m above the
    # mapped terrain - not from the terrain the height is measured against.
    assert pose.position_ned.down_m == pytest.approx(expected_down(origin, height), abs=4.0)
    assert pose.height_agl_m == pytest.approx(height, abs=4.0)
    assert wrap_pi(pose.yaw_rad - yaw) == pytest.approx(0.0, abs=math.radians(2.0))


def test_pose_solve_reports_the_geographic_position(tile, origin, solver, extrinsics):
    frame_points, tile_points = synthesise_view(
        tile, origin, 100.0, -50.0, 110.0, 0.3, extrinsics
    )
    geometry = GeometricVerifier().verify(
        frame_points, tile_points, frame_size=(FRAME_W, FRAME_H)
    )
    pose = solver.solve(geometry, tile, origin, frame_size=(FRAME_W, FRAME_H))

    expected = origin.to_geo(NEDPosition(100.0, -50.0, -110.0))
    assert pose.geo.lat_deg == pytest.approx(expected.lat_deg, abs=3e-5)
    assert pose.geo.lon_deg == pytest.approx(expected.lon_deg, abs=3e-5)


def test_the_attitude_quaternion_matches_the_recovered_yaw(tile, origin, solver, extrinsics):
    yaw = math.radians(70.0)
    frame_points, tile_points = synthesise_view(tile, origin, 0.0, 0.0, 100.0, yaw, extrinsics)
    geometry = GeometricVerifier().verify(
        frame_points, tile_points, frame_size=(FRAME_W, FRAME_H)
    )
    pose = solver.solve(geometry, tile, origin, frame_size=(FRAME_W, FRAME_H))

    _, _, quaternion_yaw = quaternion_to_euler(pose.attitude.as_tuple())
    assert wrap_pi(quaternion_yaw - pose.yaw_rad) == pytest.approx(0.0, abs=1e-6)


def test_a_rotated_camera_mount_is_compensated_for(tile, origin):
    """A camera bolted on 90 degrees off must still report the vehicle's yaw."""
    extrinsics = CameraExtrinsics(yaw_deg=90.0)
    solver = PoseSolver(
        extrinsics=extrinsics, fx=FOCAL_PX, fy=FOCAL_PX, cx=FRAME_W / 2.0, cy=FRAME_H / 2.0
    )
    vehicle_yaw = math.radians(30.0)
    frame_points, tile_points = synthesise_view(
        tile, origin, 0.0, 0.0, 100.0, vehicle_yaw, extrinsics
    )
    geometry = GeometricVerifier().verify(
        frame_points, tile_points, frame_size=(FRAME_W, FRAME_H)
    )
    pose = solver.solve(geometry, tile, origin, frame_size=(FRAME_W, FRAME_H))

    assert wrap_pi(pose.yaw_rad - vehicle_yaw) == pytest.approx(0.0, abs=math.radians(2.0))


def test_the_camera_lever_arm_shifts_the_reported_body_origin(tile, origin):
    """A camera 2 m forward must put the *body* origin 2 m behind the camera."""
    centred = CameraExtrinsics()
    offset = CameraExtrinsics(translation_m=(2.0, 0.0, 0.0))
    frame_points, tile_points = synthesise_view(tile, origin, 0.0, 0.0, 100.0, 0.0, centred)
    geometry = GeometricVerifier().verify(
        frame_points, tile_points, frame_size=(FRAME_W, FRAME_H)
    )

    common = dict(fx=FOCAL_PX, fy=FOCAL_PX, cx=FRAME_W / 2.0, cy=FRAME_H / 2.0)
    a = PoseSolver(extrinsics=centred, **common).solve(
        geometry, tile, origin, frame_size=(FRAME_W, FRAME_H)
    )
    b = PoseSolver(extrinsics=offset, **common).solve(
        geometry, tile, origin, frame_size=(FRAME_W, FRAME_H)
    )
    # Yaw is ~0, so "forward" is north.
    assert b.position_ned.north_m == pytest.approx(a.position_ned.north_m - 2.0, abs=0.1)


def test_height_is_recovered_from_the_image_scale(tile, origin, solver, extrinsics):
    """Doubling the altitude must halve the ground sampling and be detected."""
    heights = []
    for height in (80.0, 160.0):
        frame_points, tile_points = synthesise_view(
            tile, origin, 0.0, 0.0, height, 0.0, extrinsics
        )
        geometry = GeometricVerifier().verify(
            frame_points, tile_points, frame_size=(FRAME_W, FRAME_H)
        )
        pose = solver.solve(geometry, tile, origin, frame_size=(FRAME_W, FRAME_H))
        heights.append(pose.height_agl_m)
    assert heights[0] == pytest.approx(80.0, abs=3.0)
    assert heights[1] == pytest.approx(160.0, abs=6.0)


def test_a_height_hint_substitutes_for_missing_intrinsics(tile, origin, extrinsics):
    """Without calibrated intrinsics an external height makes z observable."""
    solver = PoseSolver(extrinsics=extrinsics, cx=FRAME_W / 2.0, cy=FRAME_H / 2.0)
    frame_points, tile_points = synthesise_view(tile, origin, 0.0, 0.0, 100.0, 0.0, extrinsics)
    geometry = GeometricVerifier().verify(
        frame_points, tile_points, frame_size=(FRAME_W, FRAME_H)
    )

    pose = solver.solve(
        geometry, tile, origin, frame_size=(FRAME_W, FRAME_H), height_hint_m=100.0
    )
    assert pose.height_source == "hint"
    assert pose.position_ned.down_m == pytest.approx(expected_down(origin, 100.0), abs=0.5)


def test_no_height_source_is_an_explicit_failure(tile, origin, extrinsics):
    solver = PoseSolver(extrinsics=extrinsics)
    frame_points, tile_points = synthesise_view(tile, origin, 0.0, 0.0, 100.0, 0.0, extrinsics)
    geometry = GeometricVerifier().verify(
        frame_points, tile_points, frame_size=(FRAME_W, FRAME_H)
    )
    with pytest.raises(PoseError, match="no height source"):
        solver.solve(geometry, tile, origin, frame_size=(FRAME_W, FRAME_H))


def test_an_unverified_geometry_is_refused(tile, origin, solver):
    from avinavi.features.geometric import GeometricResult

    with pytest.raises(PoseError):
        solver.solve(
            GeometricResult(ok=False, reason="too_few_inliers"),
            tile,
            origin,
            frame_size=(FRAME_W, FRAME_H),
        )


# ---------------------------------------------------------------------------
# The full engine
# ---------------------------------------------------------------------------
def build_engine(tmp_path, tile: MapTile, frame_points, tile_points, matcher=None):
    """A LocalizationEngine over a one-tile database with scripted features."""
    root = str(tmp_path / "map_database")
    directory = os.path.join(root, tile.tile_id)
    os.makedirs(directory, exist_ok=True)
    tile.metadata.directory = directory
    tile.metadata.save()
    TileFeatures(
        keypoints=np.asarray(tile_points, dtype=np.float32),
        descriptors=np.eye(len(tile_points), 8, dtype=np.float32),
        image_size=(float(TILE_PX), float(TILE_PX)),
    ).save(os.path.join(directory, "features.npz"))

    config = Config()
    config.map.database_path = root
    config.camera.fx = FOCAL_PX
    config.camera.fy = FOCAL_PX
    config.camera.cx = FRAME_W / 2.0
    config.camera.cy = FRAME_H / 2.0

    database = MapDatabase.load(root)
    return LocalizationEngine(
        config,
        database,
        extractor=ScriptedExtractor(frame_points),
        matcher=matcher or ScriptedMatcher(),
    )


def test_the_engine_produces_a_valid_georeferenced_fix(tmp_path, tile, origin, extrinsics):
    frame_points, tile_points = synthesise_view(
        tile, origin, 60.0, -40.0, 110.0, math.radians(25.0), extrinsics
    )
    engine = build_engine(tmp_path, tile, frame_points, tile_points)
    engine.initialize(origin)

    fix = engine.localize(np.zeros((FRAME_H, FRAME_W), dtype=np.uint8))

    assert fix.valid, fix.reason
    assert fix.tile_id == "tile_center"
    assert fix.position_ned.north_m == pytest.approx(60.0, abs=3.0)
    assert fix.position_ned.east_m == pytest.approx(-40.0, abs=3.0)
    assert fix.confidence > 0.5
    assert fix.quality.number_of_inliers > 50


def test_the_search_window_recentres_on_the_visual_fix(tmp_path, tile, origin, extrinsics):
    """After a fix, the window must follow the *visual* position, not the seed."""
    frame_points, tile_points = synthesise_view(tile, origin, 100.0, 100.0, 110.0, 0.0, extrinsics)
    engine = build_engine(tmp_path, tile, frame_points, tile_points)
    engine.initialize(origin)
    assert engine.search_radius_m == engine.config.tile_search.initial_search_radius_m

    fix = engine.localize(np.zeros((FRAME_H, FRAME_W), dtype=np.uint8))
    assert fix.valid

    assert engine.search_center.lat_deg == pytest.approx(fix.geo.lat_deg)
    assert engine.search_center.lon_deg == pytest.approx(fix.geo.lon_deg)
    assert engine.search_radius_m == engine.config.tile_search.search_radius_m


def test_the_search_window_grows_after_repeated_failures(tmp_path, tile, origin, extrinsics):
    frame_points, tile_points = synthesise_view(tile, origin, 0.0, 0.0, 110.0, 0.0, extrinsics)
    engine = build_engine(
        tmp_path, tile, frame_points, tile_points, matcher=NoMatchMatcher()
    )
    engine.initialize(origin)

    initial = engine.search_radius_m
    for _ in range(4):
        fix = engine.localize(np.zeros((FRAME_H, FRAME_W), dtype=np.uint8))
        assert not fix.valid
    assert engine.search_radius_m > initial
    assert engine.consecutive_failures == 4


def test_the_growth_stops_at_the_configured_ceiling(tmp_path, tile, origin, extrinsics):
    frame_points, tile_points = synthesise_view(tile, origin, 0.0, 0.0, 110.0, 0.0, extrinsics)
    engine = build_engine(
        tmp_path, tile, frame_points, tile_points, matcher=NoMatchMatcher()
    )
    engine.initialize(origin)
    for _ in range(40):
        engine.localize(np.zeros((FRAME_H, FRAME_W), dtype=np.uint8))

    assert engine.search_radius_m == engine.config.tile_search.max_search_radius_m
    assert engine.search_exhausted


def test_no_matches_gives_an_invalid_fix_not_a_guess(tmp_path, tile, origin, extrinsics):
    frame_points, tile_points = synthesise_view(tile, origin, 0.0, 0.0, 110.0, 0.0, extrinsics)
    engine = build_engine(
        tmp_path, tile, frame_points, tile_points, matcher=NoMatchMatcher()
    )
    engine.initialize(origin)

    fix = engine.localize(np.zeros((FRAME_H, FRAME_W), dtype=np.uint8))
    assert not fix.valid
    assert fix.reason == InvalidReason.TOO_FEW_MATCHES
    assert fix.geo is None
    assert fix.position_ned is None


def test_an_uninitialised_engine_refuses_to_localize(tmp_path, tile, origin, extrinsics):
    frame_points, tile_points = synthesise_view(tile, origin, 0.0, 0.0, 110.0, 0.0, extrinsics)
    engine = build_engine(tmp_path, tile, frame_points, tile_points)

    fix = engine.localize(np.zeros((FRAME_H, FRAME_W), dtype=np.uint8))
    assert not fix.valid
    assert fix.reason == InvalidReason.NOT_INITIALIZED


def test_no_tiles_in_range_is_reported_as_a_coverage_failure(
    tmp_path, tile, origin, extrinsics
):
    frame_points, tile_points = synthesise_view(tile, origin, 0.0, 0.0, 110.0, 0.0, extrinsics)
    engine = build_engine(tmp_path, tile, frame_points, tile_points)
    # Initialise a long way from the only tile in the database.
    engine.initialize(origin, GeoPoint(origin.lat_deg + 1.0, origin.lon_deg + 1.0))

    fix = engine.localize(np.zeros((FRAME_H, FRAME_W), dtype=np.uint8))
    assert not fix.valid
    assert fix.reason == InvalidReason.NO_CANDIDATE_TILES


def test_too_few_keypoints_short_circuits_the_cycle(tmp_path, tile, origin, extrinsics):
    frame_points, tile_points = synthesise_view(tile, origin, 0.0, 0.0, 110.0, 0.0, extrinsics)
    engine = build_engine(tmp_path, tile, frame_points, tile_points)
    engine.extractor = ScriptedExtractor(frame_points[:10])
    engine.initialize(origin)

    fix = engine.localize(np.zeros((FRAME_H, FRAME_W), dtype=np.uint8))
    assert not fix.valid
    assert fix.reason == InvalidReason.TOO_FEW_KEYPOINTS


def test_an_empty_frame_is_reported_rather_than_localized(tmp_path, tile, origin, extrinsics):
    frame_points, tile_points = synthesise_view(tile, origin, 0.0, 0.0, 110.0, 0.0, extrinsics)
    engine = build_engine(tmp_path, tile, frame_points, tile_points)
    engine.initialize(origin)

    fix = engine.localize(np.zeros((0, 0), dtype=np.uint8))
    assert not fix.valid
    assert fix.reason == InvalidReason.NO_FRAME


# ---------------------------------------------------------------------------
# All the way to the wire
# ---------------------------------------------------------------------------
def test_a_fix_becomes_a_correctly_framed_odometry_message(
    tmp_path, tile, origin, extrinsics
):
    """The complete chain: synthetic view -> fix -> propagator -> ODOMETRY bytes."""
    north, east, height = 75.0, -25.0, 120.0
    frame_points, tile_points = synthesise_view(
        tile, origin, north, east, height, math.radians(15.0), extrinsics
    )
    engine = build_engine(tmp_path, tile, frame_points, tile_points)
    engine.initialize(origin)

    fix = engine.localize(np.zeros((FRAME_H, FRAME_W), dtype=np.uint8))
    assert fix.valid

    propagator = PosePropagator()
    propagator.update(fix)
    solution = propagator.solution_at(fix.timestamp_monotonic)
    assert solution is not None

    connection = FakeConnection(connected=True)
    publisher = ExternalNavigationPublisher(
        connection, message="odometry", send_to_autopilot=True
    )
    assert publisher.publish(solution)

    message = connection.last("ODOMETRY")
    assert message.frame_id == 20  # MAV_FRAME_LOCAL_FRD
    assert message.child_frame_id == 12  # MAV_FRAME_BODY_FRD
    assert message.x == pytest.approx(north, abs=3.0)
    assert message.y == pytest.approx(east, abs=3.0)
    assert message.z == pytest.approx(expected_down(origin, height), abs=5.0)
    # ArduPilot reads posErr from exactly these three indices.
    position_error = math.sqrt(
        message.pose_covariance[0] + message.pose_covariance[6] + message.pose_covariance[11]
    )
    assert 0.0 < position_error < 40.0
    assert 1 <= message.quality <= 100
