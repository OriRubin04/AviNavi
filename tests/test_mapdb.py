"""The tile database and the moving search window."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from avinavi.geo import BoundingBox, LocalOrigin, haversine_distance_m
from avinavi.mapdb.database import MapCoverageError, MapDatabase
from avinavi.mapdb.tile import MapTile, TileFeatures, TileFormatError, TileMetadata
from avinavi.types import GeoPoint, NEDPosition


def write_tile(root: str, tile_id: str, bounds: BoundingBox, keypoints: int = 32) -> str:
    directory = os.path.join(root, tile_id)
    os.makedirs(directory, exist_ok=True)
    TileMetadata(
        tile_id=tile_id,
        directory=directory,
        width_px=512,
        height_px=512,
        bounds=bounds,
        terrain_alt_m=25.0,
    ).save()
    TileFeatures(
        keypoints=np.random.rand(keypoints, 2).astype(np.float32) * 512.0,
        descriptors=np.random.rand(keypoints, 16).astype(np.float32),
        image_size=(512.0, 512.0),
    ).save(os.path.join(directory, "features.npz"))
    return directory


@pytest.fixture
def grid_database(tmp_path):
    """A 3x3 grid of ~200 m tiles centred near the fixture origin."""
    root = str(tmp_path / "map_database")
    os.makedirs(root, exist_ok=True)
    origin = LocalOrigin(32.0853, 34.7818, 40.0)
    step = 200.0
    for row in range(3):
        for col in range(3):
            south_west = origin.to_geo(
                NEDPosition((row - 1.5) * step, (col - 1.5) * step, 0.0)
            )
            north_east = origin.to_geo(
                NEDPosition((row - 0.5) * step, (col - 0.5) * step, 0.0)
            )
            write_tile(
                root,
                "tile_%d%d" % (row, col),
                BoundingBox(
                    south_west.lon_deg, south_west.lat_deg, north_east.lon_deg, north_east.lat_deg
                ),
            )
    return root, origin


def test_loading_finds_every_tile(grid_database):
    root, _ = grid_database
    database = MapDatabase.load(root)
    assert len(database) == 9


def test_an_empty_directory_is_a_coverage_error(tmp_path):
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    with pytest.raises(MapCoverageError):
        MapDatabase.load(empty)


def test_a_missing_directory_is_a_coverage_error(tmp_path):
    with pytest.raises(MapCoverageError):
        MapDatabase.load(str(tmp_path / "nope"))


def test_coverage_is_the_union_of_the_tiles(grid_database):
    root, origin = grid_database
    database = MapDatabase.load(root)
    coverage = database.coverage
    assert coverage.contains(origin.geo)
    # 3 tiles of 200 m gives a 600 m span; allow for the tangent-plane rounding.
    assert coverage.distance_to_m(origin.to_geo(NEDPosition(500.0, 0.0, 0.0))) > 0


def test_a_small_radius_finds_only_nearby_tiles(grid_database):
    root, origin = grid_database
    database = MapDatabase.load(root)
    near = database.candidates(origin.geo, radius_m=10.0)
    far = database.candidates(origin.geo, radius_m=1000.0)
    assert 0 < len(near) < len(far) == 9


def test_candidates_are_ordered_nearest_first(grid_database):
    root, origin = grid_database
    database = MapDatabase.load(root)
    candidates = database.candidates(origin.geo, radius_m=1000.0)
    distances = [t.bounds.distance_to_m(origin.geo) for t in candidates]
    assert distances == sorted(distances)


def test_the_limit_truncates_the_least_useful_tiles(grid_database):
    root, origin = grid_database
    database = MapDatabase.load(root)
    limited = database.candidates(origin.geo, radius_m=1000.0, limit=3)
    assert len(limited) == 3
    # The nearest tile must survive truncation.
    nearest = database.candidates(origin.geo, radius_m=1000.0)[0]
    assert limited[0].tile_id == nearest.tile_id


def test_preferred_tiles_are_promoted(grid_database):
    root, origin = grid_database
    database = MapDatabase.load(root)
    all_tiles = database.candidates(origin.geo, radius_m=1000.0)
    distant = all_tiles[-1].tile_id

    promoted = database.candidates(origin.geo, radius_m=1000.0, prefer=[distant])
    assert promoted[0].tile_id == distant
    assert len(promoted) == len(all_tiles)


def test_an_unknown_preferred_tile_is_ignored(grid_database):
    root, origin = grid_database
    database = MapDatabase.load(root)
    candidates = database.candidates(origin.geo, radius_m=1000.0, prefer=["nonexistent"])
    assert len(candidates) == 9


def test_the_search_window_moves_with_the_position(grid_database):
    """A window centred elsewhere must return a different tile set."""
    root, origin = grid_database
    database = MapDatabase.load(root)
    here = {t.tile_id for t in database.candidates(origin.geo, radius_m=20.0)}
    # The grid spans +/- 300 m; 250 m north-east lands squarely in the corner tile.
    over_there = {
        t.tile_id
        for t in database.candidates(
            origin.to_geo(NEDPosition(250.0, 250.0, 0.0)), radius_m=20.0
        )
    }
    assert here and over_there and here != over_there


def test_a_radius_beyond_coverage_finds_nothing(grid_database):
    root, origin = grid_database
    database = MapDatabase.load(root)
    far_away = GeoPoint(origin.lat_deg + 1.0, origin.lon_deg + 1.0)
    assert database.candidates(far_away, radius_m=300.0) == []


def test_a_non_positive_radius_is_rejected(grid_database):
    root, origin = grid_database
    database = MapDatabase.load(root)
    with pytest.raises(ValueError):
        database.candidates(origin.geo, radius_m=0.0)


def test_the_feature_cache_evicts_the_least_recently_used(grid_database):
    root, _ = grid_database
    database = MapDatabase.load(root, feature_cache_size=2)
    tiles = database.tiles[:3]
    for tile in tiles:
        database.features_for(tile)
    loaded = [t.tile_id for t in database.tiles if t.features_loaded]
    assert len(loaded) == 2
    assert tiles[0].tile_id not in loaded


def test_reusing_a_cached_tile_refreshes_its_position(grid_database):
    root, _ = grid_database
    database = MapDatabase.load(root, feature_cache_size=2)
    a, b, c = database.tiles[:3]
    database.features_for(a)
    database.features_for(b)
    database.features_for(a)  # touch a again
    database.features_for(c)  # should evict b, not a
    assert a.features_loaded
    assert not b.features_loaded


def test_an_index_file_avoids_the_directory_scan(grid_database):
    root, _ = grid_database
    MapDatabase.load(root).write_index()
    assert os.path.exists(os.path.join(root, "index.json"))
    assert len(MapDatabase.load(root)) == 9


def test_a_corrupt_index_falls_back_to_scanning(grid_database):
    root, _ = grid_database
    with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as handle:
        handle.write("{not json")
    assert len(MapDatabase.load(root)) == 9


def test_overlapping_tiles_are_ordered_by_how_deeply_they_contain_the_point():
    """The ordering fix that a measured sweep pointed at.

    Tiles overlap, so the aircraft is normally inside several at once and every
    one of them has a footprint distance of exactly zero. Ordering on the
    footprint alone left them in alphabetical order, and the sweep showed the
    cost: cutting max_candidate_tiles from 4 to 2 halved the lock rate while 2
    to 1 cost nothing — the signature of the right tile landing third or fourth
    as often as first.

    A frame near a tile's centre has its whole camera footprint inside that
    tile; the same frame near the edge is half outside it. So the tile the
    aircraft is most deeply inside must come first, whatever it is called.
    """
    import tempfile

    root = tempfile.mkdtemp()
    origin = LocalOrigin(32.0853, 34.7818, 40.0)

    # Three heavily overlapping 400 m tiles, named so that alphabetical order
    # is the *opposite* of the correct order for a point near "tile_c".
    for name, centre_north in (("tile_a", -150.0), ("tile_b", 0.0), ("tile_c", 150.0)):
        south_west = origin.to_geo(NEDPosition(centre_north - 200.0, -200.0, 0.0))
        north_east = origin.to_geo(NEDPosition(centre_north + 200.0, 200.0, 0.0))
        write_tile(
            root,
            name,
            BoundingBox(
                south_west.lon_deg, south_west.lat_deg,
                north_east.lon_deg, north_east.lat_deg,
            ),
        )

    database = MapDatabase.load(root)
    # 180 m north is inside tile_b (-200..200) and tile_c (-50..350), but it is
    # 30 m from tile_c's centre and 180 m from tile_b's.
    probe = origin.to_geo(NEDPosition(180.0, 0.0, 0.0))
    candidates = database.candidates(center=probe, radius_m=50.0)

    assert len(candidates) >= 2, "the probe should be inside more than one tile"
    assert candidates[0].tile_id == "tile_c", (
        "expected the most deeply containing tile first, got %s"
        % [t.tile_id for t in candidates]
    )


def test_an_equidistant_tie_is_broken_deterministically():
    """Two runs must order the same tiles the same way."""
    import tempfile

    root = tempfile.mkdtemp()
    origin = LocalOrigin(32.0853, 34.7818, 40.0)
    for name, east in (("tile_z", -100.0), ("tile_a", 100.0)):
        south_west = origin.to_geo(NEDPosition(-200.0, east - 200.0, 0.0))
        north_east = origin.to_geo(NEDPosition(200.0, east + 200.0, 0.0))
        write_tile(
            root, name,
            BoundingBox(
                south_west.lon_deg, south_west.lat_deg,
                north_east.lon_deg, north_east.lat_deg,
            ),
        )
    database = MapDatabase.load(root)
    first = [t.tile_id for t in database.candidates(origin.geo, radius_m=50.0)]
    second = [t.tile_id for t in database.candidates(origin.geo, radius_m=50.0)]
    assert first == second


def test_an_index_left_over_from_an_earlier_build_is_not_trusted(grid_database):
    """A stale index must never override the tiles' own georeferencing.

    This is a regression test for a bug with the worst possible failure shape.
    Rebuilding a database over an existing directory rewrote every tile's
    metadata but left ``index.json`` describing the previous build; because the
    index was trusted unconditionally, the system reported positions against
    the *old* georeferencing. Imagery, features and matching were all correct,
    and every fix was offset by a constant 100 m with nothing reporting it -
    precisely the confident, plausible, consistently wrong position the EKF
    cannot reject.
    """
    root, origin = grid_database
    MapDatabase.load(root).write_index()

    # Rebuild the same tile ids 100 m further north, as a re-tiling would.
    shifted = origin.to_geo(NEDPosition(100.0, 0.0, 0.0))
    for row in range(3):
        for col in range(3):
            south_west = origin.to_geo(
                NEDPosition((row - 1.5) * 200.0 + 100.0, (col - 1.5) * 200.0, 0.0)
            )
            north_east = origin.to_geo(
                NEDPosition((row - 0.5) * 200.0 + 100.0, (col - 0.5) * 200.0, 0.0)
            )
            write_tile(
                root,
                "tile_%d%d" % (row, col),
                BoundingBox(
                    south_west.lon_deg,
                    south_west.lat_deg,
                    north_east.lon_deg,
                    north_east.lat_deg,
                ),
            )
    # The index on disk still describes the pre-shift tiling.
    os.utime(os.path.join(root, "index.json"), (0, 0))

    database = MapDatabase.load(root)
    assert database.tile_containing(shifted) is not None

    # The decisive check: the coverage must have moved north with the rebuild.
    # The old tiling reached 300 m north of the origin, the new one 400 m, so
    # a database that still reports 300 m is reading the stale index.
    north_edge_m = haversine_distance_m(
        origin.geo, GeoPoint(database.coverage.north, origin.lon_deg, 0.0)
    )
    assert north_edge_m == pytest.approx(400.0, abs=2.0), (
        "the stale index was trusted: coverage reaches %.0f m north, which is "
        "the previous build's georeferencing" % north_edge_m
    )


def test_an_index_that_disagrees_with_the_tiles_on_disk_is_rescanned(grid_database):
    """A changed tile *set* is staleness too, even with fresh timestamps.

    Copying a database between machines can preserve mtimes, so the timestamp
    check alone is not enough; the tile set itself has to agree.
    """
    root, origin = grid_database
    MapDatabase.load(root).write_index()

    extra_sw = origin.to_geo(NEDPosition(400.0, 400.0, 0.0))
    extra_ne = origin.to_geo(NEDPosition(600.0, 600.0, 0.0))
    write_tile(
        root,
        "tile_extra",
        BoundingBox(
            extra_sw.lon_deg, extra_sw.lat_deg, extra_ne.lon_deg, extra_ne.lat_deg
        ),
    )
    # Make every timestamp agree, so only the set mismatch can reveal the index
    # is out of date.
    for entry in os.listdir(root):
        path = os.path.join(root, entry)
        target = (
            os.path.join(path, "metadata.json") if os.path.isdir(path) else path
        )
        if os.path.exists(target):
            os.utime(target, (1_000_000, 1_000_000))
    os.utime(os.path.join(root, "index.json"), (1_000_000, 1_000_000))

    assert len(MapDatabase.load(root)) == 10


def test_a_tile_with_broken_metadata_is_skipped(grid_database):
    root, _ = grid_database
    broken = os.path.join(root, "tile_broken")
    os.makedirs(broken)
    with open(os.path.join(broken, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump({"tile_id": "tile_broken"}, handle)
    assert len(MapDatabase.load(root)) == 9


# ---------------------------------------------------------------------------
# Tiles and features
# ---------------------------------------------------------------------------
def test_features_survive_a_save_load_round_trip(tmp_path):
    path = str(tmp_path / "features.npz")
    original = TileFeatures(
        keypoints=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        descriptors=np.random.rand(2, 8).astype(np.float32),
        scores=np.array([0.5, 0.6], dtype=np.float32),
        image_size=(640.0, 480.0),
    )
    original.save(path)

    loaded = TileFeatures.load(path)
    assert np.allclose(loaded.keypoints, original.keypoints)
    assert np.allclose(loaded.descriptors, original.descriptors)
    assert loaded.image_size == (640.0, 480.0)


def test_a_pt_path_falls_back_to_npz_when_torch_is_absent(tmp_path):
    """Tile features written without torch must still be findable as features.pt."""
    directory = str(tmp_path / "tile")
    os.makedirs(directory)
    TileFeatures(
        keypoints=np.zeros((4, 2), dtype=np.float32),
        descriptors=np.zeros((4, 8), dtype=np.float32),
    ).save(os.path.join(directory, "features.npz"))

    loaded = TileFeatures.load(os.path.join(directory, "features.pt"))
    assert len(loaded) == 4


def test_mismatched_keypoint_and_descriptor_counts_are_rejected():
    with pytest.raises(TileFormatError):
        TileFeatures(
            keypoints=np.zeros((4, 2), dtype=np.float32),
            descriptors=np.zeros((3, 8), dtype=np.float32),
        )


def test_missing_features_raise_a_clear_error(tmp_path):
    with pytest.raises(TileFormatError, match="missing tile features"):
        TileFeatures.load(str(tmp_path / "absent.pt"))


def test_metadata_round_trips_through_json(tmp_path, tile_bounds):
    directory = str(tmp_path / "tile_x")
    metadata = TileMetadata(
        tile_id="tile_x",
        directory=directory,
        width_px=1024,
        height_px=768,
        bounds=tile_bounds,
        terrain_alt_m=17.5,
        extra={"source": "orthophoto.tif"},
    )
    metadata.save()

    loaded = TileMetadata.load(directory)
    assert loaded.tile_id == "tile_x"
    assert loaded.width_px == 1024
    assert loaded.bounds.as_tuple() == pytest.approx(tile_bounds.as_tuple())
    assert loaded.terrain_alt_m == 17.5
    assert loaded.extra["source"] == "orthophoto.tif"


def test_metadata_accepts_a_bounds_list(tmp_path, tile_bounds):
    directory = str(tmp_path / "tile_y")
    os.makedirs(directory)
    with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {"width_px": 256, "height_px": 256, "bounds": list(tile_bounds.as_tuple())},
            handle,
        )
    loaded = TileMetadata.load(directory)
    assert loaded.bounds.as_tuple() == pytest.approx(tile_bounds.as_tuple())
    assert loaded.tile_id == "tile_y"


def test_ground_resolution_is_derived_when_absent(tile_bounds):
    tile = MapTile(
        TileMetadata(
            tile_id="t", directory="/tmp/t", width_px=1000, height_px=1000, bounds=tile_bounds
        )
    )
    assert 0.5 < tile.ground_resolution_mpp < 1.5
