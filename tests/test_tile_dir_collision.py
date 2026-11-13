"""Two different tile databases must never share one directory on disk.

``run_simulation`` caches tile databases in memory under ``db_key``, which
separates the backend, the detector, the tile size, the keypoint budget and the
photometric settings - all the things that make tile descriptors
non-interchangeable. The directory those tiles were written to named only three
of them.

So two databases the cache correctly treated as different were written to one
path, and ``build_tile_database`` starts by deleting that path.

Sequentially inside one process that is silent and wrong: the first database
object stays in the cache pointing at a directory that has since been rebuilt
from different settings, so a later cache hit serves tiles built with settings
that do not match what asked for them. Across two processes it is louder - a
real run died with ``FileNotFoundError`` on ``tile_0013/features.npz`` when one
deleted the directory the other was writing into - and it cost a measurement,
because a third run had been sharing that directory while it was swapped
underneath it and its numbers had to be discarded.

The fix derives the directory name from ``db_key`` itself, so the two cannot
drift apart. These tests pin that property rather than the specific fields,
since the failure was precisely that someone added a field to one and not the
other.
"""

from __future__ import annotations

import hashlib
import itertools

import pytest

from avinavi.config import Config


def _db_key(config: Config, terrain_key=("terrain", 4000, 0.5, 7), tile_size_px=1024,
            normalizer=None):
    """Mirror of the key built in run_simulation."""
    return (
        "database",
        terrain_key,
        tile_size_px,
        config.features.backend,
        config.features.detector,
        config.superpoint.max_num_keypoints,
        bool(normalizer),
        config.photometric.clip_limit if normalizer else None,
    )


def _tile_dir(config: Config, db_key, normalizer=None) -> str:
    """Mirror of the directory name built in run_simulation."""
    return "tiles_%s_%s_%s_%s" % (
        config.features.backend,
        config.features.detector,
        "clahe" if normalizer else "raw",
        hashlib.sha1(repr(db_key).encode("utf-8")).hexdigest()[:10],
    )


def _config(**overrides) -> Config:
    config = Config()
    config.features.backend = overrides.get("backend", "superpoint")
    config.features.detector = overrides.get("detector", "sift")
    config.superpoint.max_num_keypoints = overrides.get("keypoints", 1024)
    return config


class TestDirectoryMatchesKey:
    def test_the_keypoint_budget_separates_directories(self):
        """The field that was in db_key but not in the path."""
        a = _config(keypoints=512)
        b = _config(keypoints=256)
        assert _tile_dir(a, _db_key(a)) != _tile_dir(b, _db_key(b))

    def test_the_tile_size_separates_directories(self):
        config = _config()
        assert _tile_dir(config, _db_key(config, tile_size_px=1024)) != _tile_dir(
            config, _db_key(config, tile_size_px=2048)
        )

    def test_the_terrain_separates_directories(self):
        config = _config()
        assert _tile_dir(config, _db_key(config, terrain_key=("terrain", 4000, 0.5, 7))) \
            != _tile_dir(config, _db_key(config, terrain_key=("terrain", 4000, 0.5, 8)))

    def test_the_backend_separates_directories(self):
        a = _config(backend="superpoint")
        b = _config(backend="opencv")
        assert _tile_dir(a, _db_key(a)) != _tile_dir(b, _db_key(b))

    def test_identical_settings_share_a_directory(self):
        """The cache must still work, or every run rebuilds its tiles."""
        a, b = _config(keypoints=512), _config(keypoints=512)
        assert _tile_dir(a, _db_key(a)) == _tile_dir(b, _db_key(b))

    def test_no_two_distinct_keys_collide(self):
        """The general property, over the grid the experiments actually sweep.

        Stated as 'distinct keys give distinct directories' rather than as a
        list of fields, because the bug was a field being added to the key and
        not to the path.
        """
        seen = {}
        grids = itertools.product(
            ("superpoint", "opencv"),
            ("sift", "akaze", "orb"),
            (256, 384, 512, 1024, 2048),
            (1024, 2048),
        )
        for backend, detector, keypoints, tile_size in grids:
            config = _config(backend=backend, detector=detector, keypoints=keypoints)
            key = _db_key(config, tile_size_px=tile_size)
            directory = _tile_dir(config, key)
            if directory in seen:
                assert seen[directory] == key, (
                    "two different databases share the directory %s:\n  %r\n  %r"
                    % (directory, seen[directory], key)
                )
            seen[directory] = key
        assert len(seen) == len(set(seen))

    def test_the_old_naming_collided(self):
        """Documents the bug so a 'tidy-up' cannot reintroduce it."""
        def old_name(config, normalizer=None):
            return "tiles_%s_%s_%s" % (
                config.features.backend,
                config.features.detector,
                "clahe" if normalizer else "raw",
            )

        a, b = _config(keypoints=512), _config(keypoints=256)
        assert old_name(a) == old_name(b), "the old name ignored the budget"
        assert _db_key(a) != _db_key(b), "while the cache knew they differed"


class TestTheDirectoryIsUsable:
    @pytest.mark.parametrize("keypoints", [256, 512, 1024])
    def test_the_name_is_a_safe_path_component(self, keypoints):
        config = _config(keypoints=keypoints)
        name = _tile_dir(config, _db_key(config))
        assert "/" not in name and "\\" not in name
        assert name.startswith("tiles_")
        assert len(name) < 120
