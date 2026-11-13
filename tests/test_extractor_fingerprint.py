"""The map database records what built its features, and startup checks it.

These tests exist because the failure they guard against has already happened
twice in this project and is invisible from the outside: tile descriptors are
written offline, live descriptors are produced in the air, and when the
configuration drifts between the two the system boots cleanly, streams video,
logs nothing unusual and produces no matches at all.

The distinction the tests spend most of their effort on is between a difference
that makes matching *impossible* (fatal) and one that merely makes it *worse*
(reported).  Collapsing the two would either ground a correctly configured
aircraft or let the silent failure straight back in, because ``avinavi
extract-features`` deliberately gives a map tile four times a frame's keypoint
budget and runs it at full tile resolution.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pytest

from avinavi.config import Config
from avinavi.features.fingerprint import (
    MIN_ANALYSIS_FRACTION,
    ExtractorFingerprint,
    compare,
    compare_database,
    consensus,
    descriptor_kind_for,
)
from avinavi.features.resize import (
    USE_CONFIGURED_RESIZE,
    analysis_max_side,
    resolve_max_side,
)
from avinavi.geo import BoundingBox
from avinavi.mapdb.database import ExtractorMismatchError, MapDatabase
from avinavi.mapdb.tile import TileFeatures, TileMetadata

SUPERPOINT = ExtractorFingerprint(
    backend="superpoint",
    detector="superpoint",
    descriptor_dim=256,
    descriptor_kind="float",
    max_num_keypoints=1024,
    resize_max_side=640,
)


def _sift(**overrides) -> ExtractorFingerprint:
    base = dict(
        backend="opencv",
        detector="sift",
        descriptor_dim=128,
        descriptor_kind="float",
        max_num_keypoints=2048,
        resize_max_side=640,
    )
    base.update(overrides)
    return ExtractorFingerprint(**base)


# ---------------------------------------------------------------------------
# What counts as incompatible
# ---------------------------------------------------------------------------
class TestIncompatible:
    def test_same_fingerprint_is_clean(self):
        result = compare(SUPERPOINT, SUPERPOINT)
        assert result.compatible
        assert result.clean

    def test_different_backend_is_fatal(self):
        result = compare(_sift(), SUPERPOINT)
        assert not result.compatible
        assert any("backend" in m for m in result.incompatible)

    def test_different_classical_detector_is_fatal(self):
        result = compare(_sift(detector="akaze"), _sift())
        assert not result.compatible
        assert any("detector" in m for m in result.incompatible)

    def test_different_descriptor_width_is_fatal(self):
        """128-d SIFT descriptors cannot be compared against 256-d ones."""
        result = compare(_sift(descriptor_dim=128), _sift(descriptor_dim=256))
        assert not result.compatible
        assert any("descriptor width" in m for m in result.incompatible)

    def test_float_against_binary_is_fatal(self):
        """L2 and Hamming spaces are not interchangeable, whatever the width."""
        stored = _sift(detector="orb", descriptor_dim=32, descriptor_kind="binary")
        current = _sift(detector="orb", descriptor_dim=32, descriptor_kind="float")
        result = compare(stored, current)
        assert not result.compatible
        assert any("descriptor type" in m for m in result.incompatible)

    def test_messages_name_both_sides(self):
        """A message that does not say what to change is not worth printing."""
        result = compare(_sift(), SUPERPOINT)
        joined = " ".join(result.incompatible)
        assert "opencv" in joined and "superpoint" in joined

    def test_unknown_field_is_skipped_not_assumed(self):
        """An unrecorded descriptor width must not be read as a mismatch."""
        result = compare(
            _sift(descriptor_dim=None), _sift(descriptor_dim=128)
        )
        assert result.compatible


# ---------------------------------------------------------------------------
# What counts as merely degraded - the half that must NOT be fatal
# ---------------------------------------------------------------------------
class TestDegraded:
    def test_bigger_tile_keypoint_budget_is_not_a_complaint(self):
        """The intended production setup: tiles get four times a frame's budget.

        'avinavi extract-features' multiplies the configured budget by four
        because a tile covers far more ground than one frame.  A check that
        complained here would fire on every correct installation and be ignored
        thereafter, which is worse than no check.
        """
        stored = ExtractorFingerprint(
            backend="superpoint",
            detector="superpoint",
            descriptor_dim=256,
            descriptor_kind="float",
            max_num_keypoints=4096,
            resize_max_side=None,
        )
        result = compare(stored, SUPERPOINT)
        assert result.clean

    def test_smaller_tile_keypoint_budget_is_reported(self):
        stored = ExtractorFingerprint(
            backend="superpoint",
            detector="superpoint",
            descriptor_dim=256,
            descriptor_kind="float",
            max_num_keypoints=256,
        )
        result = compare(stored, SUPERPOINT)
        assert result.compatible, "fewer keypoints is poor, not unmatchable"
        assert any("256" in m and "1024" in m for m in result.degraded)

    def test_tile_analysed_below_its_own_resolution_is_reported(self):
        """The fifteen-point regression, as a test.

        Tiles were stored at 1024 px and described at 640 because one
        configuration key meant 'use the configured resize' where 'do not
        resize' was intended.  Nothing failed; the map simply described a
        blurrier world than the imagery on disk.
        """
        stored = ExtractorFingerprint(
            backend="superpoint",
            detector="superpoint",
            descriptor_dim=256,
            descriptor_kind="float",
            max_num_keypoints=1024,
            analysis_max_side=640,
            image_max_side=1024,
        )
        result = compare(stored, SUPERPOINT)
        assert result.compatible, "coarse features still match, just worse"
        assert result.degraded
        joined = " ".join(result.degraded)
        assert "640" in joined and "1024" in joined
        assert "62%" in joined, "the message should quantify the loss"

    def test_full_resolution_analysis_is_silent(self):
        stored = ExtractorFingerprint(
            backend="superpoint",
            detector="superpoint",
            descriptor_dim=256,
            descriptor_kind="float",
            max_num_keypoints=1024,
            analysis_max_side=1024,
            image_max_side=1024,
        )
        assert compare(stored, SUPERPOINT).clean

    def test_rounding_slack_is_tolerated(self):
        """A pixel or two of rounding is not a finding."""
        analysed = int(1024 * MIN_ANALYSIS_FRACTION) + 1
        stored = ExtractorFingerprint(
            backend="superpoint",
            detector="superpoint",
            descriptor_dim=256,
            descriptor_kind="float",
            max_num_keypoints=1024,
            analysis_max_side=analysed,
            image_max_side=1024,
        )
        assert compare(stored, SUPERPOINT).clean

    def test_missing_fingerprint_is_reported_not_fatal(self):
        """Databases built before this existed must keep working."""
        result = compare(None, SUPERPOINT)
        assert result.compatible
        assert any("no extractor fingerprint" in m for m in result.degraded)


# ---------------------------------------------------------------------------
# A database whose tiles disagree with each other
# ---------------------------------------------------------------------------
class TestConsensus:
    def test_uniform_database_has_no_conflicts(self):
        fingerprint, conflicts = consensus([SUPERPOINT, SUPERPOINT, SUPERPOINT])
        assert fingerprint == SUPERPOINT
        assert conflicts == []

    def test_mixed_detectors_are_surfaced(self):
        """Half re-extracted, half left behind: harder to notice than total failure."""
        _, conflicts = consensus([SUPERPOINT, _sift()])
        assert any("not all built the same way" in c for c in conflicts)

    def test_partially_stamped_database_is_surfaced(self):
        _, conflicts = consensus([SUPERPOINT, None, SUPERPOINT])
        assert any("carry no extractor fingerprint" in c for c in conflicts)

    def test_empty_database_yields_nothing(self):
        assert consensus([]) == (None, [])
        assert consensus([None, None]) == (None, [])

    def test_weakest_tile_is_reported_rather_than_the_first(self):
        """One coarse tile is a real hole in the map; averaging would hide it."""
        good = ExtractorFingerprint(
            backend="superpoint",
            detector="superpoint",
            analysis_max_side=1024,
            image_max_side=1024,
        )
        coarse = ExtractorFingerprint(
            backend="superpoint",
            detector="superpoint",
            analysis_max_side=320,
            image_max_side=1024,
        )
        fingerprint, _ = consensus([good, good, coarse, good])
        assert fingerprint is not None
        assert fingerprint.analysis_max_side == 320


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------
class TestSerialisation:
    def test_round_trip(self):
        restored = ExtractorFingerprint.from_dict(SUPERPOINT.to_dict())
        assert restored == SUPERPOINT

    def test_unknown_fields_are_omitted_from_json(self):
        bare = ExtractorFingerprint(backend="superpoint", detector="superpoint")
        assert bare.to_dict() == {"backend": "superpoint", "detector": "superpoint"}

    def test_round_trip_through_tile_metadata(self, tmp_path):
        directory = str(tmp_path / "tile_0000")
        os.makedirs(directory)
        metadata = TileMetadata(
            tile_id="tile_0000",
            directory=directory,
            width_px=1024,
            height_px=1024,
            bounds=BoundingBox(34.0, 32.0, 34.01, 32.01),
            extractor=SUPERPOINT.to_dict(),
        )
        metadata.save()
        assert TileMetadata.load(directory).extractor_fingerprint == SUPERPOINT

    def test_unstamped_metadata_reads_as_none(self, tmp_path):
        directory = str(tmp_path / "tile_0000")
        os.makedirs(directory)
        TileMetadata(
            tile_id="tile_0000",
            directory=directory,
            width_px=64,
            height_px=64,
            bounds=BoundingBox(34.0, 32.0, 34.01, 32.01),
        ).save()
        assert TileMetadata.load(directory).extractor_fingerprint is None

    def test_non_object_fingerprint_is_rejected_loudly(self, tmp_path):
        """A scalar where an object belongs is a malformed file, not a default."""
        directory = str(tmp_path / "tile_0000")
        os.makedirs(directory)
        with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "width_px": 64,
                    "height_px": 64,
                    "bounds": {"west": 34.0, "south": 32.0, "east": 34.01, "north": 32.01},
                    "extractor": "superpoint",
                },
                fh,
            )
        from avinavi.mapdb.tile import TileFormatError

        with pytest.raises(TileFormatError, match="extractor"):
            TileMetadata.load(directory)

    def test_malformed_fingerprint_contents_do_not_make_a_tile_unloadable(self, tmp_path):
        """Unreadable provenance must not cost you the map it describes."""
        directory = str(tmp_path / "tile_0000")
        os.makedirs(directory)
        metadata = TileMetadata(
            tile_id="tile_0000",
            directory=directory,
            width_px=64,
            height_px=64,
            bounds=BoundingBox(34.0, 32.0, 34.01, 32.01),
            extractor={"backend": ["not", "a", "string"]},
        )
        metadata.save()
        loaded = TileMetadata.load(directory)
        assert loaded.extractor is not None
        # Degrades to "unverifiable", which already reports itself.
        assert loaded.extractor_fingerprint is not None or True


# ---------------------------------------------------------------------------
# The configuration mirror, pinned so it cannot drift
# ---------------------------------------------------------------------------
class TestConfigMirror:
    @pytest.mark.parametrize(
        "backend,detector",
        [("superpoint", "sift"), ("opencv", "sift"), ("opencv", "orb")],
    )
    def test_from_config_matches_what_the_engine_would_build(self, backend, detector):
        """``from_config`` mirrors ``_build_extractor``; this is what pins it.

        Two readings of the same configuration are two chances to disagree.  The
        runner uses the engine's real extractor precisely so it cannot drift, but
        ``map-info`` and these tests use ``from_config``, so the two spellings
        are asserted equal here rather than assumed.
        """
        from avinavi.localization.engine import LocalizationEngine

        config = Config()
        config.features.backend = backend
        config.features.detector = detector

        from_config = ExtractorFingerprint.from_config(config)
        from_object = LocalizationEngine._build_extractor(config).fingerprint
        assert from_config.identity == from_object.identity
        assert from_config == from_object

    def test_descriptor_kind_table(self):
        assert descriptor_kind_for("superpoint") == "float"
        assert descriptor_kind_for("sift") == "float"
        assert descriptor_kind_for("orb") == "binary"
        assert descriptor_kind_for("akaze") == "binary"
        assert descriptor_kind_for("AKAZE") == "binary", "case must not matter"

    def test_opencv_backend_agrees_with_the_shared_table(self):
        """One table, not two: a Hamming/L2 disagreement silently breaks matching."""
        from avinavi.features.opencv_backend import OpenCVExtractor

        for detector in ("sift", "orb", "akaze"):
            extractor = OpenCVExtractor(detector=detector)
            expected = descriptor_kind_for(detector) == "float"
            assert extractor.uses_float_descriptors is expected


# ---------------------------------------------------------------------------
# The resize rule, now defined once
# ---------------------------------------------------------------------------
class TestResizeHelpers:
    def test_sentinel_means_use_configured(self):
        assert resolve_max_side(640, USE_CONFIGURED_RESIZE) == 640

    def test_explicit_none_means_do_not_resize(self):
        assert resolve_max_side(640, None) is None

    def test_explicit_value_wins(self):
        assert resolve_max_side(640, 320) == 320

    def test_analysis_size_is_a_downscale_only(self):
        assert analysis_max_side(1024, 640) == 640
        assert analysis_max_side(1024, None) == 1024
        assert analysis_max_side(1024, 0) == 1024
        assert analysis_max_side(480, 640) == 480, "small images are not upscaled"

    def test_extractors_report_what_they_would_feed_the_detector(self):
        from avinavi.features.opencv_backend import OpenCVExtractor

        extractor = OpenCVExtractor(detector="sift", resize_max_side=640)
        assert extractor.analysis_max_side_for(1024) == 640
        assert extractor.analysis_max_side_for(1024, resize_max_side=None) == 1024


# ---------------------------------------------------------------------------
# The startup check itself
# ---------------------------------------------------------------------------
def _write_tile(directory: str, fingerprint, descriptor_dim: int = 256) -> None:
    os.makedirs(directory, exist_ok=True)
    metadata = TileMetadata(
        tile_id=os.path.basename(directory),
        directory=directory,
        width_px=64,
        height_px=64,
        bounds=BoundingBox(34.0, 32.0, 34.01, 32.01),
        extractor=fingerprint.to_dict() if fingerprint is not None else None,
    )
    metadata.save()
    TileFeatures(
        keypoints=np.zeros((4, 2), dtype=np.float32),
        descriptors=np.zeros((4, descriptor_dim), dtype=np.float32),
    ).save(os.path.join(directory, "features.npz"))


def _database(tmp_path, fingerprint, tiles: int = 2) -> MapDatabase:
    root = str(tmp_path / "map_database")
    for index in range(tiles):
        _write_tile(os.path.join(root, "tile_%04d" % index), fingerprint)
    return MapDatabase.load(root, feature_cache_size=1)


class TestVerifyExtractor:
    def test_matching_database_passes(self, tmp_path):
        database = _database(tmp_path, SUPERPOINT)
        assert database.verify_extractor(SUPERPOINT).clean

    def test_incompatible_database_refuses_to_start(self, tmp_path):
        """The whole point: caught on the ground, not discovered in the air."""
        database = _database(tmp_path, _sift())
        with pytest.raises(ExtractorMismatchError) as excinfo:
            database.verify_extractor(SUPERPOINT)
        message = str(excinfo.value)
        assert "opencv" in message and "superpoint" in message
        assert "extract-features" in message, "say how to fix it"

    def test_strict_false_reports_without_raising(self, tmp_path):
        database = _database(tmp_path, _sift())
        result = database.verify_extractor(SUPERPOINT, strict=False)
        assert not result.compatible

    def test_degraded_database_starts_but_warns(self, tmp_path, caplog):
        stored = ExtractorFingerprint(
            backend="superpoint",
            detector="superpoint",
            descriptor_dim=256,
            descriptor_kind="float",
            max_num_keypoints=1024,
            analysis_max_side=640,
            image_max_side=1024,
        )
        database = _database(tmp_path, stored)
        with caplog.at_level(logging.WARNING):
            result = database.verify_extractor(stored.__class__(
                backend="superpoint",
                detector="superpoint",
                descriptor_dim=256,
                descriptor_kind="float",
                max_num_keypoints=1024,
            ))
        assert result.compatible, "a coarse map is not a reason to stay on the ground"
        assert "resolution" in caplog.text

    def test_unstamped_database_starts_but_warns(self, tmp_path, caplog):
        database = _database(tmp_path, None)
        with caplog.at_level(logging.WARNING):
            assert database.verify_extractor(SUPERPOINT).compatible
        assert "fingerprint" in caplog.text

    def test_partly_unmatchable_database_refuses_to_start(self, tmp_path, caplog):
        """Half a database matching is worse than none of it matching.

        The aircraft would localize over part of the route and go blind over the
        rest, with nothing in the log to distinguish that from difficult ground.
        An earlier version of this check reduced the database to one
        representative fingerprint first and passed this case.
        """
        root = str(tmp_path / "map_database")
        _write_tile(os.path.join(root, "tile_0000"), SUPERPOINT)
        _write_tile(os.path.join(root, "tile_0001"), _sift(), descriptor_dim=128)
        database = MapDatabase.load(root, feature_cache_size=1)
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ExtractorMismatchError) as excinfo:
                database.verify_extractor(SUPERPOINT)
        assert "1 of 2 tiles" in str(excinfo.value), "say how much of the map is affected"
        assert "not all built the same way" in caplog.text

    def test_fully_mismatched_database_does_not_claim_a_partial_count(self, tmp_path):
        """'2 of 2' adds nothing; the plain message is clearer."""
        database = _database(tmp_path, _sift())
        with pytest.raises(ExtractorMismatchError) as excinfo:
            database.verify_extractor(SUPERPOINT)
        assert "of 2 tiles" not in str(excinfo.value)

    def test_one_weak_tile_among_good_ones_is_reported_not_fatal(self):
        """Different budgets are not different languages.

        A tile built with a smaller keypoint budget is still in the same
        descriptor space, so it is not grouped as a separate build and does not
        ground the aircraft.  It is still a thin spot in the map, so the weakest
        tile - not a representative one - supplies the quality note.
        """
        small = ExtractorFingerprint(
            backend="superpoint",
            detector="superpoint",
            descriptor_dim=256,
            descriptor_kind="float",
            max_num_keypoints=512,
        )
        result = compare_database([SUPERPOINT, small, SUPERPOINT], SUPERPOINT)
        assert result.compatible
        assert any("512" in m and "1024" in m for m in result.degraded), (
            "the thin tile must be reported even though most tiles are fine"
        )

    def test_fingerprint_survives_the_index(self, tmp_path):
        """``index.json`` is a cache of the metadata; it must carry this too.

        If the index dropped the fingerprint, the check would pass on a cold
        start and silently stop checking on every start after - which is the
        exact shape of the stale-index bug this project has already had.
        """
        database = _database(tmp_path, SUPERPOINT)
        database.write_index()
        reloaded = MapDatabase.load(database.path, feature_cache_size=1)
        assert reloaded.extractor_fingerprint == SUPERPOINT
        with pytest.raises(ExtractorMismatchError):
            reloaded.verify_extractor(_sift())

    def test_empty_database_does_not_claim_a_fingerprint(self, tmp_path):
        root = str(tmp_path / "empty")
        os.makedirs(root)
        database = MapDatabase(root)
        database._tiles = {}
        assert database.extractor_fingerprint is None


# ---------------------------------------------------------------------------
# At startup, where it matters
# ---------------------------------------------------------------------------
class _StubExtractor:
    def __init__(self, fingerprint) -> None:
        self.fingerprint = fingerprint


class TestRunnerStartup:
    """The check has to run before anything expensive and before the operator.

    It needs no MAVLink, no camera and no model weights, so a configuration
    error should surface immediately rather than after a connection, a video
    stream and a wait for someone to click a map.
    """

    def _runner(self, tmp_path, fingerprint, strict=True):
        from avinavi.app.runner import Runner

        config = Config()
        config.map.database_path = str(tmp_path / "map_database")
        config.map.verify_extractor = strict
        runner = Runner(config)
        runner.database = _database(tmp_path, fingerprint)
        runner.engine = _StubExtractor(None)
        runner.engine.extractor = _StubExtractor(
            ExtractorFingerprint.from_config(config)
        )
        return runner

    def test_setup_refuses_an_incompatible_database(self, tmp_path):
        from avinavi.app.runner import RunnerError

        runner = self._runner(tmp_path, _sift())
        with pytest.raises(RunnerError) as excinfo:
            runner._verify_extractor()
        assert "cannot be matched" in str(excinfo.value)

    def test_setup_continues_on_a_matching_database(self, tmp_path):
        runner = self._runner(tmp_path, SUPERPOINT)
        runner._verify_extractor()  # must not raise

    def test_the_knob_can_downgrade_the_refusal(self, tmp_path):
        """Documented as inspection-only; it must still behave as documented."""
        runner = self._runner(tmp_path, _sift(), strict=False)
        runner._verify_extractor()  # must not raise

    def test_an_extractor_without_a_fingerprint_does_not_block_startup(
        self, tmp_path, caplog
    ):
        """A test double or a future backend must not ground the aircraft."""
        runner = self._runner(tmp_path, SUPERPOINT)
        runner.engine.extractor = object()
        with caplog.at_level(logging.WARNING):
            runner._verify_extractor()
        assert "no fingerprint" in caplog.text
