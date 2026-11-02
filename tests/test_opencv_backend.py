"""The classical feature backend: SIFT/AKAZE/ORB plus a ratio-test matcher.

This exists for two reasons, and both make it worth testing properly rather
than treating it as a toy. It is the **measurement baseline** the learned stack
is compared against — the A/B in `docs/matching-robustness.md` only means
anything because the same pipeline can be run with a matcher whose properties
are understood — and it is the **fallback** for a Pi with no torch installed.

The property tested hardest is the one that makes keypoints usable at all: they
must come back in the *input* image's pixel grid, whatever internal resizing
happened. A silent factor-of-two there would not look like a bug; it would look
like a pose solution that is quietly wrong.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from avinavi.features.opencv_backend import OpenCVExtractor, RatioTestMatcher
from avinavi.sim.terrain import TerrainGenerator


def detector_available(name: str) -> bool:
    """Which detectors this OpenCV build actually provides.

    AKAZE is absent from OpenCV 5.0, so the parametrised cases skip rather than
    fail — but `test_a_detector_missing_from_this_opencv_is_explained` asserts
    that asking for it anyway produces a message naming the config key.
    """
    return hasattr(cv2, {"sift": "SIFT_create", "akaze": "AKAZE_create",
                         "orb": "ORB_create"}[name])


DETECTORS = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(
            not detector_available(name),
            reason="this OpenCV build has no %s" % name,
        ),
    )
    for name in ("sift", "akaze", "orb")
]


@pytest.fixture(scope="module")
def scene():
    """Real terrain imagery, so the detectors have genuine structure to find."""
    colour = TerrainGenerator(
        width_px=600, height_px=600, gsd_mpp=0.5, seed=9
    ).generate()
    return cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("detector", DETECTORS)
def test_every_detector_finds_features_in_real_terrain(scene, detector):
    features = OpenCVExtractor(detector=detector, resize_max_side=None).extract(scene)
    assert len(features) > 50
    assert features.keypoints.shape[1] == 2
    assert len(features.descriptors) == len(features.keypoints)


def test_an_unknown_detector_is_rejected_by_name(scene):
    with pytest.raises(ValueError, match="unknown features.detector"):
        OpenCVExtractor(detector="surf").extract(scene)


def test_a_detector_missing_from_this_opencv_is_explained(scene):
    """A bare AttributeError names a cv2 symbol, not the setting at fault.

    The same configuration works on one OpenCV and not another - AKAZE is gone
    in 5.0 - so the error has to name the config key, the installed version,
    what is available instead, and that the tile database must be rebuilt.
    """
    missing = [
        name for name in ("sift", "akaze", "orb") if not detector_available(name)
    ]
    if not missing:
        pytest.skip("this OpenCV provides every detector")
    with pytest.raises(RuntimeError) as raised:
        OpenCVExtractor(detector=missing[0]).extract(scene)
    message = str(raised.value)
    assert "features.detector" in message
    assert "extract-features" in message


def test_the_keypoint_cap_is_honoured(scene):
    """AKAZE and ORB overshoot ``nfeatures``, so the cap is applied by hand."""
    for detector in [d.values[0] for d in DETECTORS if not d.marks]:
        features = OpenCVExtractor(
            detector=detector, max_num_keypoints=40, resize_max_side=None
        ).extract(scene)
        assert len(features) <= 40, detector


def test_the_cap_keeps_the_strongest_keypoints(scene):
    """Capping by response, not by detection order."""
    many = OpenCVExtractor(
        detector="sift", max_num_keypoints=2000, resize_max_side=None
    ).extract(scene)
    few = OpenCVExtractor(
        detector="sift", max_num_keypoints=30, resize_max_side=None
    ).extract(scene)
    assert few.scores is not None
    assert float(few.scores.min()) >= float(np.sort(many.scores)[-30])


def test_keypoints_are_returned_in_the_input_pixel_grid(scene):
    """The invariant that makes internal resizing safe.

    Extraction may downscale, but a keypoint at (x, y) must describe the same
    ground in the caller's image whatever it did internally. A factor-of-two
    error here looks like a plausible but wrong pose, not like a bug.
    """
    full = OpenCVExtractor(detector="sift", resize_max_side=None).extract(scene)
    resized = OpenCVExtractor(detector="sift", resize_max_side=300).extract(scene)

    assert resized.image_size == (600.0, 600.0)
    assert full.image_size == (600.0, 600.0)
    # Coordinates must span the original grid, not the downscaled one.
    assert resized.keypoints[:, 0].max() > 320.0
    assert resized.keypoints[:, 1].max() > 320.0
    assert resized.keypoints.max() <= 600.0


def test_the_per_call_resize_overrides_the_configured_one(scene):
    extractor = OpenCVExtractor(detector="sift", resize_max_side=200)
    coarse = extractor.extract(scene)
    fine = extractor.extract(scene, resize_max_side=None)
    assert len(fine) > len(coarse)


def test_a_colour_image_is_accepted(scene):
    colour = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    assert len(OpenCVExtractor(detector="sift", resize_max_side=None).extract(colour)) > 50


def test_a_blank_image_yields_no_features_rather_than_failing():
    blank = np.full((200, 200), 128, dtype=np.uint8)
    features = OpenCVExtractor(detector="sift").extract(blank)
    assert len(features) == 0
    assert features.keypoints.shape == (0, 2)
    assert features.image_size == (200.0, 200.0)


def test_reading_a_missing_file_is_an_explicit_error(tmp_path):
    with pytest.raises(RuntimeError, match="could not read"):
        OpenCVExtractor().extract_image_file(str(tmp_path / "nope.png"))


def test_an_image_file_round_trips(tmp_path, scene):
    path = str(tmp_path / "tile.png")
    cv2.imwrite(path, scene)
    features = OpenCVExtractor(detector="sift", resize_max_side=None).extract_image_file(path)
    assert len(features) > 50


def test_binary_descriptors_are_declared_as_such():
    """ORB and AKAZE need Hamming distance, not L2 — getting it wrong matches
    nothing while raising nothing."""
    assert OpenCVExtractor(detector="sift").uses_float_descriptors is True
    assert OpenCVExtractor(detector="orb").uses_float_descriptors is False
    assert OpenCVExtractor(detector="akaze").uses_float_descriptors is False


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def shifted(image, dx=25, dy=18):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))


@pytest.mark.parametrize("detector", DETECTORS)
def test_a_translated_copy_matches_with_the_right_displacement(scene, detector):
    """The end-to-end check: real features, a known shift, recovered."""
    extractor = OpenCVExtractor(detector=detector, resize_max_side=None)
    matcher = RatioTestMatcher.for_extractor(extractor)

    reference = extractor.extract(scene)
    moved = extractor.extract(shifted(scene, 25, 18))
    result = matcher.match(moved, reference, tile_id="t")

    assert len(result) > 20, detector
    displacement = result.tile_points - result.frame_points
    assert np.median(displacement[:, 0]) == pytest.approx(-25.0, abs=2.0)
    assert np.median(displacement[:, 1]) == pytest.approx(-18.0, abs=2.0)


def test_unrelated_images_produce_few_or_no_matches(scene):
    """What keeps the system safe: silence where it should not match."""
    other = cv2.cvtColor(
        TerrainGenerator(width_px=600, height_px=600, gsd_mpp=0.5, seed=77).generate(),
        cv2.COLOR_BGR2GRAY,
    )
    extractor = OpenCVExtractor(detector="sift", resize_max_side=None)
    matcher = RatioTestMatcher.for_extractor(extractor)

    same = matcher.match(extractor.extract(shifted(scene)), extractor.extract(scene), "a")
    different = matcher.match(extractor.extract(other), extractor.extract(scene), "b")
    assert len(different) < 0.25 * len(same)


def test_a_stricter_ratio_keeps_fewer_matches(scene):
    extractor = OpenCVExtractor(detector="sift", resize_max_side=None)
    moved = extractor.extract(shifted(scene))
    reference = extractor.extract(scene)

    loose = RatioTestMatcher.for_extractor(extractor, ratio=0.9).match(moved, reference)
    strict = RatioTestMatcher.for_extractor(extractor, ratio=0.6).match(moved, reference)
    assert len(strict) < len(loose)


def test_the_mutual_check_removes_one_sided_matches(scene):
    extractor = OpenCVExtractor(detector="sift", resize_max_side=None)
    moved = extractor.extract(shifted(scene))
    reference = extractor.extract(scene)

    with_check = RatioTestMatcher.for_extractor(extractor, cross_check=True)
    without = RatioTestMatcher.for_extractor(extractor, cross_check=False)
    assert len(with_check.match(moved, reference)) <= len(without.match(moved, reference))


def test_scores_are_in_range_and_ordered_best_first(scene):
    extractor = OpenCVExtractor(detector="sift", resize_max_side=None)
    matcher = RatioTestMatcher.for_extractor(extractor)
    result = matcher.match(extractor.extract(shifted(scene)), extractor.extract(scene))
    assert len(result) > 10
    assert result.scores.min() >= 0.0
    assert result.scores.max() <= 1.0
    assert np.all(np.diff(result.scores) <= 1e-6), "scores should descend"


def test_the_match_count_is_capped(scene):
    extractor = OpenCVExtractor(detector="sift", resize_max_side=None)
    matcher = RatioTestMatcher.for_extractor(extractor, max_matches=15)
    result = matcher.match(extractor.extract(shifted(scene)), extractor.extract(scene))
    assert len(result) <= 15


def test_a_descriptor_dimension_mismatch_is_refused_not_matched(scene, caplog):
    """A tile database built with a different detector.

    Matching 128-d SIFT against 32-d ORB cannot mean anything, so it must
    produce nothing and say why — silently matching would be far worse.
    """
    sift = OpenCVExtractor(detector="sift", resize_max_side=None)
    akaze = OpenCVExtractor(detector="orb", resize_max_side=None)
    matcher = RatioTestMatcher.for_extractor(sift)

    with caplog.at_level("WARNING"):
        result = matcher.match(sift.extract(scene), akaze.extract(scene), "t")
    assert len(result) == 0
    assert "dimension mismatch" in caplog.text
    assert "different detector" in caplog.text


def test_too_few_features_on_either_side_is_handled(scene):
    extractor = OpenCVExtractor(detector="sift", resize_max_side=None)
    matcher = RatioTestMatcher.for_extractor(extractor)
    blank = extractor.extract(np.full((200, 200), 128, dtype=np.uint8))

    assert len(matcher.match(blank, extractor.extract(scene), "t")) == 0
    assert len(matcher.match(extractor.extract(scene), blank, "t")) == 0


def test_an_empty_result_still_carries_its_tile_id(scene):
    """Diagnostics need to know which tile produced nothing."""
    extractor = OpenCVExtractor(detector="sift", resize_max_side=None)
    matcher = RatioTestMatcher.for_extractor(extractor)
    blank = extractor.extract(np.full((200, 200), 128, dtype=np.uint8))
    assert matcher.match(blank, extractor.extract(scene), "tile_0042").tile_id == "tile_0042"


def test_match_many_ranks_the_best_tile_first(scene):
    """The engine takes the first result, so the ordering is load-bearing."""
    other = cv2.cvtColor(
        TerrainGenerator(width_px=600, height_px=600, gsd_mpp=0.5, seed=77).generate(),
        cv2.COLOR_BGR2GRAY,
    )
    extractor = OpenCVExtractor(detector="sift", resize_max_side=None)
    matcher = RatioTestMatcher.for_extractor(extractor)

    results = matcher.match_many(
        extractor.extract(shifted(scene)),
        [
            ("wrong", extractor.extract(other)),
            ("right", extractor.extract(scene)),
        ],
    )
    assert results[0].tile_id == "right"


def test_the_matcher_matches_the_extractor_it_was_built_for():
    """`for_extractor` has to pick Hamming for binary descriptors."""
    assert RatioTestMatcher.for_extractor(
        OpenCVExtractor(detector="orb")
    ).binary_descriptors is True
    assert RatioTestMatcher.for_extractor(
        OpenCVExtractor(detector="sift")
    ).binary_descriptors is False


def test_warmup_does_not_raise(scene):
    """Called at startup so the first real cycle is not the slow one."""
    extractor = OpenCVExtractor(detector="sift")
    extractor.warmup()
    RatioTestMatcher.for_extractor(extractor).warmup()
