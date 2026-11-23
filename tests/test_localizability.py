"""Predicting offline which map tiles will not localize.

The central claim is that a periodic scene can be told apart from a varied one
using the basemap tile alone. That claim is testable, and it is tested here on
**real SIFT features extracted from rendered imagery** — a synthetic orchard
lattice against ordinary mixed terrain — rather than on hand-made descriptor
arrays, because hand-made arrays would only test that the arithmetic does what
I wrote, not that the measure separates the two cases it exists to separate.

The repetitive case is the important one. It is not "this tile is a bit worse";
it is the only failure mode in the whole system that produces a *confident
wrong* position, because a lattice shifted by one period is still perfectly
consistent under RANSAC.
"""

from __future__ import annotations

import numpy as np
import pytest

from avinavi.features.opencv_backend import OpenCVExtractor
from avinavi.mapdb.localizability import (
    VERDICTS,
    TileLocalizability,
    descriptor_distinctiveness,
    keypoint_density,
    keypoint_spread,
    repetition,
    score_features,
    summarise_scores,
    verdict_for,
)
from avinavi.sim.terrain import TerrainGenerator

SIZE = 512


def extract(image):
    extractor = OpenCVExtractor(
        detector="sift", max_num_keypoints=2000, resize_max_side=None
    )
    features = extractor.extract(image, resize_max_side=None)
    return features.keypoints, features.descriptors


def orchard(period_px=32, size=SIZE):
    """A regular lattice of identical blobs: the dangerous case."""
    image = np.full((size, size), 90, dtype=np.uint8)
    for row in range(period_px, size - period_px, period_px):
        for col in range(period_px, size - period_px, period_px):
            image[row - 4:row + 5, col - 4:col + 5] = 210
    return image


def jittered_orchard(period_px=34, size=SIZE, seed=3):
    """A realistic orchard: repeated, but not identical.

    This is the case that decides whether the measure is of any practical use,
    because no real orchard has identical trees. Position, size and tone all
    vary; only the confusability remains.
    """
    rng = np.random.default_rng(seed)
    image = np.full((size, size), 88, dtype=np.uint8)
    for row in range(period_px, size - period_px, period_px):
        for col in range(period_px, size - period_px, period_px):
            r = row + int(rng.integers(-3, 4))
            c = col + int(rng.integers(-3, 4))
            tone = int(200 + rng.integers(-25, 26))
            radius = int(4 + rng.integers(0, 2))
            image[r - radius:r + radius + 1, c - radius:c + radius + 1] = tone
    return image


def varied(size=SIZE):
    """Ordinary mixed terrain from the simulator's own generator."""
    import cv2

    colour = TerrainGenerator(
        width_px=size, height_px=size, gsd_mpp=0.5, seed=12
    ).generate()
    return cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)


def water(size=SIZE):
    """Near-featureless: a faint gradient and a little noise."""
    rng = np.random.default_rng(4)
    base = np.linspace(100, 115, size, dtype=np.float32)
    image = np.tile(base, (size, 1)) + rng.normal(0.0, 1.0, (size, size))
    return np.clip(image, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# The claim: periodic ground is distinguishable from varied ground
# ---------------------------------------------------------------------------
def test_a_lattice_is_detected_as_repetitive():
    keypoints, descriptors = extract(orchard())
    assert len(keypoints) > 50, "the test image should yield plenty of keypoints"
    assert repetition(keypoints, descriptors) > 0.15


def test_varied_terrain_is_not_detected_as_repetitive():
    keypoints, descriptors = extract(varied())
    assert len(keypoints) > 50
    assert repetition(keypoints, descriptors) < 0.15


def test_the_measure_separates_the_two_cases():
    """The comparison, stated as one assertion.

    If this ever fails, the measure has stopped doing the only job it has, and
    the score built on it means nothing.
    """
    lattice = repetition(*extract(orchard()))
    ordinary = repetition(*extract(varied()))
    assert lattice > 2.0 * ordinary, (
        "lattice %.3f vs varied %.3f: the periodicity measure no longer "
        "separates repetitive ground from ordinary ground" % (lattice, ordinary)
    )


def test_a_realistic_orchard_is_still_detected():
    """The case that decides whether this is of practical use.

    No real orchard has identical trees, so a measure that only catches
    identical repetition would be worthless in the field. With position, size
    and tone all varying, the score must still be high — it depends on the
    elements being *confusable*, not identical.
    """
    keypoints, descriptors = extract(jittered_orchard())
    assert len(keypoints) > 50
    score = repetition(keypoints, descriptors)
    assert score > 0.5, (
        "a jittered lattice scored only %.3f; the measure depends on the "
        "repeated elements being identical, which no real orchard is" % score
    )


def test_a_realistic_orchard_is_flagged_and_not_called_good():
    score = score_features("orchard", *extract(jittered_orchard()), (SIZE, SIZE))
    assert score.repetitive is True
    assert score.verdict in ("poor", "unusable")


def test_distinctiveness_is_the_complement_of_repetition():
    keypoints, descriptors = extract(orchard())
    assert descriptor_distinctiveness(keypoints, descriptors) == pytest.approx(
        1.0 - repetition(keypoints, descriptors)
    )


def test_a_lattice_tile_is_flagged_as_repetitive_with_specific_advice():
    """"Poor" undersells it: the failure is a wrong fix, not a missing one."""
    keypoints, descriptors = extract(orchard())
    score = score_features("orchard", keypoints, descriptors, (SIZE, SIZE))
    assert score.repetitive is True
    assert "one row off" in score.advice or "confidently" in score.advice


def test_varied_terrain_scores_better_than_a_lattice():
    good = score_features("varied", *extract(varied()), (SIZE, SIZE))
    bad = score_features("orchard", *extract(orchard()), (SIZE, SIZE))
    assert good.score > bad.score


def test_featureless_ground_scores_worse_than_varied_terrain():
    good = score_features("varied", *extract(varied()), (SIZE, SIZE))
    blank = score_features("water", *extract(water()), (SIZE, SIZE))
    assert blank.score < good.score


# ---------------------------------------------------------------------------
# The components
# ---------------------------------------------------------------------------
def test_density_saturates_rather_than_rewarding_ever_more_keypoints():
    size = (1000.0, 1000.0)   # one megapixel
    few = keypoint_density(np.zeros((40, 2)), size)
    target = keypoint_density(np.zeros((400, 2)), size)
    many = keypoint_density(np.zeros((4000, 2)), size)
    assert few < target
    assert target == pytest.approx(1.0)
    assert many == pytest.approx(1.0)


def test_density_is_zero_without_keypoints_or_a_size():
    assert keypoint_density(np.zeros((0, 2)), (1000.0, 1000.0)) == 0.0
    assert keypoint_density(np.zeros((10, 2)), None) == 0.0
    assert keypoint_density(np.zeros((10, 2)), (0.0, 0.0)) == 0.0


def test_spread_distinguishes_a_corner_cluster_from_even_coverage():
    """A tile whose features all sit in one corner localizes only there."""
    rng = np.random.default_rng(1)
    corner = rng.uniform(0, 60, size=(300, 2))
    everywhere = rng.uniform(0, 512, size=(300, 2))
    assert keypoint_spread(corner, (512.0, 512.0)) < 0.2
    assert keypoint_spread(everywhere, (512.0, 512.0)) > 0.7


def test_spread_is_zero_without_keypoints():
    assert keypoint_spread(np.zeros((0, 2)), (512.0, 512.0)) == 0.0
    assert keypoint_spread(None, (512.0, 512.0)) == 0.0


def test_a_single_zero_component_makes_the_score_zero():
    """The geometric mean is the point.

    An arithmetic mean would let a high keypoint density hide zero
    distinctiveness, which is exactly the lattice case this must not miss.
    """
    keypoints = np.random.default_rng(2).uniform(0, 512, size=(400, 2))
    score = score_features("no-descriptors", keypoints, None, (512.0, 512.0))
    assert score.distinctiveness == pytest.approx(1.0)  # unmeasurable, not bad
    # Density and spread are real here, so the score is non-zero...
    assert score.score > 0.0
    # ...but remove the keypoints and everything collapses.
    empty = score_features("empty", np.zeros((0, 2)), None, (512.0, 512.0))
    assert empty.score == 0.0
    assert empty.verdict == "unusable"


def test_the_score_is_deterministic():
    """A map database must not change its verdict between runs."""
    keypoints, descriptors = extract(varied())
    first = score_features("t", keypoints, descriptors, (SIZE, SIZE))
    second = score_features("t", keypoints, descriptors, (SIZE, SIZE))
    assert first.score == second.score


def test_too_few_keypoints_is_not_an_error():
    assert repetition(np.zeros((3, 2)), np.zeros((3, 128), dtype=np.float32)) == 0.0
    assert repetition(None, None) == 0.0


def test_mismatched_keypoint_and_descriptor_counts_are_refused():
    assert repetition(np.zeros((10, 2)), np.zeros((9, 128), dtype=np.float32)) == 0.0


# ---------------------------------------------------------------------------
# Verdicts and reporting
# ---------------------------------------------------------------------------
def test_the_verdict_thresholds_are_ordered_and_cover_the_range():
    thresholds = [t for t, _v, _a in VERDICTS]
    assert thresholds == sorted(thresholds, reverse=True)
    assert thresholds[-1] == 0.0
    assert verdict_for(1.0)[0] == "good"
    assert verdict_for(0.0)[0] == "unusable"


def test_every_score_gets_a_verdict():
    for value in np.linspace(0.0, 1.0, 21):
        verdict, advice = verdict_for(float(value))
        assert verdict
        assert advice


def test_the_summary_leads_with_the_worst_tiles():
    """The operator's question is never "which tiles are good"."""
    scores = [
        TileLocalizability("a", 100, 0.9, 0.9, 0.9, 0.90, "good", "", False),
        TileLocalizability("b", 100, 0.2, 0.2, 0.2, 0.20, "poor", "", False),
        TileLocalizability("c", 100, 0.6, 0.6, 0.6, 0.60, "good", "", False),
    ]
    summary = summarise_scores(sorted(scores, key=lambda s: s.score))
    assert summary["tiles_scored"] == 3
    assert summary["worst"][0]["tile_id"] == "b"
    assert summary["score"]["min"] == 0.2
    assert summary["verdicts"]["good"] == 2


def test_the_summary_names_the_repetitive_tiles_separately():
    scores = [
        TileLocalizability("orchard", 900, 0.9, 0.1, 0.9, 0.4, "fair", "", True),
        TileLocalizability("field", 300, 0.7, 0.8, 0.8, 0.8, "good", "", False),
    ]
    summary = summarise_scores(scores)
    assert summary["repetitive_tiles"] == ["orchard"]


def test_an_empty_database_summarises_without_failing():
    assert summarise_scores([])["tiles_scored"] == 0
