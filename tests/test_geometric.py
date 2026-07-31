"""Geometric verification: fitting the planar model and rejecting bad matches.

The rejection tests matter more than the acceptance ones.  LightGlue will
happily return confident-looking correspondences between two unrelated aerial
images, and it is this stage's job - not the matcher's - to notice.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from avinavi.features.geometric import (
    GeometricVerifier,
    decompose_local,
    spatial_spread,
)
from tests.fakes import planar_correspondences, similarity_homography

FRAME = (640, 480)


@pytest.fixture
def verifier() -> GeometricVerifier:
    return GeometricVerifier()


@pytest.fixture
def tile_points() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.uniform(0.0, 1000.0, size=(250, 2))


def test_a_clean_similarity_is_accepted(verifier, tile_points):
    homography = similarity_homography(0.5, math.radians(30.0), (100.0, 50.0))
    frame_points = planar_correspondences(tile_points, homography, noise_px=0.4)

    result = verifier.verify(frame_points, tile_points, FRAME)
    assert result.ok, result.reason
    assert result.inlier_ratio > 0.9
    assert result.reprojection_error_px < 1.0
    assert result.scale_frame_per_tile == pytest.approx(0.5, abs=0.02)
    assert result.rotation_rad == pytest.approx(math.radians(30.0), abs=math.radians(1.0))


def test_outliers_are_identified_not_fitted(verifier, tile_points):
    """RANSAC must isolate the 30% garbage rather than compromise with it."""
    rng = np.random.default_rng(3)
    homography = similarity_homography(0.5, 0.2, (50.0, 20.0))
    frame_points = planar_correspondences(tile_points, homography, noise_px=0.4)
    outliers = int(0.3 * len(frame_points))
    frame_points[:outliers] = rng.uniform(0.0, 640.0, size=(outliers, 2))

    result = verifier.verify(frame_points, tile_points, FRAME)
    assert result.ok, result.reason
    assert 0.6 < result.inlier_ratio < 0.8
    assert result.num_inliers == pytest.approx(len(tile_points) - outliers, rel=0.1)


def test_pure_noise_is_rejected(verifier):
    """Two unrelated images must never produce a fix."""
    rng = np.random.default_rng(5)
    result = verifier.verify(
        rng.uniform(0, 640, (250, 2)), rng.uniform(0, 1000, (250, 2)), FRAME
    )
    assert not result.ok
    assert result.num_inliers < 20


def test_too_few_matches_short_circuits(verifier):
    result = verifier.verify(np.zeros((5, 2)), np.zeros((5, 2)), FRAME)
    assert not result.ok
    assert "too_few_matches" in result.reason


def test_clustered_inliers_are_rejected(verifier):
    """A model fitted to one corner extrapolates wildly over the rest of the frame."""
    rng = np.random.default_rng(9)
    # Tile points confined to a small patch => frame points equally confined.
    tile_points = rng.uniform(0.0, 60.0, size=(120, 2))
    homography = similarity_homography(1.0, 0.0, (20.0, 20.0))
    frame_points = planar_correspondences(tile_points, homography, noise_px=0.3)

    result = verifier.verify(frame_points, tile_points, FRAME)
    assert not result.ok
    assert "clustered" in result.reason


def test_an_extreme_scale_change_is_rejected(tile_points):
    """A 20x scale jump is not a plausible altitude for this tile."""
    verifier = GeometricVerifier(min_scale_ratio=0.25, max_scale_ratio=4.0)
    homography = similarity_homography(20.0, 0.0, (0.0, 0.0))
    frame_points = planar_correspondences(tile_points, homography, noise_px=0.2)

    result = verifier.verify(frame_points, tile_points, FRAME)
    assert not result.ok
    assert "scale_out_of_range" in result.reason


def test_a_mirrored_transform_never_produces_a_fix(tile_points):
    """A reflection cannot arise from a real camera looking at real ground.

    Two things can catch it: USAC's own orientation check may refuse to fit the
    model at all, or our explicit determinant test rejects the fitted result.
    Either is fine - what must never happen is a mirrored transform being
    accepted - so this asserts the outcome and
    :func:`test_the_orientation_check_rejects_a_reflection` covers our own check.
    """
    verifier = GeometricVerifier(min_spatial_spread=0.0)
    homography = similarity_homography(0.5, 0.0, (100.0, 50.0))
    homography[0, 0] *= -1.0  # flip the x axis
    homography[1, 0] *= -1.0
    frame_points = planar_correspondences(tile_points, homography, noise_px=0.2)

    result = verifier.verify(frame_points, tile_points, FRAME)
    assert not result.ok
    assert result.reason in ("model_fit_failed", "mirrored_transform") or "inlier" in result.reason


def test_the_orientation_check_rejects_a_reflection():
    from avinavi.features.geometric import _preserves_orientation

    upright = similarity_homography(0.5, 0.4, (10.0, 10.0))
    mirrored = upright.copy()
    mirrored[:2, 0] *= -1.0

    point = np.array([100.0, 100.0])
    assert _preserves_orientation(upright, point)
    assert not _preserves_orientation(mirrored, point)


def test_a_high_inlier_threshold_rejects_a_marginal_match(tile_points):
    strict = GeometricVerifier(min_inlier_ratio=0.95)
    rng = np.random.default_rng(13)
    homography = similarity_homography(0.5, 0.0, (0.0, 0.0))
    frame_points = planar_correspondences(tile_points, homography, noise_px=0.3)
    frame_points[:60] = rng.uniform(0, 640, (60, 2))

    result = strict.verify(frame_points, tile_points, FRAME)
    assert not result.ok
    assert "low_inlier_ratio" in result.reason


def test_the_inverse_transform_is_returned_and_consistent(verifier, tile_points):
    homography = similarity_homography(0.4, 0.5, (30.0, 10.0))
    frame_points = planar_correspondences(tile_points, homography, noise_px=0.2)

    result = verifier.verify(frame_points, tile_points, FRAME)
    assert result.ok
    product = result.frame_from_tile @ result.tile_from_frame
    assert np.allclose(product / product[2, 2], np.eye(3), atol=1e-6)


def test_the_affine_model_also_works(tile_points):
    verifier = GeometricVerifier(model="affine")
    homography = similarity_homography(0.6, 0.3, (40.0, 25.0))
    frame_points = planar_correspondences(tile_points, homography, noise_px=0.3)

    result = verifier.verify(frame_points, tile_points, FRAME)
    assert result.ok, result.reason
    assert result.scale_frame_per_tile == pytest.approx(0.6, abs=0.02)


@pytest.mark.parametrize("method", ["magsac", "ransac", "lmeds"])
def test_every_configured_method_runs(method, tile_points):
    verifier = GeometricVerifier(method=method)
    homography = similarity_homography(0.5, 0.1, (10.0, 10.0))
    frame_points = planar_correspondences(tile_points, homography, noise_px=0.3)
    assert verifier.verify(frame_points, tile_points, FRAME).ok


def test_an_unknown_method_is_rejected(tile_points):
    verifier = GeometricVerifier(method="vibes")
    homography = similarity_homography(0.5, 0.0, (0.0, 0.0))
    frame_points = planar_correspondences(tile_points, homography)
    with pytest.raises(ValueError, match="unknown geometric method"):
        verifier.verify(frame_points, tile_points, FRAME)


# ---------------------------------------------------------------------------
# Local decomposition
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scale,rotation_deg", [(1.0, 0.0), (0.25, 45.0), (2.0, -120.0)])
def test_decompose_local_recovers_scale_and_rotation(scale, rotation_deg):
    homography = similarity_homography(scale, math.radians(rotation_deg), (5.0, -5.0))
    recovered_scale, recovered_rotation = decompose_local(homography, np.array([100.0, 100.0]))
    assert recovered_scale == pytest.approx(scale, rel=1e-9)
    assert recovered_rotation == pytest.approx(math.radians(rotation_deg), abs=1e-9)


def test_decompose_local_handles_a_degenerate_transform():
    scale, rotation = decompose_local(np.zeros((3, 3)), np.array([0.0, 0.0]))
    assert math.isnan(scale) and math.isnan(rotation)


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------
def test_spread_is_near_one_for_full_frame_coverage():
    corners = np.array([[0, 0], [639, 0], [639, 479], [0, 479]], dtype=np.float32)
    assert spatial_spread(corners, FRAME) > 0.95


def test_spread_is_small_for_a_tight_cluster():
    cluster = np.array([[10, 10], [20, 12], [15, 20], [12, 15]], dtype=np.float32)
    assert spatial_spread(cluster, FRAME) < 0.1


def test_spread_is_zero_for_degenerate_input():
    assert spatial_spread(np.zeros((2, 2)), FRAME) == 0.0
    assert spatial_spread(np.zeros((0, 2)), FRAME) == 0.0
