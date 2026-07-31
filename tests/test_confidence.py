"""The confidence model and its gate."""

from __future__ import annotations

import pytest

from avinavi.features.geometric import GeometricResult
from avinavi.localization.confidence import ConfidenceModel


def result(**overrides) -> GeometricResult:
    defaults = dict(
        ok=True,
        num_matches=200,
        num_inliers=150,
        inlier_ratio=0.75,
        reprojection_error_px=1.0,
        spatial_spread=0.7,
        mean_match_score=0.8,
    )
    defaults.update(overrides)
    return GeometricResult(**defaults)


@pytest.fixture
def model() -> ConfidenceModel:
    return ConfidenceModel()


def test_a_strong_match_scores_high(model):
    quality = model.evaluate(
        result(num_inliers=200, inlier_ratio=0.95, reprojection_error_px=0.5, spatial_spread=0.9)
    )
    assert quality.localization_confidence > 0.85
    assert model.is_acceptable(quality)


def test_a_weak_match_scores_low(model):
    quality = model.evaluate(
        result(num_inliers=16, inlier_ratio=0.28, reprojection_error_px=3.8, spatial_spread=0.16)
    )
    assert quality.localization_confidence < 0.35
    assert not model.is_acceptable(quality)


def test_a_failed_geometry_scores_zero(model):
    quality = model.evaluate(result(ok=False, reason="too_few_inliers"))
    assert quality.localization_confidence == 0.0
    assert not model.is_acceptable(quality)


def test_confidence_stays_in_the_unit_interval(model):
    """Even absurd inputs must not produce a score outside [0, 1]."""
    extreme = model.evaluate(
        result(num_inliers=100000, inlier_ratio=5.0, reprojection_error_px=-3.0, spatial_spread=9.0)
    )
    assert 0.0 <= extreme.localization_confidence <= 1.0


def test_nan_reprojection_error_does_not_poison_the_score(model):
    quality = model.evaluate(result(reprojection_error_px=float("nan")))
    assert 0.0 <= quality.localization_confidence <= 1.0


def test_all_required_statistics_are_populated(model):
    """The specification names each of these fields explicitly."""
    quality = model.evaluate(result())
    data = quality.as_dict()
    for field in (
        "number_of_matches",
        "number_of_inliers",
        "inlier_ratio",
        "reprojection_error_px",
        "matching_confidence",
        "localization_confidence",
    ):
        assert field in data
    assert quality.number_of_matches == 200
    assert quality.number_of_inliers == 150
    assert quality.matching_confidence == pytest.approx(0.8)


def test_confidence_rises_with_the_inlier_ratio(model):
    low = model.evaluate(result(inlier_ratio=0.3)).localization_confidence
    high = model.evaluate(result(inlier_ratio=0.9)).localization_confidence
    assert high > low


def test_confidence_falls_with_reprojection_error(model):
    good = model.evaluate(result(reprojection_error_px=0.5)).localization_confidence
    bad = model.evaluate(result(reprojection_error_px=3.5)).localization_confidence
    assert good > bad


def test_clustered_inliers_score_worse_than_spread_ones(model):
    """A geometrically valid but clustered match is the dangerous case."""
    clustered = model.evaluate(result(spatial_spread=0.05)).localization_confidence
    spread = model.evaluate(result(spatial_spread=0.85)).localization_confidence
    assert spread > clustered


def test_sigma_shrinks_as_confidence_grows(model):
    assert model.position_sigma_m(1.0) < model.position_sigma_m(0.7)
    assert model.yaw_sigma_rad(1.0) < model.yaw_sigma_rad(0.7)


def test_sigma_bounds_match_the_configuration(model):
    assert model.position_sigma_m(1.0) == pytest.approx(model.position_sigma_best_m)
    assert model.position_sigma_m(model.min_confidence) == pytest.approx(
        model.position_sigma_worst_m
    )
    # Below the gate the sigma is clamped rather than extrapolated.
    assert model.position_sigma_m(0.0) == pytest.approx(model.position_sigma_worst_m)


def test_weights_are_normalised():
    """Weights that do not sum to 1 must still yield a score in [0, 1]."""
    model = ConfidenceModel(
        weight_inlier_ratio=4.0,
        weight_inlier_count=4.0,
        weight_reprojection=4.0,
        weight_spread=4.0,
    )
    quality = model.evaluate(
        result(num_inliers=1000, inlier_ratio=1.0, reprojection_error_px=0.0, spatial_spread=1.0)
    )
    assert quality.localization_confidence == pytest.approx(1.0)


def test_zero_weights_are_rejected():
    with pytest.raises(ValueError):
        ConfidenceModel(
            weight_inlier_ratio=0.0,
            weight_inlier_count=0.0,
            weight_reprojection=0.0,
            weight_spread=0.0,
        )
