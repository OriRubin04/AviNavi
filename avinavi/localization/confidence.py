"""Turning match statistics into a single confidence number - and a gate.

The specification requires every localization result to carry
``number_of_matches``, ``number_of_inliers``, ``inlier_ratio``,
``reprojection_error``, ``matching_confidence`` and ``localization_confidence``.
This module computes the last one from the others as a weighted sum of four
normalised terms, and translates it into the position/yaw standard deviations
that end up in the ``ODOMETRY`` covariance.

Each term is deliberately bounded in ``[0, 1]``:

``inlier_ratio``   the fraction of LightGlue matches that survived RANSAC - the
                   single most informative signal that the tile is the right one
``inlier_count``   saturating count, because 200 inliers is not twice as
                   trustworthy as 100 but 10 is much worse than 50
``reprojection``   falls linearly from 1 at zero error to 0 at
                   ``reprojection_scale_px``
``spread``         how much of the frame the inliers cover, which catches
                   geometrically valid but locally clustered matches

The gate is intentionally simple and conservative: below ``min_confidence`` the
fix is marked invalid and nothing reaches the flight controller.  It is far
better to publish nothing than to publish a confident lie.
"""

from __future__ import annotations

from typing import Optional

from ..features.geometric import GeometricResult
from ..types import MatchQuality


class ConfidenceModel:
    """Scores a geometric result and maps the score onto measurement noise."""

    def __init__(
        self,
        weight_inlier_ratio: float = 0.4,
        weight_inlier_count: float = 0.25,
        weight_reprojection: float = 0.2,
        weight_spread: float = 0.15,
        inlier_count_saturation: int = 120,
        reprojection_scale_px: float = 4.0,
        min_confidence: float = 0.5,
        position_sigma_best_m: float = 1.0,
        position_sigma_worst_m: float = 15.0,
        yaw_sigma_best_rad: float = 0.05,
        yaw_sigma_worst_rad: float = 0.5,
    ) -> None:
        self.weight_inlier_ratio = float(weight_inlier_ratio)
        self.weight_inlier_count = float(weight_inlier_count)
        self.weight_reprojection = float(weight_reprojection)
        self.weight_spread = float(weight_spread)
        self.inlier_count_saturation = max(1, int(inlier_count_saturation))
        self.reprojection_scale_px = max(1e-6, float(reprojection_scale_px))
        self.min_confidence = float(min_confidence)
        self.position_sigma_best_m = float(position_sigma_best_m)
        self.position_sigma_worst_m = float(position_sigma_worst_m)
        self.yaw_sigma_best_rad = float(yaw_sigma_best_rad)
        self.yaw_sigma_worst_rad = float(yaw_sigma_worst_rad)

        total = (
            self.weight_inlier_ratio
            + self.weight_inlier_count
            + self.weight_reprojection
            + self.weight_spread
        )
        if total <= 0:
            raise ValueError("confidence weights must sum to a positive number")
        # Normalise so the score stays in [0, 1] whatever the operator configured.
        self._weight_total = total

    @classmethod
    def from_config(cls, config) -> "ConfidenceModel":
        return cls(
            weight_inlier_ratio=config.weight_inlier_ratio,
            weight_inlier_count=config.weight_inlier_count,
            weight_reprojection=config.weight_reprojection,
            weight_spread=config.weight_spread,
            inlier_count_saturation=config.inlier_count_saturation,
            reprojection_scale_px=config.reprojection_scale_px,
            min_confidence=config.min_confidence,
            position_sigma_best_m=config.position_sigma_best_m,
            position_sigma_worst_m=config.position_sigma_worst_m,
            yaw_sigma_best_rad=config.yaw_sigma_best_rad,
            yaw_sigma_worst_rad=config.yaw_sigma_worst_rad,
        )

    def evaluate(self, geometry: GeometricResult) -> MatchQuality:
        """Build the full :class:`MatchQuality` record for a geometric result."""
        quality = MatchQuality(
            number_of_matches=geometry.num_matches,
            number_of_inliers=geometry.num_inliers,
            inlier_ratio=geometry.inlier_ratio,
            reprojection_error_px=geometry.reprojection_error_px,
            matching_confidence=geometry.mean_match_score,
            spatial_spread=geometry.spatial_spread,
        )
        if not geometry.ok:
            quality.localization_confidence = 0.0
            return quality

        ratio_term = _clamp01(geometry.inlier_ratio)
        count_term = _clamp01(geometry.num_inliers / float(self.inlier_count_saturation))
        error = geometry.reprojection_error_px
        reprojection_term = (
            0.0
            if error != error  # NaN
            else _clamp01(1.0 - error / self.reprojection_scale_px)
        )
        spread_term = _clamp01(geometry.spatial_spread)

        score = (
            self.weight_inlier_ratio * ratio_term
            + self.weight_inlier_count * count_term
            + self.weight_reprojection * reprojection_term
            + self.weight_spread * spread_term
        ) / self._weight_total
        quality.localization_confidence = _clamp01(score)
        return quality

    def is_acceptable(self, quality: MatchQuality) -> bool:
        return quality.localization_confidence >= self.min_confidence

    def position_sigma_m(self, confidence: float) -> float:
        """Interpolate a position standard deviation from the confidence.

        Confidence 1.0 gives ``position_sigma_best_m``; confidence at or below
        the acceptance threshold gives ``position_sigma_worst_m``.  ArduPilot
        derives ``posErr`` from the covariance we send, and clamps it at
        ``VISO_POS_M_NSE``, so these numbers directly set how much the EKF
        trusts each fix.
        """
        return _interpolate(
            confidence,
            self.min_confidence,
            self.position_sigma_worst_m,
            self.position_sigma_best_m,
        )

    def yaw_sigma_rad(self, confidence: float) -> float:
        return _interpolate(
            confidence,
            self.min_confidence,
            self.yaw_sigma_worst_rad,
            self.yaw_sigma_best_rad,
        )

    def describe(self, quality: MatchQuality, valid: bool) -> str:
        """One-line human summary, used by the diagnostics view and the log."""
        return (
            "valid=%s confidence=%.2f matches=%d inliers=%d ratio=%.3f "
            "reproj=%.2fpx spread=%.2f"
            % (
                "true" if valid else "false",
                quality.localization_confidence,
                quality.number_of_matches,
                quality.number_of_inliers,
                quality.inlier_ratio,
                quality.reprojection_error_px,
                quality.spatial_spread,
            )
        )


def _clamp01(value: float) -> float:
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _interpolate(
    confidence: float, low_confidence: float, value_at_low: float, value_at_high: float
) -> float:
    """Linear map from confidence in ``[low_confidence, 1]`` onto a value range."""
    span = max(1e-6, 1.0 - low_confidence)
    t = _clamp01((confidence - low_confidence) / span)
    return value_at_low + t * (value_at_high - value_at_low)
