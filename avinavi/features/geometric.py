"""Geometric verification of LightGlue correspondences.

LightGlue produces plausible-looking matches even between unrelated images, so
nothing downstream may trust a match until it has survived a robust model fit.
This module fits a planar model - a homography by default, since a downward
camera over locally flat terrain sees a plane - with MAGSAC++ (or plain RANSAC
/ LMedS) and then applies a battery of rejection tests:

* inlier count and inlier ratio,
* mean reprojection error over the inliers,
* transform validity (invertible, orientation preserving, sane scale change),
* spatial distribution of the inliers across the live frame.

The last one matters more than it looks: a homography fitted to matches
clustered in one corner of the image is numerically satisfied but wildly
extrapolates over the rest of the frame, which is exactly the failure that
produces a confident, wrong position.

Convention: the estimated homography maps **tile pixels -> live frame pixels**,
so reprojection residuals are measured in live-camera pixels where the
configured threshold is meaningful.  ``frame_from_tile`` and its inverse
``tile_from_frame`` are both returned.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)


@dataclass
class GeometricResult:
    """Outcome of fitting and validating a planar model."""

    ok: bool
    reason: str = ""
    frame_from_tile: Optional[np.ndarray] = None
    tile_from_frame: Optional[np.ndarray] = None
    inlier_mask: Optional[np.ndarray] = None
    num_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error_px: float = float("nan")
    #: Live-frame pixels per tile pixel implied by the model, at the frame centre.
    scale_frame_per_tile: float = float("nan")
    #: Rotation of the tile's +column axis as seen in the live frame, radians.
    rotation_rad: float = float("nan")
    spatial_spread: float = 0.0
    mean_match_score: float = 0.0

    @property
    def inlier_frame_points(self) -> Optional[np.ndarray]:
        return self._select(self._frame_points)

    def _select(self, points):  # pragma: no cover - helper for debugging views
        if points is None or self.inlier_mask is None:
            return points
        return points[self.inlier_mask]

    _frame_points: Optional[np.ndarray] = None
    _tile_points: Optional[np.ndarray] = None


class GeometricVerifier:
    """Fits and validates the tile -> frame planar transform."""

    def __init__(
        self,
        method: str = "magsac",
        model: str = "homography",
        reproj_threshold_px: float = 3.0,
        max_iterations: int = 5000,
        confidence: float = 0.999,
        min_matches: int = 20,
        min_inliers: int = 15,
        min_inlier_ratio: float = 0.25,
        max_reprojection_error_px: float = 4.0,
        min_scale_ratio: float = 0.25,
        max_scale_ratio: float = 4.0,
        min_spatial_spread: float = 0.15,
    ) -> None:
        self.method = method
        self.model = model
        self.reproj_threshold_px = float(reproj_threshold_px)
        self.max_iterations = int(max_iterations)
        self.confidence = float(confidence)
        self.min_matches = int(min_matches)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.min_scale_ratio = float(min_scale_ratio)
        self.max_scale_ratio = float(max_scale_ratio)
        self.min_spatial_spread = float(min_spatial_spread)

    @classmethod
    def from_config(cls, config) -> "GeometricVerifier":
        return cls(
            method=config.method,
            model=config.model,
            reproj_threshold_px=config.reproj_threshold_px,
            max_iterations=config.max_iterations,
            confidence=config.confidence,
            min_matches=config.min_matches,
            min_inliers=config.min_inliers,
            min_inlier_ratio=config.min_inlier_ratio,
            max_reprojection_error_px=config.max_reprojection_error_px,
            min_scale_ratio=config.min_scale_ratio,
            max_scale_ratio=config.max_scale_ratio,
            min_spatial_spread=config.min_spatial_spread,
        )

    def verify(
        self,
        frame_points: np.ndarray,
        tile_points: np.ndarray,
        frame_size: Tuple[int, int],
        match_scores: Optional[np.ndarray] = None,
    ) -> GeometricResult:
        """Fit and validate the model.  ``frame_size`` is ``(width, height)``."""
        frame_points = np.asarray(frame_points, dtype=np.float64).reshape(-1, 2)
        tile_points = np.asarray(tile_points, dtype=np.float64).reshape(-1, 2)
        num_matches = int(frame_points.shape[0])
        mean_score = (
            float(np.mean(match_scores)) if match_scores is not None and len(match_scores) else 0.0
        )

        if num_matches < self.min_matches:
            return GeometricResult(
                ok=False,
                reason="too_few_matches (%d < %d)" % (num_matches, self.min_matches),
                num_matches=num_matches,
                mean_match_score=mean_score,
            )

        cv2 = _require_cv2()
        matrix, mask = self._fit(cv2, tile_points, frame_points)
        if matrix is None or mask is None:
            return GeometricResult(
                ok=False,
                reason="model_fit_failed",
                num_matches=num_matches,
                mean_match_score=mean_score,
            )

        inlier_mask = mask.ravel().astype(bool)
        num_inliers = int(inlier_mask.sum())
        inlier_ratio = num_inliers / float(num_matches)

        result = GeometricResult(
            ok=False,
            frame_from_tile=matrix,
            inlier_mask=inlier_mask,
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            mean_match_score=mean_score,
            _frame_points=frame_points,
            _tile_points=tile_points,
        )

        if num_inliers < self.min_inliers:
            result.reason = "too_few_inliers (%d < %d)" % (num_inliers, self.min_inliers)
            return result
        if inlier_ratio < self.min_inlier_ratio:
            result.reason = "low_inlier_ratio (%.3f < %.3f)" % (
                inlier_ratio,
                self.min_inlier_ratio,
            )
            return result

        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            result.reason = "singular_transform"
            return result
        result.tile_from_frame = inverse

        result.reprojection_error_px = self._reprojection_error(
            matrix, tile_points[inlier_mask], frame_points[inlier_mask]
        )
        if not math.isfinite(result.reprojection_error_px):
            result.reason = "non_finite_reprojection_error"
            return result
        if result.reprojection_error_px > self.max_reprojection_error_px:
            result.reason = "high_reprojection_error (%.2f px > %.2f px)" % (
                result.reprojection_error_px,
                self.max_reprojection_error_px,
            )
            return result

        centre_tile = _apply(inverse, np.array([[frame_size[0] * 0.5, frame_size[1] * 0.5]]))[0]
        scale, rotation = decompose_local(matrix, centre_tile)
        result.scale_frame_per_tile = scale
        result.rotation_rad = rotation
        if not math.isfinite(scale) or scale <= 0.0:
            result.reason = "degenerate_scale"
            return result
        if not self.min_scale_ratio <= scale <= self.max_scale_ratio:
            result.reason = "scale_out_of_range (%.3f not in [%.3f, %.3f])" % (
                scale,
                self.min_scale_ratio,
                self.max_scale_ratio,
            )
            return result

        result.spatial_spread = spatial_spread(frame_points[inlier_mask], frame_size)
        if result.spatial_spread < self.min_spatial_spread:
            result.reason = "inliers_too_clustered (%.3f < %.3f)" % (
                result.spatial_spread,
                self.min_spatial_spread,
            )
            return result

        if not _preserves_orientation(matrix, centre_tile):
            result.reason = "mirrored_transform"
            return result

        result.ok = True
        result.reason = "ok"
        return result

    def _fit(self, cv2, src: np.ndarray, dst: np.ndarray):
        """Fit ``dst ~ M @ src`` robustly. src=tile pixels, dst=frame pixels."""
        flag = _method_flag(cv2, self.method)
        if self.model == "affine":
            matrix, mask = cv2.estimateAffinePartial2D(
                src.reshape(-1, 1, 2),
                dst.reshape(-1, 1, 2),
                method=flag if flag in (cv2.RANSAC, cv2.LMEDS) else cv2.RANSAC,
                ransacReprojThreshold=self.reproj_threshold_px,
                maxIters=self.max_iterations,
                confidence=self.confidence,
            )
            if matrix is None:
                return None, None
            # Promote the 2x3 affine to a 3x3 so downstream code is uniform.
            matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
            return matrix, mask

        matrix, mask = cv2.findHomography(
            src.reshape(-1, 1, 2),
            dst.reshape(-1, 1, 2),
            method=flag,
            ransacReprojThreshold=self.reproj_threshold_px,
            maxIters=self.max_iterations,
            confidence=self.confidence,
        )
        return matrix, mask

    @staticmethod
    def _reprojection_error(
        matrix: np.ndarray, tile_points: np.ndarray, frame_points: np.ndarray
    ) -> float:
        if tile_points.shape[0] == 0:
            return float("nan")
        projected = _apply(matrix, tile_points)
        residuals = np.linalg.norm(projected - frame_points, axis=1)
        return float(np.mean(residuals))


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "geometric verification needs OpenCV; pip install opencv-python-headless"
        ) from exc
    return cv2


def _method_flag(cv2, name: str) -> int:
    """Map the configured method name onto an OpenCV estimator flag.

    MAGSAC++ appeared in OpenCV 4.5; older builds fall back to RANSAC with a
    warning rather than failing at flight time.
    """
    name = (name or "").lower()
    if name == "ransac":
        return cv2.RANSAC
    if name == "lmeds":
        return cv2.LMEDS
    if name == "magsac":
        flag = getattr(cv2, "USAC_MAGSAC", None)
        if flag is None:
            LOG.warning("this OpenCV build has no USAC_MAGSAC; using RANSAC instead")
            return cv2.RANSAC
        return flag
    raise ValueError("unknown geometric method %r" % name)


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 projective transform to ``(N, 2)`` points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    projected = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    w = projected[:, 2:3]
    # Guard against points projecting to (or through) the horizon.
    w = np.where(np.abs(w) < 1e-12, np.nan, w)
    return projected[:, :2] / w


def local_jacobian(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    """2x2 Jacobian of the projective map at ``point`` (in source coordinates).

    A homography is only locally affine; evaluating its Jacobian at the point of
    interest is what lets us read off a meaningful scale and rotation instead of
    guessing from the raw matrix entries.
    """
    h = np.asarray(matrix, dtype=np.float64)
    x, y = float(point[0]), float(point[1])
    denominator = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    if abs(denominator) < 1e-12:
        return np.full((2, 2), np.nan)
    u = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    v = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    jacobian = np.empty((2, 2))
    jacobian[0, 0] = (h[0, 0] * denominator - u * h[2, 0]) / denominator**2
    jacobian[0, 1] = (h[0, 1] * denominator - u * h[2, 1]) / denominator**2
    jacobian[1, 0] = (h[1, 0] * denominator - v * h[2, 0]) / denominator**2
    jacobian[1, 1] = (h[1, 1] * denominator - v * h[2, 1]) / denominator**2
    return jacobian


def decompose_local(matrix: np.ndarray, point: np.ndarray) -> Tuple[float, float]:
    """Local ``(scale, rotation)`` of a projective map at ``point``.

    The scale is the geometric mean of the Jacobian's singular values (an
    area-preserving average that does not favour either axis), and the rotation
    is the direction the source +X axis ends up pointing.
    """
    jacobian = local_jacobian(matrix, point)
    if not np.all(np.isfinite(jacobian)):
        return (float("nan"), float("nan"))
    singular = np.linalg.svd(jacobian, compute_uv=False)
    scale = float(math.sqrt(max(singular[0] * singular[1], 0.0)))
    rotation = float(math.atan2(jacobian[1, 0], jacobian[0, 0]))
    return (scale, rotation)


def _preserves_orientation(matrix: np.ndarray, point: np.ndarray) -> bool:
    """Reject mirrored transforms, which are never physically possible here."""
    jacobian = local_jacobian(matrix, point)
    if not np.all(np.isfinite(jacobian)):
        return False
    return float(np.linalg.det(jacobian)) > 0.0


def spatial_spread(points: np.ndarray, frame_size: Tuple[int, int]) -> float:
    """How well inliers cover the frame, in ``[0, 1]``.

    Defined as ``sqrt(convex hull area / frame area)`` so it behaves like a
    linear fraction of the frame's width: 1.0 means the inliers span the whole
    image, 0.1 means they occupy a patch a tenth of the frame across.  Falls
    back to a bounding-box measure when OpenCV is unavailable.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    width, height = float(frame_size[0]), float(frame_size[1])
    frame_area = width * height
    if points.shape[0] < 3 or frame_area <= 0:
        return 0.0
    try:
        cv2 = _require_cv2()
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        area = float(cv2.contourArea(hull))
    except RuntimeError:  # pragma: no cover - OpenCV missing
        extent = points.max(axis=0) - points.min(axis=0)
        area = float(extent[0] * extent[1])
    return float(min(1.0, math.sqrt(max(area, 0.0) / frame_area)))
