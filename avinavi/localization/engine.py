"""The visual localization engine.

One call to :meth:`LocalizationEngine.localize` performs a full cycle::

    camera frame
        -> rectify to nadir / north-up / tile-GSD   (the keystone step)
        -> photometric normalisation
        -> sharpness gate
        -> SuperPoint                               (live features)
        -> candidate tile search                    (centred on the last fix)
        -> LightGlue                                (vs precomputed tile features)
        -> RANSAC / MAGSAC                          (geometric verification)
        -> compose the homography back to camera pixels
        -> pose solve                               (map pixel -> geo -> local NED)
        -> confidence gate
        -> VisualFix

Why the rectification step is first and is not optional
-------------------------------------------------------
SuperPoint descriptors carry no orientation channel - LightGlue is handed only
``(x, y)`` per keypoint - and LightGlue's own cross-attention blocks have no
positional encoding at all, so the pair is translation invariant **by design and
rotation variant by design**.  Its synthetic pre-training covered +/-31.5
degrees of rotation and its second fine-tuning stage used gravity-aligned
photography with essentially none, so the usable envelope is roughly +/-30
degrees, with a collapse to near-zero matches past about 75.

For a fixed-wing aircraft that is fatal without rectification.  A plane banks to
turn - ArduPlane's ``ROLL_LIMIT_DEG`` defaults to 45 and stall prevention always
permits at least 25 - and a rigidly mounted nadir camera on a banking aircraft
is a tilted camera.  At 100 m AGL with a 60 degree field of view, the ground
sample distance across the frame varies by 1.5x at only **10 degrees** of bank
and by 4x at 30, while learned sparse matching starts failing past about 1.2x.
Flying a circuit therefore means most turns produce no fix at all.

Rectifying first - using the autopilot's attitude and a height estimate - closes
the rotation gap, the perspective gap and the scale gap simultaneously, leaving
near-pure translation.  It is a matching aid only: the recovered homography is
composed back through the warp and the pose is solved in camera pixels exactly
as before, so no approximation introduced by the warp reaches the position.

Nothing in this file reads GPS.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from ..frames import CameraExtrinsics, wrap_pi
from ..geo import LocalOrigin, haversine_distance_m
from ..mapdb.database import MapCoverageError, MapDatabase
from ..mapdb.tile import MapTile, TileFormatError
from ..types import GeoPoint, InvalidReason, MatchQuality, VisualFix
from ..features.geometric import GeometricResult, GeometricVerifier
from ..features.lightglue_matcher import LightGlueMatcher, MatchResult
from ..features.superpoint import SuperPointExtractor
from .confidence import ConfidenceModel
from .pose import PoseError, PoseSolver
from .prewarp import FramePrewarper, PhotometricNormalizer, PrewarpError

LOG = logging.getLogger(__name__)


@dataclass
class PreparedFrame:
    """A frame ready for feature extraction, and how to get back to the camera."""

    image: np.ndarray
    #: Homography from live-camera pixels into this image. Identity when the
    #: frame was not rectified.
    from_camera: np.ndarray
    rectified: bool
    gsd_mpp: Optional[float] = None
    #: Variance of the Laplacian - the sharpness measure the gate uses.
    sharpness: float = float("nan")
    note: str = ""

    @property
    def size(self) -> Tuple[int, int]:
        return (int(self.image.shape[1]), int(self.image.shape[0]))


class YawBiasEstimator:
    """Running estimate of the autopilot's heading error, from the imagery.

    A free and genuinely useful by-product of matching against a north-up
    basemap: the pose solve recovers an absolute heading from the ground itself,
    so the difference against the autopilot's reported yaw *is* the compass
    error — hard-iron offset, declination mistakes and all.

    Two reasons to track it.  It improves the rectification, which depends on
    the assumed yaw and is the one place a heading error directly costs match
    rate.  And because dead-reckoning cross-track error is dominated by heading
    bias rather than scale — one degree of uncorrected heading is 1.7% of
    distance travelled — correcting it is the cheapest available reduction in
    drift during a dropout.

    The estimate is a median over a sliding window, so a single bad fix cannot
    move it, and it is only applied once enough samples agree.
    """

    def __init__(self, window: int = 20, min_samples: int = 8, max_bias_rad: float = 0.35) -> None:
        self.window = int(window)
        self.min_samples = int(min_samples)
        self.max_bias_rad = float(max_bias_rad)
        self._residuals: Deque[float] = deque(maxlen=self.window)

    def update(self, visual_yaw_rad: float, autopilot_yaw_rad: Optional[float]) -> None:
        if autopilot_yaw_rad is None:
            return
        self._residuals.append(wrap_pi(visual_yaw_rad - autopilot_yaw_rad))

    @property
    def bias_rad(self) -> float:
        """Median residual, or 0 until there is enough evidence."""
        if len(self._residuals) < self.min_samples:
            return 0.0
        # Median of wrapped angles is safe here: a compass error is small, so
        # the samples do not straddle the +/-pi discontinuity.
        bias = float(np.median(self._residuals))
        return max(-self.max_bias_rad, min(self.max_bias_rad, bias))

    @property
    def samples(self) -> int:
        return len(self._residuals)

    def reset(self) -> None:
        self._residuals.clear()


class LocalizationEngine:
    """Stateful visual localizer with a moving tile search window."""

    def __init__(
        self,
        config: Config,
        database: MapDatabase,
        extractor=None,
        matcher=None,
        verifier: Optional[GeometricVerifier] = None,
        confidence: Optional[ConfidenceModel] = None,
        pose_solver: Optional[PoseSolver] = None,
    ) -> None:
        self.config = config
        self.database = database
        self.extractor = extractor or self._build_extractor(config)
        self.matcher = matcher or self._build_matcher(config)
        self.verifier = verifier or GeometricVerifier.from_config(config.geometry)
        self.confidence = confidence or ConfidenceModel.from_config(config.confidence)

        extrinsics = CameraExtrinsics.from_config(config.camera_extrinsics)
        self.extrinsics = extrinsics
        self.pose_solver = pose_solver or PoseSolver.from_config(config.camera, extrinsics)
        self.prewarper = FramePrewarper.from_config(
            config.camera, extrinsics, config.prewarp
        )
        self.normalizer = PhotometricNormalizer.from_config(config.photometric)
        self.yaw_bias = YawBiasEstimator()

        self._origin: Optional[LocalOrigin] = None
        self._search_center: Optional[GeoPoint] = None
        #: Position in the candidate list that produced the last result.
        self._last_candidate_rank: Optional[int] = None
        self._search_radius_m: float = config.tile_search.initial_search_radius_m
        self._last_tile_id: Optional[str] = None
        self._consecutive_failures: int = 0
        self._last_valid_geo: Optional[GeoPoint] = None
        self._last_height_m: Optional[float] = None
        self._sharpness_history: Deque[float] = deque(maxlen=30)
        self._rectified_cycles = 0
        self._raw_cycles = 0

    @staticmethod
    def _build_extractor(config: Config):
        if config.features.backend == "opencv":
            from ..features.opencv_backend import OpenCVExtractor

            return OpenCVExtractor(
                detector=config.features.detector,
                max_num_keypoints=config.superpoint.max_num_keypoints,
                resize_max_side=config.superpoint.resize_max_side,
            )
        return SuperPointExtractor.from_config(config.superpoint)

    @staticmethod
    def _build_matcher(config: Config):
        if config.features.backend == "opencv":
            from ..features.opencv_backend import OpenCVExtractor, RatioTestMatcher

            probe = OpenCVExtractor(detector=config.features.detector)
            return RatioTestMatcher.for_extractor(probe, ratio=config.features.ratio)
        return LightGlueMatcher.from_config(config.lightglue)

    # -- initialisation -----------------------------------------------------
    def initialize(self, origin: LocalOrigin, start_position: Optional[GeoPoint] = None) -> None:
        """Seed the engine with the EKF origin and the operator's start position."""
        self._origin = origin
        self._search_center = start_position or origin.geo
        self._search_radius_m = self.config.tile_search.initial_search_radius_m
        self._consecutive_failures = 0
        self._last_valid_geo = None
        LOG.info(
            "localization initialised: origin=(%.7f, %.7f, %.1f m), search seed="
            "(%.7f, %.7f) r=%.0f m",
            origin.lat_deg,
            origin.lon_deg,
            origin.alt_m,
            self._search_center.lat_deg,
            self._search_center.lon_deg,
            self._search_radius_m,
        )
        if not self.prewarper.available and self.config.prewarp.enabled:
            LOG.warning(
                "frame rectification is enabled but camera.fx/fy are not set, so it "
                "cannot run. Expect a markedly lower match rate, especially in "
                "turns - calibrate the camera and set the intrinsics."
            )

    @property
    def initialized(self) -> bool:
        return self._origin is not None and self._search_center is not None

    @property
    def origin(self) -> Optional[LocalOrigin]:
        return self._origin

    @property
    def search_center(self) -> Optional[GeoPoint]:
        return self._search_center

    @property
    def search_radius_m(self) -> float:
        return self._search_radius_m

    def warmup(self) -> None:
        self.extractor.warmup()
        self.matcher.warmup()

    def reanchor(self, geo: GeoPoint, radius_m: Optional[float] = None) -> None:
        """Move the tile search to an operator-supplied point.

        Only the search window changes.  No position is asserted and nothing is
        published; the next frame still has to earn a match.
        """
        self._search_center = geo
        self._search_radius_m = radius_m or self.config.reanchor.radius_m
        self._consecutive_failures = 0
        self._last_tile_id = None
        self._last_valid_geo = None
        LOG.warning(
            "tile search re-anchored to %.7f, %.7f with a %.0f m radius",
            geo.lat_deg,
            geo.lon_deg,
            self._search_radius_m,
        )

    # -- frame preparation --------------------------------------------------
    def prepare_frame(
        self,
        frame: np.ndarray,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        yaw_rad: Optional[float] = None,
        height_m: Optional[float] = None,
        target_gsd_mpp: Optional[float] = None,
    ) -> PreparedFrame:
        """Rectify and normalise a frame, or fall back to the raw frame."""
        identity = np.eye(3)
        prewarp_config = self.config.prewarp

        rectify = (
            prewarp_config.enabled
            and self.prewarper.available
            and height_m is not None
            and height_m > 0.0
        )
        note = ""
        if prewarp_config.enabled and not self.prewarper.available:
            note = "no intrinsics"
        elif prewarp_config.enabled and (height_m is None or height_m <= 0):
            note = "no height estimate"

        if rectify:
            gsd = target_gsd_mpp or self._nominal_tile_gsd()
            # Correct the assumed heading by the measured compass bias, so a
            # systematic magnetometer error stops costing us match rate.
            corrected_yaw = (yaw_rad or 0.0) + self.yaw_bias.bias_rad
            try:
                result = self.prewarper.rectify(
                    frame,
                    height_m=height_m,
                    target_gsd_mpp=gsd,
                    roll_rad=roll_rad,
                    pitch_rad=pitch_rad,
                    yaw_rad=corrected_yaw,
                )
            except PrewarpError as exc:
                note = str(exc)
                LOG.debug("rectification unavailable this cycle: %s", exc)
            else:
                self._rectified_cycles += 1
                image = self.normalizer.apply(result.image)
                return PreparedFrame(
                    image=image,
                    from_camera=result.virtual_from_camera,
                    rectified=True,
                    gsd_mpp=result.gsd_mpp,
                    sharpness=self._sharpness(image),
                    note="rectified",
                )

        self._raw_cycles += 1
        image = self.normalizer.apply(frame)
        return PreparedFrame(
            image=image,
            from_camera=identity,
            rectified=False,
            sharpness=self._sharpness(image),
            note=note or "raw",
        )

    def _nominal_tile_gsd(self) -> float:
        """Ground resolution to rectify to: the nearest candidate tile's."""
        if self._search_center is not None:
            tile = self.database.tile_containing(self._search_center)
            if tile is not None:
                return tile.ground_resolution_mpp
        tiles = self.database.tiles
        if tiles:
            return tiles[0].ground_resolution_mpp
        return 0.5

    @staticmethod
    def _sharpness(image: np.ndarray) -> float:
        """Variance of the Laplacian: the standard cheap focus measure."""
        try:
            import cv2  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            return float("nan")
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _sharpness_acceptable(self, sharpness: float) -> bool:
        """Reject a frame far blurrier than its recent neighbours.

        Motion blur is the single most damaging degradation for this task, and
        the dangerous part is that a blurred frame does not fail cleanly - it
        produces a *confidently wrong* match.  Skipping those both protects the
        solution and saves the matching cost, which matters on a Pi.

        The threshold is relative, because absolute sharpness depends entirely
        on the terrain: a frame over featureless farmland is legitimately less
        sharp than one over a town.
        """
        if not self.config.quality.sharpness_gate_enabled or math.isnan(sharpness):
            return True
        self._sharpness_history.append(sharpness)
        if len(self._sharpness_history) < 10:
            return True
        reference = float(np.median(self._sharpness_history))
        if reference <= 0:
            return True
        return sharpness >= reference * self.config.quality.min_sharpness_ratio

    # -- the cycle ----------------------------------------------------------
    def localize(
        self,
        frame: np.ndarray,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        yaw_rad: Optional[float] = None,
        height_hint_m: Optional[float] = None,
        timestamp_unix: Optional[float] = None,
    ) -> VisualFix:
        """Run one localization cycle over ``frame`` (grayscale or BGR)."""
        started = time.monotonic()
        timestamp_unix = timestamp_unix if timestamp_unix is not None else time.time()

        if not self.initialized:
            return self._failure(started, timestamp_unix, InvalidReason.NOT_INITIALIZED)
        if frame is None or getattr(frame, "size", 0) == 0:
            return self._failure(started, timestamp_unix, InvalidReason.NO_FRAME)

        # A steep bank leaves so little usable, well-conditioned ground in the
        # rectified frame that matching is a waste of the cycle.
        bank_limit = self.config.quality.max_bank_deg
        if bank_limit > 0 and abs(math.degrees(roll_rad)) > bank_limit:
            self._register_failure()
            return self._failure(started, timestamp_unix, InvalidReason.EXCESSIVE_BANK)

        height = height_hint_m if height_hint_m is not None else self._last_height_m
        prepared = self.prepare_frame(
            frame,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            yaw_rad=yaw_rad,
            height_m=height,
        )

        if self.config.prewarp.enabled and not prepared.rectified:
            if not self.config.prewarp.fallback_to_raw:
                self._register_failure()
                fix = self._failure(started, timestamp_unix, InvalidReason.NOT_RECTIFIED)
                return fix

        if not self._sharpness_acceptable(prepared.sharpness):
            self._register_failure()
            return self._failure(started, timestamp_unix, InvalidReason.FRAME_TOO_BLURRED)

        candidates = self._candidate_tiles()
        if not candidates:
            self._register_failure()
            return self._failure(started, timestamp_unix, InvalidReason.NO_CANDIDATE_TILES)

        frame_features = self.extractor.extract(prepared.image)
        if len(frame_features) < self.config.superpoint.min_keypoints:
            self._register_failure()
            fix = self._failure(started, timestamp_unix, InvalidReason.TOO_FEW_KEYPOINTS)
            fix.candidate_tiles = len(candidates)
            return fix

        best = self._match_and_verify(frame_features, candidates, prepared.size)
        if best is None:
            self._register_failure()
            fix = self._failure(started, timestamp_unix, InvalidReason.TOO_FEW_MATCHES)
            fix.candidate_tiles = len(candidates)
            return fix

        tile, match, geometry = best
        quality = self.confidence.evaluate(geometry)

        if not geometry.ok:
            self._register_failure()
            fix = self._failure(
                started,
                timestamp_unix,
                InvalidReason.GEOMETRIC_VERIFICATION_FAILED,
                quality=quality,
                tile_id=tile.tile_id,
            )
            fix.candidate_tiles = len(candidates)
            LOG.debug("tile %s rejected: %s", tile.tile_id, geometry.reason)
            return fix

        if not self.confidence.is_acceptable(quality):
            self._register_failure()
            fix = self._failure(
                started,
                timestamp_unix,
                InvalidReason.LOW_CONFIDENCE,
                quality=quality,
                tile_id=tile.tile_id,
            )
            fix.candidate_tiles = len(candidates)
            return fix

        # Bring the homography back into live-camera pixels so the pose solve
        # is unaffected by the rectification.
        camera_geometry = self._compose_to_camera(geometry, prepared)
        if camera_geometry is None:
            self._register_failure()
            fix = self._failure(
                started,
                timestamp_unix,
                InvalidReason.GEOMETRIC_VERIFICATION_FAILED,
                quality=quality,
                tile_id=tile.tile_id,
            )
            return fix

        frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        try:
            pose = self.pose_solver.solve(
                geometry=camera_geometry,
                tile=tile,
                origin=self._origin,
                frame_size=frame_size,
                roll_rad=roll_rad,
                pitch_rad=pitch_rad,
                terrain_alt_m=self._terrain_alt(tile),
                height_hint_m=height_hint_m,
            )
        except PoseError as exc:
            LOG.warning("pose solve failed on tile %s: %s", tile.tile_id, exc)
            self._register_failure()
            fix = self._failure(
                started,
                timestamp_unix,
                InvalidReason.GEOMETRIC_VERIFICATION_FAILED,
                quality=quality,
                tile_id=tile.tile_id,
            )
            fix.candidate_tiles = len(candidates)
            return fix

        confidence_value = quality.localization_confidence
        fix = VisualFix(
            timestamp_monotonic=started,
            timestamp_unix=timestamp_unix,
            valid=True,
            reason=InvalidReason.NONE,
            geo=pose.geo,
            position_ned=pose.position_ned,
            attitude=pose.attitude,
            yaw_rad=pose.yaw_rad,
            quality=quality,
            tile_id=tile.tile_id,
            candidate_tiles=len(candidates),
            ground_resolution_mpp=pose.frame_gsd_mpp,
            processing_time_s=time.monotonic() - started,
            position_sigma_m=self.confidence.position_sigma_m(confidence_value),
            yaw_sigma_rad=self.confidence.yaw_sigma_rad(confidence_value),
        )

        self.yaw_bias.update(pose.yaw_rad, yaw_rad)
        self._last_height_m = pose.height_agl_m
        self._register_success(fix, tile)
        return fix

    def _compose_to_camera(
        self, geometry: GeometricResult, prepared: PreparedFrame
    ) -> Optional[GeometricResult]:
        """Re-express a prepared-frame homography in live-camera pixels.

        ``geometry.frame_from_tile`` maps tile pixels into the *prepared* image.
        With ``W`` the rectification (camera -> prepared), the camera-space
        transform is ``W⁻¹ @ H``.  The match statistics are left as measured,
        in prepared-frame pixels - which is the right place to measure them,
        since that is the scale the inliers were found at.
        """
        if not prepared.rectified:
            return geometry
        if geometry.frame_from_tile is None:
            return None
        try:
            camera_from_prepared = np.linalg.inv(prepared.from_camera)
        except np.linalg.LinAlgError:
            return None

        frame_from_tile = camera_from_prepared @ geometry.frame_from_tile
        try:
            tile_from_frame = np.linalg.inv(frame_from_tile)
        except np.linalg.LinAlgError:
            return None

        composed = GeometricResult(
            ok=geometry.ok,
            reason=geometry.reason,
            frame_from_tile=frame_from_tile,
            tile_from_frame=tile_from_frame,
            inlier_mask=geometry.inlier_mask,
            num_matches=geometry.num_matches,
            num_inliers=geometry.num_inliers,
            inlier_ratio=geometry.inlier_ratio,
            reprojection_error_px=geometry.reprojection_error_px,
            scale_frame_per_tile=geometry.scale_frame_per_tile,
            rotation_rad=geometry.rotation_rad,
            spatial_spread=geometry.spatial_spread,
            mean_match_score=geometry.mean_match_score,
        )
        return composed

    # -- tile search --------------------------------------------------------
    def _candidate_tiles(self) -> List[MapTile]:
        prefer = (
            [self._last_tile_id]
            if self._last_tile_id and self.config.tile_search.prefer_last_matched_tile
            else None
        )
        try:
            return self.database.candidates(
                center=self._search_center,
                radius_m=self._search_radius_m,
                limit=self.config.lightglue.max_candidate_tiles,
                prefer=prefer,
            )
        except MapCoverageError:
            return []

    def _match_and_verify(
        self,
        frame_features,
        candidates: Sequence[MapTile],
        frame_size: Tuple[int, int],
    ) -> Optional[Tuple[MapTile, MatchResult, GeometricResult]]:
        """Match every candidate and keep the strongest geometrically valid one."""
        best_valid: Optional[Tuple[MapTile, MatchResult, GeometricResult]] = None
        best_attempt: Optional[Tuple[MapTile, MatchResult, GeometricResult]] = None
        # Which position in the candidate list each result came from. Recorded
        # because the candidate limit is a direct cost multiplier - LightGlue
        # runs once per candidate - so knowing whether the winner is usually
        # first or usually fourth is what decides whether the limit can be
        # lowered. Two plausible explanations for the measured sensitivity to
        # that limit (list ordering, and the last-matched-tile preference) were
        # both tested and refuted; this is the direct observable instead.
        self._last_candidate_rank = None
        best_valid_rank: Optional[int] = None
        best_attempt_rank: Optional[int] = None

        for rank, tile in enumerate(candidates):
            try:
                tile_features = self.database.features_for(tile)
            except TileFormatError as exc:
                LOG.warning("tile %s unusable: %s", tile.tile_id, exc)
                continue

            match = self.matcher.match(frame_features, tile_features, tile_id=tile.tile_id)
            if len(match) == 0:
                continue

            geometry = self.verifier.verify(
                frame_points=match.frame_points,
                tile_points=match.tile_points,
                frame_size=frame_size,
                match_scores=match.scores,
            )
            candidate = (tile, match, geometry)

            if best_attempt is None or geometry.num_inliers > best_attempt[2].num_inliers:
                best_attempt = candidate
                best_attempt_rank = rank
            if geometry.ok and (
                best_valid is None
                or (geometry.num_inliers, geometry.num_matches)
                > (best_valid[2].num_inliers, best_valid[2].num_matches)
            ):
                best_valid = candidate
                best_valid_rank = rank

        self._last_candidate_rank = (
            best_valid_rank if best_valid is not None else best_attempt_rank
        )
        return best_valid or best_attempt

    # -- window management --------------------------------------------------
    def _register_success(self, fix: VisualFix, tile: MapTile) -> None:
        jump_m = None
        if self._last_valid_geo is not None and fix.geo is not None:
            jump_m = haversine_distance_m(self._last_valid_geo, fix.geo)

        self._search_center = fix.geo
        self._search_radius_m = self.config.tile_search.search_radius_m
        self._last_tile_id = tile.tile_id
        self._consecutive_failures = 0
        self._last_valid_geo = fix.geo

        if jump_m is not None and jump_m > self.config.safety.max_position_jump_m:
            LOG.warning(
                "visual position jumped %.0f m (limit %.0f m); downstream safety "
                "checks will flag this fix",
                jump_m,
                self.config.safety.max_position_jump_m,
            )

    def _register_failure(self) -> None:
        self._consecutive_failures += 1
        search = self.config.tile_search
        if self._consecutive_failures < search.failures_before_growth:
            return
        grown = self._search_radius_m * search.radius_growth_factor
        self._search_radius_m = min(grown, search.max_search_radius_m)
        LOG.debug(
            "%d consecutive localization failures; search radius now %.0f m",
            self._consecutive_failures,
            self._search_radius_m,
        )

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def search_exhausted(self) -> bool:
        return (
            self._consecutive_failures >= self.config.tile_search.failures_before_growth
            and self._search_radius_m >= self.config.tile_search.max_search_radius_m
        )

    @property
    def last_candidate_rank(self) -> Optional[int]:
        """Zero-based position in the candidate list of the last result.

        A diagnostic for ``lightglue.max_candidate_tiles``: that limit
        multiplies the dominant cost, so whether the winning tile is normally
        first or normally fourth is what decides if it can be lowered.
        """
        return self._last_candidate_rank

    @property
    def rectified_fraction(self) -> float:
        """Share of cycles that were rectified - a key diagnostic."""
        total = self._rectified_cycles + self._raw_cycles
        return self._rectified_cycles / total if total else 0.0

    def _terrain_alt(self, tile: MapTile) -> float:
        if tile.terrain_alt_m:
            return tile.terrain_alt_m
        return self.config.map.terrain_alt_m

    # -- helpers ------------------------------------------------------------
    def _failure(
        self,
        started: float,
        timestamp_unix: float,
        reason: InvalidReason,
        quality: Optional[MatchQuality] = None,
        tile_id: Optional[str] = None,
    ) -> VisualFix:
        return VisualFix(
            timestamp_monotonic=started,
            timestamp_unix=timestamp_unix,
            valid=False,
            reason=reason,
            quality=quality or MatchQuality(),
            tile_id=tile_id,
            processing_time_s=time.monotonic() - started,
        )
