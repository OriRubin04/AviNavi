"""The visual localization engine.

One call to :meth:`LocalizationEngine.localize` performs a full cycle:

    camera frame
        -> SuperPoint            (live features)
        -> candidate tile search (centred on the last *visual* fix)
        -> LightGlue             (against each candidate's precomputed features)
        -> RANSAC / MAGSAC       (geometric verification, best tile wins)
        -> pose solve            (map pixel -> geographic -> local NED)
        -> confidence gate
        -> VisualFix

The search window is then recentred on the result, so the next cycle looks in
the right place.  Nothing in this file reads GPS.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from ..frames import CameraExtrinsics
from ..geo import LocalOrigin, haversine_distance_m
from ..mapdb.database import MapCoverageError, MapDatabase
from ..mapdb.tile import MapTile, TileFormatError
from ..types import GeoPoint, InvalidReason, MatchQuality, VisualFix
from ..features.geometric import GeometricResult, GeometricVerifier
from ..features.lightglue_matcher import LightGlueMatcher, MatchResult
from ..features.superpoint import SuperPointExtractor
from .confidence import ConfidenceModel
from .pose import PoseError, PoseSolver

LOG = logging.getLogger(__name__)


class LocalizationEngine:
    """Stateful visual localizer with a moving tile search window."""

    def __init__(
        self,
        config: Config,
        database: MapDatabase,
        extractor: Optional[SuperPointExtractor] = None,
        matcher: Optional[LightGlueMatcher] = None,
        verifier: Optional[GeometricVerifier] = None,
        confidence: Optional[ConfidenceModel] = None,
        pose_solver: Optional[PoseSolver] = None,
    ) -> None:
        self.config = config
        self.database = database
        self.extractor = extractor or SuperPointExtractor.from_config(config.superpoint)
        self.matcher = matcher or LightGlueMatcher.from_config(config.lightglue)
        self.verifier = verifier or GeometricVerifier.from_config(config.geometry)
        self.confidence = confidence or ConfidenceModel.from_config(config.confidence)
        extrinsics = CameraExtrinsics.from_config(config.camera_extrinsics)
        self.extrinsics = extrinsics
        self.pose_solver = pose_solver or PoseSolver.from_config(config.camera, extrinsics)

        self._origin: Optional[LocalOrigin] = None
        self._search_center: Optional[GeoPoint] = None
        self._search_radius_m: float = config.tile_search.initial_search_radius_m
        self._last_tile_id: Optional[str] = None
        self._consecutive_failures: int = 0
        self._has_fixed_once: bool = False
        self._last_valid_geo: Optional[GeoPoint] = None

    # -- initialisation -----------------------------------------------------
    def initialize(self, origin: LocalOrigin, start_position: Optional[GeoPoint] = None) -> None:
        """Seed the engine with the EKF origin and the operator's start position.

        ``origin`` anchors the local NED frame and must be the *same* origin
        ArduPilot's EKF uses.  ``start_position`` seeds the tile search window;
        when omitted the origin doubles as the starting position, which is the
        normal case because the operator picks one point for both.
        """
        self._origin = origin
        self._search_center = start_position or origin.geo
        self._search_radius_m = self.config.tile_search.initial_search_radius_m
        self._consecutive_failures = 0
        self._has_fixed_once = False
        self._last_valid_geo = None
        LOG.info(
            "localization initialised: origin=(%.7f, %.7f, %.1f m), search seed=(%.7f, %.7f) r=%.0f m",
            origin.lat_deg,
            origin.lon_deg,
            origin.alt_m,
            self._search_center.lat_deg,
            self._search_center.lon_deg,
            self._search_radius_m,
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
        """Load both networks and run a dummy pass so cycle one is not an outlier."""
        self.extractor.warmup()
        self.matcher.warmup()

    # -- the cycle ----------------------------------------------------------
    def localize(
        self,
        frame: np.ndarray,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
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

        frame_size = (int(frame.shape[1]), int(frame.shape[0]))

        candidates = self._candidate_tiles()
        if not candidates:
            self._register_failure()
            return self._failure(
                started, timestamp_unix, InvalidReason.NO_CANDIDATE_TILES
            )

        frame_features = self.extractor.extract(frame)
        if len(frame_features) < self.config.superpoint.min_keypoints:
            self._register_failure()
            fix = self._failure(started, timestamp_unix, InvalidReason.TOO_FEW_KEYPOINTS)
            fix.candidate_tiles = len(candidates)
            return fix

        best = self._match_and_verify(frame_features, candidates, frame_size)
        if best is None:
            self._register_failure()
            fix = self._failure(
                started, timestamp_unix, InvalidReason.TOO_FEW_MATCHES
            )
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

        try:
            pose = self.pose_solver.solve(
                geometry=geometry,
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

        self._register_success(fix, tile)
        return fix

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
        """Match every candidate and keep the strongest geometrically valid one.

        Selection is by inlier count among tiles that pass verification, with
        the number of raw matches as a tie-break; if nothing passes, the best
        *attempt* is returned so the caller can report why it failed instead of
        just saying "no match".
        """
        best_valid: Optional[Tuple[MapTile, MatchResult, GeometricResult]] = None
        best_attempt: Optional[Tuple[MapTile, MatchResult, GeometricResult]] = None

        for tile in candidates:
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
            if geometry.ok and (
                best_valid is None
                or (geometry.num_inliers, geometry.num_matches)
                > (best_valid[2].num_inliers, best_valid[2].num_matches)
            ):
                best_valid = candidate

        return best_valid or best_attempt

    # -- window management --------------------------------------------------
    def _register_success(self, fix: VisualFix, tile: MapTile) -> None:
        """Recentre the search window on the new visual position."""
        jump_m = None
        if self._last_valid_geo is not None and fix.geo is not None:
            jump_m = haversine_distance_m(self._last_valid_geo, fix.geo)

        self._search_center = fix.geo
        self._search_radius_m = self.config.tile_search.search_radius_m
        self._last_tile_id = tile.tile_id
        self._consecutive_failures = 0
        self._has_fixed_once = True
        self._last_valid_geo = fix.geo

        if jump_m is not None and jump_m > self.config.safety.max_position_jump_m:
            LOG.warning(
                "visual position jumped %.0f m (limit %.0f m); downstream safety "
                "checks will flag this fix",
                jump_m,
                self.config.safety.max_position_jump_m,
            )

    def _register_failure(self) -> None:
        """Grow the search window so a drifted aircraft can be re-acquired."""
        self._consecutive_failures += 1
        search = self.config.tile_search
        if self._consecutive_failures < search.failures_before_growth:
            return
        grown = self._search_radius_m * search.radius_growth_factor
        self._search_radius_m = min(grown, search.max_search_radius_m)
        LOG.info(
            "%d consecutive localization failures; search radius now %.0f m",
            self._consecutive_failures,
            self._search_radius_m,
        )

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def search_exhausted(self) -> bool:
        """True once the window has grown to its ceiling and still finds nothing."""
        return (
            self._consecutive_failures >= self.config.tile_search.failures_before_growth
            and self._search_radius_m >= self.config.tile_search.max_search_radius_m
        )

    def _terrain_alt(self, tile: MapTile) -> float:
        """Terrain altitude for a tile: the tile's own value, else the global one."""
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
