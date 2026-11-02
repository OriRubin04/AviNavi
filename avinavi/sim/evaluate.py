"""Running a simulated flight through the real pipeline and scoring it.

The metrics are chosen to answer the questions that actually matter in flight,
in order of importance:

``lock_rate``
    Fraction of localization cycles that produced a valid fix.  The headline
    number, and the one that determines whether the aircraft has a position at
    all.

``false_fix_rate``
    Fraction of *accepted* fixes that were badly wrong — further from truth than
    ``false_fix_threshold_m``.  **This matters more than the lock rate.**  A
    system that locks 60% of the time and is never badly wrong is useful; one
    that locks 95% of the time and is occasionally 6 km off is dangerous,
    because the EKF will believe it.  Published work on this exact problem
    found matchers that scored *better* on match count and were far worse
    end-to-end for precisely this reason.

``separation``
    Median inliers at the true tile divided by median inliers at a deliberately
    wrong tile.  This is the structural version of the same question: can the
    matcher stay silent where it should not match at all?  A high ratio means a
    threshold exists that cleanly separates right from wrong.

``position_error_m``
    Percentiles, not just the mean.  The p90 is where a localizer's real
    character shows; a good median with a terrible tail is a filter divergence
    waiting to happen.

Absolute numbers from synthetic imagery are optimistic — see the package
docstring.  The *comparisons* are what this harness is for.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from ..frames import CameraExtrinsics, wrap_pi
from ..geo import BoundingBox, LocalOrigin, haversine_distance_m
from ..localization.engine import LocalizationEngine
from ..mapdb.database import MapDatabase
from ..types import GeoPoint, InvalidReason, NEDPosition
from .flight import TrajectoryPoint, fly_circuit, trajectory_statistics
from .render import CameraSimulator, RenderSettings
from .terrain import TerrainGenerator, seasonal_variant

LOG = logging.getLogger(__name__)


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("the simulator needs OpenCV") from exc
    return cv2


@dataclass
class SimulationReport:
    """Scored outcome of one simulated flight."""

    label: str
    cycles: int = 0
    valid_fixes: int = 0
    false_fixes: int = 0
    position_errors_m: List[float] = field(default_factory=list)
    #: Signed (north, east, down) error per accepted fix.  Recorded separately
    #: from the magnitude because the two failure modes look identical in a
    #: magnitude histogram and could not be more different: scattered errors
    #: mean the matcher is latching onto the wrong ground, while a *constant*
    #: offset means a frame-convention or georeferencing bug in the harness or
    #: the pose chain.  Keeping only the magnitude once cost a whole run.
    position_error_vectors_m: List[Tuple[float, float, float]] = field(
        default_factory=list
    )
    yaw_errors_deg: List[float] = field(default_factory=list)
    inliers: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    processing_times_s: List[float] = field(default_factory=list)
    failure_reasons: Dict[str, int] = field(default_factory=dict)
    #: Lock rate split by bank angle, which is where a fixed wing struggles.
    by_bank: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    wrong_tile_inliers: List[int] = field(default_factory=list)
    trajectory: Dict = field(default_factory=dict)
    settings: Dict = field(default_factory=dict)

    @property
    def lock_rate(self) -> float:
        return self.valid_fixes / self.cycles if self.cycles else 0.0

    @property
    def false_fix_rate(self) -> float:
        return self.false_fixes / self.valid_fixes if self.valid_fixes else 0.0

    @property
    def separation(self) -> Optional[float]:
        """Median true-tile inliers over median wrong-tile inliers."""
        if not self.inliers or not self.wrong_tile_inliers:
            return None
        wrong = float(np.median(self.wrong_tile_inliers))
        right = float(np.median(self.inliers))
        if wrong < 1.0:
            # No inliers at all on the wrong tile: cleanly separable. Report the
            # true-tile median as the ratio floor rather than dividing by zero.
            return right
        return right / wrong

    def percentile(self, values: Sequence[float], q: float) -> Optional[float]:
        return float(np.percentile(values, q)) if len(values) else None

    @property
    def bias_ned_m(self) -> Optional[Tuple[float, float, float]]:
        """Median signed error per axis.

        If this is large while the *scatter* about it is small, the run is
        measuring a systematic offset, not localization quality.
        """
        if not self.position_error_vectors_m:
            return None
        array = np.asarray(self.position_error_vectors_m, dtype=float)
        median = np.median(array, axis=0)
        return (float(median[0]), float(median[1]), float(median[2]))

    @property
    def scatter_about_bias_m(self) -> Optional[float]:
        """Median horizontal distance from the bias: the real error spread."""
        if not self.position_error_vectors_m:
            return None
        array = np.asarray(self.position_error_vectors_m, dtype=float)
        centred = array[:, :2] - np.median(array[:, :2], axis=0)
        return float(np.median(np.hypot(centred[:, 0], centred[:, 1])))

    def as_dict(self) -> Dict:
        return {
            "label": self.label,
            "cycles": self.cycles,
            "lock_rate": round(self.lock_rate, 4),
            "valid_fixes": self.valid_fixes,
            "false_fixes": self.false_fixes,
            "false_fix_rate": round(self.false_fix_rate, 4),
            "separation_true_over_wrong": (
                round(self.separation, 1) if self.separation is not None else None
            ),
            "position_error_m": {
                "median": _round(self.percentile(self.position_errors_m, 50)),
                "p90": _round(self.percentile(self.position_errors_m, 90)),
                "max": _round(max(self.position_errors_m) if self.position_errors_m else None),
            },
            "bias_ned_m": (
                [_round(v) for v in self.bias_ned_m]
                if self.bias_ned_m is not None
                else None
            ),
            "scatter_about_bias_m": _round(self.scatter_about_bias_m),
            "yaw_error_deg": {
                "median": _round(self.percentile(self.yaw_errors_deg, 50)),
                "p90": _round(self.percentile(self.yaw_errors_deg, 90)),
            },
            "inliers": {
                "median": _round(self.percentile(self.inliers, 50)),
                "p10": _round(self.percentile(self.inliers, 10)),
            },
            "confidence_median": _round(self.percentile(self.confidences, 50), 3),
            "processing_time_s": {
                "mean": _round(
                    float(np.mean(self.processing_times_s)) if self.processing_times_s else None,
                    3,
                ),
                "p95": _round(self.percentile(self.processing_times_s, 95), 3),
            },
            "lock_rate_by_bank": {
                band: round(good / total, 3) if total else None
                for band, (good, total) in sorted(self.by_bank.items())
            },
            "failure_reasons": dict(
                sorted(self.failure_reasons.items(), key=lambda kv: -kv[1])
            ),
            "trajectory": self.trajectory,
            "settings": self.settings,
        }


def _round(value, digits: int = 2):
    return None if value is None else round(float(value), digits)


def _trajectory_fingerprint(trajectory: Sequence[TrajectoryPoint]) -> tuple:
    """A cheap content-derived identity for a trajectory."""
    if not trajectory:
        return (0,)
    first, last = trajectory[0], trajectory[-1]
    return (
        len(trajectory),
        round(first.north_m, 3),
        round(first.east_m, 3),
        round(first.yaw_rad, 6),
        round(last.north_m, 3),
        round(last.east_m, 3),
        round(last.yaw_rad, 6),
        round(last.time_s, 3),
    )


def _bank_band(roll_rad: float) -> str:
    """Bucket the bank angle.

    The bands are not arbitrary: unwarped matching degrades once the
    near/far ground-sample ratio exceeds about 1.5x, which happens at only
    ~10 degrees of bank, and reaches 4x by 30 degrees.
    """
    degrees = abs(math.degrees(roll_rad))
    if degrees < 5.0:
        return "00-05deg"
    if degrees < 10.0:
        return "05-10deg"
    if degrees < 20.0:
        return "10-20deg"
    if degrees < 30.0:
        return "20-30deg"
    return "30deg+"


def build_tile_database(
    basemap: np.ndarray,
    basemap_gsd_mpp: float,
    origin: LocalOrigin,
    output_dir: str,
    extractor,
    tile_size_px: int = 1024,
    overlap: float = 0.3,
    normalizer=None,
) -> MapDatabase:
    """Cut ``basemap`` into georeferenced tiles and precompute their features."""
    cv2 = _require_cv2()
    from ..mapdb.builder import write_tile
    from ..mapdb.tile import TileFeatures

    # Start from an empty directory, always. Writing a new tiling over an old
    # one leaves orphan tiles and a stale index, and the result is a database
    # that loads cleanly while describing ground that is no longer there. The
    # measurement this harness exists to produce was invalidated exactly once
    # that way, silently; a rebuild is cheap and a wrong answer is not.
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    height, width = basemap.shape[:2]
    stride = max(1, int(round(tile_size_px * (1.0 - overlap))))
    origin_pixel = (width / 2.0, height / 2.0)

    index = 0
    for row0 in range(0, max(1, height - tile_size_px + 1), stride):
        for col0 in range(0, max(1, width - tile_size_px + 1), stride):
            col1 = min(col0 + tile_size_px, width)
            row1 = min(row0 + tile_size_px, height)
            patch = basemap[row0:row1, col0:col1]

            # Tile corners in local NED, then in geographic coordinates.
            north_top = (origin_pixel[1] - row0) * basemap_gsd_mpp
            north_bottom = (origin_pixel[1] - row1) * basemap_gsd_mpp
            east_left = (col0 - origin_pixel[0]) * basemap_gsd_mpp
            east_right = (col1 - origin_pixel[0]) * basemap_gsd_mpp
            south_west = origin.to_geo(NEDPosition(north_bottom, east_left, 0.0))
            north_east = origin.to_geo(NEDPosition(north_top, east_right, 0.0))

            tile_id = "tile_%04d" % index
            directory = os.path.join(output_dir, tile_id)
            write_tile(
                directory=directory,
                image=patch,
                bounds=BoundingBox(
                    south_west.lon_deg,
                    south_west.lat_deg,
                    north_east.lon_deg,
                    north_east.lat_deg,
                ),
                tile_id=tile_id,
                terrain_alt_m=0.0,
            )

            prepared = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if normalizer is not None:
                # Both sides must get the same transform, or normalisation
                # makes matching worse rather than better.
                prepared = normalizer.apply(prepared)
            features = extractor.extract(prepared, resize_max_side=None)
            features.image_size = (float(patch.shape[1]), float(patch.shape[0]))
            features.save(os.path.join(directory, "features.npz"))
            index += 1

    database = MapDatabase.load(output_dir, feature_cache_size=64)
    database.write_index()
    LOG.info("built %d tiles in %s", len(database), output_dir)
    return database


def run_simulation(
    config: Config,
    label: str = "run",
    basemap_px: int = 3000,
    basemap_gsd_mpp: float = 0.5,
    tile_size_px: int = 1024,
    terrain_seed: int = 7,
    appearance_strength: float = 1.0,
    trajectory: Optional[Sequence[TrajectoryPoint]] = None,
    render_settings: Optional[RenderSettings] = None,
    extractor=None,
    matcher=None,
    workdir: Optional[str] = None,
    max_cycles: Optional[int] = None,
    false_fix_threshold_m: float = 60.0,
    measure_separation: bool = True,
    cache: Optional[Dict] = None,
) -> SimulationReport:
    """Fly a simulated mission through the pipeline and score the result.

    ``cache`` lets several runs share one basemap and tile database, which is
    what makes an A/B comparison both fast and fair - every variant sees exactly
    the same ground and the same frames.
    """
    from ..features.opencv_backend import OpenCVExtractor, RatioTestMatcher
    from ..localization.prewarp import PhotometricNormalizer

    cv2 = _require_cv2()
    workdir = workdir or os.path.join(os.getcwd(), ".simulation")
    os.makedirs(workdir, exist_ok=True)
    cache = cache if cache is not None else {}

    origin = LocalOrigin(32.0853, 34.7818, 40.0)
    extractor = extractor or OpenCVExtractor(
        detector=config.features.detector,
        max_num_keypoints=config.superpoint.max_num_keypoints,
        resize_max_side=config.superpoint.resize_max_side,
    )
    matcher = matcher or RatioTestMatcher.for_extractor(
        extractor, ratio=config.features.ratio
    )
    normalizer = (
        PhotometricNormalizer.from_config(config.photometric)
        if config.photometric.enabled
        else None
    )

    # -- terrain, shared across runs ---------------------------------------
    terrain_key = ("terrain", basemap_px, basemap_gsd_mpp, terrain_seed)
    if terrain_key not in cache:
        LOG.info("generating a %dx%d basemap", basemap_px, basemap_px)
        cache[terrain_key] = TerrainGenerator(
            width_px=basemap_px,
            height_px=basemap_px,
            gsd_mpp=basemap_gsd_mpp,
            seed=terrain_seed,
        ).generate()
    basemap = cache[terrain_key]

    # The tile database is built from the basemap; the camera sees a *different
    # date* of the same ground, which is the gap a real flight faces.
    variant_key = ("variant", terrain_key, appearance_strength)
    if variant_key not in cache:
        cache[variant_key] = (
            seasonal_variant(
                basemap, np.random.default_rng(99), strength=appearance_strength
            )
            if appearance_strength > 0
            else basemap
        )
    flown_map = cache[variant_key]

    # -- tile database, shared across runs with the same normalisation ------
    db_key = (
        "database",
        terrain_key,
        tile_size_px,
        config.features.detector,
        bool(normalizer),
        config.photometric.clip_limit if normalizer else None,
    )
    if db_key not in cache:
        tile_dir = os.path.join(
            workdir, "tiles_%s_%s" % (config.features.detector, "clahe" if normalizer else "raw")
        )
        cache[db_key] = build_tile_database(
            basemap=basemap,
            basemap_gsd_mpp=basemap_gsd_mpp,
            origin=origin,
            output_dir=tile_dir,
            extractor=extractor,
            tile_size_px=tile_size_px,
            normalizer=normalizer,
        )
    database = cache[db_key]
    config.map.database_path = database.path

    # -- trajectory and frames, shared across runs -------------------------
    if trajectory is None:
        trajectory = fly_circuit(
            extent_m=basemap_px * basemap_gsd_mpp * 0.42,
            airspeed_mps=22.0,
            cruise_height_m=120.0,
            max_bank_deg=30.0,
            wind_ned=(4.0, -3.0),
            duration_s=180.0,
            dt_s=1.0 / max(config.rates.visual_localization_hz, 0.5),
            pattern="racetrack",
        )
    render_settings = render_settings or RenderSettings()
    # Keep the pipeline's intrinsics consistent with the simulated camera.
    config.camera.fx = render_settings.fx
    config.camera.fy = render_settings.fy
    config.camera.cx = render_settings.cx
    config.camera.cy = render_settings.cy
    config.camera.width = render_settings.width_px
    config.camera.height = render_settings.height_px

    extrinsics = CameraExtrinsics.from_config(config.camera_extrinsics)
    simulator = CameraSimulator(
        basemap=flown_map,
        basemap_gsd_mpp=basemap_gsd_mpp,
        settings=render_settings,
        extrinsics=extrinsics,
    )

    # Key the frame cache on the trajectory's *contents*, not on id(): a list
    # that has been garbage collected can have its id reused, which would serve
    # frames rendered for a different flight path and produce a position error
    # that looks like a localization failure.
    frames_key = ("frames", variant_key, _trajectory_fingerprint(trajectory),
                  render_settings.seed, render_settings.width_px,
                  render_settings.exposure_s)
    if frames_key not in cache:
        LOG.info("rendering %d frames", len(trajectory))
        rendered = []
        for point in trajectory:
            frame = simulator.render(point)
            if frame is not None:
                rendered.append((point, frame))
        cache[frames_key] = rendered
        LOG.info("rendered %d usable frames", len(rendered))
    frames = cache[frames_key]

    # -- run the pipeline ---------------------------------------------------
    engine = LocalizationEngine(
        config, database, extractor=extractor, matcher=matcher
    )
    engine.initialize(origin)

    report = SimulationReport(
        label=label,
        trajectory=trajectory_statistics(trajectory),
        settings={
            "prewarp": config.prewarp.enabled,
            "photometric": config.photometric.enabled,
            "detector": config.features.detector,
            "basemap_gsd_mpp": basemap_gsd_mpp,
            "appearance_strength": appearance_strength,
            "camera": "%dx%d fov=%.0f exposure=1/%.0f" % (
                render_settings.width_px,
                render_settings.height_px,
                render_settings.horizontal_fov_deg,
                1.0 / render_settings.exposure_s,
            ),
            "false_fix_threshold_m": false_fix_threshold_m,
        },
    )

    wrong_tile = database.tiles[-1] if len(database) > 1 else None

    for index, (point, frame) in enumerate(frames):
        if max_cycles is not None and index >= max_cycles:
            break

        truth_geo = origin.to_geo(
            NEDPosition(point.north_m, point.east_m, origin.alt_m - point.height_m)
        )

        started = time.monotonic()
        fix = engine.localize(
            frame,
            roll_rad=point.roll_rad,
            pitch_rad=point.pitch_rad,
            yaw_rad=point.yaw_rad,
            height_hint_m=point.height_m,
        )
        elapsed = time.monotonic() - started

        report.cycles += 1
        report.processing_times_s.append(elapsed)
        band = _bank_band(point.roll_rad)
        good, total = report.by_bank.get(band, (0, 0))

        if fix.valid and fix.geo is not None:
            error = haversine_distance_m(truth_geo, fix.geo)
            report.valid_fixes += 1
            report.position_errors_m.append(error)
            if fix.position_ned is not None:
                report.position_error_vectors_m.append(
                    (
                        fix.position_ned.north_m - point.north_m,
                        fix.position_ned.east_m - point.east_m,
                        fix.position_ned.down_m - (origin.alt_m - point.height_m),
                    )
                )
            report.inliers.append(fix.quality.number_of_inliers)
            report.confidences.append(fix.confidence)
            if fix.yaw_rad is not None:
                report.yaw_errors_deg.append(
                    abs(math.degrees(wrap_pi(fix.yaw_rad - point.yaw_rad)))
                )
            if error > false_fix_threshold_m:
                report.false_fixes += 1
                # A false fix is worse than no fix: log it loudly so a run that
                # looks good on lock rate cannot hide one.
                LOG.warning(
                    "FALSE FIX at cycle %d: %.0f m from truth, confidence %.2f, "
                    "tile %s, %d inliers",
                    index,
                    error,
                    fix.confidence,
                    fix.tile_id,
                    fix.quality.number_of_inliers,
                )
            report.by_bank[band] = (good + 1, total + 1)
        else:
            reason = (
                fix.reason.value if isinstance(fix.reason, InvalidReason) else str(fix.reason)
            )
            report.failure_reasons[reason] = report.failure_reasons.get(reason, 0) + 1
            report.by_bank[band] = (good, total + 1)

        # Separation: how many inliers does the *wrong* tile attract?  Sampled
        # rather than measured every cycle, because it doubles the run time.
        if measure_separation and wrong_tile is not None and index % 5 == 0:
            report.wrong_tile_inliers.append(
                _wrong_tile_inliers(engine, frame, wrong_tile, config)
            )

    return report


def _wrong_tile_inliers(engine, frame, wrong_tile, config: Config) -> int:
    """Inliers this frame attracts against a deliberately wrong tile.

    The measurement that decides whether the system is safe: a matcher that
    finds plausible geometry on unrelated ground will eventually hand the EKF a
    confident position kilometres away, and no confidence threshold can catch
    it because the distributions overlap.
    """
    from ..mapdb.tile import TileFormatError

    try:
        prepared = engine.prepare_frame(frame)
        frame_features = engine.extractor.extract(prepared.image)
        tile_features = engine.database.features_for(wrong_tile)
        match = engine.matcher.match(frame_features, tile_features, tile_id=wrong_tile.tile_id)
        if len(match) == 0:
            return 0
        geometry = engine.verifier.verify(
            frame_points=match.frame_points,
            tile_points=match.tile_points,
            frame_size=prepared.size,
            match_scores=match.scores,
        )
        return geometry.num_inliers if geometry.ok else 0
    except (TileFormatError, RuntimeError, ValueError):
        return 0


def compare(reports: Sequence[SimulationReport]) -> str:
    """Render several runs as a comparison table."""
    header = (
        "%-26s %8s %8s %9s %9s %9s %8s %8s %7s"
        % (
            "run",
            "lock",
            "false",
            "err p50",
            "scatter",
            "bias|NE|",
            "inl p50",
            "sep",
            "ms",
        )
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        data = report.as_dict()
        bias = report.bias_ned_m
        bias_horiz = None if bias is None else math.hypot(bias[0], bias[1])
        lines.append(
            "%-26s %7.1f%% %7.1f%% %9s %9s %9s %8s %8s %7s"
            % (
                report.label[:26],
                100.0 * report.lock_rate,
                100.0 * report.false_fix_rate,
                _fmt(data["position_error_m"]["median"]),
                _fmt(data["scatter_about_bias_m"]),
                _fmt(bias_horiz),
                _fmt(data["inliers"]["median"], 0),
                _fmt(data["separation_true_over_wrong"], 0),
                _fmt(
                    data["processing_time_s"]["mean"] * 1000.0
                    if data["processing_time_s"]["mean"] is not None
                    else None,
                    0,
                ),
            )
        )
    # A large bias with small scatter is a bug, not a localization result; say
    # so here rather than leaving it to be read off the columns.
    for report in reports:
        bias = report.bias_ned_m
        scatter = report.scatter_about_bias_m
        if bias is None or scatter is None:
            continue
        horiz = math.hypot(bias[0], bias[1])
        if horiz > 10.0 and scatter < 0.25 * horiz:
            lines.append(
                "NOTE  %s: errors are a CONSTANT offset of N%+.1f E%+.1f D%+.1f m "
                "(scatter only %.1f m). That is a georeferencing or frame bug, "
                "not localization quality." % (report.label, bias[0], bias[1], bias[2], scatter)
            )
    return "\n".join(lines)


def _fmt(value, digits: int = 1) -> str:
    if value is None:
        return "-"
    return ("%%.%df" % digits) % value
