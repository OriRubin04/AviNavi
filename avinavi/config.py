"""Typed configuration for AviNavi, loaded from YAML.

Nothing in the codebase hard-codes a device path, baud rate, rate limit or
threshold: every tunable lives here and is sourced from ``config/default.yaml``
(or a file passed with ``--config``).  Values can additionally be overridden
from the command line with ``--set section.key=value``.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, get_args, get_origin

import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "default.yaml"
)


@dataclass
class MavlinkConfig:
    """MAVLink transport and identity settings.

    ``connection_type`` selects how ``device`` is interpreted:

    ``serial``  ``device`` is a tty (e.g. ``/dev/ttyAMA0``) opened at ``baudrate``
    ``udp``     ``device`` is ``udpout:host:port`` / ``udpin:host:port``
    ``tcp``     ``device`` is ``tcp:host:port``
    """

    connection_type: str = "serial"
    device: str = "/dev/ttyAMA0"
    baudrate: int = 921600
    #: System ID the companion computer *sends* as.  Must match the autopilot's
    #: system ID for ArduPilot to accept the external navigation data.
    system_id: int = 1
    #: MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY (197) is the semantically correct
    #: component ID for a VIO source; 191 (ONBOARD_COMPUTER) also works.
    component_id: int = 197
    #: System ID of the autopilot we expect to talk to.
    target_system: int = 1
    target_component: int = 1
    #: Seconds without a HEARTBEAT before the link is declared dead.
    heartbeat_timeout_s: float = 3.0
    #: Emit our own HEARTBEAT so the autopilot and GCS see the companion alive.
    send_heartbeat: bool = True
    heartbeat_hz: float = 1.0
    #: Request these streams from the autopilot (rates in Hz, 0 disables).
    request_attitude_hz: float = 10.0
    request_gps_for_ground_truth: bool = False


@dataclass
class InitialPositionConfig:
    """How the operator-selected starting position reaches the Pi.

    See ``docs/initial-position.md``.  The default path listens for the
    autopilot's ``GPS_GLOBAL_ORIGIN`` broadcast, which is what Mission Planner's
    "Set EKF origin here" map action produces.
    """

    #: "mavlink_origin" | "manual" | "auto"  ("auto" = mavlink with manual fallback)
    source: str = "mavlink_origin"
    #: Seconds to wait for the origin before giving up (0 = wait forever).
    wait_timeout_s: float = 0.0
    #: Ask the autopilot for its origin at startup instead of waiting passively.
    request_origin_on_start: bool = True
    #: Used when ``source`` is "manual" (or as the "auto" fallback).
    manual_lat_deg: Optional[float] = None
    manual_lon_deg: Optional[float] = None
    manual_alt_m: Optional[float] = None
    #: If the Pi holds the origin but the autopilot does not, offer to set it.
    push_origin_to_autopilot: bool = False


@dataclass
class CameraConfig:
    """Camera source and intrinsics."""

    #: "picamera2" | "opencv" | "directory"
    source: str = "opencv"
    #: Device index/URL for opencv, or the image folder for "directory".
    device: str = "0"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    #: Grayscale conversion happens in the extractor; keep this for debug dumps.
    save_frames_dir: Optional[str] = None
    #: Pinhole intrinsics in pixels for the configured resolution.  Required for
    #: metric altitude/scale reasoning; leave null to fall back on the tile
    #: ground resolution only.
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    #: Radial/tangential distortion (k1, k2, p1, p2, k3); empty = already undistorted.
    distortion: List[float] = field(default_factory=list)
    #: Directory-source replay speed multiplier (1.0 = real time by fps).
    replay_speed: float = 1.0


@dataclass
class ExtrinsicsRotation:
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class ExtrinsicsTranslation:
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0


@dataclass
class CameraExtrinsicsConfig:
    """Fixed camera-to-body transform.

    ``rotation`` is the roll/pitch/yaw (ZYX intrinsic, applied yaw->pitch->roll)
    that rotates the **camera optical frame** (X right, Y down, Z forward) into
    the **body FRD frame** (X forward, Y right, Z down).  The identity rotation
    therefore corresponds to a camera looking straight down with its image
    +X axis pointing along the aircraft's +Y (right) axis - i.e. the standard
    downward-facing mount.  ``translation`` is the camera's position in the body
    frame, metres, FRD.
    """

    rotation: ExtrinsicsRotation = field(default_factory=ExtrinsicsRotation)
    translation: ExtrinsicsTranslation = field(default_factory=ExtrinsicsTranslation)


@dataclass
class RatesConfig:
    """The two conceptual rates plus the camera capture rate."""

    camera_fps: float = 30.0
    visual_localization_hz: float = 5.0
    mavlink_odometry_hz: float = 20.0
    diagnostics_hz: float = 1.0


@dataclass
class SuperPointConfig:
    max_num_keypoints: int = 1024
    detection_threshold: float = 0.0005
    nms_radius: int = 4
    #: "cpu" | "cuda" | "auto"
    device: str = "auto"
    #: Longest image side fed to the network; smaller is faster on a Pi.
    resize_max_side: int = 640
    #: Minimum keypoints in a live frame before we bother matching.
    min_keypoints: int = 64


@dataclass
class LightGlueConfig:
    #: LightGlue weights variant; must match the extractor ("superpoint").
    features: str = "superpoint"
    depth_confidence: float = 0.95
    width_confidence: float = 0.99
    filter_threshold: float = 0.1
    device: str = "auto"
    #: Evaluate at most this many candidate tiles per localization cycle.
    max_candidate_tiles: int = 6


@dataclass
class GeometryConfig:
    """RANSAC/MAGSAC geometric verification settings."""

    #: "magsac" | "ransac" | "lmeds" - MAGSAC++ where OpenCV provides it.
    method: str = "magsac"
    #: "homography" (planar ground, the normal case) or "affine".
    model: str = "homography"
    reproj_threshold_px: float = 3.0
    max_iterations: int = 5000
    confidence: float = 0.999
    min_matches: int = 20
    min_inliers: int = 15
    min_inlier_ratio: float = 0.25
    max_reprojection_error_px: float = 4.0
    #: Reject degenerate homographies whose implied scale change is extreme.
    min_scale_ratio: float = 0.25
    max_scale_ratio: float = 4.0
    #: Minimum normalised spread of inliers across the image (0..1).
    min_spatial_spread: float = 0.15


@dataclass
class ConfidenceConfig:
    """Weights and gates for the localization confidence model."""

    weight_inlier_ratio: float = 0.4
    weight_inlier_count: float = 0.25
    weight_reprojection: float = 0.2
    weight_spread: float = 0.15
    #: Inlier count that saturates the count term.
    inlier_count_saturation: int = 120
    #: Reprojection error (px) at which the reprojection term hits zero.
    reprojection_scale_px: float = 4.0
    #: Below this the fix is marked invalid and nothing is published.
    min_confidence: float = 0.5
    #: Position sigma (m) reported at confidence 1.0 and at min_confidence.
    position_sigma_best_m: float = 1.0
    position_sigma_worst_m: float = 15.0
    yaw_sigma_best_rad: float = 0.05
    yaw_sigma_worst_rad: float = 0.5


@dataclass
class TileSearchConfig:
    """Which map tiles are considered for the next localization attempt."""

    #: Radius around the current position estimate, metres.
    search_radius_m: float = 300.0
    #: Radius used for the very first fix from the operator-selected position,
    #: which is typically less accurate than subsequent visual fixes.
    initial_search_radius_m: float = 500.0
    #: Grow the radius by this factor after each consecutive failure...
    radius_growth_factor: float = 1.5
    #: ...up to this ceiling, then give up and report a coverage failure.
    max_search_radius_m: float = 2000.0
    #: Consecutive failures before the search radius starts growing.
    failures_before_growth: int = 2
    #: Prefer tiles the previous fix matched against.
    prefer_last_matched_tile: bool = True


@dataclass
class MapConfig:
    """Location and interpretation of the georeferenced tile database."""

    database_path: str = "map_database"
    #: Ground altitude (metres, same datum as the origin) of the mapped terrain.
    #: Used to convert a map-plane fix into a 3-D position when no rangefinder
    #: or baro-derived height is available.
    terrain_alt_m: float = 0.0
    #: Cache at most this many tiles' features in RAM (8 GB Pi: ~200 is safe).
    feature_cache_size: int = 64


@dataclass
class PropagationConfig:
    """Forward propagation of the last visual fix between localization cycles."""

    enabled: bool = True
    #: Never propagate a fix older than this; after it the solution is invalid.
    max_age_s: float = 1.0
    #: Estimate velocity from consecutive fixes and use it while propagating.
    use_velocity: bool = True
    #: Velocity is only trusted if the two fixes are closer together than this.
    max_velocity_dt_s: float = 2.0
    #: Sanity ceiling on the estimated velocity (m/s).
    max_speed_mps: float = 40.0
    #: Inflate the reported position sigma by this many m per second of age.
    sigma_growth_mps: float = 2.0


@dataclass
class OutputConfig:
    """What we publish to ArduPilot, and whether we publish at all."""

    #: "odometry" | "vision_position_estimate" | "none"
    #: Exactly one authoritative external-navigation source; "none" is the
    #: log-only mode mandated by the safety progression.
    message: str = "odometry"
    #: Master arm for transmission.  False = compute and log, transmit nothing.
    send_to_autopilot: bool = False
    #: Populate ODOMETRY velocity fields (child frame BODY_FRD).
    send_velocity: bool = True
    #: Populate the ODOMETRY quality field from the confidence score.
    send_quality: bool = True
    #: Timestamp source: "unix" (UNIX epoch us) or "boot" (us since our start).
    timestamp_source: str = "unix"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    directory: str = "logs"
    #: Per-solution CSV trace (timestamp, position, confidence, latency, ...).
    csv_enabled: bool = True
    #: Write the raw ODOMETRY fields that were transmitted.
    log_transmitted_messages: bool = True
    console_diagnostics: bool = True


@dataclass
class SafetyConfig:
    """Guards that keep a bad visual solution away from the flight controller."""

    #: Refuse to transmit until this many consecutive good fixes have been seen.
    required_consecutive_good_fixes: int = 5
    #: Stop transmitting after this many consecutive invalid cycles.
    max_consecutive_failures: int = 10
    #: Hard ceiling on how far a single fix may jump from the previous one (m).
    max_position_jump_m: float = 100.0
    #: Require an acknowledged MAVLink link before transmitting.
    require_heartbeat: bool = True


@dataclass
class Config:
    """Root configuration object."""

    mode: str = "offline"
    mavlink: MavlinkConfig = field(default_factory=MavlinkConfig)
    initial_position: InitialPositionConfig = field(default_factory=InitialPositionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    camera_extrinsics: CameraExtrinsicsConfig = field(default_factory=CameraExtrinsicsConfig)
    rates: RatesConfig = field(default_factory=RatesConfig)
    superpoint: SuperPointConfig = field(default_factory=SuperPointConfig)
    lightglue: LightGlueConfig = field(default_factory=LightGlueConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    tile_search: TileSearchConfig = field(default_factory=TileSearchConfig)
    map: MapConfig = field(default_factory=MapConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


class ConfigError(ValueError):
    """Raised when a configuration file cannot be interpreted."""


def _coerce(value: Any, target_type: Any, path: str) -> Any:
    """Convert a YAML scalar into the type the dataclass field declares."""
    origin = get_origin(target_type)
    if origin is not None:
        args = get_args(target_type)
        # Optional[T] / Union[T, None]
        if type(None) in args:
            if value is None:
                return None
            inner = [a for a in args if a is not type(None)]
            return _coerce(value, inner[0], path)
        if origin in (list, List):
            if not isinstance(value, (list, tuple)):
                raise ConfigError("%s: expected a list, got %r" % (path, value))
            item_type = args[0] if args else Any
            return [_coerce(v, item_type, "%s[%d]" % (path, i)) for i, v in enumerate(value)]
        if origin in (dict, Dict):
            return dict(value)
        return value

    if target_type is Any or target_type is None:
        return value
    if value is None:
        return None
    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "on", "1"):
                return True
            if lowered in ("false", "no", "off", "0"):
                return False
        raise ConfigError("%s: expected a boolean, got %r" % (path, value))
    if target_type is float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError("%s: expected a number, got %r" % (path, value)) from exc
    if target_type is int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError("%s: expected an integer, got %r" % (path, value)) from exc
    if target_type is str:
        return str(value)
    return value


def _from_mapping(cls: Any, data: Any, path: str = "") -> Any:
    """Recursively build a dataclass instance from a nested mapping."""
    if not is_dataclass(cls):
        return data
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError("%s: expected a mapping, got %r" % (path or "<root>", data))

    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(
            "%s: unknown configuration key(s): %s"
            % (path or "<root>", ", ".join(unknown))
        )

    kwargs = {}
    for name, f in known.items():
        if name not in data:
            continue
        child_path = "%s.%s" % (path, name) if path else name
        if is_dataclass(f.type) or (
            isinstance(f.type, str) and f.type in _DATACLASS_BY_NAME
        ):
            child_cls = f.type if is_dataclass(f.type) else _DATACLASS_BY_NAME[f.type]
            kwargs[name] = _from_mapping(child_cls, data[name], child_path)
        else:
            kwargs[name] = _coerce(data[name], f.type, child_path)
    return cls(**kwargs)


#: dataclass fields annotated as strings (``from __future__ import annotations``)
#: need to be resolved by name.
_DATACLASS_BY_NAME = {
    "MavlinkConfig": MavlinkConfig,
    "InitialPositionConfig": InitialPositionConfig,
    "CameraConfig": CameraConfig,
    "CameraExtrinsicsConfig": CameraExtrinsicsConfig,
    "ExtrinsicsRotation": ExtrinsicsRotation,
    "ExtrinsicsTranslation": ExtrinsicsTranslation,
    "RatesConfig": RatesConfig,
    "SuperPointConfig": SuperPointConfig,
    "LightGlueConfig": LightGlueConfig,
    "GeometryConfig": GeometryConfig,
    "ConfidenceConfig": ConfidenceConfig,
    "TileSearchConfig": TileSearchConfig,
    "MapConfig": MapConfig,
    "PropagationConfig": PropagationConfig,
    "OutputConfig": OutputConfig,
    "LoggingConfig": LoggingConfig,
    "SafetyConfig": SafetyConfig,
}


def apply_overrides(data: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``section.key=value`` CLI overrides onto a raw config mapping."""
    data = copy.deepcopy(data)
    for override in overrides:
        if "=" not in override:
            raise ConfigError("override %r is not of the form key.path=value" % override)
        key, _, raw = override.partition("=")
        parts = [p for p in key.strip().split(".") if p]
        if not parts:
            raise ConfigError("override %r has an empty key" % override)
        # YAML-parse the value so numbers, booleans and null work naturally.
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        cursor = data
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[parts[-1]] = value
    return data


def load_config(
    path: Optional[str] = None, overrides: Optional[List[str]] = None
) -> Config:
    """Load configuration from ``path`` (defaults to ``config/default.yaml``).

    Missing sections fall back to the dataclass defaults, so a config file only
    has to state what it changes.
    """
    path = path or DEFAULT_CONFIG_PATH
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    else:
        if path != DEFAULT_CONFIG_PATH:
            raise ConfigError("configuration file not found: %s" % path)
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("%s: top level of the config must be a mapping" % path)
    if overrides:
        raw = apply_overrides(raw, list(overrides))
    config = _from_mapping(Config, raw)
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Reject configurations that cannot produce a safe system."""
    if config.mode not in ("offline", "bench", "flight"):
        raise ConfigError("mode must be one of offline, bench, flight")
    if config.mavlink.connection_type not in ("serial", "udp", "tcp"):
        raise ConfigError("mavlink.connection_type must be serial, udp or tcp")
    if config.output.message not in ("odometry", "vision_position_estimate", "none"):
        raise ConfigError(
            "output.message must be odometry, vision_position_estimate or none"
        )
    if config.output.timestamp_source not in ("unix", "boot"):
        raise ConfigError("output.timestamp_source must be unix or boot")
    if config.initial_position.source not in ("mavlink_origin", "manual", "auto"):
        raise ConfigError(
            "initial_position.source must be mavlink_origin, manual or auto"
        )
    if config.initial_position.source in ("manual", "auto"):
        manual = config.initial_position
        has_manual = manual.manual_lat_deg is not None and manual.manual_lon_deg is not None
        if manual.source == "manual" and not has_manual:
            raise ConfigError(
                "initial_position.source is 'manual' but manual_lat_deg/manual_lon_deg are unset"
            )
    if config.geometry.model not in ("homography", "affine"):
        raise ConfigError("geometry.model must be homography or affine")
    if config.rates.visual_localization_hz <= 0:
        raise ConfigError("rates.visual_localization_hz must be > 0")
    if config.rates.mavlink_odometry_hz <= 0:
        raise ConfigError("rates.mavlink_odometry_hz must be > 0")
    if config.rates.mavlink_odometry_hz < config.rates.visual_localization_hz:
        raise ConfigError(
            "rates.mavlink_odometry_hz must be >= rates.visual_localization_hz"
        )
    if not 0.0 < config.confidence.min_confidence <= 1.0:
        raise ConfigError("confidence.min_confidence must be in (0, 1]")
    if config.tile_search.search_radius_m <= 0:
        raise ConfigError("tile_search.search_radius_m must be > 0")
    if config.tile_search.max_search_radius_m < config.tile_search.search_radius_m:
        raise ConfigError(
            "tile_search.max_search_radius_m must be >= tile_search.search_radius_m"
        )
    if config.propagation.max_age_s <= 0:
        raise ConfigError("propagation.max_age_s must be > 0")
