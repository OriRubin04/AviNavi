"""``avinavi`` - the command line interface.

    avinavi run --mode offline          the pipeline against recorded frames
    avinavi run --mode bench            hardware-in-the-loop on a stationary aircraft
    avinavi run --mode flight           the real thing
    avinavi build-map ...               slice a source raster into tiles
    avinavi extract-features ...        precompute SuperPoint features for tiles
    avinavi map-info ...                inspect a tile database
    avinavi mavlink-check                MAVLink diagnostics, no vision involved
    avinavi show-config                  the fully resolved configuration
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional

from ..config import Config, ConfigError, load_config
from ..diagnostics.logging_setup import setup_logging

LOG = logging.getLogger("avinavi")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avinavi",
        description="GPS-denied visual localization and navigation for ArduPilot",
    )
    parser.add_argument("--config", help="path to a YAML config (default: config/default.yaml)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set mavlink.device=/dev/ttyACM0",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run the localization system")
    run.add_argument(
        "--mode",
        choices=["offline", "bench", "flight"],
        help="operating mode (overrides the config)",
    )
    run.add_argument("--map", help="path to the map database (overrides the config)")
    run.add_argument("--camera", help="camera device or image directory")
    run.add_argument(
        "--send",
        action="store_true",
        help="enable transmission of external navigation to the autopilot",
    )
    run.add_argument("--max-cycles", type=int, help="stop after this many localization cycles")
    run.add_argument("--duration", type=float, help="stop after this many seconds")
    run.set_defaults(func=command_run)

    build_map = subparsers.add_parser(
        "build-map", help="slice a georeferenced raster into map tiles"
    )
    build_map.add_argument("source", help="source image or GeoTIFF")
    build_map.add_argument("output", help="output map database directory")
    build_map.add_argument(
        "--bounds",
        help="WGS-84 bounds of the source as west,south,east,north (required for "
        "a plain image; read from the file for a GeoTIFF)",
    )
    build_map.add_argument("--tile-size", type=int, default=1024, help="tile size in pixels")
    build_map.add_argument("--overlap", type=float, default=0.25, help="tile overlap fraction")
    build_map.add_argument(
        "--terrain-alt", type=float, default=0.0, help="mean terrain altitude in metres"
    )
    build_map.add_argument(
        "--no-features",
        action="store_true",
        help="only cut tiles; run extract-features separately",
    )
    build_map.set_defaults(func=command_build_map)

    extract = subparsers.add_parser(
        "extract-features", help="precompute SuperPoint features for every tile"
    )
    extract.add_argument("database", help="map database directory")
    extract.add_argument("--force", action="store_true", help="recompute existing features")
    extract.add_argument(
        "--max-keypoints",
        type=int,
        help="keypoint budget per tile (defaults to 4x the live-frame budget)",
    )
    extract.add_argument(
        "--resize-max-side",
        type=int,
        default=0,
        help="downscale tiles before extraction; 0 uses full resolution",
    )
    extract.set_defaults(func=command_extract_features)

    map_info = subparsers.add_parser("map-info", help="summarise a map tile database")
    map_info.add_argument("database", nargs="?", help="map database directory")
    map_info.set_defaults(func=command_map_info)

    mavlink_check = subparsers.add_parser(
        "mavlink-check", help="MAVLink link diagnostics without running the vision stack"
    )
    mavlink_check.add_argument(
        "--duration", type=float, default=15.0, help="how long to observe the link"
    )
    mavlink_check.add_argument(
        "--send-test-odometry",
        action="store_true",
        help="send a zero-position ODOMETRY so ArduPilot's reception can be verified",
    )
    mavlink_check.set_defaults(func=command_mavlink_check)

    show_config = subparsers.add_parser("show-config", help="print the resolved configuration")
    show_config.set_defaults(func=command_show_config)

    return parser


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def command_run(args, config: Config) -> int:
    from ..app.runner import Runner, RunnerError

    setup_logging(config.logging)
    runner = Runner(config)
    try:
        return runner.run(max_cycles=args.max_cycles, duration_s=args.duration)
    except RunnerError as exc:
        LOG.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOG.info("interrupted")
        runner.stop()
        return 0


def command_build_map(args, config: Config) -> int:
    from ..geo import BoundingBox
    from ..mapdb.builder import TilingSpec, slice_image

    setup_logging(config.logging)

    bounds = None
    if args.bounds:
        try:
            west, south, east, north = (float(v) for v in args.bounds.split(","))
        except ValueError:
            LOG.error("--bounds must be four comma-separated numbers: west,south,east,north")
            return 2
        try:
            bounds = BoundingBox(west, south, east, north)
        except ValueError as exc:
            LOG.error("%s", exc)
            return 2

    spec = TilingSpec(tile_size_px=args.tile_size, overlap=args.overlap)
    try:
        tiles = slice_image(
            source_image_path=args.source,
            output_dir=args.output,
            bounds=bounds,
            spec=spec,
            terrain_alt_m=args.terrain_alt,
        )
    except (RuntimeError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2

    print("wrote %d tiles to %s" % (len(tiles), args.output))
    if args.no_features:
        print("run 'avinavi extract-features %s' before flying" % args.output)
        return 0

    extract_args = argparse.Namespace(
        database=args.output, force=False, max_keypoints=None, resize_max_side=0
    )
    return command_extract_features(extract_args, config)


def command_extract_features(args, config: Config) -> int:
    from ..features.superpoint import SuperPointExtractor
    from ..mapdb.builder import build_features

    setup_logging(config.logging)

    # Map tiles cover far more ground than a single frame, so they deserve a
    # bigger keypoint budget than the live camera path uses.
    max_keypoints = args.max_keypoints or (config.superpoint.max_num_keypoints * 4)
    extractor = SuperPointExtractor(
        max_num_keypoints=max_keypoints,
        detection_threshold=config.superpoint.detection_threshold,
        nms_radius=config.superpoint.nms_radius,
        device=config.superpoint.device,
        resize_max_side=None,
    )
    try:
        count = build_features(
            database_path=args.database,
            extractor=extractor,
            force=args.force,
            resize_max_side=args.resize_max_side or None,
        )
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 2

    print("extracted features for %d tiles in %s" % (count, args.database))
    return 0


def command_map_info(args, config: Config) -> int:
    from ..mapdb.builder import summarise
    from ..mapdb.database import MapCoverageError

    path = args.database or config.map.database_path
    try:
        info = summarise(path)
    except MapCoverageError as exc:
        LOG.error("%s", exc)
        return 2
    print(json.dumps(info, indent=2))
    if info["tiles_missing_features"]:
        print(
            "\nWARNING: %d tiles have no precomputed features; run "
            "'avinavi extract-features %s'"
            % (len(info["tiles_missing_features"]), path),
            file=sys.stderr,
        )
        return 1
    return 0


def command_mavlink_check(args, config: Config) -> int:
    """Verify the link, the inbound streams and (optionally) ODOMETRY reception.

    This is the Mode 2 bring-up tool: it exercises everything except the vision
    stack, so a MAVLink or coordinate-frame problem can be found without a
    camera or a map.
    """
    import time

    from ..mavlink.connection import MavlinkConnection, MavlinkConnectionError
    from ..mavlink.external_navigation import ExternalNavigationPublisher
    from ..mavlink.initial_position import InitialPositionProvider
    from ..types import Attitude, NavigationSolution, NEDPosition

    setup_logging(config.logging)

    connection = MavlinkConnection.from_config(config.mavlink)
    try:
        connection.connect(wait_for_heartbeat=True, timeout_s=30.0)
    except MavlinkConnectionError as exc:
        LOG.error("%s", exc)
        return 2

    provider = InitialPositionProvider.from_config(config.initial_position, connection)
    provider.start_listening()

    publisher = ExternalNavigationPublisher.from_config(connection, config.output)
    if args.send_test_odometry:
        publisher.send_to_autopilot = True
        LOG.warning(
            "sending test %s at the configured rate with a zero position - the "
            "aircraft must be stationary and disarmed",
            config.output.message,
        )

    interval = 1.0 / config.rates.mavlink_odometry_hz
    deadline = time.monotonic() + args.duration
    next_send = time.monotonic()
    sent = 0
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if args.send_test_odometry and now >= next_send:
                next_send = now + interval
                solution = NavigationSolution(
                    timestamp_monotonic=now,
                    timestamp_unix=time.time(),
                    position_ned=NEDPosition(0.0, 0.0, 0.0),
                    attitude=Attitude(1.0, 0.0, 0.0, 0.0),
                    velocity_body_frd=(0.0, 0.0, 0.0),
                    position_sigma_m=1.0,
                    confidence=0.9,
                )
                if publisher.publish(solution):
                    sent += 1
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

    state = connection.state
    report = {
        "connected": connection.connected,
        "heartbeat_age_s": round(connection.heartbeat_age_s, 2)
        if connection.heartbeat_age_s != float("inf")
        else None,
        "received": connection.received_counts,
        "sent": connection.sent_counts,
        "attitude": (
            {
                "roll_deg": round(state.roll_rad * 57.2957795, 2),
                "pitch_deg": round(state.pitch_rad * 57.2957795, 2),
                "yaw_deg": round(state.yaw_rad * 57.2957795, 2),
            }
            if state.roll_rad is not None
            else None
        ),
        "ekf_origin": (
            {
                "lat": provider.position.geo.lat_deg,
                "lon": provider.position.geo.lon_deg,
                "alt_m": provider.position.geo.alt_m,
                "source": provider.position.source,
            }
            if provider.position is not None
            else None
        ),
        "test_odometry_sent": sent,
    }
    connection.close()

    print(json.dumps(report, indent=2))
    if not report["connected"]:
        print("\nFAIL: no heartbeat", file=sys.stderr)
        return 1
    if report["ekf_origin"] is None:
        print(
            "\nNOTE: no EKF origin yet. In Mission Planner right-click the map "
            "and choose Set Home Here -> Set EKF Origin Here.",
            file=sys.stderr,
        )
    return 0


def command_show_config(args, config: Config) -> int:
    from dataclasses import asdict

    print(json.dumps(asdict(config), indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = list(args.overrides)
    # Command-line conveniences are just overrides under the hood, so there is
    # exactly one place where configuration is resolved.
    for attribute, key in (
        ("mode", "mode"),
        ("map", "map.database_path"),
        ("camera", "camera.device"),
    ):
        value = getattr(args, attribute, None)
        if value:
            overrides.append("%s=%s" % (key, value))
    if getattr(args, "send", False):
        overrides.append("output.send_to_autopilot=true")

    try:
        config = load_config(args.config, overrides)
    except ConfigError as exc:
        print("configuration error: %s" % exc, file=sys.stderr)
        return 2

    # An offline run has no autopilot to talk to, so make that explicit rather
    # than letting it fail later on a serial port that is not there.
    if getattr(args, "command", None) == "run" and config.mode == "offline":
        if config.output.send_to_autopilot:
            print(
                "offline mode cannot transmit to an autopilot; "
                "use --mode bench or --mode flight",
                file=sys.stderr,
            )
            return 2

    return args.func(args, config)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
