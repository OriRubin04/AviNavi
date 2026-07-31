"""Logging configuration and the per-solution CSV trace.

Two streams:

* a normal Python log (console + rotating file) for human-readable events;
* a CSV row per localization cycle carrying timestamp, visual position,
  confidence, match statistics, frame sequence and latency, plus what was
  actually transmitted.

The CSV is what makes a flight reviewable afterwards, and what lets a
development run be compared against GPS ground truth without the localizer ever
having seen it.
"""

from __future__ import annotations

import csv
import logging
import logging.handlers
import os
import threading
import time
from datetime import datetime
from typing import Optional, TextIO

from ..types import InvalidReason, NavigationSolution, VisualFix

LOG = logging.getLogger(__name__)

CSV_FIELDS = [
    "wall_time",
    "timestamp_unix",
    "timestamp_monotonic",
    "frame_sequence",
    "valid",
    "reason",
    "tile_id",
    "candidate_tiles",
    "lat_deg",
    "lon_deg",
    "alt_m",
    "north_m",
    "east_m",
    "down_m",
    "yaw_deg",
    "confidence",
    "matches",
    "inliers",
    "inlier_ratio",
    "reprojection_error_px",
    "matching_confidence",
    "spatial_spread",
    "position_sigma_m",
    "processing_time_s",
    "transmitted",
    "tx_message",
    "tx_north_m",
    "tx_east_m",
    "tx_down_m",
    "tx_propagated",
    "tx_age_s",
    "tx_quality",
    "tx_reset_counter",
    # Ground truth, present only when GPS logging was explicitly enabled.
    "gt_lat_deg",
    "gt_lon_deg",
    "gt_alt_m",
    "gt_error_m",
]


def setup_logging(config) -> str:
    """Configure the root logger.  Returns the log file path."""
    directory = config.directory
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "avinavi.log")

    level = getattr(logging, str(config.level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-28s %(message)s", "%H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=16 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)-28s %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    return path


class SolutionLogger:
    """Writes one CSV row per localization cycle.

    Thread-safe: the localization thread writes rows while the output thread
    updates the transmitted-solution columns.
    """

    def __init__(self, directory: str, enabled: bool = True, log_transmitted: bool = True) -> None:
        self.enabled = bool(enabled)
        self.log_transmitted = bool(log_transmitted)
        self.directory = directory
        self._handle: Optional[TextIO] = None
        self._writer = None
        self._lock = threading.Lock()
        self.path: Optional[str] = None

    @classmethod
    def from_config(cls, config) -> "SolutionLogger":
        return cls(
            directory=config.directory,
            enabled=config.csv_enabled,
            log_transmitted=config.log_transmitted_messages,
        )

    def open(self) -> Optional[str]:
        if not self.enabled:
            return None
        os.makedirs(self.directory, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.directory, "solutions_%s.csv" % stamp)
        self._handle = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._handle.flush()
        LOG.info("logging solutions to %s", self.path)
        return self.path

    def write(
        self,
        fix: VisualFix,
        solution: Optional[NavigationSolution] = None,
        transmitted: bool = False,
        message: str = "",
        frame_sequence: int = 0,
        ground_truth=None,
        ground_truth_error_m: Optional[float] = None,
    ) -> None:
        """Append a row for one localization cycle."""
        if not self.enabled or self._writer is None:
            return

        import math  # local import keeps the module import cheap

        row = {name: "" for name in CSV_FIELDS}
        row["wall_time"] = datetime.now().isoformat(timespec="milliseconds")
        row["timestamp_unix"] = "%.6f" % fix.timestamp_unix
        row["timestamp_monotonic"] = "%.6f" % fix.timestamp_monotonic
        row["frame_sequence"] = frame_sequence
        row["valid"] = int(fix.valid)
        row["reason"] = (
            fix.reason.value if isinstance(fix.reason, InvalidReason) else str(fix.reason)
        )
        row["tile_id"] = fix.tile_id or ""
        row["candidate_tiles"] = fix.candidate_tiles

        if fix.geo is not None:
            row["lat_deg"] = "%.7f" % fix.geo.lat_deg
            row["lon_deg"] = "%.7f" % fix.geo.lon_deg
            row["alt_m"] = "%.2f" % fix.geo.alt_m
        if fix.position_ned is not None:
            row["north_m"] = "%.3f" % fix.position_ned.north_m
            row["east_m"] = "%.3f" % fix.position_ned.east_m
            row["down_m"] = "%.3f" % fix.position_ned.down_m
        if fix.yaw_rad is not None:
            row["yaw_deg"] = "%.2f" % math.degrees(fix.yaw_rad)

        quality = fix.quality
        row["confidence"] = "%.4f" % quality.localization_confidence
        row["matches"] = quality.number_of_matches
        row["inliers"] = quality.number_of_inliers
        row["inlier_ratio"] = "%.4f" % quality.inlier_ratio
        row["reprojection_error_px"] = (
            "" if math.isnan(quality.reprojection_error_px)
            else "%.3f" % quality.reprojection_error_px
        )
        row["matching_confidence"] = "%.4f" % quality.matching_confidence
        row["spatial_spread"] = "%.4f" % quality.spatial_spread
        row["position_sigma_m"] = (
            "" if math.isnan(fix.position_sigma_m) else "%.3f" % fix.position_sigma_m
        )
        row["processing_time_s"] = "%.4f" % fix.processing_time_s

        if self.log_transmitted and solution is not None:
            row["transmitted"] = int(transmitted)
            row["tx_message"] = message
            row["tx_north_m"] = "%.3f" % solution.position_ned.north_m
            row["tx_east_m"] = "%.3f" % solution.position_ned.east_m
            row["tx_down_m"] = "%.3f" % solution.position_ned.down_m
            row["tx_propagated"] = int(solution.propagated)
            row["tx_age_s"] = "%.3f" % solution.age_s
            row["tx_quality"] = solution.quality_percent
            row["tx_reset_counter"] = solution.reset_counter

        if ground_truth is not None:
            row["gt_lat_deg"] = "%.7f" % ground_truth.lat_deg
            row["gt_lon_deg"] = "%.7f" % ground_truth.lon_deg
            row["gt_alt_m"] = "%.2f" % ground_truth.alt_m
        if ground_truth_error_m is not None:
            row["gt_error_m"] = "%.2f" % ground_truth_error_m

        with self._lock:
            self._writer.writerow(row)
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
                self._writer = None

    def __enter__(self) -> "SolutionLogger":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
