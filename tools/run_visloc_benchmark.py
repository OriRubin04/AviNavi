"""Run the real localization pipeline on real UAV imagery (UAV-VisLoc).

Everything else in this project was measured on a synthetic basemap matched
against a synthetic seasonal variant of itself.  This tool asks the question
the simulator cannot: does tier 1 lock when the live frame is a real drone
photograph and the map is a real satellite image taken on another date?

It is meant to be run unattended by the operator, then the small results file
it writes is sent back for analysis.  Usage (from the project folder)::

    .venv\\Scripts\\python.exe tools\\run_visloc_benchmark.py ^
        --data C:\\Users\\user\\Desktop\\AviNavi_data\\uav_visloc_sample\\UAV_VisLoc_example

Steps, all automatic:

1. Tile the GeoTIFF at the project's 0.5 m/px into a tile database (cached).
2. Calibrate what the dataset does not give: the camera's field of view (EXIF
   is stripped) and the heading convention of ``Phi1``.  Done on a handful of
   frames that are then **excluded** from scoring, so the result is not
   inflated by grading on the frames used to tune.
3. Fly the remaining frames in order through ``LocalizationEngine``, seeded
   once with the first frame's GPS and never again - the way a flight would be.
4. Write ``visloc_results.json`` (send this) and ``visloc_frames.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from avinavi.config import load_config  # noqa: E402
from avinavi.geo import BoundingBox, LocalOrigin  # noqa: E402
from avinavi.localization.engine import LocalizationEngine  # noqa: E402
from avinavi.mapdb.builder import _record_fingerprint, write_tile  # noqa: E402
from avinavi.mapdb.database import MapDatabase  # noqa: E402
from avinavi.types import GeoPoint  # noqa: E402

TILE_GSD_MPP = 0.5
TILE_SIZE_PX = 1024
TILE_OVERLAP = 0.25
FRAME_WIDTH_PX = 1280
FALSE_FIX_M = 60.0
TILE_DB_VERSION = "visloc-tiles-v1"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# -- data ---------------------------------------------------------------------


def find_flight(data: Path):
    """Return (flight_dir, csv_path, tif_path) for the first flight folder."""
    for sub in sorted(p for p in data.iterdir() if p.is_dir()):
        csvs = list(sub.glob("*.csv"))
        tifs = list(sub.glob("*.tif"))
        if csvs and tifs and (sub / "drone").is_dir():
            return sub, csvs[0], tifs[0]
    raise SystemExit("no flight folder (with drone/, *.csv, *.tif) found in %s" % data)


def read_frames(csv_path: Path, drone_dir: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            path = drone_dir / r["filename"]
            if not path.exists():
                continue
            rows.append(
                {
                    "num": int(r["num"]),
                    "path": path,
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "height": float(r["height"]),
                    "pitch_deg": float(r["Omega"]),
                    "roll_deg": float(r["Kappa"]),
                    "yaw_deg": float(r["Phi1"]),
                }
            )
    return rows


def load_frame(cv2, path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    scale = FRAME_WIDTH_PX / image.shape[1]
    return cv2.resize(image, (FRAME_WIDTH_PX, int(round(image.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)


# -- tile database ------------------------------------------------------------


def build_tiles(cv2, tif: Path, out_dir: Path, extractor):
    import rasterio
    from rasterio.enums import Resampling

    stamp = out_dir / "visloc_stamp.json"
    stamp_value = {"version": TILE_DB_VERSION, "tif": str(tif),
                   "tif_size": tif.stat().st_size, "gsd": TILE_GSD_MPP,
                   "tile": TILE_SIZE_PX, "overlap": TILE_OVERLAP,
                   "extractor": type(extractor).__name__}
    if stamp.exists():
        try:
            if json.loads(stamp.read_text()) == stamp_value:
                log("tile database already built, reusing %s" % out_dir)
                return MapDatabase.load(str(out_dir), feature_cache_size=64)
        except ValueError:
            pass
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    with rasterio.open(tif) as ds:
        west, south, east, north = ds.bounds
        lat0 = math.radians((north + south) / 2)
        width_m = (east - west) * 111320.0 * math.cos(lat0)
        height_m = (north - south) * 110574.0
        out_w = int(round(width_m / TILE_GSD_MPP))
        out_h = int(round(height_m / TILE_GSD_MPP))
        log("reading map %dx%d -> %dx%d px at %.2f m/px"
            % (ds.width, ds.height, out_w, out_h, TILE_GSD_MPP))
        data = ds.read([1, 2, 3], out_shape=(3, out_h, out_w),
                       resampling=Resampling.average)
    basemap = np.ascontiguousarray(np.transpose(data, (1, 2, 0))[:, :, ::-1])
    del data
    deg_per_px_lon = (east - west) / out_w
    deg_per_px_lat = (north - south) / out_h

    stride = int(round(TILE_SIZE_PX * (1 - TILE_OVERLAP)))
    rows0 = list(range(0, max(1, out_h - TILE_SIZE_PX + 1), stride))
    cols0 = list(range(0, max(1, out_w - TILE_SIZE_PX + 1), stride))
    total = len(rows0) * len(cols0)
    index = 0
    started = time.monotonic()
    for row0 in rows0:
        for col0 in cols0:
            patch = basemap[row0:row0 + TILE_SIZE_PX, col0:col0 + TILE_SIZE_PX]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if float(np.mean(gray < 5)) > 0.5:
                continue
            bounds = BoundingBox(
                west + col0 * deg_per_px_lon,
                north - (row0 + patch.shape[0]) * deg_per_px_lat,
                west + (col0 + patch.shape[1]) * deg_per_px_lon,
                north - row0 * deg_per_px_lat,
            )
            tile_id = "tile_%04d" % index
            directory = out_dir / tile_id
            metadata = write_tile(directory=str(directory), image=patch, bounds=bounds,
                                  tile_id=tile_id, terrain_alt_m=0.0)
            features = extractor.extract(gray, resize_max_side=None)
            features.image_size = (float(patch.shape[1]), float(patch.shape[0]))
            features.save(str(directory / "features.npz"))
            _record_fingerprint(metadata, extractor, features, gray, resize_max_side=None)
            index += 1
            if index % 20 == 0:
                done = (rows0.index(row0) * len(cols0) + cols0.index(col0) + 1)
                rate = (time.monotonic() - started) / done
                log("tiles %d/%d, about %.0f min left" % (done, total, rate * (total - done) / 60))
    database = MapDatabase.load(str(out_dir), feature_cache_size=64)
    database.write_index()
    stamp.write_text(json.dumps(stamp_value))
    log("built %d tiles" % len(database))
    return database


# -- engine -------------------------------------------------------------------


def make_engine(config, database, extractor, matcher, hfov_deg, frame_shape):
    h, w = frame_shape[:2]
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    config.camera.fx = fx
    config.camera.fy = fx
    config.camera.cx = w / 2.0
    config.camera.cy = h / 2.0
    config.camera.width = w
    config.camera.height = h
    return LocalizationEngine(config, database, extractor=extractor, matcher=matcher)


def yaw_rad(frame, mode):
    sign, offset = mode
    return math.radians(sign * frame["yaw_deg"] + offset)


def localize(engine, image, frame, mode):
    return engine.localize(
        image,
        roll_rad=math.radians(frame["roll_deg"]),
        pitch_rad=math.radians(frame["pitch_deg"]),
        yaw_rad=yaw_rad(frame, mode),
        height_hint_m=frame["height"],
    )


def score_hypothesis(config, database, extractor, matcher, origin, images, frames,
                     hfov, mode):
    """Lock count and median error over calibration frames, each seeded at truth."""
    errors = []
    for image, frame in zip(images, frames):
        engine = make_engine(config, database, extractor, matcher, hfov, image.shape)
        engine.initialize(origin, start_position=GeoPoint(frame["lat"], frame["lon"], 0.0))
        fix = localize(engine, image, frame, mode)
        if fix.valid and fix.geo is not None:
            err = haversine_m(fix.geo.lat_deg, fix.geo.lon_deg, frame["lat"], frame["lon"])
            if err < FALSE_FIX_M * 2:
                errors.append(err)
    med = statistics.median(errors) if errors else float("inf")
    return len(errors), med


def calibrate(config, database, extractor, matcher, origin, cv2, calib_frames):
    images = [load_frame(cv2, f["path"]) for f in calib_frames]
    pairs = [(i, f) for i, f in zip(images, calib_frames) if i is not None]
    images, frames = [p[0] for p in pairs], [p[1] for p in pairs]
    modes = [(s, o) for s in (1, -1) for o in (0, 90, 180, 270)]
    trials = []
    log("calibration 1/2: heading convention x field of view (%d frames)" % len(frames))
    best = None
    for hfov in (45.0, 65.0, 85.0):
        for mode in modes:
            n, med = score_hypothesis(config, database, extractor, matcher, origin,
                                      images[:4], frames[:4], hfov, mode)
            trials.append({"stage": 1, "hfov": hfov, "yaw_sign": mode[0],
                           "yaw_offset": mode[1], "locks": n, "median_err_m": med})
            log("  hfov %.0f yaw %+d%+d -> %d/4 locked, median %.1f m"
                % (hfov, mode[0], mode[1], n, med))
            if best is None or (n, -med) > (best[0], -best[1]):
                best = (n, med, hfov, mode)
    if best[0] == 0:
        log("calibration found no hypothesis that locks; continuing with defaults")
        return 65.0, (1, 0), trials
    mode = best[3]
    log("calibration 2/2: field of view sweep at yaw %+d%+d" % mode)
    best2 = None
    for hfov in range(35, 100, 5):
        n, med = score_hypothesis(config, database, extractor, matcher, origin,
                                  images, frames, float(hfov), mode)
        trials.append({"stage": 2, "hfov": hfov, "yaw_sign": mode[0],
                       "yaw_offset": mode[1], "locks": n, "median_err_m": med})
        log("  hfov %d -> %d/%d locked, median %.1f m" % (hfov, n, len(frames), med))
        if best2 is None or (n, -med) > (best2[0], -best2[1]):
            best2 = (n, med, float(hfov))
    return best2[2], mode, trials


# -- main ---------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="UAV_VisLoc_example folder")
    ap.add_argument("--out", default=None, help="output folder (default: <data>/../visloc_out)")
    ap.add_argument("--limit", type=int, default=0, help="score at most N frames (0 = all)")
    ap.add_argument("--calib-frames", type=int, default=8)
    ap.add_argument("--hfov", type=float, default=None, help="skip calibration: use this HFOV")
    ap.add_argument("--yaw-sign", type=int, default=1)
    ap.add_argument("--yaw-offset", type=float, default=0.0)
    ap.add_argument("--omega-sign", type=float, default=-1.0,
                    help="engine pitch = sign * Omega (measured: -1)")
    ap.add_argument("--mount-deg", type=float, default=2.0,
                    help="fixed camera tilt added to Kappa (measured: ~2 deg)")
    ap.add_argument("--tag", default="", help="suffix for the result file names")
    args = ap.parse_args(argv)

    import cv2
    import torch

    data = Path(args.data).resolve()
    out = Path(args.out).resolve() if args.out else data.parent / "visloc_out"
    out.mkdir(parents=True, exist_ok=True)
    flight_dir, csv_path, tif = find_flight(data)
    frames = read_frames(csv_path, flight_dir / "drone")
    # Measured on this dataset (docs/visloc-benchmark-plan.md): with yaw = Phi1+90
    # the engine's pitch acts cross-track, Omega's sign is inverted, and the
    # camera has a fixed ~2 deg tilt on the along-track (Kappa) axis.
    for f in frames:
        f["pitch_deg"] = args.omega_sign * f["pitch_deg"]
        f["roll_deg"] = f["roll_deg"] + args.mount_deg
    log("flight %s: %d frames, map %s" % (flight_dir.name, len(frames), tif.name))

    config = load_config()
    extractor = LocalizationEngine._build_extractor(config)
    matcher = LocalizationEngine._build_matcher(config)
    database = build_tiles(cv2, tif, out / ("tiles_" + flight_dir.name), extractor)
    config.map.database_path = database.path

    lats = [f["lat"] for f in frames]
    lons = [f["lon"] for f in frames]
    origin = LocalOrigin((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2, 0.0)

    step = max(1, len(frames) // max(1, args.calib_frames))
    calib_idx = set(range(step // 2, len(frames), step)) if args.hfov is None else set()
    calib = [frames[i] for i in sorted(calib_idx)][: args.calib_frames]
    calib_idx = {frames.index(f) for f in calib}
    if args.hfov is None:
        hfov, mode, trials = calibrate(config, database, extractor, matcher, origin, cv2, calib)
    else:
        hfov, mode, trials = args.hfov, (args.yaw_sign, args.yaw_offset), []
    log("using hfov %.1f deg, yaw = %+d * Phi1 %+g deg" % (hfov, mode[0], mode[1]))

    test = [f for i, f in enumerate(frames) if i not in calib_idx]
    if args.limit:
        test = test[: args.limit]
    first_image = load_frame(cv2, test[0]["path"])
    engine = make_engine(config, database, extractor, matcher, hfov, first_image.shape)
    engine.initialize(origin, start_position=GeoPoint(test[0]["lat"], test[0]["lon"], 0.0))

    per_frame = []
    started = time.monotonic()
    for k, frame in enumerate(test):
        image = first_image if k == 0 else load_frame(cv2, frame["path"])
        if image is None:
            continue
        t0 = time.monotonic()
        fix = localize(engine, image, frame, mode)
        dt = time.monotonic() - t0
        err = dn = de = None
        if fix.valid and fix.geo is not None:
            err = haversine_m(fix.geo.lat_deg, fix.geo.lon_deg, frame["lat"], frame["lon"])
            dn = math.radians(fix.geo.lat_deg - frame["lat"]) * 6371008.8
            de = (math.radians(fix.geo.lon_deg - frame["lon"]) * 6371008.8
                  * math.cos(math.radians(frame["lat"])))
        quality = getattr(fix, "quality", None)
        per_frame.append({
            "num": frame["num"], "valid": bool(fix.valid),
            "reason": getattr(fix.reason, "name", str(fix.reason)),
            "error_m": None if err is None else round(err, 2),
            "err_north_m": None if dn is None else round(dn, 2),
            "err_east_m": None if de is None else round(de, 2),
            "sigma_m": None if not math.isfinite(fix.position_sigma_m) else round(fix.position_sigma_m, 2),
            "inliers": getattr(quality, "inliers", getattr(quality, "inlier_count", None)),
            "tile": fix.tile_id, "ms": round(dt * 1000),
            "rectified_px": None,
        })
        if (k + 1) % 10 == 0 or k == len(test) - 1:
            locked = [p for p in per_frame if p["valid"]]
            rate = (time.monotonic() - started) / (k + 1)
            log("frame %d/%d: locked %d (%.0f%%), about %.0f min left"
                % (k + 1, len(test), len(locked), 100.0 * len(locked) / len(per_frame),
                   rate * (len(test) - k - 1) / 60))

    errs = sorted(p["error_m"] for p in per_frame if p["error_m"] is not None)
    good = [e for e in errs if e <= FALSE_FIX_M]

    def pct(v, q):
        return None if not v else round(v[min(len(v) - 1, int(q * len(v)))], 2)

    reasons = {}
    for p in per_frame:
        if not p["valid"]:
            reasons[p["reason"]] = reasons.get(p["reason"], 0) + 1
    results = {
        "dataset": "UAV-VisLoc example, flight %s" % flight_dir.name,
        "frames_scored": len(per_frame),
        "calibration_frames_excluded": len(calib_idx),
        "hfov_deg": hfov, "yaw_sign": mode[0], "yaw_offset_deg": mode[1],
        "omega_sign": args.omega_sign, "mount_deg": args.mount_deg,
        "tile_gsd_mpp": TILE_GSD_MPP, "frame_width_px": FRAME_WIDTH_PX,
        "lock_rate": round(len(errs) / max(1, len(per_frame)), 4),
        "good_fix_rate": round(len(good) / max(1, len(per_frame)), 4),
        "false_fixes_over_%dm" % FALSE_FIX_M: len(errs) - len(good),
        "error_m_p50_good": pct(good, 0.5), "error_m_p90_good": pct(good, 0.9),
        "error_m_max": errs[-1] if errs else None,
        "failure_reasons": reasons,
        "ms_per_frame_median": statistics.median([p["ms"] for p in per_frame]) if per_frame else None,
        "torch": torch.__version__, "threads": torch.get_num_threads(),
        "superpoint_keypoints": config.superpoint.max_num_keypoints,
        "max_candidate_tiles": config.lightglue.max_candidate_tiles,
        "calibration_trials": trials,
        "note": "ground truth is the dataset lat/lon, which behaves as the aircraft "
                "position; omega_sign and mount_deg were measured on this flight",
    }
    (out / ("visloc_results%s.json" % args.tag)).write_text(json.dumps(results, indent=2))
    with open(out / ("visloc_frames%s.csv" % args.tag), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_frame[0].keys()))
        w.writeheader()
        w.writerows(per_frame)
    log("DONE. lock %.1f%%, good %.1f%%, false %d, error p50 %s m"
        % (100 * results["lock_rate"], 100 * results["good_fix_rate"],
           results["false_fixes_over_%dm" % FALSE_FIX_M], results["error_m_p50_good"]))
    log("send this file: %s" % (out / ("visloc_results%s.json" % args.tag)))


if __name__ == "__main__":
    main()
