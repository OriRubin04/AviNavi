"""Per-stage timing on UAV-VisLoc, and how each speed knob trades time for accuracy.

Flies N consecutive frames sequentially (seeded once, like the benchmark) under
several configurations, and records for every frame the time spent in each
stage, how many map tiles were matched, which candidate rank won, and the error.

Usage (from the project folder)::

    .venv\\Scripts\\python.exe -I tools\\visloc_timing.py --data <UAV_VisLoc_example> --frames 150

Writes visloc_timing.json (summary, send this) and visloc_timing.csv to visloc_out.
Note: tile features were built once at the default keypoint budget; the keypoint
variants change only the live frame's budget.
"""
import argparse, copy, csv, ctypes, json, math, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import run_visloc_benchmark as B  # noqa: E402
from avinavi.geo import LocalOrigin  # noqa: E402
from avinavi.types import GeoPoint  # noqa: E402

VARIANTS = [
    ("baseline", {}),
    ("keypoints_512", {"superpoint.max_num_keypoints": 512}),
    ("keypoints_256", {"superpoint.max_num_keypoints": 256}),
    ("tiles_2", {"lightglue.max_candidate_tiles": 2}),
    ("tiles_1", {"lightglue.max_candidate_tiles": 1}),
    ("resize_480", {"superpoint.resize_max_side": 480}),
    ("resize_800", {"superpoint.resize_max_side": 800}),
    ("resize_1024", {"superpoint.resize_max_side": 1024}),
    ("native_tiles_2", {"superpoint.resize_max_side": 768, "lightglue.max_candidate_tiles": 2}),
    ("native_tiles_2_kp512", {"superpoint.resize_max_side": 768, "lightglue.max_candidate_tiles": 2,
                              "superpoint.max_num_keypoints": 512}),
    ("native_tiles_3", {"superpoint.resize_max_side": 768, "lightglue.max_candidate_tiles": 3}),
]


def peak_rss_mb():
    if sys.platform != "win32":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    k = ctypes.windll.kernel32; p = ctypes.windll.psapi
    p.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    k.GetCurrentProcess.restype = ctypes.c_void_p
    p.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2**20


def set_path(cfg, dotted, value):
    obj = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def timed(times, key, fn):
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            times[key] = times.get(key, 0.0) + (time.perf_counter() - t)
            times[key + "_n"] = times.get(key + "_n", 0) + 1
    return wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument("--no-aim", dest="aim", action="store_false",
                    help="do not aim the tile search at truth before each frame")
    args = ap.parse_args()

    import cv2
    import torch
    data = Path(args.data).resolve()
    out = data.parent / "visloc_out"
    flight_dir, csv_path, tif = B.find_flight(data)
    frames = B.read_frames(csv_path, flight_dir / "drone")
    for f in frames:  # measured conventions, see docs/visloc-benchmark-plan.md
        f["pitch_deg"], f["roll_deg"] = -f["pitch_deg"], f["roll_deg"] + 2.0
    base_cfg = B.load_config()
    ext0 = B.LocalizationEngine._build_extractor(base_cfg)
    db = B.build_tiles(cv2, tif, out / ("tiles_" + flight_dir.name), ext0)
    lats = [f["lat"] for f in frames]; lons = [f["lon"] for f in frames]
    origin = LocalOrigin((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2, 0.0)
    test = frames[args.start: args.start + args.frames]
    images = [B.load_frame(cv2, f["path"]) for f in test]

    rows, summary = [], {"frames": len(test), "start": args.start, "torch_threads": torch.get_num_threads(),
                         "aim_at_truth": args.aim, "variants": {}}
    for name, overrides in VARIANTS:
        if args.only and name not in args.only:
            continue
        cfg = copy.deepcopy(base_cfg)
        cfg.map.database_path = db.path
        for k, v in overrides.items():
            set_path(cfg, k, v)
        ext = B.LocalizationEngine._build_extractor(cfg)
        mat = B.LocalizationEngine._build_matcher(cfg)
        eng = B.make_engine(cfg, db, ext, mat, args.hfov, images[0].shape)
        eng.initialize(origin, start_position=GeoPoint(test[0]["lat"], test[0]["lon"], 0.0))
        times = {}
        eng.prepare_frame = timed(times, "rectify", eng.prepare_frame)
        eng.extractor.extract = timed(times, "superpoint", eng.extractor.extract)
        eng.matcher.match = timed(times, "lightglue", eng.matcher.match)
        eng.verifier.verify = timed(times, "geometry", eng.verifier.verify)
        eng.database.features_for = timed(times, "tile_load", eng.database.features_for)
        errs, totals = [], []
        for k, (f, img) in enumerate(zip(test, images)):
            times.clear()
            if args.aim:
                # Stand-in for tier 2 (optical flow), which this test does not run:
                # without it nothing moves the search after a miss, and a variant
                # that misses the first frames never re-acquires.
                eng.guide_search_center(GeoPoint(f["lat"], f["lon"], 0.0))
            t = time.perf_counter()
            fix = B.localize(eng, img, f, (1, 90.0))
            total = time.perf_counter() - t
            err = B.haversine_m(fix.geo.lat_deg, fix.geo.lon_deg, f["lat"], f["lon"]) if fix.valid else None
            row = {"variant": name, "num": f["num"], "valid": fix.valid, "reason": fix.reason.name,
                   "error_m": None if err is None else round(err, 2), "total_ms": round(total * 1000),
                   "tiles_matched": times.get("lightglue_n", 0),
                   "winner_rank": getattr(eng, "_last_candidate_rank", None)}
            for s in ("rectify", "superpoint", "lightglue", "geometry", "tile_load"):
                row[s + "_ms"] = round(times.get(s, 0.0) * 1000)
            rows.append(row)
            if k > 0:  # first frame includes warm-up
                totals.append(row)
            if err is not None:
                errs.append(err)
        errs.sort()
        def med(key):
            return round(statistics.median(r[key] for r in totals), 0)
        ranks = [r["winner_rank"] for r in totals if r["valid"]]
        summary["variants"][name] = {
            "overrides": overrides,
            "lock_rate": round(len(errs) / len(test), 3),
            "error_p50_m": round(errs[len(errs) // 2], 2) if errs else None,
            "error_p90_m": round(errs[int(len(errs) * 0.9)], 2) if errs else None,
            "false_over_60m": sum(e > 60 for e in errs),
            "total_ms_p50": med("total_ms"),
            "total_ms_p90": round(sorted(r["total_ms"] for r in totals)[int(len(totals) * 0.9)]),
            "stage_ms_p50": {s: med(s + "_ms") for s in ("rectify", "superpoint", "lightglue", "geometry", "tile_load")},
            "tiles_matched_mean": round(statistics.mean(r["tiles_matched"] for r in totals), 2),
            "lightglue_ms_per_tile": round(sum(r["lightglue_ms"] for r in totals) / max(1, sum(r["tiles_matched"] for r in totals))),
            "winner_rank_counts": {str(k): ranks.count(k) for k in sorted(set(ranks), key=str)},
            "peak_rss_mb_so_far": round(peak_rss_mb()),
        }
        print(time.strftime("%H:%M:%S"), name, json.dumps(summary["variants"][name]), flush=True)
        (out / "visloc_timing.json").write_text(json.dumps(summary, indent=2))
    with open(out / "visloc_timing.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("DONE", out / "visloc_timing.json", flush=True)


if __name__ == "__main__":
    main()
