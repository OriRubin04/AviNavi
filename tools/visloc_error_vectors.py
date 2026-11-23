"""Signed error vectors on UAV-VisLoc: body fix AND image-centre ground point vs truth.

Each sampled frame is seeded at truth independently (isolates matching error).
Usage: python diag_offset.py <repo> <data> <out_csv> <every_n> <hfov> [<hfov> ...]
"""
import csv, math, sys, time
from pathlib import Path

repo, data, out_csv, every = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
hfovs = [float(x) for x in sys.argv[5:]]
sys.path.insert(0, str(repo)); sys.path.insert(0, str(repo / "tools"))

import run_visloc_benchmark as B  # noqa
from avinavi.localization import pose as P  # noqa
from avinavi.geo import LocalOrigin  # noqa
from avinavi.types import GeoPoint  # noqa
import cv2  # noqa

STASH = {}
_orig = P.PoseSolver.solve
def solve(self, geometry, tile, *a, **k):
    sol = _orig(self, geometry, tile, *a, **k)
    STASH["sol"], STASH["tile"] = sol, tile
    return sol
P.PoseSolver.solve = solve

def ne(lat, lon, lat0, lon0):
    n = math.radians(lat - lat0) * 6371008.8
    e = math.radians(lon - lon0) * 6371008.8 * math.cos(math.radians(lat0))
    return n, e

flight_dir, csv_path, tif = B.find_flight(data)
frames = B.read_frames(csv_path, flight_dir / "drone")
config = B.load_config()
ext = B.LocalizationEngine._build_extractor(config)
mat = B.LocalizationEngine._build_matcher(config)
db = B.build_tiles(cv2, tif, data.parent / "visloc_out" / ("tiles_" + flight_dir.name), ext)
config.map.database_path = db.path
lats = [f["lat"] for f in frames]; lons = [f["lon"] for f in frames]
origin = LocalOrigin((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2, 0.0)
mode = (1, 90.0)

fields = ["hfov", "num", "valid", "reason", "lat", "lon", "height", "pitch", "roll", "yaw_in",
          "yaw_fit", "range_m", "body_dn", "body_de", "gnd_dn", "gnd_de", "track_deg"]
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fields); w.writeheader()
    for hfov in hfovs:
        for i in range(2, len(frames) - 1, every):
            f = frames[i]
            pn, pe = ne(frames[i + 1]["lat"], frames[i + 1]["lon"], frames[i - 1]["lat"], frames[i - 1]["lon"])
            img = B.load_frame(cv2, f["path"])
            if img is None:
                continue
            eng = B.make_engine(config, db, ext, mat, hfov, img.shape)
            eng.initialize(origin, start_position=GeoPoint(f["lat"], f["lon"], 0.0))
            STASH.clear()
            t = time.monotonic()
            fix = B.localize(eng, img, f, mode)
            row = {"hfov": hfov, "num": f["num"], "valid": fix.valid, "reason": fix.reason.name,
                   "lat": f["lat"], "lon": f["lon"], "height": f["height"], "pitch": f["pitch_deg"],
                   "roll": f["roll_deg"], "yaw_in": f["yaw_deg"],
                   "track_deg": round(math.degrees(math.atan2(pe, pn)) % 360, 1)}
            if fix.valid and "sol" in STASH:
                s, tile = STASH["sol"], STASH["tile"]
                g = tile.pixel_to_geo(*s.tile_pixel)
                bn, be = ne(fix.geo.lat_deg, fix.geo.lon_deg, f["lat"], f["lon"])
                gn, ge = ne(g.lat_deg, g.lon_deg, f["lat"], f["lon"])
                row.update(yaw_fit=round(math.degrees(s.yaw_rad) % 360, 2), range_m=round(s.range_m, 1),
                           body_dn=round(bn, 2), body_de=round(be, 2), gnd_dn=round(gn, 2), gnd_de=round(ge, 2))
            w.writerow(row); fh.flush()
            print(time.strftime("%H:%M:%S"), hfov, f["num"], fix.reason.name,
                  row.get("gnd_dn"), row.get("gnd_de"), "%.1fs" % (time.monotonic() - t), flush=True)
print("DONE", flush=True)
