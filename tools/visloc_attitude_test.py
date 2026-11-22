"""Test the label hypotheses in the full pipeline: Omega/Kappa swapped (with signs)
and a fixed camera nose-down mount, folded into the pitch passed to the engine.
Usage: python diag2.py <repo> <data> <out_csv> <every_n> <hfov>
"""
import csv, math, sys, time
from pathlib import Path

repo, data, out_csv, every, hfov = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
sys.path.insert(0, str(repo)); sys.path.insert(0, str(repo / "tools"))
import run_visloc_benchmark as B  # noqa
from avinavi.geo import LocalOrigin  # noqa
from avinavi.types import GeoPoint  # noqa
import cv2  # noqa

flight_dir, csv_path, tif = B.find_flight(data)
frames = B.read_frames(csv_path, flight_dir / "drone")
config = B.load_config()
ext = B.LocalizationEngine._build_extractor(config)
mat = B.LocalizationEngine._build_matcher(config)
db = B.build_tiles(cv2, tif, data.parent / "visloc_out" / ("tiles_" + flight_dir.name), ext)
config.map.database_path = db.path
lats = [f["lat"] for f in frames]; lons = [f["lon"] for f in frames]
origin = LocalOrigin((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2, 0.0)

# variant: (label, pitch = sp*Kappa + mount, roll = sr*Omega)
# Engine yaw is Phi1+90, so engine pitch acts cross-track and engine roll along-track.
# variant: pitch = a*Omega, roll = b*Kappa + m
variants = [("asis", 1, 1, 0.0), ("p-O r+K", -1, 1, 0.0)]
for m in (2.0, -2.0, 1.0, -1.0):
    variants.append(("p-O r+K m%+g" % m, -1, 1, m))

sample = list(range(5, len(frames), every))
images = {i: B.load_frame(cv2, frames[i]["path"]) for i in sample}
with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["variant", "num", "valid", "err_m"])
    for label, sp, sr, mount in variants:
        errs = []
        for i in sample:
            f, img = dict(frames[i]), images[i]
            if img is None:
                continue
            om, ka = f["pitch_deg"], f["roll_deg"]  # as read: pitch<-Omega, roll<-Kappa
            f["pitch_deg"], f["roll_deg"] = sp * om, sr * ka + mount
            eng = B.make_engine(config, db, ext, mat, hfov, img.shape)
            eng.initialize(origin, start_position=GeoPoint(f["lat"], f["lon"], 0.0))
            fix = B.localize(eng, img, f, (1, 90.0))
            e = B.haversine_m(fix.geo.lat_deg, fix.geo.lon_deg, f["lat"], f["lon"]) if fix.valid else None
            w.writerow([label, f["num"], fix.valid, "" if e is None else round(e, 2)]); fh.flush()
            if e is not None and e < 80:
                errs.append(e)
        errs.sort()
        print(time.strftime("%H:%M:%S"), "%-22s locked %d/%d median %.1f" % (label, len(errs), len(sample), errs[len(errs) // 2] if errs else -1), flush=True)
print("DONE", flush=True)
