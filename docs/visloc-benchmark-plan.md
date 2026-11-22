# Real-imagery benchmark (UAV-VisLoc) — handoff notes

Status: data downloaded and inspected; the tool `tools/run_visloc_benchmark.py`
is **not yet written**. Goal: a one-command tool the operator runs alone, which
writes a small results file to send back.

## Environment (this laptop)
- Project venv: `.venv\Scripts\python.exe` (torch 2.15 CPU, LightGlue, OpenCV,
  rasterio, pytest). Tests: 610 passed, 2 skipped, 2 failed — both in
  `tests/test_camera_usb.py`, caused by Windows `time.monotonic` 15.6 ms
  resolution (measured), not a code bug.

## Data (measured)
`C:\Users\user\Desktop\AviNavi_data\uav_visloc_sample\UAV_VisLoc_example\`
- `03/drone/03_0001..0768.JPG`, 3976x2652, EXIF stripped (no intrinsics).
- `03/03.csv`: num,filename,date,lat,lon,height,Omega(pitch),Kappa(roll),Phi1(yaw, higher confidence),Phi2
  — height ~466 m, pitch/roll within ±5°, frame spacing ~95 m, 2018-10-23.
- `03/satellite03.tif`: 35092x24308 RGB, EPSG:4326 GeoTIFF, ~0.25–0.30 m/px,
  8.9 x 7.2 km. Corners also in `satellite_ coordinates_range.csv` (note the space).

## Plan
1. Tile the GeoTIFF at 0.5 m/px (downsample) with `avinavi/mapdb/builder.py`
   (`slice_image` + `build_features`), or mirror `build_tile_database` in
   `avinavi/sim/evaluate.py:345`.
2. Calibrate focal length on ~10 held-out frames (fit to map; f = height / GSD).
   Exclude them from scoring. Check Phi1 sign/zero convention against the map.
3. For each other frame: set `config.camera.fx/fy/cx/cy/width/height`,
   `engine.localize(frame, roll, pitch, yaw, height_hint_m)` — see the loop in
   `run_simulation`. Downscale frames to the configured working size first.
4. Write `visloc_results.json`: lock rate, error p50/p90 vs GPS, false fixes
   (>60 m), ms/frame. Small enough to paste back.

## First result (2026-11-22, flight 03, 760 scored frames, CPU, measured)
Lock 86.2%, good fixes (<=60 m) 85.9%, 2 fixes just over 60 m (64.7, 60.6 m),
error p50 20.1 m / p90 32.0 m, ~7.2 s per frame. Calibrated HFOV 95 deg (poorly
determined: error is flat over 35-95 deg; dataset GSD implies ~46-81 deg),
yaw = Phi1 + 90 deg. Error correlates with tilt (r = 0.48: median 18 m at
<10 m tilt offset, 32 m at 20-40 m), so part of it is image-centre vs aircraft
position; ~18 m remains unexplained. Full numbers: `visloc_results_flight03.json`.
