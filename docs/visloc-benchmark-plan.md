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

## Where the ~20 m came from (2026-11-22, measured)
Tools: `tools/visloc_error_vectors.py` (signed N/E error per frame, body fix and
image-centre ground point, each frame seeded at truth) and
`tools/visloc_attitude_test.py` (re-runs the full pipeline under attitude variants).

1. Error vectors, every 4th frame (166/192 locked, HFOV 60 and 95 alike):
   - No constant north/east bias (mean N -2.5, E +2.0 m), so **not a map offset**.
   - Constant **+16.7 m along-track** (ahead), ~0 cross-track.
   - Image-centre error follows the attitude labels at ~8 m/deg = 466 m x tan(1 deg):
     along-track with Kappa (r = +0.81), cross-track with Omega (r = -0.87).
     So the CSV lat/lon is the aircraft position, not the image centre as the
     README says, and Kappa behaves as pitch, Omega as roll.
   - Our yaw = Phi1 + 90 deg with identity extrinsics put engine-pitch
     cross-track and engine-roll along-track, so the axes were already matched;
     but the cross-track sign was wrong (body-fix cross error -14.5 m/deg,
     double the -7.9 of the raw image centre).
   - HFOV does not matter here (18.1 vs 18.3 m).
2. Prediction: flip Omega's sign and add a ~2.1 deg (16.7 m / 8.1 m/deg) fixed
   camera tilt on the along-track axis. Tested in the full pipeline on a
   disjoint sample (every 12th frame, 64 frames, HFOV 60):

   | variant | locked | median error |
   |---|---|---|
   | as run | 58/64 | 22.5 m |
   | pitch = -Omega | 58/64 | 16.7 m |
   | pitch = -Omega, roll = Kappa + 2 deg | 59/64 | **5.8 m** |
   | ... + 1 deg | 59/64 | 10.5 m |
   | ... - 2 deg | 57/64 | 29.9 m |

   A first test that also swapped Omega/Kappa failed (best 16.9 m), because the
   90 deg yaw offset had already swapped them; that test was mis-designed.
3. Conclusion: the ~20 m was a dataset-convention mismatch plus a ~2 deg camera
   mount tilt, not matching error. Matches the third-party GeoAnchor report
   (6.1 m on flight 03) and its claim that the labels are swapped.
   **Lesson for real flight:** 2 deg of uncalibrated mount tilt costs 16 m at
   466 m. Camera extrinsics must be calibrated, not assumed.
4. Full sequential flight with these settings (all 768 frames, seeded once,
   HFOV 60, yaw = Phi1 + 90, omega_sign -1, mount 2 deg), measured:
   lock 87.6%, **0 fixes over 60 m**, error p50 **6.6 m**, p90 13.0 m,
   max 21.4 m, 7.2 s/frame on CPU. Mean bias N -3.3 m, E +0.4 m.
   Failures: LOW_CONFIDENCE 78, GEOMETRIC_VERIFICATION_FAILED 11, BLURRED 6.
   Caveat: the 2 deg mount was measured on this same flight.
   Full numbers: `visloc_results_flight03_v2.json`.
