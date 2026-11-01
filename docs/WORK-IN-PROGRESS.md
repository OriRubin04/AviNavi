# Work in progress — resume here

**Status as of this commit: the architecture changed substantially from the
first version, based on three source-verified research findings. The new code
is written and imports cleanly, but the measurement run that justifies it has
not been executed yet.** Read §1 before doing anything else.

---

## 1. The three findings that reshaped the design

All verified against ArduPilot / Mission Planner / LightGlue source, not recalled.

### 1.1 A stock MATEK F405-WING cannot accept a visual position at all

`HAL_VISUALODOM_ENABLED`, `EK3_FEATURE_EXTERNAL_NAV` and `AP_GPS_MAV_ENABLED`
are **all 0** on the `MatekF405-Wing` ArduPlane target. `ODOMETRY`,
`VISION_POSITION_ESTIMATE` **and** `GPS_INPUT` are therefore all compiled out,
and every failure is silent. Full detail and the three verification methods:
`docs/flight-controller-compatibility.md`.

Consequence for the code: **`GPS_INPUT` is now the primary output path**
(`avinavi/mavlink/gps_input.py`), requiring a custom firmware build with
`AP_GPS_MAV_ENABLED` ticked at custom.ardupilot.org. `ODOMETRY` is retained for
2 MB boards.

### 1.2 SuperPoint + LightGlue is rotation- and scale-variant, and a banking plane breaks it

- LightGlue's synthetic pre-training covered **±31.5°** (`max_angle: 45` ×
  `difficulty: 0.7` in `superpoint+lightglue_homography.yaml`); its second
  stage fine-tunes on gravity-aligned MegaDepth with essentially no rotation.
- `features["superpoint"]` does not set `add_scale_ori`, so **LightGlue is
  handed only `(x, y)`** — no orientation channel exists. `CrossBlock` has no
  positional encoding: translation invariant by design, rotation variant by
  design.
- An independent 360° sweep on a fixed-wing survey pair: SP+LG mean precision
  **43.8%**, worst **0.0%**, usable over only ~165° of the circle. SIFT held
  >95% everywhere.
- Usable GSD ratio is **≤1.2×, failing by 1.5×**.
- ArduPlane's `ROLL_LIMIT_DEG` defaults to **45**, and stall prevention always
  permits **≥25**. At 100 m AGL / 60° HFOV, bank of just **10° already gives a
  1.5× GSD ratio across the frame**; 30° gives 4×.

**Conclusion: unwarped matching is impossible in any normal fixed-wing turn.**
Hence `avinavi/localization/prewarp.py` — rectify every frame to a synthetic
nadir, north-up, tile-GSD view first. It is a matching aid only: the homography
is composed back to camera pixels (`LocalizationEngine._compose_to_camera`) and
the pose solve is untouched.

Published ablation on the same problem: removing attitude/boresight correction
moved median error 6.62 → 17.13 m and within-20 m from 92.7% → 67.3% — the
largest single contributor. A **2° boresight error alone cost 2.7×** accuracy.

### 1.3 The operator's map click can only reach the Pi through specific channels

ArduPilot's `MAVLink_routing::forward` **early-returns** for a message
addressed to `target_system = 1, target_component = 1`, and **every** Mission
Planner map action is addressed to component 1. So sniffing "Point Camera Here"
cannot work when the Pi is on its own UART.

What does work, in preference order (all in `avinavi/mavlink/reanchor.py`):

1. `COMMAND_INT` / **`MAV_CMD_SPATIAL_USER_1` (31005)** addressed to our
   component — zero side effects (not in ArduPilot's
   `command_long_stores_location()`), and we can `COMMAND_ACK` it so the
   operator sees the click land. Needs the plugin in
   `tools/mission_planner_plugin/AviNaviSearchHere.cs`.
2. **`SET_GPS_GLOBAL_ORIGIN` (48)** sniffing, from "Set EKF Origin Here" — the
   only stock map action that reaches a separate UART, because the message has
   no `target_component` field so the router forwards it. Requires the Pi to
   transmit with the **same sysid as the autopilot**. Verified safe: ArduPilot
   refuses an origin change once set, in two places.
3. `DO_SET_ROI` (**201**, the legacy command — *not* 195, which is what Mission
   Planner actually sends) — telemetry-hub topology only.
4. `HOME_POSITION` (242) — reaches every link reliably but **moves the RTL
   point**, so default `false`.

---

## 2. What is done and working

| Area | File | State |
| --- | --- | --- |
| Frame rectification | `avinavi/localization/prewarp.py` | **Done**, homography verified numerically against hand-computed ground points |
| CLAHE normalisation | same file, `PhotometricNormalizer` | Done |
| `GPS_INPUT` output | `avinavi/mavlink/gps_input.py` | Done; field order + `ignore_flags` verified against pymavlink and `AP_GPS_MAV::handle_msg` |
| Manual re-anchor | `avinavi/mavlink/reanchor.py` | Done, 5 channels |
| Mission Planner plugin | `tools/mission_planner_plugin/` | Written, **not compiled or tested** |
| Engine rewrite | `avinavi/localization/engine.py` | Done: rectify → normalise → gate → match → compose back → solve |
| Yaw-bias estimator | same file, `YawBiasEstimator` | Done — median residual vs autopilot yaw, free compass calibration |
| Classical backend | `avinavi/features/opencv_backend.py` | Done (SIFT/AKAZE/ORB + ratio + mutual check) |
| Simulator: terrain | `avinavi/sim/terrain.py` | Done, ~8 s for 3000² |
| Simulator: fixed-wing flight | `avinavi/sim/flight.py` | Done; coordinated turns, wind crab, verified 19% of time >15° bank |
| Simulator: camera | `avinavi/sim/render.py` | Done; blur/rolling-shutter/exposure/vignette/JPEG, ~148 ms per 1280×720 frame |
| Evaluator | `avinavi/sim/evaluate.py` | Written, **never run to completion** |
| Experiment driver | `tools/run_experiment.py` | Written, **never run to completion** |
| FC compatibility doc | `docs/flight-controller-compatibility.md` | Done |

Also fixed: a **documentation bug** in `frames.py` and
`docs/coordinate-frames.md` that claimed identity extrinsics put image +X along
body +Y. It is along body **+X** (the nose). A 90° error here biases yaw in a
way the EKF fights rather than rejects. `tests/test_frames.py` now pins all four
mountings.

**Verified working in this environment:** real SuperPoint + LightGlue weights
load and run (`pip install -e` from a clone of `cvg/LightGlue`; the PyPI name
does not exist). LightGlue matched in 23 ms on 512×512 / 82 keypoints.

---

## 3. Resume here — ordered, highest value first

### 3.1 Run the measurement (this is the blocking item)

The whole point of the simulator is to replace assertion with numbers, and that
run was interrupted. Nothing below should be reported to the user as an
improvement until this produces a table.

```bash
cd /home/user/AviNavi
python3 -m pip install numpy pytest pymavlink PyYAML opencv-python-headless
# note: pip and python3 point at different interpreters here — always use
#       python3 -m pip, never bare pip
SCRATCH=/tmp/... python3 tools/run_experiment.py --backend opencv \
    --duration 80 --basemap-px 2400 --quick
```

Start with `--backend opencv` (seconds per cycle, not minutes). Then repeat with
`--backend superpoint` and a small `--max-cycles` — and be aware SuperPoint on
this CPU was ~1.8 s for the first 512² call.

Expect to debug the first run: `evaluate.py` has never executed end to end.
Known risks — `build_tile_database` memory on a large basemap; the
`frames_key` cache keying on `id(trajectory)`, which is fragile; `wrong_tile`
being adjacent to the true tile and so overlapping it (pick a tile that is
genuinely far away instead).

The four variants that matter: `raw`, `+CLAHE`, `+rectify`, `+rectify+CLAHE`.
**If rectification does not clearly win, the §1.2 reasoning is wrong somewhere
and must be re-examined rather than explained away.**

Also measure `separation` (true-tile inliers ÷ wrong-tile inliers). Published
work found this predicts end-to-end safety far better than match count: a
matcher scoring 13.5× was 6 km wrong on 95% of frames while one scoring 709×
was safe. **This is the number that decides whether the system is flyable.**

### 3.2 Finish the wiring that the new modules need

- `config/default.yaml` **still says `output.message: odometry`** and has no
  `prewarp`, `photometric`, `quality`, `features`, `visual_odometry` or
  `reanchor` sections. The dataclass defaults are right; the YAML is stale and
  overrides them. `tests/test_config.py::test_the_default_yaml_covers_every_dataclass_field`
  will fail until this is fixed.
- `avinavi/app/runner.py` does not yet construct `GpsInputPublisher` or
  `ReanchorListener`, and does not pass `yaw_rad` into `engine.localize`.
  Add an output-publisher factory keyed on `config.output.message`.
- `avinavi/mavlink/__init__.py` does not export the two new modules.
- `initial_position.py` should call
  `connection.request_message_interval(49, 5e6)` — `GPS_GLOBAL_ORIGIN` and
  `HOME_POSITION` are **in no stream group**, so they are sent once on change
  with no retry, and a single lost packet means the origin is never learned.

### 3.3 Not yet started

- **USB camera hardening** — the user's camera is USB, and two settings prevent
  catastrophic failure: force **MJPEG** (720p30 YUYV is 442 Mbit/s and does not
  fit in USB 2.0's ~300 Mbit/s practical ceiling) and **lock exposure ≤ 1/500 s**
  via `V4L2_CID_EXPOSURE_AUTO`/`EXPOSURE_ABSOLUTE`. Auto-exposure dropping to
  1/30 s gives 4.6 px of blur, and blur was the worst degradation measured
  anywhere in the research (58 m moderate, 500 m heavy, vs a 6.6 m baseline).
  Blur at 1/500 s is 0.28 px — a non-issue. Also drain the V4L2 queue to the
  newest frame; `CAP_PROP_BUFFERSIZE` is unreliable.
- **Frame-to-frame visual odometry** (`VisualOdometryConfig` exists, no module).
  Published ablation: removing it moved p90 from 16 m to **368 m** — it barely
  touches the median but it controls the tail, and it is what makes a 1–2 Hz
  match rate viable. Expect 1–4% of distance drift, dominated by heading bias,
  which is exactly what `YawBiasEstimator` corrects.
- **Tile localizability score, offline.** A structure score computed from the
  basemap tile alone predicted localizability at **AUC 0.852** — better than
  post-match inlier count. This would let `avinavi map-info` warn which areas
  will not work *before* flying. High value for the money.
- **Realistic rate defaults.** `rates.visual_localization_hz` is still 5.0.
  Measured expectations on a Pi 5: 0.5–0.85 Hz PyTorch, ~1.7 Hz ONNX Runtime,
  ~3.3 Hz at 384². Set the default to 2.0 and document ONNX Runtime (~2.4×) and
  XFeat + LighterGlue (4–8× faster for ~5% relative AUC) as the speed path.
  **Do not pursue a Hailo NPU** — LightGlue's dynamic keypoint count and
  data-dependent early exit are incompatible with its static dataflow compiler.
- **Tests** for `prewarp`, `gps_input`, `reanchor`, `opencv_backend` and the
  simulator. The existing 219 pass; `test_config` will fail until §3.2.
- **Docs**: `matching-robustness.md` (the §1.2 evidence plus measured results),
  `manual-reanchor.md`, `usb-camera.md`. `README.md` and
  `docs/ardupilot-configuration.md` both need the `GPS_INPUT` path added.

---

## 4. Things that will bite

- `pip` → Python 3.13, `python3` → 3.11. **Always `python3 -m pip`.**
- The container was recreated once mid-session and all pip packages vanished.
  Re-install before assuming anything works.
- `download.pytorch.org`, `ardupilot.org` and `mavlink.io` are **403 via the
  egress proxy**. `raw.githubusercontent.com` and PyPI work. Get torch from
  PyPI, ArduPilot facts from raw GitHub.
- `lightglue` is **not on PyPI** — clone `cvg/LightGlue` and `pip install -e`.
  Pillow needed a `--force-reinstall`.
- **`normalize_keypoints` must receive the true image size.** A port that
  passed a hardcoded size put coordinates 2.3× outside the trained range and
  collapsed match rates silently. `SuperPointExtractor` sets `image_size` from
  the original frame and rescales keypoints to match — keep that invariant if
  the resize logic is touched.
- `docs/ardupilot-configuration.md` was written before §1.1 was known. Its
  parameter values are correct **for a board that has the features**; it needs a
  prominent pointer to the compatibility page.
