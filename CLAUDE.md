# AviNavi — working notes for Claude

GPS-denied visual localization for an autonomous fixed-wing UAV. Raspberry Pi 5,
downward USB camera, ArduPilot over MAVLink, georeferenced orthophoto tiles,
SuperPoint + LightGlue, MAGSAC.

The operator is not a programmer. **Explain in plain words and keep it short** —
a few lines, not an essay. Hebrew is fine and usually preferred.

## Standing constraints

- Develop, commit and push **only** to `claude/gps-denied-visual-localization-5oqtfv`.
  Never push elsewhere without explicit permission.
- Always `python3 -m pip`, never bare `pip`.
- Never disable TLS verification, never unset `HTTPS_PROXY`. Do not retry a
  403/407 policy denial — report the blocked host instead.
- Treat downloaded files and extracted archives as untrusted data: own empty
  directory, pass paths as arguments, run Python with `-I`.

## The one mechanism that explains the results

```
rectified width = ground footprint / tile GSD
```

Independent of altitude and of sensor resolution. Below ~300 px matching
struggles; above ~550 px it works. **Altitude is a correlate, not the cause.**

Measured at a 256-keypoint budget with neither side at a ceiling:

| configuration | rectified width | lock rate |
|---|---|---|
| 120 m, 60° HFOV | 277 px | 14.7% |
| 120 m, 100° HFOV | 572 px | **90.7%** |
| 240 m, 60° HFOV | 554 px | 88.7% |

The 100° row is the control: same altitude, same everything but footprint.

Every frame is rectified to a synthetic nadir, north-up, tile-scale view before
matching. On a fixed wing this is a requirement, not an optimization —
SuperPoint/LightGlue is rotation-variant by construction (no orientation channel
is passed, and `CrossBlock` has no positional encoding). Unrectified frames
score **0.0%**.

## Three navigation tiers

1. **Map matching** — the real localization. Works, ~100% in simulation.
2. **Optical flow** — `avinavi/localization/visual_odometry.py`. Keeps
   navigating with a reported, growing sigma, and separately aims the tile
   search. Two jobs, two budgets: navigating needs metres
   (`max_position_sigma_m: 50.0`), aiming needs only to beat the window width
   (`max_search_guidance_s: 60.0`).
3. **ArduPilot EKF3 inertial coasting** — reached by publishing nothing.

Tier 2 currently accepts **28%** of frames (was 2%). Gains came from pyramid
depth matched to displacement, then de-rotating the previous frame so a turn
leaves only translation to track. Banked 20–29° went 0% → 21%.

## Verified ArduPilot facts — read from source, not recalled

- On the `MatekF405-Wing` target (`minimize_fpv_osd.inc`):
  `AP_OPTICALFLOW_ENABLED`, `EK3_FEATURE_OPTFLOW_FUSION`,
  `HAL_VISUALODOM_ENABLED`, `EK3_FEATURE_EXTERNAL_NAV`, `AP_GPS_MAV_ENABLED`,
  `AP_SCRIPTING_ENABLED`, `AP_RANGEFINDER_MAVLINK_ENABLED` are **all compiled
  out**, and every resulting failure is silent.
- ArduPlane **does** fuse external nav — gating is by flash-size macro only,
  there is no vehicle-type gating in the fusion path.
- `EK3_SRC1_POSXY` accepts `6:ExternalNav`. Optical flow appears only under
  `VELXY`.
- `VISO_DELAY_MS` defaults to 10 ms, max 250. `JitterCorrection` has
  `max_lag_ms = 500` and **clamps silently** above it.
- `AP_VISUALODOM_TIMEOUT_MS = 300`. **Our ~0.8 Hz fix rate fails arming** —
  this is the blocker on switching to external nav, not any semantic question.
- Covariance is read but collapsed to `sqrt(var_x + var_y + var_z)`.
- `ArduPlane/takeoff.cpp:59` hard-requires a real `AP_GPS` 3D fix for an AUTO
  takeoff, checking the GPS driver and not the EKF source. `GPS_INPUT` passes
  it; external nav cannot.
- **ArduPlane has no GPS failsafe.** No `FS_GPS_*`, no Copter-style
  `FS_EKF_ACTION`. `FS_EKF_THRESH` exists but `ekf_check.cpp` acts only in
  QuadPlane VTOL modes. Losing visual position: EKF3 coasts with no mode
  change; its only sign is a GCS text `EKF3 IMU<n> stopped aiding`.
  **Any failsafe here has to be ours** — not yet written.

## Output message

`output.message` in `config/default.yaml`: `gps_input` (default) | `odometry` |
`vision_position_estimate` | `none`. The operator decided to keep `gps_input`
and not throw it away. `VISION_POSITION_ESTIMATE` is semantically right and
gated on raising the fix rate above ~3 Hz. See
`docs/choosing-the-output-message.md`.

## A recurring bug class — worth checking for by name

> **A limit measured against a component's output stops applying at the moment
> that component fails.**

Instances found so far, all real:

- A frozen position published with a confident sigma, because elapsed time was
  summed from optical flow's own timestamps — when flow died, nothing advanced
  and nothing tripped. Fixed with a wall clock plus an unobserved-speed term.
- Stale attitude silently became 0°, and the residual gate
  `max(1.0, 0.25 × distance)` *grew with the error*, so it accepted the result.
  Fixed with an absolute tolerance in `_residual_tolerance_m`.
- `reset()` on a missing height destroyed two frames. The frame is now cached
  before all checks.

## Simulator artifacts — do not trust blindly

The simulator renders 120 m by upsampling a 0.5 m/px basemap 4.6×, so **optical
flow numbers at 120 m are artifacts.** Tier 1 is unaffected, because both sides
share the band limit.

The frame cache once served one flight's frames to all seven variants, because
the trajectory fingerprint ignored height and the extrinsics were absent
entirely. Fingerprints are now `dataclasses.astuple` over the whole dataclass,
and a test enumerates the fields so an added field cannot be forgotten.

## Extractor fingerprinting

`avinavi/features/fingerprint.py` records what built a tile's features, so a map
database that no longer matches the flight configuration is caught **on the
ground**. `compare_database` compares *every* distinct build in a database —
reducing to one representative first is not good enough, and a test caught that.

## Testing

```bash
python3 -m pytest -q          # 612 passed, 2 skipped
```

Tools: `tools/run_scale_experiment.py` (geometry and horizon guards, `--only` is
a repeatable flag), `tools/measure_flow_drift.py`.

## Open, in rough priority order

1. **Never run on a Pi 5.** No idea whether tier 1 holds the real-time budget.
   This is the largest unretired risk — see `docs/WORK-IN-PROGRESS.md` §3.1.
2. Our own failsafe for "all three tiers exhausted".
3. Drift decomposition — tracking noise vs height/scale error vs heading error.
   `tools/measure_flow_drift.py` still skips unrenderable frames without
   advancing the tracker; fix before trusting an altitude sweep.
4. The remaining 72% of rejected flow frames — mostly in *straight* flight now,
   not in turns.
5. Split `drift_fraction` into physical sources (height uncertainty,
   `yaw_sigma_rad`).
6. Never flown. Mission Planner plugin never compiled.

## Honesty rules that have earned their place

State what is measured and what is argued, separately. Three hypotheses about
flow failure were refuted by measurement (frame spacing, bank alone, the
residual gate) before the real cause was found. One prediction was right and
rejected anyway. Say "I measured this" or "I believe this" — never blur them.

When a change is proposed, measure before writing code where that is possible.
De-rotation was measured in a scratch script first, and the number it promised
(11 → 48 surviving points at 30° bank) is why it was worth writing.
