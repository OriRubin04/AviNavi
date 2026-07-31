# Safety and the testing progression

The visual position starts as **monitoring data**. It becomes a flight-control
input only after each step below has actually passed — not after it has been
skipped because the previous one looked fine.

```
Visual position calculated
        |
        v
     Log only                       Mode 1, offline
        |
        v
    Send MAVLink                    Mode 2, bench
        |
        v
Verify ArduPilot receives it        MAVLink Inspector
        |
        v
  Verify EKF accepts it             "EKF3 IMU0 is using external nav data"
        |
        v
  Stationary testing                Mode 2, armed props off
        |
        v
Controlled flight testing           Mode 3, GPS available as a safety net
        |
        v
 GPS-denied flight                  Mode 3, GPS1_TYPE = 0
```

## The three operating modes

### Mode 1 — offline

No flight controller. Recorded frames plus the map database.

```bash
avinavi run --mode offline --camera /path/to/recorded/frames --map map_database
```

- `camera.source: directory` replays a folder of images in filename order.
- No MAVLink connection is opened at all; `initial_position.source` must be
  `manual`.
- Everything from SuperPoint to the pose solve runs exactly as it will in
  flight, and every cycle is written to the CSV log.

This is where algorithm work belongs. If the localizer cannot find your
aircraft in recorded imagery on a workstation, it will not find it in the air.

### Mode 2 — bench / hardware-in-the-loop

Raspberry Pi connected to the flight controller. **The aircraft is stationary.**

```bash
# 1. Link and origin only, no vision, no transmission
avinavi mavlink-check --duration 20

# 2. Send test ODOMETRY at a fixed zero position
avinavi mavlink-check --send-test-odometry --duration 30

# 3. The full pipeline, computing and logging but transmitting nothing
avinavi run --mode bench

# 4. The full pipeline, transmitting
avinavi run --mode bench --send
```

Step 2 is the one that catches frame-ID and covariance mistakes, because it
isolates the MAVLink encoding from everything else. Verify in Mission Planner's
MAVLink Inspector (Ctrl+F) that `ODOMETRY` is arriving at the expected rate,
then check the HUD for `EKF3 IMU0 is using external nav data`.

What to confirm before moving on:

- [ ] `HEARTBEAT` received; `avinavi mavlink-check` reports `connected: true`
- [ ] The EKF origin has been set and is reported back
- [ ] `ODOMETRY` visible in MAVLink Inspector at `rates.mavlink_odometry_hz`
- [ ] `EKF3 IMU0 is using external nav data` in the HUD
- [ ] With the aircraft stationary, the reported position is stable — a
      wandering position on a stationary airframe means the localizer is
      matching noise
- [ ] Rotating the aircraft by hand changes the reported yaw in the right
      direction and by the right amount
- [ ] Carrying the aircraft a measured distance moves the position by that
      distance, in the right direction

That last pair is the cheapest possible check on the coordinate frames, and it
catches sign errors that no amount of code reading will.

### Mode 3 — flight

```bash
avinavi run --mode flight --send
```

First flights should keep GPS available as a safety net, with the EKF on
ExternalNav and a second source set on GPS bound to an RC switch. Only once the
visual solution has held up across a full sortie should you set
`GPS1_TYPE = 0` and fly genuinely GPS-denied.

## The gates between the solution and the autopilot

Three independent things must all be true before a message is transmitted.

### 1. `output.send_to_autopilot`

The master switch. **Defaults to `false`.** A fresh checkout computes
everything, logs everything and transmits nothing. `--send` on the command line
sets it for one run.

`output.message: none` is a stronger form: the publisher is disabled entirely.

### 2. The safety gate

`avinavi/app/safety.py`. Between the localizer and the publisher:

```yaml
safety:
  required_consecutive_good_fixes: 5    # before transmission is armed
  max_consecutive_failures: 10          # before it is disarmed again
  max_position_jump_m: 100.0            # a jump this big is a mis-localization
  require_heartbeat: true
```

- **Arming.** Nothing is transmitted until `required_consecutive_good_fixes`
  valid, confident fixes have arrived in a row.
- **Disarming.** `max_consecutive_failures` invalid cycles in a row withdraws
  transmission until the streak is rebuilt. A single dropout does not — that is
  what the propagator is for.
- **Jump rejection.** A fix further than `max_position_jump_m` from the previous
  one is refused *and* disarms the gate. It is either a mis-localization against
  the wrong tile or a re-acquisition after a long dropout; neither should be
  fused as if it were a continuous measurement.
- **Link check.** With `require_heartbeat: true` nothing is sent while the
  heartbeat is stale.

### 3. The propagator's age limit

```yaml
propagation:
  max_age_s: 1.0
```

Past this, `solution_at()` returns `None` and the output stage goes quiet. It
never repeats a stale pose to keep the message rate up. ArduPilot's own
external-navigation timeout then takes over, which is a failure mode the EKF
knows how to handle — unlike a confidently repeated wrong position.

## Failure handling

Every failure mode named in the specification has explicit handling. In all of
them the rule is the same: **do not generate a fake position.**

| Failure | Behaviour |
| --- | --- |
| LightGlue finds nothing | `VisualFix(valid=False, reason=TOO_FEW_MATCHES)`, no position at all |
| Geometry fails verification | `GEOMETRIC_VERIFICATION_FAILED`, with the specific reason logged |
| Confidence below threshold | `LOW_CONFIDENCE`, solution marked invalid |
| No candidate tiles | `NO_CANDIDATE_TILES`; the search radius grows, then a coverage failure is reported |
| Camera failure | `CameraError` → propagator cleared, safety gate reset, `ODOMETRY.quality = -1` sent once, run stops |
| MAVLink disconnected | Transmission stops, the loss is logged, reconnection is logged |
| Map coverage exhausted | `search_exhausted`, error logged, `STATUSTEXT` to the HUD |

An invalid fix still carries its diagnostics — which tile was tried, how many
matches, why it was rejected — so failures are debuggable. It just never
becomes an external navigation message.

## Confidence

```yaml
confidence:
  min_confidence: 0.5
```

Below this the fix is invalid. The score is a weighted sum of four normalised
terms — inlier ratio, saturating inlier count, reprojection error and the
spatial spread of the inliers across the frame. The spread term is the one that
catches geometrically valid but locally clustered matches, which is the failure
that produces a *confident* wrong position.

The score also sets the reported measurement noise, so a marginal fix reaches
the EKF with an honest covariance rather than being either dropped or
over-trusted:

```yaml
  position_sigma_best_m: 1.0      # at confidence 1.0
  position_sigma_worst_m: 15.0    # at min_confidence
```

## Diagnostics

The console view updates at `rates.diagnostics_hz`:

```
MAVLink connection: OK
Heartbeat: OK (0.3 s ago)

ODOMETRY transmission:
    Enabled: yes
    Rate: 19.8 Hz
    Sent: 3941   Suppressed: 12

Visual localization:
    Rate: 4.7 Hz (valid 4.5 Hz)
    Camera: 29.9 Hz
    Latency: mean 187 ms, max 240 ms
    Valid: True
    Position: 32.0853100, 34.7818400, 138.2 m
    Local NED: N 34.5  E -12.1  D -98.2 m

Confidence:
    0.92
Matches:
    182
RANSAC inliers:
    131 (ratio 0.720)
Reprojection error: 1.84 px
Tile: tile_0042 (of 4 candidates)
Tile search: r=300 m, 0 consecutive failures
```

Rates are measured over a sliding window, not assumed from the configuration —
the point of watching them is to catch the case where the Pi cannot keep up
with what it was asked to do.

Alongside it, `logs/solutions_<timestamp>.csv` gets one row per cycle:
timestamp, frame sequence, validity and reason, tile, geographic and NED
position, yaw, every match statistic, latency, and what was actually
transmitted. When `mavlink.request_gps_for_ground_truth` is enabled the row also
carries the GPS position and the error against it.

## Ground truth without contaminating the loop

```yaml
mavlink:
  request_gps_for_ground_truth: true    # development only
```

This requests `GLOBAL_POSITION_INT` and writes it to the CSV. The localizer
never reads it: `MavlinkConnection._update_state` stores it in a field that only
the logger touches, and the tile search is only ever recentred by
`LocalizationEngine._register_success`, which is only ever called with a visual
fix.

**Turn it off for real GPS-denied operation.** Not because it would break
anything, but because leaving it on makes it impossible to prove from the logs
alone that GPS was not involved.

## Benchmarking before choosing rates

The specification asks for the Pi to be benchmarked before the production rate
is fixed. Run a fixed number of cycles against recorded imagery:

```bash
avinavi run --mode offline --max-cycles 200
```

and read `Latency: mean … ms, max … ms` from the diagnostics, or the
`processing_time_s` column of the CSV. Set
`rates.visual_localization_hz` below `1 / max_latency`, not `1 / mean_latency` —
a rate the Pi can only sometimes sustain produces a stuttering solution, which
is worse for the EKF than a slower steady one.

Levers if it is too slow, in the order worth trying:

1. `superpoint.resize_max_side` — the single biggest one.
2. `superpoint.max_num_keypoints`.
3. `lightglue.max_candidate_tiles` — matching cost is linear in this.
4. `lightglue.depth_confidence` / `width_confidence` — LightGlue's own early
   exit and point pruning.
