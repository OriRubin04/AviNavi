# AviNavi

GPS-denied visual localization and navigation for ArduPilot, using SuperPoint
and LightGlue against a georeferenced map on a Raspberry Pi 5.

The operator picks the aircraft's starting position once, on the Mission
Planner map. From then on the system localizes by matching the downward camera
against precomputed map tiles, and feeds the result to ArduPilot's EKF as
MAVLink `GPS_INPUT` or `ODOMETRY`. **No GPS is involved in the localization
loop.** The operator can also click the map at any time to tell the system where
to resume searching after a lost lock.

> **Read [docs/flight-controller-compatibility.md](docs/flight-controller-compatibility.md)
> before wiring anything.** On a stock ArduPlane build for a 1 MB board such as
> the MATEKSYS F405-WING, `ODOMETRY`, `VISION_POSITION_ESTIMATE` *and*
> `GPS_INPUT` are all compiled out, and every one of them fails silently. Search
> your Full Parameter List for `VISO_TYPE`: if it is not there, no amount of
> configuration will make the external-navigation path work.

```
              OPERATOR
                  |  selects the starting position
                  v
           MISSION PLANNER
                  |  MAVLink: GPS_GLOBAL_ORIGIN
                  v
           RASPBERRY PI 5
                  |
                  v
          Initial tile search
                  |
        +---------+---------+
        |                   |
     CAMERA           MAP DATABASE
        |                   |
        v                   v
    SuperPoint      precomputed features
        |                   |
        +---------+---------+
                  |
                  v
              LightGlue
                  |
                  v
            RANSAC / MAGSAC
                  |
                  v
          Visual localization
                  |
        +---------+---------+
        |                   |
        v                   v
 Update tile search    MAVLink GPS_INPUT
        |                   |    (or ODOMETRY)
        +--> next frame     v
                        ArduPilot EKF
```

Each live frame is first rectified to a synthetic nadir, north-up, tile-scale
view. On a fixed wing that is not an optimization but a requirement: SuperPoint
carries no orientation channel and LightGlue was trained on pairs within roughly
±31.5°, while 10° of bank already skews the ground sample distance 1.5× across
the frame and ArduPlane permits 45°. See
[docs/matching-robustness.md](docs/matching-robustness.md) for the measured
effect.

## Documentation

| Document | What it covers |
| --- | --- |
| [docs/coordinate-frames.md](docs/coordinate-frames.md) | Every frame, every transform, and the MAVLink frame IDs ArduPilot silently requires |
| [docs/initial-position.md](docs/initial-position.md) | How the operator's map click reaches the Pi, and why it is done that way |
| [docs/ardupilot-configuration.md](docs/ardupilot-configuration.md) | Exact parameter names and values, verified against the ArduPilot source |
| [docs/map-database.md](docs/map-database.md) | Tile format, building a database, sizing and overlap |
| [docs/safety-and-testing.md](docs/safety-and-testing.md) | The testing progression, the safety gates, failure handling |
| [docs/flight-controller-compatibility.md](docs/flight-controller-compatibility.md) | **Read first.** Which boards can accept a visual position at all, and how to check yours |
| [docs/matching-robustness.md](docs/matching-robustness.md) | Why frames are rectified before matching, and what it measures |
| [docs/usb-camera.md](docs/usb-camera.md) | The three USB camera settings that fail silently on a Pi |
| [docs/manual-reanchor.md](docs/manual-reanchor.md) | Clicking the map to say where to resume searching |

## Installation

On a Raspberry Pi 5 (8 GB), Raspberry Pi OS 64-bit:

```bash
git clone <this repository> AviNavi && cd AviNavi
python3 -m venv --system-site-packages .venv && source .venv/bin/activate

# CPU build of torch for aarch64
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .

# The Pi camera stack comes from apt, not pip
sudo apt install -y python3-picamera2
```

On a workstation, for offline development and map building:

```bash
pip install -e '.[vision,maps,dev]'
```

The core package — configuration, geodesy, frame maths, MAVLink, the map
database — needs only `numpy`, `PyYAML` and `pymavlink`. torch and OpenCV are
imported lazily, so the test suite runs anywhere.

## Quick start

### 1. Build a map database

```bash
avinavi build-map orthophoto.tif map_database --tile-size 1024 --overlap 0.25
avinavi map-info map_database
```

For a plain image, add `--bounds west,south,east,north` in WGS-84 degrees.

### 2. Configure

Copy `config/default.yaml` and edit it, or override individual keys:

```bash
avinavi run --set camera.fx=520 --set camera.fy=520 --set mavlink.device=/dev/ttyACM0
```

The two things you must get right before anything works well:

- **`camera.fx` / `fy` / `cx` / `cy`** — calibrated intrinsics at the configured
  resolution. Without them the vertical channel is unobservable from imagery
  alone and you must supply a rangefinder or barometric height instead.
- **`camera_extrinsics`** — how the camera is mounted relative to the airframe.
  Identity means a nadir camera whose image +X points along the aircraft's right
  wing. See [docs/coordinate-frames.md](docs/coordinate-frames.md).

### 3. Run it offline first

```bash
avinavi run --mode offline --camera /path/to/recorded/frames
```

Nothing is transmitted. Every cycle lands in `logs/solutions_*.csv`.

### 4. Then on the bench, then in the air

```bash
avinavi mavlink-check --send-test-odometry     # verify the link and frames
avinavi run --mode bench                       # full pipeline, log only
avinavi run --mode bench --send                # transmit, aircraft stationary
avinavi run --mode flight --send               # the real thing
```

Do not skip the middle two. See
[docs/safety-and-testing.md](docs/safety-and-testing.md).

## How it works

### Two rates, deliberately decoupled

Visual localization runs at 2–5 Hz on a Pi 5; ArduPilot's EKF wants external
navigation at 10–20 Hz. Rather than throttling one to the other, the localizer
and the publisher run on separate threads and a propagator bridges them: it
holds the last fix, extrapolates it with the velocity estimated from the two
most recent fixes, and inflates the reported uncertainty with age.

Past `propagation.max_age_s` it returns nothing and the publisher goes silent.
**It never repeats a stale pose to keep the rate up** — ArduPilot's own timeout
is a failure mode the EKF can handle; a confidently repeated wrong position is
not.

### The search window moves on visual position

The operator's initial position seeds the tile search once. Every successful fix
recentres the window on the **visual** position. After repeated failures the
radius grows, up to a ceiling, so a drifted aircraft can re-acquire; past the
ceiling a map coverage failure is reported rather than searching forever.

`LocalizationEngine._register_success` is the only thing that moves the search
centre, and it is only ever called with a visual fix.

### Confidence gates everything

Every result carries `number_of_matches`, `number_of_inliers`, `inlier_ratio`,
`reprojection_error`, `matching_confidence` and `localization_confidence`. The
last is a weighted sum of four normalised terms, including the *spatial spread*
of the inliers across the frame — the term that catches geometrically valid but
locally clustered matches, which is the failure that produces a confident wrong
position.

Below `confidence.min_confidence` the fix is marked invalid and nothing is sent.
Above it, the score sets the covariance, so a marginal fix reaches the EKF with
an honest measurement noise rather than being silently over-trusted.

### One authoritative external-navigation source

Three interfaces exist and the publisher sends **exactly one**. There is no
configuration in which two go out, because two external navigation sources fuse
against each other.

| `output.message` | ArduPilot side | Use when |
| --- | --- | --- |
| `gps_input` (default) | `AP_GPS_MAV`, `GPS1_TYPE=14` | 1 MB board with a custom build; the only interface a single feature flag can provide |
| `odometry` | `GCS_MAVLINK::handle_odometry` | 2 MB+ board on stock firmware; carries velocity and a full covariance |
| `vision_position_estimate` | `AP_VisualOdom` | prototype fallback |

`gps_input` is the default because of the compatibility problem above: on the
F405-WING it is the one path reachable with a single build flag. Its cost is
that it is a *GPS*, so the satellite count and DOP values are necessarily
fabricated — but `horiz_accuracy` carries the real confidence-derived sigma, and
that is the field ArduPilot actually weights the fix by.

## Configuration

Everything is in `config/default.yaml` — no device path, rate, threshold or
gain is hard-coded. Sections:

| Section | What it controls |
| --- | --- |
| `mavlink` | Transport, IDs, heartbeat, which inbound streams to request |
| `initial_position` | How the operator's start position arrives |
| `camera` | Source, resolution, intrinsics, distortion |
| `camera_extrinsics` | The fixed camera-to-body transform |
| `rates` | Camera, visual localization and external navigation rates |
| `superpoint`, `lightglue` | Model settings and the candidate tile budget |
| `geometry` | RANSAC/MAGSAC method, thresholds and rejection tests |
| `confidence` | Score weights, the validity gate, the sigma mapping |
| `tile_search` | Search radius, growth on failure, tile preference |
| `map` | Database path, terrain altitude, feature cache size |
| `propagation` | Forward propagation and the staleness limit |
| `output` | Which message, and whether to transmit at all |
| `logging`, `safety` | Log destinations; the gates before the flight controller |

An unknown key is an error, not a silent no-op — a typo in a config file should
not cost you a flight.

## Development

```bash
pip install -e '.[dev]'
python -m pytest
```

The suite runs without torch, without a camera and without an autopilot. The
deep-learning stack is stubbed; everything downstream of it is the real code.

Two families of test carry most of the weight:

- **`tests/test_pipeline.py`** constructs a known camera pose, synthesises the
  correspondences a nadir camera at that pose would actually see, and checks
  that the pipeline recovers the pose — through geometric verification, the pose
  solve, the propagator and into encoded `ODOMETRY` bytes. Any sign error in the
  middle of the chain fails it.
- **`tests/test_external_navigation.py`** encodes real messages with pymavlink
  and decodes them back, so assertions about `frame_id` or covariance index 11
  are assertions about bytes that would genuinely reach ArduPilot.

## Module map

```
avinavi/
├── config.py            typed configuration loaded from YAML
├── types.py             shared value types and MAVLink covariance packing
├── geo.py               geographic <-> local NED <-> map pixel
├── frames.py            rotations and the camera -> body -> NED chain
├── camera/              Picamera2, OpenCV and image-directory sources
├── features/
│   ├── superpoint.py    live-frame feature extraction
│   ├── lightglue_matcher.py
│   └── geometric.py     RANSAC/MAGSAC and the rejection tests
├── mapdb/
│   ├── tile.py          tile metadata, georeferencing, stored features
│   ├── database.py      the tile database and the moving search window
│   └── builder.py       offline slicing and feature precomputation
├── localization/
│   ├── engine.py        the localization cycle
│   ├── prewarp.py       rectify to nadir/north-up/tile-GSD before matching
│   ├── pose.py          homography -> geographic position and yaw
│   ├── confidence.py    the score and the validity gate
│   └── propagator.py    forward propagation between fixes
├── features/
│   ├── superpoint.py    SuperPoint extraction
│   ├── lightglue_matcher.py   LightGlue matching
│   └── opencv_backend.py      SIFT/AKAZE/ORB, the no-torch fallback
├── mavlink/
│   ├── connection.py    transport, inbound state, heartbeat tracking
│   ├── initial_position.py   the operator-selected origin
│   ├── gps_input.py     GPS_INPUT, the default output path
│   ├── external_navigation.py   ODOMETRY and VISION_POSITION_ESTIMATE
│   └── reanchor.py      the operator's map click, over five channels
├── sim/                 fixed-wing flight, terrain and camera simulation
├── app/
│   ├── runner.py        the three modes and the main loop
│   └── safety.py        the gate before the flight controller
├── diagnostics/         rate monitoring and the CSV trace
└── cli/main.py          the avinavi command
```

The layering is deliberate: each stage can be replaced without touching the
others, which is what makes optical flow, IMU fusion, global visual retrieval or
an NPU-accelerated extractor additions rather than rewrites.

## Status and known limits

This is a complete, working implementation of the architecture, with the maths
verified against synthetic ground truth and the MAVLink encoding verified
against ArduPilot's own receiving code. It has **not** been flown. Before it is:

- Calibrate the camera and fill in `camera.fx/fy/cx/cy`.
- Measure `VISO_DELAY_MS` for your hardware rather than accepting the default.
- Benchmark the Pi and set `rates.visual_localization_hz` from the *worst-case*
  latency, not the mean.
- Work through [docs/safety-and-testing.md](docs/safety-and-testing.md) in order.

Modelling limits worth knowing:

- The terrain under each frame is assumed flat at the tile's `terrain_alt_m`.
  Significant relief will bias the position, most visibly in the vertical.
- Height from image scale is exact for a plane perpendicular to the optical axis
  and first-order accurate for small tilts. Prefer a rangefinder or the
  barometer where one is available.
- Roll and pitch are not estimated visually. They are taken from the autopilot's
  `ATTITUDE` when fresh, and the covariance tells the EKF not to trust the
  attitude channels it does receive.
