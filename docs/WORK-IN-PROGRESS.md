# Work in progress — resume here

**Status: the measurement is done and the rectification thesis is confirmed
decisively. The open question is no longer whether the matching works but
whether it fits the Pi's time budget — see §3.1, which is now the blocking
item.** Read §1 for the three findings the design rests on, §2 for measured
results, §3 for what to do next.

---

## 1. The three findings that reshaped the design

All verified against ArduPilot / Mission Planner / LightGlue source, not recalled.

### 1.1 A stock MATEK F405-WING cannot accept a visual position at all

`HAL_VISUALODOM_ENABLED`, `EK3_FEATURE_EXTERNAL_NAV` and `AP_GPS_MAV_ENABLED`
are **all 0** on the `MatekF405-Wing` ArduPlane target. `ODOMETRY`,
`VISION_POSITION_ESTIMATE` **and** `GPS_INPUT` are therefore all compiled out,
and every failure is silent. Full detail and the three verification methods:
`docs/flight-controller-compatibility.md`.

Consequence: **`GPS_INPUT` is the primary output path**
(`avinavi/mavlink/gps_input.py`), needing a custom build with GPS Drivers → MAV
at custom.ardupilot.org. `ODOMETRY` is retained for 2 MB boards.

### 1.2 SuperPoint + LightGlue is rotation-variant, and a banking plane breaks it

- LightGlue's synthetic pre-training covered **±31.5°**; its second stage
  fine-tunes on gravity-aligned MegaDepth with essentially no rotation.
- `features["superpoint"]` does not set `add_scale_ori`, so LightGlue is handed
  only `(x, y)` — **no orientation channel exists**. `CrossBlock` has no
  positional encoding: translation invariant by design, rotation variant by
  design.
- ArduPlane's `ROLL_LIMIT_DEG` defaults to **45**. At 100 m AGL / 60° HFOV,
  **10° of bank already gives a 1.5× GSD ratio** across the frame; 30° gives 4×.

**This is now measured, not argued — see §2.1. The real stack scores 0.0% on
unrectified frames.** Hence `avinavi/localization/prewarp.py`. It is a matching
aid only: the homography is composed back to camera pixels and the pose solve is
untouched.

### 1.3 The operator's map click reaches the Pi through specific channels only

ArduPilot's `MAVLink_routing::forward` does not forward a message addressed to a
specific component to other links, and **every** Mission Planner map action is
addressed to component 1. Five channels, ranked, in
`avinavi/mavlink/reanchor.py`; full table in `docs/manual-reanchor.md`.

---

## 2. Measured results

Both runs: 3000 px synthetic basemap at 0.5 m/px, 9 tiles of 1024 px at 30%
overlap, racetrack circuit at 22 m/s / 120 m AGL / 30° max bank with wind, and
a *seasonal variant* of the basemap for the camera so the live imagery is not
pixel-identical to the tiles. Reproduce with `tools/run_experiment.py`.

### 2.1 What rectification is worth

**SuperPoint + LightGlue** (60 cycles per variant):

| Variant | Lock | False fix | Error p50 |
| --- | --- | --- | --- |
| baseline (raw frames) | **0.0%** | — | — |
| + CLAHE only | **0.0%** | — | — |
| + rectify only | **41.7%** | 0.0% | 0.71 m |
| + rectify + CLAHE | *(see results JSON)* | | |

**SIFT + ratio test** (300 cycles per variant, the fast baseline):

| Variant | Lock | False fix | Error p50 | ms/cycle |
| --- | --- | --- | --- | --- |
| baseline (raw frames) | 2.7% | 0.0% | 5.5 m | 83 |
| + CLAHE only | 3.3% | 0.0% | 5.4 m | 153 |
| + rectify only | 10.7% | 0.0% | 0.5 m | 27 |
| + rectify + CLAHE | **22.0%** | 0.0% | 0.2 m | 40 |

Three things to take from this:

1. **Rectification is categorical for the learned matcher, not incremental.**
   0.0% → 41.7%. The §1.2 reasoning predicted exactly this: with no orientation
   channel, SuperPoint cannot describe a rotated patch at all. SIFT, which *is*
   rotation-invariant by construction, merely degrades (2.7%) instead of
   failing. The two backends disagreeing in exactly this way is the strongest
   available evidence that the mechanism is understood correctly.
2. **CLAHE alone achieves nothing** (0.0% → 0.0%, 2.7% → 3.3%). It only pays off
   once the geometry is aligned, where it doubles the SIFT lock rate
   (10.7% → 22.0%). Photometric normalisation cannot fix a geometric problem.
3. **Rectification makes it faster**, not slower — 40 ms against 83 ms for SIFT
   — because the rectified output is capped at 768 px.

Zero false fixes in every variant, and wrong-tile separation was clean
throughout (the deliberately-wrong tile attracted no inliers that survived
verification).

### 2.2 Sensitivity (SIFT, 300 cycles)

| Variation | Lock | vs. baseline 22.0% |
| --- | --- | --- |
| no appearance gap | 43.7% | the seasonal gap costs **half** the lock rate |
| double appearance gap | 31.7% | **non-monotonic — see below** |
| slow shutter 1/60 s | 23.7% | no measurable penalty here |
| + quality gates | 23.3% | gates cost nothing, as intended |

**Two of these are unresolved and must not be quoted as results:**

- *Double* appearance gap scoring **better** than single (31.7% vs 22.0%, 95 vs
  66 fixes — too large to be sampling noise) means `seasonal_variant`'s
  `strength` parameter is **not a valid difficulty dial**. It scales gains and
  brightness offsets together, so at strength 2 it mostly raises contrast, which
  *helps* keypoint detection. The appearance-gap model needs separating into
  "harder" and "merely different" before any sensitivity claim can be made.
- The slow-shutter result (23.7% vs 22.0%) contradicts the motion-blur argument
  in `docs/usb-camera.md`. The difference is within sampling noise at 300
  cycles, and there is a plausible real mechanism — mild blur low-passes a
  0.13 m/px live frame before it is matched against a 0.5 m/px tile, reducing
  aliasing. **This is a SIFT result and SuperPoint behaves differently**; the
  blur argument for locking exposure stands on the keypoint-detection mechanism
  and on the measured optical-flow result (§2.3), not on this row.

### 2.3 Visual odometry

`tests/test_visual_odometry.py` measures that auto-exposure hunting between
consecutive frames costs the frame-to-frame tracking outright — locked exposure
tracks ≥6 of 8 frames, hunting tracks fewer. That is the measured basis for
`camera.lock_exposure`.

Displacement accuracy against simulated ground truth is within 0.25 m over a
2.2 m step, with signs verified in all four directions and under a sustained
25° bank.

---

## 3. Resume here — ordered

### 3.1 The Pi time budget (measured; this is the blocking item)

**Done: the sweep is in `docs/performance.md`.** The finding is that
SuperPoint + LightGlue runs at ~0.4 Hz on four cores and **no tested
configuration reaches 2 Hz**. The architecture is confirmed; the configuration
is not yet flyable.

What the sweep established:

- `max_candidate_tiles` is the only real time lever (4 → 1 cuts 2438 → 1476 ms)
  and it costs half the lock rate (45% → 20%). LightGlue runs once per
  candidate, so this multiplies the dominant cost — and the lock rate falling
  that hard means the search window is not precise enough to put the right tile
  first.
- `resize_max_side` 640 → 384 changes **neither** lock rate nor latency.
  Extraction is not the bottleneck; matching is.
- `max_num_keypoints` 1024 → 512 is nearly free (45.0% → 42.5%, better median
  error); 512 → 256 collapses to 2.5%. A cliff, not a slope.

So the remaining work, in value order:

1. **Tighten the search window** so fewer candidate tiles are needed. This is
   the one change that buys time *and* hit rate together, and the sweep points
   straight at it. The frame-to-frame odometry narrowing the window between map
   fixes is the mechanism; it is implemented and wired but the window still
   comes from `tile_search.search_radius_m` alone.
2. **ONNX Runtime** (~2.4× reported) — best effort-to-benefit ratio available,
   untried here.
3. **XFeat + LighterGlue** (4–8× faster for ~5% relative AUC) would put 2 Hz in
   reach; larger change, needs its own A/B.
4. Set `rates.visual_localization_hz` from the measured **p95**.

**Do not pursue a Hailo NPU** — LightGlue's dynamic keypoint count and
data-dependent early exit are incompatible with its static dataflow compiler.

A caution the sweep itself demonstrated: **run it with nothing else on the
machine.** Several p95 figures in the first run (6456, 4681, 12545 ms) are
inconsistent with their own means because I was running tests on the same four
cores at the time. Lock rates are unaffected; timings are not.

### 3.2 Fix the harness weaknesses §2.2 exposed

- `seasonal_variant(strength=…)` is not a difficulty dial. Separate the
  *geometric/structural* change (things built and removed) from the
  *photometric* change (gain, gamma, vegetation tone), so one can be swept
  without the other accidentally helping.
- Re-run the shutter sensitivity with the SuperPoint backend and more cycles
  before either keeping or retracting the §2.2 row.

### 3.3 Remaining work, in value order

- **Validate the localizability score against outcomes.** The score exists and
  its repetition measure is shown to separate repetitive from varied ground on
  real features. What is *not* shown is that `score` ranks tiles in the order
  they actually localize. Doing it needs per-tile lock rates, which
  `SimulationReport` does not record — add a `by_tile` counter and correlate.
  Until then the module docstring says plainly what is and is not established.
- **Mission Planner plugin** (`tools/mission_planner_plugin/`) is written but
  **never compiled or run**. It needs a Windows machine with Mission Planner.
- **Tests for the simulator itself.** Everything else now has them, and the
  simulator is what every measurement rests on — which is precisely why its
  two bugs (§5) were so expensive.
- **A per-stage timing breakdown.** `docs/performance.md` establishes that
  matching dominates, but not how the time splits between rectification,
  extraction and each candidate tile's LightGlue call. Worth having before
  optimising anything.

Done since this plan was written: the odometry is wired into the runner
(§5 note: with its own tests), the localizability score is in `map-info`,
`docs/matching-robustness.md`, `docs/performance.md`, `docs/usb-camera.md` and
`docs/manual-reanchor.md` are written, and `opencv_backend` has tests.

---

## 4. Things that will bite

- `pip` → Python 3.13, `python3` → 3.11. **Always `python3 -m pip`.**
- The container was recreated once mid-session and all pip packages vanished.
  Re-install before assuming anything works.
- **`/tmp/.../scratchpad/reference_reading/` holds copies of `lightglue.py` and
  `superpoint.py`** downloaded for reading. They were originally in the
  scratchpad root, where Python put them *first* on `sys.path` for any script
  run from there — shadowing the real installed package and producing a
  confusing `ImportError: cannot import name 'SuperPoint' from 'lightglue'`.
  Run scratchpad scripts with `python3 -P` to keep the script's directory off
  `sys.path`.
- `download.pytorch.org`, `ardupilot.org` and `mavlink.io` are **403 via the
  egress proxy**. `raw.githubusercontent.com` and PyPI work.
- `lightglue` is **not on PyPI** — clone `cvg/LightGlue` and `pip install -e`.
  Pillow needed a `--force-reinstall`.
- **`normalize_keypoints` must receive the true image size.** A port that passed
  a hardcoded size put coordinates 2.3× outside the trained range and collapsed
  match rates silently. `SuperPointExtractor` sets `image_size` from the
  original frame and rescales keypoints to match — keep that invariant.
- **A background run piped through `grep` shows no interim output**, because
  grep buffers. Use `python3 -u` and filter when reading the log, not in the
  pipeline.
- **`pgrep -f` and `pkill -f` match the watcher's own command line.** A wait
  loop written as `until ! pgrep -f "tools/run_experiment"; do …` never exits,
  because the shell running it contains that string; and `pkill -f
  run_experiment` once killed the shell that was about to start a job, since the
  heredoc it was writing mentioned the name. Wait on a **PID** (`kill -0 $pid`)
  instead, and check for a job with a pattern specific enough to exclude the
  checker (`pgrep -af "python3 -u tools/…"`).
- **The repository has no base branch**, so the draft PR cannot be created yet
  — `main` does not exist and the feature branch is the only history. Creating
  it is a structural decision for the repository owner.

---

## 5. Three bugs worth remembering, all of the same shape

Both produced a *confident, plausible, consistently wrong* answer with no error
anywhere. That is the failure mode this whole project is built to refuse, and
both were caught only by checking numbers against arithmetic rather than
trusting that a passing run meant a correct one.

1. **A stale `index.json`** overrode the tiles' own georeferencing after a
   rebuild, so every reported position carried a constant 100 m offset. Found by
   recording the error *vector* instead of only its magnitude: a 141 m error
   with 0.1 m of scatter is not a matcher failing, it is an offset. Fixed in
   `MapDatabase.reload()`, which now revalidates the index against the tiles'
   timestamps and the tile set on disk; two regression tests cover it.

2. **`run_simulation` built an OpenCV extractor unconditionally**, so every run
   requested with `--backend superpoint` silently measured SIFT. The giveaway
   was the two backends agreeing to within 0.3% — which should have been
   suspicious immediately, and was not.

3. **`resize_max_side=None` could not mean "do not resize".** With
   `value if value is not None else self.value`, passing `None` to mean *full
   resolution* collapsed into *use the configured resize*, so
   `build_tile_database` — which passes `None` precisely because a 1024 px tile
   should be described at full resolution — had its tiles quietly downscaled to
   the live camera's working size. Fixed with an explicit sentinel
   (`avinavi/features/resize.py`), and found by a test that asserted the
   override worked. It did not.

The pattern across all three: **a passing run is not a correct one.** Each was
caught by checking a number against arithmetic, or by asserting that a knob did
what its name said — never by the code failing. Anything in this codebase that
reports a measurement deserves the same suspicion.
