# Why every frame is rectified before matching

**Summary: on a fixed wing, matching raw camera frames against north-up map
tiles with SuperPoint + LightGlue does not work at a reduced rate. It does not
work at all.** Measured: **0.0%** lock on raw frames, **41.7%** once each frame
is rectified to a synthetic nadir, north-up, tile-scale view.

This page gives the mechanism first, then the measurements, then what the
measurements do *not* support.

## The mechanism

Three gaps separate a live frame from a basemap tile, and a learned sparse
matcher is strong against only one of them.

**Rotation.** A fixed wing flying a circuit presents the ground rotated by up to
180° relative to a north-up tile. Two properties of the stack make this fatal
rather than merely hard:

- LightGlue's synthetic pre-training covered **±31.5°** of rotation
  (`max_angle: 45` × `difficulty: 0.7` in its homography config). Its second
  stage fine-tunes on gravity-aligned MegaDepth, which contributes essentially
  no rotation diversity.
- `features["superpoint"]` does not set `add_scale_ori`, so LightGlue receives
  only `(x, y)` per keypoint — **there is no orientation channel at all**. And
  `CrossBlock` has no positional encoding: translation invariant by design,
  rotation variant by design.

So SuperPoint cannot describe a rotated patch, and LightGlue has no mechanism to
compensate. This is a structural property, not a tuning problem.

**Perspective and scale anisotropy.** Bank tilts the optical axis, and the
ground sample distance then varies across the frame. At 100 m AGL with a 60°
horizontal field:

| Bank | Near/far GSD ratio across the frame |
| --- | --- |
| 10° | **1.5×** |
| 20° | 2.4× |
| 30° | 4.0× |
| 45° (`ROLL_LIMIT_DEG` default) | ~9× |

Learned sparse matching degrades badly past roughly 1.2× and fails by 1.5×. So
**10° of bank is already enough**, and ArduPlane permits 45 by default while
stall prevention always allows at least 25. A survey circuit spends a large
fraction of its time past that threshold — the simulated trajectory used below
is above 15° of bank for 19% of its flight.

**Altitude scale.** The basemap has one ground resolution; the aircraft flies at
whatever altitude it flies at. A 2× scale mismatch is ordinary.

## What rectification does

Each frame is warped into the view a *virtual* camera would see if it were
pointing straight down, oriented north-up, and sampling the ground at the
basemap's metres-per-pixel. All three gaps close geometrically, using
information already available: the autopilot's `ATTITUDE` and a height above
terrain. What is left for LightGlue is close to a pure translation, which is the
regime it is strongest in.

The derivation is in `avinavi/localization/prewarp.py`. The important property
is that **this is only a matching aid**. The homography recovered against the
tile is composed back through the warp:

```
frame_from_tile = camera_from_prepared @ prepared_from_tile
```

and handed to the unchanged pose solver. No approximation in the warp leaks into
the position estimate. If the attitude or height used for the warp is wrong, the
warp is simply less effective at closing the gap — the resulting match is still
verified by MAGSAC and solved exactly.

## Measurements

Synthetic 3000 px basemap at 0.5 m/px, cut into 9 tiles of 1024 px at 30%
overlap. Racetrack circuit at 22 m/s, 120 m AGL, 30° maximum bank, with wind
(so there is a crab angle between heading and track). **The camera renders from
a *seasonal variant* of the basemap** — different gains, gamma, vegetation tone,
large-scale patchiness and some structures added and removed — so the live
imagery is never pixel-identical to the tiles it is matched against. Motion
blur, rolling shutter, vignetting, JPEG artefacts and sensor noise are all
simulated.

Reproduce with `tools/run_experiment.py`.

### SuperPoint + LightGlue (60 cycles per variant)

| Variant | Lock | False fix | Error p50 | ms/cycle |
| --- | --- | --- | --- | --- |
| baseline (raw frames) | **0.0%** | — | — | 1828 |
| + CLAHE only | **0.0%** | — | — | 1870 |
| + rectify only | **41.7%** | 0.0% | 0.71 m | 2134 |
| + rectify + CLAHE | 35.0% | 0.0% | 0.78 m | 2223 |

### SIFT + ratio test (300 cycles per variant)

The classical backend, kept as a measurement baseline and a no-torch fallback.

| Variant | Lock | False fix | Error p50 | ms/cycle |
| --- | --- | --- | --- | --- |
| baseline (raw frames) | 2.7% | 0.0% | 5.5 m | 83 |
| + CLAHE only | 3.3% | 0.0% | 5.4 m | 153 |
| + rectify only | 10.7% | 0.0% | 0.5 m | 27 |
| + rectify + CLAHE | **22.0%** | 0.0% | 0.2 m | 40 |

### Reading these

**1. Rectification is categorical for the learned stack and merely helpful for
the classical one.** 0.0% → 41.7% against 2.7% → 10.7%.

That asymmetry is the strongest evidence that the mechanism above is understood
correctly, because it is what the mechanism predicts and it is not what a
generic "preprocessing helps" story would predict. SIFT computes a dominant
gradient orientation per keypoint and is rotation-invariant by construction, so
rotation *degrades* it. SuperPoint has no orientation channel, so rotation
*defeats* it. If rectification had simply improved both backends by a similar
factor, the explanation would be in doubt.

**2. Rectification also makes it faster.** 27 ms against 83 ms for SIFT, because
the rectified output is capped at `prewarp.max_output_side` (768 px) while a raw
1280×720 frame is not. For SuperPoint the warp cost shows up as a small net
increase, since extraction already resizes.

**3. CLAHE cannot fix a geometric problem.** On raw frames it achieves nothing
(0.0% → 0.0%; 2.7% → 3.3%). It only pays off once the geometry is aligned —
where it doubles the SIFT lock rate (10.7% → 22.0%).

**4. CLAHE shows no benefit for SuperPoint.** 41.7% → 35.0%, i.e. 25 fixes
against 21 out of 60 — a difference well within sampling noise at this cycle
count, so it should **not** be read as CLAHE being harmful. There is a plausible
mechanism for a real effect in that direction, namely that learned descriptors
are trained on natural image statistics and histogram equalisation moves images
away from that distribution. But this measurement does not establish it. The
default stays on; if you are running the SuperPoint backend and want the
setting decided rather than inherited, measure it with several hundred cycles.

**5. Zero false fixes everywhere**, and the deliberately-wrong tile attracted no
inliers that survived verification in any variant. This matters more than the
lock rate: a system that locks 40% of the time and is never badly wrong is
useful, while one that locks 95% of the time and is occasionally 6 km out is
dangerous, because the EKF has no way to reject it.

## What these numbers are not

**They are optimistic in absolute terms.** The terrain is synthetic and
therefore more uniformly textured than much real ground, and the appearance gap
is modelled rather than real. Treat the **comparisons** as the result and the
absolute lock rates as an upper bound.

**Two sensitivity runs came out non-monotonic and are not usable:**

| Variation (SIFT, 300 cycles) | Lock | |
| --- | --- | --- |
| no appearance gap | 43.7% | the gap costs half the lock rate, as expected |
| **double appearance gap** | **31.7%** | **higher than the single gap's 22.0%** |
| slow shutter 1/60 s | 23.7% | no penalty, against 22.0% |
| + quality gates | 23.3% | gates cost nothing, as intended |

The double-gap result (95 fixes against 66 — too large for sampling noise) means
`seasonal_variant`'s `strength` parameter is **not a valid difficulty dial**: it
scales gains and brightness offsets together, so at strength 2 it mostly raises
contrast, which *helps* keypoint detection. The appearance model needs its
structural changes separated from its photometric ones before any sensitivity
claim can be made from it.

The shutter result appears to contradict `docs/usb-camera.md`. The difference is
within sampling noise, and there is a plausible real mechanism — mild blur
low-passes a 0.13 m/px live frame before it is matched against a 0.5 m/px tile,
reducing aliasing. It is also a SIFT result, and SuperPoint's keypoint detector
behaves differently. **The case for locking the exposure does not rest on this
row**; it rests on the detection mechanism and on a separate measured result:
auto-exposure hunting between consecutive frames costs the frame-to-frame
odometry its tracking outright (`tests/test_visual_odometry.py`).

## The cost, and the open question

At 2.1 s per cycle for the real stack on a CPU, matching is no longer the
limiting factor — **time is**. A fixed wing covers 46 metres in 2.1 s, so a 41.7%
lock rate at 0.5 Hz is a very different thing from 41.7% at 2 Hz.

Two things address this, and both are in progress:

- `tools/run_speed_sweep.py` measures the lock rate against the per-cycle
  latency over `max_candidate_tiles`, `resize_max_side` and `max_num_keypoints`,
  to find the best configuration that fits a chosen budget rather than either
  extreme. Note that LightGlue runs **once per candidate tile**, so the search
  window's tightness multiplies the dominant cost directly.
- Frame-to-frame visual odometry (`avinavi/localization/visual_odometry.py`)
  carries the solution across the gap between map fixes, with the drift reported
  as a growing position sigma rather than hidden.

See `docs/WORK-IN-PROGRESS.md` §3.1 for the current state of that work.
