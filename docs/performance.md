# How fast this runs, and what that costs

**Summary: SuperPoint + LightGlue at the default settings runs at roughly
0.4 Hz on four CPU cores — not the 2 Hz the configuration assumes. No tested
configuration reaches 2 Hz. This is now the limiting factor in the whole
system, not the matching.**

Lock rate alone is not the figure of merit. A fixed wing covers 22 m/s, so 45%
of cycles locking at 0.4 Hz is a very different proposition from 45% at 2 Hz:
the first gives a map fix roughly every 5.5 seconds, or **120 metres of
flight**, and the frame-to-frame odometry has to carry everything in between.

## Measured

Four cores, `torch.get_num_threads() == 4`. A Raspberry Pi 5 also has four
cores, so these figures are a reasonable proxy — and if anything optimistic,
since these are x86 cores.

Reproduce with `tools/run_speed_sweep.py`. Latency is per localization cycle,
including rectification, extraction, and matching against every candidate tile.

| Setting | Lock | Error p50 | ms mean | ms p95 |
| --- | --- | --- | --- | --- |
| reference (4 tiles, 640 px, 1024 kp) | **45.0%** | 0.65 m | 2438 | 3054 |
| `max_candidate_tiles` 2 | 20.0% | 0.60 m | 1929 | 2357 |
| `max_candidate_tiles` 1 | 20.0% | 0.60 m | **1476** | **1701** |
| `resize_max_side` 512 | 45.0% | 0.65 m | 2339 | 2831 |
| `resize_max_side` 384 | 45.0% | 0.65 m | 2345 | 2742 |
| `max_num_keypoints` 512 | 42.5% | 0.32 m | 2414 | — † |
| `max_num_keypoints` 256 | **2.5%** | 0.42 m | 2044 | — † |
| 2 tiles + 512 px | 20.0% | 0.60 m | — † | — † |
| 2 tiles + 512 kp | 17.5% | 0.32 m | 1788 | 1978 |
| 1 tile + 512 px + 512 kp | 17.5% | 0.32 m | 1504 | 1745 |

† Some p95 figures from that run are unreliable: other work was running on the
same four cores during part of it, which shows up as p95 values of 6456, 4681
and 12545 ms that are inconsistent with their own means. The *lock rates* from
those rows are unaffected, since they do not depend on timing. The clean rows
are the ones with a plausible mean-to-p95 ratio.

Zero false fixes in every configuration.

## What the sweep says

**1. `max_candidate_tiles` is the only real time lever, and it is expensive.**
Going from 4 candidates to 1 cuts the mean from 2438 ms to 1476 ms — but halves
the lock rate, from 45% to 20%. LightGlue runs **once per candidate tile**, so
this multiplies the dominant cost directly; and the lock rate falling that
sharply says the search window is not precise enough to put the right tile
first. Which points at the cheapest available win: **a tighter search window
buys both time and hit rate at once**, because fewer candidates would be needed
for the same result. The frame-to-frame odometry narrowing the window between
map fixes is exactly that mechanism.

**2. Frame resolution is nearly free to reduce, and buys nothing.** 640 → 384
changes neither lock rate (45.0% either way) nor latency (2438 → 2345 ms). So
extraction is **not** the bottleneck; matching is. Lowering the live frame
resolution is not a useful lever here.

**3. Keypoint count has a cliff.** 1024 → 512 costs almost nothing in lock rate
(45.0% → 42.5%) and actually *improves* the median error. 512 → 256 collapses it
to 2.5%. So 512 is close to free and 256 is unusable; there is no gentle slope
to trade along.

**4. Nothing fits 2 Hz.** The fastest configuration tested is 1504 ms mean,
1745 ms p95 — three and a half times over a 500 ms budget, for 17.5% lock.

The budget has to be judged on **p95, not the mean**. A rate set from the mean
means the loop silently falls behind on exactly the hard frames, which are the
ones where a fix matters.

## What to do about it

In order of how much they are worth:

**Accept a lower rate and let dead reckoning cover the gap.** This is the
honest default, and it is what the architecture was built for:
`rates.visual_localization_hz` set from the measured p95, with
`avinavi/localization/visual_odometry.py` interpolating between map fixes and
reporting its drift as a growing sigma. A 0.4 Hz map fix rate with sound dead
reckoning is a usable navigation system; a 2 Hz rate that misses its deadline is
not.

**Change the runtime, not the model.** ONNX Runtime is reported at about 2.4×
PyTorch for this workload, which would bring the reference configuration to
roughly 1 s per cycle without touching accuracy. That is the best
effort-to-benefit ratio available and it has not been tried here.

**Change the model.** XFeat + LighterGlue is reported 4–8× faster for about 5%
relative AUC, which would put 2 Hz within reach. It is a larger change and would
need its own A/B before being trusted.

**Use the classical backend.** SIFT + ratio test runs the whole cycle in 40 ms —
sixty times faster — for 22% lock against 45%. That is a real option for a first
flight: `features.backend: opencv`. It is also rotation-invariant by
construction, so it degrades rather than fails when rectification is imperfect
(see `docs/matching-robustness.md`).

**Do not pursue a Hailo or similar NPU.** LightGlue's keypoint count varies per
frame and its early exit is data-dependent, and both are incompatible with a
static dataflow compiler. The effort would be large and the outcome is
predictable.

## Where the time is not

Two things that might look like candidates and are not:

- **Rectification.** It is a warp of an image capped at 768 px. With the
  classical backend it makes the cycle *faster* overall (40 ms against 83 ms),
  because everything downstream then works on a smaller image.
- **The tile database.** Features are precomputed offline, and the feature cache
  (`map.feature_cache_size`) keeps the working set in memory. Loading a tile's
  features is not in the inner loop.

## Measuring it on your own hardware

```bash
# The sweep, against your own map database and camera settings
python3 tools/run_speed_sweep.py --backend superpoint --max-cycles 40 --budget-s 0.5

# Then set the rate from the p95, not the mean
avinavi run --set rates.visual_localization_hz=0.4
```

Run it with nothing else on the machine. The contaminated rows in the table
above are a demonstration of why: four cores shared with anything else produce
p95 figures that are not about the pipeline at all.
