# How fast this runs, and what that costs

**Summary: SuperPoint + LightGlue at the default settings runs at roughly
0.44 Hz on four CPU cores — not the 2 Hz the configuration assumes. No tested
configuration reaches 2 Hz. This is now the limiting factor in the whole
system, not the matching.**

In navigation terms: **one map fix every ~90 metres of flight**, at 55% lock and
0.29 m median error. That is survivable — the arithmetic is below — but only
because the frame-to-frame odometry carries the gap, and only once that odometry
runs on its own schedule rather than the localization loop's.

## Measured

Four cores, `torch.get_num_threads() == 4`. A Raspberry Pi 5 also has four
cores, so these figures are a reasonable proxy — and if anything optimistic,
since these are x86 cores.

Reproduce with `tools/run_speed_sweep.py`. Latency is per localization cycle,
including rectification, extraction, and matching against every candidate tile.

The last two columns are the ones that matter for navigation. **Lock rate alone
is misleading** when cycle times differ by 50%: a configuration that locks half
as often but runs twice as fast delivers the same number of fixes, and it is the
*gap between fixes* that the dead reckoning has to cover, not the ratio.

| Setting | Lock | Error p50 | ms mean | ms p95 | fix/s | Gap at 22 m/s |
| --- | --- | --- | --- | --- | --- | --- |
| reference (4 tiles, 640 px, 1024 kp) | **55.0%** | 0.29 m | 2260 | 2759 | 0.243 | 90 m |
| `max_candidate_tiles` 2 | 27.5% | 0.51 m | 1768 | 2131 | 0.156 | 141 m |
| `max_candidate_tiles` 1 | 27.5% | 0.51 m | 1522 | 1764 | 0.181 | 122 m |
| `resize_max_side` 512 | **55.0%** | 0.29 m | 2227 | 2813 | **0.247** | **89 m** |
| `resize_max_side` 384 | **55.0%** | 0.29 m | 2230 | 2734 | **0.247** | **89 m** |
| `max_num_keypoints` 512 | 40.0% | 0.26 m | 1739 | 1903 | 0.230 | 96 m |
| `max_num_keypoints` 256 | 5.0% | 0.25 m | 1554 | 1679 | 0.032 | 684 m |
| 2 tiles + 512 px | 27.5% | 0.51 m | 1680 | 1986 | 0.164 | 134 m |
| 2 tiles + 512 kp | 12.5% | 0.21 m | 1482 | 1651 | 0.084 | 261 m |
| 1 tile + 512 px + 512 kp | 12.5% | 0.21 m | 1442 | 1647 | 0.087 | 254 m |

Zero false fixes in every configuration. Every p95 is within ~1.25× of its mean,
which is what a clean measurement looks like — an earlier version of this table
had p95 values of 6456 and 12545 ms because other work was sharing the four
cores, and those were discarded rather than published.

**The headline number is the last column: at best, one map fix every ~90 metres
of flight.** Everything about the system's viability follows from that.

## What the sweep says

**1. `max_candidate_tiles` is the only real time lever, and it is a bad
trade.** 4 → 2 cuts the mean from 2260 ms to 1768 ms but halves the lock rate,
55.0% → 27.5%, so the fixes per second get *worse* (0.243 → 0.156). LightGlue
runs **once per candidate tile**, so this multiplies the dominant cost directly.

The shape of that loss is informative: 2 → 1 candidate costs nothing further
(27.5% either way) while 4 → 2 costs half. Something about the candidate *list*
rather than its radius is responsible, since the right tile is evidently
reachable with four slots and not with two.

**One explanation has been tested and rejected.** The obvious suspect was the
ordering: `MapDatabase.candidates` ranked tiles by distance to the tile
footprint, which is exactly zero for every tile containing the aircraft, so
among the two to four overlapping tiles it sits inside, the order fell through
to an alphabetical `tile_id` tie-break. Distance to the tile *centre* is better
motivated — a frame near a tile's centre has its whole footprint inside that
tile — and the tie-break was duly fixed. **It made no difference at all:**
reference 55.0% → 55.0%, two tiles 27.5% → 27.5%, one tile 27.5% → 27.5%. The
change is retained because the old ordering was arbitrary and the new one is
not, but it does not explain the observation and is not an improvement.

The next candidate explanation, under test, is the **last-matched-tile
preference**. `tile_search.prefer_last_matched_tile` promotes the previous
cycle's tile to the front of the list, and `candidates` applies `limit`
*after* promotion — so with `max_candidate_tiles = 1` the only tile tried is
the one that matched last time, wherever the aircraft has since flown. That
would explain a halving directly, and it predicts that turning the preference
off should raise the one- and two-candidate rows without touching the
four-candidate one.

**2. Frame resolution is free to reduce, and buys nothing.** 640 → 512 → 384 all
give 55.0% lock at ~2230 ms. Measured cleanly, with the tile resolution no
longer entangled with this setting. Extraction is **not** the bottleneck;
matching is, and the cost is dominated by the ~1024 keypoints per *tile* rather
than by the live frame.

**3. Keypoint count is a real trade with a cliff at the end.** 1024 → 512 costs
15 points of lock (55.0% → 40.0%) and saves 23% of the time (2260 → 1739 ms),
which very nearly breaks even on fixes per second (0.243 → 0.230) while leaving
more headroom — a defensible choice. 512 → 256 collapses to 5.0%. Note this
setting applies to the **tile** features too, which is why it moves the clock at
all where `resize_max_side` does not.

**4. Nothing fits 2 Hz.** The fastest configuration tested is 1442 ms mean,
1647 ms p95 — more than three times over a 500 ms budget, for 12.5% lock.

The budget has to be judged on **p95, not the mean**. A rate set from the mean
means the loop silently falls behind on exactly the hard frames, which are the
ones where a fix matters.

## Is a 90 metre gap survivable?

This is the question the whole table exists to answer, and the arithmetic is
encouraging — with one condition that turned out to matter a great deal.

At `visual_odometry.drift_fraction` of 0.05, carrying the position over 90 m of
flight adds **4.5 m** of uncertainty. Against a `false_fix_threshold_m` of 60 m
and a map-fix error of 0.29 m median, that is comfortable; the dead-reckoned
sigma reported to the EKF grows from ~1 m to ~4.6 m between fixes and then
collapses again. `max_dead_reckoning_s` of 10 s covers 220 m, so the limit is
generous relative to the 90 m gap and the system goes silent only after roughly
two consecutive missed fixes.

**The condition: the odometry has to run on its own schedule, not the
localization loop's.** Lucas-Kanade needs consecutive frames to overlap heavily,
and that was measured:

| Frame gap | Baseline at 22 m/s | Frame overlap | Result |
| --- | --- | --- | --- |
| 0.1 s | 2 m | 98% | 210 points, error 0.00 m |
| 0.5 s | 11 m | 92% | 123 points, error 0.00 m |
| 1.0 s | 22 m | 84% | **unusable** |
| 2.3 s | 51 m | 63% | **unusable** |

Tracking between *localization* frames, 2.3 s apart, would produce nothing at
all — the dead reckoning would be present and inert exactly when needed. Hence
`rates.visual_odometry_hz`, default 5 Hz, on its own thread. **Do not tie it to
`visual_localization_hz`.**

And a sharper lesson from the same measurement: at the 1 s gap the tracker did
not fail cleanly. It returned 7.4 m for a true 22 m, from 23 points that
disagreed among themselves by 14.7 m — enough points to pass the minimum count,
with nothing checking whether they agreed. There is now a residual gate for
exactly this, because dead reckoning on that estimate would walk the solution
away from the aircraft at a third of its true speed, confidently.

## What to do about it

In order of how much they are worth:

**Fly higher — this is the largest single lever found so far, and it is free.**
Measured over 150 cycles per altitude on one trajectory, one map and one
camera, with every variant rendering all 150 frames:

| Cruise | Rectified frame | Lock | Error p50 | Inliers |
| --- | --- | --- | --- | --- |
| 120 m | 277 × 156 px | 59.3% | 0.49 m | 42 |
| 180 m | 416 × 234 px | 91.3% | 0.26 m | 79 |
| **240 m** | **554 × 312 px** | **100.0%** | **0.18 m** | 120 |
| 300 m | 693 × 390 px | 100.0% | 0.20 m | 141 |
| 360 m | 831 × 467 px | 100.0% | 0.25 m | 152 |

Zero false fixes throughout. Going from 120 m to 240 m is a change to the
flight plan, not to the code, the camera or the map, and it is worth **forty
points of lock rate** while nearly tripling the inlier count.

The reason is visible in the second column. Rectification resamples the frame to
the tile's ground sample distance, so the image SuperPoint actually sees is
`footprint / tile_gsd` on a side. At 120 m against a 0.5 m/px map that is 277 px
wide — from a 1280 px sensor. Ninety-five per cent of the pixels are thrown away
before any matching happens. Flying higher widens the footprint, so more of the
sensor survives the warp.

**There is an optimum, and it is not "as high as possible."** Error bottoms out
at 240 m and then worsens: 0.18 m, 0.20 m, 0.25 m. Two effects pull against each
other — more inliers stabilise the solve, while a coarser live GSD (0.108 m/px
at 120 m, 0.325 m/px at 360 m) makes every pixel of matching error worth three
times as many metres. Below 240 m the first wins; above it the second does.

**A wider lens buys the same thing without the altitude — up to a limit.**
120 m with a 100 degree lens covers 286 m of ground and rectifies to 572 px,
near-identical to 240 m with a 60 degree lens at 554 px. They measured 125
inliers at 0.17 m against 120 inliers at 0.18 m: a twofold difference in
altitude for a 4% difference in result.

The limit is hard and geometric. At 120 degrees the frame edge sits 60 degrees
from nadir, so at 30 degrees of bank it reaches the horizon and those frames
cannot be rendered at all. In the air that is not a dropped frame, it is a frame
with sky in it. **Keep `half the field of view + the bank limit` below about 80
degrees**: with ArduPlane's 30 degree stall-prevention floor that caps the
usable lens at about 100 degrees.

**Accept a lower rate and let dead reckoning cover the gap.** This is the
honest default, and it is what the architecture was built for:
`rates.visual_localization_hz` set from the measured p95, with
`avinavi/localization/visual_odometry.py` interpolating between map fixes and
reporting its drift as a growing sigma. A 0.4 Hz map fix rate with sound dead
reckoning is a usable navigation system; a 2 Hz rate that misses its deadline is
not.

**Improve the search window's ordering**, per finding 1. The true tile sitting
third or fourth in the candidate list is costing either half the lock rate or
40% of the cycle time, and neither needs to be paid.

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
