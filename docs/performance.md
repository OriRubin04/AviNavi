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

**Give the rectified frame more pixels. This is the largest lever found so far,
and altitude is only one of three ways to pull it.**

Rectification resamples the frame to the tile's ground sample distance, so the
image SuperPoint actually sees is

```
rectified width = ground footprint / tile GSD
```

which depends on neither altitude nor sensor resolution directly. At 120 m
behind a 60 degree lens against a 0.5 m/px map that is **277 px wide — from a
1280 px sensor.** Ninety-five per cent of the pixels are discarded before any
matching happens.

Measured over 150 cycles per variant on one trajectory, one map and one camera,
with every variant rendering all 150 frames:

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

### It is the pixel count, not the altitude

Altitude and rectified size move together in that table, so it does not separate
them. This does.

Starve the keypoint budget to 512 so nothing sits at the 100% ceiling, then
compare three variants — one that shares *altitude* with the failure and
*rectified size* with the success:

| Variant | Altitude | Rectified | Lock | Inliers |
| --- | --- | --- | --- | --- |
| 120 m, 60 deg | 120 m | 277 px | **17.3%** | 29 |
| 120 m, 100 deg | **120 m** | **572 px** | **100.0%** | 87 |
| 240 m, 60 deg | 240 m | 554 px | 100.0% | 82 |

The middle row flies at exactly the altitude of the 17.3% failure and scores
100%, tracking its rectified-size twin (87 inliers against 82) rather than its
altitude twin (29). The prediction was written before the run and could have
come out the other way; it did not.

**Altitude was never the cause. It is one way to widen the footprint.**

Read that 17.3% carefully, though — it is not a uniformly hard run. Its failure
runs were `[116, 4, 1, 1, 1, 1]`: **one blackout of 116 consecutive cycles**
accounts for 116 of the 124 failures. The two wide-footprint variants recorded
no failure runs at all, `count: 0`. So the honest statement is not "small
rectified frames score lower", it is **"small rectified frames can lose the
thread completely and not get it back"** — which is worse, and is the subject of
the next section.

### Three routes to the same place

| Route | Cost | Ceiling |
| --- | --- | --- |
| Fly higher | free — a flight-plan change | past 240 m precision degrades (below) |
| Wider lens | a lens | ~100 deg, set by the bank limit (below) |
| Finer map | disk and build time | none found; **not measured** |

The third is predicted by the formula and untested here: the simulator renders
the camera from the same raster it tiles, so a finer map would also be a finer
*world*, with terrain detail that does not exist at 0.5 m/px. That is a
confound, not a measurement. The other two routes agreeing is the evidence for
the formula; the third follows from it.

As a rule of thumb from these runs: **below ~300 px rectified the system
struggles, above ~550 px it works.**

### The false-fix rate is not unconditionally zero

Every variant in the first table reported 0.0% false fixes, and so has every
run in `docs/matching-robustness.md`. That is not an unconditional property of
the pipeline, and the starved run above is where it broke: **120 m at a 512
keypoint budget produced 3.8% false fixes** — the first non-zero false-fix rate
measured in this project.

The mechanism is not mysterious. Geometric verification needs enough correct
correspondences to outvote the wrong ones; with 29 inliers on a 277 px image
that margin is thin enough for RANSAC to settle on a consensus that is merely
self-consistent. Every variant at 550 px or more stayed at 0.0%.

So "no fake positions" holds where matches are plentiful and degrades where they
are scarce — which is a third, independent argument for the same change, and a
reason to treat `geometry.min_inliers` as a floor that should rise, not fall, if
the rectified frame is small.

**And a false fix does not cost one bad position — it cost the rest of this
flight.** Measured, not inferred. The run was repeated with the cycle index of
every false fix and of every failure run recorded:

```
false fix at cycle      : 33
failure runs started at : 17, 22, 26, 28, 32, 34
failure run lengths     : 4, 1, 1, 1, 1, 116
```

| Cycle | |
| --- | --- |
| 17, 22, 26, 28, 32 | short failure runs of 4, 1, 1, 1, 1 — **recovered from every one** |
| **33** | **a false fix is accepted** |
| **34** | the blackout begins, **the very next cycle** |
| 34–149 | 116 consecutive failures, to the end of the run |

34 + 116 = 150: it never re-acquired. Before cycle 33 the system fell and got up
five times; after cycle 33 it never got up.

The mechanism is structural. The tile search is centred on the last accepted
position, so a fix on the wrong ground moves the search window to the wrong
ground, every candidate tile afterwards is drawn from the wrong place, and there
is nothing left that can recover it.

This is the same path dependence that made `max_candidate_tiles: 2` halve the
lock rate while 91% of fixes came from rank 0, which had no explanation until
now. It raises the stakes on the confidence gate considerably: **a single false
fix can blind the system for the rest of the flight**, so the cost of accepting
a doubtful fix is not one bad position but every position after it. A re-anchor
path exists for exactly this (`docs/manual-reanchor.md`) and this is the
strongest argument yet for keeping it reachable in flight.

It also means **lock rate is a noisy measure wherever a blackout is possible**,
and it should not be read as difficulty. The same variant scored 17.3% at a 512
keypoint budget and 49.3% at 384 — better on the smaller budget — because the
512 run fell into a terminal blackout and the 384 run did not. Nothing about the
task got easier. Inlier counts (29 against 24) and failure-run lengths describe
these runs honestly; a single lock-rate number does not.

**A blackout does not require a false fix, though.** The same variant at a 384
keypoint budget reported `NOT supported`:

```
false fix cycles      : (none)
failure runs started at: 17, 28, 32, 86, 93, 113, 134
failure run lengths    : 53, 9, 6, 5, 1, 1, 1
```

No false fix anywhere, and still a 53-cycle run of failures. It is also certain
that the system **recovered** from it: the last failure run starts at cycle 134,
and a 53-cycle run starting there would end past cycle 150, so the long run is
not the last one — there were valid fixes after it.

So the two observations separate into a distinction worth keeping:

| | Blackout happens? | Recovers? |
| --- | --- | --- |
| No false fix (384) | yes, 53 cycles | **yes** |
| After a false fix (512) | yes, 116 cycles | **never** |

Which fits the structure: without a false fix the search centre is still roughly
right and merely drifting, so a later frame can re-acquire. After a false fix it
is actively wrong, and nothing in the search can find its way back.

*Scope. Each row of that table is a single observation. The measurement
establishes that a false fix can end a flight's localization, and that blackouts
also occur without one. It does not establish a rate for either, and it does not
establish that every false fix is terminal — that needs many runs containing
false fixes, and the only configuration found so far that produces any is one
this document recommends against flying.
`SimulationReport.blackouts_after_false_fix()` is the measurement; its tests pin
the negative cases, and it did return `NOT supported` here.*

**There is an optimum, and it is not "as high as possible."** Error bottoms out
at 240 m and then worsens: 0.18 m, 0.20 m, 0.25 m. Two effects pull against each
other — more inliers stabilise the solve, while a coarser live GSD (0.108 m/px
at 120 m, 0.325 m/px at 360 m) makes every pixel of matching error worth three
times as many metres. Below 240 m the first wins; above it the second does.

**The lens route has a hard geometric limit.** At 120 degrees the frame edge sits 60 degrees
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
