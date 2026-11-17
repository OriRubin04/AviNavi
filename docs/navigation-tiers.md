# The three navigation tiers

Where the position comes from, how one tier hands over to the next, and how
each one is bounded.

The arrangement is standard in vision-based navigation: a map-matched absolute
fix when it is available, optical-flow dead reckoning when it is not, and the
autopilot's own inertial estimate when neither works. What is specific to this
project is where the boundaries are drawn and why, because two of the three
boundaries were originally drawn in the wrong place and both mistakes were the
same mistake: **a limit measured against the output of the component it is
meant to bound.**

```
                    ┌──────────────────────────────────────────┐
  camera frame ───► │ 1. MAP MATCHING                          │
                    │    absolute position, 0.15-0.5 m         │
                    └──────────────┬───────────────────────────┘
                                   │ no match this cycle
                                   ▼
                    ┌──────────────────────────────────────────┐
                    │ 2. OPTICAL FLOW                          │
                    │    relative: last fix + tracked motion   │
                    │    bounded by position sigma, in metres   │
                    └──────────────┬───────────────────────────┘
                                   │ flow produces nothing
                                   ▼
                    ┌──────────────────────────────────────────┐
                    │ 3. INERTIAL (ArduPilot EKF3, its own)    │
                    │    we stop publishing; the EKF coasts    │
                    └──────────────────────────────────────────┘
```

Tier 3 is not code in this project. It is what ArduPilot does on its own once
this system stops sending, and stopping is the whole of our contribution to it.

## Tier 1 — map matching

`avinavi/localization/engine.py`. Rectify, extract, match against precomputed
tile features, verify geometrically, gate on confidence. This is the only tier
that produces an **absolute** position: it is the only one that knows where the
ground is, rather than how far the aircraft has moved across it.

Measured at 240 m cruise: 100% of attempts, 0.18 m median error, one attempt
every 1.26 s — a fix every 28 m of flight at 22 m/s. At 120 m, 59.3% and 46 m
between fixes. See [performance.md](performance.md); the difference is the size
of the rectified frame, not the altitude.

## Tier 2 — optical flow

`FrameToFrameOdometry` and `DeadReckoner` in
`avinavi/localization/visual_odometry.py`. Lucas-Kanade sparse flow between
consecutive live frames, projected to the ground with the autopilot's attitude
and a height estimate, with a forward-backward check and a residual gate. It
runs on its own thread at `rates.visual_odometry_hz`, because tracking needs
consecutive frames and tier 1 is far too slow to provide them.

This tier does **two different jobs**, and separating them is the main design
decision on this page.

### Job A: navigate

The aircraft can be flown on flow. It is relative and it drifts, but drift is
not the problem — **misreported drift** is. ArduPilot's EKF weights every
measurement by the accuracy it is given, so a 20 m-sigma position is useful to
it and a 300 m one is correctly down-weighted to nothing. So the bound is a
tolerance the mission sets **in metres**:

```yaml
visual_odometry:
  max_position_sigma_m: 50.0
```

At the measured 5% drift that is about a kilometre of flow navigation. It is
deliberately not a stopwatch: "roughly where we are" is a distance, and a
duration only stands in for one if the speed is known.

### Job B: aim tier 1

The tile search is centred on the last accepted position. During a dropout the
aircraft keeps flying, so without help the window stays where the last fix put
it — 66 m behind after 3 s, 440 m after 20 — while the radius grows around a
point the aircraft has left. `LocalizationEngine.guide_search_center` moves the
centre to follow the flow estimate each failed cycle, which is what lets a
dropout recover without being told.

The two jobs get separate budgets because they have very different accuracy
requirements:

| | needs | budget | at 440 m flown |
| --- | --- | --- | --- |
| Navigate | metres | `max_position_sigma_m` (50 m) | 22 m drift — too much |
| Aim the search | better than the window is wide | `max_search_guidance_s` (60 s) | 22 m against a radius of hundreds — ample |

The same number, useless for one job and excellent for the other.

`guide_search_center` is deliberately not `reanchor`: it moves the centre but
leaves the failure count and the grown radius alone. An operator click is new
information about the ground and resets both; "I have moved" is not, and
resetting the radius on every guidance update would undo the widening the
dropout earned — which is the only thing that can escape a wrong centre.

### What tier 2 cannot do

**Flow measures motion, not position.** If the last accepted fix was 500 m from
the truth, guidance keeps the window 500 m from the truth and merely moves it
along the right track. There is a test pinning exactly this
(`test_guidance_carries_a_false_fix_faithfully`) so the manual re-anchor is not
later removed as redundant — it is the only escape from a poisoned centre, and
a measured run went 116 cycles to the end of a flight without recovering from
one. See [manual-reanchor.md](manual-reanchor.md).

## Tier 3 — the hand-off, and the bug that made it unreachable

Reaching tier 3 means publishing nothing, so ArduPilot's EKF3 is left with its
IMU. The hand-off has to be reliable, and for a long time it was not reachable
at all.

The elapsed time bounding tier 2 was a **sum of optical flow's own
timestamps**. When flow failed — over water, under cloud shadow, on a
featureless field — it returned nothing, and so:

- the accumulated offset stopped changing, so the position froze;
- the sigma stopped growing, being a function of distance flown;
- the elapsed time stopped advancing, so no limit ever tripped;
- and a fresh timestamp went on it every cycle.

Reproduced before fixing: the same position with the same 3 m sigma, published
as current indefinitely while the aircraft flew away at 22 m/s. After a minute
that is **1320 m of error reported with 3 m of confidence** — the exact fake
position this system exists to refuse, produced by the component meant to
prevent one.

The general lesson, and it is worth carrying to any watchdog: **a limit
measured against a component's output stops applying at the moment that
component fails,** which is the moment it is most needed.

Two halves to the fix:

1. **Time comes from a clock.** `wall_elapsed_s` and `unobserved_s` are
   measured against `time.monotonic`, injectable so staleness can be tested
   without sleeping.
2. **Unobserved seconds enter the uncertainty at the airframe's speed.** A
   second nobody measured is a second the aircraft could have flown anywhere up
   to `unobserved_speed_mps` (40 m/s), so it enters `sigma_m` in quadrature
   alongside the base sigma and the tracked drift.

A frozen position now reports its own worthlessness:

```
flow healthy          t=0.5   north= 11.00   sigma= 3.05
flow produces nothing t=0.6   north= 11.00   sigma= 5.03
                      t=1.0   north= 11.00   sigma=20.23
                      t=1.7   north= 11.00   sigma=48.10
                      t=1.8   dead_reckoning_expired  → tier 3
```

The position does not move, which is correct — the system genuinely does not
know where the aircraft is. What changed is that it now says so.

### Two different messages to the operator

| Message | Meaning |
| --- | --- |
| `position uncertainty passed 50 m` | the system working as designed; it drifted far |
| `optical flow has produced nothing for N s` | a **failure**; nothing visual remains |

They are reported separately because only the second means the aircraft is on
its IMU alone. `flow_lost` is derived from `max_position_sigma_m /
unobserved_speed_mps` rather than configured separately: once the unmeasured
term alone exceeds the tolerance, the position is worthless whatever else is
true, so waiting longer gains nothing.

## Coming back

Nothing latches. The system tries to recover on its own and the operator
intervenes only if it is taking too long.

Automatic recovery is already corroborated by the safety gate
(`avinavi/app/safety.py`): consecutive invalid fixes **disarm** external
navigation, and re-arming requires `required_consecutive_good_fixes` consecutive
valid ones. A large position jump also disarms and re-arms from scratch. So a
single lucky match cannot resume transmission by itself.

An earlier version of this design latched the system off after the budget
expired and required an operator re-anchor to resume. It was abandoned for two
reasons: the safety gate already provides the corroboration, and the latch would
have fired four times in a measured 150-cycle run at 120 m — including on a
53-cycle stretch the system recovered from unaided.

## The settings, in one place

```yaml
visual_odometry:
  enabled: true
  max_position_sigma_m: 50.0      # tier 2 navigation bound, in METRES
  max_dead_reckoning_s: 3.0       # time backstop, secondary to the sigma
  unobserved_speed_mps: 40.0      # what a second of silence costs in sigma
  max_search_guidance_s: 60.0     # how long flow keeps aiming tier 1
  search_guidance_min_move_m: 15.0
  drift_fraction: 0.05            # measured: 5 m of error per 100 m flown
```

## Where the tests are

| File | What it pins |
| --- | --- |
| `tests/test_visual_odometry.py` | flow solving, the residual gate, the reckoner |
| `tests/test_flow_failure.py` | the frozen-position bug, including its old shape |
| `tests/test_search_guidance.py` | the window follows the aircraft, and what guidance cannot fix |
| `tests/test_runner_dead_reckoning.py` | the tier 1 → 2 transition in the runner |
| `tests/test_safety.py` | arming, disarming, and the jump gate |
