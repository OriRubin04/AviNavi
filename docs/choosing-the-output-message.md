# Choosing the output message: GPS_INPUT or external navigation

Three output paths are implemented and selected by one line of configuration:

```yaml
output:
  message: gps_input      # gps_input | odometry | vision_position_estimate | none
```

**Nothing in this document has been changed in the code.** It sets out the case
for moving the default off `gps_input`, what must be verified first, and what
is still unverified. The default remains `gps_input` until those checks pass.

## Why `gps_input` is the current default, and why that reason is narrow

It was chosen for one reason: on a 1 MB F4 board such as the MatekF405-Wing, a
stock ArduPlane build has `HAL_VISUALODOM_ENABLED`, `EK3_FEATURE_EXTERNAL_NAV`
**and** `AP_GPS_MAV_ENABLED` all compiled out, and `GPS_INPUT` is the only one a
custom build can restore with a single feature flag. See
[flight-controller-compatibility.md](flight-controller-compatibility.md).

That is a workaround for a flash-size limit. It is **not** an argument that
GPS_INPUT is the better interface, and on a board that supports external
navigation it is the weaker choice on every count below.

## The safety case against GPS_INPUT

### 1. It misrepresents what the system is

ArduPilot routes GPS_INPUT through `AP_GPS_MAV`, so the autopilot, the ground
station and the operator all see an ordinary GNSS receiver. A pilot watching
"3D Fix, 12 satellites" has no indication that the position came from a camera
matching an orthophoto. Post-flight, the dataflash log records it as GPS.

That matters most exactly when it matters: diagnosing an anomaly. Every
instinct an experienced operator has about GPS behaviour — glitches, multipath,
satellite geometry — is the wrong model for this system's failure modes, which
are matching failures, scale error and heading error.

### 2. The satellite count is fabricated

`EK3_GPS_CHECK` requires six satellites by default, so the configuration ships
`gps_input_satellites: 12` to get past it. There is no camera equivalent of a
satellite. The field exists only to satisfy a check that was written for a
different sensor, and filling it in is inventing telemetry.

### 3. The EKF applies a GNSS noise and fault model

Position from GPS is processed with GPS-specific innovation gates, glitch
detection, lag assumptions and the `EK3_POSNE_M_NSE` / `EK3_VELNE_M_NSE` noise
terms, all tuned for the error characteristics of a GNSS receiver. Our errors
do not look like GNSS errors: they are near-zero for a good match, and a
matching failure produces no measurement rather than a degraded one.

### 4. The uncertainty we compute cannot be expressed

This is the sharpest loss. Considerable work went into reporting an honest,
growing position sigma — distance-driven drift, plus time nobody observed, in
quadrature. `VISION_POSITION_ESTIMATE` carries a **6×6 covariance matrix**;
`ODOMETRY` carries pose and twist covariance. `GPS_INPUT` has `hdop` and `vdop`,
which are dimensionless geometry factors, not metres.

So the sigma is squeezed into a field that cannot represent it, and the EKF
reconstructs a variance from it using a GNSS-derived scaling. The one number
that protects the autopilot from a bad fix is the number the interface handles
worst.

### 5. Losing the fix is silent, and nothing catches it

An earlier version of this document claimed that when the system stops
publishing, the autopilot runs its GPS failsafe. **That was wrong.**
ArduPlane has no GPS failsafe at all — unlike Copter, which has
`FS_EKF_ACTION`, ArduPlane defines no equivalent. There is no `FS_GPS_*`
parameter and no code path that reacts to the loss of a position source.

The real behaviour is worse than a failsafe firing, and worse in a way that is
easy to miss: when this system stops publishing — which it is designed to do,
deliberately and often — EKF3 simply coasts on inertial dead reckoning and
**says nothing**. No mode change, no warning, no message to the operator. The
aircraft keeps flying its mission on a position estimate that drifts without
bound. In a `LOITER` or `RTL` built on that estimate, the drift becomes a
steadily growing error in where the aircraft believes it is.

This is an argument for external navigation, but not the one previously given.
With `EK3_SRC*` the EKF knows an external-nav source has gone away and the
source-set mechanism is the one intended for handling it. With `GPS_INPUT` the
loss is indistinguishable from a GNSS dropout that nothing is watching for.

**Either way, ArduPilot will not protect us here.** The behaviour on losing
visual position has to be ours: a failsafe implemented on our side, which
detects that all three navigation tiers have run out and commands something
deliberate rather than letting the aircraft coast quietly. That work is not yet
written. See [navigation-tiers.md](navigation-tiers.md) for the tiers it would
sit on top of.

## What has been verified, and what has not

**Verified** in ArduPilot source (`libraries/AP_NavEKF/AP_NavEKF_Source.cpp`):

- `EK3_SRC1_POSXY` options are `0:None, 3:GPS, 4:Beacon, 6:ExternalNav` — so
  external navigation **can** be an absolute position source, which optical
  flow cannot (flow appears only under `VELXY`).

The five questions this document originally listed as open have since been
checked against ArduPilot source. Four came back in favour of external
navigation. The fifth found a blocker that has nothing to do with the choice of
message.

1. **Does ArduPlane fuse external navigation for navigation, not just logging?
   Yes.** `AP_VisualOdom` is gated only by flash-size macros
   (`HAL_VISUALODOM_ENABLED`, `EK3_FEATURE_EXTERNAL_NAV`) — there is no
   vehicle-type gating anywhere in the fusion path, so a plane build that
   compiles it in fuses it exactly as a copter build does. The documentation
   sentence about "fast or high flying vehicles" is about optical-flow
   operation close to the ground, and does not bind an absolute position source.
2. **The `VISO_*` parameters.** `VISO_DELAY_MS` defaults to **10 ms** with a
   documented maximum of 250. Our pipeline's latency is far above the default,
   so this must be set, not left alone. There is a trap above it:
   `JitterCorrection` carries `max_lag_ms = 500` and **clamps silently** — a
   delay larger than that is discarded without a warning.
3. **Is ~0.8 Hz acceptable? No, and this is the real blocker.**
   `AP_VISUALODOM_TIMEOUT_MS` is **300 ms**. At one fix per 1.26 s the source is
   considered unhealthy almost all of the time, which fails the arming check —
   the aircraft will not arm. Raising the fix rate above ~3 Hz is a prerequisite
   for external navigation, independent of which message is used.
4. **Covariance is accepted but mostly discarded.** `VISION_POSITION_ESTIMATE`
   does carry a 6×6 packed upper triangle, and ArduPilot does read it — but it
   collapses the whole matrix to a single scalar,
   `sqrt(var_x + var_y + var_z)`. So the honest per-axis sigma still cannot be
   expressed. It is a real improvement on `hdop` (metres, not a dimensionless
   geometry factor) but far less than the message format suggests.
5. **`VISION_POSITION_ESTIMATE`, not `ODOMETRY`.** We have position and a
   usable yaw; velocity only indirectly. `ODOMETRY` additionally expects twist,
   which we would be inventing — the same mistake as the fabricated satellite
   count.

**A separate blocker, found while checking the above:** `ArduPlane/takeoff.cpp`
hard-requires a real `AP_GPS` 3D fix for an `AUTO` takeoff, and that check
looks at the GPS driver directly, not at the EKF's position source. External
navigation cannot satisfy it. So an external-nav aircraft cannot use AUTO
takeoff — it must be launched in a manual mode and switched afterwards. Notably
**`GPS_INPUT` does satisfy this check**, because it presents itself as a GPS
driver. This is the one concrete operational advantage GPS_INPUT has, and it is
a consequence of the same misrepresentation §1 objects to.

Switching sources in flight does not need Lua: `RCx_OPTION = 90` assigns a
transmitter switch to the EK3 source set, so an operator can move between a
GPS-backed and an external-nav source set directly.

## Recommendation

On a board with `EK3_FEATURE_EXTERNAL_NAV` compiled in,
`VISION_POSITION_ESTIMATE` is the correct interface, and `gps_input` should be
treated as the compatibility fallback it is.

**But the default stays `gps_input`, and now for a measured reason rather than
an unverified one: our fix rate.** At ~0.8 Hz the 300 ms health timeout keeps
the aircraft from arming. The switch is gated on raising the fix rate above
~3 Hz, not on further source reading. The semantic objections in §1–§4 all
stand; they are simply not yet actionable.
