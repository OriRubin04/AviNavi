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

### 5. Losing the fix looks like losing GPS

When this system stops publishing — which it is designed to do, deliberately and
often — the autopilot sees a GPS dropout and may run its GPS failsafe. That
failsafe typically wants to fly home, which needs the position that was just
lost. With external navigation the EKF handles source loss through
`EK3_SRC*` fallback instead, which is the mechanism intended for it.

## What has been verified, and what has not

**Verified** in ArduPilot source (`libraries/AP_NavEKF/AP_NavEKF_Source.cpp`):

- `EK3_SRC1_POSXY` options are `0:None, 3:GPS, 4:Beacon, 6:ExternalNav` — so
  external navigation **can** be an absolute position source, which optical
  flow cannot (flow appears only under `VELXY`).

**Not verified, and each one blocks the switch:**

1. **Does ArduPlane fuse external navigation for navigation, not just logging?**
   The official docs state non-GPS navigation "is not applicable to fast or high
   flying vehicles such as conventional Planes." That sentence is about
   flow-based, close-to-the-ground operation, and may not bind an absolute
   position source — but it has not been checked against the code path.
2. **The `VISO_*` parameters.** Names, units and sign conventions for the sensor
   offset and the measurement delay must be read from source, not assumed. A
   wrong delay is a position error proportional to speed: at 22 m/s, 100 ms is
   2.2 m.
3. **Is ~0.8 Hz acceptable?** Our fix rate is roughly one per 1.26 s. External
   navigation is normally fed by a VIO system at tens of hertz. Whether EKF3
   accepts a source this slow as its primary absolute position, and what it does
   between updates, needs checking.
4. **Covariance units and ordering** in `VISION_POSITION_ESTIMATE` — the field
   is a packed upper triangle, and the convention must be read from the message
   definition rather than inferred.
5. **Which is better of the two external messages**, `ODOMETRY` or
   `VISION_POSITION_ESTIMATE`, given that we have position and a usable yaw but
   velocity only indirectly.

## Recommendation

On a board with `EK3_FEATURE_EXTERNAL_NAV` compiled in, external navigation is
the correct interface and `gps_input` should be treated as the compatibility
fallback it is. The code already supports all three; this is a configuration
change plus the verification above, not new development.

Until points 1–5 are checked against ArduPilot source, the default stays as it
is. Shipping a default that is semantically right and functionally unverified
would be worse than shipping one that is semantically wrong and known to work.
