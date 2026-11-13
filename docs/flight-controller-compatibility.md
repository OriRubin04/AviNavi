# Flight controller compatibility — read this before buying or wiring anything

**Summary: on stock ArduPlane firmware, a MATEKSYS F405-WING (V1 or V2) cannot
accept a visual position estimate by any means. Not `ODOMETRY`, not
`VISION_POSITION_ESTIMATE`, not `GPS_INPUT`. There is no parameter that fixes
this — the code is not in the binary.**

This page explains why, how to verify it on your own hardware in ten seconds,
and the three ways forward.

## Why

The F405-WING flashes the `MatekF405-Wing` target. It is an STM32F405 with 1 MB
of flash, of which ~960 KB is usable (`FLASH_RESERVE_START_KB 64` — 16 KB
bootloader, 16 KB OSD, 32 KB storage). ArduPilot gates expensive features on
flash size, and this board sits one unit below every relevant threshold.

| Feature | Macro | Gate | Value here |
| --- | --- | --- | --- |
| `ODOMETRY`, `VISION_POSITION_ESTIMATE` | `HAL_VISUALODOM_ENABLED` | `HAL_PROGRAM_SIZE_LIMIT_KB > 1024` | **0** |
| EKF3 external-nav fusion | `EK3_FEATURE_EXTERNAL_NAV` | `… > 1024` | **0** |
| `GPS_INPUT` (`GPS1_TYPE = 14`) | `AP_GPS_MAV_ENABLED` | `AP_GPS_BACKEND_DEFAULT_ENABLED` | **0** |

`HAL_PROGRAM_SIZE_LIMIT_KB` is 1024 for this board, and `1024 > 1024` is false.

The `GPS_INPUT` case is different and easier to miss: it is not a flash-size
gate. The board's hwdef ends with `include ../include/minimize_fpv_osd.inc`,
which includes `minimize_common.inc`, which contains:

```
# smaller boards lose some GPS support
define AP_GPS_BACKEND_DEFAULT_ENABLED 0
define AP_GPS_UBLOX_ENABLED 1
define AP_GPS_DRONECAN_ENABLED HAL_ENABLE_DRONECAN_DRIVERS
```

`minimize_fpv_osd.inc` adds NMEA back. It never adds MAV back. Only one board in
the entire tree (`skyviper-v2450`) sets `AP_GPS_MAV_ENABLED 1`.

### The failure modes are silent, which is the dangerous part

- `ODOMETRY` arrives, is dispatched, and the entire `case` block is compiled
  out. No error, no warning, nothing in the logs.
- `GPS1_TYPE = 14` saves fine. `AP_GPS::_detect_instance` then falls through the
  disabled `GPS_TYPE_MAV` case into the DroneCAN cases and returns `nullptr`.
  You get **no GPS instance and no message** — not an error, just a GPS that
  never appears.
- `EK3_SRC1_POSXY = 6 (ExternalNav)` saves fine too: the parameter's `@Values`
  metadata is **not** conditioned on the feature flag, so Mission Planner
  cheerfully offers "ExternalNav" in the dropdown. You only find out at arming:
  `PreArm: EK3 sources require VisualOdom`.

**Do not treat the presence of a dropdown option as evidence the feature is
compiled in.**

## Verifying on your own hardware

Three checks, cheapest first.

**1. Read `features.txt`.** Every official build publishes one beside the
firmware: `firmware.ardupilot.org/Plane/stable/MatekF405-Wing/features.txt`.
Expect `!HAL_VISUALODOM_ENABLED`, `!EK3_FEATURE_EXTERNAL_NAV`,
`!AP_GPS_MAV_ENABLED` — the `!` prefix means compiled out.

**2. Look for `VISO_TYPE` in the full parameter list.** The whole `VISO`
parameter subgroup lives inside `#if HAL_VISUALODOM_ENABLED`
(`AP_Vehicle.cpp`), so **if `VISO_TYPE` does not exist, visual odometry is not
in your firmware.** Unambiguous, and it takes ten seconds in Mission Planner's
Full Parameter List.

**3. For the MAV GPS the parameter check does not work** — `GPS1_TYPE` always
advertises `14:MAVLink` regardless of the build. Set it, reboot, send a
well-formed `GPS_INPUT`, and watch whether a GPS instance ever appears in
`GPS_RAW_INT` / the `SYS_STATUS` sensor bits.

## The three ways forward

### Option A — custom firmware with the MAV GPS backend (recommended)

Build at `custom.ardupilot.org`: board `MatekF405-Wing`, vehicle ArduPlane,
tick **GPS Drivers → MAV** (`AP_GPS_MAV_ENABLED`). Then set `GPS1_TYPE = 14`
(`GPS_TYPE` on 4.5 and earlier) and run AviNavi with:

```yaml
output:
  message: gps_input
```

Why `GPS_INPUT` rather than the visual-odometry path:

- **One feature flag, no dependencies.** `ODOMETRY` needs two
  (`HAL_VISUALODOM_ENABLED` *and* `EK3_FEATURE_EXTERNAL_NAV`), and enabling only
  the first gets you message parsing that feeds nothing — easy to get half-right
  and lose days to.
- **It feeds the normal, heavily exercised GPS → EKF3 path.** `EK3_SRC1_*` stays
  at its defaults. No source-set juggling, no `VISO_*` setup, no EKF origin
  alignment.
- **It is global-frame** (lat/lon/alt MSL), which is what a fixed-wing AUTO
  mission wants anyway.

Flash is genuinely tight. The custom build will tell you if it overflows; OSD is
the obvious thing to trade on a wing flying with a companion computer.

### Option B — use a 2 MB flight controller

If the mission is "companion computer injects visual position into a fixed
wing", the F405-WING is the wrong board. A 2 MB board gets all three features in
the **stock official firmware**, with no custom build and no flash anxiety:

- `MatekF765-Wing` — same wing form factor, drop-in
- `MatekH743` / `MatekH7A3-Wing`

If you have any schedule risk, this is the cheapest fix available.

### Option C — custom firmware with visual odometry (`ODOMETRY`)

Tick **AHRS → VISUALODOM** *and* **AHRS → EKF3_EXTNAV**, then `VISO_TYPE = 1`
and `EK3_SRC1_POSXY/VELXY = 6`. More flash, two flags, and for a fixed wing
flying waypoints it buys nothing over `GPS_INPUT`. Worth it only if you
specifically need body-frame odometry with covariances.

## What works on the F405-WING with no custom build at all

One thing, and AviNavi is built around it: **`HOME_POSITION` (242) and
`GPS_GLOBAL_ORIGIN` (49) are broadcast to every MAVLink link whenever they
change**, with no request and no route learning. From `AP_AHRS::set_home()`:

```c
// send new home and ekf origin to GCS
GCS_SEND_MESSAGE(MSG_HOME);
GCS_SEND_MESSAGE(MSG_ORIGIN);
```

and `GCS::send_message` loops over every channel. So the operator's map click
reaches the Pi for free — see [manual-reanchor.md](manual-reanchor.md).

GCS *command* forwarding is the opposite: unreliable by design. A command
addressed to `target_component = 1` (the autopilot) is **never** forwarded to
other links — `MAVLink_routing::forward` returns early. Do not build the Pi's
control flow on overhearing Mission Planner.

## Other things `minimize_fpv_osd.inc` removes on this board

Relevant to this project:

- `AP_OPTICALFLOW_ENABLED 0` — no flow sensor fusion, so the Pi's own
  frame-to-frame odometry is the only dead-reckoning aid available
- `AP_BEACON_ENABLED 0`
- `AP_PLANE_OFFBOARD_GUIDED_SLEW_ENABLED 0`
- `AP_ADVANCEDFAILSAFE_ENABLED 0`
- rangefinders cut to the four Benewake drivers only — matters if you planned a
  laser altimeter to feed the vertical channel

## Arming and EKF gates to know about

- **EKF3 wants 6 satellites and a passing HDOP** before it will align and set an
  origin: `EK3_GPS_CHECK` defaults to 31 (bits 0–4), and
  `AP_NavEKF3_VehicleStatus.cpp` reports `"GPS numsats %u (needs 6)"`. A
  synthetic GPS must therefore report `satellites_visible >= 6` and a plausible
  HDOP, which AviNavi does.
- Arming requires a **3D fix** (`AP_Arming::gps_checks`), GPS/AHRS agreement
  within 10 m, and home set.
- **The arming mask was renamed.** On 4.5/4.6 it is `ARMING_CHECK` (set bits to
  *enable* checks; bit 3 = GPS). On current master it is `ARMING_SKIPCHK` (set
  bits to *skip* checks, default 0). Know which release you are on before
  scripting parameter changes.
- ArduPlane's AUTO mode, unlike Copter's, does **not** gate on position
  (`ModeAuto::_enter()` has no GPS check). It will navigate on whatever the EKF
  produces. Note also that DCM fallback is enabled by default on Plane
  (`AHRS_OPTIONS` bit 0) — understand that fallback before trusting it.
