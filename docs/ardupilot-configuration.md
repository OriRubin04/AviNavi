# Configuring ArduPilot to accept the visual navigation solution

Every parameter name and value below was checked against the ArduPilot source
rather than recalled. The authoritative definitions are:

| What | Where in the ArduPilot source |
| --- | --- |
| `EK3_SRCn_*` | `libraries/AP_NavEKF/AP_NavEKF_Source.cpp` |
| `VISO_*` | `libraries/AP_VisualOdom/AP_VisualOdom.cpp` |
| `AHRS_EKF_TYPE` | `libraries/AP_AHRS/AP_AHRS.cpp` |
| `EK3_ENABLE` | `libraries/AP_NavEKF3/AP_NavEKF3.cpp` |
| `GPS1_TYPE` / `GPS_TYPE` | `libraries/AP_GPS/AP_GPS.cpp` |
| `ODOMETRY` handling | `libraries/GCS_MAVLink/GCS_Common.cpp`, `GCS_MAVLINK::handle_odometry` |

**Check these against the firmware version you are actually flying.** Parameter
names do change between releases — `GPS_TYPE` was renamed to `GPS1_TYPE` in 4.6,
for instance. In Mission Planner, *Config → Full Parameter List* and the search
box will tell you what your board has.

## 1. Enable EKF3 and the external navigation source

| Parameter | Value | Meaning |
| --- | --- | --- |
| `EK3_ENABLE` | `1` | Enable EKF3. **Reboot required.** |
| `AHRS_EKF_TYPE` | `3` | Use EKF3 for attitude and position. |
| `VISO_TYPE` | `1` | Visual odometry arrives over MAVLink. **Reboot required.** |

`EK3_ENABLE` only makes the filter run; `AHRS_EKF_TYPE` is what makes it fly the
aircraft.

## 2. Select the EKF source set

`EK3_SRC1_*` is the primary source set. `6` is `ExternalNav` in every one of
these parameters.

| Parameter | Value | Meaning |
| --- | --- | --- |
| `EK3_SRC1_POSXY` | `6` | Horizontal position from external nav |
| `EK3_SRC1_VELXY` | `6` | Horizontal velocity from external nav |
| `EK3_SRC1_POSZ` | `1` (Baro) or `6` (ExternalNav) | See below |
| `EK3_SRC1_VELZ` | `0` (None) or `6` | See below |
| `EK3_SRC1_YAW` | `6` | Yaw from external nav |

The full value sets, from `AP_NavEKF_Source.cpp`:

```
POSXY:  0:None, 3:GPS, 4:Beacon, 6:ExternalNav
VELXY:  0:None, 3:GPS, 4:Beacon, 5:OpticalFlow, 6:ExternalNav, 7:WheelEncoder
POSZ:   0:None, 1:Baro, 2:RangeFinder, 3:GPS, 4:Beacon, 6:ExternalNav
VELZ:   0:None, 3:GPS, 4:Beacon, 6:ExternalNav
YAW:    0:None, 1:Compass, 2:GPS, 3:GPS with Compass Fallback, 6:ExternalNav, 8:GSF
```

### On the vertical channel

**Start with `EK3_SRC1_POSZ = 1` (Baro).** AviNavi derives height from the
image scale, which is the weakest of its three channels: it depends on accurate
`camera.fx`/`fy` and on the terrain actually being flat at the tile's
`terrain_alt_m`. The barometer is better at this job. Only move to
`EK3_SRC1_POSZ = 6` if you have validated the visual height against a
rangefinder over the actual flight area.

The covariance AviNavi sends already reflects this: `pose_covariance_from_sigmas`
gives the vertical channel four times the variance of the horizontal one.

### On yaw

`EK3_SRC1_YAW = 6` tells the EKF to take yaw from the visual solution rather
than the compass — appropriate when flying somewhere magnetically hostile,
which is often why you are GPS-denied in the first place. If your compass is
healthy, `1` (Compass) is a perfectly good choice and removes one dependency on
the visual solution being right.

### Keeping a GPS fallback

If you want the aircraft to fall back to GPS when it is available, put GPS in a
second source set and switch between them with an RC channel:

```
EK3_SRC2_POSXY = 3   (GPS)
EK3_SRC2_VELXY = 3
EK3_SRC2_POSZ  = 1   (Baro)
EK3_SRC2_YAW   = 1   (Compass)
```

then set an `RCn_OPTION` to *EKF Source Set*. For genuine GPS-denied testing,
leave the fallback off so a silent reversion to GPS cannot mask a failure of
the visual solution.

## 3. Camera mounting and noise parameters

| Parameter | Meaning | Suggested start |
| --- | --- | --- |
| `VISO_POS_X` | Camera X in body frame, m, forward positive | `0` — see the warning below |
| `VISO_POS_Y` | Camera Y in body frame, m, right positive | `0` |
| `VISO_POS_Z` | Camera Z in body frame, m, down positive | `0` |
| `VISO_ORIENT` | Camera orientation | `0` (Forward) |
| `VISO_SCALE` | Scale applied to incoming positions | `1.0` |
| `VISO_DELAY_MS` | Sensor delay vs. the IMU, ms | measure it — see below |
| `VISO_POS_M_NSE` | Minimum position noise, m | `0.5` |
| `VISO_VEL_M_NSE` | Minimum velocity noise, m/s | `0.5` |
| `VISO_YAW_M_NSE` | Minimum yaw noise, rad | `0.2` |
| `VISO_QUAL_MIN` | Reject samples below this quality, % | `0` initially, then `30`–`50` |

### Do not configure the lever arm twice

ArduPilot subtracts the rotated `VISO_POS_X/Y/Z` offset from every incoming
external-nav position (`NavEKF3_core::CorrectExtNavForSensorOffset`), because it
assumes you are reporting the *camera's* position.

AviNavi already reports the **body origin**: the pose solver subtracts
`camera_extrinsics.translation` rotated by the vehicle attitude.

So configure the lever arm in **exactly one** place:

- **Recommended:** set `camera_extrinsics.translation` in `config/default.yaml`
  and leave `VISO_POS_X/Y/Z` at `0`.
- Or leave the AviNavi translation at zero and set `VISO_POS_*` instead.

Setting both applies the correction twice.

`VISO_ORIENT` should stay at `0`. It rotates the incoming *pose*, and AviNavi
has already resolved the camera mount rotation into the body frame via
`camera_extrinsics.rotation`. Same rule: configure the mount rotation in one
place, not two.

### Measuring `VISO_DELAY_MS`

This is the age of the measurement when it arrives, relative to the IMU. It is
real and it matters: at 15 m/s, 100 ms of unmodelled delay is 1.5 m of position
error that the EKF will fight.

Add up the pipeline latency AviNavi reports (`Latency: mean … ms` in the
diagnostics view, which covers SuperPoint through the pose solve), the camera's
own exposure and readout time, and the serial transit time. Start with that
sum, then refine by watching the EKF innovations during gentle flight — a delay
that is too small shows up as position innovations that lead the manoeuvre.

### `VISO_QUAL_MIN`

AviNavi maps its confidence score onto `ODOMETRY.quality` as a percentage
(`-1` = failed, `0` = unknown, `1`–`100`). Leave `VISO_QUAL_MIN = 0` during
bring-up so nothing is filtered out and you can see the whole distribution in
the logs. Once you know what confidence a good match produces on your imagery,
raise it to a threshold that rejects the tail — `30` to `50` is typical.

## 4. The external navigation origin

This is the part that most often goes wrong, and it is why AviNavi does not
invent its own origin.

ArduPilot interprets incoming external-nav positions **relative to the EKF's
public origin**:

```c
// external nav data is against the public_origin, so convert to offset from EKF_origin
ext_nav_data.pos.xy() += EKF_origin.get_distance_NE_ftype(public_origin);
```

That public origin is exactly what the autopilot reports in `GPS_GLOBAL_ORIGIN`.
AviNavi listens for that message and anchors its local NED frame at the same
point, so the two frames cannot drift apart.

To set it, in Mission Planner: **right-click the map → Set Home Here → Set EKF
Origin Here**. See [initial-position.md](initial-position.md) for the full
mechanism and the fallbacks.

You should see `EKF3 IMU0 origin set` in the HUD messages.

## 5. Disabling GPS for genuine GPS-denied testing

| Parameter | Value | Meaning |
| --- | --- | --- |
| `GPS1_TYPE` (4.6+) or `GPS_TYPE` (earlier) | `0` | No GPS driver. **Reboot required.** |

With GPS off you will also need to relax the arming checks that require a
position fix. `ARMING_CHECK` is a bitmask; clear the GPS bit rather than
disabling all checks. In Mission Planner the bitmask editor names each bit.

Setting `GPS1_TYPE = 0` is the honest way to test: it makes it impossible for
GPS to be quietly propping up a visual solution that is not actually working.

If you want GPS available for ground-truth comparison *without* the EKF using
it, leave the GPS driver enabled but keep `EK3_SRC1_*` on ExternalNav, and set
`mavlink.request_gps_for_ground_truth: true` in the AviNavi config. The
localizer never reads GPS — the messages go to the CSV log and nowhere else.

## 6. Failsafe and timeout behaviour

ArduPilot stops aiding when external nav data stops arriving; you will see
`EKF3 IMU0 stopped aiding` in the HUD. Configure what the vehicle does then:

- `FS_EKF_ACTION` — what happens on an EKF failsafe (land, AltHold, etc.).
- `FS_EKF_THRESH` — the EKF variance threshold that triggers it.

Do not raise `FS_EKF_THRESH` to stop the failsafe firing during bring-up. If it
is firing, the visual solution is not good enough yet.

AviNavi's side of this is deliberate: when it has nothing valid it **sends
nothing**, rather than repeating a stale pose to keep the rate up. Silence is a
signal the EKF knows how to act on; a confidently repeated wrong position is not.

## 7. Verifying that ArduPilot is receiving the messages

Two checks, in order.

### Is the message arriving at all?

In Mission Planner press **Ctrl+F → MAVLink Inspector** and look for `ODOMETRY`
from the companion's system/component ID at the configured rate.

If it is not there:

- Is `output.send_to_autopilot: true`? It defaults to `false`.
- Does `mavlink.system_id` match the autopilot's `SYSID_THISMAV`?
- Is the serial port set up for MAVLink at a matching baud rate?
  (`SERIALn_PROTOCOL = 2` for MAVLink2, `SERIALn_BAUD = 921` for 921600.)

If `ODOMETRY` appears in the inspector but nothing happens, the most likely
cause is the frame check. `GCS_MAVLINK::handle_odometry` **silently discards**
any message whose `frame_id` is not `MAV_FRAME_LOCAL_FRD` (20) or whose
`child_frame_id` is not `MAV_FRAME_BODY_FRD` (12). AviNavi always sets both
correctly and has a test that asserts it on encoded bytes; if you have modified
the sender, this is the first thing to check.

### Is the EKF actually using it?

Look for this in the HUD messages:

```
EKF3 IMU0 is using external nav data
EKF3 IMU0 initial pos NED = 0.0,0.0,0.0 (m)
```

That message comes from `NavEKF3_core::setAidingMode` and is the definitive
confirmation. Until you see it, the EKF is not consuming the solution no matter
what the inspector shows.

Then, in the onboard log:

- `XKF*` messages: watch the position innovations. They should stay small and
  centred, not ramp.
- `VISP` / `VISO`: the external navigation samples as ArduPilot received them.
  Compare against AviNavi's own CSV to confirm nothing is being mangled in
  transit.

## 8. Rates

ArduPilot enforces a minimum interval of 20 ms between external navigation
samples (`extNavIntervalMin_ms` in `AP_NavEKF3.h`), i.e. a **50 Hz ceiling**.
The configured `rates.mavlink_odometry_hz` of 10–20 Hz is comfortably inside
that, and there is nothing to gain from going higher when the underlying visual
localization runs at 2–5 Hz.

## 9. Parameter summary

A starting point for a multirotor with a healthy barometer and compass:

```
EK3_ENABLE      1
AHRS_EKF_TYPE   3
VISO_TYPE       1

EK3_SRC1_POSXY  6     # ExternalNav
EK3_SRC1_VELXY  6     # ExternalNav
EK3_SRC1_POSZ   1     # Baro - move to 6 only after validating visual height
EK3_SRC1_VELZ   0     # None
EK3_SRC1_YAW    1     # Compass - move to 6 in magnetically hostile areas

VISO_POS_X      0     # lever arm lives in AviNavi's camera_extrinsics
VISO_POS_Y      0
VISO_POS_Z      0
VISO_ORIENT     0     # mount rotation lives in AviNavi's camera_extrinsics
VISO_SCALE      1.0
VISO_DELAY_MS   50    # measure this for your setup
VISO_POS_M_NSE  0.5
VISO_VEL_M_NSE  0.5
VISO_YAW_M_NSE  0.2
VISO_QUAL_MIN   0     # raise to 30-50 once the confidence distribution is known
```

`EK3_ENABLE` and `VISO_TYPE` need a reboot. Verify each value in the Full
Parameter List after writing — and re-verify after a firmware update.
