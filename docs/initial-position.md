# The operator-selected initial position

The operator picks one point on the Mission Planner map before the mission.
That point does two different jobs, and the second one determines the whole
mechanism.

1. **It seeds the map-tile search window** so the first localization attempt
   looks in the right few hundred metres of map instead of the whole database.
2. **It anchors the local NED frame** that every `ODOMETRY` position is
   expressed in.

## Why the origin cannot be invented locally

ArduPilot's EKF has its own notion of where local `(0, 0, 0)` is — the *EKF
origin* — and it interprets incoming external navigation positions against it.
From `NavEKF3_core::CorrectExtNavForSensorOffset`:

```c
// external nav data is against the public_origin, so convert to offset from EKF_origin
ext_nav_data.pos.xy() += EKF_origin.get_distance_NE_ftype(public_origin);
```

If the Raspberry Pi picked its own origin, every position it sent would be
offset by the difference between the two frames. The EKF would either reject
the data as inconsistent or fuse it and fly to the wrong place. Neither failure
announces itself clearly.

So AviNavi does not choose an origin. It reads the autopilot's.

## The chosen mechanism: listen for `GPS_GLOBAL_ORIGIN`

**In Mission Planner: right-click the map → Set Home Here → Set EKF Origin
Here.**

```
     OPERATOR
        |
        |  right-click the map
        v
  MISSION PLANNER
        |
        |  MAV_CMD_DO_SET_GLOBAL_ORIGIN (611), or the older
        |  SET_GPS_GLOBAL_ORIGIN (48) it supersedes
        v
     ARDUPILOT
        |
        |  GPS_GLOBAL_ORIGIN (49)  -- broadcast
        v
   RASPBERRY PI
        |
        +--> local NED origin  (LocalOrigin)
        |
        +--> tile search seed  (LocalizationEngine.initialize)
```

The MAVLink specification for `SET_GPS_GLOBAL_ORIGIN` states that the vehicle
"should emit `GPS_GLOBAL_ORIGIN` irrespective of whether the origin is changed",
and ArduPilot does. The Pi simply subscribes to that broadcast.

### Why this and not the alternatives

- **It is a standard MAVLink message.** No custom dialect, no new protocol, no
  Mission Planner plugin or Lua script.
- **It is the same number the EKF is using**, by construction. A separately
  transmitted "start position" — a parameter, a mission item, a custom command —
  could not guarantee that, and would introduce a second source of truth for
  something that must have exactly one.
- **The operator interaction already exists** in stock Mission Planner.
- **It works identically GPS-denied**, because the origin is set by hand rather
  than derived from a fix.

### Message fields

`GPS_GLOBAL_ORIGIN` (id 49):

| Field | Units |
| --- | --- |
| `latitude` | degE7 (degrees × 10⁷) |
| `longitude` | degE7 |
| `altitude` | mm, MSL, positive up |

`avinavi.mavlink.initial_position.InitialPositionProvider._on_origin` decodes
these and rejects `(0, 0)`, which is what an unset origin looks like.

### Requesting it rather than waiting

`GPS_GLOBAL_ORIGIN` is emitted on change, not streamed. If the autopilot booted
and had its origin set before the Pi started, no broadcast is coming. The
provider therefore also asks for one at startup with
`MAV_CMD_REQUEST_MESSAGE` (512, param1 = 49), and re-asks every 5 seconds
while it waits. Configured by:

```yaml
initial_position:
  source: mavlink_origin
  request_origin_on_start: true
  wait_timeout_s: 0.0       # 0 = wait indefinitely
```

### If the origin changes mid-flight

The provider logs a warning and adopts the new origin. This is worth watching
for: it means the local NED frame has moved, and positions before and after are
not in the same frame. It should not happen during a mission.

## Fallbacks

### `manual` — no autopilot attached

For offline development and bench work with no flight controller:

```yaml
initial_position:
  source: manual
  manual_lat_deg: 32.0853
  manual_lon_deg: 34.7818
  manual_alt_m: 40.0
```

Validation rejects `source: manual` without coordinates rather than silently
starting at null island.

### `auto` — MAVLink first, then manual

```yaml
initial_position:
  source: auto
  wait_timeout_s: 30.0
  manual_lat_deg: 32.0853
  manual_lon_deg: 34.7818
```

Waits for `GPS_GLOBAL_ORIGIN`, falls back to the configured values on timeout
and logs a warning. Reasonable for a repeatable test rig; not what you want in
flight, where a missing origin is information you should act on.

### `push_origin_to_autopilot` — the reverse direction

If the Pi is configured with the start position and the autopilot has no origin
yet, this sends `MAV_CMD_DO_SET_GLOBAL_ORIGIN` (611, params 5/6/7 =
lat/lon/alt) to set it.

```yaml
initial_position:
  source: manual
  manual_lat_deg: 32.0853
  manual_lon_deg: 34.7818
  push_origin_to_autopilot: true
```

Defaults to `false`, and for good reason: overwriting a live EKF origin
mid-flight shifts every local coordinate the autopilot is using. Only enable it
when you are certain the autopilot has no origin set.

Note that `MAV_CMD_DO_SET_GLOBAL_ORIGIN` (611) is newer than the pymavlink
dialect bundled with some releases, so the command ID is defined as a named
constant in `avinavi/mavlink/initial_position.py` rather than imported.

## What happens after initialization

The initial position is used **once**, to seed the search window:

```yaml
tile_search:
  initial_search_radius_m: 500.0    # first fix, from the operator's estimate
  search_radius_m: 300.0            # subsequent fixes, from visual position
```

The first radius is larger because a hand-picked map point is less accurate
than a visual fix. After the first successful localization the window recentres
on the **visual** position and stays there:

```
initial position --> candidate tiles --> first visual fix
                                              |
                          +-------------------+
                          |
                          v
                    new search centre --> candidate tiles --> next visual fix
                          ^                                        |
                          +----------------------------------------+
```

`LocalizationEngine._register_success` is the only thing that moves the search
centre, and it is only ever called with a visual fix. Nothing in the
localization loop reads GPS.

If the initial position falls outside the map database's coverage, the runner
refuses to start rather than searching a database that cannot contain the
answer — unless the gap is within `tile_search.max_search_radius_m`, in which
case it warns and continues.
