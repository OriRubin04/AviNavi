# Manual re-anchor — telling the system where to look

You click a point on the Mission Planner map and the Pi resumes its tile search
there. This is what you use when the system has lost lock and you know roughly
where the aircraft is.

**A re-anchor moves the search window and nothing else.** It is not a position
fix. The next frame still has to earn its match against the tiles around that
point, and until it does, nothing is published to the autopilot. That is what
makes a click safe to send at any time, including mid-flight with transmission
enabled: the worst a wrong click can do is make the system search in the wrong
place and stay silent, which is what it was already doing.

Concretely, accepting a request calls `engine.reanchor()`, which sets the search
centre and widens the radius to `reanchor.radius_m` (1500 m by default — a map
click is a rougher estimate than a visual fix). It also clears the pose
propagator and resets the safety gate, because any pose still in flight was
derived from the old area and must not survive the move.

## The problem: most map clicks never reach the Pi

ArduPilot's `MAVLink_routing::forward` does not forward a message addressed to a
specific component to other links. Mission Planner's map actions are addressed
to `target_component = 1` (the autopilot). So if your Pi is on its own
telemetry UART — the normal wiring — **the obvious channels do not arrive.**

This is why there are five channels rather than one. They are ranked below by
what they cost you, and the first is the one to use.

| Channel | Reaches a Pi on its own UART | ACK | Side effects | Needs |
| --- | --- | --- | --- | --- |
| `MAV_CMD_SPATIAL_USER_1` | ✅ addressed to us | ✅ | none | the MP plugin |
| `SET_GPS_GLOBAL_ORIGIN` | ✅ no `target_component` | ❌ | none¹ | `SYSID` match |
| `DO_SET_ROI` | ❌ hub topology only | ❌ | none² | — |
| `HOME_POSITION` | ✅ sent to every link | ❌ | **moves RTL** | — |
| UDP control socket | n/a | — | none | network access |

¹ ArduPilot rejects the origin change itself once the origin is set, so nothing
on the aircraft moves. ² A navigation no-op on a gimbal-less plane.

## 1. The recommended channel: the Mission Planner plugin

`tools/mission_planner_plugin/AviNaviSearchHere.cs` adds **"AviNavi: Search
Here"** to the Flight Data map's right-click menu. It sends a `COMMAND_INT` with
`MAV_CMD_SPATIAL_USER_1` (31005) addressed to our component, which the router
delivers because it is addressed to us.

This is the only channel that can **acknowledge** the click, which matters more
than it sounds: every other option leaves you clicking and hoping. With this
one, Mission Planner shows the command result, so you know the Pi received the
point.

Installation:

1. Copy `AviNaviSearchHere.cs` into Mission Planner's `plugins/` directory.
2. Restart Mission Planner. The plugin compiles on startup.
3. Right-click the Flight Data map — the item appears in the context menu.

By our own convention `param1` carries an optional search radius in metres; zero
means "use the configured default".

```yaml
reanchor:
  accept_spatial_user: true
  radius_m: 1500.0
```

## 2. The zero-code fallback: "Set EKF Origin Here"

If you cannot install a plugin, right-click the map and choose **Set EKF Origin
Here**. This sends `SET_GPS_GLOBAL_ORIGIN` (message 48), which has **no
`target_component` field** — so ArduPilot's router forwards it to every link,
including your Pi's.

It is safe on the aircraft: once the origin is set, ArduPilot rejects a change
to it, so the click does nothing to the flight controller. We simply overhear
it.

Two caveats. The Pi's `mavlink.system_id` must equal the autopilot's
`SYSID_THISMAV`, or the message is addressed to a different system and we will
not see it. And there is no ACK, so you get no confirmation beyond watching the
diagnostics line change.

Identical consecutive clicks at the same point are debounced; a genuine click
somewhere else is always treated as a new request.

## 3. Hub topology only: "Point Camera Here"

If the Pi sits between the radio and the flight controller — forwarding traffic
rather than hanging off a spare UART — then `DO_SET_ROI` traffic passes through
it and **Point Camera Here** works as a re-anchor. On a plane with no gimbal it
is a navigation no-op, so nothing is lost by using it this way.

We deliberately do not ACK it: we are eavesdropping on a command addressed to
the autopilot, and a second ACK would collide with the autopilot's own reply.

## 4. Off by default: "Set Home Here"

`HOME_POSITION` (242) is broadcast from `AP_AHRS::set_home()` to every link,
bypassing the router entirely, which makes it the most *reliable* channel. It is
also the most intrusive, because using it means you have moved the RTL point and
re-referenced every relative altitude in the mission.

That trade is not ours to make for you, so it is **off**:

```yaml
reanchor:
  accept_home_position: false
```

If you enable it: the first `HOME_POSITION` after startup is treated as a
baseline, not a request — it is simply where home already was. And a move
smaller than `min_move_m` (25 m) is ignored, because ArduPilot re-broadcasts
home on arming and on origin changes, which is not you asking for anything.

## 5. Scripted: the UDP control socket

For a ground station of your own, or a test script:

```bash
echo '{"lat": 32.0853, "lon": 34.7818, "radius_m": 900}' \
  | nc -u -w0 127.0.0.1 14555
```

Disabled by default (`control_socket_host: ""`). **Anything that can reach this
port can redirect the navigation search**, so keep it on localhost unless you
have a specific reason not to — and if you bind it to a routable address,
AviNavi logs a warning saying exactly that.

## What gets rejected, and why

A coordinate channel is a place where malformed input turns into a navigation
command, so each of these is refused rather than acted on:

- **`(0, 0)`** — ArduPilot's own "use current location" sentinel, not a point in
  the Gulf of Guinea.
- **degE7 values too small to be degE7** (below 1000, i.e. under 0.0001°) — the
  signature of a sender that packed float degrees into an integer field.
- **Out-of-range** latitude or longitude.
- **Float degrees in `COMMAND_LONG` that look like degE7.** This is the sharpest
  trap in the pair: the *same command number* carries degE7 integers in
  `COMMAND_INT` and plain float degrees in `COMMAND_LONG`. Reading one as the
  other is a factor of ten million — the point does not land slightly wrong, it
  leaves the planet. Both decodings are implemented separately and both are
  tested.

## Confirming it worked

The console diagnostics line shows the search centre and radius, and a re-anchor
logs at warning level with the source channel and how far the centre moved:

```
manual re-anchor from spatial_user_1: tile search will resume at
32.0853000, 34.7818000 with a 1500 m radius
```

If you see the aircraft's own `STATUSTEXT` in Mission Planner but no change in
the AviNavi diagnostics, the message is not reaching the Pi — go back to the
routing table above, because it is a topology problem and no amount of
configuration will fix it.
