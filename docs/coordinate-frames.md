# Coordinate frames

Every frame the system touches, what it means, and how the code converts
between them. Getting one of these backwards produces a plausible-looking
position that is wrong, so each conversion has a named function and a test.

## The frames

### Map pixel

Column and row inside a tile raster. Tiles are stored **north-up**, so:

```
        col 0                     col W
row 0   +-----------------------------+   north
        |                             |
        |          tile image         |
        |                             |
row H   +-----------------------------+   south
        west                        east
```

- `+col` is **east**
- `+row` is **south**

Pixel coordinates are continuous with `(0, 0)` at the *corner* of the top-left
pixel, matching GDAL. `avinavi.geo.Georeference` converts in both directions.

### Geographic (WGS-84)

Latitude, longitude in degrees; altitude in metres. What the tiles are
georeferenced against and what the operator picks on the Mission Planner map.

### Local NED

```
        X = North
        Y = East
        Z = Down     (positive downwards, so flying up gives a negative Z)
```

Anchored at the **EKF origin**, in metres. This is the frame `ODOMETRY.x/y/z`
are expressed in. `avinavi.geo.LocalOrigin` does the conversion using the
WGS-84 radii of curvature evaluated at the origin latitude — a tangent-plane
approximation good to well under a metre over the few kilometres a tile
database realistically covers.

**The origin must be the autopilot's origin, not one we invent.** See
[initial-position.md](initial-position.md).

### Body FRD

```
        X = Forward   (nose)
        Y = Right     (starboard)
        Z = Down
```

ArduPilot's body frame and MAVLink's `MAV_FRAME_BODY_FRD`. `ODOMETRY`'s
velocity fields live here.

### Camera optical

```
        X = right across the image
        Y = down the image
        Z = forward along the optical axis
```

The OpenCV convention, and the frame SuperPoint keypoints live in.

## The transform chain

```
   camera optical
         |
         |  R_body_from_camera        fixed, from camera_extrinsics
         v
     body FRD
         |
         |  R_ned_from_body           the vehicle attitude (roll, pitch, yaw)
         v
     local NED
         |
         |  LocalOrigin.to_geo        tangent plane about the EKF origin
         v
    geographic
```

These are **not** interchangeable. `avinavi.frames.CameraExtrinsics` holds the
first, and the second is built from Euler angles with
`euler_to_matrix(roll, pitch, yaw)` = `Rz(yaw) @ Ry(pitch) @ Rx(roll)`, which
takes a body vector into NED.

## Camera-to-body extrinsics

Configured as:

```yaml
camera_extrinsics:
  rotation:
    roll_deg: 0.0
    pitch_deg: 0.0
    yaw_deg: 0.0
  translation:
    x_m: 0.0    # forward of the body origin
    y_m: 0.0    # right of the body origin
    z_m: 0.0    # below the body origin
```

The rotation is applied as `Rz(yaw) @ Ry(pitch) @ Rx(roll)` and maps a
**camera-frame** vector into the **body frame**:

```
v_body = R_body_from_camera @ v_camera
```

**Identity means a nadir mount.** With no rotation the camera's +Z (optical
axis) coincides with body +Z (down), which forces:

- image +X (right across the image) → body +Y (aircraft right)
- image +Y (down the image) → body −X (aircraft aft)

So with an identity extrinsic the top of the image points towards the nose.
If your camera is rotated 90° clockwise in its mount, set `yaw_deg: 90` and
nothing else in the pipeline changes.

The translation is the camera's position in the body frame. The pose solver
computes the *camera* position first and then subtracts the rotated lever arm
to report the **body origin**, which is the point ArduPilot's EKF tracks:

```
p_body_ned = p_camera_ned - R_ned_from_body @ t_camera_body
```

## Map pixel → geographic → local NED

This is the chain the localizer runs on every successful match.

1. **Homography.** RANSAC/MAGSAC fits `H_frame_from_tile`, mapping tile pixels
   to live-image pixels. Residuals are therefore measured in live-camera
   pixels, where the configured threshold is meaningful.

2. **Map pixel.** Invert and evaluate at the camera's principal point
   `(cx, cy)`:

   ```
   (col, row) = H_tile_from_frame · (cx, cy)
   ```

   For a nadir camera this is the ground point directly beneath it.

3. **Geographic.** `tile.pixel_to_geo(col, row)` — a linear interpolation of
   the tile's WGS-84 bounds, or the tile's affine transform (reprojected with
   pyproj when the tile CRS is not EPSG:4326).

4. **Local NED.** `origin.to_ned(geo)` where the altitude is the mapped terrain
   altitude. This gives the *ground point*, not the aircraft.

5. **Aircraft position.** Back-project along the optical axis by the range:

   ```
   p_camera_ned = p_ground_ned − range · (R_ned_from_camera · [0, 0, 1])
   p_body_ned   = p_camera_ned − R_ned_from_body · t_camera_body
   ```

## Scale, range and height

The homography's local Jacobian at the principal point gives metres of ground
per live-image pixel (`frame_gsd`). For a pinhole camera:

```
range = sqrt(fx · fy) · frame_gsd
```

This is what makes the vertical channel observable from imagery alone, and it
is why `camera.fx`/`camera.fy` matter. Two documented assumptions apply:

- the terrain under the frame is flat at the tile's `terrain_alt_m`;
- the depth-from-scale relation is exact for a plane perpendicular to the
  optical axis and first-order accurate for the small tilts a stabilised
  multirotor holds.

If a rangefinder or barometric height is available it is fed in as
`height_hint_m` and converted to a slant range by dividing by the downward
component of the optical axis. Without either intrinsics or a height hint the
solver raises `PoseError` rather than guessing.

## Yaw

The same Jacobian gives the ground bearing of the camera image +X axis. Write
the attitude as `R_ned_from_body = Rz(yaw) @ R_level` with
`R_level = Ry(pitch) @ Rx(roll)`, and let

```
w = R_level · R_body_from_camera · [1, 0, 0]
```

Because `Rz` only rotates the horizontal components:

```
measured_bearing = atan2(w_y, w_x) + yaw
      =>    yaw  = measured_bearing − atan2(w_y, w_x)
```

`avinavi.frames.yaw_from_camera_axis_bearing` implements exactly this. Roll and
pitch come from the autopilot's `ATTITUDE` message when it is fresh (< 0.5 s)
and default to level flight otherwise.

A camera whose +X axis points straight down in the body frame makes yaw
unobservable; the function raises `ValueError` rather than returning nonsense.

## MAVLink frame IDs

This is the detail most likely to silently break a bring-up.

`ODOMETRY` carries two frame fields, and ArduPilot's
`GCS_MAVLINK::handle_odometry` **checks both and returns without any error
message if either is wrong**:

```c
if (m.frame_id != MAV_FRAME_LOCAL_FRD ||
    m.child_frame_id != MAV_FRAME_BODY_FRD) {
    // only support local FRD frame data
    return;
}
```

So:

| Field            | Required value            | Numeric |
| ---------------- | ------------------------- | ------- |
| `frame_id`       | `MAV_FRAME_LOCAL_FRD`     | 20      |
| `child_frame_id` | `MAV_FRAME_BODY_FRD`      | 12      |

`MAV_FRAME_LOCAL_NED` (1) reads more naturally for a north/east/down position
and is **wrong here** — the message will vanish with no diagnostic at either
end. Despite the name, ArduPilot treats the position in that "local FRD" frame
as ordinary north/east/down relative to the EKF origin, so the numbers we put in
are NED numbers.

### Velocity is in the child frame

ArduPilot rotates the velocity by the message quaternion to recover NED:

```c
Vector3f vel{m.vx, m.vy, m.vz};
vel = q * vel;
```

So `vx/vy/vz` must be **body FRD**, not NED. Putting an NED velocity there is
wrong by the vehicle's attitude — invisible in a hover, badly wrong in forward
flight. `avinavi.frames.ned_velocity_to_body_frd` applies the inverse rotation.

### The quaternion

`ODOMETRY.q` is `(w, x, y, z)` — w first — and rotates body FRD into local NED.

### Covariance packing

`pose_covariance` is the row-major **upper-right triangle** of a 6×6 matrix over
the states `(x, y, z, roll, pitch, yaw)`. Row 0 contributes 6 entries, row 1
five, and so on, which puts the diagonal at indices:

```
0, 6, 11, 15, 18, 20
```

ArduPilot reads exactly those:

```c
posErr = sqrtf(m.pose_covariance[0] + m.pose_covariance[6] + m.pose_covariance[11]);
angErr = sqrtf(m.pose_covariance[15] + m.pose_covariance[18] + m.pose_covariance[20]);
```

Everything else in the array is ignored, but those six entries directly set how
much the EKF trusts each fix. `avinavi.types.covariance_upper_triangle` builds
the array; set element 0 to `NaN` to declare the covariance unknown.
