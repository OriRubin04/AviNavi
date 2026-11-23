"""How fast does optical-flow dead reckoning actually drift, and why?

The whole of tier 2 is bounded by one number in the configuration:

    drift_fraction: 0.05        # 5 m of error per 100 m flown

Nobody measured it.  Tier 1 has been measured exhaustively - lock rate against
rectified frame size, false fixes, blackout structure - and the layer that takes
over when tier 1 fails has been running on a figure that was assumed.  If the
true value is 1% the system is needlessly conservative and stops transmitting
while it still knows where it is; if it is 15% it publishes positions it
presents as three times better than they are.

What this measures
------------------
Flow is integrated from a known start with **no map fixes at all**, and the
integrated position is compared against the simulator's ground truth every
cycle.  That gives the error as a function of distance flown, which answers
three separate questions:

1. **The coefficient** - what the drift fraction actually is.
2. **The shape** - linear in distance, or super-linear?  A heading error makes
   position error grow faster than distance, because the track curves away.
   ``sigma = hypot(base, fraction * distance)`` assumes linear, and if the real
   shape is super-linear the model under-reports exactly when it matters most.
3. **Where it comes from** - which is the part worth having.

The decomposition
-----------------
Flow converts pixel displacement to ground distance with
``distance = pixels * height / focal_length``, and rotates it to north with the
autopilot's yaw.  So there are three independent error sources, and they are
separated by feeding the odometry perfect or corrupted values in turn:

* **tracking noise** - perfect height and yaw: whatever the tracker itself gets
  wrong.  The floor.
* **height/scale error** - height corrupted as a barometer would be.  This is
  *systematic*: a 5% height error scales every single displacement by 5%, so it
  accumulates linearly and never averages out.
* **heading error** - yaw corrupted.  Also systematic, and it curves the
  integrated track rather than lengthening it, which is the mechanism that
  would make the error super-linear.

Knowing which dominates decides what to fix.  If it is scale, the drift
fraction should be derived from the height estimate's uncertainty rather than
configured as a constant.  If it is heading, the model shape is wrong.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from avinavi.config import CameraConfig, CameraExtrinsicsConfig, VisualOdometryConfig  # noqa: E402
from avinavi.frames import CameraExtrinsics  # noqa: E402
from avinavi.geo import LocalOrigin  # noqa: E402
from avinavi.localization.visual_odometry import FrameToFrameOdometry  # noqa: E402
from avinavi.sim.flight import fly_circuit  # noqa: E402
from avinavi.sim.render import CameraSimulator, RenderSettings  # noqa: E402
from avinavi.sim.terrain import TerrainGenerator  # noqa: E402

SENSOR_W, SENSOR_H = 1280, 720
FOV_DEG = 60.0


def build(basemap_px: int, gsd: float, seed: int):
    basemap = TerrainGenerator(
        width_px=basemap_px, height_px=basemap_px, gsd_mpp=gsd, seed=seed
    ).generate()
    settings = RenderSettings(
        width_px=SENSOR_W, height_px=SENSOR_H, horizontal_fov_deg=FOV_DEG
    )
    extrinsics = CameraExtrinsics.from_config(CameraExtrinsicsConfig())
    simulator = CameraSimulator(
        basemap=basemap, basemap_gsd_mpp=gsd, settings=settings, extrinsics=extrinsics
    )
    return simulator, settings, extrinsics


def odometry_for(settings, extrinsics, resize_max_side: int):
    camera = CameraConfig()
    camera.fx = camera.fy = settings.fx
    camera.cx, camera.cy = settings.cx, settings.cy
    camera.width, camera.height = settings.width_px, settings.height_px
    config = VisualOdometryConfig()
    config.resize_max_side = resize_max_side
    return FrameToFrameOdometry.from_config(camera, extrinsics, config)


def run_case(
    label: str,
    trajectory,
    simulator,
    settings,
    extrinsics,
    height_scale: float = 1.0,
    yaw_bias_deg: float = 0.0,
    yaw_noise_deg: float = 0.0,
    resize_max_side: int = 480,
    seed: int = 11,
) -> dict:
    """Integrate flow along the trajectory with the given corruptions.

    ``height_scale`` multiplies the height handed to the odometry, standing in
    for a barometer that is off by a fixed proportion.  ``yaw_bias_deg`` is a
    constant heading error; ``yaw_noise_deg`` is per-frame heading noise.
    """
    odometry = odometry_for(settings, extrinsics, resize_max_side)
    rng = np.random.default_rng(seed)

    north = east = 0.0
    truth_north = truth_east = 0.0
    travelled = 0.0
    samples = []
    rejected = 0
    unrenderable = 0
    first = True

    for point in trajectory:
        frame = simulator.render(point)
        if frame is None:
            unrenderable += 1
            continue

        yaw = point.yaw_rad + math.radians(yaw_bias_deg)
        if yaw_noise_deg:
            yaw += math.radians(rng.normal(0.0, yaw_noise_deg))

        delta = odometry.update(
            frame,
            timestamp_monotonic=point.time_s,
            height_m=point.height_m * height_scale,
            roll_rad=point.roll_rad,
            pitch_rad=point.pitch_rad,
            yaw_rad=yaw,
        )
        if first:
            # The first frame has nothing to track from; start integrating from
            # the truth so the measurement is of drift, not of an offset.
            north, east = point.north_m, point.east_m
            truth_north, truth_east = point.north_m, point.east_m
            first = False
            continue

        step = math.hypot(point.north_m - truth_north, point.east_m - truth_east)
        truth_north, truth_east = point.north_m, point.east_m
        travelled += step

        if delta is None:
            # The odometry refused to answer.  The honest thing is to carry the
            # position unchanged and keep counting the distance flown, because
            # that is exactly what the flight code does.
            rejected += 1
        else:
            north += delta.north_m
            east += delta.east_m

        error = math.hypot(north - truth_north, east - truth_east)
        samples.append(
            {
                "t": round(point.time_s, 3),
                "travelled_m": round(travelled, 2),
                "error_m": round(error, 3),
                "north_err": round(north - truth_north, 3),
                "east_err": round(east - truth_east, 3),
            }
        )

    return summarise(label, samples, rejected, unrenderable)


def summarise(label, samples, rejected, unrenderable) -> dict:
    if not samples:
        return {"label": label, "samples": 0}

    travelled = np.array([s["travelled_m"] for s in samples])
    error = np.array([s["error_m"] for s in samples])
    usable = travelled > 50.0

    # The coefficient the configuration calls drift_fraction: error as a
    # fraction of distance.  Fitted through the origin, because zero distance
    # flown is zero drift by construction.
    if usable.sum() >= 5:
        fraction = float(
            np.sum(travelled[usable] * error[usable]) / np.sum(travelled[usable] ** 2)
        )
    else:
        fraction = float("nan")

    # Is the growth linear?  Fit error = a * travelled**b in log space; b near 1
    # is linear, b well above 1 means the linear model under-reports with
    # distance, which is the dangerous direction.
    exponent = float("nan")
    if usable.sum() >= 5 and np.all(error[usable] > 0):
        exponent = float(
            np.polyfit(np.log(travelled[usable]), np.log(error[usable]), 1)[0]
        )

    return {
        "label": label,
        "samples": len(samples),
        "rejected_updates": rejected,
        "unrenderable_frames": unrenderable,
        "distance_m": round(float(travelled[-1]), 1),
        "final_error_m": round(float(error[-1]), 2),
        "max_error_m": round(float(error.max()), 2),
        "drift_fraction": round(fraction, 5),
        "drift_percent": round(100 * fraction, 2),
        "growth_exponent": round(exponent, 3),
        # The signed mean says whether the error is a bias or a wander; a bias
        # means something systematic (scale or heading), not tracking noise.
        "mean_north_err_m": round(
            float(np.mean([s["north_err"] for s in samples])), 2
        ),
        "mean_east_err_m": round(float(np.mean([s["east_err"] for s in samples])), 2),
        "trace": samples[:: max(1, len(samples) // 12)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=float, default=240.0)
    parser.add_argument("--duration", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=0.2, help="seconds between frames")
    parser.add_argument("--basemap-px", type=int, default=4000)
    parser.add_argument("--gsd", type=float, default=0.5)
    parser.add_argument("--extent-m", type=float, default=600.0)
    parser.add_argument("--resize-max-side", type=int, default=480)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    simulator, settings, extrinsics = build(args.basemap_px, args.gsd, seed=7)
    trajectory = fly_circuit(
        extent_m=args.extent_m,
        airspeed_mps=22.0,
        cruise_height_m=args.height,
        max_bank_deg=30.0,
        wind_ned=(4.0, -3.0),
        duration_s=args.duration,
        dt_s=args.dt,
        pattern="racetrack",
    )

    print("flow drift, %d frames at %.1f s intervals, %.0f m AGL"
          % (len(trajectory), args.dt, args.height))
    print("frame interval %.2f s at 22 m/s = %.1f m of baseline per frame\n"
          % (args.dt, args.dt * 22.0))

    cases = [
        # label,                         height scale, yaw bias, yaw noise
        ("tracking noise only",                  1.00,      0.0,       0.0),
        ("height +5% (baro high)",               1.05,      0.0,       0.0),
        ("height -5% (baro low)",                0.95,      0.0,       0.0),
        ("height +10%",                          1.10,      0.0,       0.0),
        ("heading bias +2 deg",                  1.00,      2.0,       0.0),
        ("heading bias +5 deg",                  1.00,      5.0,       0.0),
        ("heading noise 2 deg rms",              1.00,      0.0,       2.0),
        ("realistic: 5% height + 2 deg bias",    1.05,      2.0,       1.0),
    ]

    print("  %-34s %8s %9s %9s %7s %8s"
          % ("case", "flown", "final err", "drift %", "exp", "rejected"))
    results = []
    for label, height_scale, yaw_bias, yaw_noise in cases:
        row = run_case(
            label,
            trajectory,
            simulator,
            settings,
            extrinsics,
            height_scale=height_scale,
            yaw_bias_deg=yaw_bias,
            yaw_noise_deg=yaw_noise,
            resize_max_side=args.resize_max_side,
        )
        results.append(row)
        print("  %-34s %7.0fm %8.1fm %8.2f%% %7.2f %8d"
              % (label, row["distance_m"], row["final_error_m"],
                 row["drift_percent"], row["growth_exponent"],
                 row["rejected_updates"]), flush=True)

    print("\nconfigured drift_fraction is 5.00%; the first row is the floor and")
    print("the last is what a real flight would see.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"cases": results, "height_m": args.height,
                       "dt_s": args.dt}, handle, indent=2)
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
