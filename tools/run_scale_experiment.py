"""Is the lock rate limited by how many pixels survive rectification?

The suspicion
-------------
Rectification resamples the camera frame to the tile's ground sample distance.
At 120 m with a 60 degree lens the frame covers 139 m of ground, and against a
0.5 m/px map that is **278 x 155 px** - from a 1280 x 720 sensor.  Ninety-five
per cent of the pixels are discarded before SuperPoint ever sees them, and the
observed match counts (16-38) are low enough that this could be the binding
constraint rather than appearance, rotation or search.

The sharper statement, and why it is testable
----------------------------------------------
The rectified image is ``footprint / tile_gsd`` on a side.  It does **not**
depend on altitude or on sensor resolution directly - only on how much ground
the frame covers and how finely the map samples it.  So two quite different
aircraft configurations can produce exactly the same rectified image size:

    360 m altitude, 60 degree lens  ->  416 m of ground  ->  831 x 467 px
    120 m altitude, 120 degree lens ->  416 m of ground  ->  831 x 467 px

If rectified size is what matters, those two must score the same despite
differing by a factor of three in altitude.  If they do not, the hypothesis is
wrong and the altitude effect is something else - thinner air between camera and
ground, a larger search footprint, or different terrain under the track.

That is the experiment.  It is a prediction that can fail, rather than a sweep
that will always produce some ordering.

What this deliberately does not do
-----------------------------------
It does not change the map's resolution, which is the other way to close the
gap.  Doing that in the simulator means building tiles at a different GSD from
the world they are cut from, and the current harness renders the camera from the
same raster it tiles - so a finer map would also be a finer *world*, with terrain
detail that does not exist at 0.5 m/px.  That is a confound, not a measurement,
and the altitude/field-of-view pair above answers the same question without it.

Every variant shares one basemap, one tile database and one tiling, so the only
differences are the ones named.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avinavi.config import Config  # noqa: E402
from avinavi.sim.evaluate import run_simulation  # noqa: E402
from avinavi.sim.flight import fly_circuit  # noqa: E402
from avinavi.sim.render import RenderSettings  # noqa: E402

SENSOR_W, SENSOR_H = 1280, 720
TILE_GSD = 0.5
TILE_PX = 1024
MAX_BANK_DEG = 30.0


def base_config(backend: str, keypoints=None) -> Config:
    """The configuration the 56.7% measurement used, so results stay comparable."""
    config = Config()
    config.features.backend = backend
    if backend == "superpoint":
        config.superpoint.max_num_keypoints = 1024
        config.superpoint.resize_max_side = 640
    else:
        config.superpoint.max_num_keypoints = 2048
        config.superpoint.resize_max_side = 768
    if keypoints:
        config.superpoint.max_num_keypoints = int(keypoints)
    config.superpoint.min_keypoints = 40
    config.lightglue.max_candidate_tiles = 4
    config.prewarp.enabled = True
    config.photometric.enabled = False
    config.quality.sharpness_gate_enabled = False
    config.quality.max_bank_deg = 0.0
    config.geometry.min_inliers = 15
    config.geometry.min_matches = 18
    config.confidence.min_confidence = 0.35
    config.tile_search.initial_search_radius_m = 600.0
    config.tile_search.search_radius_m = 400.0
    config.rates.visual_localization_hz = 2.0
    config.camera.width, config.camera.height = SENSOR_W, SENSOR_H
    return config


def footprint_m(height_m: float, fov_deg: float) -> float:
    """Ground covered across the frame's long axis."""
    return 2.0 * height_m * math.tan(math.radians(fov_deg / 2.0))


def rectified_px(height_m: float, fov_deg: float) -> int:
    return int(round(footprint_m(height_m, fov_deg) / TILE_GSD))


def check_geometry(variants, basemap_px: int, extent_m: float, max_bank_deg: float) -> None:
    """Refuse to run a variant that would look past the edge of the map.

    Two earlier experiments in this project produced confident, wrong numbers
    because a variant was quietly seeing ground that was not there - once by
    looking 56 m sideways at different terrain, once by running off the basemap.
    A frame rendered partly off the map is not a harder frame, it is a different
    experiment, so this fails loudly instead.
    """
    half_map_m = basemap_px * TILE_GSD / 2.0
    tile_ground_m = TILE_PX * TILE_GSD
    problems = []
    for label, height, fov in variants:
        reach = extent_m + footprint_m(height, fov) / 2.0 * math.sqrt(2.0)
        if reach > half_map_m:
            problems.append(
                "%s reaches %.0f m from the centre but the map only extends %.0f m"
                % (label, reach, half_map_m)
            )
        if footprint_m(height, fov) > tile_ground_m:
            problems.append(
                "%s covers %.0f m of ground, more than one %.0f m tile, so the "
                "frame cannot fall inside a single tile"
                % (label, footprint_m(height, fov), tile_ground_m)
            )
        # A wide lens on a banking aircraft points its outer edge at the sky.
        # At 120 degrees and 30 degrees of bank the edge sits exactly on the
        # horizon, those frames cannot be rendered, and - this is the damaging
        # part - the frames lost are the *banked* ones, which are the hardest.
        # The variant then scores against an easier subset than every other
        # variant and looks better than it is.  Measured: 134 of 152 frames.
        edge_from_nadir = fov / 2.0 + max_bank_deg
        if edge_from_nadir >= 85.0:
            problems.append(
                "%s puts the frame edge %.0f degrees from nadir at %.0f degrees "
                "of bank, on or past the horizon. Those frames cannot render, "
                "and the ones lost are the banked ones, so this variant would "
                "be scored on an easier subset than the others"
                % (label, edge_from_nadir, max_bank_deg)
            )
    if problems:
        raise SystemExit(
            "the geometry of this experiment is invalid:\n  " + "\n  ".join(problems)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["superpoint", "opencv"], default="superpoint")
    parser.add_argument("--max-cycles", type=int, default=150)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="trajectory seconds; defaults to just enough for --max-cycles",
    )
    parser.add_argument("--basemap-px", type=int, default=4000)
    parser.add_argument("--extent-m", type=float, default=600.0)
    parser.add_argument(
        "--appearance",
        type=float,
        default=1.0,
        help="seasonal appearance gap between map and camera; 1.0 is the "
        "standard run, higher is harder",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="substring of a variant label; repeat the flag to select several. "
        "Not comma-separated: the labels themselves contain commas, and "
        "splitting on them turned '240 m, 60 deg' into the fragment '60 deg', "
        "which matches every 60 degree variant. The filter then silently "
        "selected all of them.",
    )
    parser.add_argument(
        "--keypoints",
        type=int,
        default=None,
        help="keypoint budget per frame and per tile; lowering it makes the "
        "task harder for every variant equally, which is how to get both "
        "halves of the matched pair out from under the 100%% ceiling",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR, format="%(levelname)-7s %(message)s")

    # (label, height_m, fov_deg).  The two arms are chosen so that the last
    # entry of each produces an identical rectified size by a different route.
    variants = [
        ("120 m, 60 deg (today)", 120.0, 60.0),
        ("180 m, 60 deg", 180.0, 60.0),
        ("240 m, 60 deg", 240.0, 60.0),
        ("300 m, 60 deg", 300.0, 60.0),
        ("360 m, 60 deg", 360.0, 60.0),
        ("120 m, 100 deg", 120.0, 100.0),
        # 120 m / 120 deg was the exact match for 360 m / 60 deg - both 831 px -
        # and had to be dropped. At 120 degrees the frame edge reaches the
        # horizon under 30 degrees of bank, so 18 of 152 frames did not render
        # and the variant was scored on the unbanked subset. It returned 100%
        # and 158 inliers, which looked like confirmation and was not.
        # check_geometry now refuses it outright.
        #
        # 240 m / 60 deg against 120 m / 100 deg (554 px against 572 px) is the
        # usable pair: both render every frame.
    ]
    if args.only:
        wanted = [piece.strip() for piece in args.only if piece.strip()]
        variants = [v for v in variants if any(w in v[0] for w in wanted)]
        if not variants:
            raise SystemExit("--only matched no variants")
        print("selected %d variant(s): %s"
              % (len(variants), ", ".join(v[0] for v in variants)))
    check_geometry(variants, args.basemap_px, args.extent_m, MAX_BANK_DEG)

    workdir = os.path.join(
        os.environ.get("SCRATCH", "/tmp"), "avinavi_scale_%s" % args.backend
    )
    os.makedirs(workdir, exist_ok=True)
    cache = {}

    print("\nRectified image size is footprint / tile GSD (%.2f m/px):" % TILE_GSD)
    print("  %-22s %-9s %-11s %-11s %s"
          % ("variant", "footprint", "live GSD", "GSD ratio", "rectified"))
    for label, height, fov in variants:
        fx = (SENSOR_W / 2.0) / math.tan(math.radians(fov / 2.0))
        live = height / fx
        width = rectified_px(height, fov)
        print("  %-22s %6.0f m  %7.3f m/px %7.1fx    %dx%d"
              % (label, footprint_m(height, fov), live, TILE_GSD / live,
                 width, int(round(width * SENSOR_H / SENSOR_W))))

    print("\n=== measuring (%s, %d cycles each) ===" % (args.backend, args.max_cycles))
    print("  %-22s %-7s %-7s %-9s %-8s %s"
          % ("variant", "lock", "false", "p50 err", "inliers", "cycles"))

    # Each variant now renders its own frames - that is the point of the fix in
    # evaluate.py - so rendering a long trajectory and then using 150 cycles of
    # it costs real time for nothing.  A dt of 0.5 s at 2 Hz means one cycle per
    # point, so ask for exactly the points that will be used.
    dt_s = 0.5
    duration_s = args.duration or (args.max_cycles + 2) * dt_s

    results = []
    for label, height, fov in variants:
        trajectory = fly_circuit(
            extent_m=args.extent_m,
            airspeed_mps=22.0,
            cruise_height_m=height,
            max_bank_deg=MAX_BANK_DEG,
            wind_ned=(4.0, -3.0),
            duration_s=duration_s,
            dt_s=dt_s,
            pattern="racetrack",
        )
        started = time.time()
        report = run_simulation(
            config=base_config(args.backend, args.keypoints),
            label=label,
            basemap_px=args.basemap_px,
            basemap_gsd_mpp=TILE_GSD,
            tile_size_px=TILE_PX,
            trajectory=trajectory,
            render_settings=RenderSettings(
                width_px=SENSOR_W, height_px=SENSOR_H, horizontal_fov_deg=fov
            ),
            workdir=workdir,
            max_cycles=args.max_cycles,
            appearance_strength=args.appearance,
            cache=cache,
        )
        data = report.as_dict()
        median_inliers = (
            sorted(report.inliers)[len(report.inliers) // 2] if report.inliers else 0
        )
        print("  %-22s %5.1f%%  %5.1f%%  %8s %7d  %d  (%.0fs)"
              % (label, 100 * report.lock_rate, 100 * report.false_fix_rate,
                 data["position_error_m"]["median"], median_inliers,
                 report.cycles, time.time() - started), flush=True)
        results.append(
            {
                "label": label,
                "height_m": height,
                "fov_deg": fov,
                "footprint_m": footprint_m(height, fov),
                "rectified_px": rectified_px(height, fov),
                "lock_rate": report.lock_rate,
                "false_fix_rate": report.false_fix_rate,
                "median_inliers": median_inliers,
                "cycles": report.cycles,
                "report": data,
            }
        )

    # -- the prediction, stated before the numbers were seen ----------------
    print("\n=== the matched pair ===")
    short = [r for r in results if r["cycles"] < args.max_cycles]
    for r in short:
        print("  WARNING  %s completed only %d of %d cycles - frames that could "
              "not be rendered are dropped silently, and the ones lost are the "
              "banked ones, so this variant is NOT comparable with the others."
              % (r["label"], r["cycles"], args.max_cycles))

    pair = [r for r in results if r["label"].startswith(("240 m, 60", "120 m, 100"))]
    if len(pair) == 2:
        a, b = pair
        if any(r in short for r in pair):
            print("  the pair cannot be compared: see the warning above.")
            return 0
        print("  %-16s %4d px rectified -> %5.1f%% lock, %3d inliers, %s m"
              % (a["label"], a["rectified_px"], 100 * a["lock_rate"],
                 a["median_inliers"], a["report"]["position_error_m"]["median"]))
        print("  %-16s %4d px rectified -> %5.1f%% lock, %3d inliers, %s m"
              % (b["label"], b["rectified_px"], 100 * b["lock_rate"],
                 b["median_inliers"], b["report"]["position_error_m"]["median"]))
        print("  nearly equal rectified size, twofold difference in altitude.")

        if min(a["lock_rate"], b["lock_rate"]) > 0.97:
            # Two lock rates at the ceiling agree about nothing, so fall back to
            # the inlier count, which does not saturate (42 -> 79 -> 120 -> 141
            # -> 152 across the altitude arm while lock sat at 100%).
            spread = abs(a["median_inliers"] - b["median_inliers"])
            scale = max(a["median_inliers"], b["median_inliers"], 1)
            print("  lock rate is at the ceiling for both and settles nothing; "
                  "comparing inliers instead, which do not saturate.")
            print("  inliers differ by %d (%.0f%%) -> %s"
                  % (spread, 100.0 * spread / scale,
                     "consistent with rectified size being the driver"
                     if spread <= 0.15 * scale
                     else "NOT consistent; rectified size is not the whole story"))
        else:
            gap = abs(a["lock_rate"] - b["lock_rate"]) * 100
            print("  gap: %.1f points -> %s" % (
                gap,
                "consistent with rectified size being the driver"
                if gap < 10
                else "NOT consistent; rectified size is not the whole story",
            ))
    else:
        print("  the matched pair is missing; the variant list was edited.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"variants": results}, handle, indent=2)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
