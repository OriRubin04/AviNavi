"""Find the highest lock rate that fits a given per-cycle time budget.

The A/B in ``run_experiment.py`` answers "does rectification help".  This
answers the question that decides whether the system is flyable at all: the
real SuperPoint + LightGlue stack takes seconds per cycle on a CPU, and a fixed
wing covers 22 metres a second, so lock rate alone is not the figure of merit.

Three settings dominate the cost, and each trades accuracy for time
differently:

``lightglue.max_candidate_tiles``
    LightGlue runs once per candidate tile, so this multiplies the dominant
    cost directly.  Usually the cheapest win: a tighter search window means
    fewer candidates for the same hit rate.

``superpoint.resize_max_side``
    Extraction cost is roughly quadratic in this, and it also sets how much
    real detail reaches the matcher.

``superpoint.max_num_keypoints``
    LightGlue's attention is quadratic in the keypoint count.

Each row is run against exactly the same basemap, trajectory and frames, so
the differences are attributable to the setting and nothing else.  Usage:

    python3 tools/run_speed_sweep.py --backend superpoint --max-cycles 40
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avinavi.sim.evaluate import run_simulation  # noqa: E402
from avinavi.sim.flight import fly_circuit  # noqa: E402
from avinavi.sim.render import RenderSettings  # noqa: E402
from run_experiment import base_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["superpoint", "opencv"], default="superpoint")
    parser.add_argument("--duration", type=float, default=150.0)
    parser.add_argument("--basemap-px", type=int, default=3000)
    parser.add_argument("--max-cycles", type=int, default=40)
    parser.add_argument(
        "--budget-s",
        type=float,
        default=0.5,
        help="per-cycle latency target; 0.5 s is 2 Hz",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(name)-24s %(message)s")

    workdir = os.path.join(
        os.environ.get("SCRATCH", "/tmp"), "avinavi_sweep_%s" % args.backend
    )
    os.makedirs(workdir, exist_ok=True)
    cache = {}

    trajectory = fly_circuit(
        extent_m=args.basemap_px * 0.5 * 0.42,
        airspeed_mps=22.0,
        cruise_height_m=120.0,
        max_bank_deg=30.0,
        wind_ned=(4.0, -3.0),
        duration_s=args.duration,
        dt_s=0.5,
        pattern="racetrack",
    )
    camera = RenderSettings(width_px=1280, height_px=720, horizontal_fov_deg=60.0)

    # (label, candidate tiles, resize, keypoints)
    rows = [
        ("reference", 4, 640, 1024),
        ("tiles 2", 2, 640, 1024),
        ("tiles 1", 1, 640, 1024),
        ("resize 512", 4, 512, 1024),
        ("resize 384", 4, 384, 1024),
        ("keypoints 512", 4, 640, 512),
        ("keypoints 256", 4, 640, 256),
        ("tiles 2 + resize 512", 2, 512, 1024),
        ("tiles 2 + kp 512", 2, 640, 512),
        ("tiles 1 + resize 512 + kp 512", 1, 512, 512),
    ]

    reports = []
    print(
        "\n=== speed/accuracy sweep (%s backend, budget %.0f ms/cycle) ===\n"
        % (args.backend, args.budget_s * 1000.0)
    )
    header = "%-30s %7s %8s %9s %9s %8s %8s %7s" % (
        "setting", "lock", "false", "err p50", "ms mean", "ms p95", "fix/s", "fits?"
    )
    print(header)
    print("-" * len(header))

    for label, tiles, resize, keypoints in rows:
        config = base_config(args.backend)
        config.lightglue.max_candidate_tiles = tiles
        config.superpoint.resize_max_side = resize
        config.superpoint.max_num_keypoints = keypoints
        started = time.time()
        report = run_simulation(
            config=config,
            label=label,
            basemap_px=args.basemap_px,
            trajectory=trajectory,
            render_settings=camera,
            workdir=workdir,
            max_cycles=args.max_cycles,
            cache=cache,
            # The separation probe costs an extra match per sample and would
            # distort the very timing this sweep exists to measure.
            measure_separation=False,
        )
        data = report.as_dict()
        mean_ms = (data["processing_time_s"]["mean"] or 0.0) * 1000.0
        p95_ms = (data["processing_time_s"]["p95"] or 0.0) * 1000.0
        # Judge on p95, not the mean: the rate has to be set from the worst
        # case or the loop silently falls behind in exactly the hard frames.
        fits = "yes" if p95_ms <= args.budget_s * 1000.0 else "no"
        # Map fixes per second of flight: the figure that actually matters.
        # Lock rate alone is misleading when cycle times differ by 50%, because
        # a configuration that locks half as often but runs twice as fast
        # delivers the same number of fixes - and the gaps between them, not the
        # ratio, are what the dead reckoning has to cover.
        fixes_per_second = (
            report.lock_rate / (mean_ms / 1000.0) if mean_ms > 0 else 0.0
        )
        print(
            "%-30s %6.1f%% %7.1f%% %9s %9.0f %8.0f %8.3f %7s"
            % (
                label,
                100.0 * report.lock_rate,
                100.0 * report.false_fix_rate,
                data["position_error_m"]["median"]
                if data["position_error_m"]["median"] is not None
                else "-",
                mean_ms,
                p95_ms,
                fixes_per_second,
                fits,
            ),
            flush=True,
        )
        reports.append(
            {
                "label": label,
                "max_candidate_tiles": tiles,
                "resize_max_side": resize,
                "max_num_keypoints": keypoints,
                "fits_budget": fits == "yes",
                "fixes_per_second": round(fixes_per_second, 4),
                "metres_between_fixes_at_22mps": (
                    round(22.0 / fixes_per_second, 1) if fixes_per_second > 0 else None
                ),
                "wall_time_s": round(time.time() - started, 1),
                **data,
            }
        )

    # Rank by fixes per second, which is what the navigation solution lives on.
    print("\n--- by map fixes per second of flight (22 m/s) ---")
    for row in sorted(reports, key=lambda r: -r["fixes_per_second"]):
        gap = row["metres_between_fixes_at_22mps"]
        print(
            "  %-30s %5.3f fix/s  one fix every %s"
            % (
                row["label"],
                row["fixes_per_second"],
                "%.0f m" % gap if gap is not None else "never",
            )
        )

    affordable = [r for r in reports if r["fits_budget"]]
    print("\n--- within the %.0f ms budget, by lock rate ---" % (args.budget_s * 1000.0))
    if not affordable:
        print(
            "  NONE. Every configuration tested is too slow for %.0f ms. The "
            "options are a larger budget (a lower localization rate, with the\n"
            "  frame-to-frame odometry covering the gap), a faster runtime (ONNX\n"
            "  Runtime, or XFeat + LighterGlue), or the opencv backend."
            % (args.budget_s * 1000.0)
        )
    else:
        for row in sorted(affordable, key=lambda r: -r["lock_rate"]):
            print(
                "  %-30s lock %5.1f%%  p95 %4.0f ms"
                % (row["label"], 100.0 * row["lock_rate"],
                   (row["processing_time_s"]["p95"] or 0.0) * 1000.0)
            )

    out = args.out or os.path.join(workdir, "sweep.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"backend": args.backend, "budget_s": args.budget_s, "rows": reports},
                  handle, indent=2)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
