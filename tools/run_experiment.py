"""Measure the effect of each pipeline choice on lock rate.

Run with the real SuperPoint + LightGlue weights:

    python3 tools/run_experiment.py --backend superpoint

or with the classical backend, which is much faster and useful for sweeps:

    python3 tools/run_experiment.py --backend opencv

Every variant sees exactly the same basemap, the same trajectory and the same
rendered frames, so the differences are attributable to the pipeline change and
nothing else.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avinavi.config import Config  # noqa: E402
from avinavi.sim.evaluate import compare, run_simulation  # noqa: E402
from avinavi.sim.flight import fly_circuit  # noqa: E402
from avinavi.sim.render import RenderSettings  # noqa: E402


def base_config(backend: str) -> Config:
    config = Config()
    config.features.backend = backend
    if backend == "superpoint":
        config.superpoint.max_num_keypoints = 1024
        config.superpoint.resize_max_side = 640
        config.superpoint.min_keypoints = 40
        config.lightglue.max_candidate_tiles = 4
    else:
        config.superpoint.max_num_keypoints = 2048
        config.superpoint.resize_max_side = 768
        config.superpoint.min_keypoints = 40
        config.lightglue.max_candidate_tiles = 4

    config.prewarp.enabled = True
    config.photometric.enabled = True
    # The gates are measured separately below, so start with them off to keep
    # the lock-rate denominators comparable across variants.
    config.quality.sharpness_gate_enabled = False
    config.quality.max_bank_deg = 0.0
    config.geometry.min_inliers = 15
    config.geometry.min_matches = 18
    config.confidence.min_confidence = 0.35
    config.tile_search.initial_search_radius_m = 600.0
    config.tile_search.search_radius_m = 400.0
    config.rates.visual_localization_hz = 2.0
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["superpoint", "opencv"], default="opencv")
    parser.add_argument("--duration", type=float, default=150.0)
    parser.add_argument("--basemap-px", type=int, default=3000)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--quick", action="store_true", help="fewer variants")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)-7s %(name)-28s %(message)s"
    )

    workdir = os.path.join(
        os.environ.get("SCRATCH", "/tmp"), "avinavi_experiment_%s" % args.backend
    )
    os.makedirs(workdir, exist_ok=True)
    cache = {}

    # One trajectory and one camera, shared by every variant.
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

    def variant(label, mutate=None, **kwargs):
        config = base_config(args.backend)
        if mutate is not None:
            mutate(config)
        started = time.time()
        report = run_simulation(
            config=config,
            label=label,
            basemap_px=args.basemap_px,
            trajectory=trajectory,
            render_settings=kwargs.pop("render_settings", camera),
            workdir=workdir,
            max_cycles=args.max_cycles,
            cache=cache,
            **kwargs,
        )
        print(
            "  %-28s lock %5.1f%%  false %4.1f%%  p50 %6s m  (%.0fs)"
            % (
                label,
                100 * report.lock_rate,
                100 * report.false_fix_rate,
                report.as_dict()["position_error_m"]["median"],
                time.time() - started,
            ),
            flush=True,
        )
        return report

    reports = []
    print("\n=== A/B: what the rectification is worth (%s backend) ===" % args.backend)

    def no_prewarp(config):
        config.prewarp.enabled = False

    def no_clahe(config):
        config.photometric.enabled = False

    def neither(config):
        config.prewarp.enabled = False
        config.photometric.enabled = False

    reports.append(variant("baseline (raw frames)", neither))
    reports.append(variant("+ CLAHE only", no_prewarp))
    reports.append(variant("+ rectify only", no_clahe))
    reports.append(variant("+ rectify + CLAHE", None))

    if not args.quick:
        print("\n=== sensitivity ===")

        def blurry(config):
            pass

        reports.append(
            variant(
                "slow shutter 1/60s",
                None,
                render_settings=RenderSettings(
                    width_px=1280, height_px=720, horizontal_fov_deg=60.0,
                    exposure_s=1.0 / 60.0,
                ),
            )
        )

        def with_gates(config):
            config.quality.sharpness_gate_enabled = True
            config.quality.max_bank_deg = 35.0

        reports.append(variant("+ quality gates", with_gates))
        reports.append(
            variant("no appearance gap", None, appearance_strength=0.0)
        )
        reports.append(
            variant("double appearance gap", None, appearance_strength=2.0)
        )

    print("\n" + compare(reports))

    payload = {
        "backend": args.backend,
        "trajectory": reports[0].trajectory if reports else {},
        "runs": [r.as_dict() for r in reports],
    }
    out = args.out or os.path.join(workdir, "results.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
