"""The simulation's frame cache must not serve frames from a different flight.

This file exists because the cache did exactly that.  An altitude experiment ran
seven variants - 120 m to 360 m, and three fields of view - and every one
returned byte-identical lock rates, p50 errors and inlier counts, six of them in
zero seconds.  The cause was that ``_trajectory_fingerprint`` summarised a
trajectory by its endpoints and headings but not its *height*, and the frame key
listed ``width_px`` but not ``horizontal_fov_deg``.

That is the most dangerous kind of bug this project can have: it does not crash,
it does not warn, and it produces a confident number that is wrong.  The only
reason it was caught is that the identical results were impossible on their face.

The lesson in the fix is that the keys no longer name a *selection* of settings.
``RenderSettings`` and the extrinsics config are dataclasses, so the key is all
of their fields, which cannot fall out of step when someone adds one.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from avinavi.config import CameraExtrinsicsConfig
from avinavi.sim.evaluate import (
    _extrinsics_fingerprint,
    _render_fingerprint,
    _trajectory_fingerprint,
)
from avinavi.sim.flight import TrajectoryPoint, fly_circuit
from avinavi.sim.render import RenderSettings


def circuit(height_m: float, **overrides):
    params = dict(
        extent_m=600.0,
        airspeed_mps=22.0,
        cruise_height_m=height_m,
        max_bank_deg=30.0,
        wind_ned=(4.0, -3.0),
        duration_s=120.0,
        dt_s=0.5,
        pattern="racetrack",
    )
    params.update(overrides)
    return fly_circuit(**params)


class TestTrajectoryFingerprint:
    def test_altitude_changes_the_fingerprint(self):
        """The exact bug: same ground track, different altitude, same key."""
        assert _trajectory_fingerprint(circuit(120.0)) != _trajectory_fingerprint(
            circuit(360.0)
        )

    @pytest.mark.parametrize("height", [120.0, 180.0, 240.0, 300.0, 360.0])
    def test_every_altitude_in_the_experiment_is_distinct(self, height):
        """All five altitude variants must key differently from the baseline."""
        baseline = _trajectory_fingerprint(circuit(120.0))
        fingerprint = _trajectory_fingerprint(circuit(height))
        if height == 120.0:
            assert fingerprint == baseline
        else:
            assert fingerprint != baseline

    def test_identical_trajectories_still_share(self):
        """The cache has to keep working, or an A/B comparison is not fair."""
        assert _trajectory_fingerprint(circuit(120.0)) == _trajectory_fingerprint(
            circuit(120.0)
        )

    def test_a_climb_is_not_mistaken_for_level_flight(self):
        """Same endpoints, same headings, different altitude in between."""
        level = [
            TrajectoryPoint(
                time_s=i * 0.5, north_m=float(i), east_m=0.0, height_m=200.0,
                roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0,
                ground_speed_mps=22.0, track_rad=0.0,
            )
            for i in range(11)
        ]
        # Dips to 100 m in the middle and returns: identical first and last point.
        climb = [
            TrajectoryPoint(
                time_s=i * 0.5, north_m=float(i), east_m=0.0,
                height_m=200.0 if i in (0, 10) else 100.0,
                roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0,
                ground_speed_mps=22.0, track_rad=0.0,
            )
            for i in range(11)
        ]
        assert _trajectory_fingerprint(level) != _trajectory_fingerprint(climb)

    def test_empty_trajectory_is_handled(self):
        assert _trajectory_fingerprint([]) == (0,)


class TestRenderFingerprint:
    def test_field_of_view_changes_the_fingerprint(self):
        """The second half of the bug: a different lens reused the old frames."""
        assert _render_fingerprint(
            RenderSettings(width_px=1280, height_px=720, horizontal_fov_deg=60.0)
        ) != _render_fingerprint(
            RenderSettings(width_px=1280, height_px=720, horizontal_fov_deg=120.0)
        )

    def test_identical_settings_still_share(self):
        assert _render_fingerprint(RenderSettings()) == _render_fingerprint(
            RenderSettings()
        )

    @pytest.mark.parametrize(
        "field", [f.name for f in dataclasses.fields(RenderSettings)]
    )
    def test_every_field_affects_the_key(self, field):
        """No setting may be silently excluded, including ones added later.

        This is the test that makes the fix durable: it enumerates the dataclass
        rather than a list someone has to remember to update.
        """
        base = RenderSettings()
        current = getattr(base, field)
        if isinstance(current, bool):
            changed = not current
        elif isinstance(current, int):
            changed = current + 7
        elif isinstance(current, float):
            changed = current + 1.5
        else:
            pytest.skip("no obvious perturbation for %s" % field)
        other = dataclasses.replace(base, **{field: changed})
        assert _render_fingerprint(base) != _render_fingerprint(other), (
            "%s does not affect the frame cache key" % field
        )


class TestExtrinsicsFingerprint:
    def test_mount_rotation_changes_the_fingerprint(self):
        """The simulator renders through the mount, so it belongs in the key."""
        base = CameraExtrinsicsConfig()
        turned = CameraExtrinsicsConfig()
        turned.rotation.yaw_deg = base.rotation.yaw_deg + 90.0
        assert _extrinsics_fingerprint(base) != _extrinsics_fingerprint(turned)

    def test_identical_mounts_still_share(self):
        assert _extrinsics_fingerprint(
            CameraExtrinsicsConfig()
        ) == _extrinsics_fingerprint(CameraExtrinsicsConfig())


class TestTheExperimentThatCaughtIt:
    """The seven variants of the scale experiment must all key differently.

    Phrased as the experiment rather than as the mechanism, because the mechanism
    is easy to re-break in a way that still passes a narrower test.
    """

    VARIANTS = [
        (120.0, 60.0),
        (180.0, 60.0),
        (240.0, 60.0),
        (300.0, 60.0),
        (360.0, 60.0),
        (120.0, 100.0),
        (120.0, 120.0),
    ]

    def test_all_seven_variants_have_distinct_frame_keys(self):
        keys = {
            (
                _trajectory_fingerprint(circuit(height)),
                _render_fingerprint(
                    RenderSettings(
                        width_px=1280, height_px=720, horizontal_fov_deg=fov
                    )
                ),
            )
            for height, fov in self.VARIANTS
        }
        assert len(keys) == len(self.VARIANTS), (
            "variants are sharing rendered frames; the experiment would measure "
            "the same imagery seven times"
        )

    def test_the_matched_pair_differs_despite_equal_rectified_size(self):
        """360 m / 60 deg and 120 m / 120 deg cover the same ground.

        They must still render separately - the whole point of the pair is that
        two *different* flights produce the same rectified size, and comparing a
        cached frame against itself would guarantee the answer.
        """
        assert math.isclose(
            2 * 360.0 * math.tan(math.radians(30.0)),
            2 * 120.0 * math.tan(math.radians(60.0)),
            rel_tol=1e-9,
        ), "the pair is no longer matched; update the experiment"

        a = (
            _trajectory_fingerprint(circuit(360.0)),
            _render_fingerprint(RenderSettings(horizontal_fov_deg=60.0)),
        )
        b = (
            _trajectory_fingerprint(circuit(120.0)),
            _render_fingerprint(RenderSettings(horizontal_fov_deg=120.0)),
        )
        assert a != b
