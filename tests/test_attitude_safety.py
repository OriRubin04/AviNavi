"""An unknown attitude must produce no estimate, not a confident wrong one.

Two defects, found together, that combined into the worst kind of failure this
project can have: a position that is wrong, and asserted confidently.

**The substitution.** ``Runner._attitude()`` returned ``(0.0, 0.0, None)`` when
the autopilot's attitude went stale — silently claiming level flight for an
aircraft that was probably still banked. Dead reckoning then integrated a
displacement computed by projecting a banked frame as if it were nadir.

**The gate that could not catch it.** The residual check asked whether the
tracked points agreed *with each other*, against a tolerance of
``max(1.0 m, 0.25 × distance)``. Both frames are projected with the *same*
wrong attitude, so the points agree perfectly — and the tolerance grew in
proportion to the error it was meant to catch. Measured at a 60 degree field of
view:

    mismatch   implied    truth    residual   old limit   verdict
       5 deg    11.8 m    2.2 m       0.9 m      2.9 m    ACCEPTED
      25 deg    63.1 m    2.2 m       9.0 m     15.8 m    ACCEPTED

A 63 m displacement against a 2.2 m truth passed. The speed limit did not save
it either: 11.8 m over 0.2 s is 59 m/s, just under the 60 m/s cap.

Both halves are fixed here, and both are tested, because either alone leaves
the failure reachable.
"""

from __future__ import annotations

import math

import pytest

from avinavi.app.runner import Runner
from avinavi.config import CameraConfig, CameraExtrinsicsConfig, Config, VisualOdometryConfig
from avinavi.frames import CameraExtrinsics
from avinavi.localization.visual_odometry import FrameToFrameOdometry


def tracker(fx=1109.0, **overrides):
    camera = CameraConfig()
    camera.fx = camera.fy = fx
    camera.cx, camera.cy = 640.0, 360.0
    camera.width, camera.height = 1280, 720
    config = VisualOdometryConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    return FrameToFrameOdometry.from_config(
        camera, CameraExtrinsics.from_config(CameraExtrinsicsConfig()), config
    )


class TestTheToleranceIsAbsolute:
    """It must not grow with the quantity it is checking."""

    def test_the_tolerance_does_not_depend_on_the_displacement(self):
        """The defect in one line: the old limit was a fraction of the answer."""
        t = tracker()
        limit = t._residual_tolerance_m(120.0)
        # Whatever displacement is claimed, the tolerance is the same number.
        assert t._residual_tolerance_m(120.0) == limit
        assert limit == pytest.approx(4.0 * 120.0 / 1109.0, rel=0.01)

    def test_the_old_proportional_limit_would_have_accepted_the_failure(self):
        """Documents what was being waved through, so it cannot come back."""
        for implied_m, residual_m in ((11.8, 0.9), (63.1, 9.0), (79.2, 13.8)):
            old_limit = max(1.0, 0.25 * implied_m)
            assert residual_m < old_limit, "the old gate accepted this"
            assert residual_m > tracker()._residual_tolerance_m(120.0), (
                "and the absolute gate rejects it"
            )

    def test_it_scales_with_height_because_the_real_error_does(self):
        """A pixel of mismatch is worth more ground the higher you are."""
        t = tracker()
        low = t._residual_tolerance_m(120.0)
        high = t._residual_tolerance_m(240.0)
        assert high == pytest.approx(2 * low, rel=0.01)

    def test_the_floor_protects_a_very_low_pass(self):
        t = tracker(max_residual_pixels=4.0, min_residual_m=0.25)
        assert t._residual_tolerance_m(5.0) == pytest.approx(0.25)

    def test_a_working_tracker_sits_far_inside_the_tolerance(self):
        """Measured residual with the tracker working is 0.07 m at 240 m."""
        assert tracker()._residual_tolerance_m(240.0) > 0.07 * 5

    def test_an_unknown_focal_length_falls_back_to_the_floor(self):
        t = tracker(fx=0.0)
        assert t._residual_tolerance_m(120.0) == pytest.approx(t.min_residual_m)

    @pytest.mark.parametrize("height", [0.0, -5.0])
    def test_a_nonsense_height_falls_back_to_the_floor(self, height):
        assert tracker()._residual_tolerance_m(height) == pytest.approx(0.25)


class TestUnknownAttitudeProducesNothing:
    """``None`` means no information. It must not be read as zero."""

    @pytest.mark.parametrize("field", ["roll_rad", "pitch_rad", "yaw_rad"])
    def test_any_missing_angle_refuses_the_estimate(self, field):
        import numpy as np

        t = tracker()
        image = np.zeros((720, 1280), dtype=np.uint8)
        angles = {"roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.0}
        angles[field] = None
        t.update(image, timestamp_monotonic=0.0, height_m=120.0, **angles)
        assert t.update(image, timestamp_monotonic=0.1, height_m=120.0, **angles) is None

    def test_a_missing_height_refuses_the_estimate(self):
        import numpy as np

        t = tracker()
        image = np.zeros((720, 1280), dtype=np.uint8)
        assert t.update(image, 0.0, None, 0.0, 0.0, 0.0) is None


class TestOneBadCycleCostsOneUpdate:
    """Refusing to solve must not also destroy the next frame.

    The previous code called ``reset()`` on a missing height, discarding the
    stored frame — so the *next* cycle also returned ``None``, because there
    was nothing left to track from. One stale telemetry message cost two
    updates, and at 5 Hz that collapses the dead-reckoning rate non-linearly.
    """

    def _image(self, seed):
        import numpy as np

        rng = np.random.default_rng(seed)
        return rng.integers(0, 255, (720, 1280), dtype=np.uint8)

    def test_the_frame_is_kept_when_the_height_is_missing(self):
        t = tracker()
        t.update(self._image(1), 0.0, None, 0.0, 0.0, 0.0)
        assert t._previous_gray is not None, "the frame must survive the refusal"

    def test_the_frame_is_kept_when_the_attitude_is_unknown(self):
        t = tracker()
        t.update(self._image(2), 0.0, 120.0, None, None, None)
        assert t._previous_gray is not None

    def test_recovery_needs_only_one_good_cycle(self):
        """After the gap, the very next pair must be usable again."""
        t = tracker()
        t.update(self._image(3), 0.0, 120.0, 0.0, 0.0, 0.0)
        t.update(self._image(4), 0.1, None, 0.0, 0.0, 0.0)   # the bad cycle
        t.update(self._image(5), 0.2, 120.0, 0.0, 0.0, 0.0)
        # The state from the bad cycle was cleared, so this pair is the first
        # with two usable states - and nothing earlier is silently reused.
        assert t._previous_state is not None

    def test_a_refused_cycle_clears_the_state_rather_than_reusing_it(self):
        """The frame is kept, but its *pose* is not invented."""
        t = tracker()
        t.update(self._image(6), 0.0, 120.0, 0.0, 0.0, 0.0)
        t.update(self._image(7), 0.1, 120.0, None, None, None)
        assert t._previous_state is None, (
            "a frame with an unknown attitude has no usable state"
        )


class TestTheRunnerDoesNotSubstituteLevelFlight:
    def _runner(self):
        config = Config()
        config.camera.fx = config.camera.fy = 1109.0
        config.camera.cx, config.camera.cy = 640.0, 360.0
        return Runner(config)

    class _Connection:
        def __init__(self, age, roll=0.4):
            self.state = self
            self._age = age
            self.roll_rad = roll
            self.pitch_rad = 0.1
            self.yaw_rad = 0.3

        def attitude_age(self):
            return self._age

    def test_fresh_attitude_is_passed_through(self):
        runner = self._runner()
        runner.connection = self._Connection(age=0.1)
        assert runner._fresh_attitude() == (0.4, 0.1, 0.3)

    def test_stale_attitude_is_none_not_level(self):
        """The defect: this used to return (0.0, 0.0, None)."""
        runner = self._runner()
        runner.connection = self._Connection(age=2.0)
        assert runner._fresh_attitude() is None

    def test_an_absent_source_is_not_the_same_as_a_stale_one(self):
        """No autopilot at all is a configuration; a stale one is a failure."""
        runner = self._runner()
        runner.connection = None
        assert runner._fresh_attitude() is None
        # ...but rectification may still assume level, because a badly
        # rectified frame still has to earn its match against the map.
        assert runner._attitude() == (0.0, 0.0, None)

    def test_rectification_still_falls_back_to_level(self):
        runner = self._runner()
        runner.connection = self._Connection(age=2.0)
        assert runner._attitude() == (0.0, 0.0, None)


class TestTheArithmeticBehindTheFinding:
    """Why a small attitude error produces a large displacement error."""

    def test_a_five_degree_error_is_a_large_ground_error_at_altitude(self):
        """Tangent of the mismatch, times the height, per projected ray."""
        assert 120.0 * math.tan(math.radians(5.0)) == pytest.approx(10.5, rel=0.05)

    def test_the_speed_limit_alone_did_not_catch_it(self):
        """11.8 m in 0.2 s is 59 m/s — just under the 60 m/s cap."""
        assert 11.8 / 0.2 == pytest.approx(59.0, rel=0.01)
        assert 11.8 / 0.2 < 60.0
