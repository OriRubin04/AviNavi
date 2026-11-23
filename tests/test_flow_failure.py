"""A frozen position must never be published as a confident current one.

This file exists because of the worst bug this project has had. Optical flow
carries the position across a map-matching dropout, and the elapsed time that
bounded it was a **sum of optical flow's own timestamps**. So when flow itself
failed - over water, under cloud shadow, on a featureless field - it produced
nothing, which meant:

* the accumulated offset stopped changing, so the position froze;
* the sigma stopped growing, because it was a function of distance flown;
* the elapsed time stopped advancing, so the time limit never tripped;
* and a fresh timestamp was stamped on it every cycle.

Measured: the same position, with the same 3 m sigma, published as current
indefinitely, while the aircraft flew away at 22 m/s. After one minute that is
1320 m of error reported with 3 m of confidence - precisely the fake position
the whole system is built to refuse, produced by the component meant to prevent
one.

The fix has two halves and both are load-bearing:

1. Time comes from a clock, not from flow's output. Anything derived from a
   failed component's output stops advancing when it fails, so every limit
   built on it silently stops applying.
2. Seconds nobody observed enter the uncertainty at the airframe's speed. A
   frozen position then reports its own worthlessness: a second of silence
   costs 40 m of sigma, and the publishing gate - a tolerance in metres -
   closes on its own.
"""

from __future__ import annotations

import math

import pytest

from avinavi.config import Config
from avinavi.localization.visual_odometry import DeadReckoner, OdometryDelta


class Clock:
    """A clock the test drives, so staleness can be tested without sleeping."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def delta(north=2.2, east=0.0, dt_s=0.1) -> OdometryDelta:
    return OdometryDelta(
        north_m=north, east_m=east, dt_s=dt_s, tracked_points=120, residual_m=0.2
    )


def reckoner(clock, **overrides) -> DeadReckoner:
    params = dict(
        drift_fraction=0.05,
        max_dead_reckoning_s=3.0,
        max_search_guidance_s=60.0,
        max_position_sigma_m=50.0,
        unobserved_speed_mps=40.0,
    )
    params.update(overrides)
    return DeadReckoner(clock=clock, **params)


class TestTheFrozenPositionBug:
    def test_uncertainty_grows_while_flow_is_silent(self):
        """The heart of it: no displacement must still mean more uncertainty."""
        clock = Clock()
        r = reckoner(clock)
        r.start(base_sigma_m=3.0)
        for _ in range(5):
            clock.advance(0.1)
            r.accumulate(delta())
        with_flow = r.sigma_m
        assert with_flow == pytest.approx(3.05, abs=0.05)

        clock.advance(1.0)  # flow produces nothing for a second
        assert r.sigma_m > with_flow + 30.0, (
            "a second unobserved at 40 m/s must cost tens of metres of sigma"
        )

    def test_the_position_itself_stays_put(self):
        """It must not invent motion - only admit that it does not know."""
        clock = Clock()
        r = reckoner(clock)
        r.start(base_sigma_m=3.0)
        clock.advance(0.1)
        r.accumulate(delta(north=10.0))
        before = r.offset_ned_m
        clock.advance(5.0)
        assert r.offset_ned_m == before

    def test_publishing_stops_once_the_sigma_passes_the_tolerance(self):
        clock = Clock()
        r = reckoner(clock)
        r.start(base_sigma_m=3.0)
        clock.advance(0.1)
        r.accumulate(delta())
        assert r.publishable is True
        clock.advance(50.0 / 40.0 + 0.1)
        assert r.publishable is False

    def test_the_old_behaviour_would_have_published_forever(self):
        """Pins the bug's shape so a 'simplification' cannot restore it.

        The old sigma was hypot(base, drift * distance) with distance frozen,
        so it was constant - and the old elapsed time was a sum of flow's dt,
        so it was frozen too. Both conditions for publishing held forever.
        """
        clock = Clock()
        r = reckoner(clock)
        r.start(base_sigma_m=3.0)
        clock.advance(0.1)
        r.accumulate(delta())

        old_style_sigma = math.hypot(3.0, 0.05 * r.distance_m)
        clock.advance(60.0)
        assert old_style_sigma == pytest.approx(math.hypot(3.0, 0.05 * r.distance_m))
        assert old_style_sigma < 5.0, "the old sigma stayed small forever"
        assert r.sigma_m > 1000.0, "the honest one does not"

    def test_a_minute_of_silence_is_reported_as_kilometres(self):
        """The number the old code hid: 22 m/s for 60 s is 1320 m of ignorance."""
        clock = Clock()
        r = reckoner(clock)
        r.start(base_sigma_m=3.0)
        clock.advance(60.0)
        assert r.sigma_m > 1320.0


class TestTheTimeLimitActuallyApplies:
    def test_the_backstop_uses_a_clock_not_flow_timestamps(self):
        """With no deltas at all, the time limit must still trip.

        Previously it could not: the elapsed time it compared against was a sum
        of the deltas that were not arriving.
        """
        clock = Clock()
        r = reckoner(clock)
        r.start(base_sigma_m=3.0)
        clock.advance(10.0)
        assert r.wall_elapsed_s == pytest.approx(10.0)
        assert r.publishable is False

    def test_unobserved_time_is_measured_from_the_last_delta(self):
        clock = Clock()
        r = reckoner(clock)
        r.start(base_sigma_m=3.0)
        clock.advance(2.0)
        r.accumulate(delta())
        assert r.unobserved_s == pytest.approx(0.0, abs=1e-9)
        clock.advance(1.5)
        assert r.unobserved_s == pytest.approx(1.5)

    def test_an_inactive_reckoner_reports_no_elapsed_time(self):
        clock = Clock()
        r = reckoner(clock)
        assert r.unobserved_s == 0.0
        assert r.wall_elapsed_s == 0.0
        r.start(base_sigma_m=3.0)
        clock.advance(5.0)
        r.clear()
        assert r.wall_elapsed_s == 0.0


class TestTheHandoffToInertial:
    """Flow failing is reported separately from flow drifting.

    They mean different things to an operator: drifting past the tolerance is
    the system working as designed, while "the camera can no longer see the
    ground move" is a failure, and it is the one that leaves nothing but the
    autopilot's own IMU.
    """

    def test_flow_loss_is_detected(self):
        clock = Clock()
        r = reckoner(clock)
        r.start(base_sigma_m=3.0)
        clock.advance(0.1)
        r.accumulate(delta())
        assert r.flow_lost is False
        clock.advance(50.0 / 40.0 + 0.1)
        assert r.flow_lost is True

    def test_flow_loss_and_drift_are_distinguishable(self):
        """Drifting far on healthy flow must not be reported as flow lost."""
        clock = Clock()
        r = reckoner(clock, max_dead_reckoning_s=600.0)
        r.start(base_sigma_m=3.0)
        for _ in range(500):
            clock.advance(0.1)
            r.accumulate(delta(north=2.2))
        assert r.sigma_m > 50.0, "far enough that publishing has stopped"
        assert r.publishable is False
        assert r.flow_lost is False, "but flow is working perfectly"

    def test_the_timeout_follows_the_tolerance(self):
        """Derived, not configured: waiting past worthlessness gains nothing."""
        clock = Clock()
        strict = reckoner(clock, max_position_sigma_m=20.0)
        lenient = reckoner(clock, max_position_sigma_m=200.0)
        assert strict._flow_timeout_s < lenient._flow_timeout_s
        assert strict._flow_timeout_s == pytest.approx(0.5)

    def test_flow_loss_needs_an_active_reckoning(self):
        clock = Clock()
        r = reckoner(clock)
        clock.advance(100.0)
        assert r.flow_lost is False

    def test_a_new_map_fix_clears_everything(self):
        clock = Clock()
        r = reckoner(clock)
        r.start(base_sigma_m=3.0)
        clock.advance(10.0)
        assert r.flow_lost is True
        r.start(base_sigma_m=2.0)
        assert r.flow_lost is False
        assert r.publishable is True
        assert r.sigma_m == pytest.approx(2.0)


class TestTheConfiguredDefaults:
    def test_the_limit_is_a_distance_not_a_duration(self):
        """Because 'roughly where we are' is a tolerance in metres."""
        assert Config().visual_odometry.max_position_sigma_m == 50.0

    def test_fifty_metres_is_about_a_kilometre_of_flow_navigation(self):
        config = Config().visual_odometry
        reach = config.max_position_sigma_m / config.drift_fraction
        assert reach == pytest.approx(1000.0)

    def test_and_about_a_second_of_flow_failure(self):
        config = Config().visual_odometry
        timeout = config.max_position_sigma_m / config.unobserved_speed_mps
        assert 1.0 <= timeout <= 2.0, (
            "a failure of flow must be caught in about a second, not a minute"
        )

    def test_the_unobserved_speed_exceeds_cruise(self):
        """It bounds what the aircraft *could* have done, not what it usually does."""
        assert Config().visual_odometry.unobserved_speed_mps > 22.0
