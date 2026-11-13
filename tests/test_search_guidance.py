"""During a dropout the search window follows the aircraft, not the last fix.

The three-tier fallback is map matching, then frame-to-frame optical flow, then
the autopilot's own inertial coasting - which is what happens once this system
stops publishing and ArduPilot's EKF3 is left with its IMU.

The middle tier was doing half its job. It carried the *published* position
across a dropout, but the accumulator stopped dead at the publishing budget, so
the tile search window stayed where the last fix put it while the aircraft flew
on at 22 m/s: 66 m behind after 3 s, 440 m after 20. The radius grew - around a
point the aircraft had left. That is why a dropout could run to the end of a
flight without re-acquiring.

Separating the two budgets fixes it, and the separation is the idea worth
protecting: publishing a position needs metre accuracy, while aiming a search
window only needs to beat the width of the window. 5% drift over 440 m is 22 m,
useless for the first job and ample for the second.

What this explicitly does **not** do is correct a wrong centre.
``test_guidance_carries_a_false_fix_faithfully`` pins that, because it is the
reason the manual re-anchor still has to exist.
"""

from __future__ import annotations

import math

import pytest

from avinavi.app.runner import Runner
from avinavi.config import Config
from avinavi.geo import LocalOrigin, haversine_distance_m
from avinavi.localization.visual_odometry import OdometryDelta
from avinavi.types import (
    Attitude,
    GeoPoint,
    InvalidReason,
    MatchQuality,
    NEDPosition,
    VisualFix,
)


class _StubEngine:
    def __init__(self, origin: LocalOrigin) -> None:
        self.search_center = origin.geo
        self.search_radius_m = 400.0
        self.guided_to = []
        self.reanchored_to = None

    def guide_search_center(self, geo):
        self.guided_to.append(geo)
        self.search_center = geo

    def reanchor(self, geo, radius_m=None):
        self.reanchored_to = geo
        self.search_center = geo


@pytest.fixture
def origin() -> LocalOrigin:
    return LocalOrigin(32.0853, 34.7818, 40.0)


@pytest.fixture
def runner(origin) -> Runner:
    config = Config()
    config.camera.fx = config.camera.fy = 900.0
    config.camera.cx, config.camera.cy = 640.0, 360.0
    config.visual_odometry.enabled = True
    config.visual_odometry.max_dead_reckoning_s = 3.0
    config.visual_odometry.max_search_guidance_s = 60.0
    config.visual_odometry.search_guidance_min_move_m = 15.0
    runner = Runner(config)
    runner.engine = _StubEngine(origin)
    runner._origin = origin
    return runner


def map_fix(north=0.0, east=0.0) -> VisualFix:
    return VisualFix(
        timestamp_monotonic=0.0,
        timestamp_unix=1_760_000_000.0,
        valid=True,
        reason=InvalidReason.NONE,
        geo=GeoPoint(32.0853, 34.7818, 40.0),
        position_ned=NEDPosition(north, east, -120.0),
        attitude=Attitude(1.0, 0.0, 0.0, 0.0),
        yaw_rad=0.0,
        quality=MatchQuality(number_of_inliers=40, localization_confidence=0.8),
        tile_id="tile_0007",
        position_sigma_m=3.0,
    )


def fly(runner, north_m=0.0, east_m=0.0, dt_s=0.1):
    """Feed one odometry delta straight into the reckoner."""
    runner.reckoner.accumulate(
        OdometryDelta(
            north_m=north_m,
            east_m=east_m,
            dt_s=dt_s,
            tracked_points=120,
            residual_m=0.2,
        )
    )


class TestTheWindowFollowsTheAircraft:
    def test_the_centre_moves_with_the_flow_estimate(self, runner, origin):
        runner._last_map_fix = map_fix()
        runner.reckoner.start(3.0)
        fly(runner, north_m=80.0)
        runner._steer_search_with_flow()

        assert runner.engine.guided_to, "the window must move"
        moved = haversine_distance_m(origin.geo, runner.engine.guided_to[-1])
        assert moved == pytest.approx(80.0, rel=0.02)

    def test_the_direction_is_right(self, runner, origin):
        """Moving north must move the window north, not merely somewhere."""
        runner._last_map_fix = map_fix()
        runner.reckoner.start(3.0)
        fly(runner, north_m=100.0)
        runner._steer_search_with_flow()
        guided = runner.engine.guided_to[-1]
        assert guided.lat_deg > origin.geo.lat_deg
        assert guided.lon_deg == pytest.approx(origin.geo.lon_deg, abs=1e-5)

    def test_it_accumulates_over_a_long_dropout(self, runner, origin):
        """440 m of flight must move the window 440 m, not freeze at the budget."""
        runner._last_map_fix = map_fix()
        runner.reckoner.start(3.0)
        for _ in range(20):
            fly(runner, north_m=22.0, dt_s=1.0)
        runner._steer_search_with_flow()
        moved = haversine_distance_m(origin.geo, runner.engine.guided_to[-1])
        assert moved == pytest.approx(440.0, rel=0.02)

    def test_steering_outlives_the_publishing_budget(self, runner):
        """The whole point: publication stops at 3 s, steering does not."""
        runner._last_map_fix = map_fix()
        runner.reckoner.start(3.0)
        for _ in range(10):
            fly(runner, north_m=22.0, dt_s=1.0)
        assert runner.reckoner.publishable is False
        runner._steer_search_with_flow()
        assert runner.engine.guided_to, "steering must continue past the budget"

    def test_a_small_drift_does_not_churn_the_window(self, runner):
        runner._last_map_fix = map_fix()
        runner.reckoner.start(3.0)
        fly(runner, north_m=5.0)
        runner._steer_search_with_flow()
        assert runner.engine.guided_to == []

    def test_guidance_stops_at_its_own_limit(self, runner):
        runner.config.visual_odometry.max_search_guidance_s = 2.0
        runner.reckoner.max_search_guidance_s = 2.0
        runner._last_map_fix = map_fix()
        runner.reckoner.start(3.0)
        for _ in range(5):
            fly(runner, north_m=22.0, dt_s=1.0)
        runner._steer_search_with_flow()
        assert runner.engine.guided_to == [], "past the guidance limit, stop guessing"

    def test_guidance_can_be_turned_off(self, runner):
        runner.config.visual_odometry.max_search_guidance_s = 0.0
        runner._last_map_fix = map_fix()
        runner.reckoner.start(3.0)
        fly(runner, north_m=80.0)
        runner._steer_search_with_flow()
        assert runner.engine.guided_to == []

    def test_nothing_happens_without_a_fix_to_reckon_from(self, runner):
        runner.reckoner.start(3.0)
        fly(runner, north_m=80.0)
        runner._steer_search_with_flow()
        assert runner.engine.guided_to == []

    def test_nothing_happens_before_the_origin_is_known(self, runner):
        runner._origin = None
        runner._last_map_fix = map_fix()
        runner.reckoner.start(3.0)
        fly(runner, north_m=80.0)
        runner._steer_search_with_flow()
        assert runner.engine.guided_to == []


class TestWhatGuidanceCannotDo:
    def test_guidance_carries_a_false_fix_faithfully(self, runner, origin):
        """Flow measures motion, not position — so it propagates an error.

        This is why the manual re-anchor is not optional. If the last accepted
        fix was 500 m from the truth, steering keeps the window 500 m from the
        truth and merely moves it along the right track. Only the growing
        radius, or an operator, can escape that.
        """
        wrong = map_fix(north=500.0, east=0.0)
        runner._last_map_fix = wrong
        runner.reckoner.start(3.0)
        fly(runner, north_m=80.0)
        runner._steer_search_with_flow()

        guided = runner.engine.guided_to[-1]
        from_truth = haversine_distance_m(origin.geo, guided)
        assert from_truth == pytest.approx(580.0, rel=0.02), (
            "the 500 m error survives; guidance only adds the 80 m flown"
        )

    def test_guidance_does_not_reset_the_search_radius(self, runner):
        """A grown radius is what eventually escapes a poisoned centre.

        guide_search_center must not behave like reanchor, which resets the
        radius and the failure count because an operator click is new
        information. 'I have moved' is not new information about the ground.
        """
        from avinavi.localization.engine import LocalizationEngine

        config = Config()
        config.map.database_path = "unused"
        engine = LocalizationEngine.__new__(LocalizationEngine)
        engine.config = config
        engine._search_center = GeoPoint(32.0853, 34.7818, 40.0)
        engine._search_radius_m = 1500.0
        engine._consecutive_failures = 9

        engine.guide_search_center(GeoPoint(32.0900, 34.7818, 40.0))

        assert engine._search_radius_m == 1500.0, "the grown radius must survive"
        assert engine._consecutive_failures == 9, "so must the failure count"


class TestTheBudgets:
    def test_the_publishing_budget_is_three_seconds(self):
        assert Config().visual_odometry.max_dead_reckoning_s == 3.0

    def test_three_seconds_is_about_two_and_a_half_attempts(self):
        """Measured ~0.79 localization attempts per second, so it is tight."""
        budget = Config().visual_odometry.max_dead_reckoning_s
        assert 2.0 <= budget * 0.79 <= 3.0

    def test_guidance_outlasts_publication_by_a_wide_margin(self):
        config = Config().visual_odometry
        assert config.max_search_guidance_s > 10 * config.max_dead_reckoning_s

    def test_the_drift_arithmetic_behind_the_split(self):
        """Why one number is fine for aiming and hopeless for navigating."""
        drift_fraction = Config().visual_odometry.drift_fraction
        assert drift_fraction * 66.0 < 4.0, "3 s: metres — publishable"
        assert drift_fraction * 440.0 > 20.0, "20 s: tens of metres — not"
        assert drift_fraction * 440.0 < 100.0, "but far inside a 400 m radius"

    def test_three_seconds_of_flight_at_cruise(self):
        assert math.isclose(3.0 * 22.0, 66.0)
