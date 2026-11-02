"""The runner's dead-reckoning path.

This is the one place in the system that **publishes a position nothing matched
this cycle**, so it gets its own tests. The properties that matter are not the
arithmetic — `test_visual_odometry.py` covers that — but the bookkeeping around
it:

- a dead-reckoned position must be distinguishable from a map fix downstream,
- it must be offset from the last *real* map fix, not from a previous guess,
- the uncertainty must grow,
- and it must stop, rather than degrading indefinitely.

The runner is constructed directly and its collaborators replaced, because
``setup()`` wants a map database, a camera and a MAVLink link, none of which
this logic depends on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from avinavi.app.runner import Runner
from avinavi.config import Config
from avinavi.geo import LocalOrigin
from avinavi.localization.visual_odometry import OdometryDelta
from avinavi.types import (
    Attitude,
    InvalidReason,
    MatchQuality,
    NEDPosition,
    VisualFix,
)


class StubEngine:
    """Only the parts of the engine the dead-reckoning path reads."""

    def __init__(self, origin) -> None:
        self.origin = origin
        self.search_radius_m = 400.0
        self.search_center = origin.geo


class StubOdometry:
    """Returns a scripted delta, and records that it was called every cycle."""

    available = True

    def __init__(self, delta=None) -> None:
        self.delta = delta
        self.calls = 0

    def update(self, *_args, **_kwargs):
        self.calls += 1
        return self.delta


class StubFrame:
    def __init__(self, timestamp=0.0) -> None:
        self.image = np.zeros((64, 64), dtype=np.uint8)
        self.timestamp_monotonic = timestamp
        self.timestamp_unix = 1_760_000_000.0 + timestamp
        self.sequence = 1


@pytest.fixture
def origin() -> LocalOrigin:
    return LocalOrigin(32.0853, 34.7818, 40.0)


@pytest.fixture
def runner(origin) -> Runner:
    config = Config()
    config.camera.fx = config.camera.fy = 900.0
    config.camera.cx, config.camera.cy = 640.0, 360.0
    config.visual_odometry.enabled = True
    config.visual_odometry.drift_fraction = 0.05
    config.visual_odometry.max_dead_reckoning_s = 1.0
    runner = Runner(config)
    runner.engine = StubEngine(origin)
    runner.odometry = StubOdometry()
    return runner


def map_fix(north=100.0, east=50.0, down=-80.0, sigma=3.0, timestamp=0.0) -> VisualFix:
    return VisualFix(
        timestamp_monotonic=timestamp,
        timestamp_unix=1_760_000_000.0 + timestamp,
        valid=True,
        reason=InvalidReason.NONE,
        geo=None,
        position_ned=NEDPosition(north, east, down),
        attitude=Attitude(1.0, 0.0, 0.0, 0.0),
        yaw_rad=0.3,
        quality=MatchQuality(number_of_inliers=40, localization_confidence=0.7),
        tile_id="tile_0001",
        position_sigma_m=sigma,
        yaw_sigma_rad=0.1,
    )


def failed_fix(reason=InvalidReason.TOO_FEW_MATCHES, timestamp=0.1) -> VisualFix:
    return VisualFix(
        timestamp_monotonic=timestamp,
        timestamp_unix=1_760_000_000.0 + timestamp,
        valid=False,
        reason=reason,
    )


def delta(north=2.2, east=0.0, dt_s=0.1) -> OdometryDelta:
    return OdometryDelta(
        north_m=north, east_m=east, dt_s=dt_s, tracked_points=120, residual_m=0.2
    )


# ---------------------------------------------------------------------------
# The odometry runs regardless
# ---------------------------------------------------------------------------
def test_the_odometry_runs_on_a_successful_cycle_too(runner):
    """It needs consecutive frames, so skipping it after a good fix would
    leave nothing to track from when the next one fails."""
    runner._apply_dead_reckoning(map_fix(), StubFrame())
    assert runner.odometry.calls == 1


def test_a_successful_map_fix_is_returned_untouched(runner):
    fix = map_fix()
    result = runner._apply_dead_reckoning(fix, StubFrame())
    assert result is fix
    assert result.dead_reckoned is False


def test_a_map_fix_resets_the_drift(runner):
    """Dead reckoning is always relative to the most recent real match."""
    runner.odometry.delta = delta()
    runner._apply_dead_reckoning(map_fix(sigma=3.0), StubFrame(0.0))
    runner._apply_dead_reckoning(failed_fix(timestamp=0.1), StubFrame(0.1))
    assert runner.reckoner.distance_m > 0.0

    runner._apply_dead_reckoning(map_fix(sigma=2.0, timestamp=0.2), StubFrame(0.2))
    assert runner.reckoner.distance_m == 0.0
    assert runner.reckoner.sigma_m == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Carrying the solution forward
# ---------------------------------------------------------------------------
def test_a_failed_match_becomes_a_position_offset_from_the_last_map_fix(runner, origin):
    runner.odometry.delta = delta(north=2.2, east=0.0)
    runner._apply_dead_reckoning(map_fix(north=100.0, east=50.0), StubFrame(0.0))
    result = runner._apply_dead_reckoning(failed_fix(), StubFrame(0.1))

    assert result.valid is True
    assert result.dead_reckoned is True
    assert result.position_ned.north_m == pytest.approx(102.2)
    assert result.position_ned.east_m == pytest.approx(50.0)
    assert result.geo is not None


def test_successive_dropouts_accumulate_from_the_map_fix_not_from_each_other(runner):
    """The offset is cumulative, but always applied to the real fix.

    Compounding guess upon guess would make a single bad delta permanent.
    """
    runner.odometry.delta = delta(north=2.2)
    runner._apply_dead_reckoning(map_fix(north=100.0), StubFrame(0.0))
    for index in range(1, 5):
        result = runner._apply_dead_reckoning(
            failed_fix(timestamp=index * 0.1), StubFrame(index * 0.1)
        )
    assert result.position_ned.north_m == pytest.approx(100.0 + 4 * 2.2)
    assert runner._last_map_fix.position_ned.north_m == pytest.approx(100.0)


def test_the_uncertainty_grows_while_dead_reckoning(runner):
    runner.odometry.delta = delta(north=10.0)
    runner._apply_dead_reckoning(map_fix(sigma=3.0), StubFrame(0.0))
    first = runner._apply_dead_reckoning(failed_fix(timestamp=0.1), StubFrame(0.1))
    second = runner._apply_dead_reckoning(failed_fix(timestamp=0.2), StubFrame(0.2))
    assert first.position_sigma_m > 3.0
    assert second.position_sigma_m > first.position_sigma_m


def test_the_dead_reckoned_fix_reports_how_long_it_has_been_guessing(runner):
    runner.odometry.delta = delta(dt_s=0.1)
    runner._apply_dead_reckoning(map_fix(), StubFrame(0.0))
    result = runner._apply_dead_reckoning(failed_fix(timestamp=0.1), StubFrame(0.1))
    assert result.dead_reckoning_s == pytest.approx(0.1)


def test_yaw_is_not_carried_forward(runner):
    """Dead reckoning does not observe heading.

    Repeating the last fix's yaw would present a stale value as a fresh
    measurement, and the EKF would weight it as one.
    """
    runner.odometry.delta = delta()
    runner._apply_dead_reckoning(map_fix(), StubFrame(0.0))
    result = runner._apply_dead_reckoning(failed_fix(timestamp=0.1), StubFrame(0.1))
    assert result.yaw_rad is None


def test_the_vertical_channel_is_held_at_the_last_map_fix(runner):
    """Frame-to-frame flow measures horizontal motion, not height."""
    runner.odometry.delta = delta()
    runner._apply_dead_reckoning(map_fix(down=-80.0), StubFrame(0.0))
    result = runner._apply_dead_reckoning(failed_fix(timestamp=0.1), StubFrame(0.1))
    assert result.position_ned.down_m == pytest.approx(-80.0)


# ---------------------------------------------------------------------------
# Refusing to guess
# ---------------------------------------------------------------------------
def test_nothing_happens_before_the_first_map_fix(runner):
    """There is no position to carry forward from."""
    runner.odometry.delta = delta()
    result = runner._apply_dead_reckoning(failed_fix(), StubFrame())
    assert result.valid is False
    assert result.dead_reckoned is False


def test_no_odometry_estimate_means_no_position(runner):
    """Over water there is nothing to track; that is not a displacement."""
    runner.odometry.delta = None
    runner._apply_dead_reckoning(map_fix(), StubFrame(0.0))
    result = runner._apply_dead_reckoning(failed_fix(timestamp=0.1), StubFrame(0.1))
    assert result.valid is False


def test_dead_reckoning_stops_at_the_configured_limit(runner):
    """Past the limit it produces nothing, and announces the expiry once.

    The announcement is a single cycle on purpose. Afterwards the honest reason
    is whatever the map matching actually reported, because that is what the
    operator needs to see — reporting ``dead_reckoning_expired`` on every
    subsequent cycle would hide *why* matching is failing.
    """
    runner.odometry.delta = delta(dt_s=0.1)
    runner._apply_dead_reckoning(map_fix(), StubFrame(0.0))

    results = [
        runner._apply_dead_reckoning(
            failed_fix(timestamp=index * 0.1), StubFrame(index * 0.1)
        )
        for index in range(1, 20)
    ]
    carried = [r for r in results if r.valid]
    expired = [r for r in results if r.reason == InvalidReason.DEAD_RECKONING_EXPIRED]

    # 1.0 s at 0.1 s per cycle, so ten cycles are carried and then it stops.
    assert len(carried) == 10
    assert len(expired) == 1, "the expiry should be announced exactly once"
    assert results[-1].valid is False
    assert results[-1].reason == InvalidReason.TOO_FEW_MATCHES
    assert runner.reckoner.active is False


def test_a_later_map_fix_revives_dead_reckoning_after_expiry(runner):
    runner.odometry.delta = delta(dt_s=0.1)
    runner._apply_dead_reckoning(map_fix(), StubFrame(0.0))
    for index in range(1, 20):
        runner._apply_dead_reckoning(
            failed_fix(timestamp=index * 0.1), StubFrame(index * 0.1)
        )
    assert runner.reckoner.active is False

    runner._apply_dead_reckoning(map_fix(timestamp=3.0), StubFrame(3.0))
    result = runner._apply_dead_reckoning(failed_fix(timestamp=3.1), StubFrame(3.1))
    assert result.valid is True
    assert result.dead_reckoned is True


def test_the_feature_can_be_turned_off(runner):
    runner.config.visual_odometry.enabled = False
    runner.odometry.delta = delta()
    runner._apply_dead_reckoning(map_fix(), StubFrame(0.0))
    result = runner._apply_dead_reckoning(failed_fix(timestamp=0.1), StubFrame(0.1))
    assert result.valid is False
    assert runner.odometry.calls == 0


def test_odometry_without_intrinsics_is_skipped(origin):
    """No focal length means a pixel displacement is not a distance."""
    config = Config()
    config.visual_odometry.enabled = True
    runner = Runner(config)   # no camera.fx / fy
    runner.engine = StubEngine(origin)
    assert runner.odometry.available is False
    runner._apply_dead_reckoning(map_fix(), StubFrame(0.0))
    result = runner._apply_dead_reckoning(failed_fix(timestamp=0.1), StubFrame(0.1))
    assert result.valid is False


def test_an_odometry_exception_does_not_break_the_cycle(runner):
    """A fault in an optional aid must not cost the primary path."""
    class Exploding:
        available = True

        def update(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    runner.odometry = Exploding()
    fix = map_fix()
    assert runner._apply_dead_reckoning(fix, StubFrame()) is fix


def test_nothing_is_produced_without_an_origin(runner):
    """A local NED position cannot become a geographic one without it."""
    runner.odometry.delta = delta()
    runner._apply_dead_reckoning(map_fix(), StubFrame(0.0))
    runner.engine.origin = None
    result = runner._apply_dead_reckoning(failed_fix(timestamp=0.1), StubFrame(0.1))
    assert result.valid is False


# ---------------------------------------------------------------------------
# Statistics must stay honest
# ---------------------------------------------------------------------------
def test_dead_reckoned_positions_do_not_count_toward_the_lock_rate(runner):
    """Otherwise a dropout would inherit its predecessor's success.

    The runner records the fix before dead reckoning is applied, so this test
    mirrors the loop's own ordering.
    """
    runner.odometry.delta = delta()

    fix = map_fix()
    runner.monitor.record_fix(fix)
    runner._apply_dead_reckoning(fix, StubFrame(0.0))

    for index in range(1, 5):
        failed = failed_fix(timestamp=index * 0.1)
        runner.monitor.record_fix(failed)
        carried = runner._apply_dead_reckoning(failed, StubFrame(index * 0.1))
        assert carried.valid is True        # published
    assert runner.monitor.valid_fixes.total == 1   # but not counted as a lock
