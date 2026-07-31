"""Pose propagation between visual fixes.

The behaviour that matters most here is the *refusal*: past ``max_age_s`` the
propagator must return ``None`` so the output stage goes quiet, rather than
repeating a stale pose to keep the message rate up.
"""

from __future__ import annotations

import math

import pytest

from avinavi.frames import quaternion_from_euler
from avinavi.localization.propagator import PosePropagator
from avinavi.types import Attitude, InvalidReason, MatchQuality, NEDPosition, VisualFix


def fix(t: float, north: float, east: float, down: float = -50.0, yaw: float = 0.0) -> VisualFix:
    quality = MatchQuality(localization_confidence=0.9)
    return VisualFix(
        timestamp_monotonic=t,
        timestamp_unix=1700000000.0 + t,
        valid=True,
        position_ned=NEDPosition(north, east, down),
        attitude=Attitude(*quaternion_from_euler(0.0, 0.0, yaw)),
        yaw_rad=yaw,
        quality=quality,
        position_sigma_m=1.0,
        yaw_sigma_rad=0.1,
    )


def test_nothing_to_send_before_the_first_fix():
    assert PosePropagator().solution_at(10.0) is None


def test_a_fresh_fix_is_returned_unchanged():
    propagator = PosePropagator()
    propagator.update(fix(100.0, 10.0, 20.0))

    solution = propagator.solution_at(100.0)
    assert solution is not None
    assert solution.position_ned.north_m == pytest.approx(10.0)
    assert solution.propagated is False
    assert solution.age_s == pytest.approx(0.0)


def test_invalid_fixes_are_ignored():
    propagator = PosePropagator()
    invalid = fix(100.0, 10.0, 20.0)
    invalid.valid = False
    invalid.reason = InvalidReason.LOW_CONFIDENCE
    propagator.update(invalid)
    assert propagator.last_fix is None


def test_a_stale_fix_produces_silence():
    """Past max_age_s we must send nothing at all."""
    propagator = PosePropagator(max_age_s=1.0)
    propagator.update(fix(100.0, 10.0, 20.0))

    assert propagator.solution_at(100.9) is not None
    assert propagator.solution_at(101.1) is None


def test_velocity_is_estimated_from_consecutive_fixes():
    propagator = PosePropagator()
    propagator.update(fix(100.0, 0.0, 0.0))
    propagator.update(fix(100.5, 5.0, -2.5))

    velocity = propagator.velocity
    assert velocity is not None
    assert velocity.north_mps == pytest.approx(10.0)
    assert velocity.east_mps == pytest.approx(-5.0)


def test_the_pose_is_propagated_forward_with_velocity():
    propagator = PosePropagator(max_age_s=1.0)
    propagator.update(fix(100.0, 0.0, 0.0))
    propagator.update(fix(100.5, 5.0, 0.0))  # 10 m/s north

    solution = propagator.solution_at(100.7)
    assert solution.position_ned.north_m == pytest.approx(7.0, abs=1e-6)
    assert solution.propagated is True
    assert solution.age_s == pytest.approx(0.2)


def test_uncertainty_grows_with_age():
    propagator = PosePropagator(max_age_s=1.0, sigma_growth_mps=2.0)
    propagator.update(fix(100.0, 0.0, 0.0))

    fresh = propagator.solution_at(100.0).position_sigma_m
    old = propagator.solution_at(100.5).position_sigma_m
    assert old == pytest.approx(fresh + 1.0)


def test_implausible_velocity_is_discarded():
    """A 1 km jump in 100 ms is a mis-localization, not 10 km/s of flight."""
    propagator = PosePropagator(max_speed_mps=40.0)
    propagator.update(fix(100.0, 0.0, 0.0))
    propagator.update(fix(100.1, 1000.0, 0.0))
    assert propagator.velocity is None


def test_velocity_is_not_estimated_across_a_long_gap():
    propagator = PosePropagator(max_velocity_dt_s=2.0)
    propagator.update(fix(100.0, 0.0, 0.0))
    propagator.update(fix(105.0, 10.0, 0.0))
    assert propagator.velocity is None


def test_velocity_can_be_switched_off():
    propagator = PosePropagator(use_velocity=False)
    propagator.update(fix(100.0, 0.0, 0.0))
    propagator.update(fix(100.5, 5.0, 0.0))
    assert propagator.velocity is None

    solution = propagator.solution_at(100.7)
    # Position is held rather than extrapolated, but age is still reported.
    assert solution.position_ned.north_m == pytest.approx(5.0)
    assert solution.age_s == pytest.approx(0.2)


def test_disabling_propagation_holds_the_position():
    propagator = PosePropagator(enabled=False, max_age_s=1.0)
    propagator.update(fix(100.0, 0.0, 0.0))
    propagator.update(fix(100.5, 5.0, 0.0))

    solution = propagator.solution_at(100.9)
    assert solution.position_ned.north_m == pytest.approx(5.0)


def test_velocity_is_reported_in_the_body_frame():
    """Flying north while facing east means moving right, per ODOMETRY's child frame."""
    propagator = PosePropagator()
    propagator.update(fix(100.0, 0.0, 0.0, yaw=math.pi / 2))
    propagator.update(fix(100.5, 5.0, 0.0, yaw=math.pi / 2))  # 10 m/s north

    solution = propagator.solution_at(100.5)
    assert solution.velocity_body_frd == pytest.approx((0.0, -10.0, 0.0), abs=1e-6)


def test_reset_counter_advances_on_a_noted_reset():
    propagator = PosePropagator()
    propagator.update(fix(100.0, 0.0, 0.0))
    assert propagator.solution_at(100.0).reset_counter == 0

    propagator.note_reset()
    assert propagator.solution_at(100.0).reset_counter == 1


def test_reset_counter_wraps_at_a_byte():
    propagator = PosePropagator()
    for _ in range(256):
        propagator.note_reset()
    assert propagator.reset_counter == 0


def test_clear_stops_all_output():
    propagator = PosePropagator()
    propagator.update(fix(100.0, 0.0, 0.0))
    propagator.clear()
    assert propagator.solution_at(100.0) is None


def test_a_timestamp_in_the_past_is_not_negative_age():
    propagator = PosePropagator()
    propagator.update(fix(100.0, 0.0, 0.0))
    solution = propagator.solution_at(99.5)
    assert solution.age_s == 0.0
