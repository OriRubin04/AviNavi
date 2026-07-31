"""The safety gate between the visual solution and the flight controller."""

from __future__ import annotations

import pytest

from avinavi.app.safety import SafetyGate
from avinavi.geo import LocalOrigin
from avinavi.types import GeoPoint, InvalidReason, MatchQuality, NEDPosition, VisualFix

ORIGIN = LocalOrigin(32.0853, 34.7818, 40.0)


def good_fix(north: float = 0.0, east: float = 0.0, t: float = 0.0) -> VisualFix:
    geo = ORIGIN.to_geo(NEDPosition(north, east, -50.0))
    return VisualFix(
        timestamp_monotonic=t,
        timestamp_unix=1700000000.0 + t,
        valid=True,
        geo=geo,
        position_ned=NEDPosition(north, east, -50.0),
        quality=MatchQuality(localization_confidence=0.9),
    )


def bad_fix(t: float = 0.0) -> VisualFix:
    return VisualFix(
        timestamp_monotonic=t,
        timestamp_unix=1700000000.0 + t,
        valid=False,
        reason=InvalidReason.LOW_CONFIDENCE,
    )


def test_transmission_is_withheld_until_the_streak_is_met():
    """A fresh system must prove itself before anything reaches the autopilot."""
    gate = SafetyGate(required_consecutive_good_fixes=3)
    assert not gate.armed

    for index in range(2):
        gate.observe(good_fix(north=index))
        assert not gate.armed
        assert gate.may_transmit(link_ok=True)[0] is False

    gate.observe(good_fix(north=2))
    assert gate.armed
    assert gate.may_transmit(link_ok=True)[0] is True


def test_an_invalid_fix_resets_the_arming_streak():
    gate = SafetyGate(required_consecutive_good_fixes=3)
    gate.observe(good_fix(north=0))
    gate.observe(good_fix(north=1))
    gate.observe(bad_fix())
    assert gate.good_streak == 0
    gate.observe(good_fix(north=2))
    assert not gate.armed


def test_sustained_failures_disarm_a_running_system():
    gate = SafetyGate(required_consecutive_good_fixes=2, max_consecutive_failures=3)
    gate.observe(good_fix(north=0))
    gate.observe(good_fix(north=1))
    assert gate.armed

    for _ in range(3):
        gate.observe(bad_fix())
    assert not gate.armed
    assert gate.may_transmit(link_ok=True)[0] is False


def test_a_single_failure_does_not_disarm():
    """Brief dropouts are normal; the propagator covers them."""
    gate = SafetyGate(required_consecutive_good_fixes=2, max_consecutive_failures=5)
    gate.observe(good_fix(north=0))
    gate.observe(good_fix(north=1))
    gate.observe(bad_fix())
    assert gate.armed


def test_a_large_position_jump_is_rejected_and_disarms():
    """A 500 m jump is a mis-localization, not flight; never fuse it silently."""
    gate = SafetyGate(required_consecutive_good_fixes=2, max_position_jump_m=100.0)
    gate.observe(good_fix(north=0))
    gate.observe(good_fix(north=5))
    assert gate.armed

    accepted, reason = gate.observe(good_fix(north=505))
    assert accepted is False
    assert "jumped" in reason
    assert not gate.armed


def test_a_small_movement_is_accepted():
    gate = SafetyGate(required_consecutive_good_fixes=1, max_position_jump_m=100.0)
    gate.observe(good_fix(north=0))
    accepted, _ = gate.observe(good_fix(north=20))
    assert accepted


def test_re_arming_is_possible_after_a_jump():
    gate = SafetyGate(required_consecutive_good_fixes=2, max_position_jump_m=100.0)
    gate.observe(good_fix(north=0))
    gate.observe(good_fix(north=5))
    gate.observe(good_fix(north=505))  # jump: disarms
    gate.observe(good_fix(north=506))
    gate.observe(good_fix(north=507))
    assert gate.armed


def test_a_dead_link_blocks_transmission_even_when_armed():
    gate = SafetyGate(required_consecutive_good_fixes=1, require_heartbeat=True)
    gate.observe(good_fix())
    assert gate.armed
    permitted, reason = gate.may_transmit(link_ok=False)
    assert permitted is False
    assert "heartbeat" in reason


def test_the_heartbeat_requirement_can_be_waived():
    gate = SafetyGate(required_consecutive_good_fixes=1, require_heartbeat=False)
    gate.observe(good_fix())
    assert gate.may_transmit(link_ok=False)[0] is True


def test_reset_returns_the_gate_to_its_initial_state():
    gate = SafetyGate(required_consecutive_good_fixes=1)
    gate.observe(good_fix())
    assert gate.armed
    gate.reset()
    assert not gate.armed
    assert gate.good_streak == 0


def test_a_fix_without_a_geographic_position_skips_the_jump_check():
    """Jump checking must not crash on a fix that carries no geo position."""
    gate = SafetyGate(required_consecutive_good_fixes=1)
    fix = good_fix()
    fix.geo = None
    accepted, _ = gate.observe(fix)
    assert accepted
