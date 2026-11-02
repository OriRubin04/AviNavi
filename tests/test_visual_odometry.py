"""Frame-to-frame odometry and dead reckoning.

The displacement tests fly the simulated camera over simulated terrain, because
that is the only way to know the *sign conventions* are right.  A sign error
here is not a crash: it produces a plausible displacement in the wrong
direction, which during a dropout walks the solution away from the aircraft at
twice its true speed.

The rest of the file is about giving up on time.  Dead reckoning is useful for
the seconds it takes to get another map fix and harmful after that, so the
limits matter as much as the arithmetic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from avinavi.config import CameraConfig, CameraExtrinsicsConfig, VisualOdometryConfig
from avinavi.frames import CameraExtrinsics
from avinavi.localization.visual_odometry import (
    DeadReckoner,
    FrameToFrameOdometry,
    OdometryDelta,
)
from avinavi.sim.flight import TrajectoryPoint
from avinavi.sim.render import CameraSimulator, RenderSettings
from avinavi.sim.terrain import TerrainGenerator

HEIGHT_M = 120.0
GSD = 0.5


@pytest.fixture(scope="module")
def terrain():
    return TerrainGenerator(
        width_px=1200, height_px=1200, gsd_mpp=GSD, seed=5
    ).generate()


@pytest.fixture
def world(terrain):
    """A fresh camera over shared terrain.

    Fresh per test because the simulator's RNG advances with every render, so a
    shared one makes these tests order-dependent.

    Exposure drift is off. Lucas-Kanade assumes brightness constancy, and the
    simulator's auto-exposure hunting swings up to 0.45 stops *between
    consecutive frames* - which is realistic for a camera left on auto, and
    exactly what ``camera.lock_exposure`` exists to prevent. Leaving it on here
    would measure the camera configuration rather than the geometry. Its effect
    is measured on purpose in its own test below.
    """
    settings = RenderSettings(
        width_px=960, height_px=540, horizontal_fov_deg=60.0,
        enable_exposure_drift=False,
    )
    extrinsics = CameraExtrinsics.from_config(CameraExtrinsicsConfig())
    simulator = CameraSimulator(
        basemap=terrain, basemap_gsd_mpp=GSD, settings=settings, extrinsics=extrinsics
    )
    return simulator, settings, extrinsics


def odometry(settings, extrinsics, **overrides) -> FrameToFrameOdometry:
    camera = CameraConfig()
    camera.fx, camera.fy = settings.fx, settings.fy
    camera.cx, camera.cy = settings.cx, settings.cy
    config = VisualOdometryConfig()
    config.min_points = 20
    for key, value in overrides.items():
        setattr(config, key, value)
    return FrameToFrameOdometry.from_config(camera, extrinsics, config)


def point(north, east, roll=0.0, pitch=0.0, yaw=0.0, height=HEIGHT_M):
    return TrajectoryPoint(
        time_s=0.0, north_m=north, east_m=east, height_m=height,
        roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw,
        ground_speed_mps=22.0, track_rad=yaw,
    )


def fly(world, first, second, dt_s=0.1, **odometry_kwargs):
    """Run two frames through the odometry and return the delta."""
    simulator, settings, extrinsics = world
    tracker = odometry(settings, extrinsics, **odometry_kwargs)
    for index, state in enumerate((first, second)):
        frame = simulator.render(state)
        assert frame is not None, "the simulated camera could not render this pose"
        delta = tracker.update(
            frame,
            timestamp_monotonic=index * dt_s,
            height_m=state.height_m,
            roll_rad=state.roll_rad,
            pitch_rad=state.pitch_rad,
            yaw_rad=state.yaw_rad,
        )
    return delta


# ---------------------------------------------------------------------------
# Displacement, with the signs checked against known motion
# ---------------------------------------------------------------------------
def test_the_first_frame_yields_nothing(world):
    simulator, settings, extrinsics = world
    tracker = odometry(settings, extrinsics)
    frame = simulator.render(point(0.0, 0.0))
    assert tracker.update(frame, timestamp_monotonic=0.0, height_m=HEIGHT_M) is None


def test_flying_north_is_measured_as_north(world):
    """The sign test that matters most.

    Get this backwards and a dropout walks the solution away from the aircraft
    at twice its true speed, with every individual number looking reasonable.
    """
    delta = fly(world, point(0.0, 0.0), point(2.2, 0.0))
    assert delta is not None
    assert delta.north_m == pytest.approx(2.2, abs=0.25)
    assert delta.east_m == pytest.approx(0.0, abs=0.25)


def test_flying_east_is_measured_as_east(world):
    delta = fly(world, point(0.0, 0.0), point(0.0, 2.2))
    assert delta is not None
    assert delta.east_m == pytest.approx(2.2, abs=0.25)
    assert delta.north_m == pytest.approx(0.0, abs=0.25)


def test_flying_south_west_is_measured_as_negative_on_both_axes(world):
    delta = fly(world, point(0.0, 0.0), point(-1.8, -1.4))
    assert delta is not None
    assert delta.north_m == pytest.approx(-1.8, abs=0.25)
    assert delta.east_m == pytest.approx(-1.4, abs=0.25)


def test_a_stationary_camera_measures_no_motion(world):
    delta = fly(world, point(0.0, 0.0), point(0.0, 0.0))
    assert delta is not None
    assert delta.distance_m == pytest.approx(0.0, abs=0.1)


def test_the_speed_follows_from_the_timestamps(world):
    delta = fly(world, point(0.0, 0.0), point(2.2, 0.0), dt_s=0.1)
    assert delta is not None
    assert delta.speed_mps == pytest.approx(22.0, rel=0.15)


def test_a_sustained_bank_does_not_become_translation(world):
    """The case that matters: a fixed wing in a turn.

    Each frame is projected with its own attitude, so a steady 25 degrees of
    bank must produce no displacement at all. If attitude leaked into the
    estimate, this would be useless exactly when it is most needed.
    """
    delta = fly(
        world,
        point(0.0, 0.0, roll=math.radians(25.0)),
        point(0.0, 0.0, roll=math.radians(25.0)),
    )
    assert delta is not None
    assert delta.distance_m == pytest.approx(0.0, abs=1.0)


def test_a_bank_change_between_frames_does_not_become_translation(world):
    """3 degrees in 0.1 s is a 30 deg/s roll rate - a brisk fixed-wing input."""
    delta = fly(
        world,
        point(0.0, 0.0, roll=math.radians(22.0)),
        point(0.0, 0.0, roll=math.radians(25.0)),
    )
    assert delta is not None
    assert delta.distance_m == pytest.approx(0.0, abs=1.5)


def test_translation_is_recovered_while_rolling(world):
    delta = fly(
        world,
        point(0.0, 0.0, roll=math.radians(-10.0)),
        point(2.2, 0.0, roll=math.radians(-7.0)),
    )
    assert delta is not None
    assert delta.north_m == pytest.approx(2.2, abs=0.6)
    assert delta.east_m == pytest.approx(0.0, abs=0.6)


def test_a_heading_change_does_not_become_translation(world):
    """Both frames are levelled north-up, so a turn must not displace."""
    delta = fly(
        world,
        point(0.0, 0.0, yaw=math.radians(0.0)),
        point(0.0, 0.0, yaw=math.radians(10.0)),
    )
    assert delta is not None
    assert delta.distance_m == pytest.approx(0.0, abs=1.5)


def test_a_climb_between_frames_does_not_become_translation(world):
    delta = fly(world, point(0.0, 0.0, height=120.0), point(0.0, 0.0, height=128.0))
    assert delta is not None
    assert delta.distance_m == pytest.approx(0.0, abs=1.0)


def test_auto_exposure_hunting_costs_the_odometry_its_tracking(terrain):
    """Why ``camera.lock_exposure`` is on by default, measured rather than asserted.

    Lucas-Kanade assumes a feature's brightness is the same in both frames.
    Auto-exposure swinging between consecutive frames breaks that assumption
    directly, and the failure is not a wrong answer - it is no answer, which is
    the better of the two outcomes but still a dropout.
    """
    extrinsics = CameraExtrinsics.from_config(CameraExtrinsicsConfig())
    results = {}
    for label, drift in (("locked", False), ("hunting", True)):
        settings = RenderSettings(
            width_px=960, height_px=540, horizontal_fov_deg=60.0,
            enable_exposure_drift=drift,
        )
        simulator = CameraSimulator(
            basemap=terrain, basemap_gsd_mpp=GSD, settings=settings,
            extrinsics=extrinsics,
        )
        tracker = odometry(settings, extrinsics)
        succeeded = 0
        for index in range(8):
            frame = simulator.render(point(index * 2.2, 0.0))
            assert frame is not None
            delta = tracker.update(frame, index * 0.1, HEIGHT_M)
            if delta is not None and delta.tracked_points >= 20:
                succeeded += 1
        results[label] = succeeded

    assert results["locked"] >= 6, "locked exposure should track almost every frame"
    assert results["hunting"] < results["locked"], (
        "exposure drift should cost tracking; it did not, so either the "
        "simulator's drift is too weak to be a fair test or this check is "
        "measuring nothing (locked=%d, hunting=%d)" % (results["locked"], results["hunting"])
    )


def test_the_estimate_reports_how_many_points_agreed(world):
    delta = fly(world, point(0.0, 0.0), point(2.2, 0.0))
    assert delta is not None
    assert delta.tracked_points >= 20
    assert delta.residual_m < 2.0


# ---------------------------------------------------------------------------
# Refusing to guess
# ---------------------------------------------------------------------------
def test_odometry_is_unavailable_without_intrinsics(world):
    _simulator, _settings, extrinsics = world
    tracker = FrameToFrameOdometry.from_config(CameraConfig(), extrinsics)
    assert tracker.available is False


def test_nothing_is_produced_without_a_height(world):
    """A pixel displacement is not a distance until the height is known."""
    simulator, settings, extrinsics = world
    tracker = odometry(settings, extrinsics)
    frame = simulator.render(point(0.0, 0.0))
    assert tracker.update(frame, timestamp_monotonic=0.0, height_m=None) is None
    assert tracker.update(frame, timestamp_monotonic=0.1, height_m=0.0) is None


def test_featureless_ground_produces_nothing_rather_than_zero(world):
    """Over water there is nothing to track, which is not the same as not moving.

    Returning a zero displacement here would be a confident claim that the
    aircraft had stopped.
    """
    _simulator, settings, extrinsics = world
    tracker = odometry(settings, extrinsics)
    blank = np.full((540, 960), 128, dtype=np.uint8)
    assert tracker.update(blank, 0.0, HEIGHT_M) is None
    assert tracker.update(blank, 0.1, HEIGHT_M) is None


def test_an_impossible_speed_is_rejected(world):
    """A fixed wing does not teleport; a flow field saying so has failed."""
    delta = fly(world, point(0.0, 0.0), point(40.0, 0.0), dt_s=0.1,
                min_points=20)
    # 400 m/s. Either tracking fails outright or the speed check rejects it;
    # what must not happen is a confident 400 m/s displacement.
    assert delta is None or delta.speed_mps <= 60.0


def test_a_non_advancing_timestamp_is_rejected(world):
    simulator, settings, extrinsics = world
    tracker = odometry(settings, extrinsics)
    first = simulator.render(point(0.0, 0.0))
    second = simulator.render(point(2.2, 0.0))
    tracker.update(first, 1.0, HEIGHT_M)
    assert tracker.update(second, 1.0, HEIGHT_M) is None


def test_a_failed_estimate_still_leaves_a_basis_for_the_next_cycle(world):
    """One bad frame must not cost the following one too."""
    simulator, settings, extrinsics = world
    tracker = odometry(settings, extrinsics)
    tracker.update(simulator.render(point(0.0, 0.0)), 0.0, HEIGHT_M)
    # A featureless frame in the middle: no estimate, but it becomes the basis.
    assert tracker.update(np.full((540, 960), 128, dtype=np.uint8), 0.1, HEIGHT_M) is None
    assert tracker._previous_gray is not None


def test_reset_forgets_the_previous_frame(world):
    simulator, settings, extrinsics = world
    tracker = odometry(settings, extrinsics)
    tracker.update(simulator.render(point(0.0, 0.0)), 0.0, HEIGHT_M)
    tracker.reset()
    assert tracker.update(simulator.render(point(2.2, 0.0)), 0.1, HEIGHT_M) is None


# ---------------------------------------------------------------------------
# Dead reckoning
# ---------------------------------------------------------------------------
def delta(north=2.2, east=0.0, dt_s=0.1) -> OdometryDelta:
    return OdometryDelta(
        north_m=north, east_m=east, dt_s=dt_s, tracked_points=120, residual_m=0.2
    )


def test_nothing_accumulates_before_a_map_fix_starts_it():
    """Dead reckoning is always relative to a real fix."""
    reckoner = DeadReckoner()
    assert reckoner.active is False
    assert reckoner.accumulate(delta()) is False
    assert reckoner.offset_ned_m == (0.0, 0.0)


def test_deltas_accumulate_onto_the_starting_fix():
    reckoner = DeadReckoner()
    reckoner.start(base_sigma_m=3.0)
    for _ in range(5):
        assert reckoner.accumulate(delta(north=2.2)) is True
    north, east = reckoner.offset_ned_m
    assert north == pytest.approx(11.0)
    assert east == pytest.approx(0.0)
    assert reckoner.elapsed_s == pytest.approx(0.5)


def test_the_uncertainty_grows_with_distance_flown():
    reckoner = DeadReckoner(drift_fraction=0.05)
    reckoner.start(base_sigma_m=3.0)
    assert reckoner.sigma_m == pytest.approx(3.0)
    for _ in range(10):
        reckoner.accumulate(delta(north=10.0))
    # 100 m flown at 5% drift is 5 m, in quadrature with the fix's own 3 m.
    assert reckoner.sigma_m == pytest.approx(math.hypot(3.0, 5.0), rel=1e-6)


def test_the_uncertainty_never_shrinks_below_the_starting_fix():
    reckoner = DeadReckoner()
    reckoner.start(base_sigma_m=4.0)
    reckoner.accumulate(delta())
    assert reckoner.sigma_m >= 4.0


def test_reckoning_stops_at_the_time_limit():
    """Past this it is actively harmful, so it produces nothing."""
    reckoner = DeadReckoner(max_dead_reckoning_s=1.0)
    reckoner.start(base_sigma_m=2.0)
    accepted = 0
    for _ in range(30):
        if reckoner.accumulate(delta(dt_s=0.1)):
            accepted += 1
    assert accepted == 10
    assert reckoner.active is False


def test_a_new_fix_restarts_the_reckoning_from_zero():
    reckoner = DeadReckoner()
    reckoner.start(base_sigma_m=3.0)
    reckoner.accumulate(delta(north=10.0))
    reckoner.start(base_sigma_m=1.0)
    assert reckoner.offset_ned_m == (0.0, 0.0)
    assert reckoner.distance_m == 0.0
    assert reckoner.sigma_m == pytest.approx(1.0)


def test_clearing_stops_it_without_a_new_fix():
    reckoner = DeadReckoner()
    reckoner.start(base_sigma_m=3.0)
    reckoner.clear()
    assert reckoner.active is False
    assert reckoner.accumulate(delta()) is False


def test_the_configured_limits_are_used():
    config = VisualOdometryConfig()
    config.drift_fraction = 0.1
    config.max_dead_reckoning_s = 4.0
    reckoner = DeadReckoner.from_config(config)
    assert reckoner.drift_fraction == 0.1
    assert reckoner.max_dead_reckoning_s == 4.0
