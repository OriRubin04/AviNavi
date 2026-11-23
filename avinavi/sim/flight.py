"""Fixed-wing trajectory generation.

A plane is not a multirotor, and the differences all matter to a vision-based
localizer:

* **It banks to turn.** A coordinated turn at bank angle ``phi`` produces a turn
  rate ``g·tan(phi)/V``.  At 22 m/s and 30 degrees of bank that is about 15
  degrees per second, so a 180-degree reversal takes ~12 seconds during which
  the camera is looking sideways-down at the ground, rotating steadily.  This is
  the hardest case for the pipeline and the simulator must produce it.
* **It cannot stop.** There is no hover to re-acquire from; whatever the lock
  rate is, the aircraft is covering 20-plus metres a second regardless.
* **It crabs in wind.** Heading and ground track differ by the crab angle, so
  the yaw the camera sees is *not* the direction of travel.  A localizer that
  conflates the two drifts.
* **It pitches with airspeed and climb.** Modest, but non-zero.

The generator produces a waypoint circuit flown with a simple coordinated-turn
model, which is enough to exercise all of the above without pretending to be a
flight dynamics model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

GRAVITY = 9.80665


@dataclass
class TrajectoryPoint:
    """The aircraft state at one instant."""

    time_s: float
    north_m: float
    east_m: float
    #: Height above the terrain, metres.
    height_m: float
    roll_rad: float
    pitch_rad: float
    #: Heading: where the nose points, not where the aircraft is going.
    yaw_rad: float
    #: Ground speed, m/s.
    ground_speed_mps: float
    #: Ground track: the direction of travel.
    track_rad: float

    @property
    def velocity_ned(self) -> Tuple[float, float, float]:
        return (
            self.ground_speed_mps * math.cos(self.track_rad),
            self.ground_speed_mps * math.sin(self.track_rad),
            0.0,
        )


@dataclass
class FixedWingTrajectory:
    """A coordinated-turn trajectory generator.

    Defaults describe a small survey aircraft: 22 m/s cruise, 120 m AGL, 30
    degrees maximum bank.
    """

    waypoints: Sequence[Tuple[float, float]]
    airspeed_mps: float = 22.0
    cruise_height_m: float = 120.0
    max_bank_deg: float = 30.0
    #: Wind, as (north, east) m/s.  Produces the crab angle.
    wind_ned: Tuple[float, float] = (0.0, 0.0)
    #: Peak-to-peak altitude wander, metres: thermals, altitude-hold error.
    height_variation_m: float = 15.0
    #: Pitch oscillation amplitude, degrees.
    pitch_amplitude_deg: float = 4.0
    dt_s: float = 0.2
    #: Metres from a waypoint at which it counts as reached.
    capture_radius_m: float = 40.0
    seed: int = 11
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError("need at least two waypoints")
        self._rng = np.random.default_rng(self.seed)

    def fly(self, max_time_s: float = 240.0) -> List[TrajectoryPoint]:
        """Fly the circuit, returning the state at every ``dt_s``."""
        north, east = float(self.waypoints[0][0]), float(self.waypoints[0][1])
        target_index = 1
        # Start pointing at the first leg.
        heading = math.atan2(
            self.waypoints[1][1] - east, self.waypoints[1][0] - north
        )
        max_bank = math.radians(self.max_bank_deg)
        points: List[TrajectoryPoint] = []
        time_s = 0.0
        # Low-frequency height wander, deterministic from the seed.
        height_phase = self._rng.uniform(0, 2 * math.pi)
        height_period = self._rng.uniform(25.0, 60.0)
        pitch_phase = self._rng.uniform(0, 2 * math.pi)

        while time_s < max_time_s:
            target = self.waypoints[target_index % len(self.waypoints)]
            to_target = (target[0] - north, target[1] - east)
            distance = math.hypot(*to_target)
            if distance < self.capture_radius_m:
                target_index += 1
                continue

            desired_heading = math.atan2(to_target[1], to_target[0])
            error = _wrap_pi(desired_heading - heading)

            # Proportional roll demand, saturated at the structural/comfort limit.
            # The 1.6 gain gives a turn that settles without overshooting badly.
            roll = max(-max_bank, min(max_bank, 1.6 * error))
            # Coordinated turn: omega = g tan(phi) / V
            turn_rate = GRAVITY * math.tan(roll) / max(self.airspeed_mps, 1.0)
            heading = _wrap_pi(heading + turn_rate * self.dt_s)

            # Air velocity plus wind gives the ground track and speed, which is
            # where the crab angle comes from.
            air_n = self.airspeed_mps * math.cos(heading)
            air_e = self.airspeed_mps * math.sin(heading)
            ground_n = air_n + self.wind_ned[0]
            ground_e = air_e + self.wind_ned[1]
            ground_speed = math.hypot(ground_n, ground_e)
            track = math.atan2(ground_e, ground_n)

            north += ground_n * self.dt_s
            east += ground_e * self.dt_s

            height = self.cruise_height_m + 0.5 * self.height_variation_m * math.sin(
                2 * math.pi * time_s / height_period + height_phase
            )
            pitch = math.radians(self.pitch_amplitude_deg) * math.sin(
                2 * math.pi * time_s / 13.0 + pitch_phase
            )

            points.append(
                TrajectoryPoint(
                    time_s=time_s,
                    north_m=north,
                    east_m=east,
                    height_m=height,
                    roll_rad=roll,
                    pitch_rad=pitch,
                    yaw_rad=heading,
                    ground_speed_mps=ground_speed,
                    track_rad=track,
                )
            )
            time_s += self.dt_s

        return points


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def fly_circuit(
    extent_m: float = 900.0,
    airspeed_mps: float = 22.0,
    cruise_height_m: float = 120.0,
    max_bank_deg: float = 30.0,
    wind_ned: Tuple[float, float] = (0.0, 0.0),
    duration_s: float = 240.0,
    dt_s: float = 0.2,
    seed: int = 11,
    pattern: str = "racetrack",
) -> List[TrajectoryPoint]:
    """Fly a standard test pattern centred on the local origin.

    ``racetrack`` is two long straight legs joined by 180-degree turns, which is
    what a survey aircraft actually flies and which deliberately spends a good
    fraction of the time in a steep turn.  ``figure8`` alternates turn direction
    so both bank signs are exercised.  ``straight`` is the easy case, useful as a
    control.
    """
    half = extent_m / 2.0
    if pattern == "racetrack":
        waypoints = [(-half, -half * 0.5), (half, -half * 0.5), (half, half * 0.5), (-half, half * 0.5)]
    elif pattern == "figure8":
        waypoints = [
            (-half, -half * 0.6),
            (half * 0.5, half * 0.6),
            (half, -half * 0.6),
            (-half * 0.5, half * 0.6),
        ]
    elif pattern == "straight":
        waypoints = [(-half, 0.0), (half, 0.0)]
    else:
        raise ValueError("unknown pattern %r" % pattern)

    return FixedWingTrajectory(
        waypoints=waypoints,
        airspeed_mps=airspeed_mps,
        cruise_height_m=cruise_height_m,
        max_bank_deg=max_bank_deg,
        wind_ned=wind_ned,
        dt_s=dt_s,
        seed=seed,
    ).fly(max_time_s=duration_s)


def trajectory_statistics(points: Sequence[TrajectoryPoint]) -> dict:
    """Summarise a trajectory, so a simulation report can describe its own input."""
    if not points:
        return {}
    rolls = np.array([abs(math.degrees(p.roll_rad)) for p in points])
    heights = np.array([p.height_m for p in points])
    speeds = np.array([p.ground_speed_mps for p in points])
    crabs = np.array([abs(math.degrees(_wrap_pi(p.track_rad - p.yaw_rad))) for p in points])
    return {
        "duration_s": round(points[-1].time_s, 1),
        "samples": len(points),
        "bank_deg": {
            "mean": round(float(rolls.mean()), 1),
            "p95": round(float(np.percentile(rolls, 95)), 1),
            "max": round(float(rolls.max()), 1),
            "fraction_over_15deg": round(float((rolls > 15.0).mean()), 3),
        },
        "height_m": {
            "mean": round(float(heights.mean()), 1),
            "min": round(float(heights.min()), 1),
            "max": round(float(heights.max()), 1),
        },
        "ground_speed_mps": {
            "mean": round(float(speeds.mean()), 1),
            "min": round(float(speeds.min()), 1),
            "max": round(float(speeds.max()), 1),
        },
        "crab_angle_deg": {
            "mean": round(float(crabs.mean()), 1),
            "max": round(float(crabs.max()), 1),
        },
    }
