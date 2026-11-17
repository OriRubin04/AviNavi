"""A fixed-wing flight simulator for measuring localization lock rate.

The point of this package is to turn "the lock rate should be high" into a
number.  It generates a synthetic basemap, flies a fixed-wing trajectory over
it, renders what a USB camera would actually see - including the appearance gap
against the basemap, motion blur, bank angles and exposure drift - and then runs
the real localization pipeline over those frames and reports how often it locked
and how far off it was.

That makes pipeline changes measurable instead of arguable: every claim about
lock rate in this repository comes from ``avinavi simulate``.

Deliberate limitation, stated up front: synthetic imagery is kinder than real
imagery.  Procedural texture has no seasonal change, no shadows that move, no
new buildings, no clouds and no sensor-specific colour response.  Absolute lock
rates from this harness are therefore optimistic. What it measures reliably is
the *relative* effect of a change, and whether the geometry is right - which is
what the pipeline is designed and tested around.
"""

from .terrain import TerrainGenerator, generate_basemap
from .flight import FixedWingTrajectory, TrajectoryPoint, fly_circuit
from .render import CameraSimulator, RenderSettings
from .evaluate import SimulationReport, run_simulation

__all__ = [
    "TerrainGenerator",
    "generate_basemap",
    "FixedWingTrajectory",
    "TrajectoryPoint",
    "fly_circuit",
    "CameraSimulator",
    "RenderSettings",
    "SimulationReport",
    "run_simulation",
]
