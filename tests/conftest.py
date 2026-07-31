"""Shared fixtures.

The tests deliberately avoid torch and a real camera: the deep-learning stack is
stubbed out (see :mod:`tests.fakes`) so that the parts most likely to be *wrong*
in a subtle, flight-relevant way - geodesy, frame transforms, the ODOMETRY
encoding, the safety gate - are exercised on every run, on any machine.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avinavi.config import Config  # noqa: E402
from avinavi.geo import BoundingBox, LocalOrigin  # noqa: E402


@pytest.fixture
def origin() -> LocalOrigin:
    """A mid-latitude origin, far enough from the equator to catch scale bugs."""
    return LocalOrigin(lat_deg=32.0853, lon_deg=34.7818, alt_m=40.0)


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def tile_bounds() -> BoundingBox:
    # Roughly 1 km square at the fixture origin's latitude.
    return BoundingBox(west=34.7760, south=32.0800, east=34.7867, north=32.0890)
