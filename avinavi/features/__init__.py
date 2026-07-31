"""Feature extraction, matching and geometric verification.

torch is imported lazily inside :mod:`~avinavi.features.superpoint` and
:mod:`~avinavi.features.lightglue_matcher` so that the configuration, geodesy,
MAVLink and map-database layers stay importable (and testable) on a machine
without the deep-learning stack installed.
"""

from .geometric import GeometricVerifier, GeometricResult

__all__ = ["GeometricVerifier", "GeometricResult"]
