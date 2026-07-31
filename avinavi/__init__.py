"""AviNavi - GPS-denied visual localization and navigation for ArduPilot.

The package is deliberately layered so that each stage can be replaced without
touching the others:

    avinavi.config        typed configuration loaded from YAML
    avinavi.geo           geodetic <-> local NED <-> map pixel conversions
    avinavi.frames        camera -> body -> navigation frame transforms
    avinavi.camera        frame sources (Picamera2, OpenCV, image directory)
    avinavi.features      SuperPoint, LightGlue, RANSAC/MAGSAC verification
    avinavi.mapdb         georeferenced tile database + offline feature builder
    avinavi.localization  the visual localization engine and pose propagation
    avinavi.mavlink       MAVLink connection, origin acquisition, ODOMETRY out
    avinavi.app           orchestration of the three operating modes
    avinavi.diagnostics   rate/health monitoring and structured logging
"""

__version__ = "0.1.0"
