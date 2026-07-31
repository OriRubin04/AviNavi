"""Build the configured camera source."""

from __future__ import annotations

from .base import CameraError, CameraSource
from .sources import DirectoryCamera, OpenCVCamera, Picamera2Camera


def create_camera(config, loop: bool = False) -> CameraSource:
    """Instantiate the camera named by ``camera.source``."""
    source = (config.source or "").lower()
    if source == "picamera2":
        return Picamera2Camera(
            width=config.width, height=config.height, fps=config.fps, device=config.device
        )
    if source == "opencv":
        return OpenCVCamera(
            device=config.device, width=config.width, height=config.height, fps=config.fps
        )
    if source == "directory":
        return DirectoryCamera(
            directory=config.device,
            fps=config.fps,
            replay_speed=config.replay_speed,
            loop=loop,
        )
    raise CameraError(
        "unknown camera.source %r; expected picamera2, opencv or directory" % config.source
    )
