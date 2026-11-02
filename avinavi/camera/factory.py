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
            device=config.device,
            width=config.width,
            height=config.height,
            fps=config.fps,
            fourcc=config.fourcc,
            lock_exposure=config.lock_exposure,
            max_exposure_s=config.max_exposure_s,
            gain=config.gain,
            drain_to_newest=config.drain_to_newest,
            max_frame_age_s=config.max_frame_age_s,
            warmup_frames=config.warmup_frames,
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
