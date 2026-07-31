"""Concrete camera sources: Picamera2, OpenCV and image-directory replay."""

from __future__ import annotations

import glob
import logging
import os
import time
from typing import List, Optional

import numpy as np

from .base import CameraError, CameraSource, Frame

LOG = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class OpenCVCamera(CameraSource):
    """Any camera or video file OpenCV can open.

    ``device`` is a numeric index for a V4L2 device, or a path/URL for a file or
    stream.  Used on the bench and for replaying recorded flights; on the Pi
    prefer :class:`Picamera2Camera`, which has a much shorter capture path.
    """

    def __init__(
        self,
        device: str = "0",
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
    ) -> None:
        super().__init__()
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._capture = None

    def open(self) -> None:
        try:
            import cv2  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover
            raise CameraError("the opencv camera source needs OpenCV") from exc

        target = int(self.device) if str(self.device).isdigit() else self.device
        capture = cv2.VideoCapture(target)
        if not capture.isOpened():
            raise CameraError("could not open camera %r" % self.device)

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            capture.set(cv2.CAP_PROP_FPS, self.fps)
        # A small buffer keeps latency down: we always want the newest frame.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # pragma: no cover - not all backends support it
            pass

        self._capture = capture
        self._opened = True
        actual = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if actual != (self.width, self.height):
            LOG.warning(
                "camera returned %dx%d, not the requested %dx%d; camera.fx/fy/cx/cy "
                "must match the actual resolution",
                actual[0],
                actual[1],
                self.width,
                self.height,
            )

    def read(self) -> Optional[Frame]:
        if self._capture is None:
            raise CameraError("camera is not open")
        ok, image = self._capture.read()
        if not ok or image is None:
            # A file source simply ends; a live camera failing is a fault.
            if str(self.device).isdigit():
                raise CameraError("camera %r stopped returning frames" % self.device)
            return None
        return self._make_frame(image)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._opened = False


class Picamera2Camera(CameraSource):
    """The Raspberry Pi camera stack (libcamera via Picamera2).

    Configured for a still-image style capture at the requested resolution.
    Frames come back as BGR888 so the rest of the pipeline sees the same layout
    as OpenCV.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        device: str = "0",
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.device = device
        self._camera = None

    def open(self) -> None:
        try:
            from picamera2 import Picamera2  # noqa: PLC0415 - Pi-only dependency
        except ImportError as exc:  # pragma: no cover - depends on hardware
            raise CameraError(
                "picamera2 is not installed; on Raspberry Pi OS: "
                "sudo apt install -y python3-picamera2"
            ) from exc

        try:
            camera_index = int(self.device) if str(self.device).isdigit() else 0
            camera = Picamera2(camera_index)
            configuration = camera.create_video_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"},
                controls={"FrameRate": self.fps} if self.fps > 0 else {},
            )
            camera.configure(configuration)
            camera.start()
        except Exception as exc:  # pragma: no cover - depends on hardware
            raise CameraError("could not start the Pi camera: %s" % exc) from exc

        # Let auto-exposure settle before the first frame reaches SuperPoint.
        time.sleep(0.5)
        self._camera = camera
        self._opened = True

    def read(self) -> Optional[Frame]:
        if self._camera is None:
            raise CameraError("camera is not open")
        try:
            image = self._camera.capture_array()
        except Exception as exc:  # pragma: no cover - depends on hardware
            raise CameraError("Pi camera capture failed: %s" % exc) from exc
        if image is None:
            raise CameraError("Pi camera returned no image")
        return self._make_frame(np.asarray(image))

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop()
                self._camera.close()
            except Exception:  # pragma: no cover
                LOG.debug("error closing the Pi camera", exc_info=True)
            self._camera = None
        self._opened = False


class DirectoryCamera(CameraSource):
    """Replays a folder of images in filename order.

    This is the backbone of Mode 1 (offline): point it at a recorded flight's
    frames and the whole localization pipeline runs with no hardware attached.
    ``replay_speed`` throttles playback to a multiple of real time; set it to 0
    to run as fast as the pipeline allows.
    """

    def __init__(
        self,
        directory: str,
        fps: float = 30.0,
        replay_speed: float = 1.0,
        loop: bool = False,
    ) -> None:
        super().__init__()
        self.directory = directory
        self.fps = float(fps)
        self.replay_speed = float(replay_speed)
        self.loop = bool(loop)
        self._paths: List[str] = []
        self._index = 0
        self._last_read_monotonic: Optional[float] = None

    def open(self) -> None:
        if not os.path.isdir(self.directory):
            raise CameraError("image directory not found: %s" % self.directory)
        paths: List[str] = []
        for extension in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(self.directory, "*" + extension)))
            paths.extend(glob.glob(os.path.join(self.directory, "*" + extension.upper())))
        self._paths = sorted(set(paths))
        if not self._paths:
            raise CameraError("no images in %s" % self.directory)
        self._index = 0
        self._opened = True
        LOG.info("replaying %d images from %s", len(self._paths), self.directory)

    def read(self) -> Optional[Frame]:
        if not self._opened:
            raise CameraError("camera is not open")
        if self._index >= len(self._paths):
            if not self.loop:
                return None
            self._index = 0

        self._throttle()

        try:
            import cv2  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover
            raise CameraError("the directory camera source needs OpenCV") from exc

        path = self._paths[self._index]
        self._index += 1
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            LOG.warning("skipping unreadable image %s", path)
            return self.read()
        frame = self._make_frame(image)
        LOG.debug("frame %d: %s", frame.sequence, os.path.basename(path))
        return frame

    def _throttle(self) -> None:
        """Pace playback so an offline run behaves like a real flight."""
        if self.replay_speed <= 0 or self.fps <= 0:
            return
        interval = 1.0 / (self.fps * self.replay_speed)
        now = time.monotonic()
        if self._last_read_monotonic is not None:
            remaining = interval - (now - self._last_read_monotonic)
            if remaining > 0:
                time.sleep(remaining)
        self._last_read_monotonic = time.monotonic()

    def close(self) -> None:
        self._paths = []
        self._opened = False

    @property
    def frame_count(self) -> int:
        return len(self._paths)
