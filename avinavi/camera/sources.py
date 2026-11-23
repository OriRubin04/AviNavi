"""Concrete camera sources: Picamera2, OpenCV and image-directory replay."""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import threading
import time
from typing import List, Optional

import numpy as np

from .base import CameraError, CameraSource, Frame

LOG = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class OpenCVCamera(CameraSource):
    """Any camera or video file OpenCV can open — and the USB flight path.

    ``device`` is a numeric index for a V4L2 device, or a path/URL for a file or
    stream.  This is the source to use with a USB camera on a Pi 5.

    A plain ``VideoCapture`` works on the bench and then disappoints in the air,
    for three reasons that all fail *quietly*:

    **Pixel format.** Most UVC cameras default to uncompressed YUYV.  At
    1280x720x30 that is about 442 Mbit/s, more than USB 2.0 actually carries, so
    the camera quietly serves 5-10 fps instead, or refuses the resolution and
    hands back something smaller — which then silently invalidates the
    calibrated intrinsics.  Requesting MJPG fixes it.  The FourCC must be set
    **before** the resolution: V4L2 validates the frame size against the
    currently selected format, so doing it the other way round can leave the
    camera on a size the new format does not support.

    **Exposure.** Auto-exposure over varying terrain reaches for a long shutter.
    At 22 m/s and 0.13 m/px, 1/60 s smears a feature across ~2.8 px, and
    SuperPoint simply does not detect smeared corners.  Noise from a higher gain
    costs far fewer keypoints than blur does, so the shutter is capped and the
    gain left to pay for it.

    **Buffer staleness.** V4L2 commonly queues four buffers and ignores
    ``CAP_PROP_BUFFERSIZE``.  A 2 Hz localization loop reading from a 30 fps
    camera then gets a frame up to ~0.5 s old, which at 22 m/s is eleven metres
    of error that looks exactly like a mis-georeferenced tile.  So capture runs
    continuously on its own thread and the loop is always handed the newest
    frame, with its true age.
    """

    def __init__(
        self,
        device: str = "0",
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        fourcc: str = "MJPG",
        lock_exposure: bool = True,
        max_exposure_s: float = 0.002,
        gain: Optional[float] = None,
        drain_to_newest: bool = True,
        max_frame_age_s: float = 0.25,
        warmup_frames: int = 5,
    ) -> None:
        super().__init__()
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.fourcc = (fourcc or "").strip()
        self.lock_exposure = bool(lock_exposure)
        self.max_exposure_s = float(max_exposure_s)
        self.gain = gain
        self.drain_to_newest = bool(drain_to_newest)
        self.max_frame_age_s = float(max_frame_age_s)
        self.warmup_frames = int(warmup_frames)

        self._capture = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None          # (image, monotonic)
        self._latest_taken = True
        self._capture_error: Optional[str] = None
        self._stale_frames = 0
        self._dropped_frames = 0

    @property
    def is_live_device(self) -> bool:
        """A numeric device is a real camera; a path is a file or stream."""
        return str(self.device).isdigit()

    # -- opening ------------------------------------------------------------
    def open(self) -> None:
        try:
            import cv2  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover
            raise CameraError("the opencv camera source needs OpenCV") from exc

        target = int(self.device) if self.is_live_device else self.device
        if self.is_live_device:
            # Ask for V4L2 explicitly. The default backend choice varies, and
            # only V4L2 honours the format and exposure controls below.
            capture = cv2.VideoCapture(target, cv2.CAP_V4L2)
            if not capture.isOpened():
                LOG.warning("V4L2 would not open camera %r; trying the default backend", self.device)
                capture = cv2.VideoCapture(target)
        else:
            capture = cv2.VideoCapture(target)
        if not capture.isOpened():
            raise CameraError("could not open camera %r" % self.device)

        if self.is_live_device:
            self._configure_live(cv2, capture)

        self._capture = capture
        self._opened = True
        self._check_resolution(cv2, capture)

        if self.is_live_device:
            self._discard_warmup_frames(capture)
            if self.drain_to_newest:
                self._start_capture_thread()

    def _configure_live(self, cv2, capture) -> None:
        """Apply the format, size and exposure settings, then verify them."""
        # FourCC first: see the class docstring.
        if self.fourcc:
            if len(self.fourcc) != 4:
                LOG.warning("ignoring camera.fourcc %r: a FourCC is 4 characters", self.fourcc)
            else:
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            capture.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # pragma: no cover - not all backends support it
            pass

        if self.fourcc:
            self._verify_fourcc(cv2, capture)
        if self.lock_exposure:
            self._apply_exposure(cv2, capture)
        if self.gain is not None:
            capture.set(cv2.CAP_PROP_GAIN, float(self.gain))

    def _verify_fourcc(self, cv2, capture) -> None:
        """Read the format back: a rejected request is reported, not assumed."""
        raw = int(capture.get(cv2.CAP_PROP_FOURCC))
        actual = "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4)) if raw else ""
        if actual and actual.upper() != self.fourcc.upper():
            LOG.warning(
                "camera is serving %r, not the requested %r. If this is an "
                "uncompressed format the USB bus may not sustain %dx%d at %.0f fps, "
                "and the camera will quietly drop frames instead of saying so. "
                "Check 'v4l2-ctl --list-formats-ext -d %s'.",
                actual,
                self.fourcc,
                self.width,
                self.height,
                self.fps,
                self.device,
            )
        else:
            LOG.info("camera pixel format: %s", actual or self.fourcc)

    def _apply_exposure(self, cv2, capture) -> None:
        """Lock the shutter below :attr:`max_exposure_s`.

        The units here are genuinely camera-dependent: V4L2 UVC drivers express
        ``exposure_time_absolute`` in 100 microsecond steps, while some OpenCV
        backends take seconds, and a few take an opaque index.  So the value is
        written, read back, and any disagreement reported rather than assumed —
        and ``v4l2-ctl`` is preferred when it is available, because it is the
        one path with a documented unit.
        """
        # 1 = V4L2_EXPOSURE_MANUAL, 3 = aperture priority (the usual "auto").
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        hundred_us = max(1, int(round(self.max_exposure_s / 100e-6)))

        if self._apply_exposure_with_v4l2(hundred_us):
            return

        capture.set(cv2.CAP_PROP_EXPOSURE, float(hundred_us))
        readback = capture.get(cv2.CAP_PROP_EXPOSURE)
        # A backend taking seconds would read back something tiny instead.
        if readback and abs(readback - hundred_us) > max(1.0, 0.25 * hundred_us):
            capture.set(cv2.CAP_PROP_EXPOSURE, float(self.max_exposure_s))
            LOG.warning(
                "exposure set to %d (100us units) read back as %.4f; retried in "
                "seconds. Verify the shutter with 'v4l2-ctl -d %s "
                "--get-ctrl=exposure_time_absolute' - an unlocked shutter is the "
                "single most common cause of a poor match rate in flight.",
                hundred_us,
                readback,
                self.device,
            )
        else:
            LOG.info(
                "exposure locked at <= 1/%.0f s (%d x 100us)",
                1.0 / self.max_exposure_s,
                hundred_us,
            )

    def _apply_exposure_with_v4l2(self, hundred_us: int) -> bool:
        """Set the shutter through ``v4l2-ctl``, which has a documented unit."""
        if not self.is_live_device or shutil.which("v4l2-ctl") is None:
            return False
        node = "/dev/video%s" % self.device
        try:
            result = subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    node,
                    "--set-ctrl=auto_exposure=1",
                    "--set-ctrl=exposure_time_absolute=%d" % hundred_us,
                ],
                capture_output=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
            LOG.debug("v4l2-ctl exposure call failed: %s", exc)
            return False
        if result.returncode != 0:
            LOG.debug(
                "v4l2-ctl could not set the exposure (%s); falling back to OpenCV",
                result.stderr.decode("utf-8", "replace").strip(),
            )
            return False
        LOG.info(
            "exposure locked via v4l2-ctl at <= 1/%.0f s (%d x 100us) on %s",
            1.0 / self.max_exposure_s,
            hundred_us,
            node,
        )
        return True

    def _check_resolution(self, cv2, capture) -> None:
        actual = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if actual != (self.width, self.height):
            LOG.warning(
                "camera returned %dx%d, not the requested %dx%d. camera.fx/fy/cx/cy "
                "describe the requested size, so rectification and every metric "
                "scale will be wrong by %.2fx until they are updated or the camera "
                "is made to comply.",
                actual[0],
                actual[1],
                self.width,
                self.height,
                (self.width / actual[0]) if actual[0] else float("nan"),
            )

    def _discard_warmup_frames(self, capture) -> None:
        """Throw away the first few frames while auto-gain settles."""
        for _ in range(max(0, self.warmup_frames)):
            if not capture.grab():
                break

    # -- continuous capture -------------------------------------------------
    def _start_capture_thread(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="camera-capture", daemon=True
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        """Read as fast as the camera allows, keeping only the newest frame.

        Discarding frames here is the point, not a shortcoming: an old frame has
        no value to a localizer that knows the aircraft has moved since.
        """
        while not self._stop.is_set():
            capture = self._capture
            if capture is None:
                return
            ok, image = capture.read()
            if not ok or image is None:
                self._capture_error = "camera %r stopped returning frames" % self.device
                return
            now = time.monotonic()
            with self._lock:
                if not self._latest_taken:
                    self._dropped_frames += 1
                self._latest = (image, now)
                self._latest_taken = False

    # -- reading ------------------------------------------------------------
    def read(self) -> Optional[Frame]:
        if self._capture is None:
            raise CameraError("camera is not open")

        if self._thread is None:
            return self._read_directly()

        if self._capture_error is not None:
            raise CameraError(self._capture_error)

        with self._lock:
            latest = self._latest
            self._latest_taken = True
        if latest is None:
            # Opened, but the first frame has not arrived yet. Not a fault.
            return None

        image, captured_at = latest
        age = time.monotonic() - captured_at
        if age > self.max_frame_age_s:
            # Report it rather than localizing it: at 22 m/s a 0.5 s old frame
            # is 11 m of position error that would look like a tile offset.
            self._stale_frames += 1
            LOG.warning(
                "newest camera frame is %.0f ms old (limit %.0f ms); the capture "
                "thread is not keeping up",
                age * 1000.0,
                self.max_frame_age_s * 1000.0,
            )
        return self._make_frame(image, timestamp_monotonic=captured_at)

    def _read_directly(self) -> Optional[Frame]:
        ok, image = self._capture.read()
        if not ok or image is None:
            # A file source simply ends; a live camera failing is a fault.
            if self.is_live_device:
                raise CameraError("camera %r stopped returning frames" % self.device)
            return None
        return self._make_frame(image)

    @property
    def dropped_frames(self) -> int:
        """Frames captured and superseded before the loop asked for one.

        Expected and healthy: a 30 fps camera feeding a 2 Hz localizer should
        drop most of what it captures.
        """
        return self._dropped_frames

    @property
    def stale_frames(self) -> int:
        return self._stale_frames

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
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
