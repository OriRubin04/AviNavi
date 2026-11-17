"""USB (V4L2) camera hardening.

A plain ``VideoCapture`` opens fine, returns frames, and looks healthy on a
bench — then matches poorly in the air.  The three causes are pixel format,
exposure and buffer staleness, and none of them raises anything.  These tests
pin the behaviour without hardware, by standing a recording double in for
``cv2.VideoCapture``, so the ordering and staleness logic is covered on CI
rather than only on a flight line.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from avinavi.camera.base import CameraError
from avinavi.camera.sources import OpenCVCamera


class FakeCapture:
    """Records every ``set`` in order and serves numbered frames."""

    def __init__(self, width=1280, height=720, fourcc="MJPG", fail_after=None,
                 fixed_size=False):
        #: A camera that will not change resolution, as one serving an
        #: uncompressed format at a size the bus cannot carry behaves.
        self.fixed_size = fixed_size
        self.calls = []
        self.props = {
            cv2.CAP_PROP_FRAME_WIDTH: float(width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(height),
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*fourcc)),
        }
        self.frames_read = 0
        self.grabs = 0
        self.released = False
        self.fail_after = fail_after
        self._opened = True

    def isOpened(self):  # noqa: N802 - OpenCV's spelling
        return self._opened

    def set(self, prop, value):
        self.calls.append((prop, value))
        # A real camera reports back what it actually accepted.
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            if not self.fixed_size:
                self.props[prop] = float(value)
        if prop == cv2.CAP_PROP_EXPOSURE:
            self.props[prop] = float(value)
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def grab(self):
        self.grabs += 1
        return True

    def read(self):
        if self.fail_after is not None and self.frames_read >= self.fail_after:
            return (False, None)
        self.frames_read += 1
        # Encode the frame number in the pixels so a test can tell them apart.
        image = np.full((8, 8, 3), self.frames_read % 256, dtype=np.uint8)
        time.sleep(0.002)
        return (True, image)

    def release(self):
        self.released = True
        self._opened = False


class CaptureRecorder:
    """Collects the FakeCaptures created, and configures the next one."""

    def __init__(self) -> None:
        self.made = []
        self.kwargs = {}

    def configure(self, **kwargs) -> None:
        """Set how the *next* capture should behave."""
        self.kwargs.update(kwargs)

    def make(self, *_args, **_kwargs) -> FakeCapture:
        capture = FakeCapture(**self.kwargs)
        self.made.append(capture)
        return capture

    def __getitem__(self, index) -> FakeCapture:
        return self.made[index]


@pytest.fixture
def captures(monkeypatch):
    """Patch VideoCapture and collect the instances that get created."""
    recorder = CaptureRecorder()
    monkeypatch.setattr(cv2, "VideoCapture", recorder.make)
    # Keep v4l2-ctl out of it so the OpenCV path is what gets exercised.
    monkeypatch.setattr("avinavi.camera.sources.shutil.which", lambda _name: None)
    return recorder


def set_values(capture, prop):
    return [value for name, value in capture.calls if name == prop]


# ---------------------------------------------------------------------------
# Pixel format
# ---------------------------------------------------------------------------
def test_the_pixel_format_is_requested_before_the_resolution(captures):
    """V4L2 validates the frame size against the *current* format.

    Setting the size first and the format second can leave the camera on a size
    the new format does not support, which it resolves by quietly serving
    something else.
    """
    camera = OpenCVCamera(device="0")
    camera.open()
    try:
        order = [name for name, _ in captures[0].calls]
        assert order.index(cv2.CAP_PROP_FOURCC) < order.index(cv2.CAP_PROP_FRAME_WIDTH)
        assert order.index(cv2.CAP_PROP_FOURCC) < order.index(cv2.CAP_PROP_FRAME_HEIGHT)
    finally:
        camera.close()


def test_mjpeg_is_requested_by_default(captures):
    camera = OpenCVCamera(device="0")
    camera.open()
    try:
        expected = float(cv2.VideoWriter_fourcc(*"MJPG"))
        assert set_values(captures[0], cv2.CAP_PROP_FOURCC) == [expected]
    finally:
        camera.close()


def test_an_uncompressed_format_the_camera_substitutes_is_reported(captures, caplog):
    """Silence here is the failure: the camera works, just not fast enough."""
    captures.configure(fourcc="YUYV")
    camera = OpenCVCamera(device="0", fourcc="MJPG")
    with caplog.at_level("WARNING"):
        camera.open()
    try:
        assert "YUYV" in caplog.text
        assert "v4l2-ctl" in caplog.text
    finally:
        camera.close()


def test_an_invalid_fourcc_is_refused_rather_than_sent(captures, caplog):
    camera = OpenCVCamera(device="0", fourcc="MJP")
    with caplog.at_level("WARNING"):
        camera.open()
    try:
        assert set_values(captures[0], cv2.CAP_PROP_FOURCC) == []
        assert "4 characters" in caplog.text
    finally:
        camera.close()


def test_an_empty_fourcc_leaves_the_driver_default(captures):
    camera = OpenCVCamera(device="0", fourcc="")
    camera.open()
    try:
        assert set_values(captures[0], cv2.CAP_PROP_FOURCC) == []
    finally:
        camera.close()


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------
def test_the_exposure_is_switched_to_manual_before_being_set(captures):
    camera = OpenCVCamera(device="0", max_exposure_s=0.002)
    camera.open()
    try:
        order = [name for name, _ in captures[0].calls]
        assert order.index(cv2.CAP_PROP_AUTO_EXPOSURE) < order.index(cv2.CAP_PROP_EXPOSURE)
        # 1 is V4L2_EXPOSURE_MANUAL; 3 is the usual "auto".
        assert set_values(captures[0], cv2.CAP_PROP_AUTO_EXPOSURE) == [1]
    finally:
        camera.close()


def test_the_shutter_is_sent_in_v4l2_hundred_microsecond_units(captures):
    """1/500 s is 20 units of 100 us, not 0.002."""
    camera = OpenCVCamera(device="0", max_exposure_s=0.002)
    camera.open()
    try:
        assert set_values(captures[0], cv2.CAP_PROP_EXPOSURE)[0] == pytest.approx(20.0)
    finally:
        camera.close()


def test_a_backend_that_wants_seconds_is_detected_and_retried(captures, caplog):
    """The units are genuinely camera-dependent, so the value is read back.

    A backend taking seconds reads back something far from the number written;
    rather than assume, retry in seconds and say so.
    """
    class SecondsCapture(FakeCapture):
        def set(self, prop, value):
            super().set(prop, value)
            if prop == cv2.CAP_PROP_EXPOSURE:
                # Whatever we write, it reports a seconds-like value.
                self.props[prop] = 0.002
            return True

    made = []
    import avinavi.camera.sources as sources

    def factory(*args, **kwargs):
        capture = SecondsCapture()
        made.append(capture)
        return capture

    original = cv2.VideoCapture
    cv2.VideoCapture = factory
    try:
        camera = OpenCVCamera(device="0", max_exposure_s=0.002)
        with caplog.at_level("WARNING"):
            camera.open()
        camera.close()
    finally:
        cv2.VideoCapture = original

    written = set_values(made[0], cv2.CAP_PROP_EXPOSURE)
    assert written[0] == pytest.approx(20.0)
    assert written[-1] == pytest.approx(0.002)
    assert "exposure" in caplog.text.lower()


def test_the_exposure_is_left_alone_when_locking_is_disabled(captures):
    camera = OpenCVCamera(device="0", lock_exposure=False)
    camera.open()
    try:
        assert set_values(captures[0], cv2.CAP_PROP_EXPOSURE) == []
        assert set_values(captures[0], cv2.CAP_PROP_AUTO_EXPOSURE) == []
    finally:
        camera.close()


def test_a_configured_gain_is_applied(captures):
    camera = OpenCVCamera(device="0", gain=8.0)
    camera.open()
    try:
        assert set_values(captures[0], cv2.CAP_PROP_GAIN) == [8.0]
    finally:
        camera.close()


def test_gain_is_left_to_the_camera_by_default(captures):
    camera = OpenCVCamera(device="0")
    camera.open()
    try:
        assert set_values(captures[0], cv2.CAP_PROP_GAIN) == []
    finally:
        camera.close()


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def test_a_substituted_resolution_is_reported_with_its_consequence(captures, caplog):
    """The intrinsics describe the requested size, so this is not cosmetic."""
    captures.configure(width=640, height=480, fixed_size=True)
    camera = OpenCVCamera(device="0", width=1280, height=720)
    with caplog.at_level("WARNING"):
        camera.open()
    try:
        assert "640x480" in caplog.text
        assert "fx" in caplog.text
    finally:
        camera.close()


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------
def test_warmup_frames_are_discarded_while_auto_gain_settles(captures):
    camera = OpenCVCamera(device="0", warmup_frames=5, drain_to_newest=False)
    camera.open()
    try:
        assert captures[0].grabs == 5
    finally:
        camera.close()


def test_the_newest_frame_is_served_and_older_ones_are_dropped(captures):
    """Discarding is the point: an old frame has no value to a localizer.

    A 30 fps camera feeding a 2 Hz loop should drop most of what it captures.
    """
    camera = OpenCVCamera(device="0", drain_to_newest=True, warmup_frames=0)
    camera.open()
    try:
        time.sleep(0.2)
        first = camera.read()
        assert first is not None
        captured_so_far = captures[0].frames_read
        # The frame handed over must be among the most recent, not the first.
        assert int(first.image[0, 0, 0]) > 1
        assert camera.dropped_frames > 0
        assert camera.dropped_frames < captured_so_far
    finally:
        camera.close()


def test_a_frame_older_than_the_limit_is_reported(captures, caplog):
    """At 22 m/s, 0.5 s of staleness is 11 m that looks like a tile offset."""
    camera = OpenCVCamera(device="0", drain_to_newest=True, warmup_frames=0,
                          max_frame_age_s=0.0)
    camera.open()
    try:
        time.sleep(0.1)
        with caplog.at_level("WARNING"):
            frame = camera.read()
        assert frame is not None
        assert camera.stale_frames >= 1
        assert "old" in caplog.text
    finally:
        camera.close()


def test_the_frame_timestamp_is_the_capture_time_not_the_read_time(captures):
    """Pose propagation and the MAVLink timestamp both depend on this.

    Timestamping at read() would hide the capture latency the whole drain
    mechanism exists to control.
    """
    camera = OpenCVCamera(device="0", drain_to_newest=True, warmup_frames=0,
                          max_frame_age_s=10.0)
    camera.open()
    try:
        time.sleep(0.15)
        before_read = time.monotonic()
        frame = camera.read()
        assert frame is not None
        assert frame.timestamp_monotonic < before_read
    finally:
        camera.close()


def test_no_frame_yet_is_not_an_error(captures):
    """Opened but the first frame has not arrived: normal, not a fault.

    Exercises the drain branch with an empty slot, which is the state between
    ``open()`` and the capture thread's first successful read.
    """
    camera = OpenCVCamera(device="0", drain_to_newest=True, warmup_frames=0)
    camera._opened = True
    camera._capture = FakeCapture()
    camera._thread = object()  # take the drain path without running a thread
    assert camera._latest is None
    assert camera.read() is None


def test_a_camera_that_stops_returning_frames_is_a_fault(captures):
    """A live camera dying must invalidate the navigation solution."""
    captures.configure(fail_after=3)
    camera = OpenCVCamera(device="0", drain_to_newest=True, warmup_frames=0)
    camera.open()
    try:
        deadline = time.monotonic() + 3.0
        while camera._capture_error is None and time.monotonic() < deadline:
            time.sleep(0.02)
        with pytest.raises(CameraError, match="stopped returning frames"):
            camera.read()
    finally:
        camera.close()


def test_closing_stops_the_capture_thread_and_releases_the_device(captures):
    camera = OpenCVCamera(device="0", drain_to_newest=True, warmup_frames=0)
    camera.open()
    time.sleep(0.05)
    camera.close()
    assert camera._thread is None
    assert captures[0].released is True
    assert camera.opened is False


# ---------------------------------------------------------------------------
# File sources must not be treated as cameras
# ---------------------------------------------------------------------------
def test_a_file_source_skips_the_v4l2_configuration(captures):
    """None of the live settings mean anything for a video file."""
    camera = OpenCVCamera(device="/tmp/flight.mp4")
    camera.open()
    try:
        assert captures[0].calls == []
        assert camera._thread is None
    finally:
        camera.close()


def test_a_file_source_ending_is_not_an_error(captures):
    captures.configure(fail_after=2)
    camera = OpenCVCamera(device="/tmp/flight.mp4")
    camera.open()
    try:
        assert camera.read() is not None
        assert camera.read() is not None
        assert camera.read() is None
    finally:
        camera.close()


def test_reading_before_opening_is_an_error():
    with pytest.raises(CameraError, match="not open"):
        OpenCVCamera(device="0").read()
