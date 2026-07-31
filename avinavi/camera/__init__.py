"""Camera frame sources.

Three interchangeable implementations behind one interface:

``picamera2``  the Raspberry Pi camera stack, for flight
``opencv``     any V4L2/USB camera or video file, for bench work
``directory``  a folder of images replayed in order, for offline development

All of them yield :class:`~avinavi.camera.base.Frame` objects and report
failures through :class:`~avinavi.camera.base.CameraError` rather than returning
empty images, so the runner can invalidate the navigation solution when the
camera dies instead of localizing against noise.
"""

from .base import CameraError, CameraSource, Frame
from .factory import create_camera

__all__ = ["CameraError", "CameraSource", "Frame", "create_camera"]
