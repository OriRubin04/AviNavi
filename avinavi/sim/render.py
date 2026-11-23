"""Rendering what a USB camera on a fixed wing would actually see.

A renderer that just crops the basemap would make the pipeline look perfect and
tell us nothing.  The failure modes that matter in flight all come from the gap
between the basemap and the live frame, so each one is modelled explicitly and
can be switched off to measure its individual cost:

``perspective``   the full pinhole projection with the aircraft's real bank and
                  pitch, so the frame is a trapezoid of ground, not a crop
``motion blur``   at 22 m/s and 120 m AGL the ground moves a measurable number
                  of pixels during the exposure; see :meth:`motion_blur_px`
``rolling shutter`` cheap USB cameras expose row by row, which shears the image
                  when the aircraft rotates - a real and often-ignored effect
``exposure``      the webcam's auto-exposure hunts, so brightness and gamma
                  drift frame to frame independently of the basemap
``vignetting``    cheap lenses darken at the corners, suppressing detections
                  exactly where the geometry most needs them
``JPEG``          MJPEG is the only way to get useful frame rates over USB 2.0,
                  and its block artifacts interact badly with corner detectors
``noise``         sensor noise, which CLAHE will happily amplify
``defocus``       a fixed-focus lens at the wrong distance

Rolling shutter deserves a note: the model here shears the image by the
rotation accumulated over the readout time, which captures the dominant effect
for a rotating aircraft. It does not re-project each row from a separately
integrated pose, so it understates the distortion in a hard, fast turn.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..frames import CameraExtrinsics, euler_to_matrix
from .flight import TrajectoryPoint

LOG = logging.getLogger(__name__)


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("the simulator needs OpenCV") from exc
    return cv2


@dataclass
class RenderSettings:
    """Camera and degradation settings.

    Defaults model a decent USB 2.0 webcam: 1280x720 MJPEG, 60 degree
    horizontal field of view, 1/500 s exposure, 25 ms rolling-shutter readout.
    """

    width_px: int = 1280
    height_px: int = 720
    horizontal_fov_deg: float = 60.0
    exposure_s: float = 1.0 / 500.0
    rolling_shutter_readout_s: float = 0.025

    enable_motion_blur: bool = True
    enable_rolling_shutter: bool = True
    enable_exposure_drift: bool = True
    enable_vignetting: bool = True
    enable_jpeg: bool = True
    enable_noise: bool = True
    enable_defocus: bool = True

    jpeg_quality: int = 72
    noise_sigma: float = 3.5
    vignette_strength: float = 0.35
    exposure_drift_stops: float = 0.45
    defocus_sigma_px: float = 0.6
    seed: int = 23

    @property
    def fx(self) -> float:
        """Focal length in pixels implied by the horizontal field of view."""
        return self.width_px / (2.0 * math.tan(math.radians(self.horizontal_fov_deg) / 2.0))

    @property
    def fy(self) -> float:
        # Square pixels, which is what a UVC webcam gives.
        return self.fx

    @property
    def cx(self) -> float:
        return self.width_px / 2.0

    @property
    def cy(self) -> float:
        return self.height_px / 2.0

    def intrinsics(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def ground_gsd_mpp(self, height_m: float) -> float:
        """Metres of ground per image pixel at nadir, at this height."""
        return height_m / self.fx

    def motion_blur_px(self, height_m: float, speed_mps: float) -> float:
        """Ground smear during one exposure, in pixels.

        ``blur = speed · exposure / gsd``.  At 22 m/s, 120 m AGL, fx≈1108 px
        and 1/500 s that is about 0.4 px - negligible.  At 1/60 s it is 3.4 px,
        which visibly costs keypoints.  This is why the exposure matters more
        than the resolution on a fast aircraft.
        """
        gsd = self.ground_gsd_mpp(height_m)
        if gsd <= 0:
            return 0.0
        return speed_mps * self.exposure_s / gsd


class CameraSimulator:
    """Renders camera frames from a basemap along a trajectory."""

    def __init__(
        self,
        basemap: np.ndarray,
        basemap_gsd_mpp: float,
        settings: Optional[RenderSettings] = None,
        extrinsics: Optional[CameraExtrinsics] = None,
        #: Where the local NED origin sits in basemap pixels (col, row).
        origin_pixel: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.basemap = basemap
        self.basemap_gsd_mpp = float(basemap_gsd_mpp)
        self.settings = settings or RenderSettings()
        self.extrinsics = extrinsics or CameraExtrinsics()
        if origin_pixel is None:
            origin_pixel = (basemap.shape[1] / 2.0, basemap.shape[0] / 2.0)
        self.origin_pixel = origin_pixel
        self._rng = np.random.default_rng(self.settings.seed)

    # -- geometry -----------------------------------------------------------
    def ground_to_basemap_pixel(self, north_m: float, east_m: float) -> Tuple[float, float]:
        """Local NED ground coordinates -> basemap pixel (col, row).

        The basemap is north-up, so +east is +col and +north is -row.
        """
        col = self.origin_pixel[0] + east_m / self.basemap_gsd_mpp
        row = self.origin_pixel[1] - north_m / self.basemap_gsd_mpp
        return (col, row)

    def camera_from_basemap_homography(self, point: TrajectoryPoint) -> np.ndarray:
        """Homography mapping basemap pixels into the camera image.

        Derived the same way as the pre-warp, but in the opposite direction: a
        basemap pixel is a ground point, which is projected through the full
        pinhole model with the aircraft's attitude.
        """
        K = self.settings.intrinsics()
        R_ned_from_body = euler_to_matrix(point.roll_rad, point.pitch_rad, point.yaw_rad)
        R_ned_from_camera = self.extrinsics.R_ned_from_camera(R_ned_from_body)
        R_camera_from_ned = R_ned_from_camera.T

        # Camera centre in local NED (down negative above the terrain).
        camera_ned = np.array([point.north_m, point.east_m, -point.height_m])

        # Basemap pixel -> ground NED: an affine map.
        g = self.basemap_gsd_mpp
        # north = (origin_row - row) * g ; east = (col - origin_col) * g
        pixel_to_ground = np.array(
            [
                [0.0, -g, self.origin_pixel[1] * g],          # north
                [g, 0.0, -self.origin_pixel[0] * g],          # east
                [0.0, 0.0, 0.0],                              # down = 0 (terrain)
                [0.0, 0.0, 1.0],
            ]
        )
        # Ground point relative to the camera, then into the camera frame.
        translate = np.hstack([np.eye(3), -camera_ned.reshape(3, 1)])
        return K @ R_camera_from_ned @ translate @ pixel_to_ground

    # -- rendering ----------------------------------------------------------
    def render(self, point: TrajectoryPoint) -> Optional[np.ndarray]:
        """Render one frame.  Returns ``None`` if the view leaves the basemap."""
        cv2 = _require_cv2()
        settings = self.settings

        H = self.camera_from_basemap_homography(point)
        try:
            H_inverse = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None

        # Check the frame's footprint lands on the basemap before warping.
        corners = np.array(
            [
                [0.0, 0.0, 1.0],
                [settings.width_px - 1.0, 0.0, 1.0],
                [settings.width_px - 1.0, settings.height_px - 1.0, 1.0],
                [0.0, settings.height_px - 1.0, 1.0],
            ]
        )
        projected = (H_inverse @ corners.T).T
        w = projected[:, 2:3]
        if np.any(np.abs(w) < 1e-9):
            return None
        footprint = projected[:, :2] / w
        margin = 8
        if (
            footprint[:, 0].min() < margin
            or footprint[:, 1].min() < margin
            or footprint[:, 0].max() > self.basemap.shape[1] - margin
            or footprint[:, 1].max() > self.basemap.shape[0] - margin
        ):
            return None

        frame = cv2.warpPerspective(
            self.basemap,
            H,
            (settings.width_px, settings.height_px),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return self.degrade(frame, point)

    def degrade(self, frame: np.ndarray, point: TrajectoryPoint) -> np.ndarray:
        """Apply the sensor and optical degradations."""
        cv2 = _require_cv2()
        settings = self.settings
        rng = self._rng
        image = frame.astype(np.float32)

        if settings.enable_rolling_shutter:
            image = self._rolling_shutter(image, point)

        if settings.enable_motion_blur:
            blur_px = settings.motion_blur_px(point.height_m, point.ground_speed_mps)
            if blur_px > 0.6:
                image = self._directional_blur(image, blur_px, point)

        if settings.enable_defocus and settings.defocus_sigma_px > 0:
            image = cv2.GaussianBlur(image, (0, 0), settings.defocus_sigma_px)

        if settings.enable_exposure_drift:
            # Auto-exposure hunting: a multiplicative gain plus a gamma shift.
            stops = rng.normal(0.0, settings.exposure_drift_stops)
            image *= float(2.0 ** stops)
            gamma = float(np.clip(1.0 + rng.normal(0.0, 0.12), 0.6, 1.6))
            image = 255.0 * np.power(np.clip(image / 255.0, 0, 1), gamma)

        if settings.enable_vignetting:
            image *= self._vignette(image.shape[:2])

        if settings.enable_noise and settings.noise_sigma > 0:
            image += rng.normal(0.0, settings.noise_sigma, image.shape)

        image = np.clip(image, 0, 255).astype(np.uint8)

        if settings.enable_jpeg:
            ok, encoded = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality]
            )
            if ok:
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        return image

    def _rolling_shutter(self, image: np.ndarray, point: TrajectoryPoint) -> np.ndarray:
        """Shear the image by the rotation accumulated over the readout."""
        cv2 = _require_cv2()
        settings = self.settings
        # Turn rate from the bank angle, as in the coordinated-turn model.
        turn_rate = 9.80665 * math.tan(point.roll_rad) / max(point.ground_speed_mps, 1.0)
        rotation_over_readout = turn_rate * settings.rolling_shutter_readout_s
        if abs(rotation_over_readout) < 1e-4:
            return image
        # Rows later in the readout are rotated slightly more; approximate the
        # whole thing as a shear whose magnitude is the accumulated rotation.
        shear = rotation_over_readout * settings.width_px / 2.0
        matrix = np.array(
            [[1.0, shear / max(settings.height_px, 1), 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        return cv2.warpAffine(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _directional_blur(
        self, image: np.ndarray, length_px: float, point: TrajectoryPoint
    ) -> np.ndarray:
        """Smear along the ground-track direction as seen in the image."""
        cv2 = _require_cv2()
        # Direction of travel in the image plane: rotate the NED track into the
        # camera frame and drop the depth component.
        R_ned_from_body = euler_to_matrix(point.roll_rad, point.pitch_rad, point.yaw_rad)
        R_camera_from_ned = self.extrinsics.R_ned_from_camera(R_ned_from_body).T
        velocity_camera = R_camera_from_ned @ np.array(point.velocity_ned)
        angle = math.atan2(float(velocity_camera[1]), float(velocity_camera[0]))

        size = max(3, int(round(length_px)) | 1)
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1.0
        rotation = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), -math.degrees(angle), 1.0)
        kernel = cv2.warpAffine(kernel, rotation, (size, size))
        total = kernel.sum()
        if total <= 0:
            return image
        return cv2.filter2D(image, -1, kernel / total)

    def _vignette(self, shape: Tuple[int, int]) -> np.ndarray:
        height, width = shape
        ys, xs = np.mgrid[0:height, 0:width]
        cx, cy = width / 2.0, height / 2.0
        radius = np.hypot(xs - cx, ys - cy) / math.hypot(cx, cy)
        falloff = 1.0 - self.settings.vignette_strength * radius**2
        return np.clip(falloff, 0.0, 1.0)[:, :, None].astype(np.float32)
