"""Pre-warping the live frame into a synthetic nadir, north-up, tile-GSD view.

This is the single largest lever on match rate, and the reason is specific:
**SuperPoint descriptors are not rotation invariant, and LightGlue is trained on
pairs that are broadly upright.**  A fixed-wing aircraft flying a 180-degree
turn presents the ground to the camera rotated by up to 180 degrees relative to
a north-up basemap tile, banked by 20-40 degrees, and at an altitude that may
differ from the basemap's by a factor of two.  Matching a raw frame against a
raw tile asks the network to bridge all three gaps at once.

So we close the gaps geometrically first, using information we already have:
the autopilot's attitude (`ATTITUDE`) and a height above terrain (barometric,
rangefinder, or the previous fix).  The live frame is warped into the view a
*virtual* camera would see if it were:

* pointing straight down (removes the perspective and bank),
* oriented north-up (removes the rotation), and
* sampling the ground at the basemap's metres-per-pixel (removes the scale).

What is left for LightGlue is close to a pure translation, which is the regime
it is strongest in.

Crucially, this is only a **matching aid**.  The homography recovered against
the tile is composed back through the warp::

    H_tile_from_camera = H_tile_from_virtual @ H_virtual_from_camera

and handed to the unchanged :class:`~avinavi.localization.pose.PoseSolver`.  The
pose solve stays exactly as rigorous as before; no approximation introduced here
leaks into the position estimate.  If the attitude or height used for the warp
is wrong, the warp is simply less effective at closing the gap - the resulting
match is still verified by RANSAC and solved for exactly.

Derivation
----------
Let ``K`` be the camera intrinsics, ``R = R_ned_from_camera`` the rotation from
the camera frame into NED, and ``h`` the camera's height above the ground plane.
A pixel ``(u, v)`` defines a ray ``p = K⁻¹ [u, v, 1]ᵀ`` in the camera frame,
which in NED is ``R p``.  Writing the rows of ``R`` as ``r₀`` (north), ``r₁``
(east) and ``r₂`` (down), that ray meets the ground plane at

    north = h · (r₀·p) / (r₂·p)        east = h · (r₁·p) / (r₂·p)

The virtual nadir image is north-up at ``g`` metres per pixel, so ``+x`` is east
and ``+y`` is south:

    x = (h/g) · (r₁·p)/(r₂·p) + cx     y = −(h/g) · (r₀·p)/(r₂·p) + cy

Clearing the denominator gives a homography ``H = M K⁻¹`` with rows

    M₀ = (h/g)·r₁ + cx·r₂
    M₁ = −(h/g)·r₀ + cy·r₂
    M₂ = r₂

which is what :func:`virtual_nadir_homography` returns.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..frames import CameraExtrinsics, euler_to_matrix

LOG = logging.getLogger(__name__)


class PrewarpError(RuntimeError):
    """The frame could not be rectified (geometry degenerate or unusable)."""


@dataclass
class PrewarpResult:
    """A rectified frame and the transform that produced it."""

    image: np.ndarray
    #: Maps live-camera pixels into the rectified image.
    virtual_from_camera: np.ndarray
    #: The inverse, for composing a tile match back to the camera.
    camera_from_virtual: np.ndarray
    #: Metres of ground per pixel in the rectified image.
    gsd_mpp: float
    #: Fraction of the rectified image that real pixels were available for.
    valid_fraction: float
    #: Height above terrain that was used, and where it came from.
    height_m: float
    height_source: str

    @property
    def size(self) -> Tuple[int, int]:
        return (int(self.image.shape[1]), int(self.image.shape[0]))


def virtual_nadir_homography(
    intrinsics: np.ndarray,
    R_ned_from_camera: np.ndarray,
    height_m: float,
    gsd_mpp: float,
    output_center: Tuple[float, float],
) -> np.ndarray:
    """Homography mapping live-camera pixels to a north-up nadir view.

    See the module docstring for the derivation.  ``output_center`` is where the
    ground point under the camera's optical axis should land in the output.
    """
    if height_m <= 0.0:
        raise PrewarpError("height above terrain must be positive, got %r" % height_m)
    if gsd_mpp <= 0.0:
        raise PrewarpError("target ground resolution must be positive")

    K = np.asarray(intrinsics, dtype=np.float64)
    R = np.asarray(R_ned_from_camera, dtype=np.float64)
    scale = height_m / gsd_mpp
    cx, cy = output_center

    north_row, east_row, down_row = R[0], R[1], R[2]
    M = np.vstack(
        [
            scale * east_row + cx * down_row,
            -scale * north_row + cy * down_row,
            down_row,
        ]
    )
    try:
        K_inverse = np.linalg.inv(K)
    except np.linalg.LinAlgError as exc:
        raise PrewarpError("camera intrinsics are not invertible") from exc
    return M @ K_inverse


class FramePrewarper:
    """Rectifies live frames to a nadir, north-up, basemap-scale view.

    Needs calibrated intrinsics.  Without them the geometry is undetermined and
    :meth:`available` is False, in which case the engine falls back to matching
    raw frames - which still works, just with a markedly lower inlier rate.
    """

    def __init__(
        self,
        extrinsics: CameraExtrinsics,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        max_output_side: int = 768,
        #: Refuse to warp when the optical axis is this far from vertical; a
        #: near-horizon view has unbounded ground footprint and rectifying it
        #: produces mostly extrapolated nonsense.
        max_tilt_deg: float = 55.0,
        #: Reject a rectification with less real coverage than this.
        min_valid_fraction: float = 0.25,
        #: Pad the output beyond the frame footprint by this factor.
        footprint_margin: float = 1.05,
    ) -> None:
        self.extrinsics = extrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.max_output_side = int(max_output_side)
        self.max_tilt_deg = float(max_tilt_deg)
        self.min_valid_fraction = float(min_valid_fraction)
        self.footprint_margin = float(footprint_margin)

    @classmethod
    def from_config(cls, camera_config, extrinsics: CameraExtrinsics, prewarp_config=None):
        kwargs = {}
        if prewarp_config is not None:
            kwargs = dict(
                max_output_side=prewarp_config.max_output_side,
                max_tilt_deg=prewarp_config.max_tilt_deg,
                min_valid_fraction=prewarp_config.min_valid_fraction,
                footprint_margin=prewarp_config.footprint_margin,
            )
        return cls(
            extrinsics=extrinsics,
            fx=camera_config.fx,
            fy=camera_config.fy,
            cx=camera_config.cx,
            cy=camera_config.cy,
            **kwargs,
        )

    @property
    def available(self) -> bool:
        return bool(self.fx and self.fy and self.fx > 0 and self.fy > 0)

    def intrinsics_matrix(self, frame_size: Tuple[int, int]) -> np.ndarray:
        if not self.available:
            raise PrewarpError(
                "pre-warping needs calibrated intrinsics; set camera.fx and camera.fy"
            )
        cx = self.cx if self.cx is not None else frame_size[0] * 0.5
        cy = self.cy if self.cy is not None else frame_size[1] * 0.5
        return np.array(
            [[self.fx, 0.0, cx], [0.0, self.fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def rectify(
        self,
        frame: np.ndarray,
        height_m: float,
        target_gsd_mpp: float,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        yaw_rad: float = 0.0,
        height_source: str = "unknown",
    ) -> PrewarpResult:
        """Rectify ``frame``.  Raises :class:`PrewarpError` if it cannot.

        ``yaw_rad`` is the vehicle heading.  Supplying it makes the output
        north-up, which is what removes the rotation gap; passing 0 still
        removes the perspective and the scale gap, leaving only rotation.
        """
        cv2 = _require_cv2()
        frame = np.asarray(frame)
        if frame.size == 0:
            raise PrewarpError("empty frame")
        frame_size = (int(frame.shape[1]), int(frame.shape[0]))

        K = self.intrinsics_matrix(frame_size)
        R_ned_from_body = euler_to_matrix(roll_rad, pitch_rad, yaw_rad)
        R_ned_from_camera = self.extrinsics.R_ned_from_camera(R_ned_from_body)

        # Tilt check: the optical axis must point meaningfully downwards.
        optical_axis_down = float(R_ned_from_camera[2, 2])
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, optical_axis_down))))
        if optical_axis_down <= 0.0 or tilt_deg > self.max_tilt_deg:
            raise PrewarpError(
                "optical axis is %.0f degrees off vertical (limit %.0f); the ground "
                "footprint is unbounded" % (tilt_deg, self.max_tilt_deg)
            )

        # Work out the ground footprint of the frame corners so the output is
        # sized to the data rather than to a guess.
        corners = np.array(
            [
                [0.0, 0.0],
                [frame_size[0] - 1.0, 0.0],
                [frame_size[0] - 1.0, frame_size[1] - 1.0],
                [0.0, frame_size[1] - 1.0],
            ]
        )
        ground = self._project_to_ground(corners, K, R_ned_from_camera, height_m)
        if ground is None:
            raise PrewarpError("frame corners do not all intersect the ground plane")

        # Ground extent in metres, north/east.
        north_span = float(ground[:, 0].max() - ground[:, 0].min())
        east_span = float(ground[:, 1].max() - ground[:, 1].min())
        if not (math.isfinite(north_span) and math.isfinite(east_span)):
            raise PrewarpError("ground footprint is not finite")

        gsd = float(target_gsd_mpp)
        width_px = east_span * self.footprint_margin / gsd
        height_px = north_span * self.footprint_margin / gsd
        longest = max(width_px, height_px)
        if longest > self.max_output_side:
            # Honest trade: cap the output size and coarsen the GSD rather than
            # spend the Pi's budget warping a huge image. The residual scale
            # mismatch against the tile is recorded in gsd_mpp and handled by
            # the verifier's scale band.
            gsd *= longest / self.max_output_side
            width_px = east_span * self.footprint_margin / gsd
            height_px = north_span * self.footprint_margin / gsd
        out_w = max(32, int(round(width_px)))
        out_h = max(32, int(round(height_px)))

        # Centre the footprint in the output.
        centre_north = 0.5 * (ground[:, 0].max() + ground[:, 0].min())
        centre_east = 0.5 * (ground[:, 1].max() + ground[:, 1].min())
        output_center = (
            out_w * 0.5 - centre_east / gsd,
            out_h * 0.5 + centre_north / gsd,
        )

        H = virtual_nadir_homography(K, R_ned_from_camera, height_m, gsd, output_center)

        warped = cv2.warpPerspective(
            frame,
            H,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_fraction = _valid_fraction(warped)
        if valid_fraction < self.min_valid_fraction:
            raise PrewarpError(
                "rectified frame is only %.0f%% real pixels (minimum %.0f%%)"
                % (valid_fraction * 100.0, self.min_valid_fraction * 100.0)
            )

        try:
            H_inverse = np.linalg.inv(H)
        except np.linalg.LinAlgError as exc:
            raise PrewarpError("rectification homography is singular") from exc

        return PrewarpResult(
            image=warped,
            virtual_from_camera=H,
            camera_from_virtual=H_inverse,
            gsd_mpp=gsd,
            valid_fraction=valid_fraction,
            height_m=float(height_m),
            height_source=height_source,
        )

    @staticmethod
    def _project_to_ground(
        pixels: np.ndarray, K: np.ndarray, R_ned_from_camera: np.ndarray, height_m: float
    ) -> Optional[np.ndarray]:
        """Intersect image rays with the ground plane; returns (N, 2) north/east."""
        homogeneous = np.hstack([pixels, np.ones((pixels.shape[0], 1))])
        rays_camera = (np.linalg.inv(K) @ homogeneous.T).T
        rays_ned = (R_ned_from_camera @ rays_camera.T).T
        down = rays_ned[:, 2]
        # A ray pointing up or along the horizon never meets the ground.
        if np.any(down <= 1e-6):
            return None
        t = height_m / down
        return np.column_stack([t * rays_ned[:, 0], t * rays_ned[:, 1]])


def _valid_fraction(warped: np.ndarray) -> float:
    """Fraction of the warped image that is not border fill."""
    if warped.size == 0:
        return 0.0
    gray = warped if warped.ndim == 2 else warped.max(axis=2)
    return float(np.count_nonzero(gray) / gray.size)


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise PrewarpError("pre-warping needs OpenCV") from exc
    return cv2


# ---------------------------------------------------------------------------
# Photometric normalisation
# ---------------------------------------------------------------------------
class PhotometricNormalizer:
    """Bridges the appearance gap between a live camera and a basemap.

    The basemap was shot by a different sensor, in a different season, at a
    different time of day, and has been through a mosaicking pipeline.  The live
    frame has whatever exposure the webcam's auto-exposure chose a moment ago.
    SuperPoint's detections are sensitive to local contrast, so normalising both
    sides to a common local-contrast statistic recovers detections that a raw
    comparison loses.

    CLAHE (contrast-limited adaptive histogram equalisation) is the workhorse:
    it equalises *locally*, so a bright field next to a dark wood both end up
    with usable contrast, which a global histogram stretch cannot do.

    **Apply the same normalisation to the map tiles when building the database**
    (``avinavi extract-features --normalize``) and to live frames at runtime.
    Normalising only one side makes things worse, not better.
    """

    def __init__(
        self,
        enabled: bool = True,
        clip_limit: float = 2.0,
        tile_grid: int = 8,
        #: Mild Gaussian blur before equalisation suppresses sensor noise that
        #: CLAHE would otherwise amplify into spurious keypoints.
        denoise_sigma: float = 0.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.clip_limit = float(clip_limit)
        self.tile_grid = int(tile_grid)
        self.denoise_sigma = float(denoise_sigma)
        self._clahe = None

    @classmethod
    def from_config(cls, config) -> "PhotometricNormalizer":
        return cls(
            enabled=config.enabled,
            clip_limit=config.clip_limit,
            tile_grid=config.tile_grid,
            denoise_sigma=config.denoise_sigma,
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Normalise an image, returning 8-bit grayscale."""
        gray = _to_gray_u8(image)
        if not self.enabled:
            return gray
        cv2 = _require_cv2()
        if self.denoise_sigma > 0:
            gray = cv2.GaussianBlur(gray, (0, 0), self.denoise_sigma)
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=(self.tile_grid, self.tile_grid),
            )
        return self._clahe.apply(gray)


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        try:
            cv2 = _require_cv2()
            channels = array.shape[2]
            code = cv2.COLOR_BGRA2GRAY if channels == 4 else cv2.COLOR_BGR2GRAY
            array = cv2.cvtColor(array, code)
        except PrewarpError:
            array = array[:, :, :3].mean(axis=2)
    if array.dtype != np.uint8:
        maximum = float(np.nanmax(array)) if array.size else 1.0
        scale = 255.0 if maximum <= 1.5 else (255.0 / max(maximum, 1e-6))
        array = np.clip(np.nan_to_num(array) * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)
