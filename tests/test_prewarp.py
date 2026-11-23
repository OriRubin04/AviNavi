"""Frame rectification to a synthetic nadir, north-up, tile-scale view.

These are mostly exact geometric assertions rather than tolerance checks,
because the homography is closed-form: a pixel's ground point is determined, so
where it lands in the rectified image is determined too.  That matters here more
than usual.  Rectification is a *matching aid* whose output is composed back
into camera pixels before the pose solve, so a sign error does not produce a
visibly broken image - it produces slightly fewer inliers, and a yaw or position
bias that looks like ordinary noise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from avinavi.config import CameraConfig, CameraExtrinsicsConfig, PrewarpConfig
from avinavi.frames import CameraExtrinsics, euler_to_matrix
from avinavi.localization.prewarp import (
    FramePrewarper,
    PhotometricNormalizer,
    PrewarpError,
    virtual_nadir_homography,
)

FX = FY = 900.0
CX, CY = 640.0, 360.0
WIDTH, HEIGHT = 1280, 720


def intrinsics() -> np.ndarray:
    return np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float64)


def level_downward_camera() -> np.ndarray:
    """R_ned_from_camera for identity extrinsics in level flight, heading north."""
    extrinsics = CameraExtrinsics.from_config(CameraExtrinsicsConfig())
    return extrinsics.R_ned_from_camera(euler_to_matrix(0.0, 0.0, 0.0))


def apply(H: np.ndarray, u: float, v: float):
    out = H @ np.array([u, v, 1.0])
    return (out[0] / out[2], out[1] / out[2])


# ---------------------------------------------------------------------------
# The homography itself
# ---------------------------------------------------------------------------
def test_the_principal_point_lands_at_the_requested_output_centre():
    """The ground point under the optical axis is the one we can place exactly."""
    H = virtual_nadir_homography(
        intrinsics(), level_downward_camera(), height_m=100.0,
        gsd_mpp=0.5, output_center=(384.0, 384.0),
    )
    x, y = apply(H, CX, CY)
    assert x == pytest.approx(384.0, abs=1e-6)
    assert y == pytest.approx(384.0, abs=1e-6)


def test_north_is_up_and_east_is_right_in_the_rectified_image():
    """The sign convention, asserted on real numbers rather than trusted.

    With identity extrinsics the image +X axis runs along the body nose, so in
    level flight heading north, a pixel to the *right* of the principal point
    looks at ground that is further *north*, and must therefore appear *above*
    the centre of a north-up image: a smaller row.
    """
    height_m, gsd = 100.0, 0.5
    H = virtual_nadir_homography(
        intrinsics(), level_downward_camera(), height_m=height_m,
        gsd_mpp=gsd, output_center=(0.0, 0.0),
    )
    # 100 px right of the principal point, at 900 px focal length and 100 m
    # height, is 100/900 * 100 = 11.11 m of ground.
    expected_m = 100.0 / FX * height_m
    x, y = apply(H, CX + 100.0, CY)
    assert y == pytest.approx(-expected_m / gsd, rel=1e-9)
    assert x == pytest.approx(0.0, abs=1e-9)

    # 100 px below the principal point is 11.11 m east: +x, same row.
    x, y = apply(H, CX, CY + 100.0)
    assert x == pytest.approx(expected_m / gsd, rel=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_the_output_scale_is_the_requested_ground_resolution():
    """The point of the warp: output pixels are tile pixels."""
    height_m = 120.0
    for gsd in (0.25, 0.5, 1.0):
        H = virtual_nadir_homography(
            intrinsics(), level_downward_camera(), height_m=height_m,
            gsd_mpp=gsd, output_center=(0.0, 0.0),
        )
        x, _ = apply(H, CX, CY + 200.0)
        ground_m = 200.0 / FX * height_m
        assert x == pytest.approx(ground_m / gsd, rel=1e-9)


def test_yaw_rotates_the_scene_so_the_output_stays_north_up():
    """Heading east must put the nose-ward ground to the image right, not up."""
    extrinsics = CameraExtrinsics.from_config(CameraExtrinsicsConfig())
    R = extrinsics.R_ned_from_camera(euler_to_matrix(0.0, 0.0, math.radians(90.0)))
    H = virtual_nadir_homography(
        intrinsics(), R, height_m=100.0, gsd_mpp=0.5, output_center=(0.0, 0.0)
    )
    expected_px = (100.0 / FX * 100.0) / 0.5
    # Same pixel as the north-up case, but the aircraft points east, so the
    # ground it sees is east of the origin: +x, row unchanged.
    x, y = apply(H, CX + 100.0, CY)
    assert x == pytest.approx(expected_px, rel=1e-6)
    assert y == pytest.approx(0.0, abs=1e-6)


def test_a_non_positive_height_is_refused():
    for bad in (0.0, -10.0):
        with pytest.raises(PrewarpError, match="height"):
            virtual_nadir_homography(
                intrinsics(), level_downward_camera(), height_m=bad,
                gsd_mpp=0.5, output_center=(0.0, 0.0),
            )


def test_a_singular_intrinsic_matrix_is_refused():
    with pytest.raises(PrewarpError, match="invertible"):
        virtual_nadir_homography(
            np.zeros((3, 3)), level_downward_camera(), height_m=100.0,
            gsd_mpp=0.5, output_center=(0.0, 0.0),
        )


# ---------------------------------------------------------------------------
# The prewarper
# ---------------------------------------------------------------------------
def camera_config(**overrides) -> CameraConfig:
    config = CameraConfig()
    config.fx, config.fy, config.cx, config.cy = FX, FY, CX, CY
    config.width, config.height = WIDTH, HEIGHT
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def prewarper(**prewarp_overrides) -> FramePrewarper:
    prewarp = PrewarpConfig()
    for key, value in prewarp_overrides.items():
        setattr(prewarp, key, value)
    return FramePrewarper.from_config(
        camera_config(),
        CameraExtrinsics.from_config(CameraExtrinsicsConfig()),
        prewarp,
    )


def textured_frame() -> np.ndarray:
    """A frame with structure, so the valid-fraction check sees real pixels."""
    rng = np.random.default_rng(3)
    return rng.integers(40, 220, size=(HEIGHT, WIDTH), dtype=np.uint8)


def test_rectification_is_unavailable_without_calibrated_intrinsics():
    """No focal length means the geometry is undetermined; say so, don't guess."""
    bare = FramePrewarper.from_config(
        CameraConfig(), CameraExtrinsics.from_config(CameraExtrinsicsConfig())
    )
    assert bare.available is False
    with pytest.raises(PrewarpError, match="intrinsics"):
        bare.rectify(textured_frame(), height_m=100.0, target_gsd_mpp=0.5)


def test_a_level_frame_rectifies_and_reports_its_scale():
    result = prewarper().rectify(
        textured_frame(), height_m=120.0, target_gsd_mpp=0.5, height_source="baro"
    )
    assert result.valid_fraction > 0.9
    assert result.height_m == 120.0
    assert result.height_source == "baro"
    assert max(result.size) <= PrewarpConfig().max_output_side
    assert result.gsd_mpp >= 0.5


def test_the_forward_and_inverse_transforms_actually_invert():
    """The engine composes the tile match back through camera_from_virtual.

    If these two drift apart the pose solve is quietly solving the wrong
    problem, so it is worth asserting rather than assuming.
    """
    result = prewarper().rectify(textured_frame(), height_m=100.0, target_gsd_mpp=0.5)
    product = result.camera_from_virtual @ result.virtual_from_camera
    np.testing.assert_allclose(product / product[2, 2], np.eye(3), atol=1e-8)


def test_a_steep_bank_is_refused_rather_than_warped_into_nonsense():
    """Past the tilt limit the ground footprint is unbounded."""
    with pytest.raises(PrewarpError, match="off vertical"):
        prewarper(max_tilt_deg=55.0).rectify(
            textured_frame(), height_m=100.0, target_gsd_mpp=0.5,
            roll_rad=math.radians(70.0),
        )


def test_a_bank_within_the_limit_still_rectifies():
    result = prewarper().rectify(
        textured_frame(), height_m=100.0, target_gsd_mpp=0.5,
        roll_rad=math.radians(30.0),
    )
    # A banked frame's footprint is a trapezoid, so some of the output is
    # necessarily border; it must still be mostly real.
    assert result.valid_fraction >= PrewarpConfig().min_valid_fraction


def test_the_output_is_capped_and_the_reported_gsd_coarsens_with_it():
    """The size cap must be honest: if we coarsen, gsd_mpp has to say so.

    The verifier's scale band and the pose solve both read this number; silently
    capping the output while still claiming the tile's GSD would put a real
    scale error into the solution.
    """
    fine = prewarper(max_output_side=256).rectify(
        textured_frame(), height_m=400.0, target_gsd_mpp=0.05
    )
    assert max(fine.size) <= 256
    assert fine.gsd_mpp > 0.05


def test_a_frame_whose_footprint_misses_the_ground_is_refused():
    """Pitched past vertical, the corner rays diverge upwards."""
    with pytest.raises(PrewarpError):
        prewarper(max_tilt_deg=89.9).rectify(
            textured_frame(), height_m=100.0, target_gsd_mpp=0.5,
            pitch_rad=math.radians(-88.0),
        )


def test_an_empty_frame_is_refused():
    with pytest.raises(PrewarpError, match="empty"):
        prewarper().rectify(
            np.zeros((0, 0), dtype=np.uint8), height_m=100.0, target_gsd_mpp=0.5
        )


# ---------------------------------------------------------------------------
# Photometric normalisation
# ---------------------------------------------------------------------------
def test_normalisation_raises_the_contrast_of_a_flat_frame():
    from avinavi.config import PhotometricConfig

    normalizer = PhotometricNormalizer.from_config(PhotometricConfig())
    rng = np.random.default_rng(11)
    # A low-contrast frame, as a hazy or badly exposed camera produces.
    flat = (120 + rng.normal(0.0, 4.0, size=(240, 320))).astype(np.uint8)
    out = normalizer.apply(flat)
    assert out.shape == flat.shape
    assert out.std() > flat.std()


def test_normalisation_is_deterministic():
    """Both sides of the match must get byte-identical treatment."""
    from avinavi.config import PhotometricConfig

    normalizer = PhotometricNormalizer.from_config(PhotometricConfig())
    frame = textured_frame()
    np.testing.assert_array_equal(normalizer.apply(frame), normalizer.apply(frame))
