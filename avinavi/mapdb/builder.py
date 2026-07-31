"""Offline construction of the map tile database.

This runs on a workstation, not on the Pi.  Two jobs:

1. **Slice** a georeferenced source raster (GeoTIFF, or any image plus an
   explicit WGS-84 bounding box) into overlapping north-up tiles.
2. **Precompute** SuperPoint keypoints and descriptors for every tile and store
   them next to the image, so the flight software never runs the detector over
   static map data.

Both steps are idempotent: re-running only touches tiles that are missing or
whose features are stale.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, List, Optional, Tuple

import numpy as np

from ..geo import BoundingBox
from .database import MapDatabase
from .tile import (
    DEFAULT_FEATURE_FILENAME,
    DEFAULT_IMAGE_FILENAME,
    MapTile,
    TileFeatures,
    TileMetadata,
)

LOG = logging.getLogger(__name__)


@dataclass
class TilingSpec:
    """How to cut a source raster into tiles."""

    tile_size_px: int = 1024
    #: Fractional overlap between neighbouring tiles.  Overlap matters: a live
    #: frame that straddles a tile boundary must still fall wholly inside at
    #: least one tile, otherwise it can never accumulate enough inliers.
    overlap: float = 0.25
    #: Skip tiles that are more than this fraction nodata/black.
    max_blank_fraction: float = 0.5
    tile_prefix: str = "tile"

    def __post_init__(self) -> None:
        if self.tile_size_px < 64:
            raise ValueError("tile_size_px must be at least 64")
        if not 0.0 <= self.overlap < 0.9:
            raise ValueError("overlap must be in [0, 0.9)")

    @property
    def stride_px(self) -> int:
        return max(1, int(round(self.tile_size_px * (1.0 - self.overlap))))


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "building map tiles needs OpenCV; pip install opencv-python-headless"
        ) from exc
    return cv2


def slice_image(
    source_image_path: str,
    output_dir: str,
    bounds: Optional[BoundingBox] = None,
    spec: Optional[TilingSpec] = None,
    crs: str = "EPSG:4326",
    terrain_alt_m: float = 0.0,
) -> List[TileMetadata]:
    """Cut a source raster into north-up tiles and write their metadata.

    ``bounds`` is the WGS-84 footprint of the whole source image and is
    required for a plain image.  For a GeoTIFF it is read from the file when
    rasterio is installed, and each tile additionally gets an exact affine
    transform so no precision is lost at the tile level.
    """
    cv2 = _require_cv2()
    spec = spec or TilingSpec()
    os.makedirs(output_dir, exist_ok=True)

    source_transform = None
    if _is_geotiff(source_image_path):
        image, geo_bounds, source_transform, crs = _read_geotiff(source_image_path)
        bounds = bounds or geo_bounds
    else:
        image = cv2.imread(source_image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("could not read source image: %s" % source_image_path)
    if bounds is None:
        raise ValueError(
            "bounds are required for a non-georeferenced source image "
            "(pass --bounds west,south,east,north)"
        )

    height, width = image.shape[:2]
    written: List[TileMetadata] = []
    for index, (col0, row0, col1, row1) in enumerate(_tile_windows(width, height, spec)):
        patch = image[row0:row1, col0:col1]
        if _blank_fraction(patch) > spec.max_blank_fraction:
            LOG.debug("skipping mostly-blank tile at (%d, %d)", col0, row0)
            continue

        tile_id = "%s_%04d" % (spec.tile_prefix, index)
        directory = os.path.join(output_dir, tile_id)
        os.makedirs(directory, exist_ok=True)
        cv2.imwrite(
            os.path.join(directory, DEFAULT_IMAGE_FILENAME),
            patch,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )

        tile_bounds = _sub_bounds(bounds, width, height, col0, row0, col1, row1)
        transform = (
            _sub_transform(source_transform, col0, row0) if source_transform else None
        )
        metadata = TileMetadata(
            tile_id=tile_id,
            directory=directory,
            width_px=col1 - col0,
            height_px=row1 - row0,
            bounds=tile_bounds,
            crs=crs,
            transform=list(transform) if transform else None,
            terrain_alt_m=terrain_alt_m,
            extra={
                "source": os.path.basename(source_image_path),
                "source_window": [col0, row0, col1, row1],
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        metadata.ground_resolution_mpp = MapTile(metadata).georeference.ground_resolution_mpp
        metadata.save()
        written.append(metadata)

    LOG.info("wrote %d tiles to %s", len(written), output_dir)
    return written


def _tile_windows(
    width: int, height: int, spec: TilingSpec
) -> Iterator[Tuple[int, int, int, int]]:
    """Yield ``(col0, row0, col1, row1)`` windows covering the whole raster.

    The last row/column is snapped back to the raster edge so coverage is
    complete even when the dimensions are not a whole number of strides.
    """
    size = spec.tile_size_px
    stride = spec.stride_px
    rows = _starts(height, size, stride)
    cols = _starts(width, size, stride)
    for row0 in rows:
        for col0 in cols:
            yield (col0, row0, min(col0 + size, width), min(row0 + size, height))


def _starts(extent: int, size: int, stride: int) -> List[int]:
    if extent <= size:
        return [0]
    starts = list(range(0, extent - size + 1, stride))
    if starts[-1] + size < extent:
        starts.append(extent - size)
    return starts


def _blank_fraction(patch: np.ndarray) -> float:
    """Fraction of near-black pixels, a decent proxy for nodata in aerial imagery."""
    if patch.size == 0:
        return 1.0
    gray = patch if patch.ndim == 2 else patch.mean(axis=2)
    return float(np.count_nonzero(gray < 5) / gray.size)


def _sub_bounds(
    bounds: BoundingBox,
    width: int,
    height: int,
    col0: int,
    row0: int,
    col1: int,
    row1: int,
) -> BoundingBox:
    lon_span = bounds.east - bounds.west
    lat_span = bounds.north - bounds.south
    return BoundingBox(
        west=bounds.west + lon_span * col0 / width,
        south=bounds.north - lat_span * row1 / height,
        east=bounds.west + lon_span * col1 / width,
        north=bounds.north - lat_span * row0 / height,
    )


def _sub_transform(transform: Tuple[float, ...], col0: int, row0: int) -> Tuple[float, ...]:
    """Shift a GDAL affine transform so pixel (0,0) lands at ``(col0, row0)``."""
    a, b, c, d, e, f = transform
    return (a, b, c + a * col0 + b * row0, d, e, f + d * col0 + e * row0)


def _is_geotiff(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".tif", ".tiff")


def _read_geotiff(path: str):
    """Read a GeoTIFF into a BGR array plus its bounds/transform/CRS."""
    try:
        import rasterio  # noqa: PLC0415 - optional dependency
        from rasterio.warp import transform_bounds  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "reading a GeoTIFF needs rasterio; pip install 'avinavi[maps]' or "
            "export the mosaic to a plain image and pass --bounds"
        ) from exc

    with rasterio.open(path) as dataset:
        band_count = min(3, dataset.count)
        bands = [dataset.read(i + 1) for i in range(band_count)]
        if band_count == 1:
            image = np.dstack([bands[0]] * 3)
        else:
            # rasterio gives RGB; OpenCV wants BGR.
            image = np.dstack(bands[:3][::-1])
        if image.dtype != np.uint8:
            finite = image[np.isfinite(image)]
            scale = float(finite.max()) if finite.size else 1.0
            image = np.clip(image / max(scale, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        west, south, east, north = transform_bounds(
            dataset.crs, "EPSG:4326", *dataset.bounds, densify_pts=21
        )
        t = dataset.transform
        transform = (t.a, t.b, t.c, t.d, t.e, t.f)
        crs = str(dataset.crs)
    return image, BoundingBox(west, south, east, north), transform, crs


def build_features(
    database_path: str,
    extractor,
    force: bool = False,
    resize_max_side: Optional[int] = None,
) -> int:
    """Run SuperPoint over every tile image and store the result.

    ``extractor`` is a :class:`~avinavi.features.superpoint.SuperPointExtractor`
    (or anything exposing the same ``extract_image_file`` interface).  Keypoints
    are rescaled back to full tile resolution before being written, so match
    geometry is always expressed in the tile's own pixel grid.

    Returns the number of tiles whose features were (re)written.
    """
    cv2 = _require_cv2()
    database = MapDatabase.load(database_path, feature_cache_size=1)
    updated = 0
    for tile in database:
        feature_path = tile.metadata.feature_path
        image_path = tile.metadata.image_path
        if not os.path.exists(image_path):
            LOG.warning("tile %s has no image at %s; skipping", tile.tile_id, image_path)
            continue
        if not force and _features_are_current(feature_path, image_path):
            continue

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            LOG.warning("could not read %s; skipping", image_path)
            continue

        features = extractor.extract(image, resize_max_side=resize_max_side)
        features.image_size = (float(image.shape[1]), float(image.shape[0]))
        features.save(feature_path)
        updated += 1
        LOG.info("tile %s: %d keypoints", tile.tile_id, len(features))

    database.write_index()
    return updated


def _features_are_current(feature_path: str, image_path: str) -> bool:
    candidates = [feature_path, os.path.splitext(feature_path)[0] + ".npz"]
    for path in candidates:
        if os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(image_path):
            return True
    return False


def write_tile(
    directory: str,
    image: np.ndarray,
    bounds: BoundingBox,
    tile_id: Optional[str] = None,
    crs: str = "EPSG:4326",
    terrain_alt_m: float = 0.0,
    features: Optional[TileFeatures] = None,
) -> TileMetadata:
    """Write a single tile from an in-memory image.

    Used by ``avinavi build-map`` for already-cut imagery and by the tests to
    build small synthetic databases without touching a real map product.
    """
    cv2 = _require_cv2()
    os.makedirs(directory, exist_ok=True)
    height, width = image.shape[:2]
    cv2.imwrite(os.path.join(directory, DEFAULT_IMAGE_FILENAME), image)
    metadata = TileMetadata(
        tile_id=tile_id or os.path.basename(directory.rstrip("/")),
        directory=directory,
        width_px=width,
        height_px=height,
        bounds=bounds,
        crs=crs,
        terrain_alt_m=terrain_alt_m,
        extra={"created_utc": datetime.now(timezone.utc).isoformat()},
    )
    metadata.ground_resolution_mpp = MapTile(metadata).georeference.ground_resolution_mpp
    metadata.save()
    if features is not None:
        features.save(os.path.join(directory, DEFAULT_FEATURE_FILENAME))
    return metadata


def summarise(database_path: str) -> dict:
    """Human-readable summary of a database, used by ``avinavi map-info``."""
    database = MapDatabase.load(database_path, feature_cache_size=1)
    coverage = database.coverage
    resolutions = [t.ground_resolution_mpp for t in database]
    missing_features = [
        t.tile_id
        for t in database
        if not (
            os.path.exists(t.metadata.feature_path)
            or os.path.exists(os.path.splitext(t.metadata.feature_path)[0] + ".npz")
        )
    ]
    return {
        "path": database_path,
        "tiles": len(database),
        "coverage": coverage.as_tuple() if coverage else None,
        "coverage_center": (
            {"lat": coverage.center.lat_deg, "lon": coverage.center.lon_deg}
            if coverage
            else None
        ),
        "ground_resolution_mpp": {
            "min": round(min(resolutions), 4) if resolutions else None,
            "max": round(max(resolutions), 4) if resolutions else None,
            "mean": round(sum(resolutions) / len(resolutions), 4) if resolutions else None,
        },
        "tiles_missing_features": missing_features,
    }
