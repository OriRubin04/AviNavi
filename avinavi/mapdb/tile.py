"""A single georeferenced map tile and its precomputed features.

Layout on disk, one directory per tile::

    map_database/
        index.json              (optional, generated - speeds up startup)
        tile_001/
            image.jpg           the tile raster, stored north-up
            features.pt         precomputed SuperPoint keypoints + descriptors
            metadata.json       georeferencing and provenance

``features.pt`` is a torch save file when torch is available at build time and
a ``.npz`` otherwise; both are loaded transparently.  The Pi never runs
SuperPoint over a map tile at flight time - that is the whole point of
precomputing them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..geo import BoundingBox, Georeference
from ..types import GeoPoint

METADATA_FILENAME = "metadata.json"
DEFAULT_IMAGE_FILENAME = "image.jpg"
DEFAULT_FEATURE_FILENAME = "features.pt"


class TileFormatError(ValueError):
    """Raised when a tile directory cannot be interpreted."""


@dataclass
class TileMetadata:
    """Everything needed to georeference a tile and find its data files."""

    tile_id: str
    directory: str
    width_px: int
    height_px: int
    bounds: BoundingBox
    crs: str = "EPSG:4326"
    transform: Optional[list] = None
    image_filename: str = DEFAULT_IMAGE_FILENAME
    feature_filename: str = DEFAULT_FEATURE_FILENAME
    #: Nominal ground sample distance, metres per pixel.  Recomputed from the
    #: bounds when absent so a hand-written metadata file stays usable.
    ground_resolution_mpp: Optional[float] = None
    #: Mean terrain altitude over the tile, same datum as the EKF origin.
    terrain_alt_m: float = 0.0
    #: What built ``features.pt``: backend, detector, descriptor width, and the
    #: resolution the detector actually saw.  ``None`` for a database written
    #: before fingerprinting existed, which is reported but not fatal.  See
    #: :mod:`avinavi.features.fingerprint` for why this is recorded at all.
    extractor: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def image_path(self) -> str:
        return os.path.join(self.directory, self.image_filename)

    @property
    def feature_path(self) -> str:
        return os.path.join(self.directory, self.feature_filename)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], directory: str) -> "TileMetadata":
        required = ("width_px", "height_px", "bounds")
        missing = [key for key in required if key not in data]
        if missing:
            raise TileFormatError(
                "%s: metadata is missing %s" % (directory, ", ".join(missing))
            )
        bounds_raw = data["bounds"]
        if isinstance(bounds_raw, dict):
            try:
                bounds = BoundingBox(
                    float(bounds_raw["west"]),
                    float(bounds_raw["south"]),
                    float(bounds_raw["east"]),
                    float(bounds_raw["north"]),
                )
            except KeyError as exc:
                raise TileFormatError(
                    "%s: bounds must have west/south/east/north" % directory
                ) from exc
        elif isinstance(bounds_raw, (list, tuple)) and len(bounds_raw) == 4:
            bounds = BoundingBox(*(float(v) for v in bounds_raw))
        else:
            raise TileFormatError("%s: unreadable bounds %r" % (directory, bounds_raw))

        known = {
            "tile_id",
            "width_px",
            "height_px",
            "bounds",
            "crs",
            "transform",
            "image",
            "image_filename",
            "features",
            "feature_filename",
            "ground_resolution_mpp",
            "terrain_alt_m",
            "extractor",
        }
        extractor = data.get("extractor")
        if extractor is not None and not isinstance(extractor, dict):
            raise TileFormatError(
                "%s: 'extractor' must be an object, got %r" % (directory, extractor)
            )
        return cls(
            tile_id=str(data.get("tile_id") or os.path.basename(directory.rstrip("/"))),
            directory=directory,
            width_px=int(data["width_px"]),
            height_px=int(data["height_px"]),
            bounds=bounds,
            crs=str(data.get("crs", "EPSG:4326")),
            transform=list(data["transform"]) if data.get("transform") else None,
            image_filename=str(
                data.get("image") or data.get("image_filename") or DEFAULT_IMAGE_FILENAME
            ),
            feature_filename=str(
                data.get("features")
                or data.get("feature_filename")
                or DEFAULT_FEATURE_FILENAME
            ),
            ground_resolution_mpp=(
                float(data["ground_resolution_mpp"])
                if data.get("ground_resolution_mpp") is not None
                else None
            ),
            terrain_alt_m=float(data.get("terrain_alt_m", 0.0)),
            extractor=dict(extractor) if extractor is not None else None,
            extra={k: v for k, v in data.items() if k not in known},
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "tile_id": self.tile_id,
            "image": self.image_filename,
            "features": self.feature_filename,
            "crs": self.crs,
            "bounds": {
                "west": self.bounds.west,
                "south": self.bounds.south,
                "east": self.bounds.east,
                "north": self.bounds.north,
            },
            "width_px": self.width_px,
            "height_px": self.height_px,
            "ground_resolution_mpp": self.ground_resolution_mpp,
            "terrain_alt_m": self.terrain_alt_m,
        }
        if self.transform:
            out["transform"] = list(self.transform)
        if self.extractor is not None:
            out["extractor"] = dict(self.extractor)
        out.update(self.extra)
        return out

    @property
    def extractor_fingerprint(self):
        """The stored fingerprint as an object, or ``None`` when absent.

        A malformed fingerprint is treated as absent rather than raising: the
        point of the record is to catch configuration drift, and a database
        should not become unloadable because one field of its provenance is
        unreadable.  The missing-fingerprint path already reports itself.
        """
        if not self.extractor:
            return None
        from ..features.fingerprint import ExtractorFingerprint  # noqa: PLC0415

        try:
            return ExtractorFingerprint.from_dict(self.extractor)
        except (ValueError, TypeError):
            return None

    @classmethod
    def load(cls, directory: str) -> "TileMetadata":
        path = os.path.join(directory, METADATA_FILENAME)
        if not os.path.exists(path):
            raise TileFormatError("%s: no %s" % (directory, METADATA_FILENAME))
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data, directory)

    def save(self) -> None:
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, METADATA_FILENAME)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


@dataclass
class TileFeatures:
    """Precomputed SuperPoint output for one tile.

    ``keypoints`` is ``(N, 2)`` float32 in tile pixel coordinates ``(col, row)``
    at *full tile resolution* - the builder rescales them if it ran SuperPoint
    on a downsampled copy, so consumers never have to think about it.
    ``descriptors`` is ``(N, D)`` float32, ``scores`` is ``(N,)`` float32.
    """

    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: Optional[np.ndarray] = None
    image_size: Optional[tuple] = None

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32).reshape(-1, 2)
        self.descriptors = np.asarray(self.descriptors, dtype=np.float32)
        if self.descriptors.ndim != 2:
            raise TileFormatError("descriptors must be 2-D, got shape %r" % (self.descriptors.shape,))
        if self.descriptors.shape[0] != self.keypoints.shape[0]:
            raise TileFormatError(
                "keypoint/descriptor count mismatch: %d vs %d"
                % (self.keypoints.shape[0], self.descriptors.shape[0])
            )
        if self.scores is not None:
            self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return int(self.keypoints.shape[0])

    def save(self, path: str) -> None:
        """Write features as a torch ``.pt`` when possible, ``.npz`` otherwise."""
        payload = {
            "keypoints": self.keypoints,
            "descriptors": self.descriptors,
        }
        if self.scores is not None:
            payload["scores"] = self.scores
        if self.image_size is not None:
            payload["image_size"] = np.asarray(self.image_size, dtype=np.float32)

        if path.endswith(".pt"):
            try:
                import torch  # noqa: PLC0415 - optional dependency
            except ImportError:
                path = os.path.splitext(path)[0] + ".npz"
            else:
                torch.save({k: torch.from_numpy(v) for k, v in payload.items()}, path)
                return
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "TileFeatures":
        if not os.path.exists(path):
            alternative = os.path.splitext(path)[0] + ".npz"
            if os.path.exists(alternative):
                path = alternative
            else:
                raise TileFormatError("missing tile features: %s" % path)

        if path.endswith(".npz"):
            with np.load(path) as data:
                return cls(
                    keypoints=data["keypoints"],
                    descriptors=data["descriptors"],
                    scores=data["scores"] if "scores" in data else None,
                    image_size=(
                        tuple(data["image_size"].tolist()) if "image_size" in data else None
                    ),
                )

        try:
            import torch  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise TileFormatError(
                "%s is a torch checkpoint but torch is not installed; rebuild the "
                "map database or install the 'vision' extra" % path
            ) from exc
        # Tile features are data produced by our own builder, never third-party
        # pickles, so weights_only keeps the load strictly to tensors.
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 1.13 has no weights_only
            data = torch.load(path, map_location="cpu")

        def to_numpy(value):
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

        image_size = data.get("image_size")
        return cls(
            keypoints=to_numpy(data["keypoints"]),
            descriptors=to_numpy(data["descriptors"]),
            scores=to_numpy(data["scores"]) if "scores" in data else None,
            image_size=tuple(to_numpy(image_size).tolist()) if image_size is not None else None,
        )


class MapTile:
    """A tile: metadata, georeferencing and lazily loaded features."""

    def __init__(self, metadata: TileMetadata) -> None:
        self.metadata = metadata
        self.georeference = Georeference(
            width_px=metadata.width_px,
            height_px=metadata.height_px,
            bounds=metadata.bounds,
            crs=metadata.crs,
            transform=metadata.transform,
        )
        self._features: Optional[TileFeatures] = None

    @property
    def tile_id(self) -> str:
        return self.metadata.tile_id

    @property
    def bounds(self) -> BoundingBox:
        return self.metadata.bounds

    @property
    def center(self) -> GeoPoint:
        return self.metadata.bounds.center

    @property
    def ground_resolution_mpp(self) -> float:
        if self.metadata.ground_resolution_mpp:
            return self.metadata.ground_resolution_mpp
        return self.georeference.ground_resolution_mpp

    @property
    def terrain_alt_m(self) -> float:
        return self.metadata.terrain_alt_m

    def features(self) -> TileFeatures:
        """Load (and memoise) the precomputed SuperPoint features."""
        if self._features is None:
            self._features = TileFeatures.load(self.metadata.feature_path)
        return self._features

    def release(self) -> None:
        """Drop cached features to free RAM."""
        self._features = None

    @property
    def features_loaded(self) -> bool:
        return self._features is not None

    def pixel_to_geo(self, col: float, row: float) -> GeoPoint:
        return self.georeference.pixel_to_geo(col, row)

    def geo_to_pixel(self, point: GeoPoint):
        return self.georeference.geo_to_pixel(point)

    def __repr__(self) -> str:
        return "MapTile(%s, %.5f..%.5f lon, %.5f..%.5f lat)" % (
            self.tile_id,
            self.bounds.west,
            self.bounds.east,
            self.bounds.south,
            self.bounds.north,
        )
