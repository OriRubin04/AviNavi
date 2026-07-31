"""LightGlue matching between a live camera frame and map tiles.

LightGlue is fed one pair at a time (live frame vs. one candidate tile).  The
live frame's features are extracted once per localization cycle and reused
across every candidate, which is where most of the saving comes from - the map
tile side is already precomputed on disk.

LightGlue normalises keypoints using the image size it is told about, so the
image sizes carried by :class:`~avinavi.mapdb.tile.TileFeatures` are passed
through rather than inferred.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..mapdb.tile import TileFeatures
from .superpoint import resolve_device

LOG = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Correspondences between a live frame and one map tile.

    ``frame_points`` and ``tile_points`` are ``(M, 2)`` arrays of pixel
    coordinates in the live image and the tile respectively, index-aligned.
    ``scores`` holds LightGlue's per-match confidence in ``[0, 1]``.
    """

    tile_id: str
    frame_points: np.ndarray
    tile_points: np.ndarray
    scores: np.ndarray

    def __len__(self) -> int:
        return int(self.frame_points.shape[0])

    @property
    def mean_score(self) -> float:
        return float(self.scores.mean()) if self.scores.size else 0.0


class LightGlueMatcher:
    """LightGlue wrapper operating on numpy feature arrays.

    Like the extractor, the network is loaded lazily so importing this module
    never requires torch.
    """

    def __init__(
        self,
        features: str = "superpoint",
        depth_confidence: float = 0.95,
        width_confidence: float = 0.99,
        filter_threshold: float = 0.1,
        device: str = "auto",
    ) -> None:
        self.features = features
        self.depth_confidence = depth_confidence
        self.width_confidence = width_confidence
        self.filter_threshold = filter_threshold
        self.device = resolve_device(device)
        self._model = None
        self._torch = None
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config) -> "LightGlueMatcher":
        return cls(
            features=config.features,
            depth_confidence=config.depth_confidence,
            width_confidence=config.width_confidence,
            filter_threshold=config.filter_threshold,
            device=config.device,
        )

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import torch  # noqa: PLC0415 - optional dependency
                from lightglue import LightGlue  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RuntimeError(
                    "LightGlue needs torch and the lightglue package; install "
                    "the 'vision' extra (see requirements.txt)"
                ) from exc

            LOG.info("loading LightGlue (%s) on %s", self.features, self.device)
            model = LightGlue(
                features=self.features,
                depth_confidence=self.depth_confidence,
                width_confidence=self.width_confidence,
                filter_threshold=self.filter_threshold,
            )
            model = model.eval().to(self.device)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._torch = torch
            self._model = model
        return self._model

    def warmup(self) -> None:
        """Load weights and run a dummy pair so the first real match is fast."""
        self._ensure_model()
        dummy = TileFeatures(
            keypoints=np.random.rand(32, 2).astype(np.float32) * 100.0,
            descriptors=np.random.rand(32, 256).astype(np.float32),
            image_size=(128.0, 128.0),
        )
        self.match(dummy, dummy, tile_id="warmup")

    def match(
        self,
        frame_features: TileFeatures,
        tile_features: TileFeatures,
        tile_id: str = "",
    ) -> MatchResult:
        """Match one live frame against one map tile."""
        model = self._ensure_model()
        torch = self._torch

        if len(frame_features) == 0 or len(tile_features) == 0:
            return _empty_match(tile_id)

        with torch.inference_mode():
            output = model(
                {
                    "image0": self._pack(frame_features, torch),
                    "image1": self._pack(tile_features, torch),
                }
            )
            matches = output["matches"][0].detach().cpu().numpy()
            scores = output.get("scores")
            if scores is None:
                scores = output.get("matching_scores0")
            scores = (
                scores[0].detach().cpu().numpy().astype(np.float32)
                if scores is not None
                else np.ones(len(matches), dtype=np.float32)
            )

        if matches.size == 0:
            return _empty_match(tile_id)

        frame_points = frame_features.keypoints[matches[:, 0]]
        tile_points = tile_features.keypoints[matches[:, 1]]
        if scores.shape[0] != frame_points.shape[0]:
            # Some LightGlue revisions return a per-keypoint score vector.
            scores = scores[matches[:, 0]] if scores.shape[0] >= matches.shape[0] else (
                np.ones(frame_points.shape[0], dtype=np.float32)
            )
        return MatchResult(
            tile_id=tile_id,
            frame_points=np.ascontiguousarray(frame_points, dtype=np.float32),
            tile_points=np.ascontiguousarray(tile_points, dtype=np.float32),
            scores=np.ascontiguousarray(scores, dtype=np.float32),
        )

    def match_many(
        self,
        frame_features: TileFeatures,
        tile_features: List[Tuple[str, TileFeatures]],
    ) -> List[MatchResult]:
        """Match the live frame against several tiles, best (most matches) first."""
        results = [
            self.match(frame_features, features, tile_id=tile_id)
            for tile_id, features in tile_features
        ]
        results.sort(key=lambda r: (len(r), r.mean_score), reverse=True)
        return results

    def _pack(self, features: TileFeatures, torch) -> dict:
        keypoints = torch.from_numpy(features.keypoints).to(self.device)[None]
        descriptors = torch.from_numpy(features.descriptors).to(self.device)[None]
        size = features.image_size
        if size is None:
            # Fall back to the keypoint extent; LightGlue only uses this to
            # normalise coordinates, so a tight bound is acceptable.
            extent = features.keypoints.max(axis=0) if len(features) else np.array([1.0, 1.0])
            size = (float(extent[0]) + 1.0, float(extent[1]) + 1.0)
        image_size = torch.tensor(
            [[float(size[0]), float(size[1])]], dtype=torch.float32, device=self.device
        )
        return {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "image_size": image_size,
        }


def _empty_match(tile_id: str) -> MatchResult:
    return MatchResult(
        tile_id=tile_id,
        frame_points=np.zeros((0, 2), dtype=np.float32),
        tile_points=np.zeros((0, 2), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
    )
