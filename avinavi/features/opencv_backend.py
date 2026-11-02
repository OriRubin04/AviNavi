"""A classical OpenCV feature backend, as a baseline and a fallback.

Two honest reasons this exists alongside SuperPoint + LightGlue:

1. **It is the baseline the learned pipeline must beat.** Claims about lock
   rate mean little without something to compare against, and SIFT with a ratio
   test is the strongest classical option. Note the direction of the handicap:
   SIFT is *designed* to be rotation and scale invariant, while SuperPoint
   descriptors are not. So when the simulation shows that pre-warping the frame
   helps SIFT, it is understating how much it helps SuperPoint.

2. **It runs anywhere.** Getting a working PyTorch onto a Raspberry Pi is not
   always quick, and a system that cannot run at all while that is sorted out is
   not much use. AKAZE on a Pi 5 handles a 640x480 frame in a few milliseconds,
   so this backend is a viable - if less capable - production path.

Interface-compatible with :class:`~avinavi.features.superpoint.SuperPointExtractor`
and :class:`~avinavi.features.lightglue_matcher.LightGlueMatcher`, so the engine
takes either pair without knowing the difference.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from ..features.lightglue_matcher import MatchResult
from ..mapdb.tile import TileFeatures
from .resize import USE_CONFIGURED_RESIZE

LOG = logging.getLogger(__name__)


#: Detectors that produce float descriptors (L2 distance) vs binary (Hamming).
_FLOAT_DESCRIPTOR_DETECTORS = ("sift",)


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("the OpenCV feature backend needs OpenCV") from exc
    return cv2


class OpenCVExtractor:
    """SIFT / AKAZE / ORB keypoints and descriptors."""

    def __init__(
        self,
        detector: str = "sift",
        max_num_keypoints: int = 2048,
        resize_max_side: Optional[int] = 640,
        contrast_threshold: float = 0.02,
    ) -> None:
        self.detector_name = detector.lower()
        self.max_num_keypoints = int(max_num_keypoints)
        self.resize_max_side = resize_max_side
        self.contrast_threshold = float(contrast_threshold)
        self._detector = None

    @classmethod
    def from_config(cls, config) -> "OpenCVExtractor":
        return cls(
            detector=getattr(config, "detector", "sift"),
            max_num_keypoints=config.max_num_keypoints,
            resize_max_side=config.resize_max_side,
        )

    @property
    def uses_float_descriptors(self) -> bool:
        return self.detector_name in _FLOAT_DESCRIPTOR_DETECTORS

    def _ensure_detector(self):
        """Build the configured detector, or explain precisely why it cannot.

        The explanation matters because the available detectors **depend on the
        installed OpenCV**. ``cv2.AKAZE_create`` is absent from OpenCV 5.0, for
        instance, so a configuration that works on one machine raises a bare
        ``AttributeError: module 'cv2' has no attribute 'AKAZE_create'`` on
        another — which says nothing about the YAML key that caused it.
        """
        if self._detector is not None:
            return self._detector
        cv2 = _require_cv2()

        factories = {
            "sift": lambda: cv2.SIFT_create(
                nfeatures=self.max_num_keypoints,
                contrastThreshold=self.contrast_threshold,
            ),
            "akaze": lambda: cv2.AKAZE_create(),
            "orb": lambda: cv2.ORB_create(nfeatures=self.max_num_keypoints),
        }
        required_attribute = {"sift": "SIFT_create", "akaze": "AKAZE_create",
                              "orb": "ORB_create"}

        if self.detector_name not in factories:
            raise ValueError(
                "unknown features.detector %r; expected one of %s"
                % (self.detector_name, ", ".join(sorted(factories)))
            )
        if not hasattr(cv2, required_attribute[self.detector_name]):
            available = sorted(
                name for name, attribute in required_attribute.items()
                if hasattr(cv2, attribute)
            )
            raise RuntimeError(
                "features.detector is %r, but the installed OpenCV (%s) has no "
                "cv2.%s. Available here: %s. Note that the tile database stores "
                "descriptors from whichever detector built it, so switching "
                "means rebuilding it with 'avinavi extract-features'."
                % (
                    self.detector_name,
                    getattr(cv2, "__version__", "unknown"),
                    required_attribute[self.detector_name],
                    ", ".join(available) or "none",
                )
            )
        self._detector = factories[self.detector_name]()
        return self._detector

    def warmup(self) -> None:
        self._ensure_detector()

    def extract(
        self, image: np.ndarray, resize_max_side=USE_CONFIGURED_RESIZE
    ) -> TileFeatures:
        """Extract features, with keypoints in the input image's pixel grid.

        ``resize_max_side`` omitted uses the configured value; ``None`` means do
        not resize at all.  See :mod:`avinavi.features.resize`.
        """
        cv2 = _require_cv2()
        detector = self._ensure_detector()

        gray = _to_gray_u8(image)
        max_side = (
            self.resize_max_side
            if resize_max_side is USE_CONFIGURED_RESIZE
            else resize_max_side
        )
        scaled, scale = _resize_max_side(gray, max_side)

        keypoints, descriptors = detector.detectAndCompute(scaled, None)
        if descriptors is None or len(keypoints) == 0:
            return TileFeatures(
                keypoints=np.zeros((0, 2), dtype=np.float32),
                descriptors=np.zeros((0, 128), dtype=np.float32),
                image_size=(float(gray.shape[1]), float(gray.shape[0])),
            )

        # AKAZE and ORB ignore nfeatures/produce more than asked; cap by response.
        if len(keypoints) > self.max_num_keypoints:
            order = np.argsort([-kp.response for kp in keypoints])[: self.max_num_keypoints]
            keypoints = [keypoints[i] for i in order]
            descriptors = descriptors[order]

        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        scores = np.array([kp.response for kp in keypoints], dtype=np.float32)
        if scale != 1.0:
            points = points / scale

        return TileFeatures(
            keypoints=points,
            descriptors=np.ascontiguousarray(descriptors.astype(np.float32)),
            scores=scores,
            image_size=(float(gray.shape[1]), float(gray.shape[0])),
        )

    def extract_image_file(
        self, path: str, resize_max_side=USE_CONFIGURED_RESIZE
    ) -> TileFeatures:
        cv2 = _require_cv2()
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError("could not read image: %s" % path)
        return self.extract(image, resize_max_side=resize_max_side)


class RatioTestMatcher:
    """Nearest-neighbour matching with Lowe's ratio test and a mutual check.

    The ratio test is what makes classical matching usable: a descriptor whose
    best and second-best candidates are similarly close is ambiguous, and
    keeping it would feed RANSAC noise. The mutual (cross-check) consistency
    requirement on top of it removes most of what survives.
    """

    def __init__(
        self,
        ratio: float = 0.8,
        cross_check: bool = True,
        binary_descriptors: bool = False,
        max_matches: int = 2048,
    ) -> None:
        self.ratio = float(ratio)
        self.cross_check = bool(cross_check)
        self.binary_descriptors = bool(binary_descriptors)
        self.max_matches = int(max_matches)
        self._matcher = None

    @classmethod
    def for_extractor(cls, extractor: OpenCVExtractor, **kwargs) -> "RatioTestMatcher":
        return cls(binary_descriptors=not extractor.uses_float_descriptors, **kwargs)

    def _ensure_matcher(self):
        if self._matcher is not None:
            return self._matcher
        cv2 = _require_cv2()
        norm = cv2.NORM_HAMMING if self.binary_descriptors else cv2.NORM_L2
        # crossCheck must stay False here: knnMatch needs k=2, and the mutual
        # check is applied manually below so it composes with the ratio test.
        self._matcher = cv2.BFMatcher(norm, crossCheck=False)
        return self._matcher

    def warmup(self) -> None:
        self._ensure_matcher()

    def match(
        self,
        frame_features: TileFeatures,
        tile_features: TileFeatures,
        tile_id: str = "",
    ) -> MatchResult:
        matcher = self._ensure_matcher()

        if len(frame_features) < 2 or len(tile_features) < 2:
            return _empty(tile_id)

        query = _as_descriptors(frame_features.descriptors, self.binary_descriptors)
        train = _as_descriptors(tile_features.descriptors, self.binary_descriptors)
        if query.shape[1] != train.shape[1]:
            LOG.warning(
                "descriptor dimension mismatch (%d vs %d); was the tile database "
                "built with a different detector?",
                query.shape[1],
                train.shape[1],
            )
            return _empty(tile_id)

        forward = matcher.knnMatch(query, train, k=2)
        kept: List[Tuple[int, int, float]] = []
        for pair in forward:
            if len(pair) < 2:
                continue
            best, second = pair[0], pair[1]
            if second.distance <= 0:
                continue
            if best.distance < self.ratio * second.distance:
                # Score in [0, 1]: how much better the best match is.
                kept.append(
                    (best.queryIdx, best.trainIdx, 1.0 - best.distance / second.distance)
                )

        if not kept:
            return _empty(tile_id)

        if self.cross_check:
            reverse = matcher.knnMatch(train, query, k=1)
            best_back = {
                pair[0].queryIdx: pair[0].trainIdx for pair in reverse if len(pair) >= 1
            }
            kept = [
                (q, t, s) for (q, t, s) in kept if best_back.get(t, -1) == q
            ]
            if not kept:
                return _empty(tile_id)

        kept.sort(key=lambda item: -item[2])
        kept = kept[: self.max_matches]

        frame_indices = np.array([k[0] for k in kept], dtype=int)
        tile_indices = np.array([k[1] for k in kept], dtype=int)
        scores = np.array([k[2] for k in kept], dtype=np.float32)

        return MatchResult(
            tile_id=tile_id,
            frame_points=np.ascontiguousarray(
                frame_features.keypoints[frame_indices], dtype=np.float32
            ),
            tile_points=np.ascontiguousarray(
                tile_features.keypoints[tile_indices], dtype=np.float32
            ),
            scores=scores,
        )

    def match_many(self, frame_features, tile_features):
        results = [
            self.match(frame_features, features, tile_id=tile_id)
            for tile_id, features in tile_features
        ]
        results.sort(key=lambda r: (len(r), r.mean_score), reverse=True)
        return results


def _empty(tile_id: str) -> MatchResult:
    return MatchResult(
        tile_id=tile_id,
        frame_points=np.zeros((0, 2), dtype=np.float32),
        tile_points=np.zeros((0, 2), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
    )


def _as_descriptors(descriptors: np.ndarray, binary: bool) -> np.ndarray:
    """OpenCV's matchers are strict about dtype: uint8 for Hamming, float32 for L2."""
    if binary:
        return np.ascontiguousarray(descriptors.astype(np.uint8))
    return np.ascontiguousarray(descriptors.astype(np.float32))


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    array = np.asarray(image)
    if array.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if array.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        array = cv2.cvtColor(array, code)
    if array.dtype != np.uint8:
        maximum = float(np.nanmax(array)) if array.size else 1.0
        scale = 255.0 if maximum <= 1.5 else 255.0 / max(maximum, 1e-6)
        array = np.clip(np.nan_to_num(array) * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _resize_max_side(image: np.ndarray, max_side: Optional[int]):
    if not max_side or max_side <= 0:
        return image, 1.0
    cv2 = _require_cv2()
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized), scale
