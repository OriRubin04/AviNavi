"""SuperPoint keypoint detection and description.

Wraps the reference implementation shipped with `cvg/LightGlue
<https://github.com/cvg/LightGlue>`_ (``lightglue.SuperPoint``), which is the
same detector the matcher's weights were trained against.

Two call sites use this class and they have very different constraints:

* the **offline builder** runs it once per map tile on a workstation, at full
  tile resolution, with a generous keypoint budget;
* the **flight loop** runs it on every live camera frame on a Raspberry Pi 5,
  where the image is downscaled to ``superpoint.resize_max_side`` first.

Both get keypoints back in *original image pixel coordinates*: the rescale is
undone here so no caller ever has to remember it.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

from ..mapdb.tile import TileFeatures

LOG = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    """Resolve ``"auto"`` to cuda when available, otherwise cpu."""
    if requested and requested != "auto":
        return requested
    try:
        import torch  # noqa: PLC0415 - optional dependency
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SuperPointExtractor:
    """SuperPoint wrapper with a numpy in / numpy out interface.

    The model is loaded on first use rather than in ``__init__`` so that
    constructing the object (for example while parsing configuration) is cheap
    and cannot fail on a machine without torch.
    """

    def __init__(
        self,
        max_num_keypoints: int = 1024,
        detection_threshold: float = 0.0005,
        nms_radius: int = 4,
        device: str = "auto",
        resize_max_side: Optional[int] = 640,
    ) -> None:
        self.max_num_keypoints = int(max_num_keypoints)
        self.detection_threshold = float(detection_threshold)
        self.nms_radius = int(nms_radius)
        self.device = resolve_device(device)
        self.resize_max_side = resize_max_side
        self._model = None
        self._torch = None
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config) -> "SuperPointExtractor":
        return cls(
            max_num_keypoints=config.max_num_keypoints,
            detection_threshold=config.detection_threshold,
            nms_radius=config.nms_radius,
            device=config.device,
            resize_max_side=config.resize_max_side,
        )

    # -- model lifecycle ----------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import torch  # noqa: PLC0415 - optional dependency
                from lightglue import SuperPoint  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RuntimeError(
                    "SuperPoint needs torch and the lightglue package; install "
                    "the 'vision' extra (see requirements.txt)"
                ) from exc

            LOG.info("loading SuperPoint on %s", self.device)
            model = SuperPoint(
                max_num_keypoints=self.max_num_keypoints,
                detection_threshold=self.detection_threshold,
                nms_radius=self.nms_radius,
            )
            model = model.eval().to(self.device)
            # Inference only: gradients would waste memory and time on a Pi.
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._torch = torch
            self._model = model
        return self._model

    def warmup(self) -> None:
        """Load weights and run one dummy frame so the first real frame is fast."""
        self._ensure_model()
        dummy = np.zeros((240, 320), dtype=np.uint8)
        self.extract(dummy)

    # -- extraction ---------------------------------------------------------
    def extract(
        self, image: np.ndarray, resize_max_side: Optional[int] = None
    ) -> TileFeatures:
        """Extract features from a grayscale or BGR image.

        Returns keypoints in the coordinate system of the *input* image.
        """
        model = self._ensure_model()
        torch = self._torch

        gray = _to_grayscale_float(image)
        max_side = resize_max_side if resize_max_side is not None else self.resize_max_side
        scaled, scale = _resize_max_side(gray, max_side)

        tensor = torch.from_numpy(scaled).to(self.device)[None, None]
        with torch.inference_mode():
            output = model.extract(tensor)

        keypoints = output["keypoints"][0].detach().cpu().numpy().astype(np.float32)
        descriptors = output["descriptors"][0].detach().cpu().numpy().astype(np.float32)
        scores = output.get("keypoint_scores")
        scores = (
            scores[0].detach().cpu().numpy().astype(np.float32) if scores is not None else None
        )

        if scale != 1.0 and keypoints.size:
            # Undo the downscale so keypoints refer to the original image grid.
            keypoints = keypoints / scale

        return TileFeatures(
            keypoints=keypoints,
            descriptors=descriptors,
            scores=scores,
            image_size=(float(gray.shape[1]), float(gray.shape[0])),
        )

    def extract_image_file(self, path: str, resize_max_side: Optional[int] = None) -> TileFeatures:
        try:
            import cv2  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("reading image files needs OpenCV") from exc
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError("could not read image: %s" % path)
        return self.extract(image, resize_max_side=resize_max_side)


def _to_grayscale_float(image: np.ndarray) -> np.ndarray:
    """Any input image -> float32 grayscale in [0, 1]."""
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[:, :, :3]
        # OpenCV order is BGR; the luma weights below are applied accordingly.
        array = (
            0.114 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.299 * array[:, :, 2]
        )
    elif array.ndim != 2:
        raise ValueError("expected a 2-D or 3-D image, got shape %r" % (array.shape,))
    array = array.astype(np.float32)
    if array.max(initial=0.0) > 1.5:
        array = array / 255.0
    return np.ascontiguousarray(np.clip(array, 0.0, 1.0))


def _resize_max_side(image: np.ndarray, max_side: Optional[int]):
    """Downscale so the longest side is ``max_side``; returns (image, scale)."""
    if not max_side or max_side <= 0:
        return image, 1.0
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    try:
        import cv2  # noqa: PLC0415 - optional dependency
    except ImportError:  # pragma: no cover - fall back to a cheap decimation
        step = int(round(1.0 / scale))
        return np.ascontiguousarray(image[::step, ::step]), 1.0 / step
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized), scale
