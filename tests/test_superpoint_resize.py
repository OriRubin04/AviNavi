"""SuperPoint must see the image at resize_max_side, not lightglue's default 1024.

lightglue's ``SuperPoint.extract`` rescales its input to a 1024 px long side
unless told ``resize=None``.  That silently undid ``resize_max_side``: a frame
downscaled to 640 was upsampled back to 1024 before the network ran.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from avinavi.features.superpoint import SuperPointExtractor  # noqa: E402


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def extract(self, tensor, **conf):
        self.calls.append((tuple(tensor.shape), conf))
        return {
            "keypoints": torch.zeros((1, 1, 2)),
            "descriptors": torch.zeros((1, 1, 256)),
            "keypoint_scores": torch.zeros((1, 1)),
        }


def _extractor(resize_max_side):
    ex = SuperPointExtractor(device="cpu", resize_max_side=resize_max_side)
    ex._model = _RecordingModel()
    ex._torch = torch
    return ex


def test_lightglue_internal_resize_is_disabled():
    ex = _extractor(640)
    ex.extract(np.zeros((800, 1200), dtype=np.uint8))
    shape, conf = ex._model.calls[0]
    assert conf.get("resize", "missing") is None
    assert max(shape[-2:]) == 640


def test_full_resolution_reaches_the_network_unchanged():
    ex = _extractor(640)
    ex.extract(np.zeros((1300, 1500), dtype=np.uint8), resize_max_side=None)
    shape, conf = ex._model.calls[0]
    assert conf.get("resize", "missing") is None
    assert shape[-2:] == (1300, 1500)
