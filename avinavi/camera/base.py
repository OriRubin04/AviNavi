"""The camera source interface."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


class CameraError(RuntimeError):
    """The camera failed.  The navigation solution must be invalidated."""


@dataclass
class Frame:
    """One captured image and its timestamps."""

    image: np.ndarray
    #: ``time.monotonic()`` at capture - use this for rate and age arithmetic.
    timestamp_monotonic: float
    #: ``time.time()`` at capture - use this for MAVLink timestamps.
    timestamp_unix: float
    sequence: int = 0

    @property
    def size(self):
        """``(width, height)`` in pixels."""
        return (int(self.image.shape[1]), int(self.image.shape[0]))


class CameraSource(abc.ABC):
    """Base class for frame sources."""

    def __init__(self) -> None:
        self._sequence = 0
        self._opened = False

    @abc.abstractmethod
    def open(self) -> None:
        """Start the camera.  Raises :class:`CameraError` on failure."""

    @abc.abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next frame, or ``None`` when the source is exhausted.

        Raises :class:`CameraError` for an actual fault, which is a different
        thing from a replay source running out of images.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release the camera."""

    @property
    def opened(self) -> bool:
        return self._opened

    def _make_frame(self, image: np.ndarray, timestamp_monotonic: Optional[float] = None) -> Frame:
        self._sequence += 1
        return Frame(
            image=image,
            timestamp_monotonic=(
                timestamp_monotonic if timestamp_monotonic is not None else time.monotonic()
            ),
            timestamp_unix=time.time(),
            sequence=self._sequence,
        )

    def __enter__(self) -> "CameraSource":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
