"""Stand-ins for SuperPoint, LightGlue and the MAVLink transport.

These let the full pipeline run without torch, without a camera and without an
autopilot, which is what makes an end-to-end test possible in CI.  They are
intentionally *simple* rather than clever: the fake matcher returns exactly the
correspondences the fake extractor planted, so any geometric error the test
detects belongs to the code under test rather than to the fake.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from avinavi.features.lightglue_matcher import MatchResult
from avinavi.mapdb.tile import TileFeatures


class ScriptedExtractor:
    """Returns pre-planted keypoints for whatever image it is given."""

    def __init__(self, keypoints: np.ndarray, descriptor_dim: int = 8) -> None:
        self.keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
        # Descriptors are distinct per keypoint index so the fake matcher can
        # pair them up unambiguously.
        self.descriptors = np.eye(
            len(self.keypoints), descriptor_dim, dtype=np.float32
        )
        self.calls = 0

    def extract(self, image, resize_max_side=None) -> TileFeatures:
        self.calls += 1
        return TileFeatures(
            keypoints=self.keypoints.copy(),
            descriptors=self.descriptors.copy(),
            image_size=(float(image.shape[1]), float(image.shape[0])),
        )

    def warmup(self) -> None:
        pass


class ScriptedMatcher:
    """Pairs frame keypoint *i* with tile keypoint *i*, up to ``num_matches``."""

    def __init__(self, num_matches: Optional[int] = None, score: float = 0.9) -> None:
        self.num_matches = num_matches
        self.score = score
        self.calls = 0

    def match(self, frame_features, tile_features, tile_id: str = "") -> MatchResult:
        self.calls += 1
        count = min(len(frame_features), len(tile_features))
        if self.num_matches is not None:
            count = min(count, self.num_matches)
        return MatchResult(
            tile_id=tile_id,
            frame_points=frame_features.keypoints[:count].copy(),
            tile_points=tile_features.keypoints[:count].copy(),
            scores=np.full(count, self.score, dtype=np.float32),
        )

    def match_many(self, frame_features, tile_features):
        return [self.match(frame_features, f, tile_id=t) for t, f in tile_features]

    def warmup(self) -> None:
        pass


class NoMatchMatcher(ScriptedMatcher):
    """A matcher that never finds anything, for failure-path tests."""

    def match(self, frame_features, tile_features, tile_id: str = "") -> MatchResult:
        self.calls += 1
        return MatchResult(
            tile_id=tile_id,
            frame_points=np.zeros((0, 2), dtype=np.float32),
            tile_points=np.zeros((0, 2), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
        )


class LoopbackMavlink:
    """Captures outgoing MAVLink messages and decodes them back.

    Messages are encoded by the real pymavlink codec and immediately decoded,
    so a test asserting on ``frame_id`` or a covariance index is asserting on
    bytes that would genuinely have gone down the wire.
    """

    def __init__(self, system_id: int = 1, component_id: int = 197) -> None:
        from pymavlink.dialects.v20 import ardupilotmega

        self._module = ardupilotmega
        self.messages: List = []

        outer = self

        class _CapturingFile:
            def write(self, data):
                outer._decode(data)

        self._decoder = ardupilotmega.MAVLink(None)
        self.mav = ardupilotmega.MAVLink(
            _CapturingFile(), srcSystem=system_id, srcComponent=component_id
        )

    def _decode(self, data: bytes) -> None:
        for message in self._decoder.parse_buffer(data) or []:
            if message.get_type() != "BAD_DATA":
                self.messages.append(message)

    def of_type(self, message_type: str) -> List:
        return [m for m in self.messages if m.get_type() == message_type]

    def last(self, message_type: str):
        found = self.of_type(message_type)
        return found[-1] if found else None

    def close(self) -> None:
        pass


class FakeConnection:
    """A :class:`~avinavi.mavlink.connection.MavlinkConnection` work-alike."""

    def __init__(self, connected: bool = True) -> None:
        from pymavlink import mavutil

        import threading

        from avinavi.mavlink.connection import VehicleState

        self._loopback = LoopbackMavlink()
        self._mavutil = mavutil
        self._lock = threading.RLock()
        self.state = VehicleState()
        self._connected = connected
        self.sent_counts = {}
        self.received_counts = {}
        self.system_id = 1
        self.component_id = 197
        self.handlers = {}

    @property
    def connected(self) -> bool:
        return self._connected

    def set_connected(self, value: bool) -> None:
        self._connected = value

    @property
    def master(self):
        return self._loopback

    @property
    def mavutil(self):
        return self._mavutil

    @property
    def lock(self):
        return self._lock

    def note_sent(self, message_type: str) -> None:
        self.sent_counts[message_type] = self.sent_counts.get(message_type, 0) + 1

    def add_handler(self, message_type: str, handler) -> None:
        self.handlers.setdefault(message_type, []).append(handler)

    def dispatch(self, message) -> None:
        """Feed a message in as the receive thread would."""
        for handler in self.handlers.get(message.get_type(), ()):
            handler(message)

    def send_command_ack(
        self, command: int, result: int, target_system: int = 0,
        target_component: int = 0,
    ) -> None:
        with self._lock:
            self._loopback.mav.command_ack_send(
                int(command), int(result), 0, 0,
                int(target_system), int(target_component),
            )
        self.note_sent("COMMAND_ACK")

    def send_command_long(self, command: int, **params) -> None:
        with self._lock:
            self._loopback.mav.command_long_send(
                self.system_id, self.component_id, int(command), 0,
                float(params.get("param1", 0.0)), float(params.get("param2", 0.0)),
                float(params.get("param3", 0.0)), float(params.get("param4", 0.0)),
                float(params.get("param5", 0.0)), float(params.get("param6", 0.0)),
                float(params.get("param7", 0.0)),
            )
        self.note_sent("COMMAND_LONG")

    def request_message_interval(self, message_id: int, interval_us: float) -> None:
        self.send_command_long(511, param1=float(message_id), param2=float(interval_us))

    # -- test helpers -------------------------------------------------------
    @property
    def messages(self):
        return self._loopback.messages

    def of_type(self, message_type: str):
        return self._loopback.of_type(message_type)

    def last(self, message_type: str):
        return self._loopback.last(message_type)


def planar_correspondences(
    tile_points: np.ndarray, homography: np.ndarray, noise_px: float = 0.0, seed: int = 0
) -> np.ndarray:
    """Project tile points through a homography to make synthetic frame points."""
    points = np.asarray(tile_points, dtype=np.float64)
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    projected = homogeneous @ np.asarray(homography, dtype=np.float64).T
    result = projected[:, :2] / projected[:, 2:3]
    if noise_px > 0:
        rng = np.random.default_rng(seed)
        result = result + rng.normal(0.0, noise_px, size=result.shape)
    return result.astype(np.float32)


def similarity_homography(
    scale: float, rotation_rad: float, translation: Tuple[float, float]
) -> np.ndarray:
    """A pure similarity transform as a 3x3 homography."""
    cos_r, sin_r = np.cos(rotation_rad), np.sin(rotation_rad)
    matrix = np.eye(3)
    matrix[:2, :2] = scale * np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    matrix[:2, 2] = translation
    return matrix


class Clock:
    """A monotonic clock the test drives.

    Dead reckoning's limits are measured against a clock rather than against a
    sum of optical flow's own timestamps, because a component that has failed
    stops emitting timestamps and every limit built on them silently stops
    applying - which is exactly how a frozen position came to be published as
    a confident current one. Tests therefore have to advance time explicitly,
    and this makes that possible without sleeping.
    """

    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> "Clock":
        self.t += float(seconds)
        return self
