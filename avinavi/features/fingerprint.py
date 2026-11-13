"""What built a tile's features, recorded so a mismatch cannot pass silently.

The problem
-----------
Tile descriptors are written **offline, once**, and compared against live frame
descriptors **in the air, every cycle**.  Nothing connects the two beyond the
hope that the configuration did not move in between, and when it does move the
failure is silent: the system boots, the camera streams, the log looks healthy,
and there are simply no matches.  That has cost this project twice, once for
fifteen points of lock rate (see ``docs/matching-robustness.md``).

So the builder writes a fingerprint of the extractor into every tile's
``metadata.json``, and :func:`compare` is run at startup against the extractor
the flight configuration would build.

Two kinds of difference, deliberately kept apart
------------------------------------------------
**Incompatible.** The descriptors are not in the same space at all - a different
backend, a different classical detector, a different descriptor width or a float
descriptor compared against a binary one.  Matching cannot work, so the right
behaviour is to refuse to start.

**Degraded.** The descriptors still compare, there are just fewer or coarser
ones than intended.  Refusing here would be wrong, because the offline path
*intentionally* differs from the flight path on exactly these settings:
``avinavi extract-features`` gives a map tile four times the keypoint budget of
a live frame (a tile covers far more ground) and runs at full tile resolution
while the Pi downscales to ``superpoint.resize_max_side``.  Both are correct.
So these are reported, never fatal.

The single most useful recorded number is neither of those settings: it is
``analysis_max_side``, the longest side the detector *actually saw*.  A tile
raster stored at 1024 px but analysed at 640 is the fifteen-point bug, and it is
visible in that one number without having to reason about which configuration
key was in force at build time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

#: Descriptor width per classical detector, for the flight-side fingerprint
#: where no descriptors exist yet to measure.  ``None`` means "do not claim to
#: know", which makes the comparison skip the field rather than invent a
#: mismatch.
_DESCRIPTOR_DIMS: Dict[str, Optional[int]] = {
    "superpoint": 256,
    "sift": 128,
    "akaze": 61,
    "orb": 32,
}

#: Detectors whose descriptors are float vectors compared by L2 distance.  All
#: others are binary strings compared by Hamming distance, and the two can
#: never be matched against each other.
_FLOAT_DESCRIPTOR_DETECTORS = ("superpoint", "sift")

#: Fields that must agree or matching is impossible.
IDENTITY_FIELDS = ("backend", "detector", "descriptor_dim", "descriptor_kind")

#: A tile analysed at less than this fraction of its stored raster resolution is
#: reported: its features are coarser than the imagery it was built from, which
#: is almost always an accident rather than a choice.
MIN_ANALYSIS_FRACTION = 0.9


@dataclass(frozen=True)
class ExtractorFingerprint:
    """Identity and settings of the extractor that produced a set of features.

    ``descriptor_dim`` and ``analysis_max_side`` are ``None`` when unknown - on
    the flight side nothing has been extracted yet, and an older database has no
    fingerprint at all.  Unknown fields are skipped by :func:`compare` instead
    of being treated as a difference.
    """

    backend: str
    detector: str
    descriptor_dim: Optional[int] = None
    descriptor_kind: Optional[str] = None
    max_num_keypoints: Optional[int] = None
    resize_max_side: Optional[int] = None
    #: Longest side of the image actually fed to the detector.  Build time only.
    analysis_max_side: Optional[int] = None
    #: Longest side of the stored raster the features describe.  Build time only.
    image_max_side: Optional[int] = None

    @classmethod
    def from_extractor(cls, extractor) -> "ExtractorFingerprint":
        """Fingerprint a live extractor object, without running it."""
        detector = str(getattr(extractor, "detector_name", "superpoint")).lower()
        backend = "opencv" if hasattr(extractor, "detector_name") else "superpoint"
        return cls(
            backend=backend,
            detector=detector,
            descriptor_dim=_DESCRIPTOR_DIMS.get(detector),
            descriptor_kind=descriptor_kind_for(detector),
            max_num_keypoints=_as_optional_int(
                getattr(extractor, "max_num_keypoints", None)
            ),
            resize_max_side=_as_optional_int(
                getattr(extractor, "resize_max_side", None)
            ),
        )

    @classmethod
    def from_config(cls, config) -> "ExtractorFingerprint":
        """Fingerprint the extractor the flight configuration describes.

        Mirrors :meth:`avinavi.localization.engine.LocalizationEngine._build_extractor`
        without constructing anything, so a startup check costs nothing and
        cannot fail on a machine without torch.
        """
        backend = str(config.features.backend).lower()
        detector = (
            str(config.features.detector).lower() if backend == "opencv" else "superpoint"
        )
        return cls(
            backend=backend,
            detector=detector,
            descriptor_dim=_DESCRIPTOR_DIMS.get(detector),
            descriptor_kind=descriptor_kind_for(detector),
            max_num_keypoints=_as_optional_int(config.superpoint.max_num_keypoints),
            resize_max_side=_as_optional_int(config.superpoint.resize_max_side),
        )

    def with_measurements(
        self,
        descriptor_dim: Optional[int] = None,
        analysis_max_side: Optional[int] = None,
        image_max_side: Optional[int] = None,
    ) -> "ExtractorFingerprint":
        """Return a copy carrying what was observed while extracting.

        The builder knows all three for real - the descriptor width from the
        array it just wrote, and both sizes from the image it fed in - so the
        stored fingerprint never has to rely on the static table above.
        """
        return ExtractorFingerprint(
            backend=self.backend,
            detector=self.detector,
            descriptor_dim=(
                _as_optional_int(descriptor_dim)
                if descriptor_dim is not None
                else self.descriptor_dim
            ),
            descriptor_kind=self.descriptor_kind,
            max_num_keypoints=self.max_num_keypoints,
            resize_max_side=self.resize_max_side,
            analysis_max_side=(
                _as_optional_int(analysis_max_side)
                if analysis_max_side is not None
                else self.analysis_max_side
            ),
            image_max_side=(
                _as_optional_int(image_max_side)
                if image_max_side is not None
                else self.image_max_side
            ),
        )

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Only the fields that are known, so the JSON stays readable."""
        out: Dict[str, Any] = {"backend": self.backend, "detector": self.detector}
        for key in (
            "descriptor_dim",
            "descriptor_kind",
            "max_num_keypoints",
            "resize_max_side",
            "analysis_max_side",
            "image_max_side",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractorFingerprint":
        if not isinstance(data, dict):
            raise ValueError("an extractor fingerprint must be an object, got %r" % (data,))
        detector = str(data.get("detector") or "superpoint").lower()
        backend = str(data.get("backend") or "superpoint").lower()
        kind = data.get("descriptor_kind")
        return cls(
            backend=backend,
            detector=detector,
            descriptor_dim=_as_optional_int(data.get("descriptor_dim")),
            descriptor_kind=str(kind).lower() if kind is not None else None,
            max_num_keypoints=_as_optional_int(data.get("max_num_keypoints")),
            resize_max_side=_as_optional_int(data.get("resize_max_side")),
            analysis_max_side=_as_optional_int(data.get("analysis_max_side")),
            image_max_side=_as_optional_int(data.get("image_max_side")),
        )

    def describe(self) -> str:
        """One line for a log or ``avinavi map-info``."""
        parts = ["%s/%s" % (self.backend, self.detector)]
        if self.descriptor_dim is not None:
            parts.append("%d-d" % self.descriptor_dim)
        if self.descriptor_kind is not None:
            parts.append(self.descriptor_kind)
        if self.max_num_keypoints is not None:
            parts.append("<=%d kp" % self.max_num_keypoints)
        if self.analysis_max_side is not None:
            parts.append("analysed at %d px" % self.analysis_max_side)
        return ", ".join(parts)

    @property
    def identity(self) -> tuple:
        """The part that must agree for matching to be possible at all."""
        return tuple(getattr(self, field) for field in IDENTITY_FIELDS)


@dataclass(frozen=True)
class FingerprintComparison:
    """The outcome of comparing a map database against a flight configuration.

    ``incompatible`` is fatal; ``degraded`` is worth saying out loud and nothing
    more.  Keeping them in separate lists is the whole point - collapsing them
    would either ground a correct setup or let the silent failure back in.
    """

    incompatible: List[str]
    degraded: List[str]

    @property
    def compatible(self) -> bool:
        return not self.incompatible

    @property
    def clean(self) -> bool:
        return not self.incompatible and not self.degraded


def descriptor_kind_for(detector: str) -> str:
    """``"float"`` (L2) or ``"binary"`` (Hamming) for a detector name."""
    return "float" if str(detector).lower() in _FLOAT_DESCRIPTOR_DETECTORS else "binary"


def compare(
    stored: Optional[ExtractorFingerprint],
    current: ExtractorFingerprint,
) -> FingerprintComparison:
    """Compare the fingerprint a database carries against the live one.

    ``stored`` of ``None`` means the database predates fingerprinting.  That is
    reported as degraded rather than incompatible: refusing would break every
    database built before this existed, and the honest statement is "this cannot
    be checked", not "this is wrong".
    """
    if stored is None:
        return FingerprintComparison(
            incompatible=[],
            degraded=[
                "the map database carries no extractor fingerprint, so it cannot "
                "be checked against this configuration - rebuild it with "
                "'avinavi extract-features' to enable the check"
            ],
        )

    incompatible: List[str] = []
    degraded: List[str] = []

    if stored.backend != current.backend:
        incompatible.append(
            "feature backend: tiles were built with %r, this configuration uses %r"
            % (stored.backend, current.backend)
        )
    if stored.detector != current.detector:
        incompatible.append(
            "detector: tiles were built with %r, this configuration uses %r"
            % (stored.detector, current.detector)
        )
    if (
        stored.descriptor_dim is not None
        and current.descriptor_dim is not None
        and stored.descriptor_dim != current.descriptor_dim
    ):
        incompatible.append(
            "descriptor width: tiles hold %d-dimensional descriptors, this "
            "configuration produces %d" % (stored.descriptor_dim, current.descriptor_dim)
        )
    if (
        stored.descriptor_kind is not None
        and current.descriptor_kind is not None
        and stored.descriptor_kind != current.descriptor_kind
    ):
        incompatible.append(
            "descriptor type: tiles hold %s descriptors, this configuration "
            "produces %s - these cannot be compared to each other"
            % (stored.descriptor_kind, current.descriptor_kind)
        )

    degraded.extend(_resolution_loss(stored))

    if (
        stored.max_num_keypoints is not None
        and current.max_num_keypoints is not None
        and stored.max_num_keypoints < current.max_num_keypoints
    ):
        degraded.append(
            "tiles hold at most %d keypoints while a live frame may produce %d; a "
            "map tile covers far more ground than one frame, so it normally "
            "deserves the larger budget"
            % (stored.max_num_keypoints, current.max_num_keypoints)
        )

    return FingerprintComparison(incompatible=incompatible, degraded=degraded)


def _resolution_loss(stored: ExtractorFingerprint) -> List[str]:
    """Report a database whose features are coarser than its own rasters.

    This is the check that would have caught the fifteen-point regression: the
    tiles were stored at full resolution and analysed at 640 px, so every map
    descriptor described a blurrier world than the imagery on disk.
    """
    analysed = stored.analysis_max_side
    image = stored.image_max_side
    if analysed is None or image is None or image <= 0:
        return []
    fraction = analysed / float(image)
    if fraction >= MIN_ANALYSIS_FRACTION:
        return []
    return [
        "tile features were extracted at %d px from rasters stored at %d px "
        "(%.0f%% of the available resolution), so the map describes a coarser "
        "world than the imagery it was built from - rebuild with "
        "'avinavi extract-features --force --resize-max-side 0' to use the "
        "full tile" % (analysed, image, 100.0 * fraction)
    ]


def compare_database(
    fingerprints: List[Optional[ExtractorFingerprint]],
    current: ExtractorFingerprint,
) -> FingerprintComparison:
    """Compare *every* distinct build in a database against the live extractor.

    Reducing the database to one representative fingerprint first is not good
    enough, and a test caught it: in a database half re-extracted with a new
    detector, whichever group the representative came from decided the verdict,
    so a database with half its tiles unmatchable passed the check.  That is the
    exact failure this module exists to prevent, arriving by a different door -
    and it is worse than a total mismatch, because the aircraft would localize
    over part of the route and go blind over the rest.

    So each distinct build is compared on its own and the tile count is named,
    which also makes the message far more useful: "3 of 240 tiles" says a
    partial re-extraction, "240 of 240" says the configuration is simply wrong.
    """
    known = [fp for fp in fingerprints if fp is not None]
    total = len(fingerprints)
    if not known:
        return compare(None, current)

    groups: List[tuple] = []
    for fingerprint in known:
        for identity, members in groups:
            if identity == fingerprint.identity:
                members.append(fingerprint)
                break
        else:
            groups.append((fingerprint.identity, [fingerprint]))

    incompatible: List[str] = []
    degraded: List[str] = []
    for _, members in groups:
        result = compare(members[0], current)
        prefix = "%d of %d tiles - " % (len(members), total) if len(groups) > 1 else ""
        incompatible.extend(prefix + message for message in result.incompatible)

    if len(groups) > 1:
        degraded.append(
            "the tiles in this database were not all built the same way: %s"
            % "; ".join(sorted({fp.describe() for fp in known}))
        )
    if len(known) != total:
        degraded.append(
            "%d of %d tiles carry no extractor fingerprint, so part of this "
            "database cannot be checked" % (total - len(known), total)
        )

    # Quality notes come from the weakest tile, not a representative one: a
    # single tile described at reduced resolution is a real hole in the map.
    weakest = _weakest(known)
    for message in compare(weakest, current).degraded:
        if message not in degraded:
            degraded.append(message)

    return FingerprintComparison(incompatible=incompatible, degraded=degraded)


def consensus(
    fingerprints: List[Optional[ExtractorFingerprint]],
) -> tuple:
    """Reduce per-tile fingerprints to one for display, and name disagreements.

    For ``avinavi map-info`` and logging only.  The startup check uses
    :func:`compare_database`, which never reduces the database to a single
    fingerprint - see that function for why.

    Returns ``(fingerprint_or_None, conflicts)``.
    """
    known = [fp for fp in fingerprints if fp is not None]
    if not known:
        return None, []

    conflicts: List[str] = []
    if len({fp.identity for fp in known}) > 1:
        conflicts.append(
            "the tiles in this database were not all built the same way: %s"
            % "; ".join(sorted({fp.describe() for fp in known}))
        )
    if len(known) != len(fingerprints):
        conflicts.append(
            "%d of %d tiles carry no extractor fingerprint, so part of this "
            "database cannot be checked"
            % (len(fingerprints) - len(known), len(fingerprints))
        )
    return _weakest(known), conflicts


def _weakest(known: List[ExtractorFingerprint]) -> ExtractorFingerprint:
    """The tile least able to match: coarsest analysis, then smallest budget."""

    def rank(fingerprint: ExtractorFingerprint) -> tuple:
        if fingerprint.analysis_max_side and fingerprint.image_max_side:
            fraction = fingerprint.analysis_max_side / float(fingerprint.image_max_side)
        else:
            fraction = 1.0
        budget = (
            fingerprint.max_num_keypoints
            if fingerprint.max_num_keypoints is not None
            else 1 << 30
        )
        return (fraction, budget)

    return min(known, key=rank)


def _as_optional_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
