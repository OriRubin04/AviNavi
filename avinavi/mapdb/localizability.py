"""Predicting, offline, which map tiles will not localize.

Some ground cannot be matched against, and the reasons are visible in the
basemap tile alone, before any flight:

* **Open water, snow, desert sand, dense uniform forest canopy.** There is
  nothing to detect. Few keypoints, and the ones there are come from noise.
* **Orchards, vineyards, ploughed fields, solar farms, car parks.** The opposite
  problem, and the dangerous one: *plenty* of keypoints, all of which look
  identical to one another. A matcher given a regular lattice will find a
  confident, geometrically consistent match one row off. That is a false fix,
  which is worse than no fix, because it is confident and the EKF has no way to
  reject it.
* **A single distinctive feature in an otherwise blank tile.** A lone building
  in a field gives keypoints that cluster in one corner, so the tile localizes
  only for frames that happen to see that corner.

Knowing this before the flight is worth a great deal more than discovering it
during one, and it is cheap: ``avinavi map-info`` can report it from the
precomputed features that are already on disk.

What the score is
-----------------
Three components, each measuring one of the failure modes above, combined into
``score`` in [0, 1]:

``density``
    Keypoints per megapixel against a target, saturating. Answers "is there
    anything here at all".

``distinctiveness``
    The fraction of this tile's keypoints whose descriptor has **no close rival
    elsewhere in the same tile**. This is the one that catches the dangerous
    case, because a repetitive lattice scores near zero while genuinely varied
    ground scores high. Computed exactly as a matcher's ratio test would, but
    against the tile itself.

``spread``
    How evenly keypoints cover the tile, as the fraction of a coarse grid that
    holds at least one. Answers "will this tile work for a frame anywhere over
    it, or only in one corner".

The combination is a weighted geometric mean, deliberately: a tile that fails
badly on **any** one of the three is unusable, and an arithmetic mean would let
a high density hide zero distinctiveness — which is precisely the lattice case.

Honesty about this score
------------------------
It is a **predictor, not a measurement**, and it is **not yet validated against
localization outcomes in this codebase**. What is established is narrower than
the score's presentation might suggest:

* The repetition measure separates repetitive ground from varied ground on real
  features, by a wide margin, including a *jittered* lattice standing in for a
  real orchard. That comparison is asserted in
  ``tests/test_localizability.py`` and will fail loudly if it stops holding.
* Whether ``score`` *ranks tiles in the order they actually localize* has not
  been measured. Doing so needs per-tile lock rates out of the simulator, which
  ``SimulationReport`` does not currently record. It is listed in
  ``docs/WORK-IN-PROGRESS.md``.

So: a **low score is a reliable warning** about the imagery, and the
``repetitive`` flag in particular identifies a mechanism that genuinely produces
confident wrong fixes. A **high score is not a promise** — it cannot know about
the weather on the day, the season the basemap was shot in, or the altitude
flown.

One measured caveat on resolution. Scoring the *worst region* of a tile rather
than its average matters a great deal: an orchard stamped over textured terrain
moved the tile-wide figure only from 0.89 to 0.78, because the surrounding
texture kept supplying distinctive keypoints, while the region-wise measure
separates the same tiles across 0.60 to 0.92. Even so, a large tile averages
over heterogeneous ground; smaller tiles give a finer-grained warning.

The thresholds in :data:`VERDICTS` are set from the mechanisms above rather than
fitted to data. Treat them as a triage ordering.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

#: Keypoints per megapixel at which ``density`` saturates.  A 1024x1024 tile
#: holds about one megapixel, and a few hundred well-spread keypoints there is
#: ample for matching, so the target is set at that order.
DENSITY_TARGET_PER_MEGAPIXEL = 400.0

#: Grid resolution for the spread measure.
SPREAD_GRID = 6

#: Cosine similarity above which two descriptors elsewhere in the same tile
#: count as the same patch.  Both SIFT and SuperPoint descriptors are
#: L2-normalised, so this is comparable across them, and it is set from measured
#: distributions rather than chosen: on real SIFT features, an identical-blob
#: lattice puts *every* keypoint's best far-away twin at 1.000, while mixed
#: terrain has a median of 0.797 and only 8% above 0.9.  0.92 sits in that gap.
#:
#: Measured scores at this threshold, on real SIFT features:
#:
#:   =============================  ==========  =======  ===========
#:   Scene                          repetition  score    verdict
#:   =============================  ==========  =======  ===========
#:   lattice of identical blobs     1.00        0.00     unusable
#:   lattice with jittered position,
#:   size and tone (a real orchard) 0.92        0.32     poor
#:   mixed terrain                  0.05        0.98     good
#:   near-featureless (water)       0.00        0.00     unusable
#:   =============================  ==========  =======  ===========
#:
#: The jittered case matters: it confirms the measure does not depend on the
#: repeated elements being *identical*, only on them being confusable, which is
#: what a real orchard or vineyard offers.
TWIN_SIMILARITY = 0.92

#: Score thresholds and what to do about them.
VERDICTS: Sequence[Tuple[float, str, str]] = (
    (0.55, "good", "should localize reliably"),
    (0.35, "fair", "usable; expect dropouts"),
    (0.20, "poor", "frequent dropouts likely — plan a route that avoids relying on this tile"),
    (0.0, "unusable", "do not rely on this tile; the imagery cannot support a match"),
)


@dataclass
class TileLocalizability:
    """Per-tile prediction of whether matching will work over this ground."""

    tile_id: str
    keypoints: int
    density: float
    distinctiveness: float
    spread: float
    score: float
    verdict: str
    advice: str
    #: Set when the failure mode is specifically repetition, which is the one
    #: that produces confident *wrong* fixes rather than no fix.
    repetitive: bool = False

    def as_dict(self) -> Dict:
        return {
            "tile_id": self.tile_id,
            "keypoints": self.keypoints,
            "density": round(self.density, 3),
            "distinctiveness": round(self.distinctiveness, 3),
            "spread": round(self.spread, 3),
            "score": round(self.score, 3),
            "verdict": self.verdict,
            "advice": self.advice,
            "repetitive": self.repetitive,
        }


def verdict_for(score: float) -> Tuple[str, str]:
    for threshold, verdict, advice in VERDICTS:
        if score >= threshold:
            return (verdict, advice)
    return (VERDICTS[-1][1], VERDICTS[-1][2])


def keypoint_density(
    keypoints: np.ndarray, image_size: Optional[Tuple[float, float]]
) -> float:
    """Keypoints per megapixel, normalised against the target and clipped."""
    if keypoints is None or len(keypoints) == 0:
        return 0.0
    if not image_size or image_size[0] <= 0 or image_size[1] <= 0:
        return 0.0
    megapixels = (image_size[0] * image_size[1]) / 1e6
    if megapixels <= 0:
        return 0.0
    per_megapixel = len(keypoints) / megapixels
    return float(min(1.0, per_megapixel / DENSITY_TARGET_PER_MEGAPIXEL))


def repetition(
    keypoints: Optional[np.ndarray],
    descriptors: Optional[np.ndarray],
    max_sample: int = 500,
    min_offset_px: float = 12.0,
    twin_similarity: float = TWIN_SIMILARITY,
) -> float:
    """How repetitive this tile is, in [0, 1].  Higher is worse.

    This measures the failure mode as directly as a single number can.  A
    matcher fails on an orchard not because the descriptors are poor but because
    the tile **matches itself somewhere else**: shift the frame by one row of
    trees and every correspondence is still geometrically consistent, so RANSAC
    confirms a position one row out and reports it confidently.

    So, for each keypoint: how similar is the most similar *other* keypoint that
    is at least ``min_offset_px`` away?  If that twin is near-identical, this
    keypoint cannot be used to tell one part of the tile from another.  The
    score is the fraction of keypoints with such a twin.

    The spatial exclusion is essential, not a refinement — feature detectors
    emit several keypoints at one location (different scales and orientations),
    and those are duplicates of one piece of structure rather than evidence of
    repetition.

    Two approaches that **do not** work, both tried:

    * A within-tile Lowe ratio test.  In a lattice every keypoint has many
      near-identical rivals, so its two nearest are both close; on varied ground
      both are far.  The *ratio* is near one in both cases.
    * Clustering the spatial offsets to the nearest twin, looking for a common
      lattice vector.  In a lattice of near-identical blobs a keypoint's best
      twin is at an essentially arbitrary one of the many lattice sites, so the
      offsets spread across every lattice vector instead of concentrating on
      one.  Measured 0.048 on a perfect lattice — useless.

    Measured values for the chosen statistic, on real SIFT features
    (``tests/test_localizability.py`` holds the comparison):

    ========================  ================  ==========
    Scene                     median best-twin  this score
    ========================  ================  ==========
    identical-blob lattice    1.000             1.00
    jittered lattice          -                 0.92
    mixed terrain             0.797             0.05
    ========================  ================  ==========
    """
    if keypoints is None or descriptors is None:
        return 0.0
    points = np.asarray(keypoints, dtype=np.float64)
    data = np.asarray(descriptors, dtype=np.float32)
    if points.ndim != 2 or data.ndim != 2 or len(points) != len(data) or len(points) < 8:
        return 0.0

    confusable, _points = _confusable_keypoints(
        points, data, max_sample, min_offset_px, twin_similarity
    )
    if confusable is None:
        return 0.0
    return float(np.count_nonzero(confusable) / len(confusable))


def _confusable_keypoints(
    points: np.ndarray,
    data: np.ndarray,
    max_sample: int,
    min_offset_px: float,
    twin_similarity: float,
):
    """Per-keypoint: does a near-identical twin exist elsewhere in the tile?

    Returns ``(boolean array, the points they correspond to)``, or
    ``(None, None)`` when there is not enough to judge.
    """
    if len(data) > max_sample:
        # Deterministic: the same tile must always score the same.
        rng = np.random.default_rng(0)
        chosen = rng.choice(len(data), size=max_sample, replace=False)
        points = points[chosen]
        data = data[chosen]

    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = data / norms
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)

    # Only a twin somewhere *else* in the tile counts.
    separation = np.linalg.norm(points[:, None, :2] - points[None, :, :2], axis=2)
    similarity[separation < min_offset_px] = -np.inf

    best_twin = similarity.max(axis=1)
    usable = np.isfinite(best_twin)
    if usable.sum() < 8:
        # Every keypoint sits in one small cluster: not repetition, and the
        # spread component is the one that should penalise it.
        return (None, None)
    return (best_twin[usable] > twin_similarity, points[usable])


def worst_region_repetition(
    keypoints: Optional[np.ndarray],
    descriptors: Optional[np.ndarray],
    image_size: Optional[Tuple[float, float]],
    grid: int = 3,
    min_per_region: int = 20,
    max_sample: int = 500,
    min_offset_px: float = 12.0,
    twin_similarity: float = TWIN_SIMILARITY,
) -> float:
    """Repetition in the *worst* part of the tile, not averaged over it.

    The tile-wide average is misleading, and measurably so: an orchard stamped
    over textured terrain leaves the surrounding texture supplying plenty of
    distinctive keypoints, so the average barely moves (0.78 against 0.89 for a
    clean tile) even though frames flown over the orchard itself will mismatch.

    What the operator needs to know is whether there is *any* ground in this
    tile that will mis-localize, because the aircraft will fly over part of a
    tile, not over its average. So the keypoints are bucketed into a coarse
    grid and each region is scored against the whole tile; the worst region is
    what gets reported.

    Regions holding fewer than ``min_per_region`` keypoints are skipped: a
    handful of points cannot support a fraction, and the ``spread`` component
    is what should penalise an empty region.
    """
    if keypoints is None or descriptors is None or not image_size:
        return 0.0
    if image_size[0] <= 0 or image_size[1] <= 0:
        return 0.0
    points = np.asarray(keypoints, dtype=np.float64)
    data = np.asarray(descriptors, dtype=np.float32)
    if points.ndim != 2 or data.ndim != 2 or len(points) != len(data) or len(points) < 8:
        return 0.0

    confusable, kept = _confusable_keypoints(
        points, data, max_sample, min_offset_px, twin_similarity
    )
    if confusable is None:
        return 0.0

    columns = np.clip((kept[:, 0] / image_size[0] * grid).astype(int), 0, grid - 1)
    rows = np.clip((kept[:, 1] / image_size[1] * grid).astype(int), 0, grid - 1)

    worst = 0.0
    for row in range(grid):
        for column in range(grid):
            inside = (rows == row) & (columns == column)
            if inside.sum() < min_per_region:
                continue
            worst = max(worst, float(np.count_nonzero(confusable[inside]) / inside.sum()))
    # Never report less than the tile-wide figure: a tile whose keypoints are
    # too sparse to bucket still has whatever repetition it has.
    overall = float(np.count_nonzero(confusable) / len(confusable))
    return max(worst, overall)


def descriptor_distinctiveness(
    keypoints: Optional[np.ndarray],
    descriptors: Optional[np.ndarray],
    image_size: Optional[Tuple[float, float]] = None,
) -> float:
    """``1 - repetition``: the share of keypoints with no twin elsewhere.

    Uses the worst *region* of the tile when an image size is available, for
    the reason given in :func:`worst_region_repetition`: the aircraft flies
    over part of a tile, not over its average.
    """
    if image_size:
        worst = worst_region_repetition(keypoints, descriptors, image_size)
    else:
        worst = repetition(keypoints, descriptors)
    return float(max(0.0, 1.0 - worst))


def keypoint_spread(
    keypoints: Optional[np.ndarray],
    image_size: Optional[Tuple[float, float]],
    grid: int = SPREAD_GRID,
) -> float:
    """Fraction of a coarse grid containing at least one keypoint."""
    if keypoints is None or len(keypoints) == 0:
        return 0.0
    if not image_size or image_size[0] <= 0 or image_size[1] <= 0:
        return 0.0
    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        return 0.0
    columns = np.clip((points[:, 0] / image_size[0] * grid).astype(int), 0, grid - 1)
    rows = np.clip((points[:, 1] / image_size[1] * grid).astype(int), 0, grid - 1)
    occupied = len(set(zip(rows.tolist(), columns.tolist())))
    return float(occupied / (grid * grid))


def score_features(
    tile_id: str,
    keypoints: Optional[np.ndarray],
    descriptors: Optional[np.ndarray],
    image_size: Optional[Tuple[float, float]],
) -> TileLocalizability:
    """Score one tile from its precomputed features."""
    density = keypoint_density(keypoints, image_size)
    distinctiveness = descriptor_distinctiveness(keypoints, descriptors, image_size)
    spread = keypoint_spread(keypoints, image_size)

    # Weighted geometric mean. Deliberately not arithmetic: a tile that fails
    # badly on any one component is unusable, and an arithmetic mean would let a
    # high keypoint density hide zero distinctiveness - which is exactly the
    # repetitive-lattice case that produces confident wrong fixes.
    weights = (0.25, 0.45, 0.30)
    parts = (density, distinctiveness, spread)
    if min(parts) <= 0.0:
        score = 0.0
    else:
        score = float(
            math.exp(sum(w * math.log(p) for w, p in zip(weights, parts)))
        )

    verdict, advice = verdict_for(score)
    # Many keypoints but little distinctiveness is the signature of repetition,
    # and it needs saying separately: the failure is a *wrong* fix, not a
    # missing one, so "poor" undersells it.
    repetitive = density > 0.5 and distinctiveness < 0.25
    if repetitive:
        advice = (
            "repetitive structure: many keypoints but few distinctive ones. This "
            "is the dangerous case — a matcher can lock one row off and report it "
            "confidently. Raise geometry.min_inlier_ratio and "
            "confidence.min_confidence if the route must cross it."
        )

    return TileLocalizability(
        tile_id=tile_id,
        keypoints=0 if keypoints is None else len(keypoints),
        density=density,
        distinctiveness=distinctiveness,
        spread=spread,
        score=score,
        verdict=verdict,
        advice=advice,
        repetitive=repetitive,
    )


def score_database(database, limit: Optional[int] = None) -> List[TileLocalizability]:
    """Score every tile in a database, worst first.

    Worst first because that is the order the operator needs: the question is
    never "which of my tiles are good" but "is there anything on my route that
    will not work".
    """
    from .tile import TileFormatError

    results = []
    for tile in database:
        try:
            features = database.features_for(tile)
        except (TileFormatError, OSError, ValueError) as exc:
            LOG.warning("cannot score %s: %s", tile.tile_id, exc)
            continue
        size = features.image_size or (
            float(tile.metadata.width_px),
            float(tile.metadata.height_px),
        )
        results.append(
            score_features(
                tile_id=tile.tile_id,
                keypoints=features.keypoints,
                descriptors=features.descriptors,
                image_size=size,
            )
        )
    results.sort(key=lambda r: r.score)
    return results[:limit] if limit else results


def summarise_scores(scores: Sequence[TileLocalizability]) -> Dict:
    """Aggregate for ``avinavi map-info``."""
    if not scores:
        return {"tiles_scored": 0}
    values = [s.score for s in scores]
    counts: Dict[str, int] = {}
    for score in scores:
        counts[score.verdict] = counts.get(score.verdict, 0) + 1
    return {
        "tiles_scored": len(scores),
        "score": {
            "min": round(min(values), 3),
            "median": round(float(np.median(values)), 3),
            "max": round(max(values), 3),
        },
        "verdicts": counts,
        "repetitive_tiles": [s.tile_id for s in scores if s.repetitive],
        "worst": [s.as_dict() for s in scores[:5]],
    }
