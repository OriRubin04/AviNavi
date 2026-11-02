"""The tile database and the moving search window.

The search window is what makes this a GPS-denied system.  It is seeded *once*
from the operator-selected initial position and, from then on, is recentred on
the most recent **visual** fix.  No GPS message ever touches this class.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..geo import BoundingBox, bounding_box_around
from ..types import GeoPoint
from .tile import MapTile, TileFormatError, TileMetadata, METADATA_FILENAME

LOG = logging.getLogger(__name__)

INDEX_FILENAME = "index.json"


class MapCoverageError(RuntimeError):
    """No tile covers the requested area - a map coverage failure."""


class MapDatabase:
    """An on-disk collection of georeferenced tiles.

    Tiles are discovered by scanning for ``metadata.json`` files.  A generated
    ``index.json`` at the database root is used when present to avoid stat-ing
    every tile directory at startup, which matters once a database has
    thousands of tiles on an SD card.
    """

    def __init__(self, path: str, feature_cache_size: int = 64) -> None:
        self.path = path
        self.feature_cache_size = max(1, int(feature_cache_size))
        self._tiles: Dict[str, MapTile] = {}
        #: tile_id -> None, ordered by least-recently-used feature access.
        self._feature_lru: "OrderedDict[str, None]" = OrderedDict()

    # -- loading ------------------------------------------------------------
    @classmethod
    def load(cls, path: str, feature_cache_size: int = 64) -> "MapDatabase":
        db = cls(path, feature_cache_size=feature_cache_size)
        db.reload()
        return db

    def reload(self) -> None:
        if not os.path.isdir(self.path):
            raise MapCoverageError("map database directory not found: %s" % self.path)

        metadatas = self._read_index()
        if metadatas is None:
            metadatas = self._scan()

        self._tiles = {}
        for metadata in metadatas:
            if metadata.tile_id in self._tiles:
                LOG.warning(
                    "duplicate tile_id %r in %s; keeping the first",
                    metadata.tile_id,
                    metadata.directory,
                )
                continue
            self._tiles[metadata.tile_id] = MapTile(metadata)
        self._feature_lru.clear()

        if not self._tiles:
            raise MapCoverageError("map database %s contains no tiles" % self.path)
        LOG.info("loaded %d map tiles from %s", len(self._tiles), self.path)

    def _read_index(self) -> Optional[List[TileMetadata]]:
        index_path = os.path.join(self.path, INDEX_FILENAME)
        if not os.path.exists(index_path):
            return None
        if self._index_is_stale(index_path):
            return None
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            entries = payload["tiles"] if isinstance(payload, dict) else payload
            metadatas = [
                TileMetadata.from_dict(
                    entry, os.path.join(self.path, entry["directory"])
                )
                for entry in entries
            ]
        except (OSError, ValueError, KeyError) as exc:
            LOG.warning("ignoring unreadable %s (%s); rescanning", index_path, exc)
            return None

        # The mtime check above catches a rebuilt database. This catches one
        # whose tile *set* changed - tiles added, removed, or renamed - which
        # can happen with timestamps that look perfectly fresh, for instance
        # after copying a database between machines.
        on_disk = {
            entry
            for entry in os.listdir(self.path)
            if os.path.exists(os.path.join(self.path, entry, METADATA_FILENAME))
        }
        indexed = {
            os.path.basename(m.directory.rstrip("/\\")) for m in metadatas
        }
        if on_disk != indexed:
            LOG.warning(
                "%s lists %d tiles but %d are on disk (%d only in the index, "
                "%d only on disk); ignoring the index and rescanning",
                index_path,
                len(indexed),
                len(on_disk),
                len(indexed - on_disk),
                len(on_disk - indexed),
            )
            return None
        return metadatas

    def _index_is_stale(self, index_path: str) -> bool:
        """Is ``index.json`` older than the tiles it claims to describe?

        This check exists because of a bug that cost a whole measurement run and
        would have been far worse in the air.  The index is a cache of the
        per-tile ``metadata.json`` files, written to spare a directory scan at
        startup.  Rebuilding a database over an existing directory rewrites
        every tile's own metadata but leaves the index describing the *previous*
        build, and because the index was trusted unconditionally, the whole
        system then used the old georeferencing: a measurement with correct
        imagery, correct features, correct matching, and a constant 100 m
        position offset that nothing reported.

        In flight that is the worst possible failure shape - a confident,
        plausible, consistently wrong position, which is exactly what the EKF
        has no way to reject.  So the index is now treated as what it is, a
        cache, and revalidated on every load.  One ``stat`` per tile is nothing
        against startup I/O, and if the check is wrong it is wrong in the safe
        direction: a needless rescan.
        """
        try:
            index_mtime = os.path.getmtime(index_path)
        except OSError:
            return True

        newest_tile_mtime = 0.0
        newest_tile = None
        try:
            entries = sorted(os.listdir(self.path))
        except OSError:
            return True
        for entry in entries:
            metadata_path = os.path.join(self.path, entry, METADATA_FILENAME)
            try:
                mtime = os.path.getmtime(metadata_path)
            except OSError:
                continue
            if mtime > newest_tile_mtime:
                newest_tile_mtime = mtime
                newest_tile = entry

        # A one-second grace: the builder writes the tiles and then the index,
        # so on a coarse-grained filesystem they can share a timestamp.
        if newest_tile_mtime > index_mtime + 1.0:
            LOG.warning(
                "%s is older than the tiles it describes (%s is newer); it is a "
                "stale cache from an earlier build and is being ignored. "
                "Rescanning the tiles themselves - run 'avinavi map-info' to "
                "rewrite the index.",
                index_path,
                newest_tile,
            )
            return True
        return False

    def _scan(self) -> List[TileMetadata]:
        found: List[TileMetadata] = []
        for entry in sorted(os.listdir(self.path)):
            directory = os.path.join(self.path, entry)
            if not os.path.isdir(directory):
                continue
            if not os.path.exists(os.path.join(directory, METADATA_FILENAME)):
                continue
            try:
                found.append(TileMetadata.load(directory))
            except (TileFormatError, ValueError, OSError) as exc:
                LOG.warning("skipping tile %s: %s", directory, exc)
        return found

    def write_index(self) -> str:
        """Write ``index.json`` so subsequent startups skip the directory scan."""
        entries = []
        for tile in self._tiles.values():
            data = tile.metadata.to_dict()
            data["directory"] = os.path.relpath(tile.metadata.directory, self.path)
            entries.append(data)
        index_path = os.path.join(self.path, INDEX_FILENAME)
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump({"tiles": entries}, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return index_path

    # -- access -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._tiles)

    def __iter__(self):
        return iter(self._tiles.values())

    @property
    def tiles(self) -> List[MapTile]:
        return list(self._tiles.values())

    def get(self, tile_id: str) -> Optional[MapTile]:
        return self._tiles.get(tile_id)

    @property
    def coverage(self) -> Optional[BoundingBox]:
        """Union bounding box of every tile in the database."""
        if not self._tiles:
            return None
        west = min(t.bounds.west for t in self._tiles.values())
        south = min(t.bounds.south for t in self._tiles.values())
        east = max(t.bounds.east for t in self._tiles.values())
        north = max(t.bounds.north for t in self._tiles.values())
        return BoundingBox(west, south, east, north)

    # -- the moving search window ------------------------------------------
    def candidates(
        self,
        center: GeoPoint,
        radius_m: float,
        limit: Optional[int] = None,
        prefer: Optional[Sequence[str]] = None,
    ) -> List[MapTile]:
        """Tiles whose footprint comes within ``radius_m`` of ``center``.

        Results are ordered nearest-first by the distance from ``center`` to the
        tile footprint (0 when ``center`` is inside the tile), which puts the
        most likely match first so ``limit`` truncates the least useful tiles.
        ``prefer`` promotes named tiles - normally the tile that produced the
        previous fix - to the front, since spatial continuity makes it by far
        the most likely match for the next frame.
        """
        if radius_m <= 0:
            raise ValueError("search radius must be positive")

        box = bounding_box_around(center, radius_m)
        scored: List[Tuple[float, MapTile]] = []
        for tile in self._tiles.values():
            if not _boxes_intersect(tile.bounds, box):
                continue
            distance = tile.bounds.distance_to_m(center)
            if distance <= radius_m:
                scored.append((distance, tile))

        scored.sort(key=lambda item: (item[0], item[1].tile_id))
        ordered = [tile for _, tile in scored]

        if prefer:
            preferred_ids = [tid for tid in prefer if tid in self._tiles]
            if preferred_ids:
                promoted = [t for t in ordered if t.tile_id in preferred_ids]
                remainder = [t for t in ordered if t.tile_id not in preferred_ids]
                # Keep the caller's preference order among promoted tiles.
                promoted.sort(key=lambda t: preferred_ids.index(t.tile_id))
                ordered = promoted + remainder

        if limit is not None and limit > 0:
            ordered = ordered[:limit]
        return ordered

    def tile_containing(self, point: GeoPoint) -> Optional[MapTile]:
        for tile in self._tiles.values():
            if tile.bounds.contains(point):
                return tile
        return None

    # -- feature cache ------------------------------------------------------
    def features_for(self, tile: MapTile):
        """Load a tile's features, evicting the least recently used if needed."""
        already_loaded = tile.features_loaded
        features = tile.features()
        self._feature_lru.pop(tile.tile_id, None)
        self._feature_lru[tile.tile_id] = None
        if not already_loaded:
            self._evict_if_needed()
        return features

    def _evict_if_needed(self) -> None:
        while len(self._feature_lru) > self.feature_cache_size:
            tile_id, _ = self._feature_lru.popitem(last=False)
            tile = self._tiles.get(tile_id)
            if tile is not None:
                tile.release()

    def preload(self, tiles: Iterable[MapTile]) -> int:
        """Warm the feature cache for a set of tiles; returns how many loaded."""
        count = 0
        for tile in tiles:
            try:
                self.features_for(tile)
                count += 1
            except TileFormatError as exc:
                LOG.warning("tile %s has no usable features: %s", tile.tile_id, exc)
        return count

    def release_all(self) -> None:
        for tile in self._tiles.values():
            tile.release()
        self._feature_lru.clear()


def _boxes_intersect(a: BoundingBox, b: BoundingBox) -> bool:
    return not (a.east < b.west or a.west > b.east or a.north < b.south or a.south > b.north)
