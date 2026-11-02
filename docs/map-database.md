# The map database

A georeferenced tile database with SuperPoint features precomputed offline. The
Raspberry Pi never runs the detector over static map data — that is the whole
point.

## Layout

```
map_database/
├── index.json              generated; skips the directory scan at startup
├── tile_0000/
│   ├── image.jpg           the tile raster, stored north-up
│   ├── features.pt         precomputed SuperPoint keypoints + descriptors
│   └── metadata.json       georeferencing and provenance
├── tile_0001/
│   ├── image.jpg
│   ├── features.pt
│   └── metadata.json
└── ...
```

`features.pt` is a torch save file when torch was available at build time and a
`.npz` otherwise. Both are loaded transparently: asking for `features.pt` falls
back to `features.npz` if that is what is on disk.

## `metadata.json`

```json
{
  "tile_id": "tile_0042",
  "image": "image.jpg",
  "features": "features.pt",
  "crs": "EPSG:4326",
  "bounds": {
    "west": 34.7760,
    "south": 32.0800,
    "east": 34.7867,
    "north": 32.0890
  },
  "transform": [8.98e-06, 0.0, 34.7760, 0.0, -8.98e-06, 32.0890],
  "width_px": 1024,
  "height_px": 1024,
  "ground_resolution_mpp": 0.5,
  "terrain_alt_m": 25.0,
  "source": "orthophoto.tif",
  "source_window": [2048, 1024, 3072, 2048],
  "created_utc": "2026-07-31T09:14:22+00:00"
}
```

Required fields, per the specification:

| Field | Purpose |
| --- | --- |
| `bounds` | Geographic bounding box (WGS-84), used by the tile search |
| `crs` | Coordinate reference system of `transform` |
| `width_px`, `height_px` | Pixel dimensions |
| `ground_resolution_mpp` | Ground sample distance, metres per pixel |
| `image` | Path to the raster, relative to the tile directory |
| `features` | Path to the SuperPoint keypoints and descriptors |

`terrain_alt_m` is the mean terrain altitude over the tile, in the same datum as
the EKF origin. It is what turns a map-plane fix into a 3-D position, and it
falls back to the global `map.terrain_alt_m` when zero.

### Georeferencing: two forms

**Bounds only.** A north-up linear interpolation between the corners. The common
case for tiles cut from a lat/lon mosaic, and all that is needed when the tile
is small enough that the projection is locally linear.

**Affine transform.** A GDAL-style 6-tuple `(a, b, c, d, e, f)`:

```
x = a·col + b·row + c
y = d·col + e·row + f
```

with `(x, y)` in the tile's CRS. Written automatically when the source is a
GeoTIFF, and preferred when present because it loses no precision at the tile
level. A CRS other than EPSG:4326 is reprojected with pyproj, imported lazily so
the flight runtime never needs it when tiles are already in EPSG:4326.

`bounds` is always present regardless, because the tile search needs a cheap
WGS-84 footprint for every tile.

### Pixel convention

`(col, row)` are continuous pixel coordinates with `(0, 0)` at the *corner* of
the top-left pixel, matching GDAL. Tiles are north-up: `+col` is east, `+row`
is south.

## Building a database

### From a GeoTIFF

```bash
pip install 'avinavi[maps]'          # rasterio + pyproj
avinavi build-map orthophoto.tif map_database --tile-size 1024 --overlap 0.25
```

Bounds, CRS and per-tile affine transforms are read from the file.

### From a plain image plus bounds

```bash
avinavi build-map mosaic.png map_database \
    --bounds 34.7600,32.0700,34.8100,32.1000 \
    --tile-size 1024 --overlap 0.25 --terrain-alt 25
```

`--bounds` is `west,south,east,north` in WGS-84 degrees, describing the whole
source image.

### Features separately

`build-map` runs feature extraction at the end. To do it as its own step (on a
different machine, or after adding tiles by hand):

```bash
avinavi build-map mosaic.png map_database --bounds ... --no-features
avinavi extract-features map_database
```

`extract-features` is idempotent: it skips tiles whose features are newer than
their image. `--force` recomputes everything.

Tiles get a larger keypoint budget than live frames — four times
`superpoint.max_num_keypoints` by default — because a tile covers far more
ground than a single camera frame. Override with `--max-keypoints`.

### Inspecting

```bash
avinavi map-info map_database
```

Reports tile count, coverage bounding box, the range of ground resolutions, and
any tiles missing features (exit code 1 if there are any).

It also scores each tile for **localizability**: whether the imagery can support
a match at all. This reads every tile's features, so it is the slow part of the
command; `--no-localizability` skips it.

## Ground that cannot be matched

Some terrain cannot be localized against, and it is much cheaper to find out
before the flight than during one. Two opposite failure modes:

**Nothing to detect.** Open water, snow, desert sand, dense uniform forest
canopy. Few keypoints, and the ones that exist come from sensor noise. The
system fails safe here — no match, no position, and the dead reckoning carries
on until it expires.

**Too much of the same thing.** Orchards, vineyards, ploughed fields, solar
farms, large car parks. This is the dangerous one, and it does **not** fail
safe. There are plenty of keypoints and they are all confusable, so a matcher
can lock onto the ground one row of trees away and the geometry is still
perfectly consistent — RANSAC confirms it, the confidence score is high, and the
reported position is wrong by exactly one period. The EKF has no way to reject
that.

`map-info` reports three components per tile:

| Component | Question it answers |
| --- | --- |
| `density` | Is there anything here at all? |
| `distinctiveness` | Or is it the same thing repeated? |
| `spread` | Does it work anywhere over the tile, or only in one corner? |

They combine into `score` in [0, 1] as a weighted **geometric** mean, so a tile
that fails badly on any one of them scores badly overall — an arithmetic mean
would let a high keypoint density mask zero distinctiveness, which is precisely
the orchard case.

Tiles flagged `"repetitive": true` get their own stderr warning, because
"poor" undersells them: the failure is a confident wrong position rather than a
missing one. If your route must cross such ground, raise
`geometry.min_inlier_ratio` and `confidence.min_confidence` — and treat the
dead-reckoning path as the primary source over that stretch rather than the map.

Measured behaviour, on real SIFT features:

| Scene | `distinctiveness` | `score` | Verdict |
| --- | --- | --- | --- |
| mixed terrain | 0.95 | 0.98 | good |
| orchard with varying tree size, tone and spacing | 0.08 | 0.32 | poor, flagged repetitive |
| lattice of identical blobs | 0.00 | 0.00 | unusable, flagged repetitive |
| near-featureless water | — (no keypoints) | 0.00 | unusable |

**Two honest limits.** The score is computed per *region* within a tile rather
than averaged over it, which matters — an orchard stamped over textured terrain
moves the tile-wide figure only from 0.89 to 0.78 because the surrounding
texture keeps supplying good keypoints, while the region-wise measure separates
the same tiles across 0.60 to 0.92. Even so, a large tile averages over
heterogeneous ground, so **smaller tiles give a finer-grained warning**. And
whether the score ranks tiles in the order they *actually* localize has not yet
been measured against flight or simulation outcomes; a low score is a reliable
warning about the imagery, but a high score is not a promise.

## Choosing tile size and overlap

**Overlap matters more than it looks.** A live frame that straddles a tile
boundary sees part of two tiles and may not accumulate enough inliers on
either. Enough overlap that a whole camera footprint always fits inside at
least one tile is the requirement. The default 25% is a reasonable starting
point for a typical camera footprint against 1024 px tiles.

Work out your camera's ground footprint:

```
footprint_m = altitude_m × image_width_px / fx
```

At 100 m with a 640 px wide image and `fx = 500`, that is 128 m across. With
0.5 m/px tiles, a 1024 px tile covers 512 m — so a 25% overlap (128 m) is
exactly enough. Fly lower or use a narrower lens and you need more.

Tile size trades RAM and matching cost against how many tiles the search has to
consider. 1024 px is a sensible default for a Pi 5 with 8 GB.

## Ground resolution

The tile GSD should be *comparable* to the live frame's, because SuperPoint and
LightGlue degrade when the two images differ greatly in scale. As a rule of
thumb keep the ratio within roughly 2× either way. If your basemap is 0.3 m/px
and you fly at an altitude that gives 0.15 m/px, consider downsampling the
tiles when building the database.

`geometry.min_scale_ratio` / `max_scale_ratio` (default 0.25 to 4.0) reject
matches whose implied scale change is outside a plausible band, which catches
the case where the aircraft is at a wildly different altitude than assumed.

## The feature cache

Loading every tile's descriptors would exhaust RAM on a large database, so the
database keeps an LRU cache:

```yaml
map:
  feature_cache_size: 64      # tiles' features held in RAM
```

64 tiles of 4096 keypoints × 256 float32 descriptors is roughly 270 MB —
comfortable on an 8 GB Pi 5. Raise it if your search window regularly evicts
tiles it is about to need again; the diagnostics show how many candidates each
cycle considers.

## The moving search window

```yaml
tile_search:
  search_radius_m: 300.0
  initial_search_radius_m: 500.0
  radius_growth_factor: 1.5
  max_search_radius_m: 2000.0
  failures_before_growth: 2
  prefer_last_matched_tile: true
```

Candidates are ordered by the distance from the search centre to the tile
*footprint* (zero when the centre is inside the tile), so `lightglue.
max_candidate_tiles` truncates the least likely tiles rather than an arbitrary
set. The tile that produced the previous fix is promoted to the front, because
spatial continuity makes it by far the most likely match for the next frame.

After `failures_before_growth` consecutive failures the radius grows by
`radius_growth_factor` each cycle, up to `max_search_radius_m` — this is how a
drifted aircraft re-acquires. Once the radius is at its ceiling and still
finding nothing, `search_exhausted` goes true and the runner reports a map
coverage failure rather than searching forever.

Every successful fix resets the radius to `search_radius_m` and recentres the
window on the **visual** position.

## Building a database by hand

For tests and small experiments, `avinavi.mapdb.builder.write_tile` writes a
single tile from an in-memory image:

```python
from avinavi.geo import BoundingBox
from avinavi.mapdb.builder import write_tile

write_tile(
    directory="map_database/tile_0000",
    image=image,                       # numpy array, BGR or grayscale
    bounds=BoundingBox(34.776, 32.080, 34.787, 32.089),
    terrain_alt_m=25.0,
)
```

Then run `avinavi extract-features map_database`.
