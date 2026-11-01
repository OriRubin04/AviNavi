"""Procedural generation of a synthetic aerial basemap.

Feature matchers need *distinctive, localisable* structure, and the two ways to
get a synthetic test wrong are opposite: pure noise gives keypoints that match
anywhere (so every tile looks equally good and the test measures nothing), while
a regular grid gives keypoints that match everywhere ambiguously (so the test
measures aliasing rather than localization).

So the generator builds something closer to real terrain: a textured ground with
large-scale variation, irregular field parcels, a road network, scattered
buildings with shadows, and tree clumps.  The result has structure at several
scales and is locally unique, which is what real farmland and suburbs look like
to a downward camera.

Everything is seeded, so a simulation is exactly reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("the simulator needs OpenCV") from exc
    return cv2


def fractal_noise(
    shape: Tuple[int, int],
    rng: np.random.Generator,
    octaves: int = 6,
    persistence: float = 0.55,
    base_cells: int = 4,
) -> np.ndarray:
    """Value-noise fractal in ``[0, 1]``, used as the terrain substrate.

    Built by upsampling successively finer random lattices, which gives the
    1/f-ish spectrum natural terrain has - unlike white noise, whose keypoints
    carry no multi-scale structure for a detector to lock onto.
    """
    cv2 = _require_cv2()
    height, width = shape
    total = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    normalisation = 0.0
    cells = base_cells
    for _ in range(octaves):
        lattice = rng.random((max(2, cells), max(2, cells))).astype(np.float32)
        layer = cv2.resize(lattice, (width, height), interpolation=cv2.INTER_CUBIC)
        total += amplitude * layer
        normalisation += amplitude
        amplitude *= persistence
        cells *= 2
    total /= max(normalisation, 1e-6)
    low, high = float(total.min()), float(total.max())
    return (total - low) / max(high - low, 1e-6)


@dataclass
class TerrainGenerator:
    """Settings for the synthetic basemap."""

    width_px: int = 4096
    height_px: int = 4096
    #: Ground sample distance of the generated basemap, metres per pixel.
    gsd_mpp: float = 0.5
    seed: int = 7
    num_fields: int = 70
    num_roads: int = 14
    num_buildings: int = 420
    num_tree_clumps: int = 180
    #: Sun azimuth/elevation, which decides where building shadows fall.
    sun_azimuth_deg: float = 135.0
    sun_elevation_deg: float = 45.0

    def generate(self) -> np.ndarray:
        """Return an 8-bit BGR basemap."""
        cv2 = _require_cv2()
        rng = np.random.default_rng(self.seed)
        shape = (self.height_px, self.width_px)

        # Substrate: large-scale soil/vegetation variation.
        base = fractal_noise(shape, rng, octaves=7, base_cells=3)
        image = np.zeros((self.height_px, self.width_px, 3), dtype=np.float32)
        # Loosely earth-toned: more red/green than blue, varying with the noise.
        image[:, :, 0] = 70 + 55 * base          # B
        image[:, :, 1] = 95 + 75 * base          # G
        image[:, :, 2] = 85 + 80 * (1.0 - base)  # R

        self._draw_fields(image, rng)
        self._draw_roads(image, rng)
        self._draw_tree_clumps(image, rng, base)
        self._draw_buildings(image, rng)

        # Fine grain so the detector has sub-field texture to work with.
        grain = fractal_noise(shape, rng, octaves=4, base_cells=96)
        image += (grain[:, :, None] - 0.5) * 26.0

        return np.clip(image, 0, 255).astype(np.uint8)

    # -- layers -------------------------------------------------------------
    def _draw_fields(self, image: np.ndarray, rng: np.random.Generator) -> None:
        """Irregular agricultural parcels, some with plough lines."""
        cv2 = _require_cv2()
        for _ in range(self.num_fields):
            centre = rng.integers(0, [self.width_px, self.height_px])
            radius = rng.integers(80, 420)
            sides = int(rng.integers(4, 7))
            angles = np.sort(rng.random(sides) * 2 * math.pi)
            points = np.stack(
                [
                    centre[0] + radius * (0.6 + 0.4 * rng.random(sides)) * np.cos(angles),
                    centre[1] + radius * (0.6 + 0.4 * rng.random(sides)) * np.sin(angles),
                ],
                axis=1,
            ).astype(np.int32)

            tint = np.array(
                [
                    rng.uniform(45, 130),
                    rng.uniform(80, 170),
                    rng.uniform(60, 150),
                ],
                dtype=np.float32,
            )
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            alpha = rng.uniform(0.45, 0.8)
            selection = mask > 0
            image[selection] = (1 - alpha) * image[selection] + alpha * tint

            # Plough furrows: strong oriented texture, very characteristic of
            # farmland and a rich source of keypoints.
            if rng.random() < 0.45:
                spacing = int(rng.integers(7, 20))
                angle = rng.random() * math.pi
                self._draw_furrows(image, mask, spacing, angle, rng)

    def _draw_furrows(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        spacing: int,
        angle: float,
        rng: np.random.Generator,
    ) -> None:
        cv2 = _require_cv2()
        height, width = mask.shape
        overlay = np.zeros((height, width), dtype=np.uint8)
        diagonal = int(math.hypot(width, height))
        dx, dy = math.cos(angle), math.sin(angle)
        # Perpendicular offsets, stepped by the furrow spacing.
        for offset in range(-diagonal, diagonal, spacing):
            x0 = width / 2 + offset * -dy - diagonal * dx
            y0 = height / 2 + offset * dx - diagonal * dy
            x1 = width / 2 + offset * -dy + diagonal * dx
            y1 = height / 2 + offset * dx + diagonal * dy
            cv2.line(overlay, (int(x0), int(y0)), (int(x1), int(y1)), 255, 1)
        selection = (overlay > 0) & (mask > 0)
        image[selection] *= rng.uniform(0.82, 0.93)

    def _draw_roads(self, image: np.ndarray, rng: np.random.Generator) -> None:
        """A road network: long, high-contrast, and excellent for matching."""
        cv2 = _require_cv2()
        for _ in range(self.num_roads):
            # Gently curving polyline across the map.
            points = []
            x = rng.uniform(0, self.width_px)
            y = rng.uniform(0, self.height_px)
            heading = rng.random() * 2 * math.pi
            for _ in range(int(rng.integers(6, 14))):
                points.append((int(x), int(y)))
                step = rng.uniform(200, 600)
                heading += rng.uniform(-0.45, 0.45)
                x += step * math.cos(heading)
                y += step * math.sin(heading)
            polyline = np.array(points, dtype=np.int32)
            width_px = int(rng.integers(3, 11))
            grey = rng.uniform(95, 150)
            cv2.polylines(
                image, [polyline], False, (grey, grey, grey * 1.02), width_px, cv2.LINE_AA
            )
            # Verge markings on the wider roads.
            if width_px >= 7:
                cv2.polylines(image, [polyline], False, (210, 210, 205), 1, cv2.LINE_AA)

    def _draw_tree_clumps(
        self, image: np.ndarray, rng: np.random.Generator, base: np.ndarray
    ) -> None:
        cv2 = _require_cv2()
        for _ in range(self.num_tree_clumps):
            centre = rng.integers(0, [self.width_px, self.height_px])
            for _ in range(int(rng.integers(4, 18))):
                offset = rng.normal(0, 28, 2)
                radius = int(rng.integers(3, 9))
                shade = rng.uniform(0.5, 0.95)
                cv2.circle(
                    image,
                    (int(centre[0] + offset[0]), int(centre[1] + offset[1])),
                    radius,
                    (38 * shade, 72 * shade, 42 * shade),
                    -1,
                    cv2.LINE_AA,
                )

    def _draw_buildings(self, image: np.ndarray, rng: np.random.Generator) -> None:
        """Rectangular roofs with cast shadows.

        The shadow is the point: a building plus its shadow is a strongly
        asymmetric, locally unique pattern, which is exactly what a descriptor
        needs to be confident about *which* building it is looking at.
        """
        cv2 = _require_cv2()
        sun = math.radians(self.sun_azimuth_deg)
        shadow_length = 1.0 / max(math.tan(math.radians(self.sun_elevation_deg)), 0.2)
        for _ in range(self.num_buildings):
            cx = int(rng.uniform(0, self.width_px))
            cy = int(rng.uniform(0, self.height_px))
            w = int(rng.integers(8, 46))
            h = int(rng.integers(8, 46))
            angle = rng.uniform(0, 180)
            storeys = rng.uniform(3.0, 12.0) / self.gsd_mpp * 0.1

            rectangle = ((cx, cy), (w, h), angle)
            box = cv2.boxPoints(rectangle).astype(np.int32)

            shift = np.array(
                [
                    storeys * shadow_length * math.sin(sun),
                    -storeys * shadow_length * math.cos(sun),
                ]
            )
            shadow = (box + shift).astype(np.int32)
            cv2.fillConvexPoly(image, shadow, (28, 30, 32), cv2.LINE_AA)

            roof = rng.uniform(120, 215)
            cv2.fillConvexPoly(
                image,
                box,
                (roof * rng.uniform(0.85, 1.0), roof * rng.uniform(0.9, 1.0), roof),
                cv2.LINE_AA,
            )
            cv2.polylines(image, [box], True, (roof * 0.6,) * 3, 1, cv2.LINE_AA)


def generate_basemap(
    width_px: int = 4096,
    height_px: int = 4096,
    gsd_mpp: float = 0.5,
    seed: int = 7,
) -> np.ndarray:
    """Convenience wrapper around :class:`TerrainGenerator`."""
    return TerrainGenerator(
        width_px=width_px, height_px=height_px, gsd_mpp=gsd_mpp, seed=seed
    ).generate()


def seasonal_variant(
    basemap: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    strength: float = 1.0,
) -> np.ndarray:
    """Return a plausibly *different-date* version of the same ground.

    This is the harness's attempt at honesty.  In reality the basemap was shot
    months or years before the flight: vegetation has changed colour, fields
    have been harvested or replanted, the sun is at a different angle, and some
    structures have appeared or gone.  Matching a live frame against a basemap
    that is pixel-identical to it flatters the pipeline enormously.

    So the simulator renders camera frames from *this* variant while the tile
    database is built from the original - the two differ the way a real flight
    differs from its basemap.  ``strength`` scales the whole effect so the
    sensitivity to appearance change can be measured directly.
    """
    cv2 = _require_cv2()
    rng = rng or np.random.default_rng(1234)
    image = basemap.astype(np.float32)

    # Global colour/illumination shift: different season, different sun.
    gains = 1.0 + strength * rng.uniform(-0.18, 0.18, 3)
    image *= gains[None, None, :]
    image += strength * rng.uniform(-18, 18)

    # Vegetation response: fields change tone substantially between dates.
    green_dominant = (image[:, :, 1] > image[:, :, 2] * 1.02).astype(np.float32)
    green_mask = cv2.GaussianBlur(green_dominant, (0, 0), 6.0)[:, :, None]
    vegetation_shift = np.array(
        [
            strength * rng.uniform(-10, 25),
            strength * rng.uniform(-45, 20),
            strength * rng.uniform(-10, 45),
        ],
        dtype=np.float32,
    )
    image += green_mask * vegetation_shift[None, None, :]

    # Large-scale patchiness: harvested parcels, irrigation, soil moisture.
    patches = fractal_noise(basemap.shape[:2], rng, octaves=4, base_cells=6)
    image *= 1.0 + strength * 0.22 * (patches[:, :, None] - 0.5)

    # A few local changes: structures built or removed since the basemap.
    changed = int(strength * 60)
    for _ in range(changed):
        cx = int(rng.uniform(0, basemap.shape[1]))
        cy = int(rng.uniform(0, basemap.shape[0]))
        size = int(rng.integers(10, 40))
        tone = rng.uniform(60, 200)
        cv2.rectangle(
            image, (cx, cy), (cx + size, cy + size), (tone, tone * 0.97, tone * 0.95), -1
        )

    return np.clip(image, 0, 255).astype(np.uint8)
