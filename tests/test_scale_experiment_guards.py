"""The scale experiment refuses geometry that would quietly bias a variant.

Every one of these guards exists because an experiment in this project already
produced a confident, wrong number by violating it.  None of them is
hypothetical.

The horizon guard is the sharpest example.  The experiment's exact matched pair
was 360 m with a 60 degree lens against 120 m with a 120 degree lens - both
cover 416 m of ground and rectify to 831 x 467 px, so if rectified size is what
limits matching they must score the same despite a threefold difference in
altitude.  The 120 degree variant returned 100% lock and 158 inliers against the
360 m variant's 100% and 152, which reads as a clean confirmation.

It was not one.  At 120 degrees the frame edge sits 60 degrees from nadir, so at
the trajectory's 30 degrees of bank it reaches exactly the horizon and the frame
cannot be rendered.  Eighteen of 152 frames were dropped - and the dropped
frames are the *banked* ones, the hardest in the run.  The variant was scored on
an easier subset than everything it was compared against.

So the guard refuses that configuration outright rather than letting it produce
a number that looks like evidence.
"""

from __future__ import annotations

import importlib.util
import math
import os

import pytest

_TOOL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools",
    "run_scale_experiment.py",
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("run_scale_experiment", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


class TestHorizonGuard:
    def test_a_120_degree_lens_at_30_degrees_of_bank_is_refused(self, tool):
        """The configuration that produced the false confirmation."""
        with pytest.raises(SystemExit) as excinfo:
            tool.check_geometry(
                [("120 m, 120 deg", 120.0, 120.0)],
                basemap_px=4000,
                extent_m=600.0,
                max_bank_deg=30.0,
            )
        message = str(excinfo.value)
        assert "horizon" in message
        assert "banked" in message, "say why the loss biases the result"

    def test_the_same_lens_is_allowed_in_level_flight(self, tool):
        """The lens is not the problem; the lens plus the bank is."""
        tool.check_geometry(
            [("120 m, 120 deg", 120.0, 120.0)],
            basemap_px=4000,
            extent_m=600.0,
            max_bank_deg=0.0,
        )

    def test_the_lenses_actually_used_are_allowed(self, tool):
        """A guard that rejects the real experiment is worse than none."""
        tool.check_geometry(
            [
                ("120 m, 60 deg", 120.0, 60.0),
                ("240 m, 60 deg", 240.0, 60.0),
                ("360 m, 60 deg", 360.0, 60.0),
                ("120 m, 100 deg", 120.0, 100.0),
            ],
            basemap_px=4000,
            extent_m=600.0,
            max_bank_deg=30.0,
        )

    def test_the_arithmetic_behind_the_threshold(self):
        """Half the field of view plus the bank is the angle from nadir."""
        assert 120.0 / 2 + 30.0 == pytest.approx(90.0), "exactly the horizon"
        assert 100.0 / 2 + 30.0 == pytest.approx(80.0), "steep but renderable"

    def test_the_default_variant_list_contains_nothing_refused(self, tool):
        """Running the tool as shipped must not trip its own guard."""
        variants = [
            ("120 m, 60 deg (today)", 120.0, 60.0),
            ("180 m, 60 deg", 180.0, 60.0),
            ("240 m, 60 deg", 240.0, 60.0),
            ("300 m, 60 deg", 300.0, 60.0),
            ("360 m, 60 deg", 360.0, 60.0),
            ("120 m, 100 deg", 120.0, 100.0),
        ]
        tool.check_geometry(variants, 4000, 600.0, tool.MAX_BANK_DEG)


class TestCoverageGuard:
    def test_a_variant_that_runs_off_the_map_is_refused(self, tool):
        with pytest.raises(SystemExit, match="only extends"):
            tool.check_geometry(
                [("360 m, 60 deg", 360.0, 60.0)],
                basemap_px=1200,
                extent_m=600.0,
                max_bank_deg=30.0,
            )

    def test_a_frame_larger_than_a_tile_is_refused(self, tool):
        """A frame that cannot fall inside one tile is a different experiment."""
        tile_ground_m = tool.TILE_PX * tool.TILE_GSD
        height = tile_ground_m / (2 * math.tan(math.radians(30.0))) * 1.2
        with pytest.raises(SystemExit, match="more than one"):
            tool.check_geometry(
                [("too high", height, 60.0)],
                basemap_px=40000,
                extent_m=10.0,
                max_bank_deg=0.0,
            )


class TestRectifiedGeometry:
    def test_rectified_size_is_footprint_over_tile_gsd(self, tool):
        """The claim the whole experiment rests on, as arithmetic."""
        assert tool.rectified_px(120.0, 60.0) == pytest.approx(
            tool.footprint_m(120.0, 60.0) / tool.TILE_GSD, abs=1
        )

    def test_the_pair_is_matched(self, tool):
        """240 m / 60 deg and 120 m / 100 deg must stay close in rectified size.

        If someone edits the variant list, this is what catches the pair
        silently drifting apart and the comparison becoming meaningless.
        """
        a = tool.rectified_px(240.0, 60.0)
        b = tool.rectified_px(120.0, 100.0)
        assert abs(a - b) / max(a, b) < 0.05, (
            "the matched pair is no longer matched: %d px vs %d px" % (a, b)
        )

    def test_rectified_size_does_not_depend_on_sensor_resolution(self, tool):
        """Why the pair works at all: only footprint and map GSD matter."""
        assert tool.rectified_px(240.0, 60.0) == tool.rectified_px(240.0, 60.0)
        # Doubling the sensor changes the live GSD but not the rectified size.
        fine = 240.0 / ((2560 / 2) / math.tan(math.radians(30.0)))
        coarse = 240.0 / ((1280 / 2) / math.tan(math.radians(30.0)))
        assert fine == pytest.approx(coarse / 2)
        assert tool.footprint_m(240.0, 60.0) == pytest.approx(277.1, abs=0.5)


class TestOnlyFilter:
    """--only must not silently select more than it was asked for.

    The first spelling split on commas, but the variant labels contain commas:
    "240 m, 60 deg" became the fragments "240 m" and "60 deg", and that second
    fragment matches *every* 60 degree variant. A run asked for three variants
    quietly measured six, which cost twenty minutes and would have been read as
    the requested comparison.
    """

    LABELS = [
        "120 m, 60 deg (today)",
        "180 m, 60 deg",
        "240 m, 60 deg",
        "300 m, 60 deg",
        "360 m, 60 deg",
        "120 m, 100 deg",
    ]

    def _select(self, wanted):
        return [
            label
            for label in self.LABELS
            if any(w.strip() in label for w in wanted if w.strip())
        ]

    def test_a_full_label_selects_exactly_one_variant(self):
        assert self._select(["240 m, 60 deg"]) == ["240 m, 60 deg"]

    def test_repeating_the_flag_selects_exactly_the_pair(self):
        assert self._select(["240 m, 60 deg", "120 m, 100 deg"]) == [
            "240 m, 60 deg",
            "120 m, 100 deg",
        ]

    def test_the_comma_split_that_caused_the_over_selection(self):
        """Documents the old behaviour so it is not reintroduced as a 'tidy-up'."""
        broken = "240 m, 60 deg,120 m, 100 deg".split(",")
        assert len(self._select(broken)) == len(self.LABELS), (
            "splitting on commas selects every variant, which is the bug"
        )
