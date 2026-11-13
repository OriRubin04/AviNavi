"""Does a false fix start a blackout, or did two things merely co-occur?

A starved run produced exactly one false fix and a 116-cycle run of failures out
of 150 - 116 of its 124 failures in a single event, while the two wide-footprint
variants recorded no failure runs at all. The obvious reading is that the tile
search, being centred on the last accepted position, cannot recover once it
accepts a position on the wrong ground: every candidate tile afterwards is drawn
from the wrong place.

Obvious is not measured. The report recorded failure-run *lengths* and a false
fix *count*, and neither carries a cycle index, so the two could not be placed
relative to each other in time. The claim was inference from a co-occurrence in
one run.

Recording both indices makes it testable, and these tests pin the test itself -
including the cases that must come out negative, since a detector that always
says yes measures nothing.
"""

from __future__ import annotations

from avinavi.sim.evaluate import SimulationReport


def _report(false_fix_cycles, failure_run_starts, failure_runs) -> SimulationReport:
    report = SimulationReport(label="synthetic")
    report.false_fix_cycles = list(false_fix_cycles)
    report.failure_run_starts = list(failure_run_starts)
    report.failure_runs = list(failure_runs)
    return report


class TestBlackoutDetection:
    def test_the_observed_shape_is_detected(self):
        """One false fix at cycle 33, then 116 failures from cycle 34."""
        report = _report([33], [34], [116])
        assert report.blackouts_after_false_fix() == [(33, 34, 116)]

    def test_a_blackout_that_starts_long_after_is_not_attributed(self):
        """The claim is 'the next cycle', not 'somewhere later in the run'."""
        report = _report([10], [80], [60])
        assert report.blackouts_after_false_fix() == []

    def test_a_false_fix_the_system_recovered_from_counts_against(self):
        """A false fix followed by more good fixes is evidence the other way."""
        report = _report([10], [], [])
        assert report.blackouts_after_false_fix() == []

    def test_a_blackout_with_no_false_fix_before_it_counts_against(self):
        """Long failure runs also happen on their own; those must not be claimed."""
        report = _report([], [40], [90])
        assert report.blackouts_after_false_fix() == []

    def test_only_the_blackout_that_follows_is_attributed(self):
        report = _report([5], [6, 70], [40, 3])
        assert report.blackouts_after_false_fix() == [(5, 6, 40)]

    def test_several_false_fixes_are_each_checked(self):
        report = _report([5, 60], [6, 61], [20, 30])
        assert report.blackouts_after_false_fix() == [(5, 6, 20), (60, 61, 30)]

    def test_the_gap_is_not_widened_by_default(self):
        """A two-cycle delay is not 'immediately after' unless asked for."""
        report = _report([10], [12], [50])
        assert report.blackouts_after_false_fix() == []
        assert report.blackouts_after_false_fix(gap=2) == [(10, 12, 50)]

    def test_a_run_with_neither_is_empty(self):
        assert _report([], [], []).blackouts_after_false_fix() == []


class TestRecording:
    """The indices have to survive into the JSON, or the analysis cannot be run."""

    def test_as_dict_carries_the_indices_and_the_verdict(self):
        report = _report([33], [34], [116])
        report.cycles = 150
        report.valid_fixes = 26
        report.false_fixes = 1
        data = report.as_dict()
        assert data["false_fix_cycles"] == [33]
        assert data["failure_runs"]["starts"] == [34]
        assert data["blackouts_after_false_fix"] == [[33, 34, 116]]

    def test_a_clean_run_reports_an_empty_verdict_not_a_missing_key(self):
        """An absent key reads as 'not measured'; an empty list reads as 'no'."""
        report = _report([], [], [])
        report.cycles = 150
        report.valid_fixes = 150
        data = report.as_dict()
        assert data["blackouts_after_false_fix"] == []
        assert data["false_fix_cycles"] == []
