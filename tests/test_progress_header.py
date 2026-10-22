"""The clock belongs to the widget, not to the worker.

The bug these exist for: Step 3 spends the first minutes of a run inside one
call -- ``from_pretrained`` fetching 1.3 GB -- during which the worker sends
nothing. "Elapsed" stopped at 1s, the last moment a message had arrived, and a
working application looked wedged. The same silence happens while TensorRT
builds an engine and while a cold dataloader spins up, so the fix belongs here
rather than in any one command.

Time is faked rather than slept through: a test that waits a real minute to
check a minute has passed is a test nobody runs.
"""

from __future__ import annotations

import pytest

from yolo_assistant.app.ui.common.progress_header import ProgressHeader


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr("time.monotonic", fake)
    return fake


@pytest.fixture
def header(qapp, clock):
    widget = ProgressHeader()
    widget.start()
    return widget


def tick(widget) -> None:
    """What the widget's own timer does once a second."""
    widget._paint_times()


def test_elapsed_keeps_moving_while_the_worker_says_nothing(header, clock):
    header.update_progress({"phase": "download", "current": 0, "total": None})

    # A minute passes with no further message, as it does while a model
    # downloads. Nothing arrives to repaint the line, so the line repaints
    # itself.
    clock.advance(65)
    tick(header)

    assert header._times.text() == "Elapsed 1m 05s"
    assert header._clock.isActive()


def test_the_estimate_counts_down_between_messages(header, clock):
    header.update_progress({"phase": "label", "current": 1, "total": 100, "eta_s": 100})

    clock.advance(30)
    tick(header)

    # Not "1m 40s remaining" held for thirty seconds and then a jump: the
    # estimate was made thirty seconds ago and ages like one.
    assert "remaining 1m 10s" in header._times.text()


def test_an_overrun_estimate_stops_at_zero_rather_than_going_negative(header, clock):
    header.update_progress({"phase": "label", "current": 1, "total": 100, "eta_s": 10})

    clock.advance(600)
    tick(header)

    assert "remaining 0s" in header._times.text()


def test_a_fresh_estimate_replaces_the_ageing_one(header, clock):
    header.update_progress({"phase": "label", "current": 1, "total": 100, "eta_s": 100})
    clock.advance(30)

    header.update_progress({"phase": "label", "current": 40, "total": 100, "eta_s": 50})

    assert "remaining 50s" in header._times.text()


def test_no_estimate_is_shown_until_one_arrives(header, clock):
    header.update_progress({"phase": "preprocess", "current": 3, "total": 90})

    # An estimate made from one sample is worse than no estimate at all.
    assert "Elapsed" in header._times.text()
    assert "remaining" not in header._times.text()


def test_a_finished_run_keeps_the_time_it_took(header, clock):
    header.update_progress({"phase": "label", "current": 90, "total": 90, "eta_s": 0})
    clock.advance(125)
    tick(header)

    header.stop("Done")

    assert not header._clock.isActive(), "a finished run must not keep a timer running"
    assert "Elapsed 2m 05s" in header._times.text(), "how long it took is the first question"
    assert header._phase.text() == "Done"


def test_progress_before_a_run_starts_is_harmless(qapp, clock):
    widget = ProgressHeader()

    widget.update_progress({"phase": "label", "current": 1, "total": 2, "eta_s": 5})

    assert widget._times.text() == ""
