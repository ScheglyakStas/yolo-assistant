"""End-to-end through the real widgets: page -> runner -> subprocess -> display."""

import time

import pytest

from yolo_assistant.app.core.job import JobState


@pytest.fixture
def page(qapp):
    from yolo_assistant.app.ui.steps.demo_step import DemoStep

    widget = DemoStep()
    yield widget
    widget.shutdown()


def pump(app, predicate, timeout=40.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


def test_a_run_populates_the_metrics_table(qapp, page):
    page._epochs.setValue(3)
    page._batches.setValue(10)
    page._epoch_seconds.setValue(0.3)
    page._start()

    assert pump(qapp, lambda: not page._runner.is_active()), "run never finished"
    assert page._runner.state is JobState.SUCCEEDED
    assert page._table.rowCount() == 3, "one row per epoch"
    assert page.is_complete()

    # mAP50 column is populated and improving.
    first = float(page._table.item(0, 1).text())
    last = float(page._table.item(2, 1).text())
    assert 0.0 < first < 1.0 and last > first

    assert page._console.toPlainText(), "worker output was not shown"


def test_buttons_reflect_run_state(qapp, page):
    assert page._start_button.isEnabled()
    assert not page._cancel_button.isEnabled()

    page._epochs.setValue(20)
    page._epoch_seconds.setValue(1.0)
    page._start()

    assert pump(qapp, lambda: page._runner.state is JobState.RUNNING)
    assert not page._start_button.isEnabled()
    assert page._cancel_button.isEnabled()

    page._cancel()
    assert pump(qapp, lambda: not page._runner.is_active(), timeout=15.0)
    assert page._start_button.isEnabled()
    assert not page._cancel_button.isEnabled()
    assert not page.is_complete(), "a cancelled run must not count as complete"


def test_wizard_marks_completed_steps(qapp, page):
    from yolo_assistant.app.ui.wizard import WizardWindow

    window = WizardWindow([page])
    assert "Simulated run" in window._rail.item(0).text()
    assert "✓" not in window._rail.item(0).text()

    page._epochs.setValue(1)
    page._batches.setValue(5)
    page._epoch_seconds.setValue(0.2)
    page._start()
    assert pump(qapp, lambda: not page._runner.is_active())

    assert "✓" in window._rail.item(0).text(), "completed step is not ticked in the rail"
    window.close()
