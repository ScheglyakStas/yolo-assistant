"""Step 6's page: what it offers, what it refuses, and when it lets the user on.

Exporting cannot run here -- no torch and no GPU -- so the one end-to-end test
drives the real page, the real runner and a real subprocess as far as this
machine goes, and checks the failure arrives as something a user can act on.
Everything else is about what the page decides before the button is pressed.
"""

from __future__ import annotations

import importlib.util
import sys
import time

import pytest

from yolo_assistant.app.core import export as export_model
from yolo_assistant.app.core import project as project_model
from yolo_assistant.app.core import runtime
from yolo_assistant.app.core.paths import DATA_DIR_ENV
from yolo_assistant.app.core.project import Project

from test_export import add_export, add_run
from test_train_worker import labelled_project

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "appdata"))
    monkeypatch.setenv(runtime.PYTHON_ENV, sys.executable)


@pytest.fixture
def project(tmp_path, environment):
    project = Project(labelled_project(tmp_path / "project"))
    project.runs_dir.mkdir(parents=True, exist_ok=True)
    project.exports_dir.mkdir(parents=True, exist_ok=True)
    project_model.set_current(project)
    return project


@pytest.fixture
def make_page(qapp, monkeypatch):
    """Builds pages on demand, so a test can set the scene first."""
    from PySide6.QtWidgets import QMessageBox

    # An unattended modal dialog blocks a test run for ever.
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    from yolo_assistant.app.ui.steps.export_step import ExportStep

    made = []

    def build():
        page = ExportStep()
        made.append(page)
        return page

    yield build
    for page in made:
        page.shutdown()


@pytest.fixture
def trained(project):
    add_run(project, "bottles", imgsz=1024)
    return project


@pytest.fixture
def page(make_page, trained):
    return make_page()


@pytest.fixture
def captured(page, monkeypatch):
    """Stops the job at the point where it would launch a process."""
    specs = []
    monkeypatch.setattr(page._runner, "start", specs.append)
    return specs


def pump(app, predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


def select(combo, value) -> None:
    index = combo.findData(value)
    assert index >= 0, f"{value} is not on offer"
    combo.setCurrentIndex(index)


# --------------------------------------------------------------------------- #
# What it shows
# --------------------------------------------------------------------------- #


def test_it_opens_ready_to_run(page):
    assert page._plan.is_runnable
    assert page._start_button.isEnabled()
    assert "bottles_best_fp32.onnx" in page._start_button.text()


def test_both_checkpoints_of_a_run_are_on_offer(page):
    keys = [page._checkpoint.itemData(row) for row in range(page._checkpoint.count())]

    assert keys == ["bottles/best", "bottles/last"]


def test_the_size_defaults_to_16_9_at_the_trained_size(page):
    assert (page._imgsz_h.value(), page._imgsz_w.value()) == (576, 1024)
    assert "accuracy" not in page._warnings.text()


def test_neither_side_can_be_set_above_the_trained_size(page):
    page._imgsz_h.setValue(2048)
    page._imgsz_w.setValue(2048)

    assert page._imgsz_h.maximum() == page._imgsz_w.maximum() == 1024
    assert (page._imgsz_h.value(), page._imgsz_w.value()) == (1024, 1024)


def test_a_run_with_nothing_recorded_falls_back_to_the_ordinary_default(
    make_page, project
):
    add_run(project, "copied-in", imgsz=0)
    page = make_page()

    assert (page._imgsz_h.value(), page._imgsz_w.value()) == (352, 640)


def test_a_typed_size_survives_picking_another_checkpoint(make_page, project):
    add_run(project, "first", imgsz=1024)
    add_run(project, "second", imgsz=1024)
    page = make_page()

    page._imgsz_h.setValue(736)
    page._imgsz_w.setValue(736)
    page._checkpoint.setCurrentIndex(page._checkpoint.count() - 1)

    assert (page._imgsz_h.value(), page._imgsz_w.value()) == (736, 736)


def test_picking_a_run_trained_smaller_brings_the_size_within_it(make_page, project):
    add_run(project, "large", imgsz=1024)
    add_run(project, "small", imgsz=640)
    page = make_page()

    select(page._checkpoint, "large/best")
    page._imgsz_h.setValue(736)
    page._imgsz_w.setValue(1024)
    select(page._checkpoint, "small/best")

    assert page._imgsz_w.maximum() == 640
    assert (page._imgsz_h.value(), page._imgsz_w.value()) == (640, 640)
    assert page._plan.is_runnable


def test_an_existing_export_lets_the_user_move_on(make_page, project):
    add_run(project, "bottles")
    page = make_page()
    assert not page.is_complete()

    add_export(project, "bottles_best_fp32.onnx", format="onnx", precision="fp32")
    page.on_enter()

    assert page.is_complete()
    assert page._exports_row.state == "ok"
    assert "Step 7 can test" in page._exports_row._detail.text()


# --------------------------------------------------------------------------- #
# What it refuses
# --------------------------------------------------------------------------- #


def test_an_impossible_precision_disables_the_button(page):
    select(page._precision, export_model.INT8)

    assert not page._start_button.isEnabled()
    assert "ONNX cannot be exported at INT8" in page._problems.text()


def test_a_qat_model_is_pushed_towards_int8(make_page, project):
    """Refused here as well as caught by the worker: the run already recorded
    that it was trained for INT8, so waiting through a failing export to be
    told so is a worse way to say it."""
    add_run(project, "quantised", qat=True)
    page = make_page()

    select(page._format, "engine")
    select(page._precision, export_model.FP16)

    assert not page._start_button.isEnabled()
    assert "Choose INT8" in page._problems.text()

    select(page._precision, export_model.INT8)
    assert page._start_button.isEnabled()


def test_a_size_off_the_detection_grid_stops_the_export(page):
    page._imgsz_h.setValue(700)

    assert not page._start_button.isEnabled()
    assert "multiple of 32" in page._problems.text()


def test_a_setting_the_app_manages_is_refused_outright(page):
    page._extra.setText("opset=11")

    assert not page._start_button.isEnabled()
    assert "set by the app" in page._problems.text()


def test_a_misspelt_setting_is_caught_before_the_button(page):
    page._extra.setText("dynammic=true")

    # Not a refusal: the app's list ages, and the worker has the final say.
    assert page._start_button.isEnabled()
    assert "dynammic" in page._warnings.text()


def test_an_engine_says_it_will_not_travel(page):
    select(page._format, "engine")

    assert page._start_button.isEnabled()
    assert "will not work" in page._warnings.text()


# --------------------------------------------------------------------------- #
# What the worker is asked for
# --------------------------------------------------------------------------- #


def test_the_job_carries_the_form(page, captured, project):
    select(page._format, "engine")
    select(page._precision, export_model.FP16)
    page._imgsz_h.setValue(320)
    page._imgsz_w.setValue(640)
    page._deepstream.setChecked(True)
    page._extra.setText("batch=1")

    page._start()

    assert len(captured) == 1
    job = captured[0].job
    assert captured[0].command == "export"
    assert job["format"] == "engine"
    assert job["precision"] == "fp16"
    assert job["imgsz_hw"] == [320, 640]
    assert job["deepstream"] is True
    assert job["args"] == {"nms": False, "agnostic_nms": True, "batch": 1}
    assert job["output"] == "bottles_best_fp16.engine"
    assert job["weights"].endswith("best.pt")


def test_each_export_gets_its_own_log_file(page, captured):
    page._start()

    assert captured[0].log_path.name == "export-bottles_best_fp32.onnx.log"


def test_replacing_an_export_is_asked_about_first(make_page, project, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    add_run(project, "bottles")
    add_export(project, "bottles_best_fp32.onnx")
    page = make_page()
    monkeypatch.setattr(page._runner, "start", lambda spec: pytest.fail("it started anyway"))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)

    page._start()  # must return quietly, having asked and been told no


def test_the_settings_are_remembered_for_next_time(page, captured, project):
    select(page._format, "engine")
    select(page._precision, export_model.FP16)
    page._deepstream.setChecked(True)

    page._start()

    restored = export_model.read_settings(project)
    assert restored["format"] == "engine"
    assert restored["precision"] == "fp16"
    assert restored["deepstream"] is True


def test_a_remembered_size_wins_over_the_runs(make_page, project):
    first = make_page()
    first._imgsz_h.setValue(320)
    first._imgsz_w.setValue(320)
    export_model.write_settings(project, first._plan)

    second = make_page()

    assert (second._imgsz_h.value(), second._imgsz_w.value()) == (320, 320)


def test_the_form_is_locked_while_an_export_is_going(page, monkeypatch):
    monkeypatch.setattr(page._runner, "is_active", lambda: True)

    page._paint_buttons()

    assert not page._format.isEnabled()
    assert not page._precision.isEnabled()
    assert not page._start_button.isEnabled()
    assert page._cancel_button.isEnabled()


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #


def test_a_project_with_no_trained_model_keeps_this_step_shut(make_page, project):
    page = make_page()

    assert not page.can_enter()
    assert "no trained model" in page.blocked_reason()


def test_a_trained_model_opens_it(page):
    assert page.can_enter()


def test_no_managed_environment_keeps_it_shut(page, monkeypatch):
    monkeypatch.delenv(runtime.PYTHON_ENV)

    assert not page.can_enter()
    assert "Step 1" in page.blocked_reason()


def test_a_run_deleted_between_visits_disappears(page, trained):
    import shutil

    shutil.rmtree(trained.runs_dir / "bottles")
    page.on_enter()

    assert page._checkpoint.count() == 0
    assert not page._start_button.isEnabled()


# --------------------------------------------------------------------------- #
# End to end, as far as this machine can go
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(HAS_TORCH, reason="would load a real checkpoint")
def test_a_missing_environment_surfaces_as_something_actionable(qapp, page):
    """The page, the runner and a real subprocess, up to torch itself."""
    failures = []
    page._runner.failed.connect(lambda code, msg, hint, tb: failures.append((code, hint)))

    page._start()
    assert pump(qapp, lambda: not page._runner.is_active()), "the worker never finished"

    assert failures, "the worker died without saying why"
    code, hint = failures[0]
    assert code in {"venv_broken", "no_cuda_runtime", "unknown"}
    assert hint, "a failure a beginner cannot act on is not a failure report"
    assert page._exports_row.state == "bad"
    assert not page.is_complete()


def test_the_ultralytics_export_guide_is_linked(page):
    link = page._docs_link
    assert 'href="https://docs.ultralytics.com/modes/export"' in link.text()
    assert link.openExternalLinks()
    assert link.toolTip()
