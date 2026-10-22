"""Step 5's page: what it refuses, what it asks the worker for, and when it
lets the user move on.

Training itself cannot run here -- no GPU, no torch, and hours of it -- so the
one end-to-end test drives the real page, the real runner and a real subprocess
as far as this machine goes, and checks that the failure arrives as something a
user can act on. Everything else is about the decisions the page makes before
the button is pressed, which is where this step's mistakes actually live.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time

import pytest

from yolo_assistant.app.core import project as project_model
from yolo_assistant.app.core import runtime
from yolo_assistant.app.core import training as training_model
from yolo_assistant.app.core.paths import DATA_DIR_ENV
from yolo_assistant.app.core.project import Project

from test_train_worker import labelled_project

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "appdata"))
    monkeypatch.setenv(runtime.PYTHON_ENV, sys.executable)


@pytest.fixture
def project(tmp_path, environment):
    project = Project(labelled_project(tmp_path / "project"))
    project_model.set_current(project)
    return project


@pytest.fixture
def make_page(qapp, monkeypatch):
    """Builds pages on demand, so a test can set the scene first."""
    from PySide6.QtWidgets import QMessageBox

    # An unattended modal dialog blocks a test run for ever.
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    from yolo_assistant.app.ui.steps.train_step import TrainStep

    made = []

    def build():
        page = TrainStep()
        made.append(page)
        return page

    yield build
    for page in made:
        page.shutdown()


@pytest.fixture
def page(make_page, project):
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


def add_run(project: Project, name: str, *, weights=("best.pt",)):
    folder = project.runs_dir / name / "weights"
    folder.mkdir(parents=True, exist_ok=True)
    for weight in weights:
        (folder / weight).write_bytes(b"weights")


# --------------------------------------------------------------------------- #
# What it shows
# --------------------------------------------------------------------------- #


def test_it_opens_ready_to_run(page):
    assert page._plan.is_runnable
    assert page._start_button.isEnabled()
    assert "5 photos" in page._summary.text()
    assert page._runs_row.state == "unknown"


def test_the_suggested_name_does_not_replace_an_existing_run(make_page, project):
    add_run(project, "train")
    page = make_page()

    assert page._name.text() == "train2"
    assert page._warnings.isHidden()


def test_a_project_with_a_trained_model_can_move_on(make_page, project):
    page = make_page()
    assert not page.is_complete()

    add_run(project, "bottles")
    page.on_enter()

    assert page.is_complete()
    assert page._runs_row.state == "ok"
    assert "Step 6 can export" in page._runs_row._detail.text()


def test_a_run_that_saved_nothing_is_not_a_trained_model(make_page, project):
    (project.runs_dir / "crashed").mkdir(parents=True)
    page = make_page()

    assert not page.is_complete()
    assert page._runs_row.state == "warn"


# --------------------------------------------------------------------------- #
# What it refuses
# --------------------------------------------------------------------------- #


def test_a_size_off_the_detection_grid_stops_the_run(page):
    page._imgsz.setValue(700)

    assert not page._start_button.isEnabled()
    assert not page._problems.isHidden()
    assert "multiple of 32" in page._problems.text()


def test_a_misspelt_setting_is_caught_before_the_button(page):
    page._extra.setText("batch=8, patients=20")

    # Not a refusal: the app's list of settings ages, and the worker has the
    # final say against the Ultralytics that is actually installed.
    assert page._start_button.isEnabled()
    assert not page._warnings.isHidden()
    assert "patients" in page._warnings.text()


def test_a_setting_the_app_manages_is_refused_outright(page):
    page._extra.setText("data=/somewhere/else")

    assert not page._start_button.isEnabled()
    assert "set by the app" in page._problems.text()


def test_an_empty_name_stops_the_run(page):
    page._name.setText("")

    assert not page._start_button.isEnabled()
    assert "Give this run a name" in page._problems.text()


def test_a_name_that_cannot_be_a_folder_stops_the_run(page):
    page._name.setText("bottles/v2")

    assert not page._start_button.isEnabled()
    assert "cannot contain" in page._problems.text()


def test_there_is_one_size_field_and_no_talk_of_squares(page):
    page._imgsz.setValue(1024)

    assert not hasattr(page, "_imgsz_h") and not hasattr(page, "_imgsz_w")
    assert page._start_button.isEnabled()
    assert "square" not in page._warnings.text()


# --------------------------------------------------------------------------- #
# What the worker is asked for
# --------------------------------------------------------------------------- #


def test_the_job_carries_the_form(page, captured, project):
    page._name.setText("bottles")
    page._epochs.setValue(25)
    page._imgsz.setValue(640)
    page._extra.setText("batch=4")
    page._model.setCurrentIndex(1)

    page._start()

    assert len(captured) == 1
    job = captured[0].job
    assert captured[0].command == "train"
    assert job["name"] == "bottles"
    assert job["epochs"] == 25
    assert job["imgsz"] == 640
    assert job["args"] == {"batch": 4}
    assert job["model"] == training_model.MODELS[1].key
    assert job["data"] == str(project.data_yaml)


def test_each_run_gets_its_own_log_file(page, captured):
    page._name.setText("bottles")

    page._start()

    assert captured[0].log_path.name == "train-bottles.log"


def test_replacing_a_run_is_asked_about_first(make_page, project, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    add_run(project, "bottles")
    page = make_page()
    monkeypatch.setattr(page._runner, "start", lambda spec: pytest.fail("it started anyway"))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )

    page._name.setText("bottles")
    page._start()  # must return quietly, having asked and been told no


def test_the_settings_are_remembered_for_next_time(page, captured, project):
    page._epochs.setValue(42)
    page._imgsz.setValue(320)

    page._start()

    assert training_model.read_settings(project)["epochs"] == 42
    assert training_model.read_settings(project)["imgsz"] == 320


def test_settings_are_restored_when_the_project_is_opened_again(make_page, project):
    first = make_page()
    first._epochs.setValue(7)
    first._extra.setText("batch=2")
    training_model.write_settings(project, first._plan)

    second = make_page()
    assert second._epochs.value() == 7
    assert second._extra.text() == "batch=2"


def test_returning_to_the_page_does_not_undo_what_was_just_typed(page):
    page._epochs.setValue(7)

    page.on_enter()

    assert page._epochs.value() == 7


# --------------------------------------------------------------------------- #
# While it runs
# --------------------------------------------------------------------------- #


def test_metrics_arrive_with_the_best_epoch_beside_the_last(page):
    page._metrics.set_total_epochs(10)
    page._metrics.add_epoch(3, {"metrics/mAP50-95(B)": 0.61}, True)
    page._metrics.add_epoch(5, {"metrics/mAP50-95(B)": 0.52}, False)

    heading = page._metrics._heading.text()
    assert "Epoch 5 of 10" in heading
    assert "best so far was epoch 3" in heading
    # Both columns, so "is this still improving?" is answerable at a glance.
    assert page._metrics._table.item(0, 1).text() == "0.5200"
    assert page._metrics._table.item(0, 2).text() == "0.6100"


def labels_on(page) -> list[str]:
    return [
        page._metrics._table.item(row, 0).text()
        for row in range(page._metrics._table.rowCount())
    ]


def test_only_the_listed_metrics_reach_the_table(page):
    """The held-back losses, the learning rate and an end-to-end head's own L1
    and L2 terms are diagnostics for tuning a recipe, not answers to "is this
    improving?". They stay in results.csv in the run folder."""
    page._metrics.add_epoch(
        1,
        {
            "metrics/mAP50-95(B)": 0.5,
            "train/box_loss": 1.2,
            "train/cls_loss": 0.9,
            "train/l1_loss": 0.3,
            "train/l2_loss": 0.2,
            "val/cls_loss": 0.8,
            "val/dfl_loss": 0.7,
            "lr/pg0": 0.01,
            "lr/pg2": 0.01,
        },
    )

    assert labels_on(page) == ["mAP50-95", "Box loss", "Class loss"]


def test_a_metric_the_app_does_not_list_is_left_off(page):
    """Which numbers a run reports depends on the model, so rendering whatever
    turns up is a table that changes shape with the checkpoint."""
    page._metrics.add_epoch(1, {"metrics/mAP50-95(B)": 0.5, "custom/thing": 1.25})

    assert labels_on(page) == ["mAP50-95"]


def test_a_metric_this_model_never_reports_gets_no_empty_row(page):
    """A row sitting at "-" for a whole run reads as a broken display."""
    page._metrics.add_epoch(1, {"metrics/mAP50-95(B)": 0.5})

    assert "Box loss" not in labels_on(page)


def test_typing_does_not_re_read_the_dataset(page, monkeypatch):
    """Counting a large dataset's label files per keystroke is a page that
    stutters while you type. Nothing typed here can change the answer."""
    from yolo_assistant.app.ui.steps import train_step

    scans = []
    monkeypatch.setattr(
        train_step.project_model,
        "status",
        lambda project: scans.append(project) or page._status,
    )

    page._extra.setText("batch=8")
    page._epochs.setValue(30)
    page._name.setText("bottles")

    assert scans == []

    page.on_enter()
    assert len(scans) == 1


def test_the_form_is_locked_while_a_run_is_going(page, monkeypatch):
    monkeypatch.setattr(page._runner, "is_active", lambda: True)

    page._paint_buttons()

    assert not page._epochs.isEnabled()
    assert not page._model.isEnabled()
    assert not page._start_button.isEnabled()
    assert page._cancel_button.isEnabled()


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #


def test_an_unlabelled_dataset_keeps_this_step_shut(make_page, tmp_path, environment):
    from test_autolabel_worker import build_dataset

    project = Project(build_dataset(tmp_path / "half"))
    project_model.set_current(project)
    page = make_page()

    assert not page.can_enter()
    assert "no labels" in page.blocked_reason()


def test_a_half_labelled_dataset_keeps_this_step_shut(make_page, tmp_path, environment):
    from test_autolabel_worker import build_dataset

    root = build_dataset(tmp_path / "half")
    (root / "dataset" / "labels" / "cat" / "cat_000001.txt").write_text("", encoding="utf-8")
    project_model.set_current(Project(root))
    page = make_page()

    assert not page.can_enter()
    assert "4 photo(s) still have no labels" in page.blocked_reason()


def test_a_fully_labelled_dataset_opens_it(page):
    assert page.can_enter()


def test_no_managed_environment_keeps_it_shut(page, monkeypatch):
    monkeypatch.delenv(runtime.PYTHON_ENV)

    assert not page.can_enter()
    assert "Step 1" in page.blocked_reason()


# --------------------------------------------------------------------------- #
# End to end, as far as this machine can go
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(HAS_TORCH, reason="would download a checkpoint and train it")
def test_a_missing_environment_surfaces_as_something_actionable(qapp, page, project):
    """The page, the runner and a real subprocess, up to torch itself.

    The test interpreter has no torch, so the worker gets exactly as far as
    asking which device to train on and then fails. What matters is that the
    failure comes back coded and explained rather than as a silent exit, and
    that the run folder it had already created records what was asked for.
    """
    failures = []
    page._runner.failed.connect(lambda code, msg, hint, tb: failures.append((code, hint)))
    page._name.setText("bottles")

    page._start()
    assert pump(qapp, lambda: not page._runner.is_active()), "the worker never finished"

    assert failures, "the worker died without saying why"
    code, hint = failures[0]
    assert code in {"venv_broken", "no_cuda_runtime", "unknown"}
    assert hint, "a failure a beginner cannot act on is not a failure report"
    assert page._runs_row.state == "bad"
    assert not page.is_complete()

    record = json.loads((project.runs_dir / "bottles" / "umla.json").read_text())
    assert record["epochs"] == training_model.DEFAULT_EPOCHS
    assert "finished_at" not in record


def test_the_ultralytics_training_guide_is_linked(page):
    link = page._docs_link
    assert 'href="https://docs.ultralytics.com/modes/train"' in link.text()
    assert link.openExternalLinks()
    assert link.toolTip()


def test_fitness_is_not_shown_beside_the_map_it_duplicates(page):
    page._metrics.add_epoch(
        1, {"metrics/mAP50-95(B)": 0.5, "metrics/mAP50(B)": 0.7, "fitness": 0.5}, best=True
    )
    table = page._metrics._table
    names = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert "mAP50-95" in names
    assert not any("fitness" in name.lower() for name in names)


# --------------------------------------------------------------------------- #
# The results table
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("code", ["de", "pl", "ru", "be"])
def test_the_accuracy_metrics_keep_their_english_names(qapp, code):
    """They are the search terms -- in results.csv, tutorials and every answer
    online -- so they are not translated. The losses are descriptions, and are."""
    from yolo_assistant.app.core import language
    from yolo_assistant.app.ui.common.metrics_panel import MetricsPanel

    language.apply(code)
    try:
        panel = MetricsPanel()
        panel.add_epoch(
            1,
            {
                "metrics/mAP50-95(B)": 0.5,
                "metrics/mAP50(B)": 0.7,
                "metrics/precision(B)": 0.8,
                "metrics/recall(B)": 0.6,
                "train/box_loss": 1.2,
            },
            best=True,
        )
        names = [panel._table.item(row, 0).text() for row in range(panel._table.rowCount())]
    finally:
        language.apply("en")

    assert names[:4] == ["mAP50-95", "mAP50", "Precision", "Recall"]
    assert names[4] != "Box loss"
