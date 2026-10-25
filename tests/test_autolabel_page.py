"""Step 3's page: what it shows, what it refuses, and what it asks the worker for.

The model cannot run here, so the one end-to-end test drives the real page, the
real runner and a real subprocess up to the point where OWLv2 would be loaded,
and checks that the failure arrives as something a user can act on. Everything
else is about the numbers on screen, which come off the filesystem and change
while the page is not looking.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time

import pytest

from yolo_assistant.app.core import autolabel as autolabel_model
from yolo_assistant.app.core import project as project_model
from yolo_assistant.app.core import runtime
from yolo_assistant.app.core.paths import DATA_DIR_ENV
from yolo_assistant.app.core.project import Project

from test_autolabel_worker import build_dataset

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "appdata"))
    monkeypatch.setenv(runtime.PYTHON_ENV, sys.executable)


@pytest.fixture
def project(tmp_path, environment):
    root = build_dataset(tmp_path / "project", classes=(("cat", 3), ("dog", 2)))
    project = Project(root)
    project_model.set_current(project)
    return project


@pytest.fixture
def make_page(qapp, monkeypatch):
    """Builds pages on demand, so a test can set the scene first."""
    from PySide6.QtWidgets import QMessageBox

    # An unattended modal dialog blocks a test run for ever.
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    from yolo_assistant.app.ui.steps.autolabel_step import AutolabelStep

    made = []

    def build():
        page = AutolabelStep()
        made.append(page)
        return page

    yield build
    for page in made:
        page.shutdown()


@pytest.fixture
def page(make_page, project):
    return make_page()


def prompt_cell(page, row: int):
    from yolo_assistant.app.ui.steps.autolabel_step import PROMPT_COLUMN

    return page._table.item(row, PROMPT_COLUMN)


def describe_every_class(page) -> None:
    """Types a prompt for every class, as the user has to before Start works."""
    for row in range(page._table.rowCount()):
        name = page._table.item(row, 0).text()
        prompt_cell(page, row).setText(f"a photo of a {name}")


def label_everything(project: Project) -> None:
    for images in sorted(project.images_dir.iterdir()):
        labels = project.class_labels_dir(images.name)
        labels.mkdir(parents=True, exist_ok=True)
        for photo in images.iterdir():
            (labels / (photo.stem + ".txt")).write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def pump(app, predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


# --------------------------------------------------------------------------- #
# What it shows
# --------------------------------------------------------------------------- #


def test_every_class_arrives_with_an_empty_prompt(page):
    assert page._table.rowCount() == 2
    assert page._table.item(0, 0).text() == "cat"
    # Nothing is filled in for the user: a suggested prompt was too easy to
    # run without reading, so Start waits until every one has been written.
    assert prompt_cell(page, 0).text() == ""
    assert prompt_cell(page, 1).text() == ""
    assert not page._start_button.isEnabled()
    assert '"cat"' in page._problems.text()
    assert '"dog"' in page._problems.text()


def test_the_button_says_how_many_photos_it_will_run(page):
    describe_every_class(page)

    assert page._start_button.isEnabled()
    assert page._start_button.text() == "Label 5 photos"


def test_a_class_with_no_description_blocks_the_run(page):
    describe_every_class(page)
    prompt_cell(page, 0).setText("   ")

    assert not page._start_button.isEnabled()
    assert page._problems.isVisibleTo(page)
    assert '"cat"' in page._problems.text()


def test_an_already_labelled_class_needs_no_description(page, project):
    for photo in project.class_images_dir("dog").iterdir():
        (project.class_labels_dir("dog") / (photo.stem + ".txt")).write_text("")
    page.on_enter()

    prompt_cell(page, 0).setText("a photo of a cat")

    assert page._start_button.isEnabled(), "dog is finished; its prompt is irrelevant"
    assert page._start_button.text() == "Label 3 photos"


def test_labels_made_elsewhere_are_noticed_on_returning(page, project):
    assert not page.is_complete()

    label_everything(project)
    page.on_enter()

    # Nothing recorded that Step 3 was done; the files are the answer.
    assert page.is_complete()
    assert page._labels_row.state == "ok"
    assert not page._start_button.isEnabled()


def test_prompts_from_a_previous_run_come_back(make_page, project):
    autolabel_model.record_path(project).write_text(
        json.dumps(
            {
                "threshold": 0.4,
                "max_detections": 2,
                "classes": [{"slug": "cat", "prompt": "a tabby cat on a sofa"}],
            }
        ),
        encoding="utf-8",
    )

    page = make_page()

    assert prompt_cell(page, 0).text() == "a tabby cat on a sofa"
    assert prompt_cell(page, 1).text() == "", "unrecorded classes are left for the user"
    assert page._threshold.value() == pytest.approx(0.4)
    assert page._max_detections.value() == 2


# --------------------------------------------------------------------------- #
# What it asks for
# --------------------------------------------------------------------------- #


def test_the_job_says_what_the_page_says(page, monkeypatch):
    captured = {}
    monkeypatch.setattr(type(page._runner), "start", lambda self, spec: captured.update(spec=spec))
    describe_every_class(page)
    prompt_cell(page, 0).setText("a ginger cat")
    page._threshold.setValue(0.3)

    page._start()

    spec = captured["spec"]
    assert spec.command == "autolabel"
    assert spec.job["threshold"] == pytest.approx(0.3)
    assert spec.job["classes"][0] == {
        "index": 0,
        "name": "cat",
        "slug": "cat",
        "prompt": "a ginger cat",
    }
    assert spec.project_dir == page.project.root
    assert spec.log_path == page.project.log_path("autolabel")


def test_relabelling_asks_before_replacing_labels(page, project, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    label_everything(project)
    page.on_enter()
    started = []
    monkeypatch.setattr(type(page._runner), "start", lambda self, spec: started.append(spec))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)

    page._relabel.setChecked(True)
    describe_every_class(page)
    assert page._start_button.isEnabled()
    page._start()

    # Those labels may have been corrected by hand in Step 4.
    assert started == []


def test_settings_survive_a_visit_to_another_step(page):
    page._threshold.setValue(0.42)

    page.on_leave()
    page.on_enter()

    assert page._threshold.value() == pytest.approx(0.42)


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #


def test_the_page_is_locked_until_the_dataset_is_prepared(make_page, environment, tmp_path):
    project_model.set_current(Project(tmp_path / "not-prepared"))
    page = make_page()

    assert not page.can_enter()
    assert "Step 2" in page.blocked_reason()


def test_the_page_is_locked_without_a_python_environment(page, monkeypatch):
    monkeypatch.delenv(runtime.PYTHON_ENV)
    monkeypatch.setattr(runtime, "managed_environment", lambda: None)

    assert not page.can_enter()
    assert "Step 1" in page.blocked_reason()


def test_the_wizard_locks_step_3_until_step_2_has_run(make_page, environment, tmp_path):
    from yolo_assistant.app.ui.steps.base import StepPage
    from yolo_assistant.app.ui.wizard import WizardWindow

    project_model.set_current(None)
    page = make_page()
    earlier = StepPage()
    earlier.title = "Dataset preparation"

    window = WizardWindow([earlier, page])
    try:
        assert "(locked)" in window._rail.item(1).text()
        assert not window._next.isEnabled()
    finally:
        window.close()


# --------------------------------------------------------------------------- #
# End to end, as far as this machine can go
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(HAS_TORCH, reason="would download OWLv2 and run it")
def test_a_missing_environment_surfaces_as_something_actionable(qapp, page, project):
    """The page, the runner and a real subprocess, up to the model itself.

    The test interpreter has no torch, so the worker gets exactly as far as
    loading OWLv2 and then fails. What matters is that the failure comes back
    coded and explained rather than as a silent exit -- and that the prompts
    the user typed are on disk regardless.
    """
    failures = []
    page._runner.failed.connect(lambda code, msg, hint, tb: failures.append((code, hint)))
    describe_every_class(page)

    page._start()
    assert pump(qapp, lambda: not page._runner.is_active()), "the worker never finished"

    assert failures, "the worker died without saying why"
    code, hint = failures[0]
    assert code in {"venv_broken", "download_failed", "unknown"}
    assert hint, "a failure a beginner cannot act on is not a failure report"
    assert page._labels_row.state == "bad"
    assert not page.is_complete()
    assert autolabel_model.prompts_from(autolabel_model.read_record(project)) == {
        "cat": "a photo of a cat",
        "dog": "a photo of a dog",
    }
