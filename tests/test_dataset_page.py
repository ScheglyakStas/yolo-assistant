"""Step 2 end to end: page -> runner -> subprocess -> prepared dataset.

The worker runs under the test interpreter rather than a managed environment --
``prepare_dataset`` needs only Pillow, which the GUI process has -- so this
exercises the real JobRunner, the real subprocess and the real protocol without
a GPU or an 8 GB install.
"""

from __future__ import annotations

import sys
import time

import pytest
from PIL import Image

from yolo_assistant.app.core import project as project_model
from yolo_assistant.app.core import runtime
from yolo_assistant.app.core.job import JobState
from yolo_assistant.app.core.paths import DATA_DIR_ENV


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "appdata"))
    monkeypatch.setenv(runtime.PYTHON_ENV, sys.executable)


@pytest.fixture
def photos(tmp_path):
    root = tmp_path / "photos"
    for class_index, (class_name, count) in enumerate((("cat", 4), ("dog", 3))):
        directory = root / class_name
        directory.mkdir(parents=True)
        for index in range(count):
            # Every photo a different size, across classes as well as within
            # them: two flat-colour JPEGs of the same size are byte-identical,
            # and preparation would quite correctly collapse them.
            Image.new("RGB", (64 + 8 * class_index, 48 + index), (10, 120, 200)).save(
                directory / f"{class_name}_{index}.jpg"
            )
    return root


@pytest.fixture
def page(qapp, environment, monkeypatch):
    # No modal dialogs in a test run: an unattended QMessageBox blocks for ever.
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    from yolo_assistant.app.ui.steps.dataset_step import DatasetStep

    widget = DatasetStep()
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


def choose(qapp, page, source, destination=None):
    page._source_edit.setText(str(source))
    page._on_source_typed()
    if destination is not None:
        page._project_edit.setText(str(destination))
        page._on_project_typed()
    assert pump(qapp, lambda: page._scan is not None, timeout=10.0), "scan never finished"


# --------------------------------------------------------------------------- #


def test_choosing_a_folder_shows_what_is_in_it(qapp, page, photos):
    choose(qapp, page, photos)

    assert page._table.rowCount() == 2
    assert page._table.item(0, 0).text() == "cat"
    assert page._table.item(0, 1).text() == "4"
    assert "2 classes, 7 photos" in page._summary.text()
    assert page._prepare_button.isEnabled()


def test_a_project_folder_is_suggested_beside_the_photos(qapp, page, photos):
    choose(qapp, page, photos)

    assert page._project_edit.text() == str(photos.parent / "photos-project")


def test_a_typed_project_folder_is_not_overwritten(qapp, page, photos, tmp_path):
    chosen = tmp_path / "somewhere else"
    page._project_edit.setText(str(chosen))
    page._on_project_typed()

    choose(qapp, page, photos)

    assert page._project_edit.text() == str(chosen)


def test_a_broken_folder_cannot_be_prepared(qapp, page, tmp_path):
    bad = tmp_path / "flat"
    bad.mkdir()
    (bad / "loose.jpg").write_bytes(b"x")

    choose(qapp, page, bad, destination=tmp_path / "project")

    assert not page._prepare_button.isEnabled()
    assert page._problems.isVisibleTo(page)
    assert "one folder per class" in page._problems.text()


def test_preparing_produces_a_dataset_and_completes_the_step(qapp, page, photos, tmp_path):
    project_root = tmp_path / "project"
    choose(qapp, page, photos, destination=project_root)
    assert not page.is_complete()

    page._prepare()
    assert pump(qapp, lambda: not page._runner.is_active()), "preparation never finished"

    assert page._runner.state is JobState.SUCCEEDED
    dataset = project_root / "dataset"
    assert (dataset / "images" / "cat" / "cat_000001.jpg").exists()
    assert (dataset / "data.yaml").exists()
    assert page.is_complete()
    assert page._console.toPlainText(), "the worker's output was not shown"


def test_the_prepared_project_is_remembered_for_the_next_session(qapp, page, photos, tmp_path):
    project_root = tmp_path / "project"
    choose(qapp, page, photos, destination=project_root)
    page._prepare()
    assert pump(qapp, lambda: not page._runner.is_active())

    assert project_model.current().root == project_root


def test_a_failed_preparation_is_not_remembered(qapp, page, photos, tmp_path):
    # A project folder inside the photo folder is refused by the worker.
    choose(qapp, page, photos, destination=photos / "inside")

    page._prepare()
    assert pump(qapp, lambda: not page._runner.is_active())

    assert page._runner.state is JobState.FAILED
    assert not page.is_complete()
    assert project_model.current() is None, "a failed run must not become the current project"


def test_the_status_row_reports_the_finished_dataset(qapp, page, photos, tmp_path):
    choose(qapp, page, photos, destination=tmp_path / "project")
    page._prepare()
    assert pump(qapp, lambda: not page._runner.is_active())

    assert page._dataset_row.state == "ok"
    assert "7 photos" in page._dataset_row._detail.text()
    assert page._prepare_button.text() == "Prepare again"


def test_the_page_is_locked_without_a_python_environment(qapp, page, monkeypatch):
    monkeypatch.delenv(runtime.PYTHON_ENV)
    monkeypatch.setattr(runtime, "managed_environment", lambda: None)

    assert not page.can_enter()
    assert "Step 1" in page.blocked_reason()


def test_the_wizard_locks_step_2_until_the_environment_exists(qapp, page, monkeypatch):
    from yolo_assistant.app.ui.steps.base import StepPage
    from yolo_assistant.app.ui.wizard import WizardWindow

    monkeypatch.delenv(runtime.PYTHON_ENV)
    monkeypatch.setattr(runtime, "managed_environment", lambda: None)

    first = StepPage()
    first.title = "System requirements"
    window = WizardWindow([first, page])
    try:
        assert "(locked)" in window._rail.item(1).text()
        assert not window._next.isEnabled()
    finally:
        window.close()


def test_losing_focus_does_not_rescan(qapp, page, photos, monkeypatch):
    choose(qapp, page, photos)
    scans = []
    monkeypatch.setattr(page, "start_scan", lambda root: scans.append(root))

    page._on_source_typed()  # what editingFinished does on a plain focus change

    assert scans == [], "walking the folder again would learn nothing new"


def test_choosing_the_same_folder_again_does_rescan(qapp, page, photos, monkeypatch):
    choose(qapp, page, photos)
    scans = []
    monkeypatch.setattr(page, "start_scan", lambda root: scans.append(root))

    # Re-picking a folder is how a user asks the app to look again after
    # adding photos to it.
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getExistingDirectory", lambda *a, **k: str(photos)
    )
    page._choose_source()

    assert scans == [photos]
