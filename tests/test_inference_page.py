"""Step 7's page: what it offers, what it refuses, and what it shows.

No model runs here. What is tested is everything around that: which files are
listed, which device and source combinations are allowed through, what the
worker would be handed, what arrives back on screen, and the two behaviours
this page has that no other page does -- stopping when the user navigates away,
and looking for cameras in a job of its own.
"""

from __future__ import annotations

import base64
import sys
import time

import pytest

from yolo_assistant.app.core import inference as inference_model
from yolo_assistant.app.core import project as project_model
from yolo_assistant.app.core import runtime
from yolo_assistant.app.core.paths import DATA_DIR_ENV
from yolo_assistant.app.core.project import Project

from test_export import add_export
from test_train_worker import labelled_project


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "appdata"))
    monkeypatch.setenv(runtime.PYTHON_ENV, sys.executable)


@pytest.fixture
def project(tmp_path, environment):
    project = Project(labelled_project(tmp_path / "project"))
    project.exports_dir.mkdir(parents=True, exist_ok=True)
    project_model.set_current(project)
    return project


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "shelf.jpg"
    path.write_bytes(b"photo")
    return path


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "aisle.mp4"
    path.write_bytes(b"video")
    return path


@pytest.fixture
def make_page(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)

    from yolo_assistant.app.ui.steps.inference_step import InferenceStep

    made = []

    def build():
        page = InferenceStep()
        made.append(page)
        return page

    yield build
    for page in made:
        page.shutdown()


@pytest.fixture
def exported(project):
    add_export(
        project,
        "bottles_best_fp32.onnx",
        format="onnx",
        precision="fp32",
        imgsz_hw=[640, 640],
        run="bottles",
        which="best",
    )
    return project


@pytest.fixture
def page(make_page, exported):
    return make_page()


@pytest.fixture
def captured(page, monkeypatch):
    """Stops the job at the point where it would launch a process."""
    specs = []
    monkeypatch.setattr(page._runner, "start", specs.append)
    return specs


def select(combo, value) -> None:
    index = combo.findData(value)
    assert index >= 0, f"{value} is not on offer"
    combo.setCurrentIndex(index)


def choose(page, path) -> None:
    """Pick a file, as the file dialog would."""
    page._sources[page._current_mode()] = str(path)
    page._source.setText(str(path))
    page.refresh()


def a_frame(width: int = 32, height: int = 24) -> str:
    """One real PNG, base64, so the decoding is tested rather than mocked.

    The ``QByteArray`` is held in a local on purpose: handing a temporary to
    ``QBuffer`` leaves the buffer pointing at something Python has already
    collected, and PySide6 follows the pointer.
    """
    from PySide6.QtCore import QBuffer, QByteArray
    from PySide6.QtGui import QColor, QImage

    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("#336699"))
    store = QByteArray()
    buffer = QBuffer(store)
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    assert image.save(buffer, "PNG")
    buffer.close()
    return base64.b64encode(bytes(store)).decode("ascii")


# --------------------------------------------------------------------------- #
# What it offers
# --------------------------------------------------------------------------- #


def test_it_lists_what_step_6_produced(page):
    keys = [page._model.itemData(row) for row in range(page._model.count())]

    assert keys == ["bottles_best_fp32.onnx"]


def test_it_will_not_start_until_a_photo_is_chosen(page):
    assert not page._start_button.isEnabled()
    assert "Choose a photo" in page._problems.text()


def test_choosing_a_photo_makes_it_runnable(page, photo):
    choose(page, photo)

    assert page._start_button.isEnabled()
    assert not page._problems.isVisible()


def test_the_camera_row_replaces_the_file_row(page, monkeypatch):
    monkeypatch.setattr(page, "_scan_cameras", lambda: None)

    select(page._mode, inference_model.CAMERA)

    assert page._source_stack.currentIndex() == 1


def test_the_camera_mode_row_is_there_only_for_a_camera(page, monkeypatch):
    monkeypatch.setattr(page, "_scan_cameras", lambda: None)
    assert not page._form.isRowVisible(page._camera_mode_row)

    select(page._mode, inference_model.CAMERA)
    assert page._form.isRowVisible(page._camera_mode_row)

    select(page._mode, inference_model.VIDEO)
    assert not page._form.isRowVisible(page._camera_mode_row)


def test_the_chosen_camera_mode_is_what_the_job_asks_for(page, captured, monkeypatch):
    monkeypatch.setattr(page, "_scan_cameras", lambda: None)
    select(page._mode, inference_model.CAMERA)
    page._on_cameras_found(True, {"cameras": [{"index": 0, "name": "Integrated Webcam"}]})
    select(page._camera_size, "1280x720")
    select(page._camera_fps, 15)

    page._start()

    assert captured[0].job["camera_size"] == [1280, 720]
    assert captured[0].job["camera_fps"] == 15


def test_the_photo_and_the_video_are_kept_apart(page, photo, video):
    choose(page, photo)
    select(page._mode, inference_model.VIDEO)
    choose(page, video)
    select(page._mode, inference_model.PHOTO)

    assert page._source.text() == str(photo)


def test_an_engine_on_the_processor_is_refused_in_the_page(page, project, photo):
    add_export(project, "bottles_best_int8.engine", format="engine", precision="int8")
    page.on_enter()
    select(page._model, "bottles_best_int8.engine")
    choose(page, photo)

    select(page._device, inference_model.CPU)

    assert not page._start_button.isEnabled()
    assert "TensorRT" in page._problems.text()


def test_a_file_deleted_between_visits_disappears(page, project, photo):
    choose(page, photo)
    (project.exports_dir / "bottles_best_fp32.onnx").unlink()

    page.on_enter()

    assert page._model.count() == 0
    assert "Step 6" in page._problems.text()


# --------------------------------------------------------------------------- #
# What the worker is asked for
# --------------------------------------------------------------------------- #


def test_start_launches_the_infer_command(page, photo, captured):
    choose(page, photo)

    page._start()

    assert len(captured) == 1
    assert captured[0].command == "infer"
    assert captured[0].job["source"] == str(photo)
    assert captured[0].job["mode"] == inference_model.PHOTO


def test_the_chosen_device_reaches_the_job(page, photo, captured):
    choose(page, photo)
    select(page._device, inference_model.CPU)

    page._start()

    assert captured[0].job["device"] == inference_model.CPU


def test_what_was_chosen_is_remembered_for_next_time(page, photo, captured, project):
    choose(page, photo)
    select(page._device, inference_model.CPU)
    page._confidence.setValue(0.4)

    page._start()

    settings = inference_model.read_settings(project)
    assert settings["photo"] == str(photo)
    assert settings["device"] == inference_model.CPU
    assert settings["conf"] == 0.4


def test_a_second_visit_restores_the_last_choices(make_page, exported, photo, monkeypatch):
    page = make_page()
    monkeypatch.setattr(page._runner, "start", lambda spec: None)
    choose(page, photo)
    select(page._device, inference_model.CPU)
    page._start()

    revisited = make_page()

    assert revisited._device.currentData() == inference_model.CPU
    assert revisited._source.text() == str(photo)


def test_nothing_is_started_when_the_plan_is_not_runnable(page, captured):
    page._start()

    assert captured == []


# --------------------------------------------------------------------------- #
# What comes back
# --------------------------------------------------------------------------- #


def test_a_frame_is_shown(page):
    page._on_preview(
        {
            "index": 0,
            "image": a_frame(),
            "width": 32,
            "height": 24,
            "detections": [{"name": "cat", "conf": 0.91}],
            "timings": {"inference_ms": 8.2},
            "stats": {"frames": 1, "fps": 24.0, "inference_ms_mean": 8.2},
        }
    )

    assert page._canvas.has_frame
    assert "cat 0.91" in page._found.text()


def test_a_corrupt_frame_costs_one_picture_and_nothing_else(page):
    page._on_preview({"index": 0, "image": "not base64 at all", "stats": {"frames": 3}})

    assert not page._canvas.has_frame  # nothing was painted
    assert page._found.text()  # and the rest of the message still landed


def test_an_empty_frame_says_so_rather_than_showing_nothing(page):
    page._on_preview({"index": 4, "image": a_frame(), "detections": []})

    assert "Nothing found" in page._found.text()


def test_the_last_frame_stays_on_screen_when_the_run_ends(page):
    page._on_preview({"index": 0, "image": a_frame()})

    page._on_finished(True, {"mode": "photo", "frames": 1, "inference_ms_mean": 8.0})

    assert page._canvas.has_frame


def test_finishing_marks_the_step_done(page):
    assert not page.is_complete()

    page._on_finished(True, {"mode": "photo", "frames": 1, "inference_ms_mean": 8.0})

    assert page.is_complete()


def test_a_stopped_camera_is_not_a_failure(page):
    page._on_finished(True, {"mode": "camera", "frames": 240, "inference_ms_mean": 9.0,
                             "stopped": True})

    assert page.is_complete()


def test_the_startup_cost_is_shown_apart_from_the_per_frame_numbers(page):
    page._on_finished(
        True, {"mode": "photo", "frames": 1, "inference_ms_mean": 8.0, "startup_ms": 2400.0}
    )

    assert "2.4 s" in page._speed._footnote.text()


def test_a_video_is_not_shown_while_it_is_being_run(page, video):
    """On a fast card the frames arrive far faster than they were filmed."""
    select(page._mode, "video")
    choose(page, video)

    page._on_preview(
        {"index": 0, "image": a_frame(), "stats": {"frames": 1, "fps": 400.0}}
    )

    assert not page._canvas.has_frame
    assert "once that is done" in page._canvas._message


def test_a_finished_video_plays_its_annotated_copy(page, video, tmp_path):
    if page._player is None:
        pytest.skip("this Qt build has no multimedia module")
    select(page._mode, "video")
    choose(page, video)
    recording = tmp_path / "aisle__bottles.mp4"
    recording.write_bytes(b"")

    page._on_finished(
        True, {"mode": "video", "frames": 3, "inference_ms_mean": 2.0, "output": str(recording)}
    )

    assert page._picture.currentWidget() is page._player
    assert page._player.path == recording
    assert "Save As" in page._found.text()


def test_starting_again_lets_go_of_the_last_recording(page, video, tmp_path, captured):
    """The next run may overwrite that file, and the player must not be holding it."""
    if page._player is None:
        pytest.skip("this Qt build has no multimedia module")
    select(page._mode, "video")
    choose(page, video)
    recording = tmp_path / "aisle__bottles.mp4"
    recording.write_bytes(b"")
    page._on_finished(True, {"mode": "video", "frames": 1, "output": str(recording)})

    page._start()

    assert page._picture.currentWidget() is page._canvas
    assert page._player.path is None


# --------------------------------------------------------------------------- #
# The temporary copy, and Save As
# --------------------------------------------------------------------------- #


def a_copy(project, name="shelf__bottles_best_fp32.jpg"):
    path = inference_model.temp_dir(project) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"annotated")
    return path


def finish_photo(page, project, photo):
    choose(page, photo)
    copy = a_copy(project)
    page._on_finished(
        True, {"mode": "photo", "frames": 1, "source": str(photo), "output": str(copy)}
    )
    return copy


def pick(monkeypatch, path, method="getOpenFileName"):
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, method, lambda *a, **k: (str(path), ""))


def test_save_as_is_offered_once_there_is_a_copy(page, project, photo):
    assert not page._save_button.isEnabled()

    finish_photo(page, project, photo)

    assert page._save_button.isEnabled()


def test_choosing_another_file_deletes_the_copy(page, project, photo, tmp_path, monkeypatch):
    copy = finish_photo(page, project, photo)
    other = tmp_path / "other.jpg"
    other.write_bytes(b"photo")
    pick(monkeypatch, other)

    page._choose_source()

    assert not copy.exists()
    assert not page._save_button.isEnabled()
    assert not page._canvas.has_frame


def test_choosing_the_same_file_again_keeps_the_copy(page, project, photo, monkeypatch):
    copy = finish_photo(page, project, photo)
    pick(monkeypatch, photo)

    page._choose_source()

    assert copy.exists()
    assert page._save_button.isEnabled()


def test_cancelling_the_file_dialog_keeps_the_copy(page, project, photo, monkeypatch):
    copy = finish_photo(page, project, photo)
    pick(monkeypatch, "")

    page._choose_source()

    assert copy.exists()


def test_starting_again_deletes_the_last_copy(page, project, photo, captured):
    copy = finish_photo(page, project, photo)

    page._start()

    assert captured
    assert not copy.exists()


def test_closing_the_app_deletes_the_copy(page, project, photo):
    copy = finish_photo(page, project, photo)

    page.shutdown()

    assert not copy.exists()


def test_a_finished_video_copy_is_deleted_after_the_player_lets_go(
    page, project, video, tmp_path, monkeypatch
):
    select(page._mode, "video")
    choose(page, video)
    copy = a_copy(project, "aisle__bottles_best_fp32.mp4")
    page._on_finished(
        True, {"mode": "video", "frames": 3, "source": str(video), "output": str(copy)}
    )
    other = tmp_path / "other.mp4"
    other.write_bytes(b"video")
    pick(monkeypatch, other)

    page._choose_source()

    assert not copy.exists()
    if page._player is not None:
        assert page._player.path is None
        assert page._picture.currentWidget() is page._canvas


def test_leftovers_from_a_crashed_session_are_swept_on_arrival(make_page, exported):
    stale = a_copy(exported, "old__model.mp4")

    make_page()

    assert not stale.exists()


def test_save_as_copies_where_the_user_chose(page, project, photo, tmp_path, monkeypatch):
    copy = finish_photo(page, project, photo)
    destination = tmp_path / "kept" / "result.jpg"
    pick(monkeypatch, destination, "getSaveFileName")

    page._save_output()

    assert destination.read_bytes() == b"annotated"
    assert str(destination) in page._saved.text()


def test_a_saved_file_outlives_the_temporary_copy(page, project, photo, tmp_path, monkeypatch):
    finish_photo(page, project, photo)
    destination = tmp_path / "result.jpg"
    pick(monkeypatch, destination, "getSaveFileName")
    page._save_output()
    other = tmp_path / "other.jpg"
    other.write_bytes(b"photo")
    pick(monkeypatch, other)

    page._choose_source()
    page.shutdown()

    assert destination.is_file()


def test_save_as_refuses_the_temp_folder(page, project, photo, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[1]))
    finish_photo(page, project, photo)
    pick(monkeypatch, inference_model.temp_dir(project) / "keep.jpg", "getSaveFileName")

    page._save_output()

    assert warned == ["Choose another folder"]
    assert not (inference_model.temp_dir(project) / "keep.jpg").exists()


def test_a_failure_replaces_the_picture_with_the_reason(page):
    page._on_failed("camera_unavailable", "Camera 0 could not be opened.", "Close Zoom.", "")

    assert not page._canvas.has_frame
    assert "Camera 0" in page._canvas._message


def test_the_silent_cpu_fallback_is_repeated_where_the_numbers_are(page):
    page._on_warned(
        inference_model.ONNX_ON_CPU_WARNING,
        "onnxruntime has no CUDA provider in this environment, so this ONNX model "
        "runs on the processor even though the GPU was chosen.",
        "",
    )

    assert "processor timings" in page._speed._footnote.text()


ASPECT_MESSAGE = (
    "The model was exported at 640x640, but this input is 1920x1080 -- a different "
    "aspect ratio. Inference is slower than it needs to be."
)


def test_an_aspect_mismatch_stays_on_screen_beside_the_picture(page, photo, captured):
    choose(page, photo)
    page._start()

    page._on_warned(inference_model.ASPECT_MISMATCH_WARNING, ASPECT_MESSAGE, "")
    page._on_finished(True, {"mode": inference_model.PHOTO})

    assert not page._warnings.isHidden()
    assert "1920x1080" in page._warnings.text()
    assert "1920x1080" in page._console.toPlainText()


def test_an_aspect_mismatch_goes_when_the_input_changes(
    page, photo, video, captured
):
    choose(page, photo)
    page._start()
    page._on_warned(inference_model.ASPECT_MISMATCH_WARNING, ASPECT_MESSAGE, "")

    choose(page, video.with_suffix(".png"))

    assert "1920x1080" not in page._warnings.text()


def test_starting_again_clears_the_last_aspect_mismatch(page, photo, captured):
    choose(page, photo)
    page._start()
    page._on_warned(inference_model.ASPECT_MISMATCH_WARNING, ASPECT_MESSAGE, "")

    page._start()

    assert "1920x1080" not in page._warnings.text()


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #


def test_the_camera_list_is_filled_from_the_scan(page):
    page._on_cameras_found(
        True, {"cameras": [{"index": 0, "name": "Integrated Webcam", "width": 640, "height": 480}]}
    )

    assert page._camera.count() == 1
    assert "Integrated Webcam" in page._camera.itemText(0)


def test_switching_to_camera_mode_looks_for_cameras_once(page, monkeypatch):
    scans = []
    monkeypatch.setattr(page._scanner, "start", scans.append)

    select(page._mode, inference_model.CAMERA)
    select(page._mode, inference_model.PHOTO)
    select(page._mode, inference_model.CAMERA)

    assert len(scans) == 1
    assert scans[0].command == "cameras"


def test_a_page_that_opens_in_camera_mode_looks_for_cameras(make_page, exported, project):
    """Restoring the mode fires no signal, so nothing else would trigger it."""
    from yolo_assistant.app.ui.steps.inference_step import InferenceStep

    inference_model.write_settings(
        project, inference_model.build_plan(project=project, mode=inference_model.CAMERA)
    )
    scans = []
    original = InferenceStep._scan_cameras
    try:
        InferenceStep._scan_cameras = lambda self: scans.append(True)
        make_page()
    finally:
        InferenceStep._scan_cameras = original

    assert scans, "the page opened in camera mode with no way to find a camera"


def test_cameras_are_not_looked_for_while_a_test_is_running(page, monkeypatch):
    scans = []
    monkeypatch.setattr(page._scanner, "start", scans.append)
    monkeypatch.setattr(page._runner, "is_active", lambda: True)

    page._scan_cameras()

    assert scans == []


# --------------------------------------------------------------------------- #
# Gating and lifecycle
# --------------------------------------------------------------------------- #


def test_it_is_locked_until_something_has_been_exported(make_page, project):
    page = make_page()

    assert not page.can_enter()
    assert "Step 6" in page.blocked_reason()


def test_one_exported_file_unlocks_it(page):
    assert page.can_enter()


def test_it_is_locked_without_a_managed_environment(page, monkeypatch):
    monkeypatch.setattr(runtime, "worker_python", lambda: None)

    assert not page.can_enter()
    assert "Step 1" in page.blocked_reason()


def test_leaving_the_page_stops_the_run(page, monkeypatch):
    """Unlike every other step: nothing is lost, and a camera is held for nothing."""
    cancelled = []
    monkeypatch.setattr(page._runner, "is_active", lambda: True)
    monkeypatch.setattr(page._runner, "cancel", lambda: cancelled.append(True))

    page.on_leave()

    assert cancelled == [True]


def test_closing_the_app_stops_both_jobs(page, monkeypatch):
    stopped = []
    for runner, name in ((page._runner, "test"), (page._scanner, "scan")):
        monkeypatch.setattr(runner, "is_active", lambda: True)
        monkeypatch.setattr(runner, "cancel", lambda name=name: stopped.append(name))

    page.shutdown()

    assert sorted(stopped) == ["scan", "test"]


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def pump(app, predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


def test_a_machine_that_cannot_run_it_says_so_in_words(qapp, page, photo):
    """The page, the runner and a real subprocess, as far as this machine goes.

    Nothing here can actually infer -- the GUI's own interpreter has no
    onnxruntime and no torch -- so what is being checked is that the failure
    comes back coded, with a remedy, rather than as a traceback or as silence.
    """
    failures = []
    page._runner.failed.connect(lambda code, msg, hint, tb: failures.append((code, hint)))
    choose(page, photo)

    page._start()
    assert pump(qapp, lambda: not page._runner.is_active()), "the worker never finished"

    assert failures, "the worker died without saying why"
    code, hint = failures[0]
    assert code in {"inference_failed", "no_cuda_runtime", "venv_broken", "unknown"}
    assert hint, "a failure a beginner cannot act on is not a failure report"
    assert not page.is_complete()
