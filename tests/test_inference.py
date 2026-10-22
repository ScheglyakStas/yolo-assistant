"""What Step 7 decides before anything is loaded.

The model cannot be run here -- no torch, no onnxruntime, no camera -- and the
decisions worth testing are all made before it would be. Which files are
offered, which of them can run where, what a missing or wrong source is called,
and what the worker is handed.
"""

from __future__ import annotations

import json

import pytest

from yolo_assistant.app.core import export as export_model
from yolo_assistant.app.core import inference as inference_model
from yolo_assistant.app.core.inference import CAMERA, CPU, GPU, PHOTO, VIDEO
from yolo_assistant.app.core.project import Project

from test_export import add_export


@pytest.fixture
def project(tmp_path):
    project = Project(tmp_path / "project")
    project.exports_dir.mkdir(parents=True)
    project.dataset_dir.mkdir(parents=True, exist_ok=True)
    project.classes_file.write_text("cat\ndog\n", encoding="utf-8")
    return project


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "shelf.jpg"
    path.write_bytes(b"not really a jpeg, and never opened here")
    return path


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "aisle.mp4"
    path.write_bytes(b"not really an mp4")
    return path


def plan_for(project, **kwargs):
    settings = {"mode": PHOTO}
    settings.update(kwargs)
    return inference_model.build_plan(project=project, **settings)


def add_folder_export(project, name: str, **record) -> None:
    """A format that exports as a folder rather than a file, as OpenVINO does."""
    folder = project.exports_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.xml").write_bytes(b"model")
    folder.with_name(folder.name + export_model.RECORD_SUFFIX).write_text(
        json.dumps(record), encoding="utf-8"
    )


def onnx(project, name="bottles_best_fp32.onnx"):
    add_export(project, name, format="onnx", precision="fp32", imgsz_hw=[640, 640])
    return name


# --------------------------------------------------------------------------- #
# What there is to test
# --------------------------------------------------------------------------- #


def test_nothing_exported_is_the_only_thing_it_says(project):
    problems = plan_for(project).problems()

    assert len(problems) == 1
    assert "Step 6" in problems[0]


def test_every_exported_file_is_offered(project):
    onnx(project)
    add_export(project, "bottles_best_int8.engine", format="engine", precision="int8")

    names = [target.key for target in inference_model.targets(project)]

    assert set(names) == {"bottles_best_fp32.onnx", "bottles_best_int8.engine"}


def test_a_stray_file_in_the_exports_folder_is_not_a_model(project):
    onnx(project)
    (project.exports_dir / "notes.txt").write_text("remember to test this", encoding="utf-8")

    assert [t.key for t in inference_model.targets(project)] == ["bottles_best_fp32.onnx"]


def test_every_format_step_6_offers_can_be_asked_about():
    """A format that can be written and not run has to say so here.

    Otherwise it silently disappears from Step 7's list, and the user is left
    holding a file the app exported and will not open.
    """
    missing = [fmt.key for fmt in export_model.FORMATS if fmt.key not in inference_model.RUNTIMES]

    assert not missing, f"no runtime entry for: {missing}"


# --------------------------------------------------------------------------- #
# Where it can run
# --------------------------------------------------------------------------- #


def test_an_engine_cannot_be_run_on_the_processor(project, photo):
    add_export(project, "bottles_best_int8.engine", format="engine", precision="int8")
    plan = plan_for(project, device=CPU, source=str(photo))

    problems = " ".join(plan.problems())

    assert "TensorRT" in problems
    assert "GPU" in problems


def test_an_engine_on_the_gpu_is_fine(project, photo):
    add_export(project, "bottles_best_int8.engine", format="engine", precision="int8")

    assert plan_for(project, device=GPU, source=str(photo)).is_runnable


def test_openvino_is_refused_on_the_nvidia_gpu(project, photo):
    add_folder_export(project, "bottles_best_fp16_openvino_model", format="openvino")

    problems = " ".join(plan_for(project, device=GPU, source=str(photo)).problems())

    assert "OpenVINO" in problems
    assert "Intel" in problems


def test_onnx_runs_on_either(project, photo):
    onnx(project)

    for device in (CPU, GPU):
        assert plan_for(project, device=device, source=str(photo)).is_runnable


def test_a_format_whose_runtime_is_not_installed_warns_rather_than_refuses(project, photo):
    add_export(project, "bottles_best_fp32.tflite", format="tflite")
    plan = plan_for(project, device=CPU, source=str(photo))

    assert plan.is_runnable
    assert any("tensorflow" in warning for warning in plan.warnings())


def test_the_engine_portability_warning_is_always_there(project, photo):
    add_export(project, "bottles_best_int8.engine", format="engine")
    plan = plan_for(project, device=GPU, source=str(photo))

    assert any("built it" in warning for warning in plan.warnings())


def test_onnx_on_the_gpu_warns_about_the_silent_fallback(project, photo):
    onnx(project)
    plan = plan_for(project, device=GPU, source=str(photo))

    assert any("onnxruntime" in warning for warning in plan.warnings())


# --------------------------------------------------------------------------- #
# What it runs against
# --------------------------------------------------------------------------- #


def test_a_photo_has_to_be_chosen(project):
    onnx(project)

    assert "Choose a photo" in " ".join(plan_for(project).problems())


def test_a_photo_that_has_gone_is_said_to_have_gone(project, tmp_path):
    onnx(project)
    plan = plan_for(project, source=str(tmp_path / "deleted.jpg"))

    assert "not there any more" in " ".join(plan.problems())


def test_a_folder_is_not_a_photo(project, tmp_path):
    onnx(project)

    assert "is a folder" in " ".join(plan_for(project, source=str(tmp_path)).problems())


def test_a_video_is_not_a_photo(project, video):
    onnx(project)

    assert "not a photo" in " ".join(plan_for(project, source=str(video)).problems())


def test_a_video_is_a_video(project, video):
    onnx(project)

    assert plan_for(project, mode=VIDEO, source=str(video)).is_runnable


def test_a_photo_is_not_a_video(project, photo):
    onnx(project)
    plan = plan_for(project, mode=VIDEO, source=str(photo))

    assert "not a video" in " ".join(plan.problems())


def test_camera_mode_needs_a_camera(project):
    onnx(project)
    problems = " ".join(plan_for(project, mode=CAMERA).problems())

    assert "No camera was found" in problems


def test_camera_mode_with_a_camera_is_runnable(project):
    onnx(project)
    plan = plan_for(
        project,
        mode=CAMERA,
        cameras=(inference_model.Camera(0, "Integrated Webcam", 1280, 720),),
        camera_index=0,
    )

    assert plan.is_runnable
    assert "Integrated Webcam" in plan.summary()


def test_a_camera_that_is_gone_falls_back_to_the_first_one(project):
    onnx(project)
    cameras = (inference_model.Camera(2, "Studio"),)
    plan = plan_for(project, mode=CAMERA, cameras=cameras, camera_index=9)

    assert plan.camera is cameras[0]


@pytest.mark.parametrize("value", [0.0, 1.5])
def test_an_impossible_confidence_is_refused(project, photo, value):
    onnx(project)
    plan = plan_for(project, source=str(photo), confidence=value)

    assert any("confidence" in problem for problem in plan.problems())


# --------------------------------------------------------------------------- #
# What the worker is handed
# --------------------------------------------------------------------------- #


def test_the_job_names_the_file_the_device_and_the_source(project, photo):
    name = onnx(project)
    job = plan_for(project, device=CPU, source=str(photo), confidence=0.4).job(project)

    assert job["model"] == str(project.exports_dir / name)
    assert job["device"] == CPU
    assert job["mode"] == PHOTO
    assert job["source"] == str(photo)
    assert job["conf"] == 0.4


def test_a_video_job_says_where_its_annotated_copy_goes(project, video):
    """Into temp/, named for the clip and the model."""
    onnx(project)
    job = plan_for(project, mode=VIDEO, source=str(video)).job(project)

    assert job["output"] == str(
        project.root / "temp" / f"{video.stem}__bottles_best_fp32.mp4"
    )


def test_a_photo_job_writes_its_copy_to_temp_in_the_photos_format(project, photo):
    onnx(project)
    job = plan_for(project, source=str(photo)).job(project)

    assert job["output"] == str(project.root / "temp" / "shelf__bottles_best_fp32.jpg")


def test_a_photo_in_a_format_opencv_cannot_write_becomes_a_png(project, tmp_path, monkeypatch):
    onnx(project)
    odd = tmp_path / "shelf.jpg"
    odd.write_bytes(b"x")
    monkeypatch.setattr(inference_model, "WRITABLE_PHOTO_SUFFIXES", frozenset({".bmp"}))

    output = plan_for(project, source=str(odd)).output_path(project)

    assert output.suffix == ".png"


def test_a_camera_job_writes_nothing(project):
    onnx(project)
    plan = plan_for(
        project, mode=CAMERA, cameras=(inference_model.Camera(index=0),), camera_index=0
    )

    assert plan.job(project)["output"] == ""


# --------------------------------------------------------------------------- #
# The temporary copies
# --------------------------------------------------------------------------- #


def a_copy(project, name="shelf__bottles.jpg"):
    path = inference_model.temp_dir(project) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"annotated")
    return path


def test_a_copy_in_temp_is_deleted(project):
    copy = a_copy(project)

    assert inference_model.discard(copy, project) is True
    assert not copy.exists()


def test_a_file_outside_temp_is_never_deleted(project, photo):
    """Whatever a payload says, only temp/ is the app's to delete."""
    assert inference_model.discard(photo, project) is False
    assert photo.exists()


def test_a_path_that_climbs_out_of_temp_is_not_temporary(project, photo):
    sneaky = inference_model.temp_dir(project) / ".." / ".." / photo.name

    assert not inference_model.is_temporary(sneaky, project)
    assert inference_model.discard(sneaky, project) is False
    assert photo.exists()


def test_discarding_what_is_already_gone_is_not_an_error(project):
    assert inference_model.discard(inference_model.temp_dir(project) / "gone.mp4", project) is False
    assert inference_model.discard(None, project) is False
    assert inference_model.discard("", project) is False


def test_the_sweep_empties_temp_and_nothing_else(project, photo):
    a_copy(project, "one.jpg")
    a_copy(project, "two.partial.mp4")

    assert inference_model.sweep_temp(project) == 2
    assert list(inference_model.temp_dir(project).iterdir()) == []
    assert photo.exists()


def test_the_sweep_does_not_mind_there_being_no_temp_folder(project):
    assert inference_model.sweep_temp(project) == 0
    assert inference_model.sweep_temp(None) == 0


def test_save_as_copies_and_leaves_the_temporary_copy_in_place(project, tmp_path):
    copy = a_copy(project)
    destination = tmp_path / "kept" / "result.jpg"

    saved = inference_model.save_copy(copy, destination)

    assert saved == destination
    assert destination.read_bytes() == b"annotated"
    assert copy.exists()


def test_save_as_without_an_extension_gets_the_copys(project, tmp_path):
    copy = a_copy(project, "aisle__bottles.mp4")

    saved = inference_model.save_copy(copy, tmp_path / "result")

    assert saved == tmp_path / "result.mp4"
    assert saved.is_file()


def test_a_saved_copy_survives_the_sweep(project, tmp_path):
    saved = inference_model.save_copy(a_copy(project), tmp_path / "result.jpg")

    inference_model.sweep_temp(project)
    inference_model.discard(saved, project)

    assert saved.is_file()


def test_the_save_filter_offers_the_copys_own_format_first():
    assert inference_model.save_filter("a.mp4").startswith("Video (*.mp4)")
    assert inference_model.save_filter("a.png").startswith("Image (*.png)")


def test_the_job_carries_the_datasets_class_names(project, photo):
    """Some formats lose their names on the way out; the project still has them."""
    onnx(project)
    job = plan_for(project, source=str(photo)).job(project)

    assert job["names"] == ["cat", "dog"]


def test_the_job_says_what_the_format_needs_installed(project, photo):
    add_export(project, "bottles_best_fp32.tflite", format="tflite")
    job = plan_for(project, device=CPU, source=str(photo)).job(project)

    assert job["needs"] == ["tensorflow"]


def test_the_job_carries_the_camera_index(project):
    onnx(project)
    plan = plan_for(
        project,
        mode=CAMERA,
        cameras=(inference_model.Camera(0), inference_model.Camera(3)),
        camera_index=3,
    )

    assert plan.job(project)["camera"] == 3


def test_the_job_carries_the_chosen_camera_mode(project):
    onnx(project)
    plan = plan_for(
        project,
        mode=CAMERA,
        cameras=(inference_model.Camera(0),),
        camera_size=(1280, 720),
        camera_fps=15,
    )

    job = plan.job(project)
    assert job["camera_size"] == [1280, 720]
    assert job["camera_fps"] == 15


def test_a_camera_mode_by_default_is_1080p_at_the_camera_s_own_rate(project):
    onnx(project)
    job = plan_for(project, mode=CAMERA, cameras=(inference_model.Camera(0),)).job(project)

    assert job["camera_size"] == [1920, 1080]
    assert job["camera_fps"] == 0


def test_a_camera_mode_not_on_offer_is_replaced_by_the_default(project):
    plan = plan_for(project, mode=CAMERA, camera_size=(123, 45), camera_fps=7)

    assert plan.camera_size == inference_model.DEFAULT_CAMERA_SIZE
    assert plan.camera_fps == inference_model.DEFAULT_CAMERA_FPS


def test_the_summary_describes_the_chosen_mode_not_the_scanned_one(project):
    onnx(project)
    plan = plan_for(
        project,
        mode=CAMERA,
        cameras=(inference_model.Camera(0, "Studio", 1920, 1080, 30),),
        camera_index=0,
        camera_size=(640, 480),
        camera_fps=15,
    )

    assert "Studio (640x480, 15 fps)" in plan.summary()


# --------------------------------------------------------------------------- #
# Remembering
# --------------------------------------------------------------------------- #


def test_settings_round_trip(project, photo):
    onnx(project)
    plan = plan_for(project, device=CPU, source=str(photo), confidence=0.35)

    inference_model.write_settings(project, plan)
    settings = inference_model.read_settings(project)

    assert settings["device"] == CPU
    assert settings["mode"] == PHOTO
    assert settings["photo"] == str(photo)
    assert settings["conf"] == 0.35


def test_the_camera_mode_is_remembered(project):
    onnx(project)
    plan = plan_for(project, mode=CAMERA, camera_size=(1280, 720), camera_fps=24)

    inference_model.write_settings(project, plan)
    settings = inference_model.read_settings(project)

    assert settings["camera_size"] == (1280, 720)
    assert settings["camera_fps"] == 24


def test_a_remembered_camera_mode_that_is_nonsense_is_forgotten(project):
    path = inference_model.settings_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"camera_size": "huge", "camera_fps": 1000}', encoding="utf-8")

    settings = inference_model.read_settings(project)

    assert settings["camera_size"] == inference_model.DEFAULT_CAMERA_SIZE
    assert settings["camera_fps"] == inference_model.DEFAULT_CAMERA_FPS


def test_the_photo_and_the_video_are_remembered_separately(project, photo, video):
    onnx(project)
    inference_model.write_settings(project, plan_for(project, source=str(photo)))
    previous = inference_model.read_settings(project)
    inference_model.write_settings(
        project, plan_for(project, mode=VIDEO, source=str(video)), previous=previous
    )

    settings = inference_model.read_settings(project)

    assert settings["photo"] == str(photo)
    assert settings["video"] == str(video)
    assert settings["mode"] == VIDEO


def test_nonsense_settings_fall_back_to_the_defaults(project):
    inference_model.settings_path(project).parent.mkdir(parents=True, exist_ok=True)
    inference_model.settings_path(project).write_text(
        json.dumps({"device": "quantum", "mode": "telepathy", "conf": "soon"}),
        encoding="utf-8",
    )

    settings = inference_model.read_settings(project)

    assert settings["device"] == inference_model.DEFAULT_DEVICE
    assert settings["mode"] == inference_model.DEFAULT_MODE
    assert settings["conf"] == inference_model.DEFAULT_CONFIDENCE


def test_a_read_only_project_still_runs(project, photo, monkeypatch):
    onnx(project)

    def refuse(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr("pathlib.Path.write_text", refuse)
    inference_model.write_settings(project, plan_for(project, source=str(photo)))


# --------------------------------------------------------------------------- #
# Cameras as the worker reports them
# --------------------------------------------------------------------------- #


def test_cameras_are_read_out_of_the_scan_result():
    cameras = inference_model.cameras_from(
        {"cameras": [{"index": 0, "name": "Integrated Webcam", "width": 1280, "height": 720}]}
    )

    assert cameras[0].index == 0
    assert "1280x720" in cameras[0].caption()


def test_a_malformed_camera_entry_is_skipped():
    payload = {"cameras": [{"index": "left"}, "nonsense", {"index": 1}]}

    assert [c.index for c in inference_model.cameras_from(payload)] == [1]


def test_a_camera_with_no_name_is_still_choosable():
    assert inference_model.cameras_from({"cameras": [{"index": 2}]})[0].caption() == "Camera 2"
