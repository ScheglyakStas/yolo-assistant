"""What Step 6 decides before anything is loaded.

Export's mistakes are the quiet kind -- a file that exists, is the right size,
and is the wrong thing -- so almost all of this is about refusals: which
combinations of checkpoint, format and precision cannot exist, which merely
deserve a warning, and what the resulting file is called.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_assistant.app.core import export as export_model
from yolo_assistant.app.core.export import (
    FP16,
    FP32,
    INT8,
    ExportPlan,
    checkpoints,
)
from yolo_assistant.app.core.project import Project
from yolo_assistant.app.core.training import list_runs


def add_run(
    project: Project,
    name: str,
    *,
    weights=("best.pt", "last.pt"),
    qat: bool = False,
    imgsz: int = 640,
    epochs: int = 50,
    fitness: float = 0.61,
) -> None:
    folder = project.runs_dir / name
    (folder / "weights").mkdir(parents=True, exist_ok=True)
    for weight in weights:
        (folder / "weights" / weight).write_bytes(b"weights")
    (folder / "umla.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "qat": qat,
                "imgsz": imgsz,
                "epochs_completed": epochs,
                "best_fitness": fitness,
                "finished_at": 1.0,
            }
        ),
        encoding="utf-8",
    )


def add_export(project: Project, name: str, **record) -> Path:
    project.exports_dir.mkdir(parents=True, exist_ok=True)
    path = project.exports_dir / name
    path.write_bytes(b"model")
    if record:
        path.with_name(path.name + export_model.RECORD_SUFFIX).write_text(
            json.dumps(record), encoding="utf-8"
        )
    return path


@pytest.fixture
def project(tmp_path):
    project = Project(tmp_path / "project")
    project.runs_dir.mkdir(parents=True)
    project.exports_dir.mkdir(parents=True)
    return project


def plan_for(project: Project, **kwargs) -> ExportPlan:
    settings = {"imgsz_h": 640, "imgsz_w": 640}
    settings.update(kwargs)
    return export_model.build_plan(project=project, **settings)


# --------------------------------------------------------------------------- #
# What there is to export
# --------------------------------------------------------------------------- #


def test_both_checkpoints_of_a_run_are_offered(project):
    add_run(project, "bottles")

    keys = [c.key for c in checkpoints(list_runs(project))]

    assert keys == ["bottles/best", "bottles/last"]


def test_a_run_that_only_reached_last_offers_only_last(project):
    add_run(project, "stopped", weights=("last.pt",))

    offered = checkpoints(list_runs(project))

    assert [c.key for c in offered] == ["stopped/last"]
    assert offered[0].path.name == "last.pt"


def test_a_checkpoint_says_which_epoch_it_is(project):
    add_run(project, "bottles", fitness=0.6123, qat=True)

    best, last = checkpoints(list_runs(project))

    assert "best epoch" in best.caption()
    assert "0.6123" in best.caption()
    assert "last epoch" in last.caption()
    # The one fact that changes what can be done with it.
    assert "INT8 QAT" in best.caption()


def test_a_project_with_no_runs_has_nothing_to_export(project):
    assert not plan_for(project).is_runnable
    assert "no trained model" in plan_for(project).problems()[0]


# --------------------------------------------------------------------------- #
# What it refuses
# --------------------------------------------------------------------------- #


def test_a_precision_the_format_cannot_produce_is_refused(project):
    add_run(project, "bottles")

    plan = plan_for(project, fmt="onnx", precision=INT8)

    assert not plan.is_runnable
    assert "ONNX cannot be exported at INT8" in " ".join(plan.problems())
    # And it names what it can do, rather than leaving the user to guess.
    assert "FP32, FP16" in " ".join(plan.problems())


def test_a_qat_model_is_refused_at_anything_but_int8(project):
    add_run(project, "bottles", qat=True)

    for precision in (FP32, FP16):
        plan = plan_for(project, fmt="engine", precision=precision)
        assert not plan.is_runnable
        assert "Choose INT8" in " ".join(plan.problems())


def test_a_qat_model_at_int8_is_allowed(project):
    add_run(project, "bottles", qat=True)

    assert plan_for(project, fmt="engine", precision=INT8).is_runnable


def test_a_qat_model_is_refused_a_format_that_cannot_carry_it(project):
    add_run(project, "bottles", qat=True)

    plan = plan_for(project, fmt="tflite", precision=INT8)

    assert not plan.is_runnable
    assert "cannot carry an INT8 QAT model" in " ".join(plan.problems())
    assert "ONNX or TensorRT" in " ".join(plan.problems())


def test_a_size_off_the_detection_grid_is_refused(project):
    add_run(project, "bottles")

    plan = plan_for(project, imgsz_h=700, imgsz_w=640)

    assert not plan.is_runnable
    assert "multiple of 32" in " ".join(plan.problems())
    assert "nearest that is, is 704" in " ".join(plan.problems())


def test_a_setting_the_app_manages_is_refused_outright(project):
    add_run(project, "bottles")

    plan = plan_for(project, extra_text="opset=11")

    assert not plan.is_runnable
    assert "set by the app" in " ".join(plan.problems())


def test_the_deepstream_toggle_and_a_typed_nms_is_a_contradiction(project):
    add_run(project, "bottles")

    plan = plan_for(project, deepstream=True, extra_text="nms=true")

    assert not plan.is_runnable
    assert "already set by the DeepStream toggle" in " ".join(plan.problems())


def test_a_misspelt_setting_is_a_warning_not_a_refusal(project):
    """The app's list of export settings ages; the worker has the final say."""
    add_run(project, "bottles")

    plan = plan_for(project, extra_text="dynammic=true")

    assert plan.is_runnable
    assert "dynammic" in " ".join(plan.warnings())


# --------------------------------------------------------------------------- #
# What it warns about
# --------------------------------------------------------------------------- #


def test_a_tensorrt_engine_says_it_will_not_travel(project):
    add_run(project, "bottles")

    plan = plan_for(project, fmt="engine", precision=FP16)

    assert plan.is_runnable
    assert "will not work" in " ".join(plan.warnings())


def test_a_shape_other_than_the_trained_square_says_nothing_about_accuracy(project):
    """A square model for 16:9 input is the slow choice, not the accurate one,
    and a warning nudging beginners towards it did more harm than good."""
    add_run(project, "bottles", imgsz=1024)

    plan = plan_for(project, imgsz_h=576, imgsz_w=1024)

    assert plan.is_runnable
    assert not any("accuracy" in note for note in plan.warnings())
    assert not any("was trained at" in note for note in plan.warnings())


@pytest.mark.parametrize("height, width", [(1088, 1024), (576, 1056)])
def test_neither_side_may_exceed_the_trained_size(project, height, width):
    add_run(project, "bottles", imgsz=1024)

    plan = plan_for(project, imgsz_h=height, imgsz_w=width)

    assert not plan.is_runnable
    assert "can be at most 1024" in " ".join(plan.problems())


def test_a_run_with_no_recorded_size_is_not_capped(project):
    add_run(project, "copied-in", imgsz=0)

    plan = plan_for(project, imgsz_h=1088, imgsz_w=1920)

    assert plan.is_runnable


@pytest.mark.parametrize(
    "trained, expected",
    [(1024, (576, 1024)), (640, (352, 640)), (1280, (704, 1280)), (0, (352, 640))],
)
def test_the_default_size_is_16_9_at_the_trained_size(project, trained, expected):
    add_run(project, "bottles", imgsz=trained)
    run = list_runs(project)[0]

    assert export_model.default_imgsz(run) == expected
    assert export_model.max_imgsz(run) == (trained or export_model.MAX_IMGSZ)


def test_deepstream_on_a_format_that_ignores_it_says_so(project):
    add_run(project, "bottles")

    plan = plan_for(project, fmt="torchscript", deepstream=True)

    assert plan.is_runnable
    assert "will ignore them" in " ".join(plan.warnings())


def test_replacing_an_export_is_announced(project):
    add_run(project, "bottles")
    plan = plan_for(project)
    add_export(project, plan.output_name)

    replaced = plan_for(project)

    assert replaced.is_runnable
    assert "will be replaced" in " ".join(replaced.warnings())


# --------------------------------------------------------------------------- #
# What the worker is asked for
# --------------------------------------------------------------------------- #


def test_the_output_name_records_what_it_came_from(project):
    add_run(project, "bottles")

    plan = plan_for(project, checkpoint_key="bottles/last", fmt="engine", precision=FP16)

    assert plan.output_name == "bottles_last_fp16.engine"


def test_a_directory_format_is_named_as_a_directory(project):
    add_run(project, "bottles")

    plan = plan_for(project, fmt="openvino", precision=INT8)

    assert plan.output_name == "bottles_best_int8_openvino_model"


def test_a_run_name_with_a_space_still_makes_a_sane_filename(project):
    add_run(project, "red bottles")

    assert plan_for(project).output_name == "red_bottles_best_fp32.onnx"


def test_the_deepstream_toggle_contributes_exactly_what_the_spec_says(project):
    add_run(project, "bottles")

    plan = plan_for(project, deepstream=True, extra_text="batch=1")

    assert plan.args == {"nms": False, "agnostic_nms": True, "batch": 1}


def test_the_job_carries_the_form(project):
    add_run(project, "bottles", qat=True)

    plan = plan_for(
        project,
        checkpoint_key="bottles/best",
        fmt="engine",
        precision=INT8,
        imgsz_h=480,
        imgsz_w=640,
        deepstream=True,
    )
    job = plan.job(project)

    assert job["weights"] == str(project.runs_dir / "bottles" / "weights" / "best.pt")
    assert job["format"] == "engine"
    assert job["precision"] == INT8
    # The pair, not the longer edge: export is where both numbers finally count.
    assert job["imgsz_hw"] == [480, 640]
    assert job["qat"] is True
    assert job["deepstream"] is True
    assert job["output"] == "bottles_best_int8.engine"
    assert job["args"] == {"nms": False, "agnostic_nms": True}


# --------------------------------------------------------------------------- #
# What is already on disk
# --------------------------------------------------------------------------- #


def test_an_exported_file_is_found_and_described(project):
    add_export(
        project,
        "bottles_best_fp16.onnx",
        format="onnx",
        precision="fp16",
        run="bottles",
        imgsz_hw=[480, 640],
    )

    found = export_model.list_exports(project)

    assert len(found) == 1
    assert export_model.has_export(project)
    caption = found[0].caption()
    assert "ONNX" in caption and "FP16" in caption and "640x480" in caption


def test_the_sidecar_record_is_not_itself_an_export(project):
    add_export(project, "bottles_best_fp32.onnx", format="onnx", precision="fp32")

    assert [e.name for e in export_model.list_exports(project)] == [
        "bottles_best_fp32.onnx"
    ]


def test_a_stray_file_in_the_exports_folder_is_not_a_model(project):
    (project.exports_dir / "notes.txt").write_text("todo", encoding="utf-8")

    assert not export_model.has_export(project)


def test_a_directory_export_is_recognised(project):
    (project.exports_dir / "bottles_best_int8_openvino_model").mkdir(parents=True)

    found = export_model.list_exports(project)

    assert [e.format for e in found] == ["openvino"]


def test_an_export_copied_in_by_hand_still_counts(project):
    """Nothing records that Step 6 ran; what is in the folder is the answer."""
    add_export(project, "somebody_elses.onnx")

    assert export_model.has_export(project)


# --------------------------------------------------------------------------- #
# Remembering what was asked for
# --------------------------------------------------------------------------- #


def test_settings_survive_a_round_trip(project):
    add_run(project, "bottles")
    plan = plan_for(project, fmt="engine", precision=FP16, deepstream=True, extra_text="batch=1")

    export_model.write_settings(project, plan)

    restored = export_model.read_settings(project)
    assert restored["format"] == "engine"
    assert restored["precision"] == FP16
    assert restored["deepstream"] is True
    assert restored["args"] == "batch=1"


def test_an_unreadable_settings_file_costs_nothing(project):
    export_model.settings_path(project).parent.mkdir(parents=True, exist_ok=True)
    export_model.settings_path(project).write_text("{ not json", encoding="utf-8")

    assert export_model.read_settings(project)["format"] == export_model.DEFAULT_FORMAT
