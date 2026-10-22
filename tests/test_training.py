"""What Step 5 decides before anything is loaded.

Everything here runs in the GUI process, with no torch and no Ultralytics, and
every one of these checks exists because the alternative is finding out minutes
into a run: a misspelt setting, a size off the detection grid, a run name that
cannot be a folder, or a name that would quietly replace yesterday's results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_assistant.app.core import project as project_model
from yolo_assistant.app.core import training
from yolo_assistant.app.core.project import Project

from test_train_worker import labelled_project


@pytest.fixture
def project(tmp_path):
    return Project(labelled_project(tmp_path / "project"))


@pytest.fixture
def plan(project):
    return training.build_plan(project_model.status(project), project=project)


def add_run(project: Project, name: str, *, weights=("best.pt", "last.pt"), record=None):
    run = project.runs_dir / name
    (run / "weights").mkdir(parents=True, exist_ok=True)
    for weight in weights:
        (run / "weights" / weight).write_bytes(b"weights")
    if record is not None:
        (run / training.RUN_RECORD_FILENAME).write_text(json.dumps(record), encoding="utf-8")
    return run


# --------------------------------------------------------------------------- #
# The additional settings box
# --------------------------------------------------------------------------- #


def test_values_are_typed_by_what_they_look_like():
    parsed = training.parse_extra_args("patience=20, lr0=0.001, rect=true, cache=none")

    assert parsed.values == {"patience": 20, "lr0": 0.001, "rect": True, "cache": None}
    assert parsed.is_valid


def test_batch_one_is_a_number_and_not_a_yes():
    """The example in the specification is "batch=1, save=false". Read as a
    boolean it becomes batch=True, which Ultralytics resolves to 16."""
    parsed = training.parse_extra_args("batch=1, save=false")

    assert parsed.values["batch"] == 1
    assert parsed.values["save"] is False


def test_a_quoted_value_stays_text():
    parsed = training.parse_extra_args('optimizer="auto", split=\'val\'')

    assert parsed.values == {"optimizer": "auto", "split": "val"}


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("batch", "not a setting"),
        ("=8", "no setting name"),
        ("batch=8, batch=4", "given twice"),
        ("data=/somewhere", "set by the app"),
        ("model=yolo11x.pt", "set by the app"),
        ("resume=true", "set by the app"),
    ],
)
def test_broken_settings_are_explained(text, fragment):
    parsed = training.parse_extra_args(text)

    assert not parsed.is_valid
    assert any(fragment in problem for problem in parsed.problems)


def test_an_unrecognised_name_warns_rather_than_blocks():
    """Step 1 can move the environment to a release that added a setting, and a
    stale list must not refuse one that now exists. The worker has the final say."""
    parsed = training.parse_extra_args("some_new_setting=3")

    assert parsed.is_valid
    assert parsed.unknown == ("some_new_setting",)


def test_every_problem_is_reported_at_once():
    parsed = training.parse_extra_args("batch, model=x, patience=20")

    assert len(parsed.problems) == 2
    assert parsed.values == {"patience": 20}


# --------------------------------------------------------------------------- #
# Image size
# --------------------------------------------------------------------------- #


def test_a_size_off_the_detection_grid_is_refused_with_the_nearest_one(plan):
    off_grid = training.build_plan(plan.status, imgsz=700, runs=())

    problems = " ".join(off_grid.problems())
    assert "multiple of 32" in problems
    assert "704" in problems


def test_a_size_out_of_range_is_refused(plan):
    tiny = training.build_plan(plan.status, imgsz=32, runs=())

    assert "between 64 and 4096" in " ".join(tiny.problems())


def test_training_takes_one_size_and_says_nothing_about_squares(plan):
    sized = training.build_plan(plan.status, imgsz=1024, runs=())

    assert sized.imgsz == 1024
    assert sized.is_runnable
    assert not any("square" in note for note in sized.warnings())


def test_the_trained_size_is_read_from_the_record(project):
    add_run(project, "new", record={"imgsz": 1024})
    # Runs from before the size was one number recorded the pair as well.
    add_run(project, "old", record={"imgsz_hw": [480, 640]})

    sizes = {run.name: run.imgsz for run in training.list_runs(project)}
    assert sizes == {"new": 1024, "old": 640}


def test_a_copied_in_run_falls_back_to_ultralytics_own_args(project):
    run = add_run(project, "copied")
    (run / "args.yaml").write_text("task: detect\nimgsz: 800\nbatch: 16\n", encoding="utf-8")
    add_run(project, "bare")

    sizes = {run.name: run.imgsz for run in training.list_runs(project)}
    assert sizes == {"copied": 800, "bare": 0}


# --------------------------------------------------------------------------- #
# The output name
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name, fragment",
    [
        ("", "Give this run a name"),
        ("my/run", "cannot contain"),
        ("run.", "full stop"),
        ("aux", "Windows reserves"),
        ("con.pt", "Windows reserves"),
        ("x" * 200, "too long"),
    ],
)
def test_a_name_that_cannot_be_a_folder_is_refused(name, fragment):
    assert fragment in training.name_problem(name)


def test_an_ordinary_name_is_accepted():
    assert training.name_problem("bottles-v2") == ""


def test_the_suggested_name_is_one_that_is_free(project):
    assert training.suggest_name(project) == "train"

    add_run(project, "train")
    assert training.suggest_name(project) == "train2"

    add_run(project, "train2")
    assert training.suggest_name(project) == "train3"


# --------------------------------------------------------------------------- #
# What is already on disk
# --------------------------------------------------------------------------- #


def test_runs_are_read_off_the_filesystem_every_time(project):
    assert training.list_runs(project) == ()

    add_run(project, "bottles", record={"epochs_completed": 40, "finished_at": 1.0})
    runs = training.list_runs(project)

    assert [run.name for run in runs] == ["bottles"]
    assert runs[0].has_weights
    assert runs[0].finished
    assert runs[0].epochs == 40
    assert runs[0].weights.name == "best.pt"


def test_a_run_that_saved_nothing_is_not_a_trained_model(project):
    (project.runs_dir / "empty").mkdir(parents=True)

    assert len(training.list_runs(project)) == 1
    assert training.trained_runs(project) == ()
    assert not training.has_trained_model(project)


def test_weights_copied_in_by_hand_count_as_trained(project):
    """Nothing records that this app did the training, because nothing should:
    a model is a model however it got there."""
    add_run(project, "from-elsewhere", weights=("best.pt",))

    assert training.has_trained_model(project)


def test_only_the_best_weights_are_needed_to_export(project):
    add_run(project, "stopped", weights=("last.pt",))

    run = training.list_runs(project)[0]
    assert run.weights.name == "last.pt"
    assert "stopped part-way" in run.caption()


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


def test_a_dataset_with_missing_labels_cannot_be_trained(tmp_path):
    from test_autolabel_worker import build_dataset

    root = Path(build_dataset(tmp_path / "half"))
    (root / "dataset" / "labels" / "cat" / "cat_000001.txt").write_text("", encoding="utf-8")
    status = project_model.status(Project(root))

    problems = training.build_plan(status, runs=()).problems()
    assert any("still have no labels" in problem for problem in problems)


def test_an_unprepared_project_says_so_before_anything_else(tmp_path):
    status = project_model.status(Project(tmp_path / "nothing"))

    assert training.build_plan(status, runs=()).problems() == [
        "No prepared dataset. Finish Step 2 first."
    ]


def test_reusing_a_name_warns_that_results_would_be_replaced(project, plan):
    add_run(project, "train")

    warned = training.build_plan(plan.status, project=project, name="train")
    assert any("already holds a trained model" in note for note in warned.warnings())
    # A warning, not a refusal: replacing a run on purpose is a normal thing to
    # want, and the page asks before it happens.
    assert warned.is_runnable


def test_a_project_with_a_trained_model_can_skip_this_step(project, plan):
    assert not plan.is_complete

    add_run(project, "bottles")
    refreshed = training.build_plan(plan.status, project=project)
    assert refreshed.is_complete


def test_the_job_tells_the_worker_what_the_app_decided(project, plan):
    job = training.build_plan(
        plan.status,
        project=project,
        name="bottles",
        epochs=50,
        imgsz=640,
        qat=True,
        extra_text="batch=4",
    ).job(project)

    assert job["name"] == "bottles"
    assert job["epochs"] == 50
    assert job["imgsz"] == 640
    assert "imgsz_hw" not in job
    assert job["qat"] is True
    assert job["args"] == {"batch": 4}
    assert job["data"] == str(project.data_yaml)
    assert job["runs"] == str(project.runs_dir)


def test_the_summary_says_what_is_about_to_happen(plan):
    assert training.MODELS_BY_KEY[training.DEFAULT_MODEL].label in plan.summary()
    assert "5 photos" in plan.summary()
    assert "100 epochs" in plan.summary()


# --------------------------------------------------------------------------- #
# The model catalogue
# --------------------------------------------------------------------------- #


def test_every_model_is_a_checkpoint_ultralytics_can_fetch():
    """These strings are handed to Ultralytics, which downloads them by name
    from its assets release. A typo here is a download failure minutes in."""
    assets = {
        f"yolo{family}{size}.pt"
        for family in ("26", "11", "v8")
        for size in "nsmlx"
    }
    assert {model.key for model in training.MODELS} <= assets


def test_the_families_are_offered_newest_first():
    families = []
    for model in training.MODELS:
        family = model.key[: model.key.index(".")].rstrip("nsmlx")
        if family not in families:
            families.append(family)
    assert families == ["yolo26", "yolo11", "yolov8"]


def test_the_default_is_the_smallest_of_the_newest_family():
    """The model most likely to get someone through the whole workflow once."""
    assert training.DEFAULT_MODEL == "yolo26n.pt"
    assert training.MODELS[0].key == training.DEFAULT_MODEL


def test_every_model_says_what_it_costs_and_what_it_buys():
    for model in training.MODELS:
        assert model.params_m > 0 and model.coco_map > 0, model.key
        assert model.note.strip(), model.key
        assert "mAP50-95 on COCO" in model.detail


def test_the_selector_shows_a_name_and_the_tooltip_carries_the_rest():
    """Thirteen lines of prose is a list nobody scans, so the note moves to the
    tooltip rather than being lost."""
    for model in training.MODELS:
        assert model.caption == model.label
        assert model.note.rstrip(".").lower() in model.detail.lower()


def test_qat_warns_about_the_export_it_commits_you_to(plan):
    """A model trained this way cannot be exported as FP16 or FP32, and finding
    that out in Step 6 is finding it out after the training."""
    quantised = training.build_plan(plan.status, qat=True, runs=())

    assert quantised.is_runnable
    assert any("exported as INT8" in note for note in quantised.warnings())


# --------------------------------------------------------------------------- #
# Remembering the settings
# --------------------------------------------------------------------------- #


def test_settings_come_back_next_time(project, plan):
    chosen = training.build_plan(
        plan.status,
        project=project,
        model="yolo11s.pt",
        epochs=42,
        imgsz=320,
        qat=True,
        extra_text="batch=4",
    )
    training.write_settings(project, chosen)

    restored = training.read_settings(project)
    assert restored == {
        "model": "yolo11s.pt",
        "epochs": 42,
        "imgsz": 320,
        "qat": True,
        "args": "batch=4",
    }


def test_settings_saved_as_a_pair_come_back_as_the_longer_side(project):
    path = training.settings_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"imgsz_h": 480, "imgsz_w": 640}), encoding="utf-8")

    assert training.read_settings(project)["imgsz"] == 640


def test_a_project_that_has_never_been_trained_gets_the_defaults(project):
    assert training.read_settings(project)["model"] == training.DEFAULT_MODEL
    assert training.read_settings(None)["epochs"] == training.DEFAULT_EPOCHS


def test_a_corrupt_settings_file_costs_nothing(project):
    path = training.settings_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    assert training.read_settings(project)["epochs"] == training.DEFAULT_EPOCHS


def test_a_model_that_no_longer_exists_falls_back(project):
    path = training.settings_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": "yolo99z.pt"}), encoding="utf-8")

    assert training.read_settings(project)["model"] == training.DEFAULT_MODEL
