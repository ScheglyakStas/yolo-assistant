"""What Step 3 would do, before it does it.

These are about the arithmetic the page shows and the worker is handed: how many
photos still need a label, which classes are missing a description, and what a
previous run left behind to restore.
"""

from __future__ import annotations

import json

import pytest

from yolo_assistant.app.core import autolabel
from yolo_assistant.app.core.project import ClassStat, DatasetStatus, Project


def status_for(*classes) -> DatasetStatus:
    """``classes`` as (name, images, labels) triples."""
    return DatasetStatus(
        state="prepared",
        message="",
        classes=tuple(
            ClassStat(index=index, name=name, slug=name, images=images, labels=labels)
            for index, (name, images, labels) in enumerate(classes)
        ),
    )


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


def test_only_unlabelled_photos_count_as_work():
    plan = autolabel.build_plan(status_for(("cat", 10, 4), ("dog", 6, 6)), {"cat": "a cat"})

    assert plan.total_images == 16
    assert plan.todo == 6
    assert [c.name for c in plan.working_classes] == ["cat"]


def test_relabelling_counts_every_photo():
    plan = autolabel.build_plan(
        status_for(("cat", 10, 10)), {"cat": "a cat"}, relabel=True
    )

    assert plan.todo == 10
    assert plan.is_runnable


def test_a_fully_labelled_dataset_is_complete():
    plan = autolabel.build_plan(status_for(("cat", 3, 3), ("dog", 2, 2)))

    # An empty label file means "nothing in this photo", which is a real answer,
    # so this counts files and never their contents.
    assert plan.is_complete
    assert "straight to Step 4" in plan.summary()


def test_a_class_with_work_and_no_prompt_blocks_the_run():
    plan = autolabel.build_plan(status_for(("cat", 4, 0), ("dog", 4, 4)), {"cat": "   "})

    assert not plan.is_runnable
    assert '"cat"' in plan.problems()[0]


def test_a_class_with_nothing_to_do_needs_no_prompt():
    plan = autolabel.build_plan(status_for(("cat", 4, 0), ("dog", 4, 4)), {"cat": "a cat"})

    assert plan.is_runnable, "dog is finished; its empty prompt is irrelevant"


def test_nothing_to_do_is_reported_rather_than_run():
    plan = autolabel.build_plan(status_for(("cat", 4, 4)), {"cat": "a cat"})

    assert not plan.is_runnable
    assert "already has a label" in plan.problems()[0]


def test_an_empty_dataset_is_not_complete():
    plan = autolabel.build_plan(DatasetStatus("absent", ""))

    assert not plan.is_complete
    assert not plan.is_runnable
    assert "Step 2" in plan.problems()[0]


def test_the_worker_is_given_every_class_and_a_trimmed_prompt():
    plan = autolabel.build_plan(
        status_for(("cat", 4, 0), ("dog", 4, 4)), {"cat": "  a cat  ", "dog": "a dog"}
    )

    # Finished classes are sent too: the worker decides what to skip by looking
    # at the filesystem, and two places deciding that is how they disagree.
    assert plan.job_classes() == [
        {"index": 0, "name": "cat", "slug": "cat", "prompt": "a cat"},
        {"index": 1, "name": "dog", "slug": "dog", "prompt": "a dog"},
    ]


# --------------------------------------------------------------------------- #
# Remembering what was asked for
# --------------------------------------------------------------------------- #


def test_prompts_come_back_from_a_previous_run(tmp_path):
    project = Project(tmp_path / "p")
    project.dataset_dir.mkdir(parents=True)
    autolabel.record_path(project).write_text(
        json.dumps(
            {
                "model": autolabel.MODELS[1].key,
                "threshold": 0.35,
                "max_detections": 3,
                "classes": [{"slug": "cat", "prompt": "a tabby cat"}],
            }
        ),
        encoding="utf-8",
    )

    record = autolabel.read_record(project)

    assert autolabel.prompts_from(record) == {"cat": "a tabby cat"}
    assert autolabel.settings_from(record) == {
        "threshold": 0.35,
        "max_detections": 3,
        "model": autolabel.MODELS[1].key,
    }


def test_a_damaged_record_costs_a_retyped_prompt_and_nothing_else(tmp_path):
    project = Project(tmp_path / "p")
    project.dataset_dir.mkdir(parents=True)
    autolabel.record_path(project).write_text("{ not json", encoding="utf-8")

    assert autolabel.read_record(project) is None
    assert autolabel.prompts_from(None) == {}
    assert autolabel.settings_from(None)["model"] == autolabel.DEFAULT_MODEL


def test_nonsense_settings_fall_back_to_the_defaults():
    settings = autolabel.settings_from(
        {"threshold": "high", "max_detections": -4, "model": "some/other-model"}
    )

    assert settings == {
        "threshold": autolabel.DEFAULT_THRESHOLD,
        "max_detections": 1,
        "model": autolabel.DEFAULT_MODEL,
    }
