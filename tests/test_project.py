"""The project's state is read off the filesystem, every time.

The rule this module exists to enforce: nothing records "step 2 is done".
These tests are mostly about what happens *after* the user goes behind the
app's back -- deletes a folder, moves the photos, labels half a
class in another session -- because that is when a stored flag becomes a lie.
"""

from __future__ import annotations

import json

import pytest

from yolo_assistant.app.core import project as project_model
from yolo_assistant.app.core.paths import DATA_DIR_ENV
from yolo_assistant.app.core.project import Project


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    root = tmp_path / "appdata"
    monkeypatch.setenv(DATA_DIR_ENV, str(root))
    return root


def build_project(root, classes=("cat", "dog"), photos=3, labels=0, **record_extra):
    project = Project(root)
    for index, name in enumerate(classes):
        images = project.class_images_dir(name)
        images.mkdir(parents=True, exist_ok=True)
        project.class_labels_dir(name).mkdir(parents=True, exist_ok=True)
        for number in range(1, photos + 1):
            (images / f"{name}_{number:06d}.jpg").write_bytes(b"x")
        for number in range(1, labels + 1):
            (project.class_labels_dir(name) / f"{name}_{number:06d}.txt").write_text("")
    record = {
        "schema": project_model.PROJECT_SCHEMA,
        "source": str(root.parent / "photos"),
        "prepared_at": 1.0,
        "classes": [{"name": name, "slug": name} for name in classes],
        **record_extra,
    }
    project.prepared_file.write_text(json.dumps(record), encoding="utf-8")
    return project


def test_a_prepared_project_reports_its_classes_and_counts(tmp_path):
    project = build_project(tmp_path / "p")

    status = project_model.status(project)

    assert status.is_prepared
    assert status.names == ["cat", "dog"]
    assert status.total_images == 6
    assert not status.fully_labelled


def test_counts_follow_the_filesystem_not_the_record(tmp_path):
    project = build_project(tmp_path / "p", photos=3)
    (project.class_images_dir("cat") / "cat_000003.jpg").unlink()

    # Nothing was rewritten; the number simply has to be right next time it is
    # asked for.
    assert project_model.status(project).total_images == 5


def test_labels_are_counted_so_step_3_can_be_skipped(tmp_path):
    project = build_project(tmp_path / "p", photos=2, labels=2)

    status = project_model.status(project)

    assert status.fully_labelled
    assert status.total_labels == 4


def test_an_empty_label_file_counts_as_labelled(tmp_path):
    project = build_project(tmp_path / "p", classes=("cat",), photos=1, labels=1)
    assert (project.class_labels_dir("cat") / "cat_000001.txt").read_text() == ""

    # An empty label means "there is nothing in this photo", which is a real
    # answer. Counting contents instead of files would send the user back to
    # label it again, for ever.
    assert project_model.status(project).fully_labelled


def test_a_partly_labelled_class_is_reported_as_such(tmp_path):
    project = build_project(tmp_path / "p", classes=("cat",), photos=4, labels=1)

    status = project_model.status(project)

    assert not status.fully_labelled
    assert status.classes[0].missing_labels == 3
    assert "3 still to do" in status.message


def test_an_interrupted_preparation_reads_as_unfinished(tmp_path):
    project = Project(tmp_path / "p")
    project.class_images_dir("cat").mkdir(parents=True)
    (project.class_images_dir("cat") / "cat_000001.jpg").write_bytes(b"x")

    status = project_model.status(project)

    # Photos, but no proof. Half a dataset must not look like a whole one.
    assert status.state == project_model.PARTIAL
    assert not status.is_prepared


def test_an_empty_folder_has_no_dataset(tmp_path):
    assert project_model.status(Project(tmp_path / "p")).state == project_model.ABSENT


def test_a_record_from_another_version_is_stale(tmp_path):
    project = build_project(tmp_path / "p")
    record = json.loads(project.prepared_file.read_text())
    record["schema"] = project_model.PROJECT_SCHEMA + 99
    project.prepared_file.write_text(json.dumps(record))

    assert project_model.status(project).state == project_model.STALE


def test_a_record_whose_photos_vanished_is_not_prepared(tmp_path):
    import shutil

    project = build_project(tmp_path / "p")
    shutil.rmtree(project.images_dir)

    status = project_model.status(project)

    assert status.state == project_model.PARTIAL
    assert "photos are gone" in status.message


def test_class_order_comes_from_classes_txt(tmp_path):
    project = Project(tmp_path / "p")
    project.dataset_dir.mkdir(parents=True)
    project.classes_file.write_text("dog\ncat\n", encoding="utf-8")

    # Alphabetical would give ["cat", "dog"] and silently swap every label in
    # the dataset. The recorded order is the only correct answer.
    assert project_model.read_classes(project) == ["dog", "cat"]


def test_the_label_path_matches_the_rule_ultralytics_uses(tmp_path):
    project = build_project(tmp_path / "p", classes=("cat",), photos=1)
    image = project.class_images_dir("cat") / "cat_000001.jpg"

    assert project.label_for(image) == project.class_labels_dir("cat") / "cat_000001.txt"


def test_class_images_are_listed_in_viewing_order(tmp_path):
    project = build_project(tmp_path / "p", classes=("cat",), photos=12)

    names = [p.name for p in project_model.class_images(project, "cat")]

    assert names[0] == "cat_000001.jpg"
    assert names[-1] == "cat_000012.jpg", "zero-padding keeps the order sane"


# --------------------------------------------------------------------------- #
# The current-project pointer
# --------------------------------------------------------------------------- #


def test_the_current_project_survives_a_restart(tmp_path, data_dir):
    project = build_project(tmp_path / "p")

    project_model.set_current(project)

    assert project_model.current().root == project.root


def test_a_pointer_to_a_deleted_folder_is_simply_no_project(tmp_path, data_dir):
    import shutil

    project = build_project(tmp_path / "p")
    project_model.set_current(project)
    shutil.rmtree(project.root)

    assert project_model.current() is None


def test_a_corrupt_pointer_file_is_not_fatal(tmp_path, data_dir):
    from yolo_assistant.app.core.paths import state_file

    state_file().parent.mkdir(parents=True, exist_ok=True)
    state_file().write_text("{ not json", encoding="utf-8")

    assert project_model.current() is None


def test_the_suggested_project_folder_sits_beside_the_photos(tmp_path):
    photos = tmp_path / "holiday snaps"
    photos.mkdir()

    suggested = project_model.default_root_for(photos)

    assert suggested == tmp_path / "holiday snaps-project"


def test_an_unwritable_photo_folder_falls_back_to_the_app_directory(tmp_path, data_dir, monkeypatch):
    photos = tmp_path / "readonly" / "photos"
    photos.mkdir(parents=True)
    monkeypatch.setattr(project_model, "_writable", lambda path: False)

    suggested = project_model.default_root_for(photos)

    assert suggested == data_dir / "projects" / "photos-project"
