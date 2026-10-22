"""Step 4's model: what is in a label file, and what a class's labels amount to.

Two things are being pinned down here. The first is that reading and rewriting
a label file is lossless -- Step 4 is the only part of the app that rewrites
what the worker wrote, and a rounding difference or a dropped box would be a
correction nobody made. The second is that a class is summarised without
reading its labels, because the alternative is ten thousand file reads every
time the user clicks a class name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_assistant.app.core import project as project_model
from yolo_assistant.app.core import review
from yolo_assistant.app.core.project import Project
from yolo_assistant.app.core.review import Box

from test_autolabel_worker import build_dataset


@pytest.fixture
def project(tmp_path) -> Project:
    return Project(build_dataset(tmp_path / "project", classes=(("cat", 3), ("dog", 2))))


def label(project: Project, slug: str, number: int) -> Path:
    return project.class_labels_dir(slug) / f"{slug}_{number:06d}.txt"


# --------------------------------------------------------------------------- #
# Boxes
# --------------------------------------------------------------------------- #


def test_a_box_survives_the_trip_to_the_screen_and_back():
    box = Box(0.4, 0.6, 0.2, 0.1, index=2)
    boxes, unreadable = review.parse_boxes(review.format_boxes([box]))

    assert unreadable == 0
    assert boxes[0].index == 2
    assert (boxes[0].cx, boxes[0].cy) == pytest.approx((0.4, 0.6))
    assert (boxes[0].w, boxes[0].h) == pytest.approx((0.2, 0.1))


def test_the_file_written_here_is_the_file_the_worker_writes(tmp_path):
    """Step 4 rewrites files Step 3 created, so the two formats cannot drift."""
    from worker.commands.autolabel import _write_label

    theirs = tmp_path / "worker.txt"
    _write_label(theirs, 1, [(0.5, 0.25, 0.2, 0.125)])
    ours = tmp_path / "app.txt"
    review.write_label(ours, [Box(0.5, 0.25, 0.2, 0.125, index=1)])

    assert ours.read_bytes() == theirs.read_bytes()


def test_a_box_drawn_off_the_edge_is_clipped_to_the_photo():
    box = Box.from_corners(-0.2, 0.5, 0.4, 1.3)

    assert box is not None
    assert box.corners == pytest.approx((0.0, 0.5, 0.4, 1.0))


def test_corners_in_any_order_describe_the_same_box():
    dragged_up = Box.from_corners(0.6, 0.7, 0.2, 0.3)
    dragged_down = Box.from_corners(0.2, 0.3, 0.6, 0.7)

    assert dragged_up == dragged_down


def test_a_click_that_wobbled_is_not_a_box():
    assert Box.from_corners(0.5, 0.5, 0.5008, 0.5008) is None


def test_reading_keeps_the_small_boxes_the_worker_kept():
    """The drawing minimum guards a mouse slip. It must not filter a file.

    The worker keeps boxes down to two pixels, so on a 4000-pixel photo a real
    box can be a twentieth of the size a user could draw by hand.
    """
    boxes, unreadable = review.parse_boxes("0 0.5 0.5 0.0006 0.0006\n")

    assert unreadable == 0
    assert boxes[0].w == pytest.approx(0.0006)


def test_a_line_that_makes_no_sense_is_counted_rather_than_guessed_at():
    boxes, unreadable = review.parse_boxes(
        "0 0.5 0.5 0.2 0.2\nnonsense\n0 0.1\n\n1 0.2 0.2 0.1 0.1\n"
    )

    assert len(boxes) == 2
    assert unreadable == 2


# --------------------------------------------------------------------------- #
# Label files
# --------------------------------------------------------------------------- #


def test_the_three_states_of_a_photo_are_told_apart(project):
    label(project, "cat", 1).write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    label(project, "cat", 2).write_text("", encoding="utf-8")

    assert review.read_label(label(project, "cat", 1)).state == review.LABELLED
    assert review.read_label(label(project, "cat", 2)).state == review.EMPTY
    assert review.read_label(label(project, "cat", 3)).state == review.MISSING


def test_deleting_the_last_box_leaves_a_file_rather_than_none(project):
    path = label(project, "cat", 1)
    review.write_label(path, [Box(0.5, 0.5, 0.2, 0.2)])

    review.write_label(path, [])

    # "I looked and there is nothing here" is a different statement from
    # "nobody has looked at this photo", and the whole pipeline reads the
    # second one off the file's absence.
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == ""
    assert review.state_of(path) == review.EMPTY


def test_a_label_is_never_half_written(project, monkeypatch):
    path = label(project, "cat", 1)
    review.write_label(path, [Box(0.5, 0.5, 0.2, 0.2)])
    original = path.read_bytes()

    def explode(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(review.os, "replace", explode)
    with pytest.raises(OSError):
        review.write_label(path, [])

    assert path.read_bytes() == original


# --------------------------------------------------------------------------- #
# A class, counted without reading it
# --------------------------------------------------------------------------- #


def test_a_class_is_counted_from_the_label_sizes_alone(project, monkeypatch):
    label(project, "cat", 1).write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    label(project, "cat", 2).write_text("", encoding="utf-8")

    status = project_model.status(project)

    def refuse(*_args, **_kwargs):
        raise AssertionError("a class summary must not read its label files")

    monkeypatch.setattr(Path, "read_text", refuse)
    counted = review.count_class(project, status.classes[0])

    assert (counted.total, counted.with_boxes, counted.empty, counted.missing) == (3, 1, 1, 1)


def test_a_photos_state_comes_off_the_same_two_scans(project):
    label(project, "cat", 1).write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    label(project, "cat", 2).write_text("", encoding="utf-8")

    photos = review.class_photos(project, "cat")

    assert [p.state for p in photos] == [review.LABELLED, review.EMPTY, review.MISSING]
    assert [p.name for p in photos] == [f"cat_{n:06d}.jpg" for n in (1, 2, 3)]
    assert photos[0].label == label(project, "cat", 1)


def test_the_filter_offers_the_photos_where_nothing_was_found(project):
    label(project, "cat", 1).write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    label(project, "cat", 2).write_text("", encoding="utf-8")
    label(project, "cat", 3).write_text("", encoding="utf-8")
    photos = review.class_photos(project, "cat")

    assert review.visible(photos) == (0, 1, 2)
    assert review.visible(photos, only_empty=True) == (1, 2)


def test_a_class_where_nothing_was_found_is_reported_as_a_probable_mistake(project):
    for number in (1, 2, 3):
        label(project, "cat", number).write_text("", encoding="utf-8")
    for number in (1, 2):
        label(project, "dog", number).write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    found = review.review(project, project_model.status(project))

    assert [c.looks_wrong for c in found.classes] == [True, False]
    problem = found.problems()
    assert len(problem) == 1
    # Named, counted, and told what to do about it.
    assert '"cat"' in problem[0] and "100%" in problem[0] and "Step 3" in problem[0]


def test_the_summary_counts_the_whole_dataset(project):
    label(project, "cat", 1).write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    label(project, "cat", 2).write_text("", encoding="utf-8")

    found = review.review(project, project_model.status(project))

    assert found.total == 5
    assert "1 of 5 photos have boxes" in found.summary()
    assert "nothing was found in 1" in found.summary()
    assert "3 were never labelled" in found.summary()


def test_an_unprepared_project_reviews_as_nothing(tmp_path):
    empty = Project(tmp_path / "nowhere")

    found = review.review(empty, project_model.status(empty))

    assert found.classes == ()
    assert found.summary() == "No prepared dataset yet."
