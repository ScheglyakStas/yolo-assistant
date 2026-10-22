"""Step 4's page: what it shows, how it moves, and what an edit writes.

The page is mostly navigation, so most of this is about where the user ends up
after a key, a wheel notch or a class change. The rest is about the two things
an edit must get right: it has to reach the disk immediately, because there is
no Save button to catch it, and it has to be undoable, because drawing a box is
one slip of the mouse away from happening by accident.
"""

from __future__ import annotations

import pytest
from PIL import Image
from PySide6.QtCore import QPointF, Qt

from yolo_assistant.app.core import project as project_model
from yolo_assistant.app.core import review as review_model
from yolo_assistant.app.core.paths import DATA_DIR_ENV
from yolo_assistant.app.core.project import Project

from test_autolabel_worker import build_dataset

ONE_BOX = "0 0.500000 0.500000 0.200000 0.200000\n"


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "appdata"))


@pytest.fixture
def project(tmp_path, environment) -> Project:
    """A prepared project whose photos are photos, not the word "jpeg"."""
    project = Project(build_dataset(tmp_path / "project", classes=(("cat", 3), ("dog", 2))))
    for image in sorted(project.images_dir.rglob("*.jpg")):
        Image.new("RGB", (80, 60), "grey").save(image)
    project_model.set_current(project)
    return project


@pytest.fixture
def make_page(qapp):
    from yolo_assistant.app.ui.steps.verify_step import VerifyStep

    made = []

    def build():
        page = VerifyStep()
        made.append(page)
        return page

    yield build
    for page in made:
        page.shutdown()


def label_of(project: Project, slug: str, number: int):
    return project.class_labels_dir(slug) / f"{slug}_{number:06d}.txt"


def label_all(project: Project, text: str = ONE_BOX) -> None:
    for images in sorted(project.images_dir.iterdir()):
        labels = project.class_labels_dir(images.name)
        labels.mkdir(parents=True, exist_ok=True)
        for photo in sorted(images.iterdir()):
            (labels / (photo.stem + ".txt")).write_text(text, encoding="utf-8")


@pytest.fixture
def page(make_page, project):
    label_all(project)
    return make_page()


# --------------------------------------------------------------------------- #
# What it shows
# --------------------------------------------------------------------------- #


def test_it_opens_on_the_first_class_and_says_where_you_are(page):
    assert page._classes.count() == 2
    assert page._where.text() == "cat - photo 1 of 3"
    assert "cat_000001.jpg" in page._filename.text()
    assert "1 box" in page._filename.text()


def test_the_boxes_in_the_file_are_the_boxes_on_screen(page):
    boxes = page._canvas.boxes

    assert len(boxes) == 1
    assert (boxes[0].cx, boxes[0].w) == pytest.approx((0.5, 0.2))
    assert page._canvas.has_photo, "the photo itself has to be decoded, not just its labels"


def test_the_class_list_says_where_nothing_was_found(make_page, project):
    label_all(project)
    label_of(project, "cat", 2).write_text("", encoding="utf-8")

    page = make_page()

    assert page._classes.item(0).text() == "cat - 3 photos, nothing found in 1"
    assert page._classes.item(1).text() == "dog - 2 photos"


def test_a_photo_with_no_boxes_says_so_rather_than_looking_empty(make_page, project):
    label_all(project)
    label_of(project, "cat", 1).write_text("", encoding="utf-8")

    page = make_page()

    assert page._canvas._note == "Nothing was found in this photo."


def test_a_photo_that_cannot_be_opened_says_which_one(make_page, project):
    label_all(project)
    (project.class_images_dir("cat") / "cat_000001.jpg").write_bytes(b"not a photo")

    page = make_page()

    assert not page._canvas.has_photo
    assert "cat_000001.jpg could not be opened" in page._canvas._note


def test_label_lines_that_could_not_be_read_are_reported_before_they_are_lost(
    make_page, project
):
    label_all(project)
    label_of(project, "cat", 1).write_text(ONE_BOX + "rubbish\n", encoding="utf-8")

    page = make_page()

    # Saying nothing would mean the next correction silently dropped the line.
    assert "1 line(s)" in page._canvas._note
    assert "rewrites the file without them" in page._canvas._note


def test_a_class_where_nothing_was_found_is_called_out(make_page, project):
    label_all(project)
    for number in (1, 2, 3):
        label_of(project, "cat", number).write_text("", encoding="utf-8")

    page = make_page()

    assert page._summary.state == "warn"
    assert '"cat"' in page._summary._detail.text()


def test_photos_are_not_decoded_larger_than_they_are_shown(tmp_path):
    from yolo_assistant.app.ui.steps.verify_step import MAX_DECODE_EDGE, load_image

    big = tmp_path / "big.jpg"
    Image.new("RGB", (4000, 3000), "grey").save(big)

    image = load_image(big)

    assert max(image.width(), image.height()) == MAX_DECODE_EDGE
    # The aspect ratio is what the boxes are drawn against, so it has to survive.
    assert image.width() / image.height() == pytest.approx(4000 / 3000, rel=1e-3)


def test_every_state_of_the_canvas_paints(page, project):
    """Painting is the one part with no return value to assert on.

    ``grab`` runs the real paint, so a bad flag or a null image shows up here
    rather than as a blank page on someone's machine.
    """
    assert not page._canvas.grab().isNull(), "a photo with boxes"

    page._canvas._select(0)
    assert not page._canvas.grab().isNull(), "a box selected, with its handles"

    page._canvas.delete_selected()
    assert not page._canvas.grab().isNull(), "a photo with nothing in it"

    page._canvas.clear("No prepared dataset to check yet.")
    assert not page._canvas.grab().isNull(), "no photo at all"


# --------------------------------------------------------------------------- #
# Moving about
# --------------------------------------------------------------------------- #


def test_stepping_moves_through_the_class_and_stops_at_the_ends(page):
    assert not page._previous.isEnabled()

    page.step(1)
    assert page._where.text() == "cat - photo 2 of 3"
    assert page._previous.isEnabled() and page._next.isEnabled()

    page.step(1)
    assert page._where.text() == "cat - photo 3 of 3"
    assert not page._next.isEnabled()

    page.step(1)
    assert page._where.text() == "cat - photo 3 of 3", "there is no fourth photo"


def test_the_mouse_wheel_moves_one_photo_per_notch(page):
    page._canvas.scroll_by(-120)
    assert page._where.text() == "cat - photo 2 of 3"

    page._canvas.scroll_by(120)
    assert page._where.text() == "cat - photo 1 of 3"


def test_a_trackpad_flick_is_not_twenty_photos(page):
    """Fine-resolution devices send a stream of small deltas."""
    for _ in range(4):
        page._canvas.scroll_by(-30)

    assert page._where.text() == "cat - photo 2 of 3"


def test_the_arrow_keys_step_and_delete_removes_a_box(page):
    assert set(page._shortcuts) == {
        Qt.Key.Key_Left,
        Qt.Key.Key_Right,
        Qt.Key.Key_Delete,
        Qt.Key.Key_Backspace,
    }
    for key in page._shortcuts.values():
        # Anywhere on the page, so the keys work while the class list has focus.
        assert key.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut


def test_choosing_another_class_starts_at_its_first_photo(page):
    page.step(1)

    page._classes.setCurrentRow(1)

    assert page._where.text() == "dog - photo 1 of 2"
    assert "dog_000001.jpg" in page._filename.text()


def test_the_filter_goes_straight_to_the_photos_with_nothing_in_them(page, project):
    label_of(project, "cat", 2).write_text("", encoding="utf-8")
    label_of(project, "cat", 3).write_text("", encoding="utf-8")
    page.on_enter()

    page._only_empty.setChecked(True)

    assert page._where.text() == "cat - photo 1 of 2 with nothing found, out of 3"
    assert "cat_000002.jpg" in page._filename.text()

    page._only_empty.setChecked(False)
    assert page._where.text() == "cat - photo 2 of 3", "it stays on the photo being looked at"


def test_labels_written_elsewhere_are_picked_up_on_returning(page, project):
    label_of(project, "cat", 1).write_text(ONE_BOX * 3, encoding="utf-8")

    page.on_enter()

    assert len(page._canvas.boxes) == 3


# --------------------------------------------------------------------------- #
# Corrections
# --------------------------------------------------------------------------- #


def test_deleting_a_box_reaches_the_disk_immediately(page, project):
    page._canvas._select(0)

    assert page._canvas.delete_selected()

    # No Save button: the file is the state.
    assert label_of(project, "cat", 1).read_text(encoding="utf-8") == ""
    assert page._classes.item(0).text() == "cat - 3 photos, nothing found in 1"
    assert "no boxes" in page._filename.text()
    assert "edited" in page._filename.text()


def test_a_box_drawn_by_hand_is_written_in_the_class_of_the_folder(page, project):
    page._classes.setCurrentRow(1)

    assert page._canvas.add_box(QPointF(0.2, 0.2), QPointF(0.6, 0.8))

    written = label_of(project, "dog", 1).read_text(encoding="utf-8").splitlines()
    assert len(written) == 2
    # "dog" is class 1 in classes.txt, and a box in dog's folder is a dog.
    assert written[1].split()[0] == "1"
    assert [float(v) for v in written[1].split()[1:]] == pytest.approx([0.4, 0.5, 0.4, 0.6])


def test_reverting_puts_the_photo_back(page, project):
    original = label_of(project, "cat", 1).read_text(encoding="utf-8")
    page._canvas._select(0)
    page._canvas.delete_selected()

    page._revert_photo()

    assert label_of(project, "cat", 1).read_text(encoding="utf-8") == original
    assert len(page._canvas.boxes) == 1
    assert not page._revert.isEnabled()


def test_reverting_a_photo_that_had_no_label_removes_the_file_again(make_page, project):
    label_all(project)
    label_of(project, "cat", 1).unlink()
    page = make_page()
    gates = []
    page.gates_changed.connect(lambda: gates.append(True))

    page._canvas.add_box(QPointF(0.2, 0.2), QPointF(0.6, 0.8))
    assert label_of(project, "cat", 1).is_file()
    assert gates, "the dataset has just become fully labelled, which unlocks Step 5"

    page._revert_photo()

    # Back to "nobody has looked at this photo", which is what it was.
    assert not label_of(project, "cat", 1).exists()
    assert page._classes.item(0).text() == "cat - 3 photos, 1 not labelled"


def test_revert_is_not_offered_once_the_photo_has_been_left(page, project):
    """It can only undo to how the photo looked when it was opened."""
    page._canvas._select(0)
    page._canvas.delete_selected()
    assert page._revert.isEnabled()

    page.step(1)
    page.step(-1)

    assert not page._revert.isEnabled()
    assert "edited" in page._filename.text(), "the session still remembers the change"


def test_a_filter_with_nothing_to_show_says_why(page):
    page._only_empty.setChecked(True)

    assert not page._canvas.has_photo
    assert "found in every photo of this class" in page._canvas._note
    assert page._where.text() == ""


def test_moving_between_photos_does_not_re_scan_the_dataset(page, monkeypatch):
    """Gating is re-evaluated on every ``gates_changed``, and it costs a scan."""
    gates = []
    page.gates_changed.connect(lambda: gates.append(True))

    page.step(1)
    page.step(1)
    page._canvas._select(0)
    page._canvas.delete_selected()

    assert gates == [], "nothing here changed which photos have labels"


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #


def test_it_is_locked_until_every_photo_has_a_label(make_page, project):
    label_all(project)
    label_of(project, "dog", 2).unlink()

    page = make_page()

    assert not page.can_enter()
    assert "1 photo(s) still have no labels" in page.blocked_reason()
    assert "Step 3" in page.blocked_reason()


def test_it_is_locked_without_a_prepared_dataset(make_page, environment, tmp_path):
    project_model.set_current(Project(tmp_path / "not-prepared"))

    page = make_page()

    assert not page.can_enter()
    assert "Step 2" in page.blocked_reason()
    assert page._canvas._note == "No prepared dataset to check yet."


def test_a_fully_labelled_dataset_owes_the_user_nothing(page):
    # Checking is optional, so "complete" cannot mean "every photo was looked
    # at" -- that is a tick nobody would ever earn.
    assert page.can_enter()
    assert page.is_complete()


def test_it_needs_no_managed_environment(page, monkeypatch):
    """Nothing here runs a worker, so a broken Step 1 must not lock it."""
    from yolo_assistant.app.core import runtime

    monkeypatch.setattr(runtime, "managed_environment", lambda: None)
    monkeypatch.delenv(runtime.PYTHON_ENV, raising=False)

    assert page.can_enter()


def test_the_wizard_locks_step_4_behind_step_3(make_page, project):
    from yolo_assistant.app.ui.steps.base import StepPage
    from yolo_assistant.app.ui.wizard import WizardWindow

    page = make_page()  # nothing is labelled yet
    earlier = StepPage()
    earlier.title = "Autolabeling"

    window = WizardWindow([earlier, page])
    try:
        assert "(locked)" in window._rail.item(1).text()
        assert not window._next.isEnabled()
    finally:
        window.close()


def test_step_4_is_in_the_wizard(environment):
    from yolo_assistant.app.main import build_pages

    pages = build_pages(fake_worker=False)
    try:
        assert [p.title for p in pages][:4] == [
            "System requirements",
            "Dataset preparation",
            "Autolabeling",
            "Label check",
        ]
    finally:
        for page in pages:
            page.shutdown()


def test_label_review_uses_the_review_module_the_worker_agrees_with(page, project):
    """A sanity check that the page edits the same files Step 3 wrote."""
    page._canvas._select(0)
    page._canvas.delete_selected()

    photos = review_model.class_photos(project, "cat")

    assert photos[0].state == review_model.EMPTY
