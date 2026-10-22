"""Step 4: look at what OWLv2 actually drew, and fix what it got wrong.

This is the only optional step, and the only one whose value depends entirely
on how fast it is to use. Nobody checks ten thousand photos with a mouse; they
check forty, decide whether the labels are trustworthy, and either carry on or
go back to Step 3 with a better sentence. So the page is built around moving,
not around editing: arrow keys, the mouse wheel and two buttons all step
through the photos of a class, and the filter jumps straight to the photos
where nothing was found -- which is where a bad prompt shows itself.

Three things are worth knowing about what it does.

**Nothing found is the finding.** An empty label file looks like every other
label file from Step 3's counts: written, valid, and a photo the trainer will
be told contains no objects. Those photos are counted per class, listed first
in the filter, and called out in the class list, because a class where OWLv2
found nothing in most photos is a prompt that missed rather than a class that
is hard.

**Corrections are saved as they are made.** Drawing a box or deleting one
rewrites that photo's label file immediately. A verification pass is dozens of
tiny changes across dozens of photos, and a Save button turns that into a
question the user has to answer correctly every time they press an arrow key.
Revert restores the photo to how it was when it was opened.

**Photos are decoded no larger than they will be shown.** A 12-megapixel phone
photo takes most of a tenth of a second to decode at full size, which is the
difference between holding the arrow key down and pressing it forty times. The
boxes are stored as fractions of the photo, so decoding at a quarter size
changes nothing about where they land.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QImageReader, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core import project as project_model
from ...core import review as review_model
from ...core.i18n import N_, tr, trn
from ...core.project import Project
from ...core.review import ClassReview, Photo
from ..common.image_canvas import ImageCanvas
from ..common.param_form import apply_help
from ..common.status_row import OK, UNKNOWN, WARN, StatusRow
from .base import StepPage

#: Photos are decoded to fit inside this square, in pixels. Larger than any
#: window this page will be shown in, so the photo is never upscaled from the
#: decoded copy; small enough that a phone photo decodes in a fraction of the
#: time. JPEG decoding at a reduced scale is a libjpeg feature, not a resize
#: after the fact, so the saving is real.
MAX_DECODE_EDGE = 2048


def load_image(path: Path) -> QImage | None:
    """Decode a photo for display, no larger than it will be shown.

    EXIF rotation is deliberately *not* applied: Step 2 already rotated the
    pixels and cleared the tag, and the worker read the file the same way. A
    viewer that honoured a tag the labeller ignored would draw every box on a
    photo turned ninety degrees from the one the boxes were measured against.
    """
    reader = QImageReader(str(path))
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > MAX_DECODE_EDGE:
        reader.setScaledSize(
            size.scaled(
                QSize(MAX_DECODE_EDGE, MAX_DECODE_EDGE),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        )
    image = reader.read()
    return None if image.isNull() else image


class VerifyStep(StepPage):
    title = N_("Label check")
    subtitle = N_(
        "Look at the boxes OWLv2 drew, and correct the ones it got wrong. This step is "
        "optional -- but a few minutes here is the cheapest way to find out that a "
        "description did not match before training on it."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._project: Project | None = None
        self._status = project_model.status(None)
        self._review = review_model.Review()
        self._photos: list[Photo] = []
        self._visible: tuple[int, ...] = ()
        self._position = -1
        self._original: tuple = ()
        self._original_state = review_model.MISSING
        self._touched = False
        """Whether the photo on screen has been changed since it was opened.

        Kept apart from ``_edited``, which remembers the whole session: Revert
        can only undo back to how this photo looked when it was opened, so
        offering it for a photo corrected an hour ago would be a button that
        does nothing.
        """
        self._edited: set[Path] = set()

        self._build_ui()
        self._wire()
        self._adopt_project()
        self.refresh()

    # -- construction ------------------------------------------------------ #

    def _build_ui(self) -> None:
        heading = QLabel(f"<h2>{tr(self.title)}</h2>")
        subtitle = QLabel(tr(self.subtitle))
        subtitle.setWordWrap(True)

        classes_box = QGroupBox(tr("Classes"))
        self._classes = QListWidget()
        self._classes.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # The caption carries the counts, and a class name can be long. Wrapping
        # it is the difference between reading "nothing found in 812" and
        # scrolling sideways to find out.
        self._classes.setWordWrap(True)
        self._classes.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._classes.setResizeMode(QListWidget.ResizeMode.Adjust)
        apply_help(self._classes, "verify.classes")
        classes_layout = QVBoxLayout(classes_box)
        classes_layout.addWidget(self._classes)

        self._where = QLabel("")
        where_font = self._where.font()
        where_font.setBold(True)
        self._where.setFont(where_font)
        self._filename = QLabel("")
        self._filename.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self._canvas = ImageCanvas()

        self._previous = QPushButton(tr("< Previous"))
        self._next = QPushButton(tr("Next >"))
        for button in (self._previous, self._next):
            apply_help(button, "verify.navigation")

        self._only_empty = QCheckBox(tr("Only photos where nothing was found"))
        apply_help(self._only_empty, "verify.only_empty")

        self._delete = QPushButton(tr("Delete box"))
        apply_help(self._delete, "verify.delete_box")
        self._delete.setEnabled(False)

        self._revert = QPushButton(tr("Revert photo"))
        apply_help(self._revert, "verify.revert")
        self._revert.setEnabled(False)

        controls = QHBoxLayout()
        controls.addWidget(self._previous)
        controls.addWidget(self._next)
        controls.addSpacing(16)
        controls.addWidget(self._only_empty)
        controls.addStretch(1)
        controls.addWidget(self._delete)
        controls.addWidget(self._revert)

        hint = QLabel(
            tr(
                "Arrow keys or the mouse wheel move between photos. Drag on the photo to "
                "add a box the model missed; click a box and press Delete to remove it. "
                "Changes are saved as you make them."
            )
        )
        hint.setWordWrap(True)

        self._summary = StatusRow(tr("Labels"))

        viewer = QWidget()
        viewer_layout = QVBoxLayout(viewer)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.addWidget(self._where)
        viewer_layout.addWidget(self._filename)
        viewer_layout.addWidget(self._canvas, 1)
        viewer_layout.addLayout(controls)
        viewer_layout.addWidget(hint)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(classes_box)
        splitter.addWidget(viewer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 900])

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(subtitle)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._summary)

    def _wire(self) -> None:
        self._classes.currentRowChanged.connect(self._select_class)
        self._previous.clicked.connect(lambda: self.step(-1))
        self._next.clicked.connect(lambda: self.step(1))
        self._only_empty.toggled.connect(self._apply_filter)
        self._delete.clicked.connect(self._canvas.delete_selected)
        self._revert.clicked.connect(self._revert_photo)

        self._canvas.stepped.connect(self.step)
        self._canvas.boxes_changed.connect(self._on_boxes_changed)
        self._canvas.selection_changed.connect(
            lambda index: self._delete.setEnabled(index >= 0)
        )

        # Page-level rather than on the canvas, so the keys work while the
        # class list has the focus -- which it does immediately after the user
        # has chosen a class, exactly when they want to start paging.
        self._shortcuts = {}
        for key, handler in (
            (Qt.Key.Key_Left, lambda: self.step(-1)),
            (Qt.Key.Key_Right, lambda: self.step(1)),
            (Qt.Key.Key_Delete, self._canvas.delete_selected),
            (Qt.Key.Key_Backspace, self._canvas.delete_selected),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(handler)
            self._shortcuts[key] = shortcut

    # -- state ------------------------------------------------------------- #

    def _adopt_project(self) -> None:
        """Re-read the project from disk, keeping the user where they were.

        Step 3 may have labelled another few thousand photos since this page
        was last open, and the user may have deleted a class from under it. The
        class the user was looking at is kept by name rather than by row, so a
        class disappearing moves them to the first one rather than to whichever
        class inherited its position.
        """
        previous_slug = self._current_class().slug if self._current_class() else None
        self._project = project_model.current()
        self._status = project_model.status(self._project)
        self._review = review_model.review(self._project, self._status)

        self._classes.blockSignals(True)
        self._classes.clear()
        for entry in self._review.classes:
            self._classes.addItem(entry.caption())
        self._classes.blockSignals(False)

        row = 0
        if previous_slug is not None:
            for index, entry in enumerate(self._review.classes):
                if entry.slug == previous_slug:
                    row = index
                    break
        if self._review.classes:
            # Selected with the signal blocked and loaded by hand: the signal
            # does not fire when the row has not changed, and coming back to
            # this page has to re-read the class either way.
            self._classes.blockSignals(True)
            self._classes.setCurrentRow(row)
            self._classes.blockSignals(False)
            self._select_class(row)
        else:
            self._photos = []
            self._visible = ()
            self._position = -1
            self._canvas.clear(tr("No prepared dataset to check yet."))

    def _current_class(self) -> ClassReview | None:
        row = self._classes.currentRow()
        if 0 <= row < len(self._review.classes):
            return self._review.classes[row]
        return None

    def _current_photo(self) -> Photo | None:
        if 0 <= self._position < len(self._visible):
            return self._photos[self._visible[self._position]]
        return None

    def _select_class(self, row: int) -> None:
        if self._project is None or not 0 <= row < len(self._review.classes):
            return
        entry = self._review.classes[row]
        # The photo list is the expensive part, so it is built for the class
        # being looked at and no other.
        self._photos = list(review_model.class_photos(self._project, entry.slug))
        # A new class starts at its first photo. Clearing the position first is
        # what says so: the filter below otherwise reads the old position as a
        # photo to stay on, and lands on whichever photo of this class happens
        # to sit at the same number.
        self._position = -1
        self._recount()
        self._apply_filter()

    def _apply_filter(self, _checked: bool = False) -> None:
        """Rebuild the list being paged through, staying on the same photo."""
        staying = self._current_photo()
        self._visible = review_model.visible(
            self._photos, only_empty=self._only_empty.isChecked()
        )
        position = 0
        if staying is not None:
            for index, photo_index in enumerate(self._visible):
                if self._photos[photo_index].path == staying.path:
                    position = index
                    break
        self._show(position)

    def refresh(self) -> None:
        self._paint_classes()
        self._paint_header()
        self._paint_buttons()
        self._paint_summary()

    # -- the photo on screen ----------------------------------------------- #

    def _show(self, position: int) -> None:
        if not self._visible:
            self._position = -1
            self._original = ()
            self._original_state = review_model.MISSING
            self._touched = False
            self._canvas.clear(self._nothing_to_show())
            self.refresh()
            return

        self._position = min(max(position, 0), len(self._visible) - 1)
        self._touched = False
        photo = self._photos[self._visible[self._position]]
        entry = self._current_class()
        label = review_model.read_label(photo.label)
        self._original = label.boxes
        self._original_state = label.state

        image = load_image(photo.path)
        self._canvas.set_photo(
            image,
            label.boxes,
            label=entry.name if entry else "",
            note=self._note_for(photo, label, image),
            class_index=entry.index if entry else 0,
        )
        self.refresh()

    def _note_for(self, photo: Photo, label, image) -> str:
        if image is None:
            return tr(
                "{name} could not be opened. The file may have been moved or "
                "damaged since the dataset was prepared.",
                name=photo.name,
            )
        if label.unreadable:
            return trn(
                "{n} line(s) of this photo's label file could not be "
                "read and are not shown. Changing the boxes here rewrites the file "
                "without them.",
                "{n} line(s) of this photo's label file could not be "
                "read and are not shown. Changing the boxes here rewrites the file "
                "without them.",
                label.unreadable,
            )
        if not label.exists:
            return tr("This photo has no label file yet -- autolabeling never reached it.")
        if not label.boxes:
            return tr("Nothing was found in this photo.")
        return ""

    def _nothing_to_show(self) -> str:
        if not self._review.classes:
            return tr("No prepared dataset to check yet.")
        if self._only_empty.isChecked() and self._photos:
            return tr(
                "Something was found in every photo of this class, so the filter has "
                "nothing to show. Untick it to look through them."
            )
        return tr("This class has no photos in it.")

    def step(self, delta: int) -> None:
        """Move ``delta`` photos through the class. Stops at either end."""
        if not self._visible:
            return
        target = self._position + int(delta)
        if not 0 <= target < len(self._visible):
            return
        self._show(target)

    # -- editing ------------------------------------------------------------ #

    def _on_boxes_changed(self, boxes) -> None:
        photo = self._current_photo()
        if photo is None:
            return
        review_model.write_label(photo.label, boxes)
        self._touched = True
        self._edited.add(photo.path)
        self._restate(photo, review_model.state_of(photo.label))

    def _revert_photo(self) -> None:
        photo = self._current_photo()
        if photo is None:
            return
        if self._original_state == review_model.MISSING:
            # The only way to get here is a box drawn on a photo Step 3 never
            # reached: putting the label file back means removing it.
            review_model.clear_label(photo.label)
        else:
            review_model.write_label(photo.label, self._original)
        self._edited.discard(photo.path)
        self._restate(photo, review_model.state_of(photo.label))
        self._show(self._position)

    def _restate(self, photo: Photo, state: str) -> None:
        """Bring the counts up to date after one photo's label changed."""
        was_labelled = photo.state != review_model.MISSING
        index = self._visible[self._position]
        self._photos[index] = Photo(path=photo.path, label=photo.label, state=state)
        self._recount()
        self.refresh()
        if was_labelled != (state != review_model.MISSING):
            # Whether every photo has a label is what unlocks the later steps;
            # nothing else on this page can change that answer, and saying so
            # on every arrow key would re-scan the dataset for no reason.
            self.gates_changed.emit()

    def _recount(self) -> None:
        entry = self._current_class()
        if entry is None:
            return
        self._review = self._review.with_class(
            ClassReview.from_photos(entry.index, entry.name, entry.slug, self._photos)
        )

    # -- painting ----------------------------------------------------------- #

    def _paint_classes(self) -> None:
        for row, entry in enumerate(self._review.classes):
            item = self._classes.item(row)
            if item is None:
                continue
            caption = entry.caption()
            if item.text() != caption:
                item.setText(caption)
            item.setToolTip(
                tr(
                    "Every photo of this class was looked at and nothing was found in "
                    "{share:.0%} of them.",
                    share=entry.empty_share,
                )
                if entry.looks_wrong
                else caption
            )

    def _paint_header(self) -> None:
        entry = self._current_class()
        photo = self._current_photo()
        if entry is None or photo is None:
            self._where.setText("")
            self._filename.setText("")
            return
        total = len(self._photos)
        shown = len(self._visible)
        if shown != total:
            position = tr(
                "photo {position:,} of {shown:,} with nothing found, out of {total:,}",
                position=self._position + 1,
                shown=shown,
                total=total,
            )
        else:
            position = tr(
                "photo {position:,} of {shown:,}", position=self._position + 1, shown=shown
            )
        self._where.setText(f"{entry.name} - {position}")

        boxes = len(self._canvas.boxes)
        described = tr("no boxes") if not boxes else trn("{n:,} box", "{n:,} boxes", boxes)
        edited = f" - {tr('edited')}" if photo.path in self._edited else ""
        self._filename.setText(f"{photo.name} - {described}{edited}")

    def _paint_buttons(self) -> None:
        self._previous.setEnabled(self._position > 0)
        self._next.setEnabled(0 <= self._position < len(self._visible) - 1)
        self._delete.setEnabled(self._canvas.selected >= 0)
        self._revert.setEnabled(self._touched)
        self._only_empty.setEnabled(bool(self._photos))

    def _paint_summary(self) -> None:
        if not self._review.classes:
            self._summary.set_state(UNKNOWN, tr("No prepared dataset yet."))
            return
        problems = self._review.problems()
        detail = self._review.summary()
        if self._edited:
            detail += " " + trn(
                "You have corrected {n:,} photo(s) this session.",
                "You have corrected {n:,} photo(s) this session.",
                len(self._edited),
            )
        if problems:
            self._summary.set_state(WARN, _escape(" ".join(problems)))
        elif self._review.missing:
            self._summary.set_state(WARN, _escape(detail))
        else:
            self._summary.set_state(OK, _escape(detail))

    # -- StepPage ------------------------------------------------------------ #

    def can_enter(self) -> bool:
        # Asked by the wizard whenever any page's gates change, so the cheap
        # half of the question is answered first.
        project = project_model.current()
        if project is None or not project.prepared_file.is_file():
            return False
        return project_model.status(project).fully_labelled

    def is_complete(self) -> bool:
        """Whether this step owes the user anything.

        Checking labels is optional, so "complete" cannot mean "the user looked
        at every photo" -- that would leave a tick that nobody ever earns. It
        means there is nothing here holding up Step 5.
        """
        return self.can_enter()

    def blocked_reason(self) -> str:
        status = project_model.status(project_model.current())
        if not status.is_prepared:
            return tr("Prepare your photos in Step 2 first; there are no labels to check yet.")
        missing = status.total_images - status.total_labels
        return trn(
            "{n:,} photo(s) still have no labels. Finish autolabeling in Step 3 "
            "first, or there is nothing to check for them.",
            "{n:,} photo(s) still have no labels. Finish autolabeling in Step 3 "
            "first, or there is nothing to check for them.",
            missing,
        )

    def on_enter(self) -> None:
        self._adopt_project()
        self.refresh()
        self._canvas.setFocus(Qt.FocusReason.OtherFocusReason)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
