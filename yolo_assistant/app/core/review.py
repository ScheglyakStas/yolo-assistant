"""What Step 4 shows: one photo's boxes, and how to move between photos.

Step 3 can finish perfectly and still have produced a dataset that is wrong in
a way no number reports. A prompt that describes the background rather than the
object labels the background; a prompt one word too general boxes the sofa the
cat is sitting on. Both look exactly like success from the outside: every photo
has a label file, every count adds up, and the mistake only surfaces as an
accuracy figure hours later. The only way to find out is to look, so this step
exists to make looking cheap.

Two things follow for this module.

**The state of a photo comes from the label file's size, not its contents.**
A class of ten thousand photos would otherwise mean ten thousand file reads
every time the user clicks a class. An empty file means "nothing was found
here" -- the one outcome worth counting across a whole class -- and that is
exactly what a zero-byte file already says, so one directory scan answers it for
the entire class. Only the photo actually on screen is parsed.

**Edits are written straight through.** There is no Save button, because a
verification pass is dozens of small corrections and a dialog asking whether to
keep them is a dialog that will eventually be answered wrongly. Deleting the
last box leaves an empty file rather than no file: "I looked, there is nothing
here" is a different statement from "nobody has looked at this photo", and the
whole pipeline is built on the second one being visible.

Standard library only, and no Qt -- the same rule the rest of ``core`` follows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .i18n import tr, trn
from .project import LABEL_SUFFIX, ClassStat, DatasetStatus, Project, class_images

#: The smallest box a user can draw, as a fraction of the photo's shorter side.
#: Below this it is a slip of the mouse while clicking, not an object.
MIN_BOX_FRACTION = 0.004

#: A class with nothing found in at least this share of its photos is reported
#: as a probable prompt failure rather than left to be discovered later. The
#: worker says the same thing at the end of a run; this says it again to the
#: user who came back a week afterwards and never saw that message.
EMPTY_SHARE_WARN = 0.5

#: Photo states, in the order they are worth the user's attention.
MISSING = "missing"
"""No label file: Step 3 never got to this photo."""

EMPTY = "empty"
"""A label file with nothing in it: the model looked and found nothing."""

LABELLED = "labelled"
"""At least one box."""


# --------------------------------------------------------------------------- #
# Boxes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Box:
    """One box, in YOLO's own units: centre and size as fractions of the photo.

    Fractions rather than pixels all the way up to the screen, because the
    photo on screen is a scaled copy -- often decoded at a quarter of its real
    size -- and a box stored in pixels would then have to know which pixels it
    was measured against.
    """

    cx: float
    cy: float
    w: float
    h: float
    index: int = 0
    """The class this box belongs to, as written in the label file.

    Carried rather than assumed so that rewriting a file the app did not create
    cannot silently move its boxes to another class.
    """

    @property
    def corners(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)``, still as fractions."""
        return (
            self.cx - self.w / 2,
            self.cy - self.h / 2,
            self.cx + self.w / 2,
            self.cy + self.h / 2,
        )

    @property
    def area(self) -> float:
        return self.w * self.h

    def contains(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.corners
        return x0 <= x <= x1 and y0 <= y <= y1

    def to_pixels(self, width: float, height: float) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = self.corners
        return x0 * width, y0 * height, x1 * width, y1 * height

    @classmethod
    def from_corners(
        cls,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        index: int = 0,
        minimum: float = MIN_BOX_FRACTION,
    ) -> "Box | None":
        """A box from two corners in any order, or ``None`` if it is not one.

        Corners are clipped to the photo: a user dragging a box around an object
        that runs off the edge drags past the edge, and that is not an error.

        ``minimum`` is the size below which the result is no box at all. It
        guards a mouse slip while drawing, so reading a file leaves it at zero:
        the worker keeps boxes down to two pixels, and a photo losing one of its
        boxes on the way to the screen would be a correction nobody made.
        """
        left, right = sorted((float(x0), float(x1)))
        top, bottom = sorted((float(y0), float(y1)))
        left, right = max(0.0, left), min(1.0, right)
        top, bottom = max(0.0, top), min(1.0, bottom)
        w, h = right - left, bottom - top
        if w <= 0 or h <= 0 or w < minimum or h < minimum:
            return None
        return cls(left + w / 2, top + h / 2, w, h, index=index)


def parse_boxes(text: str) -> tuple[tuple[Box, ...], int]:
    """Boxes from a label file's text, and how many lines made no sense.

    Unreadable lines are counted and dropped rather than raised on. The file may
    have been edited by hand or written by another tool, and refusing to show a
    photo because one of its five lines has a stray character helps nobody --
    but silently rewriting the file without that line would be worse, so the
    count is reported and the page says so before anything is saved.
    """
    boxes: list[Box] = []
    unreadable = 0
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            unreadable += 1
            continue
        try:
            index = int(float(fields[0]))
            cx, cy, w, h = (float(v) for v in fields[1:])
        except ValueError:
            unreadable += 1
            continue
        box = Box.from_corners(
            cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, index=index, minimum=0.0
        )
        if box is None:
            unreadable += 1
            continue
        boxes.append(box)
    return tuple(boxes), unreadable


def format_boxes(boxes) -> str:
    """The text of a label file. Byte-for-byte what the worker writes."""
    return "".join(
        f"{b.index} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}\n" for b in boxes
    )


@dataclass(frozen=True)
class Label:
    """One label file as it is right now."""

    path: Path
    exists: bool
    boxes: tuple[Box, ...] = ()
    unreadable: int = 0

    @property
    def state(self) -> str:
        if not self.exists:
            return MISSING
        return LABELLED if self.boxes else EMPTY


def read_label(path: Path) -> Label:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Label(path, exists=False)
    except OSError:
        # Unreadable is not the same as absent, and saying "not labelled yet"
        # about a file that is there would invite the user to relabel over it.
        return Label(path, exists=True, unreadable=1)
    boxes, unreadable = parse_boxes(text)
    return Label(path, exists=True, boxes=boxes, unreadable=unreadable)


def write_label(path: Path, boxes) -> None:
    """Replace a label file, atomically.

    An empty list writes an empty file, which YOLO reads as "no objects in this
    photo" -- a real answer, and the one the user gives by deleting the last
    wrong box off a photo of nothing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(format_boxes(boxes), encoding="utf-8")
    os.replace(tmp, path)


def clear_label(path: Path) -> None:
    """Remove a label file, returning the photo to "nobody has looked at this".

    Only ever the undo of a box drawn on a photo Step 3 never reached: the app
    otherwise treats deleting a label as a claim it has no business making.
    """
    try:
        Path(path).unlink()
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# What there is to look at
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Photo:
    """One photo and the state of its label, without having read either."""

    path: Path
    label: Path
    state: str = MISSING

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def found_nothing(self) -> bool:
        return self.state == EMPTY


@dataclass(frozen=True)
class ClassReview:
    """What one class holds, counted rather than listed.

    Counts and photos are separated because listing is the expensive half. A
    twenty-class dataset means twenty directory listings of ten thousand
    entries each just to open the page, and nineteen of them are for classes
    the user is not looking at. The counts come from the label folder alone --
    it has one small file per photo and their sizes answer everything the class
    list shows -- and the photos themselves are listed for one class at a time.
    """

    index: int
    name: str
    slug: str
    total: int = 0
    empty: int = 0
    missing: int = 0

    @property
    def with_boxes(self) -> int:
        return max(0, self.total - self.empty - self.missing)

    @property
    def empty_share(self) -> float:
        looked_at = self.total - self.missing
        return self.empty / looked_at if looked_at > 0 else 0.0

    @property
    def looks_wrong(self) -> bool:
        """Whether this class's prompt probably matched nothing at all."""
        return self.total > 0 and self.empty_share >= EMPTY_SHARE_WARN

    def caption(self) -> str:
        """The line shown in the class list."""
        if not self.total:
            return tr("{name} - no photos", name=self.name)
        detail = trn("{n:,} photos", "{n:,} photos", self.total)
        if self.missing:
            detail += trn(", {n:,} not labelled", ", {n:,} not labelled", self.missing)
        if self.empty:
            detail += trn(", nothing found in {n:,}", ", nothing found in {n:,}", self.empty)
        return f"{self.name} - {detail}"

    @classmethod
    def from_photos(cls, index: int, name: str, slug: str, photos) -> "ClassReview":
        photos = tuple(photos)
        return cls(
            index=index,
            name=name,
            slug=slug,
            total=len(photos),
            empty=sum(1 for p in photos if p.state == EMPTY),
            missing=sum(1 for p in photos if p.state == MISSING),
        )


@dataclass(frozen=True)
class Review:
    """Every class of a prepared dataset, counted off the filesystem."""

    classes: tuple[ClassReview, ...] = ()

    @property
    def total(self) -> int:
        return sum(c.total for c in self.classes)

    @property
    def empty(self) -> int:
        return sum(c.empty for c in self.classes)

    @property
    def missing(self) -> int:
        return sum(c.missing for c in self.classes)

    def with_class(self, entry: ClassReview) -> "Review":
        """The same review, with one class's counts brought up to date."""
        return Review(tuple(entry if c.slug == entry.slug else c for c in self.classes))

    def summary(self) -> str:
        if not self.classes:
            return tr("No prepared dataset yet.")
        if not self.total:
            return tr("This dataset has no photos in it.")
        if self.missing == self.total:
            return trn(
                "None of the {n:,} photos have been labelled yet.",
                "None of the {n:,} photos have been labelled yet.",
                self.total,
            )
        found = self.total - self.missing - self.empty
        detail = trn(
            "{found:,} of {n:,} photos have boxes",
            "{found:,} of {n:,} photos have boxes",
            self.total,
            found=found,
        )
        if self.empty:
            detail += tr("; nothing was found in {empty:,}", empty=self.empty)
        if self.missing:
            detail += trn(
                "; {n:,} were never labelled", "; {n:,} were never labelled", self.missing
            )
        return detail + "."

    def problems(self) -> list[str]:
        """Classes whose labels are probably a prompt that matched nothing.

        Worth saying out loud, because the alternative is discovering it as a
        class the trained model never predicts -- which looks like a hard class
        rather than an empty one.
        """
        return [
            trn(
                'Nothing was found in {empty:,} of "{name}"\'s '
                "{n:,} photos ({share:.0%}). Its description "
                "probably does not match what is in them -- change it in Step 3 and "
                "label that class again.",
                'Nothing was found in {empty:,} of "{name}"\'s '
                "{n:,} photos ({share:.0%}). Its description "
                "probably does not match what is in them -- change it in Step 3 and "
                "label that class again.",
                entry.total,
                empty=entry.empty,
                name=entry.name,
                share=entry.empty_share,
            )
            for entry in self.classes
            if entry.looks_wrong
        ]


def visible(photos, *, only_empty: bool = False) -> tuple[int, ...]:
    """Which photos the user is currently paging through."""
    if not only_empty:
        return tuple(range(len(photos)))
    return tuple(i for i, photo in enumerate(photos) if photo.found_nothing)


def class_photos(project: Project, slug: str) -> tuple[Photo, ...]:
    """Every photo of one class with the state of its label, in viewing order.

    Two directory scans and no file reads: an empty label file is a zero-byte
    file, so its size already says "the model looked here and found nothing".
    """
    labels_dir = project.class_labels_dir(slug)
    sizes = _label_sizes(labels_dir)
    photos = []
    for image in class_images(project, slug):
        name = image.stem + LABEL_SUFFIX
        size = sizes.get(name)
        state = MISSING if size is None else (EMPTY if size == 0 else LABELLED)
        photos.append(Photo(path=image, label=labels_dir / name, state=state))
    return tuple(photos)


def count_class(project: Project, stat: ClassStat) -> ClassReview:
    """One class's counts, from the label folder alone."""
    sizes = _label_sizes(project.class_labels_dir(stat.slug))
    # A label file whose photo has been deleted still shows up here. Counting
    # against the photos rather than the labels keeps the total honest even
    # then, at the cost of a stale label hiding one unlabelled photo -- which
    # the class's own photo list corrects as soon as the user opens it.
    labelled = min(len(sizes), stat.images)
    return ClassReview(
        index=stat.index,
        name=stat.name,
        slug=stat.slug,
        total=stat.images,
        empty=min(sum(1 for size in sizes.values() if size == 0), labelled),
        missing=max(0, stat.images - labelled),
    )


def review(project: Project | None, status: DatasetStatus) -> Review:
    if project is None or not status.is_prepared:
        return Review()
    return Review(tuple(count_class(project, stat) for stat in status.classes))


def state_of(label: Path) -> str:
    """The state of one label file, after it has been written to."""
    try:
        return EMPTY if os.stat(label).st_size == 0 else LABELLED
    except OSError:
        return MISSING


def _label_sizes(directory: Path) -> dict[str, int]:
    try:
        with os.scandir(directory) as entries:
            return {
                entry.name: entry.stat().st_size
                for entry in entries
                if entry.is_file() and entry.name.lower().endswith(LABEL_SUFFIX)
            }
    except OSError:
        return {}
