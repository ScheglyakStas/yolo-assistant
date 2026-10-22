"""The project: one folder holding a prepared dataset, its runs and its exports.

Everything after Step 2 works inside a project, so this module answers three
questions for every later step: where does a given kind of file live, what is
currently in there, and which project is the user working on.

The layout is deliberately plain, and deliberately the layout Ultralytics
already understands::

    <project>/
        dataset/
            images/<class>/<class>_000001.jpg
            labels/<class>/<class>_000001.txt
            classes.txt        index -> name, the order baked into every label
            data.yaml          what `yolo train data=...` is pointed at
            train.txt val.txt  the split, fixed once so runs stay comparable
            manifest.jsonl     where each photo came from
            prepared.json      proof that preparation finished
            autolabel.json     what the last autolabeling run was asked to find
        runs/                  training output
        exports/               converted models

so that a user who outgrows the app can point the ``yolo`` command at
``dataset/data.yaml`` and carry on without unpicking anything.

``prepared.json`` is the only file here that records anything about the dataset
rather than deriving it, and it records the same kind of thing ``ready.json``
does: a fact about an event that
happened -- which folder the photos came from, and what order the class indices
were assigned in. Counts are never stored, because users delete folders between
sessions and a stored count becomes a lie the moment they do.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .dataset import IMAGE_SUFFIXES
from .i18n import tr, trn
from .paths import projects_dir, state_file

#: Bump when the on-disk layout changes in a way an existing project cannot be
#: read with. Step 2 then offers to prepare it again rather than misreading it.
PROJECT_SCHEMA = 1

PREPARED_FILENAME = "prepared.json"
CLASSES_FILENAME = "classes.txt"
DATA_YAML_FILENAME = "data.yaml"
MANIFEST_FILENAME = "manifest.jsonl"

LABEL_SUFFIX = ".txt"

#: Kept in step with ``TINY_PIXELS`` in ``worker/commands/prepare_dataset.py``,
#: which counts the photos smaller than this that the warnings here report.
TINY_PIXELS = 32

#: Dataset states, in the order a project moves through them.
ABSENT = "absent"
PARTIAL = "partial"
STALE = "stale"
PREPARED = "prepared"


@dataclass(frozen=True)
class Project:
    """A project directory. Cheap to construct; touches no disk until asked."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser())

    # -- layout ------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def images_dir(self) -> Path:
        return self.dataset_dir / "images"

    @property
    def labels_dir(self) -> Path:
        return self.dataset_dir / "labels"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def internal_dir(self) -> Path:
        """The app's own bookkeeping: job logs, plans. Nothing the user needs."""
        return self.root / ".umla"

    @property
    def prepared_file(self) -> Path:
        return self.dataset_dir / PREPARED_FILENAME

    @property
    def classes_file(self) -> Path:
        return self.dataset_dir / CLASSES_FILENAME

    @property
    def data_yaml(self) -> Path:
        return self.dataset_dir / DATA_YAML_FILENAME

    @property
    def manifest_file(self) -> Path:
        return self.dataset_dir / MANIFEST_FILENAME

    def log_path(self, name: str) -> Path:
        return self.internal_dir / "logs" / f"{name}.log"

    def class_images_dir(self, class_slug: str) -> Path:
        return self.images_dir / class_slug

    def class_labels_dir(self, class_slug: str) -> Path:
        return self.labels_dir / class_slug

    def label_for(self, image: Path) -> Path:
        """The label file belonging to ``image``.

        Mirrors Ultralytics' own rule -- swap ``/images/`` for ``/labels/`` and
        change the extension -- rather than inventing a second one, because the
        two disagreeing is a whole afternoon of "why does it say no labels
        found".
        """
        relative = Path(image).resolve().relative_to(self.images_dir.resolve())
        return (self.labels_dir / relative).with_suffix(LABEL_SUFFIX)

    def to_job(self) -> dict:
        """The subset a worker command needs."""
        return {
            "root": str(self.root),
            "dataset": str(self.dataset_dir),
            "images": str(self.images_dir),
            "labels": str(self.labels_dir),
        }


@dataclass(frozen=True)
class ClassStat:
    """One class of a prepared dataset, counted now rather than remembered."""

    index: int
    name: str
    slug: str
    images: int = 0
    labels: int = 0

    @property
    def is_labelled(self) -> bool:
        return self.images > 0 and self.labels >= self.images

    @property
    def missing_labels(self) -> int:
        return max(0, self.images - self.labels)


@dataclass(frozen=True)
class DatasetStatus:
    """What is in a project's dataset folder right now."""

    state: str
    message: str
    classes: tuple[ClassStat, ...] = ()
    source: Path | None = None
    prepared_at: float = 0.0
    schema: int = PROJECT_SCHEMA
    warnings: tuple[str, ...] = ()

    @property
    def is_prepared(self) -> bool:
        return self.state == PREPARED

    @property
    def total_images(self) -> int:
        return sum(c.images for c in self.classes)

    @property
    def total_labels(self) -> int:
        return sum(c.labels for c in self.classes)

    @property
    def fully_labelled(self) -> bool:
        """Whether Step 3 can be skipped.

        Note that an empty label file is a perfectly valid label meaning "no
        objects in this photo", so this counts files and not their contents.
        """
        return bool(self.classes) and all(c.is_labelled for c in self.classes)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.classes]


# --------------------------------------------------------------------------- #
# Reading a project
# --------------------------------------------------------------------------- #


def read_classes(project: Project) -> list[str]:
    """The class names in index order, as recorded when the dataset was prepared.

    This ordering is load-bearing: every label file refers to a class by its
    number, so recomputing the list from whatever folders happen to exist would
    silently relabel the whole dataset the first time a user deletes one.
    """
    try:
        text = project.classes_file.read_text(encoding="utf-8")
    except OSError:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_prepared(project: Project) -> dict | None:
    try:
        obj = json.loads(project.prepared_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def status(project: Project | None) -> DatasetStatus:
    """Decide, from the filesystem alone, what state the dataset is in."""
    if project is None:
        return DatasetStatus(ABSENT, tr("No project selected yet."))

    record = read_prepared(project)
    if record is None:
        if project.images_dir.exists():
            return DatasetStatus(
                PARTIAL,
                tr(
                    "This folder holds a dataset that was never finished -- preparation "
                    "was interrupted or cancelled. Preparing it again starts from a "
                    "clean folder."
                ),
            )
        return DatasetStatus(
            ABSENT,
            tr("No dataset has been prepared in this folder yet."),
        )

    schema = int(record.get("schema", 0))
    if schema != PROJECT_SCHEMA:
        return DatasetStatus(
            STALE,
            tr(
                "This dataset was prepared by a different version of the app and has to "
                "be prepared again before it can be used."
            ),
            schema=schema,
        )

    classes = _count_classes(project, record)
    source = record.get("source")
    warnings = _warnings(record)

    if not classes:
        return DatasetStatus(
            PARTIAL,
            tr(
                "The dataset record is here but its photos are gone. Prepare the dataset "
                "again, or point the app at the folder they were moved to."
            ),
            source=Path(source) if source else None,
            warnings=warnings,
        )

    total = sum(c.images for c in classes)
    labelled = sum(c.labels for c in classes)
    counts = (
        trn("{n} class", "{n} classes", len(classes))
        + ", "
        + trn("{n:,} photos", "{n:,} photos", total)
    )
    if labelled >= total:
        detail = tr("{counts} -- all labelled.", counts=counts)
    elif labelled:
        detail = tr(
            "{counts} -- {labelled:,} labelled, {remaining:,} still to do.",
            counts=counts,
            labelled=labelled,
            remaining=total - labelled,
        )
    else:
        detail = counts + "."
    return DatasetStatus(
        PREPARED,
        detail,
        classes=tuple(classes),
        source=Path(source) if source else None,
        prepared_at=float(record.get("prepared_at") or 0.0),
        schema=schema,
        warnings=warnings,
    )


def _warnings(record: dict) -> tuple[str, ...]:
    """What preparation found wrong with the photos, in the current language.

    Rebuilt from the recorded counts rather than read back from the sentences
    the worker stored beside them, which are in whatever language the app was
    in on the day the dataset was prepared.
    """
    counts = record.get("counts")
    if not isinstance(counts, dict):
        return tuple(str(w) for w in (record.get("warnings") or []))

    def count(key: str) -> int:
        try:
            return int(counts.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    lines = []
    if unreadable := count("unreadable"):
        lines.append(
            trn(
                "{n} photo(s) could not be read and were left out.",
                "{n} photo(s) could not be read and were left out.",
                unreadable,
            )
        )
    if duplicates := count("duplicates"):
        lines.append(
            trn(
                "{n} photo(s) were exact duplicates and were kept only once.",
                "{n} photo(s) were exact duplicates and were kept only once.",
                duplicates,
            )
        )
    if repaired := count("repaired"):
        lines.append(
            trn(
                "{n} photo(s) were incomplete and were repaired on the way in.",
                "{n} photo(s) were incomplete and were repaired on the way in.",
                repaired,
            )
        )
    if tiny := count("tiny"):
        lines.append(
            trn(
                "{n} photo(s) are smaller than {pixels} pixels; "
                "objects in them may be too small to learn.",
                "{n} photo(s) are smaller than {pixels} pixels; "
                "objects in them may be too small to learn.",
                tiny,
                pixels=TINY_PIXELS,
            )
        )
    return tuple(lines)


def _count_classes(project: Project, record: dict) -> list[ClassStat]:
    stats: list[ClassStat] = []
    for index, entry in enumerate(record.get("classes") or []):
        if not isinstance(entry, dict):
            continue
        slug = str(entry.get("slug") or entry.get("name") or "")
        name = str(entry.get("name") or slug)
        images = _count_suffixes(project.class_images_dir(slug), IMAGE_SUFFIXES)
        labels = _count_suffixes(project.class_labels_dir(slug), {LABEL_SUFFIX})
        if images == 0 and labels == 0 and not project.class_images_dir(slug).exists():
            continue
        stats.append(ClassStat(index=index, name=name, slug=slug, images=images, labels=labels))
    return stats


def _count_suffixes(directory: Path, suffixes) -> int:
    try:
        with os.scandir(directory) as entries:
            return sum(
                1
                for e in entries
                if e.is_file() and os.path.splitext(e.name)[1].lower() in suffixes
            )
    except OSError:
        return 0


def class_images(project: Project, class_slug: str) -> list[Path]:
    """Every photo of one class, in the order Step 4 will page through them."""
    directory = project.class_images_dir(class_slug)
    try:
        found = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    except OSError:
        return []
    found.sort(key=lambda p: p.name.casefold())
    return found


# --------------------------------------------------------------------------- #
# Which project the user is working on
# --------------------------------------------------------------------------- #


def current() -> Project | None:
    """The project the user last worked on, if it is still there."""
    try:
        obj = json.loads(state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    root = obj.get("project") if isinstance(obj, dict) else None
    if not root:
        return None
    path = Path(root)
    # A pointer to a folder the user has since deleted or unplugged is not an
    # error worth reporting; it just means there is no current project.
    return Project(path) if path.is_dir() else None


def set_current(project: Project | None) -> None:
    path = state_file()
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            obj = {}
    except (OSError, ValueError):
        obj = {}
    obj["project"] = str(project.root) if project is not None else ""
    obj["updated_at"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


_UNSAFE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def default_root_for(source: Path) -> Path:
    """Where to suggest putting the project for a given photo folder.

    Beside the photos, not under the app's data directory. Datasets are large
    and the copy has to land on a drive with room for it, which is far more
    likely to be the drive the photos are already on than the one holding
    ``AppData``. When that is not writable -- a mounted share, a memory card,
    a read-only archive -- the app's own projects folder is the fallback.
    """
    source = Path(source).expanduser()
    stem = _UNSAFE_NAME.sub("_", source.name).strip(" .") or "dataset"
    candidate = source.parent / f"{stem}-project"
    if _writable(source.parent):
        return candidate
    return projects_dir() / f"{stem}-project"


def _writable(directory: Path) -> bool:
    return directory.is_dir() and os.access(directory, os.W_OK)
