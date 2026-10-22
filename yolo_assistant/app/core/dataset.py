"""Reading an unprepared photo folder, and saying what is wrong with it.

The specification describes the input as "a single folder, which contains one
folder for each class, and each class folder contains the photos. Nothing else
is included". Real folders are never that tidy: they have a ``Thumbs.db``, a
stray PDF, an empty class left over from an abandoned attempt, a class named
``aux`` that Windows refuses to create, and two classes whose names differ only
in case. Every one of those either breaks the run or -- worse -- silently
changes what gets trained.

So this module does not validate. It *describes*: what was found, what will be
done with it, and what the user should know before committing. The page shows
that description and lets the user decide, because the alternative is an error
dialog that says "invalid dataset" about a folder the user can see is fine.

Standard library only, and no Qt: the page scans on a worker thread and the
tests scan directly.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .i18n import tr, trn

#: What counts as a photo. Kept in step with the same list in
#: ``worker/commands/prepare_dataset.py`` -- the worker cannot import this
#: module, since it runs under the managed venv's interpreter with only
#: ``worker`` on its path.
IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)

#: Below this, a class has too few examples for the result to mean anything.
#: Not an error: people do run tiny proof-of-concept datasets deliberately.
THIN_CLASS = 10

ERROR = "error"
WARNING = "warning"

#: Characters no filesystem the app targets will accept, plus the ones that
#: turn a path into an argument. Everything else -- including non-ASCII -- is
#: kept, because a user whose classes are named in Cyrillic or Chinese should
#: not find them renamed to ``class_1``.
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')

#: Windows refuses these as file names whatever the extension, and has since
#: DOS. A class folder called "aux" or "con" is rare but not hypothetical.
_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


@dataclass(frozen=True)
class Problem:
    """Something the user should know before preparing."""

    level: str
    text: str

    @property
    def is_error(self) -> bool:
        return self.level == ERROR


@dataclass(frozen=True)
class SourceClass:
    """One class folder, as found on disk."""

    name: str
    """The folder's name, shown to the user and written into ``classes.txt``."""

    slug: str
    """The name used for folders and files in the prepared dataset."""

    path: Path
    images: tuple[Path, ...] = ()
    #: Files that are not photos, and sub-folders. Counted, never touched.
    ignored_files: int = 0
    ignored_dirs: int = 0

    @property
    def count(self) -> int:
        return len(self.images)

    @property
    def renamed(self) -> bool:
        return self.slug != self.name


@dataclass(frozen=True)
class SourceScan:
    """Everything a photo folder turned out to contain."""

    root: Path
    classes: tuple[SourceClass, ...] = ()
    problems: tuple[Problem, ...] = ()
    loose_files: int = 0

    @property
    def total_images(self) -> int:
        return sum(cls.count for cls in self.classes)

    @property
    def errors(self) -> tuple[Problem, ...]:
        return tuple(p for p in self.problems if p.is_error)

    @property
    def warnings(self) -> tuple[Problem, ...]:
        return tuple(p for p in self.problems if not p.is_error)

    @property
    def is_usable(self) -> bool:
        return not self.errors and self.total_images > 0

    def summary(self) -> str:
        if not self.classes:
            return tr("Nothing to prepare.")
        return (
            trn("{n} class", "{n} classes", len(self.classes))
            + ", "
            + trn("{n:,} photos", "{n:,} photos", self.total_images)
            + "."
        )

    def job_classes(self) -> list[dict]:
        """The class list as the worker wants it: name, slug, and source folder."""
        return [
            {"name": cls.name, "slug": cls.slug, "dir": cls.path.name}
            for cls in self.classes
        ]


def slugify(name: str) -> str:
    """A folder name that every target filesystem will accept.

    Returns ``name`` unchanged whenever it already is one, which is the common
    case -- renaming ``cat`` to ``cat`` would only make the preview noisy.
    """
    # NFC rather than NFKD: the goal is a canonical spelling of the same name,
    # not an ASCII approximation of it. macOS hands out NFD, Windows NFC, and
    # two folders that look identical would otherwise slug differently.
    slug = unicodedata.normalize("NFC", name)
    slug = _ILLEGAL.sub("_", slug)
    # Windows silently strips these, so a folder created as "dog." is opened as
    # "dog" and every path built from the original name misses.
    slug = slug.strip(" .")
    if not slug:
        return "class"
    if slug.lower() in _RESERVED:
        slug = f"{slug}_"
    return slug


def scan(root: str | os.PathLike[str]) -> SourceScan:
    """Describe the photo folder at ``root``."""
    root = Path(root)
    problems: list[Problem] = []

    if not root.exists():
        return SourceScan(
            root, problems=(Problem(ERROR, tr("{path} does not exist.", path=root)),)
        )
    if not root.is_dir():
        return SourceScan(
            root, problems=(Problem(ERROR, tr("{path} is a file, not a folder.", path=root)),)
        )

    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name.casefold())
    except OSError as exc:
        return SourceScan(
            root,
            problems=(Problem(ERROR, tr("{path} cannot be read: {error}", path=root, error=exc)),),
        )

    loose = 0
    classes: list[SourceClass] = []
    for entry in entries:
        if entry.is_dir():
            if entry.name.startswith("."):
                continue  # .git, .DS_Store folders and our own .umla
            classes.append(_scan_class(entry))
        elif entry.is_file():
            loose += 1

    if not classes:
        problems.append(
            Problem(
                ERROR,
                tr(
                    "This folder contains no sub-folders. It needs one folder per class, "
                    "each holding that class's photos -- for example a folder named "
                    '"cat" with photos of cats in it.'
                ),
            )
        )
        return SourceScan(root, problems=tuple(problems), loose_files=loose)

    problems.extend(_collision_problems(classes))

    empty = [cls.name for cls in classes if cls.count == 0]
    if empty and len(empty) == len(classes):
        problems.append(
            Problem(
                ERROR,
                tr(
                    "None of the sub-folders contain photos. Supported types are {types}.",
                    types=", ".join(sorted(s.lstrip(".") for s in IMAGE_SUFFIXES)),
                ),
            )
        )
    elif empty:
        problems.append(
            Problem(
                WARNING,
                trn(
                    "{names} contain no photos and will be skipped. A class with no "
                    "examples cannot be trained and would only add an unused name.",
                    "{names} contain no photos and will be skipped. A class with no "
                    "examples cannot be trained and would only add an unused name.",
                    len(empty),
                    names=_join(empty),
                ),
            )
        )

    thin = [cls.name for cls in classes if 0 < cls.count < THIN_CLASS]
    if thin:
        problems.append(
            Problem(
                WARNING,
                trn(
                    "{names} have fewer than {limit} photos. That is enough to "
                    "run the workflow end to end, but not enough for the trained model to "
                    "be much good -- a few hundred per class is a realistic target.",
                    "{names} have fewer than {limit} photos. That is enough to "
                    "run the workflow end to end, but not enough for the trained model to "
                    "be much good -- a few hundred per class is a realistic target.",
                    len(thin),
                    names=_join(thin),
                    limit=THIN_CLASS,
                ),
            )
        )

    renamed = [cls for cls in classes if cls.renamed]
    if renamed:
        problems.append(
            Problem(
                WARNING,
                tr(
                    "Some folder names cannot be used as-is and will be stored under a "
                    "safe spelling: {renames}. The original name is kept as the class name.",
                    renames=", ".join(
                        tr('"{name}" as "{slug}"', name=c.name, slug=c.slug) for c in renamed
                    ),
                ),
            )
        )

    ignored = sum(cls.ignored_files for cls in classes) + loose
    if ignored:
        problems.append(
            Problem(
                WARNING,
                trn(
                    "{n} file(s) that are not photos were found and will be ignored.",
                    "{n} file(s) that are not photos were found and will be ignored.",
                    ignored,
                ),
            )
        )
    nested = sum(cls.ignored_dirs for cls in classes)
    if nested:
        problems.append(
            Problem(
                WARNING,
                trn(
                    "{n} sub-folder(s) inside the class folders will be ignored. "
                    "Photos have to sit directly in their class folder.",
                    "{n} sub-folder(s) inside the class folders will be ignored. "
                    "Photos have to sit directly in their class folder.",
                    nested,
                ),
            )
        )

    # Drop the empty classes only after reporting them: preparing a class with
    # no images would create a name with nothing behind it, and every later step
    # would have to special-case it.
    keep = tuple(cls for cls in classes if cls.count)
    return SourceScan(root, classes=keep, problems=tuple(problems), loose_files=loose)


def _scan_class(path: Path) -> SourceClass:
    images: list[Path] = []
    ignored_files = 0
    ignored_dirs = 0
    try:
        entries = list(path.iterdir())
    except OSError:
        return SourceClass(path.name, slugify(path.name), path)

    for entry in entries:
        if entry.is_dir():
            ignored_dirs += 1
        elif entry.suffix.lower() in IMAGE_SUFFIXES and not entry.name.startswith("."):
            images.append(entry)
        else:
            ignored_files += 1

    # Sorted so that the numbering the worker assigns is reproducible: running
    # preparation twice on the same folder must give the same file names, or the
    # labels from the first run stop matching the images from the second.
    images.sort(key=lambda p: p.name.casefold())
    return SourceClass(
        name=path.name,
        slug=slugify(path.name),
        path=path,
        images=tuple(images),
        ignored_files=ignored_files,
        ignored_dirs=ignored_dirs,
    )


def _collision_problems(classes: list[SourceClass]) -> list[Problem]:
    """Two classes that would land in the same folder.

    Case is folded because Windows and macOS fold it too: ``Cat`` and ``cat``
    are two classes here and one folder there, and the second would silently
    absorb the first's photos.
    """
    seen: dict[str, list[str]] = {}
    for cls in classes:
        seen.setdefault(cls.slug.casefold(), []).append(cls.name)
    problems = []
    for slug, names in seen.items():
        if len(names) > 1:
            problems.append(
                Problem(
                    ERROR,
                    tr(
                        '{names} would all be stored as "{slug}" and would '
                        "overwrite each other. Rename them so they differ by more than "
                        "capitalisation or punctuation.",
                        names=_join(names),
                        slug=slug,
                    ),
                )
            )
    return problems


def _join(names: list[str]) -> str:
    quoted = [tr('"{name}"', name=n) for n in sorted(names, key=str.casefold)]
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) <= 4:
        return tr("{names} and {last}", names=", ".join(quoted[:-1]), last=quoted[-1])
    return trn(
        "{names} and {n} others", "{names} and {n} others", len(quoted) - 3,
        names=", ".join(quoted[:3]),
    )
