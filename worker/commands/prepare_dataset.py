"""Turn a folder of photos into a dataset YOLO can train on.

The input is what people actually have: ``holiday/cat/IMG_4821.JPG``. The output
is the layout every later step and Ultralytics itself expect, with the photos
copied rather than moved -- the user's originals are never touched, so a mistake
here costs time and not data.

Most of this file is not copying. Copying is four lines. The rest is the set of
problems that are cheap to find now and expensive to find later:

**Unreadable and truncated photos.** A half-downloaded JPEG opens fine, trains
fine for six epochs, and then kills the run at 2am. Every photo is decoded here,
once, and a JPEG missing its end-of-image marker is re-encoded from whatever
decoded successfully rather than carried into the dataset.

**EXIF rotation.** A phone photo stores landscape pixels plus "rotate 90". PIL
honours that tag and OpenCV does not, so the autolabeller would draw boxes on a
rotated image and the trainer would learn them on an upright one -- a dataset
that is wrong in a way no metric reveals. Rotation is applied once, here, and
the tag cleared, so every consumer downstream sees the same pixels.

**Duplicates.** The same photo in two classes is a contradiction the model
cannot resolve and will quietly average out. Identical files are detected by
hash, kept once, and reported.

**The split.** ``train.txt`` and ``val.txt`` are written now and not at training
time, on purpose. A split that is redrawn per run makes two runs incomparable
and leaks held-back photos into training across runs; fixed once per dataset, it
means what it claims to.

Needs only Pillow, which the GUI environment has too -- so the tests drive this
command under the app's own interpreter, with no managed venv in sight.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

from worker.channel import Channel
from worker.i18n import tr, trn
from worker.protocol import Phase
from worker.util.errors import DatasetError

#: Kept in step with ``IMAGE_SUFFIXES`` in the GUI's ``app/core/dataset.py``.
#: The duplication is deliberate: this module runs under the managed venv's
#: interpreter, which has only ``worker`` on its path.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})

#: ``.JPEG``, ``.jpe`` and friends all become this, so a class folder does not
#: end up with four spellings of the same format.
CANONICAL_SUFFIX = {".jpeg": ".jpg", ".tiff": ".tif"}

#: EXIF tag 0x0112. Values other than 1 mean the pixels are stored rotated or
#: mirrored relative to how the photo should be shown.
ORIENTATION_TAG = 0x0112

#: Photos smaller than this in either direction are reported. Not refused:
#: they are legal, just unlikely to be what the user meant.
TINY_PIXELS = 32

#: How many problem photos to name individually before summarising. A dataset
#: where every photo is broken should not produce ten thousand log lines.
MAX_REPORTED = 20


def run(channel: Channel, job: dict) -> dict:
    from PIL import ImageFile

    # Only for the deliberate repair path below: a photo is judged damaged by
    # the explicit checks in ``_place``, never by whether the decoder happened
    # to raise, so softening the decoder here costs nothing and lets a salvaged
    # photo through instead of discarding it.
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    source = Path(job["source"]).expanduser()
    project = Path(job["dest"]).expanduser()
    classes = list(job.get("classes") or [])
    val_split = float(job.get("val_split", 0.2))
    seed = int(job.get("seed", 0))

    if not classes:
        raise DatasetError(tr("No classes to prepare."))
    if not source.is_dir():
        raise DatasetError(tr("The photo folder {folder} is not there any more.", folder=source))
    if not 0.0 <= val_split < 1.0:
        raise DatasetError(
            tr("The validation share must be between 0 and 1, not {value}.", value=val_split)
        )

    dataset = project / "dataset"
    images_root = dataset / "images"
    labels_root = dataset / "labels"

    _guard_destination(source, project)

    channel.progress(Phase.SETUP, 0, 1, unit="steps")
    plan = _enumerate(source, classes)
    total = sum(len(files) for _, files in plan)
    if total == 0:
        raise DatasetError(
            tr(
                "No photos were found in {folder}. Supported types are {types}.",
                folder=source,
                types=", ".join(sorted(s.lstrip(".") for s in IMAGE_SUFFIXES)),
            )
        )

    # A previous attempt's images would otherwise survive under names this run
    # never assigns -- a cancelled run of 900 photos followed by a run of 600
    # leaves 300 orphans with no source and no label.
    _clear(dataset)
    dataset.mkdir(parents=True, exist_ok=True)

    channel.ready(total=total, classes=len(classes))
    channel.log(
        trn(
            "Preparing {n:,} photos from {source} into {project}.",
            "Preparing {n:,} photos from {source} into {project}.",
            total,
            source=source,
            project=project,
        )
    )
    channel.progress(Phase.SETUP, 1, 1, unit="steps")

    report = _Report()
    seen: dict[str, str] = {}
    prepared: list[dict] = []
    started = time.monotonic()
    done = 0
    rate: float | None = None

    with (dataset / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for entry, files in plan:
            slug = entry["slug"]
            class_images = images_root / slug
            class_images.mkdir(parents=True, exist_ok=True)
            # Created empty and left empty. A label file that exists means the
            # photo has been labelled, so pre-filling them would tell Step 3
            # there was nothing left to do.
            (labels_root / slug).mkdir(parents=True, exist_ok=True)

            kept: list[str] = []
            for src in files:
                channel.check_cancelled()
                step_started = time.monotonic()

                digest = _digest(src)
                first = seen.get(digest)
                if first is not None:
                    report.duplicate(channel, src, first)
                else:
                    name = f"{slug}_{len(kept) + 1:06d}{_suffix(src)}"
                    record = _place(src, class_images / name, channel=channel, report=report)
                    if record is not None:
                        relative = f"images/{slug}/{name}"
                        seen[digest] = relative
                        kept.append(relative)
                        record.update({"path": relative, "sha256": digest, "class": entry["name"]})
                        manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

                done += 1
                elapsed = time.monotonic() - step_started
                # An EMA, not a mean over the run: the first few photos pay for
                # a cold page cache and a spinning disk waking up, and a mean
                # would still be apologising for them halfway through.
                rate = elapsed if rate is None else 0.9 * rate + 0.1 * elapsed
                channel.progress(
                    Phase.PREPROCESS,
                    done,
                    total,
                    unit="photos",
                    eta_s=round(rate * (total - done), 1) if rate else None,
                )

            prepared.append({**entry, "images": len(kept), "files": kept})
            channel.log(
                trn("{name}: {n:,} photos.", "{name}: {n:,} photos.", len(kept), name=entry["name"])
            )

    usable = [c for c in prepared if c["images"]]
    if not usable:
        raise DatasetError(
            tr(
                "Not one photo could be read. The files may be damaged, or they may be "
                "something other than photos with a photo's extension."
            )
        )

    channel.progress(Phase.SETUP, 0, 1, unit="steps")
    splits = _split(usable, val_split=val_split, seed=seed)
    _write_text(dataset / "classes.txt", "\n".join(c["name"] for c in usable) + "\n")
    _write_text(dataset / "train.txt", "".join(f"./{p}\n" for p in splits["train"]))
    _write_text(dataset / "val.txt", "".join(f"./{p}\n" for p in splits["val"]))
    _write_text(dataset / "data.yaml", _data_yaml(dataset, usable))

    record = {
        "schema": int(job.get("schema", 1)),
        "source": str(source),
        "prepared_at": time.time(),
        "val_split": val_split,
        "seed": seed,
        "classes": [
            {"name": c["name"], "slug": c["slug"], "source_dir": c["dir"]} for c in usable
        ],
        "counts": {
            "photos": sum(c["images"] for c in usable),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            **report.counts(),
        },
        "warnings": report.warnings(),
    }
    # Written last and by itself: an interrupted run leaves a folder with no
    # prepared.json, which reads as unfinished rather than as ready to use.
    _write_text(dataset / "prepared.json", json.dumps(record, indent=2, ensure_ascii=False))
    channel.progress(Phase.SETUP, 1, 1, unit="steps")

    channel.artifact("dataset", str(dataset / "data.yaml"), classes=len(usable))
    for line in report.warnings():
        channel.warning("dataset", line)
    channel.log(
        tr(
            "Prepared {photos:,} photos in {classes} classes "
            "({train:,} for training, {val:,} held back) in {seconds:.1f}s.",
            photos=record["counts"]["photos"],
            classes=len(usable),
            train=len(splits["train"]),
            val=len(splits["val"]),
            seconds=time.monotonic() - started,
        )
    )

    return {
        "project": str(project),
        "dataset": str(dataset),
        "data_yaml": str(dataset / "data.yaml"),
        "classes": [c["name"] for c in usable],
        **record["counts"],
        "warnings": report.warnings(),
    }


# --------------------------------------------------------------------------- #
# One photo
# --------------------------------------------------------------------------- #


def _place(src: Path, dest: Path, *, channel: Channel, report: "_Report") -> dict | None:
    """Copy one photo into the dataset, or report why it could not be.

    Returns the manifest record, or ``None`` if the photo was rejected.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(src) as probe:
            probe.verify()  # reads the whole file; the object is unusable after
        with Image.open(src) as probe:
            width, height = probe.size
            mode = probe.mode
            orientation = int(probe.getexif().get(ORIENTATION_TAG, 1) or 1)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        report.unreadable(channel, src, exc)
        return None

    if width < TINY_PIXELS or height < TINY_PIXELS:
        report.tiny(channel, src, width, height)

    truncated = _jpeg_is_truncated(src)
    rotated = orientation != 1

    if not rotated and not truncated:
        shutil.copy2(src, dest)
        return {"source": str(src), "width": width, "height": height}

    # Anything that needs fixing is re-encoded rather than copied. Only a small
    # minority of photos take this path, so the recompression cost is bounded
    # and the alternative -- carrying the fault into training -- is not.
    try:
        with Image.open(src) as opened:
            opened.load()
            fixed = ImageOps.exif_transpose(opened) or opened
            fixed = _encodable(fixed, dest.suffix)
            # exif_transpose already drops the orientation tag; saving without
            # the original EXIF block makes sure nothing else re-applies it.
            fixed.save(dest, **_save_options(dest.suffix))
            width, height = fixed.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        report.unreadable(channel, src, exc)
        dest.unlink(missing_ok=True)
        return None

    if truncated:
        report.repaired(channel, src)
    return {
        "source": str(src),
        "width": width,
        "height": height,
        "rotated": rotated or None,
        "repaired": truncated or None,
        "mode": mode,
    }


def _jpeg_is_truncated(path: Path) -> bool:
    """Whether a JPEG is missing its end-of-image marker.

    The overwhelmingly common cause is an interrupted download or a copy off a
    card that was pulled early. PIL's ``verify()`` does not always object, and
    the decoder only gives up once training reaches that file.
    """
    if path.suffix.lower() not in (".jpg", ".jpeg"):
        return False
    try:
        with path.open("rb") as handle:
            handle.seek(-2, os.SEEK_END)
            return handle.read() != b"\xff\xd9"
    except OSError:
        return False


def _encodable(image, suffix: str):
    """Convert a mode the target format cannot store."""
    from PIL import Image

    if suffix not in (".jpg", ".jpeg") or image.mode in ("RGB", "L"):
        return image
    if image.mode in ("RGBA", "LA", "P"):
        # JPEG has no alpha channel and no palette. Flattening onto white is
        # what every viewer shows for a transparent PNG, so the dataset matches
        # what the user saw when they chose the photo.
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        return background
    return image.convert("RGB")


def _save_options(suffix: str) -> dict:
    if suffix in (".jpg", ".jpeg"):
        # High enough that a re-encode is visually lossless, low enough that a
        # rotated photo does not double in size.
        return {"quality": 95, "optimize": True}
    return {}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    return CANONICAL_SUFFIX.get(suffix, suffix)


# --------------------------------------------------------------------------- #
# The dataset as a whole
# --------------------------------------------------------------------------- #


def _enumerate(source: Path, classes: list[dict]) -> list[tuple[dict, list[Path]]]:
    plan: list[tuple[dict, list[Path]]] = []
    for entry in classes:
        directory = source / str(entry.get("dir") or entry["name"])
        try:
            files = [
                p
                for p in directory.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_SUFFIXES
                and not p.name.startswith(".")
            ]
        except OSError as exc:
            raise DatasetError(
                tr("{path} cannot be read: {error}", path=directory, error=exc)
            ) from exc
        # Same ordering rule as the GUI's scan, so preparing twice assigns the
        # same numbers and labels made against the first run still match.
        files.sort(key=lambda p: p.name.casefold())
        plan.append((entry, files))
    return plan


def _split(classes: list[dict], *, val_split: float, seed: int) -> dict[str, list[str]]:
    """A per-class split, held back in the same proportion from each.

    Splitting the dataset as a whole would, with an unbalanced dataset and bad
    luck, hold back nearly all of a small class or none of it -- and a class
    with no validation examples reports an accuracy of zero that means nothing.
    """
    train: list[str] = []
    val: list[str] = []
    for entry in classes:
        files = list(entry["files"])
        # Seeded per class name, so adding a class does not reshuffle the
        # others and invalidate a comparison with an earlier run.
        random.Random(f"{seed}:{entry['slug']}").shuffle(files)
        count = len(files)
        held = int(round(count * val_split))
        if val_split > 0 and count >= 2:
            held = min(max(held, 1), count - 1)  # never all of a class, never none
        val.extend(files[:held])
        train.extend(files[held:])
    train.sort()
    val.sort()
    return {"train": train, "val": val}


def _data_yaml(dataset: Path, classes: list[dict]) -> str:
    """The file ``yolo train data=...`` is pointed at.

    Written by hand rather than with PyYAML, which the managed environment has
    but this command should not need. Values go through ``json.dumps``: YAML 1.2
    is a superset of JSON, so a JSON string is always a valid, correctly escaped
    YAML scalar -- which matters for a class named ``50% off`` or ``a: b``.
    """
    lines = [
        "# Generated by YOLO Assistant.",
        "# train/val list the same photos every run, so two runs stay comparable.",
        f"path: {json.dumps(str(dataset))}",
        "train: train.txt",
        "val: val.txt",
        "",
        "names:",
    ]
    lines += [f"  {index}: {json.dumps(entry['name'])}" for index, entry in enumerate(classes)]
    return "\n".join(lines) + "\n"


def _guard_destination(source: Path, project: Path) -> None:
    """Refuse a destination that would eat its own input.

    Preparing ``~/photos`` into ``~/photos`` would have the app copy files into
    the folder it is walking. The check is on resolved paths because
    ``photos/../photos`` is the same folder under a different name.
    """
    source = source.resolve()
    try:
        destination = project.resolve()
    except OSError as exc:  # pragma: no cover - only on a vanished parent
        raise DatasetError(tr("{path} cannot be used: {error}", path=project, error=exc)) from exc
    if destination == source or source in destination.parents:
        raise DatasetError(
            tr(
                "The project folder cannot be inside the photo folder. Choose a folder "
                "next to it instead, so the originals stay untouched."
            )
        )


def _clear(dataset: Path) -> None:
    if dataset.exists():
        shutil.rmtree(dataset, ignore_errors=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class _Report:
    """Counts and a bounded list of examples for everything that went wrong."""

    def __init__(self) -> None:
        self.unreadable_count = 0
        self.duplicate_count = 0
        self.repaired_count = 0
        self.tiny_count = 0
        self._lines: list[str] = []

    def unreadable(self, channel: Channel, path: Path, exc: BaseException) -> None:
        self.unreadable_count += 1
        self._note(
            channel,
            tr("Skipped {name}: it could not be read ({error}).", name=path.name, error=exc),
            "warning",
        )

    def duplicate(self, channel: Channel, path: Path, first: str) -> None:
        self.duplicate_count += 1
        self._note(
            channel,
            tr(
                "Skipped {name}: identical to {first}. The same photo in two classes "
                "teaches the model two different answers to the same question.",
                name=path.name,
                first=first,
            ),
            "warning",
        )

    def repaired(self, channel: Channel, path: Path) -> None:
        self.repaired_count += 1
        self._note(
            channel, tr("Repaired {name}: it was incomplete and has been re-saved.", name=path.name)
        )

    def tiny(self, channel: Channel, path: Path, width: int, height: int) -> None:
        self.tiny_count += 1
        self._note(
            channel,
            tr(
                "{name} is only {width}x{height} pixels.",
                name=path.name,
                width=width,
                height=height,
            ),
        )

    def _note(self, channel: Channel, text: str, level: str = "info") -> None:
        total = self.unreadable_count + self.duplicate_count + self.repaired_count + self.tiny_count
        if total <= MAX_REPORTED:
            channel.log(text, level=level)
        elif total == MAX_REPORTED + 1:
            channel.log(tr("Further problems will be counted but not listed."), level="warning")

    def counts(self) -> dict:
        return {
            "unreadable": self.unreadable_count,
            "duplicates": self.duplicate_count,
            "repaired": self.repaired_count,
            "tiny": self.tiny_count,
        }

    def warnings(self) -> list[str]:
        lines = []
        # The GUI rebuilds these from the counts in prepared.json, in whatever
        # language it is in by then; see app/core/project.py.
        if self.unreadable_count:
            lines.append(
                trn(
                    "{n} photo(s) could not be read and were left out.",
                    "{n} photo(s) could not be read and were left out.",
                    self.unreadable_count,
                )
            )
        if self.duplicate_count:
            lines.append(
                trn(
                    "{n} photo(s) were exact duplicates and were kept only once.",
                    "{n} photo(s) were exact duplicates and were kept only once.",
                    self.duplicate_count,
                )
            )
        if self.repaired_count:
            lines.append(
                trn(
                    "{n} photo(s) were incomplete and were repaired on the way in.",
                    "{n} photo(s) were incomplete and were repaired on the way in.",
                    self.repaired_count,
                )
            )
        if self.tiny_count:
            lines.append(
                trn(
                    "{n} photo(s) are smaller than {pixels} pixels; "
                    "objects in them may be too small to learn.",
                    "{n} photo(s) are smaller than {pixels} pixels; "
                    "objects in them may be too small to learn.",
                    self.tiny_count,
                    pixels=TINY_PIXELS,
                )
            )
        return lines
