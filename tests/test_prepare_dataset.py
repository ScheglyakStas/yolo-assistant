"""Preparation, driven directly against real image files.

Real files rather than mocks, because every interesting thing this command does
-- detecting a truncated JPEG, honouring an EXIF rotation, hashing a duplicate
-- is a property of the bytes on disk. A mocked Pillow would only test that the
code calls the functions it calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from worker.commands import prepare_dataset
from worker.util.errors import DatasetError


class FakeChannel:
    """The subset of ``Channel`` a command touches, with a cancel switch."""

    def __init__(self, cancel_after: int | None = None) -> None:
        self.logs: list[tuple[str, str]] = []
        self.warnings: list[str] = []
        self.artifacts: list[tuple[str, str]] = []
        self.progress_calls: list[tuple[str, int, int | None]] = []
        self.ready_info: dict = {}
        self._cancel_after = cancel_after
        self._checks = 0

    def progress(self, phase, current, total=None, *, unit="items", eta_s=None):
        self.progress_calls.append((getattr(phase, "value", phase), current, total))

    def ready(self, **info):
        self.ready_info = info

    def log(self, msg, *, level="info"):
        self.logs.append((level, msg))

    def warning(self, code, msg, *, hint=""):
        self.warnings.append(msg)

    def artifact(self, kind, path, **meta):
        self.artifacts.append((kind, path))

    def check_cancelled(self):
        self._checks += 1
        if self._cancel_after is not None and self._checks > self._cancel_after:
            from worker.channel import Cancelled

            raise Cancelled()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def write_image(path: Path, size=(64, 48), colour=(10, 120, 200), **save):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, **save)
    return path


def write_series(directory: Path, count: int, suffix: str = ".jpg") -> None:
    """``count`` photos that are genuinely different files.

    Varying the size rather than the colour, because a flat-colour JPEG of two
    nearly identical colours compresses to identical bytes -- which the
    duplicate detector then quite correctly collapses into one photo, and the
    test that was meant to be about something else silently stops testing it.
    """
    for index in range(count):
        write_image(directory / f"{index}{suffix}", size=(64, 48 + index))


@pytest.fixture
def source(tmp_path):
    """Two classes, four photos, all sound."""
    root = tmp_path / "photos"
    for index in range(3):
        write_image(root / "cat" / f"IMG_{index}.jpg", size=(64, 48 + index))
    write_image(root / "dog" / "a.png", colour=(0, 200, 0))
    return root


def job_for(source: Path, dest: Path, classes=None, **extra):
    return {
        "source": str(source),
        "dest": str(dest),
        "classes": classes
        or [
            {"name": "cat", "slug": "cat", "dir": "cat"},
            {"name": "dog", "slug": "dog", "dir": "dog"},
        ],
        "val_split": 0.0,
        **extra,
    }


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_photos_are_copied_renamed_and_laid_out(tmp_path, source):
    project = tmp_path / "project"
    channel = FakeChannel()

    result = prepare_dataset.run(channel, job_for(source, project))

    dataset = project / "dataset"
    assert sorted(p.name for p in (dataset / "images" / "cat").iterdir()) == [
        "cat_000001.jpg",
        "cat_000002.jpg",
        "cat_000003.jpg",
    ]
    assert (dataset / "images" / "dog" / "dog_000001.png").exists()
    assert result["photos"] == 4

    # The label folders exist and are empty: an existing label file is what
    # tells Step 3 a photo has been done.
    assert (dataset / "labels" / "cat").is_dir()
    assert list((dataset / "labels" / "cat").iterdir()) == []


def test_the_originals_are_left_alone(tmp_path, source):
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}

    prepare_dataset.run(FakeChannel(), job_for(source, tmp_path / "project"))

    assert {p: p.read_bytes() for p in source.rglob("*") if p.is_file()} == before


def test_classes_txt_records_the_index_order(tmp_path, source):
    project = tmp_path / "project"
    prepare_dataset.run(FakeChannel(), job_for(source, project))

    text = (project / "dataset" / "classes.txt").read_text(encoding="utf-8")
    assert text.splitlines() == ["cat", "dog"]


def test_data_yaml_points_at_the_split_and_names_the_classes(tmp_path, source):
    project = tmp_path / "project"
    prepare_dataset.run(FakeChannel(), job_for(source, project))

    yaml = pytest.importorskip("yaml")
    loaded = yaml.safe_load((project / "dataset" / "data.yaml").read_text(encoding="utf-8"))

    assert loaded["path"] == str(project / "dataset")
    assert loaded["train"] == "train.txt"
    assert loaded["names"] == {0: "cat", 1: "dog"}


def test_class_names_that_would_break_yaml_survive_it(tmp_path):
    source = tmp_path / "photos"
    write_image(source / "odd" / "a.jpg")
    yaml = pytest.importorskip("yaml")

    prepare_dataset.run(
        FakeChannel(),
        job_for(
            source,
            tmp_path / "project",
            classes=[{"name": 'a: b "c" #d', "slug": "odd", "dir": "odd"}],
        ),
    )

    loaded = yaml.safe_load((tmp_path / "project" / "dataset" / "data.yaml").read_text("utf-8"))
    assert loaded["names"] == {0: 'a: b "c" #d'}


def test_split_lists_are_relative_so_the_project_can_be_moved(tmp_path, source):
    project = tmp_path / "project"
    prepare_dataset.run(FakeChannel(), job_for(source, project, val_split=0.5))

    lines = (project / "dataset" / "train.txt").read_text(encoding="utf-8").splitlines()

    # Ultralytics expands a leading "./" against the list file's own folder.
    assert lines and all(line.startswith("./images/") for line in lines)


def test_the_split_holds_back_a_share_of_every_class(tmp_path):
    source = tmp_path / "photos"
    write_series(source / "cat", 10)
    write_series(source / "dog", 4, suffix=".png")

    project = tmp_path / "project"
    result = prepare_dataset.run(FakeChannel(), job_for(source, project, val_split=0.2))

    val = (project / "dataset" / "val.txt").read_text(encoding="utf-8").splitlines()
    assert result["train"] + result["val"] == 14
    # Stratified: a class must never be entirely in or entirely out of the
    # held-back set, or its reported accuracy means nothing.
    assert sum("/cat/" in line for line in val) == 2
    assert sum("/dog/" in line for line in val) == 1


def test_a_tiny_class_keeps_at_least_one_photo_for_training(tmp_path):
    source = tmp_path / "photos"
    write_image(source / "cat" / "only.jpg")

    project = tmp_path / "project"
    result = prepare_dataset.run(
        FakeChannel(),
        job_for(source, project, classes=[{"name": "cat", "slug": "cat", "dir": "cat"}],
                val_split=0.5),
    )

    assert result["train"] == 1 and result["val"] == 0


def test_the_split_is_the_same_every_time(tmp_path):
    source = tmp_path / "photos"
    write_series(source / "cat", 20)

    classes = [{"name": "cat", "slug": "cat", "dir": "cat"}]
    splits = []
    for run_index in range(2):
        project = tmp_path / f"project{run_index}"
        prepare_dataset.run(FakeChannel(), job_for(source, project, classes, val_split=0.25))
        splits.append((project / "dataset" / "val.txt").read_text(encoding="utf-8"))

    assert splits[0] == splits[1], "preparing twice must not reshuffle the split"


def test_the_manifest_says_where_each_photo_came_from(tmp_path, source):
    project = tmp_path / "project"
    prepare_dataset.run(FakeChannel(), job_for(source, project))

    lines = (project / "dataset" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]

    assert len(records) == 4
    first = next(r for r in records if r["path"] == "images/cat/cat_000001.jpg")
    assert Path(first["source"]) == source / "cat" / "IMG_0.jpg"
    assert first["width"] == 64 and first["height"] == 48
    assert first["class"] == "cat"


def test_prepared_json_is_the_proof_and_is_written_last(tmp_path, source):
    project = tmp_path / "project"
    prepare_dataset.run(FakeChannel(), job_for(source, project))

    record = json.loads((project / "dataset" / "prepared.json").read_text(encoding="utf-8"))
    assert record["source"] == str(source)
    assert [c["slug"] for c in record["classes"]] == ["cat", "dog"]
    assert record["counts"]["photos"] == 4


def test_progress_covers_every_photo(tmp_path, source):
    channel = FakeChannel()
    prepare_dataset.run(channel, job_for(source, tmp_path / "project"))

    preprocess = [c for c in channel.progress_calls if c[0] == "preprocess"]
    assert preprocess[-1] == ("preprocess", 4, 4), "the bar has to reach its end"
    assert channel.ready_info == {"total": 4, "classes": 2}


# --------------------------------------------------------------------------- #
# The photos people actually have
# --------------------------------------------------------------------------- #


def test_an_unreadable_file_is_skipped_and_reported(tmp_path):
    source = tmp_path / "photos"
    write_image(source / "cat" / "good.jpg")
    (source / "cat" / "bad.jpg").write_bytes(b"this is not a JPEG at all")

    project = tmp_path / "project"
    channel = FakeChannel()
    result = prepare_dataset.run(
        channel,
        job_for(source, project, classes=[{"name": "cat", "slug": "cat", "dir": "cat"}]),
    )

    assert result["photos"] == 1
    assert result["unreadable"] == 1
    # Numbering is contiguous: the skipped file must not leave a gap, or the
    # count of files and the highest number would disagree.
    assert [p.name for p in (project / "dataset" / "images" / "cat").iterdir()] == [
        "cat_000001.jpg"
    ]
    assert any("could not be read" in msg for _, msg in channel.logs)


def test_a_duplicate_photo_is_kept_once(tmp_path):
    source = tmp_path / "photos"
    write_image(source / "cat" / "a.jpg")
    (source / "cat" / "copy.jpg").write_bytes((source / "cat" / "a.jpg").read_bytes())

    project = tmp_path / "project"
    result = prepare_dataset.run(
        FakeChannel(),
        job_for(source, project, classes=[{"name": "cat", "slug": "cat", "dir": "cat"}]),
    )

    assert result["photos"] == 1 and result["duplicates"] == 1


def test_the_same_photo_in_two_classes_is_kept_once(tmp_path):
    source = tmp_path / "photos"
    write_image(source / "cat" / "a.jpg")
    write_image(source / "dog" / "b.jpg")
    (source / "dog" / "b.jpg").write_bytes((source / "cat" / "a.jpg").read_bytes())

    project = tmp_path / "project"
    channel = FakeChannel()
    result = prepare_dataset.run(channel, job_for(source, project))

    # Two classes claiming the same pixels is a contradiction, not extra data.
    assert result["photos"] == 1 and result["duplicates"] == 1
    assert any("two classes" in msg for _, msg in channel.logs)


def test_an_exif_rotation_is_baked_into_the_pixels(tmp_path):
    source = tmp_path / "photos"
    path = source / "cat" / "phone.jpg"
    path.parent.mkdir(parents=True)
    image = Image.new("RGB", (100, 50), (200, 0, 0))
    exif = image.getexif()
    exif[prepare_dataset.ORIENTATION_TAG] = 6  # "rotate 90 clockwise"
    image.save(path, exif=exif)

    project = tmp_path / "project"
    prepare_dataset.run(
        FakeChannel(),
        job_for(source, project, classes=[{"name": "cat", "slug": "cat", "dir": "cat"}]),
    )

    with Image.open(project / "dataset" / "images" / "cat" / "cat_000001.jpg") as out:
        # Pillow honours the tag and OpenCV does not, so the two would disagree
        # about this photo's shape unless the rotation is applied now.
        assert out.size == (50, 100)
        assert out.getexif().get(prepare_dataset.ORIENTATION_TAG, 1) == 1


def test_a_truncated_jpeg_is_repaired_rather_than_carried_in(tmp_path):
    source = tmp_path / "photos"
    path = write_image(source / "cat" / "cut.jpg", size=(128, 128))
    path.write_bytes(path.read_bytes()[:-64])  # lose the end-of-image marker

    project = tmp_path / "project"
    channel = FakeChannel()
    result = prepare_dataset.run(
        channel,
        job_for(source, project, classes=[{"name": "cat", "slug": "cat", "dir": "cat"}]),
    )

    assert result["repaired"] == 1
    out = project / "dataset" / "images" / "cat" / "cat_000001.jpg"
    assert out.read_bytes()[-2:] == b"\xff\xd9", "the copy is still truncated"
    assert any("repaired" in w.lower() for w in channel.warnings)


def test_a_transparent_png_saved_as_jpeg_does_not_fail(tmp_path):
    source = tmp_path / "photos"
    path = source / "cat" / "logo.jpg"
    path.parent.mkdir(parents=True)
    # A PNG-style RGBA image with a .jpg name, plus a rotation to force the
    # re-encode path. JPEG has no alpha channel, so this used to be a crash.
    image = Image.new("RGBA", (40, 60), (255, 0, 0, 128))
    exif = image.getexif()
    exif[prepare_dataset.ORIENTATION_TAG] = 3
    image.convert("RGB").save(path, exif=exif)

    result = prepare_dataset.run(
        FakeChannel(),
        job_for(
            source, tmp_path / "project",
            classes=[{"name": "cat", "slug": "cat", "dir": "cat"}],
        ),
    )
    assert result["photos"] == 1


def test_jpeg_spellings_are_normalised(tmp_path):
    source = tmp_path / "photos"
    write_image(source / "cat" / "a.JPEG")

    prepare_dataset.run(
        FakeChannel(),
        job_for(source, tmp_path / "project",
                classes=[{"name": "cat", "slug": "cat", "dir": "cat"}]),
    )

    images = tmp_path / "project" / "dataset" / "images" / "cat"
    assert [p.name for p in images.iterdir()] == ["cat_000001.jpg"]


def test_a_tiny_photo_is_reported_but_kept(tmp_path):
    source = tmp_path / "photos"
    write_image(source / "cat" / "speck.png", size=(8, 8))

    result = prepare_dataset.run(
        FakeChannel(),
        job_for(source, tmp_path / "project",
                classes=[{"name": "cat", "slug": "cat", "dir": "cat"}]),
    )

    assert result["photos"] == 1 and result["tiny"] == 1


# --------------------------------------------------------------------------- #
# Refusals and restarts
# --------------------------------------------------------------------------- #


def test_a_project_inside_the_photo_folder_is_refused(tmp_path, source):
    with pytest.raises(DatasetError, match="cannot be inside"):
        prepare_dataset.run(FakeChannel(), job_for(source, source / "output"))


def test_a_photo_folder_with_nothing_in_it_is_refused(tmp_path):
    source = tmp_path / "photos"
    (source / "cat").mkdir(parents=True)

    with pytest.raises(DatasetError, match="No photos were found"):
        prepare_dataset.run(
            FakeChannel(),
            job_for(source, tmp_path / "project",
                    classes=[{"name": "cat", "slug": "cat", "dir": "cat"}]),
        )


def test_a_dataset_error_carries_its_own_code():
    from worker.__main__ import _classify
    from worker.protocol import ErrorCode

    code, hint = _classify(DatasetError("nope"))

    assert code is ErrorCode.DATASET_INVALID
    assert "one sub-folder per class" in hint


def test_cancelling_leaves_no_proof_behind(tmp_path, source):
    project = tmp_path / "project"
    from worker.channel import Cancelled

    with pytest.raises(Cancelled):
        prepare_dataset.run(FakeChannel(cancel_after=2), job_for(source, project))

    # Photos were copied, but nothing claims the dataset is ready -- which is
    # what makes the interrupted state recognisable rather than plausible.
    assert (project / "dataset" / "images" / "cat").exists()
    assert not (project / "dataset" / "prepared.json").exists()


def test_preparing_again_does_not_leave_orphans_from_the_first_attempt(tmp_path, source):
    project = tmp_path / "project"
    prepare_dataset.run(FakeChannel(), job_for(source, project))

    # The user deletes two photos and prepares again. The old cat_000003.jpg
    # has no source and no label; leaving it behind would train on a photo the
    # user believes they removed.
    (source / "cat" / "IMG_1.jpg").unlink()
    (source / "cat" / "IMG_2.jpg").unlink()
    result = prepare_dataset.run(FakeChannel(), job_for(source, project))

    assert result["photos"] == 2
    assert [p.name for p in (project / "dataset" / "images" / "cat").iterdir()] == [
        "cat_000001.jpg"
    ]
