"""The photo folder is described, not judged.

These tests pin the distinction the page depends on: an error means preparation
cannot produce a usable dataset, a warning means it can but the user should know
something first. Getting that boundary wrong either blocks a perfectly good
folder or lets a broken one through.
"""

from __future__ import annotations

import pytest

from yolo_assistant.app.core import dataset


def make_tree(root, layout: dict[str, list[str]]):
    for class_name, files in layout.items():
        directory = root / class_name
        directory.mkdir(parents=True, exist_ok=True)
        for name in files:
            (directory / name).write_bytes(b"not really an image, but named like one")
    return root


def test_a_tidy_folder_scans_clean(tmp_path):
    make_tree(tmp_path, {"cat": ["a.jpg", "b.JPG"], "dog": ["c.png"]})

    scan = dataset.scan(tmp_path)

    assert scan.is_usable
    assert [c.name for c in scan.classes] == ["cat", "dog"]
    assert scan.total_images == 3
    assert scan.errors == ()
    # Three photos is fewer than THIN_CLASS, so the only warning is about size.
    assert all("fewer than" in p.text for p in scan.warnings)


def test_extensions_are_matched_regardless_of_case(tmp_path):
    make_tree(tmp_path, {"cat": ["a.JPEG", "b.Png", "c.TIFF"]})
    assert dataset.scan(tmp_path).total_images == 3


def test_a_folder_with_no_subfolders_is_an_error(tmp_path):
    (tmp_path / "loose.jpg").write_bytes(b"x")

    scan = dataset.scan(tmp_path)

    assert not scan.is_usable
    assert scan.errors
    assert "one folder per class" in scan.errors[0].text


def test_a_missing_folder_is_an_error_not_a_crash(tmp_path):
    scan = dataset.scan(tmp_path / "nowhere")
    assert not scan.is_usable
    assert "does not exist" in scan.errors[0].text


def test_classes_that_would_share_a_folder_are_an_error(tmp_path):
    make_tree(tmp_path, {"Cat": ["a.jpg"], "cat": ["b.jpg"]})

    scan = dataset.scan(tmp_path)

    assert not scan.is_usable
    assert "overwrite each other" in scan.errors[0].text


def test_an_empty_class_is_dropped_with_a_warning(tmp_path):
    make_tree(tmp_path, {"cat": ["a.jpg"], "ghost": []})

    scan = dataset.scan(tmp_path)

    assert scan.is_usable
    assert [c.name for c in scan.classes] == ["cat"], "a class with no photos is not prepared"
    assert any('"ghost"' in p.text for p in scan.warnings)


def test_classes_that_are_all_empty_are_an_error(tmp_path):
    make_tree(tmp_path, {"cat": [], "dog": []})

    scan = dataset.scan(tmp_path)

    assert not scan.is_usable
    assert "None of the sub-folders contain photos" in scan.errors[0].text


def test_non_photos_and_nested_folders_are_counted_not_taken(tmp_path):
    make_tree(tmp_path, {"cat": ["a.jpg", "notes.txt", "Thumbs.db"]})
    (tmp_path / "cat" / "rejects").mkdir()
    (tmp_path / "cat" / "rejects" / "b.jpg").write_bytes(b"x")

    scan = dataset.scan(tmp_path)

    assert scan.total_images == 1, "photos in a nested folder are not class photos"
    assert scan.classes[0].ignored_files == 2
    assert scan.classes[0].ignored_dirs == 1
    assert any("not photos" in p.text for p in scan.warnings)
    assert any("sub-folder(s) inside" in p.text for p in scan.warnings)


def test_hidden_folders_are_skipped(tmp_path):
    make_tree(tmp_path, {"cat": ["a.jpg"], ".git": ["b.jpg"]})
    assert [c.name for c in dataset.scan(tmp_path).classes] == ["cat"]


def test_photos_are_listed_in_a_reproducible_order(tmp_path):
    make_tree(tmp_path, {"cat": ["b.jpg", "A.jpg", "c.jpg"]})

    names = [p.name for p in dataset.scan(tmp_path).classes[0].images]

    # Case-folded, so two runs on a case-insensitive and a case-sensitive
    # filesystem assign the same numbers to the same photos.
    assert names == ["A.jpg", "b.jpg", "c.jpg"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("cat", "cat"),
        ("red apple", "red apple"),
        ("a/b", "a_b"),
        ("what?", "what_"),
        ("trailing.", "trailing"),
        (" padded ", "padded"),
        ("aux", "aux_"),
        ("COM1", "COM1_"),
        ("...", "class"),
        ("кот", "кот"),
    ],
)
def test_slugify(name, expected):
    assert dataset.slugify(name) == expected


def test_a_renamed_class_keeps_its_original_name(tmp_path):
    # Legal on Linux, illegal on Windows: the class has to be storable on both.
    make_tree(tmp_path, {"50% off?": ["a.jpg"]})

    cls = dataset.scan(tmp_path).classes[0]

    # The name is what the user sees and what goes in classes.txt; the slug is
    # only ever a file name.
    assert cls.name == "50% off?"
    assert cls.slug == "50% off_"


def test_renaming_is_reported(tmp_path):
    make_tree(tmp_path, {"aux": ["a.jpg"]})

    scan = dataset.scan(tmp_path)

    assert scan.classes[0].name == "aux"
    assert scan.classes[0].slug == "aux_"
    assert any("safe spelling" in p.text for p in scan.warnings)


def test_job_classes_carries_the_source_folder_name(tmp_path):
    make_tree(tmp_path, {"aux": ["a.jpg"]})

    entry = dataset.scan(tmp_path).job_classes()[0]

    # The worker has to find the folder again, and it is not called "aux_".
    assert entry == {"name": "aux", "slug": "aux_", "dir": "aux"}
