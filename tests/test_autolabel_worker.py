"""Autolabeling, driven against a stand-in for OWLv2.

The model itself is not the interesting part and cannot run here -- no GPU, no
torch, and a 1.3 GB download. Everything around it is interesting and is exactly
where a mistake would be invisible: which photos get sent, what the boxes become
once they are written down, what happens when the card runs out of memory, and
what is left behind when the user stops halfway.

``load_detector`` is therefore replaced wholesale, which is why it is a module
function and not an import at the top of the command.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from worker.channel import Cancelled
from worker.commands import autolabel
from worker.util.errors import DatasetError

REPO_ROOT = Path(__file__).resolve().parents[1]


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
        self.artifacts.append((kind, str(path)))

    def check_cancelled(self):
        self._checks += 1
        if self._cancel_after is not None and self._checks > self._cancel_after:
            raise Cancelled()


class StubDetector:
    """Answers with whatever the test says OWLv2 would have found."""

    def __init__(self, detections=None, *, size=(100, 50), fails_above: int | None = None):
        self.calls: list[tuple[list[Path], list[str]]] = []
        self.size = size
        self._detections = detections
        self._fails_above = fails_above

    def detect(self, paths, prompts, *, threshold):
        if self._fails_above is not None and len(paths) > self._fails_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.calls.append((list(paths), list(prompts)))
        width, height = self.size
        return [
            autolabel.Photo(width, height, tuple(self._for(path, prompts)))
            for path in paths
        ]

    def _for(self, path, prompts):
        if self._detections is None:
            return [autolabel.Detection(score=0.9, box=(10, 10, 30, 30))]
        if callable(self._detections):
            return self._detections(Path(path), list(prompts))
        return list(self._detections)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def build_dataset(root: Path, classes=(("cat", 3), ("dog", 2)), labelled=()) -> Path:
    """A prepared project, as Step 2 would have left it."""
    dataset = root / "dataset"
    for name, count in classes:
        (dataset / "images" / name).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / name).mkdir(parents=True, exist_ok=True)
        for number in range(1, count + 1):
            (dataset / "images" / name / f"{name}_{number:06d}.jpg").write_bytes(b"jpeg")
    (dataset / "classes.txt").write_text(
        "\n".join(name for name, _ in classes) + "\n", encoding="utf-8"
    )
    (dataset / "prepared.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "classes": [{"name": name, "slug": name} for name, _ in classes],
            }
        ),
        encoding="utf-8",
    )
    for relative in labelled:
        path = dataset / "labels" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    return root


def job_for(root: Path, classes=(("cat", "a cat"), ("dog", "a dog")), **extra) -> dict:
    return {
        "project": str(root),
        "classes": [{"name": name, "slug": name, "prompt": prompt} for name, prompt in classes],
        "threshold": 0.2,
        "max_detections": 10,
        **extra,
    }


@pytest.fixture
def project(tmp_path):
    return build_dataset(tmp_path / "project")


@pytest.fixture
def detector(monkeypatch):
    stub = StubDetector()
    monkeypatch.setattr(autolabel, "load_detector", lambda *a, **k: stub)
    return stub


def label_of(project: Path, relative: str) -> str:
    return (project / "dataset" / "labels" / relative).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_every_photo_gets_a_label_file(project, detector):
    channel = FakeChannel()

    result = autolabel.run(channel, job_for(project))

    assert result["labelled"] == 5
    assert result["boxes"] == 5
    # 100x50 photo, box from (10,10) to (30,30): centre (20,20), size 20x20.
    assert label_of(project, "cat/cat_000001.txt") == "0 0.200000 0.400000 0.200000 0.400000\n"
    assert label_of(project, "dog/dog_000002.txt").startswith("1 ")


def test_the_class_index_comes_from_classes_txt(tmp_path, detector):
    # Prepared with dog first, asked for cat first. The number in a label file
    # means whatever classes.txt says it means, and nothing else.
    project = build_dataset(tmp_path / "p", classes=(("dog", 1), ("cat", 1)))

    autolabel.run(FakeChannel(), job_for(project, classes=(("cat", "a cat"),)))

    assert label_of(project, "cat/cat_000001.txt").startswith("1 ")


def test_a_photo_with_no_objects_still_gets_a_label_file(project, monkeypatch):
    monkeypatch.setattr(autolabel, "load_detector", lambda *a, **k: StubDetector(detections=[]))
    channel = FakeChannel()

    result = autolabel.run(channel, job_for(project))

    # An empty file is YOLO's way of saying "no objects here", and it is also
    # what makes the file's existence mean "this photo has been looked at".
    assert label_of(project, "cat/cat_000001.txt") == ""
    assert result["empty"] == 5
    assert "Not a single box" in " ".join(channel.warnings)


def test_a_class_where_almost_nothing_was_found_is_reported(tmp_path, monkeypatch):
    project = build_dataset(tmp_path / "p", classes=(("cat", 4),))
    found = lambda path, prompts: (  # noqa: E731 - a one-line stub
        [autolabel.Detection(0.9, (10, 10, 30, 30))] if path.name.endswith("1.jpg") else []
    )
    monkeypatch.setattr(autolabel, "load_detector", lambda *a, **k: StubDetector(found))

    channel = FakeChannel()
    autolabel.run(channel, job_for(project, classes=(("cat", "a cat"),)))

    # The failure that looks like success: four label files, three of them
    # saying the photo contains nothing.
    assert any("3 of 4" in line for line in channel.warnings), channel.warnings


def test_progress_is_reported_for_every_photo(project, detector):
    channel = FakeChannel()

    autolabel.run(channel, job_for(project))

    labelling = [call for call in channel.progress_calls if call[0] == "label"]
    assert [c[1] for c in labelling] == [1, 2, 3, 4, 5]
    assert {c[2] for c in labelling} == {5}
    assert channel.ready_info["total"] == 5


def test_the_run_is_recorded_for_the_next_session(project, detector):
    autolabel.run(FakeChannel(), job_for(project))

    record = json.loads((project / "dataset" / "autolabel.json").read_text())

    assert record["classes"][0]["prompt"] == "a cat"
    assert record["classes"][0]["boxes"] == 3
    assert record["finished_at"] > 0


# --------------------------------------------------------------------------- #
# Resuming
# --------------------------------------------------------------------------- #


def test_photos_that_already_have_labels_are_skipped(project, detector):
    existing = project / "dataset" / "labels" / "cat" / "cat_000001.txt"
    existing.write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")

    result = autolabel.run(FakeChannel(), job_for(project))

    sent = [path.name for paths, _ in detector.calls for path in paths]
    assert "cat_000001.jpg" not in sent
    assert result["labelled"] == 4
    assert existing.read_text() == "0 0.1 0.1 0.1 0.1\n", "a finished photo was redone"


def test_relabelling_replaces_what_is_there(project, detector):
    existing = project / "dataset" / "labels" / "cat" / "cat_000001.txt"
    existing.write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")

    result = autolabel.run(FakeChannel(), job_for(project, relabel=True))

    assert result["labelled"] == 5
    assert existing.read_text() != "0 0.1 0.1 0.1 0.1\n"


def test_nothing_to_do_returns_without_loading_the_model(project, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the model must not be loaded when there is no work")

    monkeypatch.setattr(autolabel, "load_detector", refuse)
    for name, count in (("cat", 3), ("dog", 2)):
        for number in range(1, count + 1):
            (project / "dataset" / "labels" / name / f"{name}_{number:06d}.txt").write_text("")

    result = autolabel.run(FakeChannel(), job_for(project))

    assert result["skipped"] is True
    assert result["labelled"] == 0


def test_stopping_keeps_the_photos_already_labelled(project, detector):
    # Cancel is checked once per batch; one batch of five here, so the check
    # lands before any photo is written.
    with pytest.raises(Cancelled):
        autolabel.run(FakeChannel(cancel_after=0), job_for(project))

    assert not list((project / "dataset" / "labels" / "cat").glob("*.txt"))
    # The prompts survive, so resuming does not mean retyping them.
    assert json.loads((project / "dataset" / "autolabel.json").read_text())["classes"]


def test_the_prompts_are_saved_before_the_model_loads(project, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("no internet")

    monkeypatch.setattr(autolabel, "load_detector", fail)

    with pytest.raises(RuntimeError):
        autolabel.run(FakeChannel(), job_for(project))

    # The download is the longest part of a first run. Losing eight typed
    # sentences because it failed would be unforgivable.
    record = json.loads((project / "dataset" / "autolabel.json").read_text())
    assert [c["prompt"] for c in record["classes"]] == ["a cat", "a dog"]
    assert "finished_at" not in record


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_a_class_with_work_and_no_prompt_is_refused(project, detector):
    with pytest.raises(DatasetError, match="cat"):
        autolabel.run(FakeChannel(), job_for(project, classes=(("cat", ""), ("dog", "a dog"))))


def test_a_class_the_dataset_does_not_have_is_refused(project, detector):
    with pytest.raises(DatasetError, match="prepare it again"):
        autolabel.run(FakeChannel(), job_for(project, classes=(("bird", "a bird"),)))


def test_an_unprepared_folder_is_refused(tmp_path, detector):
    with pytest.raises(DatasetError, match="Step 2"):
        autolabel.run(FakeChannel(), job_for(tmp_path / "empty"))


def test_a_missing_optional_dependency_names_the_remedy(project, monkeypatch):
    from worker.__main__ import _classify
    from worker.protocol import ErrorCode

    # transformers answers a missing scipy -- which OWLv2's image processor
    # needs and nothing else installs -- with a plain ImportError. It used to
    # reach the user as an uncoded failure with no hint at all.
    code, hint = _classify(ImportError("Owlv2ImageProcessor requires the scipy library"))

    assert code is ErrorCode.VENV_BROKEN
    assert "Repair" in hint


def test_an_impossible_threshold_is_refused(project, detector):
    with pytest.raises(DatasetError, match="between 0 and 1"):
        autolabel.run(FakeChannel(), job_for(project, threshold=1.5))


def test_a_deleted_class_folder_does_not_end_the_run(project, detector):
    import shutil

    shutil.rmtree(project / "dataset" / "images" / "dog")

    result = autolabel.run(FakeChannel(), job_for(project))

    assert result["labelled"] == 3, "the classes that are still there were labelled"


# --------------------------------------------------------------------------- #
# Turning detections into labels
# --------------------------------------------------------------------------- #


def test_boxes_are_clipped_to_the_photo(tmp_path, monkeypatch):
    project = build_dataset(tmp_path / "p", classes=(("cat", 1),))
    # OWLv2 is allowed to predict past the edge, and does: a half-visible
    # object is still an object.
    monkeypatch.setattr(
        autolabel,
        "load_detector",
        lambda *a, **k: StubDetector([autolabel.Detection(0.9, (-40, -10, 60, 40))]),
    )

    autolabel.run(FakeChannel(), job_for(project, classes=(("cat", "a cat"),)))

    _index, cx, cy, w, h = label_of(project, "cat/cat_000001.txt").split()
    assert (float(cx), float(cy)) == (0.3, 0.4)
    assert (float(w), float(h)) == (0.6, 0.8)


def test_the_same_object_found_by_two_wordings_is_kept_once(tmp_path, monkeypatch):
    project = build_dataset(tmp_path / "p", classes=(("van", 1),))
    stub = StubDetector(
        [
            autolabel.Detection(0.8, (10, 10, 30, 30), query=0),
            autolabel.Detection(0.7, (11, 11, 31, 31), query=1),
        ]
    )
    monkeypatch.setattr(autolabel, "load_detector", lambda *a, **k: stub)

    autolabel.run(FakeChannel(), job_for(project, classes=(("van", "a lorry; a truck"),)))

    assert stub.calls[0][1] == ["a lorry", "a truck"], "both wordings are asked for"
    # Two boxes on one van would teach the model that vans come in pairs.
    assert label_of(project, "van/van_000001.txt").count("\n") == 1


def test_only_the_best_boxes_are_kept(tmp_path, monkeypatch):
    project = build_dataset(tmp_path / "p", classes=(("cat", 1),))
    detections = [
        autolabel.Detection(0.9 - index / 10, (index * 20, 0, index * 20 + 10, 10))
        for index in range(5)
    ]
    monkeypatch.setattr(autolabel, "load_detector", lambda *a, **k: StubDetector(detections))

    autolabel.run(
        FakeChannel(), job_for(project, classes=(("cat", "a cat"),), max_detections=2)
    )

    rows = label_of(project, "cat/cat_000001.txt").strip().splitlines()
    assert len(rows) == 2
    assert rows[0].split()[1] == "0.050000", "the highest-scoring box comes first"


def test_boxes_below_the_threshold_are_dropped(tmp_path, monkeypatch):
    project = build_dataset(tmp_path / "p", classes=(("cat", 1),))
    monkeypatch.setattr(
        autolabel,
        "load_detector",
        lambda *a, **k: StubDetector(
            [
                autolabel.Detection(0.9, (10, 10, 30, 30)),
                autolabel.Detection(0.05, (60, 10, 80, 30)),
            ]
        ),
    )

    autolabel.run(FakeChannel(), job_for(project, classes=(("cat", "a cat"),), threshold=0.5))

    assert label_of(project, "cat/cat_000001.txt").count("\n") == 1


@pytest.mark.parametrize(
    "box, expected",
    [
        ((0, 0, 100, 50), (0.5, 0.5, 1.0, 1.0)),
        ((50, 25, 100, 50), (0.75, 0.75, 0.5, 0.5)),
        ((100, 25, 50, 0), (0.75, 0.25, 0.5, 0.5)),  # corners the wrong way round
        ((10, 10, 11, 40), None),  # a sliver, always an artefact of clipping
        ((200, 10, 300, 40), None),  # entirely off the photo
    ],
)
def test_a_box_becomes_a_centre_and_a_size(box, expected):
    assert autolabel._to_yolo(box, 100, 50) == expected


# --------------------------------------------------------------------------- #
# Running out of VRAM
# --------------------------------------------------------------------------- #


def test_running_out_of_memory_halves_the_batch_instead_of_ending_the_run(tmp_path, monkeypatch):
    project = build_dataset(tmp_path / "p", classes=(("cat", 8),))
    stub = StubDetector(fails_above=2)
    monkeypatch.setattr(autolabel, "load_detector", lambda *a, **k: stub)
    channel = FakeChannel()

    result = autolabel.run(
        channel, job_for(project, classes=(("cat", "a cat"),), batch=8)
    )

    # Another application taking VRAM forty minutes into a run must not cost
    # the run.
    assert result["labelled"] == 8
    assert max(len(paths) for paths, _ in stub.calls) <= 2
    assert any("out of GPU memory" in msg for _level, msg in channel.logs)


def test_the_retry_happens_with_the_failed_batch_already_released(tmp_path, monkeypatch):
    """The halved batch must not run inside the ``except`` block.

    While that block is open the exception holds its traceback, the traceback
    holds the frames of the pass that failed, and those frames hold the
    activations that did not fit. Retrying in there retries on a card that is
    still full, so every halving fails in turn and a recoverable hiccup ends
    the run -- which is precisely what "large crashes immediately with OOM"
    was. ``sys.exc_info()`` is how a running call can tell where it was called
    from.
    """
    seen: list[type | None] = []

    class WatchfulDetector(StubDetector):
        def detect(self, paths, prompts, *, threshold):
            seen.append(sys.exc_info()[0])
            return super().detect(paths, prompts, threshold=threshold)

    project = build_dataset(tmp_path / "p", classes=(("cat", 4),))
    monkeypatch.setattr(
        autolabel, "load_detector", lambda *a, **k: WatchfulDetector(fails_above=2)
    )

    autolabel.run(FakeChannel(), job_for(project, classes=(("cat", "a cat"),), batch=4))

    assert seen, "the detector was never called"
    assert all(kind is None for kind in seen), (
        f"a retry ran while an exception was still being handled: {seen}"
    )


def test_the_batch_is_measured_on_this_card_rather_than_assumed(tmp_path, monkeypatch):
    """Eight photos was a guess, and it was wrong in both directions."""
    asked: list[dict] = []

    class MeasuringDetector(StubDetector):
        def suggest_batch(self, path, prompts, *, threshold, ceiling):
            asked.append({"path": Path(path), "prompts": list(prompts), "ceiling": ceiling})
            return 3, "1.7 GB of GPU memory per photo, 10.0 GB free: labelling 3 at a time."

    stub = MeasuringDetector()
    monkeypatch.setattr(autolabel, "load_detector", lambda *a, **k: stub)
    project = build_dataset(tmp_path / "p", classes=(("cat", 7),))
    channel = FakeChannel()

    autolabel.run(channel, job_for(project, classes=(("cat", "a cat"),)))

    assert [call["ceiling"] for call in asked] == [autolabel.MAX_BATCH]
    # Measured on a photo this run is about to label, with the prompt it will
    # be labelled with: the cost depends on both.
    assert asked[0]["path"].name == "cat_000001.jpg"
    assert asked[0]["prompts"] == ["a cat"]
    assert [len(paths) for paths, _ in stub.calls] == [3, 3, 1]
    assert any("1.7 GB of GPU memory per photo" in msg for _level, msg in channel.logs)


def test_a_job_that_fixes_the_batch_is_not_second_guessed(tmp_path, monkeypatch):
    class MeasuringDetector(StubDetector):
        def suggest_batch(self, path, prompts, *, threshold, ceiling):  # pragma: no cover
            raise AssertionError("the job said 2")

    stub = MeasuringDetector()
    monkeypatch.setattr(autolabel, "load_detector", lambda *a, **k: stub)
    project = build_dataset(tmp_path / "p", classes=(("cat", 4),))

    autolabel.run(FakeChannel(), job_for(project, classes=(("cat", "a cat"),), batch=2))

    assert [len(paths) for paths, _ in stub.calls] == [2, 2]


def test_one_lesson_is_learned_once_for_the_whole_run(tmp_path, monkeypatch):
    """A size that just failed is not tried again on the next photos.

    Halving splits a batch in two, and the second half is the same size as the
    first -- so without remembering, every half re-discovers the same limit and
    the user reads the same warning over and over.
    """
    attempts: list[int] = []

    class CountingDetector(StubDetector):
        def detect(self, paths, prompts, *, threshold):
            attempts.append(len(paths))
            return super().detect(paths, prompts, threshold=threshold)

    project = build_dataset(tmp_path / "p", classes=(("cat", 8),))
    monkeypatch.setattr(
        autolabel, "load_detector", lambda *a, **k: CountingDetector(fails_above=2)
    )
    channel = FakeChannel()

    autolabel.run(channel, job_for(project, classes=(("cat", "a cat"),), batch=8))

    assert [n for n in attempts if n > 2] == [8, 4], attempts
    assert sum("out of GPU memory" in msg for _level, msg in channel.logs) == 2


def test_a_card_too_small_for_one_photo_still_fails(tmp_path, monkeypatch):
    project = build_dataset(tmp_path / "p", classes=(("cat", 2),))
    monkeypatch.setattr(
        autolabel, "load_detector", lambda *a, **k: StubDetector(fails_above=0)
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        autolabel.run(FakeChannel(), job_for(project, classes=(("cat", "a cat"),)))


# --------------------------------------------------------------------------- #
# The command as a command
# --------------------------------------------------------------------------- #


def test_it_runs_standalone_from_a_shell(tmp_path):
    """Every worker command is runnable without a GUI. This one included.

    A fully-labelled dataset is used so the model is never loaded -- which is
    what lets this run on a machine with no GPU and no managed environment, and
    still exercise the real argument parsing, the real channel and the real
    result frame.
    """
    project = build_dataset(
        tmp_path / "p",
        classes=(("cat", 2),),
        labelled=("cat/cat_000001.txt", "cat/cat_000002.txt"),
    )
    job = json.dumps(job_for(project, classes=(("cat", "a cat"),)))

    completed = subprocess.run(
        [sys.executable, "-m", "worker", "autolabel", "--job", job],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin", "PYTHONUNBUFFERED": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    results = [
        json.loads(line.lstrip("\x1e"))
        for line in completed.stdout.splitlines()
        if '"type":"result"' in line
    ]
    assert results and results[0]["data"]["ok"] is True
    assert results[0]["data"]["payload"]["skipped"] is True
