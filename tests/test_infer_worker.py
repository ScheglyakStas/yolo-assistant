"""Step 7's worker, driven against stand-ins for the model and the camera.

Nothing here runs a model: there is no torch, no onnxruntime and no camera on a
test machine, and none of them is the interesting part. The interesting parts
are what the command checks before it loads anything, what it does with the
frames it is given, which timings it counts and which it refuses to count, and
what stopping a live feed halfway is reported as.

Every door to the outside world is a module function for that reason --
``load_yolo``, ``read_photo``, ``open_video``, ``open_camera``, ``open_writer``,
``encode_frame`` -- and each is replaced wholesale.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from worker.channel import Cancelled
from worker.commands import infer
from worker.util.errors import (
    CameraUnavailable,
    GpuUnavailable,
    InferenceFailed,
    SourceUnreadable,
)

from test_train_worker import FakeChannel


class Channel(FakeChannel):
    """``FakeChannel`` plus the one message only this step sends."""

    def __init__(self, cancel_after: int | None = None, *, preview_every: bool = True) -> None:
        super().__init__(cancel_after=cancel_after)
        self.previews: list[dict] = []
        self.due_calls = 0
        self._preview_every = preview_every

    def preview_due(self) -> bool:
        """The real channel decides this from a clock; here it is scripted."""
        self.due_calls += 1
        return self._preview_every

    def preview(
        self,
        index,
        image,
        *,
        width,
        height,
        mime="image/jpeg",
        detections=None,
        timings=None,
        stats=None,
    ) -> None:
        self.previews.append(
            {
                "index": index,
                "image": image,
                "width": width,
                "height": height,
                "detections": list(detections or ()),
                "timings": dict(timings or {}),
                "stats": dict(stats or {}),
            }
        )
        self.events.append(("preview", index))

    def warning(self, code, msg, *, hint=""):  # the base class drops the hint
        self.warnings.append((code, msg))


# --------------------------------------------------------------------------- #
# Stand-ins
# --------------------------------------------------------------------------- #


class StubBoxes:
    def __init__(self, classes, confidences) -> None:
        self.cls = list(classes)
        self.conf = list(confidences)


class StubResult:
    def __init__(self, *, inference_ms=8.0, detections=((0, 0.9),), names=None) -> None:
        self.speed = {"preprocess": 1.0, "inference": inference_ms, "postprocess": 0.5}
        self.boxes = StubBoxes([c for c, _ in detections], [s for _, s in detections])
        self.names = names if names is not None else {0: "cat", 1: "dog"}
        self.plotted = 0

    def plot(self):
        self.plotted += 1
        return "annotated-frame"


class StubYOLO:
    """Records every call, and hands back whatever the test lined up."""

    def __init__(self, results=None, raises: Exception | None = None) -> None:
        self._results = list(results or [])
        self._raises = raises
        self.calls: list[dict] = []

    def predict(self, frame, **kwargs):
        self.calls.append({"frame": frame, **kwargs})
        if self._raises is not None:
            raise self._raises
        if self._results:
            return [self._results.pop(0)]
        return [StubResult()]


class StubCapture:
    """The handful of OpenCV capture calls the command makes."""

    def __init__(
        self, frames=(), *, total: int = 0, props=None, opened: bool = True, max_size=None
    ) -> None:
        self._frames = list(frames)
        self._opened = opened
        self.released = False
        self.settings: list[tuple[int, object]] = []
        #: What this camera can actually do. A driver clamps a request it
        #: cannot meet rather than refusing it, which is the behaviour the
        #: command has to read back instead of assume.
        self.max_size = max_size
        self.props = {PROPS["frame_count"]: total}
        self.props.update(props or {})

    def isOpened(self):  # noqa: N802 - OpenCV naming
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def get(self, prop):
        return self.props.get(prop, 0)

    def set(self, prop, value):
        self.settings.append((prop, value))
        if self.max_size and prop == PROPS["width"]:
            value = min(value, self.max_size[0])
        elif self.max_size and prop == PROPS["height"]:
            value = min(value, self.max_size[1])
        self.props[prop] = value
        return True

    def release(self):
        self.released = True


class StubWriter:
    """An OpenCV video writer that keeps the frames instead of encoding them.

    It does create its file, empty, because what happens to that file -- renamed
    into place, or removed -- is part of what is under test.
    """

    opened: list["StubWriter"] = []

    def __init__(self, path, *, fps, first) -> None:
        self.path = path
        self.fps = fps
        self.frames: list = []
        self.released = False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        StubWriter.opened.append(self)

    def write(self, image) -> None:
        self.frames.append(image)

    def release(self) -> None:
        self.released = True


#: The property ids the stub cv2 hands out, so a test can ask what was set.
PROPS = {
    "frame_count": 7,
    "width": 3,
    "height": 4,
    "fps": 5,
    "fourcc": 6,
    "buffersize": 38,
}


@pytest.fixture(autouse=True)
def stubs(monkeypatch):
    """The four doors to the outside, closed. Individual tests reopen one."""
    monkeypatch.setattr(infer, "read_photo", lambda path: f"photo:{path.name}")
    monkeypatch.setattr(infer, "encode_frame", lambda image, **kw: ("BASE64", 640, 480))
    monkeypatch.setattr(infer, "open_writer", StubWriter)
    StubWriter.opened = []
    monkeypatch.setitem(sys.modules, "cv2", stub_cv2())


def stub_cv2():
    cv2 = types.ModuleType("cv2")
    cv2.CAP_PROP_FRAME_COUNT = PROPS["frame_count"]
    cv2.CAP_PROP_FRAME_WIDTH = PROPS["width"]
    cv2.CAP_PROP_FRAME_HEIGHT = PROPS["height"]
    cv2.CAP_PROP_FPS = PROPS["fps"]
    cv2.CAP_PROP_FOURCC = PROPS["fourcc"]
    cv2.CAP_PROP_BUFFERSIZE = PROPS["buffersize"]
    cv2.CAP_DSHOW = 700
    cv2.CAP_V4L2 = 200
    cv2.VideoWriter_fourcc = lambda *chars: sum(
        ord(char) << (8 * shift) for shift, char in enumerate(chars)
    )
    return cv2


@pytest.fixture
def gpu(monkeypatch):
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


@pytest.fixture
def no_gpu(monkeypatch):
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


@pytest.fixture
def model(monkeypatch):
    stub = StubYOLO()
    monkeypatch.setattr(infer, "load_yolo", lambda *a, **k: stub)
    return stub


@pytest.fixture
def exported(tmp_path):
    path = tmp_path / "exports" / "bottles_best_fp32.onnx"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"model")
    return path


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "shelf.jpg"
    path.write_bytes(b"photo")
    return path


def job_for(exported, **extra) -> dict:
    job = {
        "model": str(exported),
        "format": "onnx",
        "device": "cpu",
        "mode": "photo",
        "conf": 0.25,
        "names": ["cat", "dog"],
        "needs": [],
        "schema": 1,
    }
    job.update(extra)
    return job


def run(job, *, channel=None) -> tuple[dict, Channel]:
    channel = channel or Channel()
    return infer.run(channel, job), channel


# --------------------------------------------------------------------------- #
# Before anything is loaded
# --------------------------------------------------------------------------- #


def test_a_deleted_model_is_reported_before_torch_is_imported(tmp_path):
    """No ``torch`` fixture here on purpose: it must not get that far."""
    with pytest.raises(InferenceFailed, match="not there any more"):
        run(job_for(tmp_path / "gone.onnx", source=str(tmp_path)))


def test_a_deleted_source_is_reported_before_torch_is_imported(exported, tmp_path):
    with pytest.raises(SourceUnreadable, match="is not there"):
        run(job_for(exported, source=str(tmp_path / "gone.jpg")))


def test_no_source_at_all_says_which_kind_is_missing(exported):
    with pytest.raises(SourceUnreadable, match="No photo was chosen"):
        run(job_for(exported, source=""))


def test_a_missing_runtime_names_the_package(exported, photo):
    with pytest.raises(InferenceFailed, match="no_such_runtime"):
        run(job_for(exported, source=str(photo), needs=["no_such_runtime"], format="tflite"))


def test_an_unknown_mode_is_refused(exported, photo):
    with pytest.raises(InferenceFailed, match="not something this app can test"):
        run(job_for(exported, source=str(photo), mode="telepathy"))


# --------------------------------------------------------------------------- #
# Which device
# --------------------------------------------------------------------------- #


def test_an_engine_has_no_processor_version(exported, photo):
    with pytest.raises(InferenceFailed, match="no processor version"):
        run(job_for(exported, source=str(photo), format="engine", device="cpu"))


def test_the_gpu_being_absent_is_an_error_not_a_fallback(exported, photo, no_gpu, model):
    """Substituting the CPU would answer the user's question with another one."""
    with pytest.raises(GpuUnavailable, match="cannot see one"):
        run(job_for(exported, source=str(photo), device="gpu"))


def test_the_gpu_is_passed_to_the_model(exported, photo, gpu, model):
    run(job_for(exported, source=str(photo), device="gpu"))

    assert model.calls[0]["device"] == 0


def test_the_processor_is_passed_to_the_model(exported, photo, model):
    run(job_for(exported, source=str(photo), device="cpu"))

    assert model.calls[0]["device"] == "cpu"


def test_onnx_on_the_gpu_without_a_cuda_provider_says_so(
    exported, photo, gpu, model, monkeypatch
):
    """The quiet failure: onnxruntime uses the processor and reports success."""
    runtime = types.ModuleType("onnxruntime")
    runtime.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)

    _payload, channel = run(job_for(exported, source=str(photo), device="gpu"))

    assert any("processor" in msg for _code, msg in channel.warnings)
    # The page keys its footnote on the code, not on the (translated) wording.
    from yolo_assistant.app.core.inference import ONNX_ON_CPU_WARNING

    assert ONNX_ON_CPU_WARNING in {code for code, _msg in channel.warnings}


def test_a_cuda_provider_is_not_worth_mentioning(exported, photo, gpu, model, monkeypatch):
    runtime = types.ModuleType("onnxruntime")
    runtime.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)

    _payload, channel = run(job_for(exported, source=str(photo), device="gpu"))

    assert not channel.warnings


# --------------------------------------------------------------------------- #
# A photo
# --------------------------------------------------------------------------- #


def test_a_photo_is_run_twice_and_timed_once(exported, photo, model):
    """The first call pays for the session; counting it would be a lie."""
    payload, _channel = run(job_for(exported, source=str(photo)))

    assert len(model.calls) == 2
    assert payload["frames"] == 1
    assert payload["startup_ms"] >= 0


def test_a_photo_writes_nothing_unless_a_copy_is_asked_for(exported, photo, model, tmp_path):
    before = sorted(tmp_path.rglob("*"))

    payload, _channel = run(job_for(exported, source=str(photo)))

    assert payload["output"] == ""
    assert sorted(tmp_path.rglob("*")) == before


def test_a_photo_copy_is_written_full_size_where_the_page_said(
    exported, photo, model, monkeypatch, tmp_path
):
    written = {}

    def fake_write(path, image):
        written[path.name] = image
        path.write_bytes(b"annotated")

    monkeypatch.setattr(infer, "write_photo", fake_write)
    output = tmp_path / "temp" / "shelf__bottles.jpg"

    payload, _channel = run(job_for(exported, source=str(photo), output=str(output)))

    assert payload["output"] == str(output)
    assert output.read_bytes() == b"annotated"
    # Drawn by Ultralytics, not the shrunk preview, and via a partial file.
    assert written == {"shelf__bottles.partial.jpg": "annotated-frame"}
    assert not list(output.parent.glob("*.partial*"))


def test_a_photo_copy_that_cannot_be_written_leaves_nothing(
    exported, photo, model, monkeypatch, tmp_path
):
    def broken_write(path, image):
        path.write_bytes(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(infer, "write_photo", broken_write)
    output = tmp_path / "temp" / "shelf__bottles.jpg"

    with pytest.raises(InferenceFailed, match="disk full"):
        run(job_for(exported, source=str(photo), output=str(output)))

    assert not list(output.parent.glob("*"))


def test_a_photo_sends_exactly_one_frame_to_look_at(exported, photo, model):
    _payload, channel = run(job_for(exported, source=str(photo)))

    assert len(channel.previews) == 1
    assert channel.previews[0]["image"] == "BASE64"


def test_the_frame_carries_this_frames_timings(exported, photo, monkeypatch):
    monkeypatch.setattr(
        infer, "load_yolo", lambda *a, **k: StubYOLO([StubResult(), StubResult(inference_ms=12.5)])
    )
    _payload, channel = run(job_for(exported, source=str(photo)))

    assert channel.previews[0]["timings"]["inference_ms"] == 12.5
    assert channel.previews[0]["timings"]["total_ms"] == 14.0


def test_the_detections_are_named_and_strongest_first(exported, photo, monkeypatch):
    result = StubResult(detections=((1, 0.4), (0, 0.95)))
    monkeypatch.setattr(infer, "load_yolo", lambda *a, **k: StubYOLO([StubResult(), result]))

    _payload, channel = run(job_for(exported, source=str(photo)))

    found = channel.previews[0]["detections"]
    assert [item["name"] for item in found] == ["cat", "dog"]
    assert found[0]["conf"] == 0.95


def test_a_photo_with_nothing_in_it_is_reported_rather_than_buried(
    exported, photo, monkeypatch
):
    empty = StubResult(detections=())
    monkeypatch.setattr(infer, "load_yolo", lambda *a, **k: StubYOLO([StubResult(), empty]))

    _payload, channel = run(job_for(exported, source=str(photo)))

    assert any("Nothing was found" in msg for _code, msg in channel.warnings)


def test_the_confidence_reaches_the_model(exported, photo, model):
    run(job_for(exported, source=str(photo), conf=0.4))

    assert model.calls[0]["conf"] == 0.4


def test_a_nonsense_confidence_is_clamped_rather_than_fatal(exported, photo, model):
    run(job_for(exported, source=str(photo), conf="soon"))

    assert model.calls[0]["conf"] == 0.25


def test_the_image_size_is_never_passed(exported, photo, model):
    """An exported model's input size is fixed; asking for another one breaks it."""
    run(job_for(exported, source=str(photo)))

    assert "imgsz" not in model.calls[0]


# --------------------------------------------------------------------------- #
# A video
# --------------------------------------------------------------------------- #


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "aisle.mp4"
    path.write_bytes(b"video")
    return path


def play(monkeypatch, frames, *, total=0, fps=0):
    capture = StubCapture(frames, total=total, props={PROPS["fps"]: fps})
    monkeypatch.setattr(infer, "open_video", lambda path: capture)
    monkeypatch.setattr(infer, "open_camera", lambda index, **_kw: capture)
    return capture


def test_every_frame_of_a_video_is_run(exported, video, model, monkeypatch):
    play(monkeypatch, ["a", "b", "c"], total=3)

    payload, _channel = run(job_for(exported, source=str(video), mode="video"))

    assert payload["frames"] == 3
    assert payload["stopped"] is False
    # Three frames plus the one discarded to wake the runtime.
    assert len(model.calls) == 4


def test_the_video_is_always_released(exported, video, model, monkeypatch):
    capture = play(monkeypatch, ["a"], total=1)

    run(job_for(exported, source=str(video), mode="video"))

    assert capture.released


def test_progress_counts_frames_against_the_total(exported, video, model, monkeypatch):
    play(monkeypatch, ["a", "b"], total=2)

    _payload, channel = run(job_for(exported, source=str(video), mode="video"))

    assert channel.progress_calls[-1]["total"] == 2
    assert channel.progress_calls[-1]["current"] == 2


def test_an_implausible_frame_count_is_no_total_at_all(exported, video, model, monkeypatch):
    """Some containers store a duration the header got wrong."""
    play(monkeypatch, ["a"], total=infer.MAX_PLAUSIBLE_FRAMES + 1)

    _payload, channel = run(job_for(exported, source=str(video), mode="video"))

    assert channel.progress_calls[-1]["total"] is None


def test_the_last_frame_is_always_shown(exported, video, model, monkeypatch):
    """Whatever the preview throttle was doing when the stream ended."""
    play(monkeypatch, ["a", "b", "c"], total=3)

    _payload, channel = run(job_for(exported, source=str(video), mode="video"))

    assert channel.previews, "the run ended without ever showing a frame"
    assert channel.previews[-1]["index"] == 2


def test_stopping_a_video_keeps_what_it_measured(exported, video, model, monkeypatch):
    """Cancelling is not a failure here: the summary is the whole result."""
    play(monkeypatch, ["a", "b", "c", "d", "e"], total=5)
    channel = Channel(cancel_after=2)

    payload, _channel = run(job_for(exported, source=str(video), mode="video"), channel=channel)

    assert payload["stopped"] is True
    assert 0 < payload["frames"] < 5
    assert payload["inference_ms_mean"] > 0


def test_frames_are_only_drawn_when_one_would_be_shown(exported, video, model, monkeypatch):
    """Drawing and compressing a frame nobody sees is time taken off the clock."""
    play(monkeypatch, ["a", "b", "c", "d"], total=4)
    channel = Channel(preview_every=False)

    payload, _channel = run(job_for(exported, source=str(video), mode="video"), channel=channel)

    assert payload["frames"] == 4
    assert channel.due_calls == 4
    # Only the closing frame, which is sent whatever the throttle says.
    assert len(channel.previews) == 1


def test_a_video_that_reads_nothing_says_so(exported, video, model, monkeypatch):
    play(monkeypatch, [], total=0)

    payload, channel = run(job_for(exported, source=str(video), mode="video"))

    assert payload["frames"] == 0
    assert "No frames" in channel.text
    assert payload["output"] == ""


def test_every_frame_of_a_video_is_recorded_with_its_boxes(
    exported, video, model, monkeypatch, tmp_path
):
    """The copy is what gets played, so it has every frame, not the previewed ones."""
    play(monkeypatch, ["a", "b", "c"], total=3, fps=25)
    output = tmp_path / "temp" / "aisle__bottles.mp4"
    channel = Channel(preview_every=False)

    payload, _channel = run(
        job_for(exported, source=str(video), mode="video", output=str(output)),
        channel=channel,
    )

    (writer,) = StubWriter.opened
    assert writer.frames == ["annotated-frame"] * 3
    assert writer.released
    assert payload["output"] == str(output)
    assert output.is_file()
    assert not list(output.parent.glob("*.partial*"))


def test_the_copy_plays_at_the_source_frame_rate(exported, video, model, monkeypatch):
    """However fast the frames were run, the copy is at the speed it was filmed."""
    play(monkeypatch, ["a"], total=1, fps=25)

    payload, _channel = run(job_for(exported, source=str(video), mode="video"))

    assert StubWriter.opened[0].fps == 25.0
    assert payload["output_fps"] == 25.0


@pytest.mark.parametrize("fps", [0, 90_000])
def test_a_frame_rate_the_file_gets_wrong_is_replaced(exported, video, model, monkeypatch, fps):
    play(monkeypatch, ["a"], total=1, fps=fps)

    payload, _channel = run(job_for(exported, source=str(video), mode="video"))

    assert payload["output_fps"] == infer.DEFAULT_FPS


def test_without_an_output_path_the_copy_goes_beside_the_video(
    exported, video, model, monkeypatch
):
    play(monkeypatch, ["a"], total=1)

    payload, _channel = run(job_for(exported, source=str(video), mode="video"))

    assert payload["output"] == str(video.with_name("aisle_detections.mp4"))


def test_a_stopped_video_keeps_the_frames_it_reached(exported, video, model, monkeypatch):
    play(monkeypatch, ["a", "b", "c", "d", "e"], total=5)
    channel = Channel(cancel_after=2)

    payload, _channel = run(job_for(exported, source=str(video), mode="video"), channel=channel)

    assert payload["stopped"] is True
    assert len(StubWriter.opened[0].frames) == payload["frames"]
    assert Path(payload["output"]).is_file()


def test_a_run_that_fails_leaves_no_half_written_copy(exported, video, monkeypatch, tmp_path):
    """Nothing that looks like a finished video, and no partial file either."""
    broken = StubResult()
    broken.plot = lambda: (_ for _ in ()).throw(RuntimeError("no"))
    monkeypatch.setattr(infer, "load_yolo", lambda *a, **k: StubYOLO([StubResult(), StubResult(), broken]))
    play(monkeypatch, ["a", "b", "c"], total=3)
    output = tmp_path / "out" / "aisle.mp4"

    with pytest.raises(InferenceFailed):
        run(job_for(exported, source=str(video), mode="video", output=str(output)))

    assert StubWriter.opened[0].released
    assert not output.exists()
    assert not list(output.parent.glob("*"))


def test_a_camera_records_nothing(exported, model, monkeypatch):
    play(monkeypatch, ["a"] * 10)
    channel = Channel(cancel_after=3)

    payload, _channel = run(job_for(exported, mode="camera", camera=0), channel=channel)

    assert StubWriter.opened == []
    assert "output" not in payload


# --------------------------------------------------------------------------- #
# A camera
# --------------------------------------------------------------------------- #


def test_a_camera_runs_until_it_is_stopped(exported, model, monkeypatch):
    play(monkeypatch, ["a"] * 50)
    channel = Channel(cancel_after=4)

    payload, _channel = run(job_for(exported, mode="camera", camera=1), channel=channel)

    assert payload["stopped"] is True
    assert payload["camera"] == 1
    assert payload["frames"] < 50


def test_the_camera_is_released_even_when_stopped(exported, model, monkeypatch):
    capture = play(monkeypatch, ["a"] * 50)
    channel = Channel(cancel_after=2)

    run(job_for(exported, mode="camera"), channel=channel)

    assert capture.released


# --------------------------------------------------------------------------- #
# How a camera is opened
# --------------------------------------------------------------------------- #


@pytest.fixture
def camera(monkeypatch):
    """A stubbed OpenCV whose capture records everything asked of it."""
    cv2 = stub_cv2()
    capture = StubCapture(["a"] * 10)

    def video_capture(index, backend=0):
        capture.index, capture.backend = index, backend
        return capture

    cv2.VideoCapture = video_capture
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return capture


def test_a_camera_is_asked_for_a_usable_mode(camera):
    infer.open_camera(0)

    asked = dict(camera.settings)
    assert asked[PROPS["width"]] == infer.CAMERA_TARGET_SIZE[0]
    assert asked[PROPS["height"]] == infer.CAMERA_TARGET_SIZE[1]
    assert asked[PROPS["buffersize"]] == infer.CAMERA_BUFFER_FRAMES


def test_on_v4l2_the_format_is_asked_for_before_the_size(camera, monkeypatch):
    """Order is the whole fix: 1080p in raw YUYV measured 3 frames a second."""
    monkeypatch.setattr(infer, "capture_backend", lambda cv2: cv2.CAP_V4L2)
    infer.open_camera(0, fps=30)

    order = [prop for prop, _value in camera.settings]
    assert order.index(PROPS["fourcc"]) < order.index(PROPS["width"])
    assert order.index(PROPS["fourcc"]) < order.index(PROPS["fps"])


def test_on_directshow_the_format_is_asked_for_after_the_size_and_rate(camera, monkeypatch):
    """DirectShow forgets a format at each size change: asked first, 3 fps; last, 30."""
    monkeypatch.setattr(infer, "capture_backend", lambda cv2: cv2.CAP_DSHOW)
    infer.open_camera(0, fps=30)

    order = [prop for prop, _value in camera.settings]
    assert camera.backend == 700
    assert order.index(PROPS["fourcc"]) > order.index(PROPS["height"])
    assert order.index(PROPS["fourcc"]) > order.index(PROPS["fps"])


def test_a_chosen_size_and_rate_are_what_is_asked_for(camera):
    infer.open_camera(0, size=(1280, 720), fps=15)

    asked = dict(camera.settings)
    assert (asked[PROPS["width"]], asked[PROPS["height"]], asked[PROPS["fps"]]) == (1280, 720, 15)


def test_no_rate_is_asked_for_unless_one_was_chosen(camera):
    infer.open_camera(0)

    assert PROPS["fps"] not in dict(camera.settings)


def test_an_unknown_frame_rate_is_not_minus_one(camera):
    """DirectShow reports -1 for a rate nobody asked for, which once read "-1fps"."""
    camera.props[PROPS["fps"]] = -1.0

    assert infer.granted_mode(infer.open_camera(0))["fps"] == 0


def test_a_reported_rate_is_rounded_rather_than_truncated(camera):
    camera.props[PROPS["fps"]] = 29.97

    assert infer.granted_mode(infer.open_camera(0))["fps"] == 30


def test_the_scan_does_not_list_minus_one_fps(camera):
    from worker.commands import cameras

    camera.props[PROPS["fps"]] = -1.0

    assert cameras._fps_of(camera) == 0


def test_a_camera_that_will_not_open_is_not_used(monkeypatch):
    cv2 = stub_cv2()
    cv2.VideoCapture = lambda index, backend=0: StubCapture(opened=False)
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    with pytest.raises(CameraUnavailable):
        infer.open_camera(0)


def test_what_the_camera_granted_is_read_back_not_assumed(camera):
    """Asked for 1080p, this camera has 720p, and the app must say 720p."""
    camera.max_size = (1280, 720)
    camera.props[PROPS["fps"]] = 30

    mode = infer.granted_mode(infer.open_camera(0))

    assert (mode["width"], mode["height"], mode["fps"]) == (1280, 720, 30)


def test_a_camera_smaller_than_asked_for_says_so(exported, model, monkeypatch, camera):
    camera.max_size = (640, 480)
    camera.props[PROPS["fps"]] = 30
    channel = Channel(cancel_after=2)

    payload, _channel = run(job_for(exported, mode="camera"), channel=channel)

    assert payload["camera_mode"]["width"] == 640
    assert any("most this camera offers" in msg for _level, msg in channel.logs)


def test_the_mode_it_did_get_is_reported(exported, model, monkeypatch, camera):
    camera.props[PROPS["fps"]] = 30
    channel = Channel(cancel_after=2)

    run(job_for(exported, mode="camera"), channel=channel)

    assert any("1920x1080" in msg for _level, msg in channel.logs)


def test_the_job_s_size_and_rate_reach_the_camera(exported, model, monkeypatch):
    asked = {}

    def open_camera(index, **kwargs):
        asked.update(kwargs)
        return StubCapture(["a"] * 10, props={PROPS["width"]: 1280, PROPS["height"]: 720})

    monkeypatch.setattr(infer, "open_camera", open_camera)
    channel = Channel(cancel_after=2)

    payload, _channel = run(
        job_for(exported, mode="camera", camera_size=[1280, 720], camera_fps=15),
        channel=channel,
    )

    assert asked == {"size": (1280, 720), "fps": 15}
    assert payload["camera_request"] == {"width": 1280, "height": 720, "fps": 15}


@pytest.mark.parametrize(
    "size, fps",
    [(None, None), ("big", "fast"), ([0, 720], -5), ([99999, 720], 100000)],
)
def test_a_nonsense_request_falls_back_to_the_default(size, fps):
    assert infer._camera_request({"camera_size": size, "camera_fps": fps}) == (
        infer.CAMERA_TARGET_SIZE,
        0,
    )


def test_a_size_the_camera_rounded_says_so(exported, model, monkeypatch, camera):
    """Not smaller -- different: 2560x1440 asked, 2592x1944 given."""
    camera.max_size = (2592, 1944)
    camera.props.update({PROPS["width"]: 2592, PROPS["height"]: 1944})
    camera.set = lambda prop, value: camera.settings.append((prop, value))
    channel = Channel(cancel_after=2)

    run(job_for(exported, mode="camera", camera_size=[2560, 1440]), channel=channel)

    assert any("nearest" in msg for _level, msg in channel.logs)
    assert not any("most this camera offers" in msg for _level, msg in channel.logs)


class Clock:
    """A clock that moves only when a frame is read from the stub camera."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def camera_at(clock: Clock, fps: float, count: int, jitter: float = 0.0):
    """``count`` frames arriving at ``fps``, each a little early or late in turn."""
    for index in range(count):
        clock.now = index / fps + (jitter if index % 2 else -jitter)
        yield index


def test_a_fast_camera_is_paced_down_to_the_chosen_rate():
    clock = Clock()
    pacer = infer.Pacer(camera_at(clock, 30, 300, jitter=0.002), 15, clock=clock)

    passed = list(pacer)

    # Ten seconds of a 30 fps camera, paced to 15.
    assert 148 <= len(passed) <= 151
    assert pacer.read == 300
    assert round(pacer.delivered_fps) == 30


def test_a_rate_that_does_not_divide_the_camera_s_is_still_met_on_average():
    clock = Clock()

    passed = list(infer.Pacer(camera_at(clock, 30, 300), 24, clock=clock))

    assert 235 <= len(passed) <= 241


def test_a_camera_slower_than_the_chosen_rate_loses_nothing():
    clock = Clock()

    passed = list(infer.Pacer(camera_at(clock, 15, 100), 30, clock=clock))

    assert len(passed) == 100


def test_a_paced_camera_reports_what_it_delivered(exported, model, monkeypatch):
    play(monkeypatch, ["a"] * 20)
    channel = Channel(cancel_after=5)

    payload, _channel = run(job_for(exported, mode="camera", camera_fps=15), channel=channel)

    assert payload["camera_request"]["fps"] == 15
    assert any("at most 15 frames a second" in msg for _level, msg in channel.logs)


@pytest.mark.parametrize(
    "published, shown",
    [
        ("UCB Camera: UCB Camera", "UCB Camera"),
        ("Integrated Camera: Integrated C", "Integrated Camera"),
        ("Logitech BRIO: Front", "Logitech BRIO: Front"),
        ("", ""),
    ],
)
def test_a_device_name_is_made_fit_to_choose_from(published, shown):
    """Linux publishes the card and bus names joined, usually as one thing twice."""
    from worker.commands import cameras

    assert cameras._tidy(published) == shown


# --------------------------------------------------------------------------- #
# Class names
# --------------------------------------------------------------------------- #


def test_a_model_that_lost_its_names_borrows_the_projects(exported, photo, monkeypatch):
    """Some exporters drop the metadata; ``{0: "0"}`` is not a name."""
    result = StubResult(names={0: "0", 1: "1"})
    monkeypatch.setattr(infer, "load_yolo", lambda *a, **k: StubYOLO([StubResult(), result]))

    run(job_for(exported, source=str(photo)))

    assert result.names == {0: "cat", 1: "dog"}


def test_the_list_and_the_drawn_labels_cannot_disagree(exported, photo, monkeypatch):
    """One answer, rendered twice: the picture's labels and the list beside it."""
    result = StubResult(detections=((0, 0.8),), names={0: "bottle"})
    monkeypatch.setattr(infer, "load_yolo", lambda *a, **k: StubYOLO([StubResult(), result]))

    _payload, channel = run(job_for(exported, source=str(photo), names=["cat", "dog"]))

    assert channel.previews[0]["detections"][0]["name"] == "bottle"
    assert result.names == {0: "bottle"}


def test_a_model_that_kept_its_names_keeps_them(exported, photo, monkeypatch):
    result = StubResult(names={0: "bottle"})
    monkeypatch.setattr(infer, "load_yolo", lambda *a, **k: StubYOLO([StubResult(), result]))

    run(job_for(exported, source=str(photo), names=["cat", "dog"]))

    assert result.names == {0: "bottle"}


# --------------------------------------------------------------------------- #
# When the runtime itself fails
# --------------------------------------------------------------------------- #


def test_running_out_of_memory_arrives_with_a_remedy(exported, photo, monkeypatch):
    monkeypatch.setattr(
        infer,
        "load_yolo",
        lambda *a, **k: StubYOLO(raises=RuntimeError("CUDA out of memory")),
    )

    with pytest.raises(InferenceFailed, match="ran out of memory"):
        run(job_for(exported, source=str(photo)))


def test_an_engine_from_another_machine_is_explained(exported, photo, gpu, monkeypatch):
    monkeypatch.setattr(
        infer,
        "load_yolo",
        lambda *a, **k: StubYOLO(raises=RuntimeError("failed to deserialize engine")),
    )

    with pytest.raises(InferenceFailed, match="deserialize") as raised:
        run(job_for(exported, source=str(photo), format="engine", device="gpu"))

    assert "built it" in raised.value.hint


def test_a_model_that_returns_nothing_is_a_failure(exported, photo, monkeypatch):
    class Silent(StubYOLO):
        def predict(self, frame, **kwargs):
            return []

    monkeypatch.setattr(infer, "load_yolo", lambda *a, **k: Silent())

    with pytest.raises(InferenceFailed, match="returned nothing"):
        run(job_for(exported, source=str(photo)))


# --------------------------------------------------------------------------- #
# The statistics
# --------------------------------------------------------------------------- #


def test_the_slow_end_is_a_real_measurement_not_an_interpolation():
    assert infer._percentile([1.0, 2.0, 3.0, 100.0], 0.95) == 100.0
    assert infer._percentile([], 0.95) == 0.0


def test_the_mean_is_over_the_frames_that_were_measured(exported, video, monkeypatch):
    results = [StubResult(inference_ms=ms) for ms in (999.0, 10.0, 20.0)]
    monkeypatch.setattr(infer, "load_yolo", lambda *a, **k: StubYOLO(results))
    play(monkeypatch, ["a", "b"], total=2)

    payload, _channel = run(job_for(exported, source=str(video), mode="video"))

    # 999 was the warmup frame and is not in the average.
    assert payload["inference_ms_mean"] == 15.0


# --------------------------------------------------------------------------- #
# The shape of the input
# --------------------------------------------------------------------------- #


def frame_of(height: int, width: int):
    return types.SimpleNamespace(shape=(height, width, 3))


def run_mode(mode, frame, exported, tmp_path, monkeypatch, **extra):
    """One run of ``mode`` on frames of one shape; returns the warning codes."""
    source = tmp_path / ("clip.mp4" if mode == "video" else "shot.jpg")
    source.write_bytes(b"source")
    monkeypatch.setattr(infer, "read_photo", lambda path: frame)
    play(monkeypatch, [frame] * 5)
    job = job_for(exported, mode=mode, source=str(source), camera=0, **extra)
    channel = Channel(cancel_after=3) if mode == "camera" else Channel()
    _payload, channel = run(job, channel=channel)
    return [code for code, _msg in channel.warnings if code == infer.ASPECT_MISMATCH_WARNING]


@pytest.mark.parametrize("mode", ["photo", "video", "camera"])
def test_a_square_model_on_wide_input_warns_about_speed_once(
    mode, exported, model, tmp_path, monkeypatch
):
    found = run_mode(
        mode, frame_of(1080, 1920), exported, tmp_path, monkeypatch, imgsz_hw=[640, 640]
    )

    assert found == [infer.ASPECT_MISMATCH_WARNING]


@pytest.mark.parametrize("mode", ["photo", "video", "camera"])
def test_a_model_the_shape_of_its_input_says_nothing(
    mode, exported, model, tmp_path, monkeypatch
):
    found = run_mode(
        mode, frame_of(1080, 1920), exported, tmp_path, monkeypatch, imgsz_hw=[576, 1024]
    )

    assert found == []


def test_the_message_names_both_shapes(exported, model, tmp_path, monkeypatch):
    monkeypatch.setattr(infer, "read_photo", lambda path: frame_of(1080, 1920))
    photo = tmp_path / "shot.jpg"
    photo.write_bytes(b"photo")

    _payload, channel = run(job_for(exported, source=str(photo), imgsz_hw=[1024, 1024]))

    message = next(m for c, m in channel.warnings if c == infer.ASPECT_MISMATCH_WARNING)
    assert "1024x1024" in message and "1920x1080" in message
    assert "slower" in message


def test_an_export_with_no_recorded_size_is_not_judged(exported, model, tmp_path, monkeypatch):
    found = run_mode("photo", frame_of(1080, 1920), exported, tmp_path, monkeypatch)

    assert found == []


def test_rounding_to_the_stride_is_not_a_mismatch():
    # 640x352 is as close to 16:9 as a multiple of 32 gets at that width.
    assert infer.aspect_padding((352, 640), (1080, 1920)) < infer.ASPECT_TOLERANCE
    assert infer.aspect_padding((480, 640), (1080, 1920)) > infer.ASPECT_TOLERANCE
    assert infer.aspect_padding((640, 640), (1080, 1920)) == pytest.approx(0.4375)
    assert infer.aspect_padding((1920, 1080), (1080, 1920)) > infer.ASPECT_TOLERANCE
