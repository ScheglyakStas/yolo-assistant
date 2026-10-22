"""Run an exported model on a photo, a video or a camera, and show what it sees.

Step 7 is the only step whose output is a picture. Everything before it can be
judged from numbers -- photos copied, boxes drawn, mAP reached, a file written
-- and none of those answer the question this step exists for, which is whether
the thing in the exports folder actually finds the object when it is pointed at
one. So this command draws the boxes and sends the frame, and the numbers it
sends alongside are about speed rather than accuracy.

Six decisions are worth a note.

**The first call is not a frame.** Loading is not the same as being ready: the
first inference through any of these runtimes pays for a session, a CUDA
context and, for TensorRT, deserialising the engine -- seconds, against
milliseconds for every frame afterwards. Averaging that in would misreport a
fast model as a slow one on the exact screen someone uses to decide whether it
is fast enough. So one frame is run and thrown away, its cost is reported
separately as what starting up costs, and the timings begin afterwards.

**Photos are read through Pillow, not OpenCV.** Two reasons, both of which bite
exactly once and confusingly. ``cv2.imread`` opens a path through the C
library's ``fopen``, which on Windows cannot see a file whose name is not in
the system code page -- so a photo in a folder with a non-English name is "not
readable" when it is plainly there. And ``cv2.imdecode``, the usual way around
that, silently ignores EXIF rotation: the phone photo that Step 2 takes care to
straighten once would come back sideways here, against a model that was trained
on the straightened version. Pillow gets both right.

**The preview is throttled at the source.** A frame that is no longer the
newest is worth nothing -- see ``Channel.preview`` -- and encoding one costs
real milliseconds that would otherwise come out of the number this step is
measuring. So frames are encoded only when one is due, at a rate a person can
see, while inference and its timings run flat out.

**Stopping is how a camera ends.** A live feed has no last frame, so cancelling
one is not a failure and is not reported as one: the loop breaks, the summary is
written, and the job finishes normally with what it measured. A video that is
stopped halfway says the same thing about the frames it did reach.

**A video is recorded, not played.** A file is not a live feed: on a fast GPU
it is read and run at hundreds of frames a second, and a preview shown as fast
as it is produced is the video on fast-forward. So every frame, boxes drawn, is
written to a copy of the video at the source's own frame rate, and the page
plays that copy once the run is over -- at the speed it was filmed, whatever
speed it was processed at. The copy goes to the path the page chose, through a
``.partial`` file renamed only once it is complete, so an interrupted run never
leaves something that looks finished. A stopped run keeps the frames it reached.

**A photo's copy is written only when asked for.** The page asks, so that the
user can keep it; the copy is full size, not the shrunk preview, and goes
through the same ``.partial`` rename. Whether either copy is temporary is the
page's business, not this command's. A camera leaves no file, folder or record
behind, which is why this command can be run again, differently, as many times
as anyone likes.
"""

from __future__ import annotations

import base64
import os
import time
from collections import deque
from pathlib import Path

from worker.channel import Cancelled, Channel
from worker.commands.cameras import capture_backend
from worker.i18n import tr, trn
from worker.protocol import Phase
from worker.util.errors import (
    CameraUnavailable,
    GpuUnavailable,
    InferenceFailed,
    ModelUnavailable,
    SourceUnreadable,
)

PHOTO = "photo"
VIDEO = "video"
CAMERA = "camera"
MODES = (PHOTO, VIDEO, CAMERA)

CPU = "cpu"
GPU = "gpu"

#: The warning code for an ONNX test on the GPU that onnxruntime runs on the
#: processor. The page matches it to change what its numbers are labelled as;
#: ``app/core/inference.py`` holds the same string.
ONNX_ON_CPU_WARNING = "onnx_on_cpu"

#: The warning code for an input whose shape is not the model's. The page shows
#: it beside the picture rather than only in the log; ``app/core/inference.py``
#: holds the same string.
ASPECT_MISMATCH_WARNING = "aspect_mismatch"

#: How far the input's aspect ratio may be from the model's before it is worth
#: a warning, as the share of the model's input that ends up as padding. Sides
#: are multiples of 32, so an export meant to be 16:9 is only close to it --
#: 640x352 pads a 1920x1080 frame by 2%, 320x192 by 6% -- and neither is the
#: mistake this is here to catch. 4:3 into 16:9 is 25%; a square model on 16:9
#: video is 44%.
ASPECT_TOLERANCE = 0.10

#: What a camera is asked for when the page does not say otherwise.
#:
#: Not "the highest it has", which is the obvious choice and the wrong one.
#: Measured on a UVC webcam: asking for 3840x2160 lands on 3264x2448 at 15
#: frames a second, which is a worse live picture than 1920x1080 at 30 and four
#: times the pixels to push through a detector whose input is 640 across.
#: Drivers pick the nearest mode they have, so a lesser camera quietly gets its
#: own maximum instead.
CAMERA_TARGET_SIZE = (1920, 1080)

#: A size or a rate outside these is a job that is wrong, not a camera that is
#: unusually good, and is replaced by the default rather than sent to a driver.
MAX_CAMERA_EDGE = 8192
MAX_CAMERA_FPS = 240

#: The single most important request in this file for anyone with a webcam.
#: Most UVC cameras offer their higher modes in motion-JPEG only; in raw YUYV
#: the same camera measured 3 frames a second at 1920x1080 against 30 in MJPG.
#: *When* it is asked for depends on the backend -- see :func:`camera_requests`
#: -- and getting that wrong produces a working, unusable camera.
CAMERA_FOURCC = "MJPG"

#: How early a frame may arrive and still count as on time when the camera is
#: paced to a frame rate, as a fraction of the interval. Cameras deliver with a
#: few milliseconds of jitter; without slack, a 30 fps camera paced to 15 drops
#: the frame that is due a hair early and waits a whole extra frame for the
#: next, landing nearer 10 than 15.
PACING_SLACK = 0.2

#: How many frames the driver may have in hand for us.
#:
#: This is a latency-against-throughput dial and the obvious setting is the
#: wrong one. At one frame the driver hands over only what we are already
#: waiting for and drops the rest: measured on a 30 fps camera with 5 ms of
#: work per frame, reading collapsed to 20 frames a second. Left alone, the
#: same camera delivered 29.8 -- and left up to four frames queued behind a
#: consumer slower than it, which is a seventh of a second of the picture
#: lagging the room. Two gives the full 29.9 and halves the queue. A consumer
#: faster than the camera never fills it at all.
CAMERA_BUFFER_FRAMES = 2

#: Preview sizes. A live frame is for watching -- the boxes are legible at 960
#: and the JPEG is a tenth the size -- while a photo is for inspecting, and is
#: the one frame where the cost of a larger image is paid once.
LIVE_MAX_EDGE = 960
LIVE_QUALITY = 80
PHOTO_MAX_EDGE = 1280
PHOTO_QUALITY = 92

#: Detections listed beside the picture. Past this the list stops being
#: something a person reads, and the boxes are on the frame regardless.
MAX_DETECTIONS_REPORTED = 30

#: A frame count OpenCV reports above this is not a frame count. Some
#: containers store a duration the header got wrong, and a progress bar built
#: on 2 billion frames is worse than no progress bar.
MAX_PLAUSIBLE_FRAMES = 10_000_000

#: The codec the annotated copy of a video is written in. MPEG-4 Part 2 rather
#: than H.264, because it is the one the pip builds of OpenCV can always write:
#: asking them for ``avc1`` fails, loudly, on a machine without an H.264
#: encoder. Qt's FFmpeg backend plays it without complaint.
RECORD_FOURCC = "mp4v"

#: What a video that does not say its own frame rate is played back at. A rate
#: outside this range is a header that is wrong, not a video that is fast.
DEFAULT_FPS = 30.0
PLAUSIBLE_FPS = (1.0, 240.0)


def run(channel: Channel, job: dict) -> dict:
    mode = str(job.get("mode") or PHOTO)
    if mode not in MODES:
        raise InferenceFailed(
            tr("{mode!r} is not something this app can test against.", mode=mode)
        )

    model_path = Path(str(job.get("model") or "")).expanduser()
    fmt = str(job.get("format") or "")
    conf = _confidence(job.get("conf"))
    names = {index: str(name) for index, name in enumerate(job.get("names") or ())}

    # Everything that can be checked without importing torch is checked first,
    # so that a deleted file or a missing runtime is reported in a second
    # rather than after a ten-second import.
    if not model_path.exists():
        raise InferenceFailed(
            tr("{path} is not there any more.", path=model_path),
            hint=tr(
                "It has been deleted or renamed since this page listed it. Go back to "
                "Step 7 and pick another file, or export it again in Step 6."
            ),
        )
    source = _resolve_source(job, mode)
    channel.progress(Phase.SETUP, 0, 4, unit="steps")

    _require_runtime(job.get("needs") or (), fmt=fmt)
    channel.progress(Phase.SETUP, 1, 4, unit="steps")

    device = _resolve_device(str(job.get("device") or GPU), fmt=fmt, channel=channel)
    channel.progress(Phase.SETUP, 2, 4, unit="steps")

    model = load_yolo(str(model_path), channel=channel)
    channel.progress(Phase.SETUP, 3, 4, unit="steps")

    channel.ready(
        model=model_path.name,
        format=fmt,
        device=str(device),
        mode=mode,
        conf=conf,
    )
    if device != CPU:
        channel.log(
            tr(
                "Testing {model} on the GPU at confidence {conf:g}.",
                model=model_path.name,
                conf=conf,
            )
        )
    else:
        channel.log(
            tr(
                "Testing {model} on the processor at confidence {conf:g}.",
                model=model_path.name,
                conf=conf,
            )
        )

    runner = Runner(
        channel,
        model,
        device=device,
        conf=conf,
        names=names,
        model_hw=_model_hw(job.get("imgsz_hw")),
    )
    if mode == PHOTO:
        raw_output = str(job.get("output") or "").strip()
        output = Path(raw_output).expanduser() if raw_output else None
        return _run_photo(channel, runner, Path(source), output)
    if mode == VIDEO:
        return _run_video(channel, runner, Path(source), _output_path(job, Path(source)))
    size, fps = _camera_request(job)
    return _run_camera(channel, runner, int(source), size=size, fps=fps)


# --------------------------------------------------------------------------- #
# The three modes
# --------------------------------------------------------------------------- #


def _run_photo(
    channel: Channel, runner: "Runner", path: Path, output: Path | None = None
) -> dict:
    """One image, run twice: once to wake the runtime, once to be measured."""
    frame = read_photo(path)
    startup_ms = runner.warm_up(frame)
    channel.progress(Phase.INFER, 0, 1, unit="photos")

    outcome = runner.infer(frame)
    stats = Stats()
    stats.add(outcome)
    painted = _plot(outcome.result)
    runner.send_preview(
        0, outcome, stats, painted=painted, max_edge=PHOTO_MAX_EDGE, quality=PHOTO_QUALITY
    )
    channel.progress(Phase.INFER, 1, 1, unit="photos")

    written = None
    if output is not None:
        written = _save_photo(output, painted)
        channel.log(tr("Saved the photo, boxes drawn, to {path}.", path=written))

    found = len(outcome.detections)
    channel.log(
        trn(
            "{name}: {n} object found in {ms:.1f} ms. Getting the model ready took "
            "{seconds:.1f} s, which is paid once rather than per frame.",
            "{name}: {n} objects found in {ms:.1f} ms. Getting the model ready took "
            "{seconds:.1f} s, which is paid once rather than per frame.",
            found,
            name=path.name,
            ms=outcome.inference_ms,
            seconds=startup_ms / 1000,
        )
    )
    if not found:
        # The failure that does not look like one, exactly as in Step 3: a
        # model that finds nothing returns cleanly and looks like a success.
        channel.warning(
            "infer",
            tr(
                "Nothing was found in this photo. That is a real answer if there is "
                "nothing there -- otherwise try a lower confidence threshold, or a "
                "photo more like the ones it was trained on."
            ),
        )
    return {
        "mode": PHOTO,
        "source": str(path),
        "output": str(written) if written is not None else "",
        "startup_ms": round(startup_ms, 1),
        **stats.snapshot(),
    }


def _save_photo(path: Path, image) -> Path:
    """Write the annotated photo through a ``.partial`` file, as a video is."""
    partial = path.with_name(f"{path.stem}.partial{path.suffix}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_photo(partial, image)
        os.replace(partial, path)
    except OSError as exc:
        try:
            partial.unlink()
        except OSError:
            pass
        raise InferenceFailed(
            tr("The photo with its boxes could not be saved: {error}", error=exc),
            hint=tr("The project folder may be read-only or out of space."),
        ) from exc
    return path


def _run_video(channel: Channel, runner: "Runner", path: Path, output: Path) -> dict:
    capture = open_video(path)
    recorder = Recorder(output, fps=_video_fps(capture))
    try:
        total = _video_total(capture)
        if total:
            channel.log(
                trn("{name}: {n:,} frames.", "{name}: {n:,} frames.", total, name=path.name)
            )
        else:
            channel.log(
                tr(
                    "{name}: this file does not say how many frames it has, so "
                    "there is no estimate of how long it will take.",
                    name=path.name,
                )
            )
        channel.log(
            tr(
                "Each frame is written, boxes drawn, to {output}. It plays at "
                "{fps:g} frames a second once every frame has been run.",
                output=output.name,
                fps=recorder.fps,
            )
        )
        summary = _play(
            channel, runner, _frames(capture), total=total, unit="frames", recorder=recorder
        )
    except BaseException:
        recorder.discard()
        raise
    finally:
        _release(capture)
    written = recorder.finish()
    if written is not None:
        channel.log(
            trn(
                "Saved {n:,} annotated frames to {path}.",
                "Saved {n:,} annotated frames to {path}.",
                recorder.frames,
                path=written,
            )
        )
    return {
        "mode": VIDEO,
        "source": str(path),
        "output": str(written) if written is not None else "",
        "output_fps": recorder.fps,
        **summary,
    }


def _run_camera(
    channel: Channel,
    runner: "Runner",
    index: int,
    *,
    size: tuple[int, int] = CAMERA_TARGET_SIZE,
    fps: int = 0,
) -> dict:
    capture = open_camera(index, size=size, fps=fps)
    pacer = None
    try:
        mode = granted_mode(capture)
        channel.log(
            tr("Camera {index} is open at {mode}.", index=index, mode=_describe_mode(mode))
        )
        width, height = size
        if _is_smaller_than_asked(mode, size):
            # Said plainly rather than left for the user to notice in a blurry
            # picture: this is the camera's own limit, not a setting in the app.
            channel.log(
                tr(
                    "That is the most this camera offers; {width}x{height} was asked for.",
                    width=width,
                    height=height,
                ),
                level="warning",
            )
        elif _is_other_than_asked(mode, size):
            channel.log(
                tr(
                    "This camera has no {width}x{height} mode, so it chose the nearest "
                    "one it has.",
                    width=width,
                    height=height,
                ),
                level="warning",
            )
        frames = _frames(capture)
        if fps:
            # Asked of the driver as well, but not left to it: plenty of
            # Windows drivers accept a rate, report it back, and go on
            # delivering their own. Pacing here makes the chosen rate the rate
            # the model runs at whatever the camera does.
            pacer = Pacer(frames, fps)
            frames = pacer
            channel.log(
                tr(
                    "The model is given at most {fps} frames a second; the camera's "
                    "others are read and skipped, so the picture stays current.",
                    fps=fps,
                )
            )
        channel.log(tr("It runs until you press Stop. Nothing is saved."))
        summary = _play(channel, runner, frames, total=None, unit="frames")
    finally:
        _release(capture)
    if pacer is not None and pacer.delivered_fps:
        summary["camera_fps_delivered"] = round(pacer.delivered_fps, 2)
        channel.log(
            tr(
                "The camera delivered {delivered:.1f} frames a second; {fps} were asked for.",
                delivered=pacer.delivered_fps,
                fps=fps,
            )
        )
    return {
        "mode": CAMERA,
        "camera": index,
        "camera_mode": mode,
        "camera_request": {"width": size[0], "height": size[1], "fps": fps},
        **summary,
    }


def _describe_mode(mode: dict) -> str:
    parts = [f"{mode.get('width', 0)}x{mode.get('height', 0)}"]
    if mode.get("fps"):
        parts.append(tr("{fps} frames a second", fps=mode["fps"]))
    if mode.get("fourcc"):
        parts.append(mode["fourcc"])
    return ", ".join(parts)


def _is_smaller_than_asked(mode: dict, size=CAMERA_TARGET_SIZE) -> bool:
    width, height = size
    return 0 < int(mode.get("width") or 0) < width or 0 < int(mode.get("height") or 0) < height


def _is_other_than_asked(mode: dict, size=CAMERA_TARGET_SIZE) -> bool:
    """Granted a size that is not the one asked for, not merely a smaller one."""
    granted = (int(mode.get("width") or 0), int(mode.get("height") or 0))
    return all(granted) and granted != tuple(size)


def _camera_request(job: dict) -> tuple[tuple[int, int], int]:
    """The size and frame rate the page asked for, or the defaults.

    A frame rate of 0 means "whatever the camera runs at": nothing is asked of
    the driver and nothing is paced.
    """
    size = CAMERA_TARGET_SIZE
    raw = job.get("camera_size")
    try:
        width, height = (int(value) for value in raw)
        if 0 < width <= MAX_CAMERA_EDGE and 0 < height <= MAX_CAMERA_EDGE:
            size = (width, height)
    except (TypeError, ValueError):
        pass
    try:
        fps = int(job.get("camera_fps") or 0)
    except (TypeError, ValueError):
        fps = 0
    if not 0 <= fps <= MAX_CAMERA_FPS:
        fps = 0
    return size, fps


def _play(
    channel: Channel,
    runner: "Runner",
    frames,
    *,
    total: int | None,
    unit: str,
    recorder: "Recorder | None" = None,
) -> dict:
    """The loop the video and the camera share.

    Identical in every way that matters: the only difference between a file and
    a camera is whether the frames run out, whether there is a total to measure
    progress against, and whether each frame is also recorded.
    """
    stats = Stats()
    startup_ms = 0.0
    stopped = False
    index = 0

    try:
        for index, frame in enumerate(frames):
            channel.check_cancelled()
            if index == 0:
                startup_ms = runner.warm_up(frame)

            outcome = runner.infer(frame)
            stats.add(outcome)
            painted = None
            if recorder is not None:
                painted = _plot(outcome.result)
                recorder.write(painted)

            # The channel owns the rate: it is the thing that knows whether a
            # frame sent now would be shown or superseded, and drawing and
            # compressing one that will not be shown is measurable work taken
            # out of the measurement.
            if channel.preview_due():
                runner.send_preview(
                    index,
                    outcome,
                    stats,
                    painted=painted,
                    max_edge=LIVE_MAX_EDGE,
                    quality=LIVE_QUALITY,
                )
            channel.progress(
                Phase.INFER,
                stats.frames,
                total,
                unit=unit,
                eta_s=stats.eta(total),
                detail=tr("{fps:.1f} frames a second", fps=stats.fps),
            )
    except Cancelled:
        # Stopping a live feed is how it ends, not a failure: the summary is
        # what the user was watching for, and losing it to a cancellation would
        # throw away the only result this step produces.
        stopped = True

    # The last frame is always shown, whatever the throttle was doing when the
    # stream ended -- otherwise a stopped feed freezes on a frame up to a
    # twelfth of a second older than the one it really ended on.
    if stats.frames:
        outcome = runner.last
        if outcome is not None:
            runner.send_preview(
                index, outcome, stats, max_edge=LIVE_MAX_EDGE, quality=LIVE_QUALITY
            )

    snapshot = stats.snapshot()
    if stats.frames:
        timings = {
            "mean": snapshot["inference_ms_mean"],
            "p95": snapshot["inference_ms_p95"],
            "fps": stats.fps,
        }
        # Both English forms are the same on purpose; Polish needs its own three.
        if stopped:
            channel.log(
                trn(
                    "Stopped after {n:,} frames, {mean:.1f} ms a frame on average "
                    "({p95:.1f} ms at the slow end), {fps:.1f} frames a second end to end.",
                    "Stopped after {n:,} frames, {mean:.1f} ms a frame on average "
                    "({p95:.1f} ms at the slow end), {fps:.1f} frames a second end to end.",
                    stats.frames,
                    **timings,
                )
            )
        else:
            channel.log(
                trn(
                    "Finished: {n:,} frames, {mean:.1f} ms a frame on average "
                    "({p95:.1f} ms at the slow end), {fps:.1f} frames a second end to end.",
                    "Finished: {n:,} frames, {mean:.1f} ms a frame on average "
                    "({p95:.1f} ms at the slow end), {fps:.1f} frames a second end to end.",
                    stats.frames,
                    **timings,
                )
            )
    else:
        channel.log(tr("No frames were read."), level="warning")
    return {"stopped": stopped, "startup_ms": round(startup_ms, 1), **snapshot}


class Recorder:
    """The annotated copy of a video, written frame by frame.

    The writer is opened at the first frame rather than up front, because its
    size is the size of the frame Ultralytics draws on, and that is known for
    certain only once there is one.
    """

    def __init__(self, path: Path, *, fps: float) -> None:
        self.path = path
        self.fps = fps
        self.frames = 0
        self._partial = path.with_name(f"{path.stem}.partial{path.suffix}")
        self._writer = None

    def write(self, image) -> None:
        if self._writer is None:
            self._writer = open_writer(self._partial, fps=self.fps, first=image)
        self._writer.write(image)
        self.frames += 1

    def finish(self) -> Path | None:
        """Close the file and put it in place, or ``None`` if nothing was written."""
        if self._writer is None:
            return None
        _release(self._writer)
        self._writer = None
        try:
            os.replace(self._partial, self.path)
        except OSError as exc:
            raise InferenceFailed(
                tr("The annotated video could not be saved: {error}", error=exc)
            ) from exc
        return self.path

    def discard(self) -> None:
        if self._writer is not None:
            _release(self._writer)
            self._writer = None
        try:
            self._partial.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Running the model
# --------------------------------------------------------------------------- #


class Outcome:
    """One frame, its boxes, and what it cost."""

    __slots__ = ("frame", "result", "speed", "detections")

    def __init__(self, frame, result, speed: dict, detections: list[dict]) -> None:
        self.frame = frame
        self.result = result
        self.speed = speed
        self.detections = detections

    @property
    def inference_ms(self) -> float:
        return float(self.speed.get("inference") or 0.0)

    @property
    def total_ms(self) -> float:
        return float(sum(float(value or 0.0) for value in self.speed.values()))


class Runner:
    """Holds the model and turns one frame into an :class:`Outcome`."""

    def __init__(
        self,
        channel: Channel,
        model,
        *,
        device,
        conf: float,
        names: dict,
        model_hw: tuple[int, int] | None = None,
    ) -> None:
        self._channel = channel
        self._model = model
        self._device = device
        self._conf = conf
        self._names = names
        self._model_hw = model_hw
        self.last: Outcome | None = None

    def warm_up(self, frame) -> float:
        """One inference, discarded, so the next one is worth timing.

        Also where the input's shape is first known in every mode, so it is
        where it is compared with the model's.
        """
        self._check_aspect(frame)
        self._channel.log(
            tr(
                "Getting the model ready. The first frame through is much slower than "
                "the rest and is not counted."
            )
        )
        started = time.monotonic()
        self._predict(frame)
        return (time.monotonic() - started) * 1000.0

    def _check_aspect(self, frame) -> None:
        """Warn when the input is a different shape from the model.

        An exported model's input is fixed, so a frame of another shape is
        scaled to fit and the rest is padding -- run through the network like
        any other pixels. Nothing is lost in accuracy, which is why it goes
        unnoticed; it is lost in speed, on every frame.
        """
        if not self._model_hw:
            return  # an export with no record of its size
        model_h, model_w = self._model_hw
        try:
            height, width = int(frame.shape[0]), int(frame.shape[1])
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        if aspect_padding((model_h, model_w), (height, width)) <= ASPECT_TOLERANCE:
            return
        self._channel.warning(
            ASPECT_MISMATCH_WARNING,
            tr(
                "The model was exported at {model_w}x{model_h}, but this input is "
                "{width}x{height} -- a different aspect ratio. Every frame is padded out "
                "to the model's shape and the padding is processed too, so inference is "
                "slower than it needs to be. Export again in Step 6 at a size with the "
                "input's shape for the best speed.",
                model_w=model_w,
                model_h=model_h,
                width=width,
                height=height,
            ),
        )

    def infer(self, frame) -> Outcome:
        result = self._predict(frame)
        # The names are settled once, here, before anything reads them. The
        # drawn label and the list beside the picture are two renderings of one
        # answer, and deciding separately is how they come to disagree.
        _apply_names(result, self._names)
        outcome = Outcome(frame, result, _speed_of(result), _detections(result))
        self.last = outcome
        return outcome

    def send_preview(
        self, index: int, outcome: Outcome, stats: "Stats", *, painted=None, **encoding
    ) -> None:
        """Send one frame. ``painted`` is the frame already drawn, if it has been."""
        if painted is None:
            painted = _plot(outcome.result)
        image, width, height = encode_frame(painted, **encoding)
        self._channel.preview(
            index,
            image,
            width=width,
            height=height,
            detections=outcome.detections[:MAX_DETECTIONS_REPORTED],
            timings={
                "inference_ms": round(outcome.inference_ms, 2),
                "total_ms": round(outcome.total_ms, 2),
            },
            stats=stats.snapshot(),
        )

    def _predict(self, frame):
        try:
            # Deliberately no ``imgsz``: an exported model's input size is fixed
            # at export time and recorded in its own metadata, and passing a
            # different one either is ignored or fails a fixed-shape engine.
            results = self._model.predict(
                frame, device=self._device, conf=self._conf, verbose=False
            )
        except Exception as exc:  # noqa: BLE001 - every runtime raises its own
            raise _explain(exc) from exc
        if not results:
            raise InferenceFailed(tr("The model returned nothing for this frame."))
        return results[0]


class Stats:
    """Per-frame timings, summarised as they arrive.

    The mean answers "how fast is this model" and the 95th percentile answers
    "how slow does it get", which is the one that decides whether a live feed
    keeps up. Both are over the frames that were measured; the startup cost is
    reported beside them and never folded in.

    Nothing is accumulated without a bound. A summary is taken for every frame
    displayed, and an hour of camera at thirty frames a second is a hundred
    thousand measurements -- so the means are running totals, and the
    percentile is taken over a window of recent frames. That window is also the
    more useful answer: a feed that has been fine for an hour and is stuttering
    now should say so.
    """

    #: Frames the percentile looks back over. At thirty a second this is the
    #: last minute or so, which is the horizon someone watching a live feed is
    #: actually judging.
    WINDOW = 2048

    def __init__(self) -> None:
        self._recent: deque[float] = deque(maxlen=self.WINDOW)
        self._inference_sum = 0.0
        self._total_sum = 0.0
        self._last_inference = 0.0
        self._started: float | None = None
        self.frames = 0
        self.detections = 0

    def add(self, outcome: Outcome) -> None:
        if self._started is None:
            self._started = time.monotonic()
        self.frames += 1
        self.detections += len(outcome.detections)
        self._last_inference = outcome.inference_ms
        self._inference_sum += outcome.inference_ms
        self._total_sum += outcome.total_ms
        self._recent.append(outcome.inference_ms)

    @property
    def elapsed(self) -> float:
        return 0.0 if self._started is None else max(1e-6, time.monotonic() - self._started)

    @property
    def fps(self) -> float:
        """End to end, decoding included, because that is what the user sees."""
        return self.frames / self.elapsed if self.frames else 0.0

    def eta(self, total: int | None) -> float | None:
        if not total or not self.frames or self.frames >= total:
            return None
        return round((total - self.frames) / max(self.fps, 1e-6), 1)

    def snapshot(self) -> dict:
        frames = max(1, self.frames)
        return {
            "frames": self.frames,
            "fps": round(self.fps, 2),
            "detections": self.detections,
            "inference_ms": round(self._last_inference, 2),
            "inference_ms_mean": round(self._inference_sum / frames, 2),
            "inference_ms_p95": round(_percentile(self._recent, 0.95), 2),
            "total_ms_mean": round(self._total_sum / frames, 2),
        }


# --------------------------------------------------------------------------- #
# Reading frames
# --------------------------------------------------------------------------- #


def read_photo(path: Path):
    """One photo as a BGR array, straightened. Replaced wholesale in tests.

    See the module docstring for why this is not ``cv2.imread``.
    """
    import numpy as np
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(path) as handle:
            upright = ImageOps.exif_transpose(handle)
            array = np.asarray(upright.convert("RGB"))
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise SourceUnreadable(
            tr("{name} could not be read: {error}", name=path.name, error=exc)
        ) from exc
    if array.size == 0:
        raise SourceUnreadable(
            tr("{name} opened but contains no image data.", name=path.name)
        )
    # Ultralytics and OpenCV both expect blue first.
    return array[:, :, ::-1].copy()


def open_video(path: Path):
    """An OpenCV capture over a video file. Replaced wholesale in tests."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        hint = tr(
            "The codec inside it may not be one this OpenCV build reads. An MP4 "
            "with H.264 video is the safest thing to test with."
        )
        if not str(path).isascii():
            hint += " " + tr(
                "The path also contains characters outside this system's default "
                "encoding, which some video backends cannot open; a folder with a "
                "plain ASCII name rules that out."
            )
        raise SourceUnreadable(
            tr("{name} could not be opened.", name=path.name),
            hint=hint,
        )
    return capture


def open_camera(index: int, size=CAMERA_TARGET_SIZE, fps: int = 0):
    """An OpenCV capture over a camera, asked for a usable mode.

    Opening a camera and reading from it is not enough, which is the trap this
    function exists to avoid. OpenCV takes whatever the driver offers by
    default, and for most webcams that is 640x480 -- on a camera perfectly
    capable of 1920x1080 at the same 30 frames a second. Four things are asked
    for, in an order that matters and that :func:`camera_requests` explains:

    * **Motion-JPEG.** See :data:`CAMERA_FOURCC`.
    * **A size**, which the driver rounds to the nearest mode it has.
    * **A frame rate**, only if one was chosen, and which some drivers ignore.
    * **A short queue**, so that a model slower than the camera shows a
      picture of now rather than of a moment ago. Short, not absent: see
      :data:`CAMERA_BUFFER_FRAMES` for why asking for a single frame costs a
      third of the frame rate.

    What was actually granted is read back rather than assumed, because none of
    them is guaranteed and a camera running at something other than what was
    asked for is exactly the thing the user needs told.
    """
    import cv2

    backend = capture_backend(cv2)
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        raise CameraUnavailable(tr("Camera {index} could not be opened.", index=index))

    for prop, value in camera_requests(cv2, backend, size=size, fps=fps):
        try:
            capture.set(prop, value)
        except Exception:  # noqa: BLE001 - a backend may refuse any of them
            pass
    return capture


def camera_requests(cv2, backend: int, *, size=CAMERA_TARGET_SIZE, fps: int = 0) -> list:
    """The ``(property, value)`` pairs to set on a new capture, in order.

    The order is per backend because the backends disagree about what a request
    means, and each was measured on a webcam that does 1920x1080 at 30 in MJPG
    and at 3 in raw YUYV:

    * **DirectShow** re-negotiates the whole format each time the size or the
      rate changes, choosing the first pixel format the camera lists for it --
      raw, on most webcams -- and forgetting an earlier MJPG request. So the
      format goes *last*, when it re-negotiates once more with the size and
      rate already in place. Asked for first, as V4L2 wants, it gave 1080p at
      3 frames a second; asked for last, 30.
    * **V4L2** and the rest take the format as the thing the next size request
      is looked up in, so it goes *first*: a size the camera only has in MJPG,
      asked for while it is still in raw mode, gets the size and a frame rate in
      the single digits.
    """
    width, height = (int(value) for value in size)
    fourcc = (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
    mode = [(cv2.CAP_PROP_FRAME_WIDTH, width), (cv2.CAP_PROP_FRAME_HEIGHT, height)]
    if fps:
        mode.append((cv2.CAP_PROP_FPS, int(fps)))
    buffer = (cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_FRAMES)

    dshow = getattr(cv2, "CAP_DSHOW", None)
    if dshow is not None and backend == dshow:
        return [*mode, fourcc, buffer]
    return [fourcc, *mode, buffer]


def granted_mode(capture) -> dict:
    """What a capture is actually running at, as opposed to what it was asked for.

    A property the backend does not know comes back as -1 or 0 -- DirectShow
    reports the frame rate as -1 unless one was asked for -- and is reported as
    0, unknown, rather than as a camera running at minus one frame a second.
    """
    import cv2

    def number(prop, default=0):
        try:
            value = round(float(capture.get(prop) or default))
        except Exception:  # noqa: BLE001 - a stub or a backend without the property
            return default
        return value if value > 0 else default

    raw = number(cv2.CAP_PROP_FOURCC)
    fourcc = "".join(chr((raw >> (8 * shift)) & 0xFF) for shift in range(4)).strip()
    return {
        "width": number(cv2.CAP_PROP_FRAME_WIDTH),
        "height": number(cv2.CAP_PROP_FRAME_HEIGHT),
        "fps": number(cv2.CAP_PROP_FPS),
        "fourcc": fourcc if fourcc.isprintable() else "",
    }


def _frames(capture):
    """Yield BGR frames until the source runs out.

    A camera never runs out, which is what makes this the same loop for both: a
    file ends by returning nothing, and a camera ends by being unplugged.
    """
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            return
        yield frame


class Pacer:
    """Frames from a camera, at no more than ``fps`` a second.

    Every frame is still read -- skipping reads would leave the driver's queue
    full of old pictures -- but only the ones that are due are passed on. The
    schedule advances by one interval per frame passed on rather than restarting
    from each, so a 30 fps camera paced to 24 averages 24 rather than rounding
    down to 15. A model slower than the rate is never held back: once it falls
    a whole interval behind, the schedule restarts from now.
    """

    def __init__(self, frames, fps: float, *, clock=time.monotonic) -> None:
        self._frames = frames
        self._interval = 1.0 / float(fps)
        self._clock = clock
        self.read = 0
        self.passed = 0
        self._started: float | None = None
        self._last: float | None = None

    def __iter__(self):
        due = None
        slack = self._interval * PACING_SLACK
        for frame in self._frames:
            now = self._clock()
            if self._started is None:
                self._started = now
            self._last = now
            self.read += 1
            if due is not None and now < due - slack:
                continue
            if due is None or now - due > self._interval:
                due = now + self._interval
            else:
                due += self._interval
            self.passed += 1
            yield frame

    @property
    def delivered_fps(self) -> float:
        """How fast the camera itself was handing frames over, skipped ones included."""
        if self._started is None or self._last is None or self.read < 2:
            return 0.0
        elapsed = self._last - self._started
        return (self.read - 1) / elapsed if elapsed > 0 else 0.0


def _release(capture) -> None:
    try:
        capture.release()
    except Exception:  # noqa: BLE001 - a stub or a dead device
        pass


def open_writer(path: Path, *, fps: float, first):
    """An OpenCV writer sized for ``first``, the first frame. Replaced in tests."""
    import cv2

    size = (int(first.shape[1]), int(first.shape[0]))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InferenceFailed(
            tr("{folder} could not be created: {error}", folder=path.parent, error=exc)
        ) from exc
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*RECORD_FOURCC), fps, size)
    if not writer.isOpened():
        if str(path).isascii():
            hint = tr("The project folder may be read-only or out of space.")
        else:
            hint = tr(
                "The project folder may be read-only or out of space, or its path "
                "contains characters this system's video backend cannot handle."
            )
        raise InferenceFailed(
            tr("{name} could not be opened for writing.", name=path.name),
            hint=hint,
        )
    return writer


def write_photo(path: Path, image) -> None:
    """A BGR array to an image file in ``path``'s format. Replaced in tests.

    Encoded in memory and written by Python rather than with ``cv2.imwrite``,
    for the same reason photos are not read with ``cv2.imread``: OpenCV's own
    file handling cannot open a path outside the system code page on Windows.
    """
    import cv2

    suffix = path.suffix.lower()
    # The suffix of a ``.partial`` name is still the real one.
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if suffix in (".jpg", ".jpeg") else []
    ok, buffer = cv2.imencode(suffix, image, params)
    if not ok:
        raise OSError(tr("OpenCV could not encode a {suffix} image", suffix=suffix))
    path.write_bytes(buffer.tobytes())


def _video_fps(capture) -> float:
    import cv2

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    except Exception:  # noqa: BLE001
        return DEFAULT_FPS
    low, high = PLAUSIBLE_FPS
    return round(fps, 3) if low <= fps <= high else DEFAULT_FPS


def _video_total(capture) -> int | None:
    import cv2

    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    except Exception:  # noqa: BLE001
        return None
    return total if 0 < total <= MAX_PLAUSIBLE_FRAMES else None


def encode_frame(image, *, max_edge: int, quality: int) -> tuple[str, int, int]:
    """A BGR array as base64 JPEG, shrunk to fit. Replaced wholesale in tests."""
    import cv2

    height, width = int(image.shape[0]), int(image.shape[1])
    longest = max(height, width)
    if longest > max_edge:
        scale = max_edge / float(longest)
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise InferenceFailed(tr("A frame could not be prepared for display."))
    return base64.b64encode(buffer.tobytes()).decode("ascii"), width, height


# --------------------------------------------------------------------------- #
# What the model said
# --------------------------------------------------------------------------- #


def _apply_names(result, names: dict) -> None:
    """Give the result usable class names, if it does not have its own.

    Some formats carry the names in their metadata and some lose them on the
    way out of Step 6. Where they are lost, the project's own names are better
    than "0" -- and where they are not, the model's are authoritative: a
    checkpoint copied in from elsewhere knows what it was trained on and this
    project's ``classes.txt`` does not.
    """
    if not names or not _names_are_placeholders(getattr(result, "names", None)):
        return
    try:
        result.names = dict(names)
    except Exception:  # noqa: BLE001 - a read-only attribute is not fatal
        pass


def _plot(result):
    """The frame with the boxes on it, as Ultralytics draws them.

    Its own drawing rather than ours, so that what the user sees here is what
    every other Ultralytics tool would show for the same result -- and because
    a second drawing routine is a second thing that can disagree about a box.
    """
    try:
        return result.plot()
    except Exception as exc:  # noqa: BLE001
        raise InferenceFailed(tr("The result could not be drawn: {error}", error=exc)) from exc


def _names_are_placeholders(names) -> bool:
    """Whether the model's own class names say anything.

    An exporter that lost the metadata leaves numbers behind -- ``{0: "0"}`` --
    and a name that is its own index is not a name.
    """
    if not names:
        return True
    try:
        items = names.items() if hasattr(names, "items") else enumerate(names)
        return all(str(name).strip() in ("", str(index)) for index, name in items)
    except Exception:  # noqa: BLE001
        return False


def _detections(result) -> list[dict]:
    """The boxes as plain data, strongest first, for the list beside the picture.

    Named from the result's own names, which :func:`_apply_names` has already
    settled, so that this list and the labels drawn on the frame cannot differ.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    names = getattr(result, "names", None) or {}

    found: list[dict] = []
    try:
        classes = [int(value) for value in _as_list(getattr(boxes, "cls", None))]
        scores = [float(value) for value in _as_list(getattr(boxes, "conf", None))]
    except (TypeError, ValueError):
        return []

    for index, score in zip(classes, scores):
        found_name = names.get(index) if hasattr(names, "get") else None
        name = str(found_name) if found_name else tr("class {index}", index=index)
        found.append({"index": index, "name": name, "conf": round(score, 3)})
    found.sort(key=lambda item: -item["conf"])
    return found


def _as_list(value) -> list:
    if value is None:
        return []
    for attribute in ("tolist",):
        method = getattr(value, attribute, None)
        if callable(method):
            return list(method())
    return list(value)


def _speed_of(result) -> dict:
    """Ultralytics' own per-stage timings, in milliseconds.

    Read rather than measured. A wall-clock timing around ``predict`` would
    also count the frame being handed across, and the number the user is
    comparing against a datasheet is the model's own.
    """
    speed = getattr(result, "speed", None)
    if not isinstance(speed, dict):
        return {}
    out: dict[str, float] = {}
    for key in ("preprocess", "inference", "postprocess"):
        try:
            out[key] = float(speed.get(key) or 0.0)
        except (TypeError, ValueError):
            out[key] = 0.0
    return out


# --------------------------------------------------------------------------- #
# Getting ready
# --------------------------------------------------------------------------- #


def _output_path(job: dict, source: Path) -> Path:
    """Where the annotated copy of a video goes: where the page said, or beside it."""
    raw = str(job.get("output") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return source.with_name(f"{source.stem}_detections.mp4")


def _resolve_source(job: dict, mode: str):
    """The photo, the video or the camera index, checked before anything loads."""
    if mode == CAMERA:
        try:
            return int(job.get("camera") or 0)
        except (TypeError, ValueError) as exc:
            raise CameraUnavailable(
                tr("{camera!r} is not a camera.", camera=job.get("camera"))
            ) from exc

    raw = str(job.get("source") or "").strip()
    if not raw:
        raise SourceUnreadable(
            tr("No video was chosen to run the model on.")
            if mode == VIDEO
            else tr("No photo was chosen to run the model on."),
            hint=tr("Pick a file in Step 7 and press Start again."),
        )
    path = Path(raw).expanduser()
    if not path.is_file():
        raise SourceUnreadable(
            tr("{path} is not there.", path=path),
            hint=tr("It has been moved or deleted. Choose another file."),
        )
    return path


def _require_runtime(needs, *, fmt: str) -> None:
    """Fail before a frame is read if the format's runtime is not installed.

    Ultralytics would otherwise notice partway through the first inference and
    pip-install it into the managed environment -- slow, surprising, and
    impossible behind a strict proxy. Finding out now costs a fraction of a
    second and names the package.
    """
    import importlib.util

    missing = []
    for name in needs:
        try:
            if importlib.util.find_spec(str(name)) is None:
                missing.append(str(name))
        except (ImportError, ValueError):
            missing.append(str(name))
    if not missing:
        return
    listed = ", ".join(tr('"{package}"', package=name) for name in missing)
    if fmt:
        msg = tr(
            "Running a {format} file needs {packages}, which is not installed in "
            "this environment.",
            format=fmt,
            packages=listed,
        )
    else:
        msg = tr(
            "Running a model in this file needs {packages}, which is not installed "
            "in this environment.",
            packages=listed,
        )
    raise InferenceFailed(
        msg,
        hint=tr(
            "If this is ONNX or TensorRT, open Step 1 and press Repair. The other "
            "formats need runtimes this app does not install, so export the model to "
            "ONNX in Step 6 and test that instead."
        ),
    )


def _resolve_device(requested: str, *, fmt: str, channel: Channel):
    """Which device to run on, refusing rather than quietly substituting.

    Falling back from the GPU to the processor would be the friendly-looking
    thing to do and would ruin the measurement: this step's whole output is a
    number of milliseconds, and one measured on a device the user did not
    choose is worse than an error. So a GPU that is not there is an error.
    """
    if requested == CPU:
        if fmt == "engine":
            raise InferenceFailed(
                tr(
                    "A TensorRT engine is compiled machine code for a graphics card and "
                    "has no processor version."
                ),
                hint=tr("Choose the GPU, or test an ONNX export to see CPU speeds."),
            )
        return CPU

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the entry point explains this
        raise GpuUnavailable(tr("PyTorch could not be imported: {error}", error=exc)) from exc

    if not torch.cuda.is_available():
        raise GpuUnavailable(
            tr("The GPU was chosen and PyTorch cannot see one."),
            hint=tr(
                "Choose CPU to test the model anyway, or go back to Step 1 and let "
                "the app set up its environment again."
            ),
        )
    if fmt == "onnx":
        _check_onnx_provider(channel)
    return 0


def _check_onnx_provider(channel: Channel) -> None:
    """Say so when onnxruntime is about to use the processor regardless.

    This is the quiet failure of the whole step. Asked for a GPU it does not
    have a provider for, onnxruntime does not complain -- it runs on the
    processor and returns the right boxes at a fifth of the speed, and the
    number on screen is then a CPU timing under a GPU heading.
    """
    try:
        import onnxruntime
        providers = list(onnxruntime.get_available_providers())
    except Exception:  # noqa: BLE001 - reported by the import check already
        return
    if not any("CUDA" in name or "TensorRT" in name for name in providers):
        channel.warning(
            ONNX_ON_CPU_WARNING,
            tr(
                "onnxruntime has no CUDA provider in this environment, so this ONNX "
                "model runs on the processor even though the GPU was chosen. The "
                "timings below are processor timings."
            ),
            hint=tr(
                "Open Step 1 and press Repair, which reinstalls onnxruntime-gpu. A "
                "TensorRT export in Step 6 is the faster answer on this machine."
            ),
        )


def load_yolo(model_id: str, *, channel: Channel | None = None):
    """Load an exported model. Replaced wholesale in tests, as in ``export.py``."""
    from ultralytics import YOLO

    try:
        # ``task`` explicitly: an exported file's metadata usually carries it,
        # and a file that lost it would otherwise be guessed at by extension.
        return YOLO(model_id, task="detect")
    except Exception as exc:  # noqa: BLE001 - a bad file arrives as many types
        raise ModelUnavailable(
            tr("{name} could not be loaded: {error}", name=Path(model_id).name, error=exc),
            hint=tr(
                "It may be from an incompatible Ultralytics release, or -- if it is a "
                "TensorRT engine -- built on another machine. Export it again in Step 6."
            ),
        ) from exc


def _explain(exc: Exception) -> Exception:
    """Turn what a runtime raised into something with a remedy attached."""
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "out of memory" in lowered:
        return InferenceFailed(
            tr("The GPU ran out of memory: {error}", error=exc),
            hint=tr(
                "Close anything else using the card, or test on the CPU. A model "
                "exported at a smaller image size also needs less."
            ),
        )
    if "deserialize" in lowered or "serialization" in lowered:
        return InferenceFailed(
            tr(
                "This engine could not be loaded by the TensorRT on this machine: {error}",
                error=exc,
            ),
            hint=tr(
                "A TensorRT engine only works on the GPU and driver that built it. "
                "Export it again in Step 6, on this machine."
            ),
        )
    return InferenceFailed(text)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _confidence(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.25
    return min(0.95, max(0.01, number))


def _model_hw(value) -> tuple[int, int] | None:
    """The export's recorded ``[height, width]``, or ``None`` if it has none."""
    try:
        height, width = (int(v) for v in list(value)[:2])
    except (TypeError, ValueError):
        return None
    return (height, width) if height > 0 and width > 0 else None


def aspect_padding(model_hw: tuple[int, int], input_hw: tuple[int, int]) -> float:
    """The share of the model's input left as padding when this input is fitted in.

    0 when the two shapes match; 0.44 for a 16:9 frame in a square model.
    """
    model_h, model_w = model_hw
    input_h, input_w = input_hw
    if min(model_h, model_w, input_h, input_w) <= 0:
        return 0.0
    model_ratio = model_w / model_h
    input_ratio = input_w / input_h
    return 1.0 - min(model_ratio, input_ratio) / max(model_ratio, input_ratio)


def _percentile(values, fraction: float) -> float:
    """The slow end, without a dependency.

    Nearest-rank rather than interpolated: with a handful of frames the
    interpolated figure invents a value between two real measurements, and this
    number exists to answer "how slow did it actually get".
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(fraction * len(ordered)))))
    return ordered[rank - 1]
