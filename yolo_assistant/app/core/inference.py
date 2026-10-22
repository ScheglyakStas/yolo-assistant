"""What Step 7 can run, where it can run it, and on what.

Testing is the shortest step and the only one whose output is a picture rather
than a number, and both of those shape this module.

**What can be written and what can be run are different questions.** Step 6
offers nine formats because Ultralytics can produce nine. Producing one needs
an exporter; running one needs a *runtime* -- a TensorFlow for a SavedModel, an
ncnn for an ncnn folder, an Apple device for a Core ML bundle -- and the app
installs the runtimes for exactly two of those nine. So there is a second table
here, saying what each format needs in order to execute and which devices it
can execute on, and a test asserts that every format Step 6 offers has an entry
in it.

**The refusals that matter here are about the device.** The specification asks
for a CPU/GPU choice, and for three of the formats one of those two answers is
not merely slower but impossible: a TensorRT engine is compiled machine code
for a particular card and has no CPU path at all, and OpenVINO's "GPU" means an
Intel one, which is not the device this app's GPU button refers to. Those are
refused as they are picked, in the page, exactly as Step 6 refuses an INT8 ONNX.

**A test is not a benchmark, and the first frame is not a frame.** The first
call into any of these runtimes pays for a session, a CUDA context and -- for
TensorRT -- deserialising an engine, which together run to seconds. Averaging
that into "inference time per frame" would misreport a fast model as a slow one
on exactly the screen someone uses to decide whether it is fast enough. The
worker measures it separately and says so; nothing in this module ever folds it
in.

**What is produced is temporary unless the user keeps it.** A photo or a
video leaves a copy with the boxes drawn on, in the project's ``temp/`` folder --
the video's is how it gets played back at the speed it was filmed rather than
the speed the GPU got through it, and either can be kept with Save As. A copy
nobody kept is deleted as soon as it stops being the result on screen: when
another file is chosen, another run starts, or the app closes. Anything saved
outside ``temp/`` is the user's and is never touched. A camera leaves nothing.
None of that is a record of the step having been done, so its completion is
still a fact about this session; it is also the last step, so nothing is gated
on that answer.

Standard library only, and no Qt.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import export as export_model
from .dataset import IMAGE_SUFFIXES
from .export import FORMATS_BY_KEY, ExportInfo
from .i18n import N_, tr
from .project import Project, read_classes

#: Bump when the settings file's shape changes incompatibly.
TEST_SCHEMA = 1

SETTINGS_FILENAME = "test.json"

#: Where the annotated copies of photos and videos go, under the project root.
#: Everything in it is deleted by the app; see the module docstring.
TEMP_DIRNAME = "temp"

#: The warning code the worker sends when onnxruntime has no CUDA provider and
#: an ONNX test asked for the GPU runs on the processor. ``worker/commands/infer.py``
#: sends the same string.
ONNX_ON_CPU_WARNING = "onnx_on_cpu"

#: The warning code the worker sends when the input's aspect ratio is not the
#: exported model's, which wastes time on padding every frame.
#: ``worker/commands/infer.py`` sends the same string.
ASPECT_MISMATCH_WARNING = "aspect_mismatch"


# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #

CPU = "cpu"
GPU = "gpu"


@dataclass(frozen=True)
class Device:
    key: str
    label: str
    note: str

    @property
    def caption(self) -> str:
        return self.label


#: GPU first, because it is the answer for anyone who has one, and because the
#: interesting comparison -- "do I need a graphics card in the machine this
#: ships on?" -- is only worth making after seeing the fast number.
DEVICES: tuple[Device, ...] = (
    Device(
        GPU,
        "GPU",
        N_(
            "The NVIDIA card in this machine. Usually five to twenty times faster, "
            "and what a TensorRT engine requires."
        ),
    ),
    Device(
        CPU,
        "CPU",
        N_(
            "The processor. Slower, and the honest figure for a deployment machine "
            "with no graphics card."
        ),
    ),
)

DEVICES_BY_KEY = {device.key: device for device in DEVICES}

DEFAULT_DEVICE = GPU


# --------------------------------------------------------------------------- #
# What each format needs in order to run
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Runtime:
    """What it takes to execute one of Step 6's formats on this machine."""

    devices: tuple[str, ...]
    """Which of :data:`DEVICES` this format can run on at all."""

    needs: tuple[str, ...] = ()
    """Modules that must import before it can run.

    Checked by the worker before a frame is read, so that a missing runtime is
    reported as "this format needs TensorFlow" rather than as a failure in the
    middle of the first inference.
    """

    provided: bool = True
    """Whether Step 1 installs :attr:`needs`.

    ``False`` is not a refusal. The file is a perfectly good export and the user
    may well have the runtime installed for their own reasons; it is a warning
    that if it fails, the failure is about this machine rather than about the
    model. Ultralytics would otherwise pip-install the missing runtime partway
    through the first frame -- see ``requirements/heavy.txt`` for why that is
    not something this app lets happen.
    """

    note: str = ""


#: Keyed by Step 6's format key. Every one of them appears; a format that can be
#: written and not run says so here rather than by failing later.
RUNTIMES: dict[str, Runtime] = {
    "onnx": Runtime(
        (GPU, CPU),
        needs=("onnxruntime",),
        note=N_("Runs on both, and is the format to test first when anything else misbehaves."),
    ),
    "engine": Runtime(
        (GPU,),
        needs=("tensorrt",),
        note=N_(
            "Compiled for the exact GPU and driver that built it. There is no CPU "
            "version of a TensorRT engine."
        ),
    ),
    "torchscript": Runtime(
        (GPU, CPU),
        note=N_("PyTorch itself runs it, so it works wherever this app does."),
    ),
    "openvino": Runtime(
        (CPU,),
        needs=("openvino",),
        provided=False,
        note=N_(
            "Intel's runtime. Its own 'GPU' means Intel integrated graphics, not the "
            "NVIDIA card this app's GPU button refers to, so it is tested on the CPU here."
        ),
    ),
    "tflite": Runtime(
        (CPU,),
        needs=("tensorflow",),
        provided=False,
        note=N_("Built for phones and small boards. On a desktop it needs TensorFlow to run."),
    ),
    "saved_model": Runtime(
        (CPU,),
        needs=("tensorflow",),
        provided=False,
        note=N_("Needs TensorFlow, which this app does not install."),
    ),
    "pb": Runtime(
        (CPU,),
        needs=("tensorflow",),
        provided=False,
        note=N_("Needs TensorFlow, which this app does not install."),
    ),
    "ncnn": Runtime(
        (CPU,),
        needs=("ncnn",),
        provided=False,
        note=N_("Built for ARM phones. Running it here needs the ncnn package."),
    ),
    "coreml": Runtime(
        (CPU,),
        needs=("coremltools",),
        provided=False,
        note=N_("Apple's format. It can only be run on an Apple device."),
    ),
}


def runtime_for(fmt: str) -> Runtime | None:
    return RUNTIMES.get(str(fmt))


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

PHOTO = "photo"
VIDEO = "video"
CAMERA = "camera"


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    note: str

    @property
    def caption(self) -> str:
        return tr(self.label)


MODES: tuple[Mode, ...] = (
    Mode(
        PHOTO,
        N_("Photo"),
        N_(
            "One still image, run once. The quickest way to find out whether the "
            "exported file works at all."
        ),
    ),
    Mode(
        VIDEO,
        N_("Video"),
        N_(
            "A video file, frame by frame. The boxes are drawn onto a temporary copy "
            "of it, which plays at the video's own speed once every frame has been run."
        ),
    ),
    Mode(
        CAMERA,
        N_("Camera"),
        N_("A camera attached to this machine, live, until you stop it."),
    ),
)

MODES_BY_KEY = {mode.key: mode for mode in MODES}

DEFAULT_MODE = PHOTO

#: Photo suffixes are the dataset's, so that anything Step 2 would have accepted
#: can also be tested.
PHOTO_SUFFIXES = frozenset(IMAGE_SUFFIXES)

VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}
)

#: Photo formats the annotated copy can be written in, keeping the source's.
#: Anything else -- a format Step 2 accepts and OpenCV cannot write -- becomes
#: a PNG, which loses nothing.
WRITABLE_PHOTO_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
FALLBACK_PHOTO_SUFFIX = ".png"


def file_filter(mode: str) -> str:
    """A Qt file-dialog filter for the mode, built from the lists above."""
    if mode == VIDEO:
        patterns = " ".join(f"*{suffix}" for suffix in sorted(VIDEO_SUFFIXES))
        return tr("Videos ({patterns});;All files (*)", patterns=patterns)
    patterns = " ".join(f"*{suffix}" for suffix in sorted(PHOTO_SUFFIXES))
    return tr("Photos ({patterns});;All files (*)", patterns=patterns)


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Camera:
    """One camera the worker managed to open, as it described itself.

    The size and rate are the ones the worker was *granted* when it asked for
    the mode a run uses by default, not the ones the camera would have chosen
    for itself -- so what the list says is what the test will do unless another
    size or rate is chosen. A rate of 0 is one the driver would not say.
    """

    index: int
    name: str = ""
    width: int = 0
    height: int = 0
    fps: int = 0

    @property
    def label(self) -> str:
        return self.name or tr("Camera {index}", index=self.index)

    def caption(self) -> str:
        name = self.label
        if not (self.width and self.height):
            return name
        rate = f", {self.fps} fps" if self.fps else ""
        return f"{name} ({self.width}x{self.height}{rate})"


#: The sizes a camera can be asked for. There is no portable way to ask a
#: camera which modes it has -- see ``worker/commands/cameras.py`` -- so these
#: are the common ones, and the driver rounds to its nearest; the run says what
#: it was actually given. 1920x1080 is the default for the reason given at
#: ``CAMERA_TARGET_SIZE`` in ``worker/commands/infer.py``, which holds the same
#: value.
CAMERA_SIZES = (
    (640, 480),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
)
DEFAULT_CAMERA_SIZE = (1920, 1080)

#: Frame rates a camera can be run at. 0 is "whatever the camera runs at":
#: nothing is asked of it and nothing is paced. Any other is both asked of the
#: driver and enforced by the worker, because many drivers accept a rate and
#: ignore it.
CAMERA_RATES = (0, 5, 10, 15, 24, 30, 60)
DEFAULT_CAMERA_FPS = 0


def size_caption(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}"


def rate_caption(fps: int) -> str:
    return f"{fps} fps" if fps else tr("As the camera runs")


def cameras_from(payload: dict | None) -> tuple[Camera, ...]:
    """Read the camera list out of the scan job's result."""
    found: list[Camera] = []
    for entry in (payload or {}).get("cameras") or ():
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        found.append(
            Camera(
                index=index,
                name=str(entry.get("name") or ""),
                width=_int(entry.get("width")),
                height=_int(entry.get("height")),
                fps=_int(entry.get("fps")),
            )
        )
    return tuple(found)


# --------------------------------------------------------------------------- #
# What there is to test
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Target:
    """One exported file, paired with what it takes to run it."""

    export: ExportInfo
    runtime: Runtime

    @property
    def key(self) -> str:
        return self.export.name

    @property
    def path(self) -> Path:
        return self.export.path

    def supports(self, device: str) -> bool:
        return device in self.runtime.devices

    def caption(self) -> str:
        return self.export.caption()


def targets(project: Project | None, exports=None) -> tuple[Target, ...]:
    """Everything in ``exports/`` that this module knows how to run.

    Read off the filesystem on every visit, like the runs and exports lists are,
    because the exports folder is a folder: users delete from it, drop files
    into it, and expect the app to have noticed.
    """
    found: list[Target] = []
    listing = export_model.list_exports(project) if exports is None else tuple(exports)
    for info in listing:
        runtime = RUNTIMES.get(info.format)
        if runtime is not None:
            found.append(Target(info, runtime))
    return tuple(found)


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #

#: Ultralytics' own default, kept rather than lowered. A test run at a threshold
#: the deployment will not use is a test of something else.
DEFAULT_CONFIDENCE = 0.25
MIN_CONFIDENCE = 0.01
MAX_CONFIDENCE = 0.95


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TestPlan:
    """What pressing Start would do, in the sentences the page shows."""

    targets: tuple[Target, ...] = ()
    target_key: str = ""
    device: str = DEFAULT_DEVICE
    mode: str = DEFAULT_MODE
    source: str = ""
    camera_index: int = -1
    cameras: tuple[Camera, ...] = ()
    camera_size: tuple[int, int] = DEFAULT_CAMERA_SIZE
    camera_fps: int = DEFAULT_CAMERA_FPS
    confidence: float = DEFAULT_CONFIDENCE
    names: tuple[str, ...] = ()

    # -- what was chosen --------------------------------------------------- #

    @property
    def target(self) -> Target | None:
        for candidate in self.targets:
            if candidate.key == self.target_key:
                return candidate
        return self.targets[0] if self.targets else None

    @property
    def camera(self) -> Camera | None:
        for candidate in self.cameras:
            if candidate.index == self.camera_index:
                return candidate
        return self.cameras[0] if self.cameras else None

    @property
    def source_path(self) -> Path | None:
        text = str(self.source).strip()
        return Path(text).expanduser() if text else None

    @property
    def wants_file(self) -> bool:
        return self.mode in (PHOTO, VIDEO)

    @property
    def suffixes(self) -> frozenset[str]:
        return VIDEO_SUFFIXES if self.mode == VIDEO else PHOTO_SUFFIXES

    # -- verdicts ---------------------------------------------------------- #

    def problems(self) -> list[str]:
        """Why Start is disabled, phrased for the person who has to fix it."""
        if not self.targets:
            return [
                tr(
                    "There is nothing to test yet. Export a model in Step 6, or copy an "
                    "exported file into the project's exports folder."
                )
            ]

        found: list[str] = []
        target = self.target
        if target is not None and not target.supports(self.device):
            found.append(self._wrong_device(target))

        if self.mode not in MODES_BY_KEY:
            found.append(tr("Choose what to test the model on."))
        elif self.wants_file:
            found.extend(self._source_problems())
        elif not self.cameras:
            found.append(
                tr(
                    "No camera was found. Connect one and press Refresh; a camera another "
                    "application already has open cannot be used here."
                )
            )

        if not MIN_CONFIDENCE <= float(self.confidence) <= MAX_CONFIDENCE:
            found.append(
                tr(
                    "The confidence threshold has to be between {low:g} and {high:g}.",
                    low=MIN_CONFIDENCE,
                    high=MAX_CONFIDENCE,
                )
            )
        return found

    def _wrong_device(self, target: Target) -> str:
        """Why this file cannot run where the user pointed it.

        Named per format rather than generically: "TensorRT cannot run on the
        processor" is a fact someone can act on, and "unsupported device" is not.
        """
        fmt = FORMATS_BY_KEY.get(target.export.format)
        label = fmt.label if fmt is not None else target.export.format
        can = ", ".join(
            DEVICES_BY_KEY[key].label for key in DEVICES_BY_KEY if key in target.runtime.devices
        )
        chosen = DEVICES_BY_KEY.get(self.device)
        chosen_label = chosen.label if chosen is not None else self.device
        detail = f" {tr(target.runtime.note)}" if target.runtime.note else ""
        return (
            tr(
                "{format} cannot run on the {device}. It runs on: {devices}.",
                format=label,
                device=chosen_label,
                devices=can,
            )
            + detail
        )

    def _source_problems(self) -> list[str]:
        path = self.source_path
        video = self.mode == VIDEO
        if path is None:
            if video:
                return [tr("Choose a video to run the model on.")]
            return [tr("Choose a photo to run the model on.")]
        if not path.exists():
            if video:
                return [tr("{path} is not there any more. Choose another video.", path=path)]
            return [tr("{path} is not there any more. Choose another photo.", path=path)]
        if path.is_dir():
            if video:
                return [tr("{name} is a folder. Choose a single video file.", name=path.name)]
            return [tr("{name} is a folder. Choose a single photo file.", name=path.name)]
        if path.suffix.lower() not in self.suffixes:
            expected = ", ".join(sorted(self.suffixes))
            if video:
                return [
                    tr(
                        '"{name}" is not a video this app recognises. '
                        "Expected one of: {expected}.",
                        name=path.name,
                        expected=expected,
                    )
                ]
            return [
                tr(
                    '"{name}" is not a photo this app recognises. '
                    "Expected one of: {expected}.",
                    name=path.name,
                    expected=expected,
                )
            ]
        return []

    def warnings(self) -> list[str]:
        """Things worth saying that are not reasons to refuse."""
        found: list[str] = []
        target = self.target
        if target is None:
            return found

        if not target.runtime.provided and target.runtime.needs:
            names = _and_list(
                [tr('"{package}"', package=name) for name in target.runtime.needs]
            )
            fmt = FORMATS_BY_KEY.get(target.export.format)
            label = fmt.label if fmt is not None else target.export.format
            found.append(
                tr(
                    "Running {format} needs {packages}, which this app does not install. "
                    "If the test fails to start, that is this machine rather than your model.",
                    format=label,
                    packages=names,
                )
            )

        if target.export.format == "engine":
            found.append(
                tr(
                    "A TensorRT engine only runs on the GPU and driver that built it. If "
                    "this one was exported elsewhere, it will not load."
                )
            )

        if self.device == GPU and target.export.format == "onnx":
            # The quiet one. onnxruntime falls back to the CPU rather than
            # failing when its CUDA provider is missing, so a "GPU" test can
            # silently report CPU timings. The worker checks the providers it
            # actually got and says so; this is the warning before the fact.
            found.append(
                tr(
                    "ONNX on the GPU needs onnxruntime's CUDA provider. If it is missing, "
                    "onnxruntime quietly uses the processor instead -- the test says which "
                    "it ended up on."
                )
            )

        if self.mode == CAMERA:
            found.append(
                tr("The camera runs until you press Stop. Nothing is recorded or saved.")
            )
        return found

    @property
    def is_runnable(self) -> bool:
        return not self.problems()

    def summary(self) -> str:
        target = self.target
        if target is None:
            return tr("Nothing to test yet.")
        device = DEVICES_BY_KEY.get(self.device)
        values = {
            "model": target.export.name,
            "device": device.label if device else self.device,
        }
        if self.mode == CAMERA:
            camera = self.camera
            if camera is None:
                return tr("{model} on the {device} against a camera", **values)
            # The chosen mode rather than the scanned one: the list describes
            # the camera at its default, and this describes the run.
            asked = size_caption(self.camera_size)
            if self.camera_fps:
                asked = f"{asked}, {rate_caption(self.camera_fps)}"
            return tr(
                "{model} on the {device} against {source}",
                source=f"{camera.label} ({asked})",
                **values,
            )
        if self.source_path is not None:
            return tr(
                "{model} on the {device} against {source}",
                source=self.source_path.name,
                **values,
            )
        if self.mode == VIDEO:
            return tr("{model} on the {device} -- choose a video", **values)
        if self.mode == PHOTO:
            return tr("{model} on the {device} -- choose a photo", **values)
        return tr("{model} on the {device}", **values)

    def output_path(self, project: Project) -> Path | None:
        """Where a photo or video run writes its annotated copy; ``None`` for a camera.

        Named for the source and the model both, so that the name Save As
        offers already says what the file is.
        """
        target = self.target
        source = self.source_path
        if not self.wants_file or target is None or source is None:
            return None
        if self.mode == VIDEO:
            suffix = ".mp4"
        else:
            suffix = source.suffix.lower()
            if suffix not in WRITABLE_PHOTO_SUFFIXES:
                suffix = FALLBACK_PHOTO_SUFFIX
        return temp_dir(project) / f"{source.stem}__{target.path.stem}{suffix}"

    def job(self, project: Project) -> dict:
        """The job payload the worker is given."""
        target = self.target
        camera = self.camera
        output = self.output_path(project)
        return {
            "project": str(project.root),
            "model": str(target.path) if target is not None else "",
            "name": target.export.name if target is not None else "",
            "format": target.export.format if target is not None else "",
            "needs": list(target.runtime.needs) if target is not None else [],
            "imgsz_hw": list(target.export.imgsz) if target is not None else [0, 0],
            "device": self.device,
            "mode": self.mode,
            "source": str(self.source_path) if self.source_path is not None else "",
            "output": str(output) if output is not None else "",
            "camera": int(camera.index) if camera is not None else 0,
            "camera_size": [int(value) for value in self.camera_size],
            "camera_fps": int(self.camera_fps),
            "conf": round(float(self.confidence), 3),
            # Only some formats carry their class names. The dataset's are sent
            # as a fallback so that a box never has to be labelled "class 2"
            # when the project knows perfectly well it is a cat.
            "names": list(self.names),
            "schema": TEST_SCHEMA,
        }


def build_plan(
    *,
    project: Project | None = None,
    exports=None,
    target_key: str = "",
    device: str = DEFAULT_DEVICE,
    mode: str = DEFAULT_MODE,
    source: str = "",
    camera_index: int = -1,
    cameras: tuple[Camera, ...] = (),
    camera_size: tuple[int, int] = DEFAULT_CAMERA_SIZE,
    camera_fps: int = DEFAULT_CAMERA_FPS,
    confidence: float = DEFAULT_CONFIDENCE,
) -> TestPlan:
    return TestPlan(
        targets=targets(project, exports),
        target_key=target_key,
        device=device,
        mode=mode,
        source=source,
        camera_index=camera_index,
        cameras=tuple(cameras),
        camera_size=_camera_size(camera_size),
        camera_fps=_camera_fps(camera_fps),
        confidence=confidence,
        names=tuple(read_classes(project)) if project is not None else (),
    )


# --------------------------------------------------------------------------- #
# The temporary copies, and keeping one
# --------------------------------------------------------------------------- #


def temp_dir(project: Project) -> Path:
    return project.root / TEMP_DIRNAME


def is_temporary(path: Path | str, project: Project | None) -> bool:
    """Whether ``path`` is inside the project's ``temp/`` folder, and so the app's to delete."""
    if project is None or not str(path).strip():
        return False
    try:
        resolved = Path(path).expanduser().resolve()
        return resolved.is_relative_to(temp_dir(project).resolve())
    except (OSError, ValueError):
        return False


def discard(path: Path | str | None, project: Project | None) -> bool:
    """Delete an annotated copy, if it is a temporary one. Returns whether it went.

    The check is the whole point of this function: a path that arrived from a
    worker payload is deleted only if it is inside ``temp/``, so a file the
    user saved elsewhere cannot be lost to it whatever the payload said.
    """
    if path is None or not is_temporary(path, project):
        return False
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return False
    except OSError:
        # Still open somewhere -- a video player on Windows, say. The next
        # sweep gets it.
        return False
    return True


def sweep_temp(project: Project | None) -> int:
    """Empty ``temp/`` of files left behind by a session that did not close cleanly.

    Only files directly inside it, which is all the worker ever writes there.
    Returns how many went.
    """
    if project is None:
        return 0
    folder = temp_dir(project)
    removed = 0
    try:
        entries = list(folder.iterdir())
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_file():
                entry.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def save_copy(source: Path | str, destination: Path | str) -> Path:
    """Copy an annotated file to where the user asked, and return where it went.

    A destination typed without an extension gets the source's, because a video
    saved as ``result`` is a file nothing will open. Raises ``OSError`` for the
    page to report.
    """
    source = Path(source)
    destination = Path(destination).expanduser()
    if not destination.suffix:
        destination = destination.with_suffix(source.suffix)
    if destination.resolve() == source.resolve():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def save_filter(path: Path | str) -> str:
    """A Save As dialog filter that offers the copy's own format first."""
    suffix = Path(path).suffix.lower()
    if not suffix:
        return tr("All files (*)")
    if suffix in VIDEO_SUFFIXES:
        return tr("Video (*{suffix});;All files (*)", suffix=suffix)
    return tr("Image (*{suffix});;All files (*)", suffix=suffix)


# --------------------------------------------------------------------------- #
# Remembering what was asked for
# --------------------------------------------------------------------------- #


def settings_path(project: Project) -> Path:
    return project.internal_dir / SETTINGS_FILENAME


def read_settings(project: Project | None) -> dict:
    """What was chosen here last time, with defaults filled in.

    The source file is remembered by path. A path that has since been deleted is
    still offered rather than silently cleared: "that file is not there any
    more" is more use than an empty box, and the plan already says it.
    """
    record = _read_json(settings_path(project)) if project is not None else None
    record = record if isinstance(record, dict) else {}

    device = str(record.get("device") or DEFAULT_DEVICE)
    if device not in DEVICES_BY_KEY:
        device = DEFAULT_DEVICE
    mode = str(record.get("mode") or DEFAULT_MODE)
    if mode not in MODES_BY_KEY:
        mode = DEFAULT_MODE

    confidence = record.get("conf", DEFAULT_CONFIDENCE)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = DEFAULT_CONFIDENCE
    confidence = min(MAX_CONFIDENCE, max(MIN_CONFIDENCE, confidence))

    return {
        "model": str(record.get("model") or ""),
        "device": device,
        "mode": mode,
        "photo": str(record.get("photo") or ""),
        "video": str(record.get("video") or ""),
        "camera": _int(record.get("camera"), -1),
        "camera_size": _camera_size(record.get("camera_size")),
        "camera_fps": _camera_fps(record.get("camera_fps")),
        "conf": confidence,
    }


def write_settings(project: Project, plan: TestPlan, *, previous: dict | None = None) -> None:
    """Record what was chosen.

    The photo and the video are kept apart, because switching modes and
    switching files are different acts: someone who tests a photo, then a video,
    then comes back should find both where they left them.
    """
    record = dict(previous or {})
    target = plan.target
    record.update(
        {
            "schema": TEST_SCHEMA,
            "model": target.export.name if target is not None else "",
            "device": plan.device,
            "mode": plan.mode,
            "camera": int(plan.camera_index),
            "camera_size": [int(value) for value in plan.camera_size],
            "camera_fps": int(plan.camera_fps),
            "conf": round(float(plan.confidence), 3),
            "updated_at": time.time(),
        }
    )
    if plan.mode in (PHOTO, VIDEO):
        record[plan.mode] = str(plan.source or "")

    path = settings_path(project)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # A project on a read-only share still tests; it just forgets.
        pass


def _and_list(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(items) <= 1:
        return "".join(items)
    return tr("{items} and {last}", items=", ".join(items[:-1]), last=items[-1])


def _camera_size(value) -> tuple[int, int]:
    """One of :data:`CAMERA_SIZES`, or the default for anything else."""
    try:
        size = tuple(int(part) for part in value)
    except (TypeError, ValueError):
        return DEFAULT_CAMERA_SIZE
    return size if size in CAMERA_SIZES else DEFAULT_CAMERA_SIZE


def _camera_fps(value) -> int:
    """One of :data:`CAMERA_RATES`, or the default for anything else."""
    fps = _int(value, DEFAULT_CAMERA_FPS)
    return fps if fps in CAMERA_RATES else DEFAULT_CAMERA_FPS


def _int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None
