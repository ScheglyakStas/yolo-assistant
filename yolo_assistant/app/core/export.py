"""What Step 6 is about to convert, worked out before anything is loaded.

Export is short next to training -- seconds for ONNX, minutes for a TensorRT
engine -- but it is the step where a mistake is hardest to see. An export that
"worked" can still be a file that runs nowhere useful, or one that quietly lost
several points of accuracy, and neither announces itself. So the same shape as
``training.py``: everything that can be decided without torch is decided here,
in the GUI process, and the page renders what this module concludes.

Four decisions are worth a note.

**A checkpoint is chosen, not a run.** Every run holds two: ``best.pt`` is the
epoch with the highest mAP50-95, ``last.pt`` is where the run stopped. They are
usually the same file's worth of accuracy and sometimes very much not, and the
user has to be able to say which. So the selector lists checkpoints rather than
folders, and each one says what it is.

**The output name is derived, never typed.** ``bottles_best_fp32.onnx`` says
which run, which checkpoint and what precision, which is exactly what someone
staring at an ``exports/`` folder six weeks later needs to know. A name the user
invents carries none of that, and a name that collides silently replaces an
hour of engine building.

**Precision is refused here when it is impossible and reported by the worker
when it is merely wrong.** Some combinations cannot exist -- Ultralytics' ONNX
exporter has no INT8 path at all -- and those are refused as the user picks
them. But whether *this* checkpoint survives *this* precision is a question
only the exporter can answer, and a model trained with INT8 QAT is the case the
specification names: it fails from inside the export, and the worker turns that
into "choose INT8" rather than a traceback.

**Image size really is a pair here.** Training takes one number because
Ultralytics trains on squares; export does not, and a model that will see
1920x1080 frames does not want a square input -- a square one spends almost
half of every frame's work on the padding around it. So the default is 16:9
with the long side at the training size (1024 trained, 1024x576 exported), and
neither side may exceed the training size: the model has never seen anything
larger, and a bigger input buys time spent, not accuracy.

Standard library only, and no Qt: the page renders what this module decides and
the worker is handed the same numbers.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .i18n import N_, tr
from .project import Project
from .training import (
    DEFAULT_IMGSZ,
    MAX_IMGSZ,
    MIN_IMGSZ,
    STRIDE,
    RunInfo,
    list_runs,
    nearest_on_grid,
)

#: Bump when the per-export record's shape changes incompatibly.
EXPORT_SCHEMA = 1

#: Written beside every exported artifact: what it was made from and how. The
#: exported file itself records none of that -- an ``.onnx`` does not know which
#: run it came from or what precision was asked for -- and Step 7 needs both to
#: describe what it is about to test.
RECORD_SUFFIX = ".umla.json"

#: The app's copy of the last settings used, so that coming back tomorrow does
#: not mean re-picking them. Like ``train.json`` it records what was asked for,
#: never that anything finished.
SETTINGS_FILENAME = "export.json"

BEST = "best"
LAST = "last"


# --------------------------------------------------------------------------- #
# Precision
# --------------------------------------------------------------------------- #

FP32 = "fp32"
FP16 = "fp16"
INT8 = "int8"


@dataclass(frozen=True)
class Precision:
    """One numeric precision the exported model can be written at."""

    key: str
    label: str
    note: str

    @property
    def caption(self) -> str:
        return self.label


#: Widest to narrowest, which is also safest to riskiest. The default is FP32:
#: it is what a plain export produces and the only one that cannot cost
#: accuracy, and a default that silently halves precision is a bad default even
#: when halving it is usually fine.
PRECISIONS: tuple[Precision, ...] = (
    Precision(
        FP32,
        "FP32",
        N_(
            "Full precision. The largest file and the slowest, and the only one that "
            "cannot cost any accuracy at all."
        ),
    ),
    Precision(
        FP16,
        "FP16",
        N_(
            "Half precision. About half the size and usually faster on a GPU, for an "
            "accuracy difference most datasets cannot measure."
        ),
    ),
    Precision(
        INT8,
        "INT8",
        N_(
            "8-bit integers. The smallest and fastest, and the only precision an "
            "INT8 QAT model can be exported at."
        ),
    ),
)

DEFAULT_PRECISION = FP32

PRECISIONS_BY_KEY = {p.key: p for p in PRECISIONS}


# --------------------------------------------------------------------------- #
# Formats
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Format:
    """One deployment format, and what it can and cannot be asked for."""

    key: str
    """What Ultralytics' ``format=`` is given."""

    label: str
    suffix: str
    """What the exported artifact is called: an extension, or a folder name."""

    is_directory: bool = False
    int8: bool = False
    half: bool = False

    carries_qat: bool = False
    """Whether a model trained with INT8 QAT survives this format.

    Quantisation-aware training leaves fake-quantisation nodes in the graph.
    ONNX carries them through and TensorRT reads them and builds an INT8 engine
    from them; every other exporter here either drops them or refuses. A format
    without this flag is not a slower way to deploy a QAT model, it is a way to
    lose the entire point of having trained one.
    """

    needs_gpu: bool = False
    note: str = ""

    @property
    def caption(self) -> str:
        return self.label

    @property
    def detail(self) -> str:
        return tr(self.note)

    def supports(self, precision: str) -> bool:
        if precision == INT8:
            return self.int8
        if precision == FP16:
            return self.half
        return True


#: The formats worth offering, most useful first. Deliberately not every format
#: Ultralytics can emit: each one here is something the app can describe, and a
#: list of twenty names with no guidance is a worse answer than a list of nine.
FORMATS: tuple[Format, ...] = (
    Format(
        "onnx",
        "ONNX",
        ".onnx",
        half=True,
        carries_qat=True,
        note=N_(
            "Runs almost anywhere -- onnxruntime on CPU or GPU, and most other "
            "deployment stacks read it. Start here."
        ),
    ),
    Format(
        "engine",
        "TensorRT",
        ".engine",
        int8=True,
        half=True,
        carries_qat=True,
        needs_gpu=True,
        note=N_(
            "The fastest thing NVIDIA hardware can run, and the least portable: "
            "the file is built for one GPU model and driver and will not load on "
            "another."
        ),
    ),
    Format(
        "openvino",
        "OpenVINO",
        "_openvino_model",
        is_directory=True,
        int8=True,
        half=True,
        note=N_("Intel processors and their built-in graphics."),
    ),
    Format(
        "torchscript",
        "TorchScript",
        ".torchscript",
        note=N_("PyTorch without Python. A good fallback when nothing else loads."),
    ),
    Format(
        "tflite",
        "TensorFlow Lite",
        ".tflite",
        int8=True,
        half=True,
        note=N_("Phones and small boards."),
    ),
    Format(
        "ncnn",
        "NCNN",
        "_ncnn_model",
        is_directory=True,
        half=True,
        note=N_("ARM phones, where it is usually quicker than TensorFlow Lite."),
    ),
    Format(
        "coreml",
        "Core ML",
        ".mlpackage",
        is_directory=True,
        int8=True,
        half=True,
        note=N_("Apple devices, using the Neural Engine."),
    ),
    Format(
        "saved_model",
        "TensorFlow SavedModel",
        "_saved_model",
        is_directory=True,
        int8=True,
        note=N_("TensorFlow Serving, and the input to most TensorFlow tooling."),
    ),
    Format(
        "pb",
        "TensorFlow GraphDef",
        ".pb",
        note=N_("The older frozen-graph format, for stacks that still want one."),
    ),
)

DEFAULT_FORMAT = FORMATS[0].key

FORMATS_BY_KEY = {fmt.key: fmt for fmt in FORMATS}

#: Formats a QAT model can be exported to, named in the refusal so the user is
#: not left to work it out from a list of nine.
QAT_FORMATS = tuple(fmt for fmt in FORMATS if fmt.carries_qat)


#: What the DeepStream toggle adds, exactly as the specification words it.
#: DeepStream does its own clustering after the network, so a model that has
#: already suppressed and merged its boxes gives it less to work with, not more.
DEEPSTREAM_ARGS: dict[str, object] = {"nms": False, "agnostic_nms": True}

#: The toggle is shorthand for "what a DeepStream pipeline expects", and both
#: of those are only meaningful to a graph that has a detection head to
#: configure. Ticking it for a Core ML bundle is not an error, it just does
#: nothing, and saying so is cheaper than letting someone wonder.
DEEPSTREAM_FORMATS = frozenset({"onnx", "engine"})


# --------------------------------------------------------------------------- #
# Extra arguments
# --------------------------------------------------------------------------- #

#: Settings the app decides, which the user therefore cannot pass. ``model``,
#: ``format`` and ``imgsz`` are the form; ``device`` is chosen from what the
#: machine has; ``simplify`` and ``opset`` are fixed by the specification. The
#: precision flags are here under all three spellings the 8.4 line has used,
#: because letting one through would mean two sources of truth for it.
RESERVED_ARGS: frozenset[str] = frozenset(
    {
        "model",
        "format",
        "imgsz",
        "device",
        "half",
        "int8",
        "quantize",
        "simplify",
        "opset",
        "task",
        "mode",
        "project",
        "name",
    }
)

#: Ultralytics' own export argument names, as of the pinned release. Advisory,
#: for the same reason the training list is: Step 1's update button can move the
#: environment to a release that added one, and a stale list must not block a
#: setting that now exists. The authoritative check is in the worker, against
#: the defaults of whichever Ultralytics is installed.
KNOWN_ARGS: frozenset[str] = frozenset(
    """
    keras optimize dynamic batch workspace nms agnostic_nms data fraction
    iou conf max_det verbose
    """.split()
)


@dataclass(frozen=True)
class ExtraArgs:
    """The "additional settings" box, parsed. See ``training.parse_extra_args``."""

    values: dict[str, object] = field(default_factory=dict)
    problems: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.problems


def parse_extra_args(text: str, *, deepstream: bool = False) -> ExtraArgs:
    """``"dynamic=true"`` -> ``{"dynamic": True}``.

    The same parser as Step 5's, with one addition: when the DeepStream toggle
    is on it already sets two settings, and typing either of them again is a
    contradiction the user should be told about rather than one the app
    resolves by picking a winner.
    """
    values: dict[str, object] = {}
    problems: list[str] = []
    unknown: list[str] = []

    for chunk in str(text).split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            problems.append(
                tr(
                    '"{item}" is not a setting. Each one is a name, an equals sign and a '
                    'value, like "dynamic=true".',
                    item=item,
                )
            )
            continue
        raw_key, raw_value = item.split("=", 1)
        key = raw_key.strip()
        value = raw_value.strip()
        if not key:
            problems.append(
                tr('"{item}" has no setting name in front of the equals sign.', item=item)
            )
            continue
        if not _ARG_NAME.fullmatch(key):
            problems.append(tr('"{key}" is not a valid setting name.', key=key))
            continue
        if key in RESERVED_ARGS:
            problems.append(
                tr(
                    '"{key}" is set by the app and cannot be changed here -- it decides '
                    "what is exported, in what format and at what precision.",
                    key=key,
                )
            )
            continue
        if deepstream and key in DEEPSTREAM_ARGS:
            problems.append(
                tr(
                    '"{key}" is already set by the DeepStream toggle. Turn the toggle off '
                    "if you want to set it yourself.",
                    key=key,
                )
            )
            continue
        if key in values:
            problems.append(tr('"{key}" is given twice. Keep only one of them.', key=key))
            continue
        if key not in KNOWN_ARGS:
            unknown.append(key)
        values[key] = _typed(value)

    return ExtraArgs(values=values, problems=tuple(problems), unknown=tuple(unknown))


_ARG_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Deliberately no "1" and no "0"; see ``training._TRUE`` for why.
_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}
_NONE = {"none", "null", ""}


def _typed(value: str):
    lowered = value.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    if lowered in _NONE:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


# --------------------------------------------------------------------------- #
# What there is to export
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Checkpoint:
    """One ``.pt`` file a run left behind, and what it is.

    Runs hold two, and which one someone wants is a real question rather than a
    detail: ``best.pt`` is the epoch that scored highest on the held-back
    photos, ``last.pt`` is wherever the run stopped. On a run that finished and
    then overfitted they are meaningfully different models.
    """

    run: RunInfo
    which: str

    @property
    def key(self) -> str:
        return f"{self.run.name}/{self.which}"

    @property
    def path(self) -> Path:
        """The file itself. Never ``None``: :func:`checkpoints` only offers
        checkpoints whose file is on disk."""
        chosen = self.run.best if self.which == BEST else self.run.last
        return chosen if chosen is not None else self.run.path

    @property
    def qat(self) -> bool:
        return self.run.qat

    def caption(self) -> str:
        name = self.run.name
        if self.which == BEST and self.run.best_fitness:
            detail = tr(
                "{run} - best epoch, mAP50-95 {fitness:.4f}",
                run=name,
                fitness=self.run.best_fitness,
            )
        elif self.which == BEST:
            detail = tr("{run} - best epoch", run=name)
        elif self.run.epochs:
            detail = tr("{run} - last epoch of {epochs}", run=name, epochs=self.run.epochs)
        else:
            detail = tr("{run} - last epoch", run=name)
        if self.run.qat:
            detail += ", INT8 QAT"
        return detail


#: The shape offered by default: most cameras and most video are 16:9, and a
#: model exported at the shape of its input spends nothing on padding.
DEFAULT_ASPECT = (16, 9)


def max_imgsz(run: RunInfo | None) -> int:
    """The largest side an export of this run may have: what it was trained at.

    A run that does not say (weights copied in with no record) is capped only by
    the app's own limit.
    """
    if run is None or run.imgsz <= 0:
        return MAX_IMGSZ
    return min(MAX_IMGSZ, int(run.imgsz))


def default_imgsz(run: RunInfo | None) -> tuple[int, int]:
    """``(height, width)`` to offer for this run: 16:9, long side at the training size.

    The short side is rounded to the stride, so it is near 16:9 rather than
    exactly it at most sizes -- 640 gives 640x352 -- and never larger than the
    long side.
    """
    width = int(run.imgsz) if run is not None and run.imgsz > 0 else DEFAULT_IMGSZ
    width = min(width, MAX_IMGSZ)
    aspect_w, aspect_h = DEFAULT_ASPECT
    height = min(width, nearest_on_grid(width * aspect_h / aspect_w))
    return height, width


def checkpoints(runs: tuple[RunInfo, ...]) -> tuple[Checkpoint, ...]:
    """Every checkpoint that is actually on disk, newest run first.

    Only the files that exist: a run stopped before Ultralytics had chosen a
    best epoch holds ``last.pt`` alone, and offering a ``best`` that is not
    there would be a selector entry that fails when it is used.
    """
    found: list[Checkpoint] = []
    for run in runs:
        if run.best is not None:
            found.append(Checkpoint(run, BEST))
        if run.last is not None:
            found.append(Checkpoint(run, LAST))
    return tuple(found)


# --------------------------------------------------------------------------- #
# What has already been exported
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExportInfo:
    """One artifact in ``exports/``, described by what is there right now."""

    name: str
    path: Path
    format: str = ""
    precision: str = ""
    run: str = ""
    which: str = ""
    imgsz: tuple[int, int] = (0, 0)
    deepstream: bool = False
    size_bytes: int = 0
    modified_at: float = 0.0

    def caption(self) -> str:
        fmt = FORMATS_BY_KEY.get(self.format)
        parts = [self.name]
        if fmt is not None:
            parts.append(fmt.label)
        precision = PRECISIONS_BY_KEY.get(self.precision)
        if precision is not None:
            parts.append(precision.label)
        if self.imgsz != (0, 0):
            parts.append(f"{self.imgsz[1]}x{self.imgsz[0]}")
        if self.size_bytes:
            parts.append(_size(self.size_bytes))
        return " - ".join(parts)


def list_exports(project: Project | None) -> tuple[ExportInfo, ...]:
    """Everything in the project's exports folder, newest first.

    Read off the filesystem every time, for the reason ``list_runs`` is: a file
    the user deleted has to disappear without restarting the app, and one they
    dropped in by hand has to appear. Recognition is by the name an export
    ends with, so a stray note or a zip in that folder is not mistaken for a
    model -- and neither is the sidecar record beside each artifact.
    """
    if project is None:
        return ()
    try:
        entries = list(project.exports_dir.iterdir())
    except OSError:
        return ()

    found: list[ExportInfo] = []
    for entry in entries:
        if entry.name.endswith(RECORD_SUFFIX):
            continue
        fmt = _format_of(entry)
        if fmt is None:
            continue
        found.append(_read_export(entry, fmt))
    found.sort(key=lambda info: (-info.modified_at, info.name.casefold()))
    return tuple(found)


def _format_of(path: Path) -> Format | None:
    """Which format wrote this, judged by its name alone.

    Longest suffix first, because ``.pb`` would otherwise claim a name that
    a longer suffix matches better.
    """
    name = path.name
    is_dir = path.is_dir()
    for fmt in sorted(FORMATS, key=lambda f: -len(f.suffix)):
        if fmt.is_directory != is_dir:
            continue
        if name.endswith(fmt.suffix):
            return fmt
    return None


def _read_export(path: Path, fmt: Format) -> ExportInfo:
    record = _read_json(path.with_name(path.name + RECORD_SUFFIX)) or {}
    imgsz = record.get("imgsz_hw") or []
    try:
        pair = (int(imgsz[0]), int(imgsz[1]))
    except (IndexError, TypeError, ValueError):
        pair = (0, 0)
    try:
        modified = path.stat().st_mtime
    except OSError:
        modified = 0.0
    return ExportInfo(
        name=path.name,
        path=path,
        format=str(record.get("format") or fmt.key),
        precision=str(record.get("precision") or ""),
        run=str(record.get("run") or ""),
        which=str(record.get("which") or ""),
        imgsz=pair,
        deepstream=bool(record.get("deepstream")),
        size_bytes=_size_of(path),
        modified_at=modified,
    )


def has_export(project: Project | None) -> bool:
    """Whether Step 7 has something to test. What "Step 6 is done" means."""
    return bool(list_exports(project))


def _size_of(path: Path) -> int:
    try:
        if path.is_dir():
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return path.stat().st_size
    except OSError:
        return 0


def _size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{size:.0f} {unit}"
            return tr("{size:.1f} {unit}", size=size, unit=unit)
        size /= 1024
    return tr("{size:.1f} {unit}", size=size, unit="GB")


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExportPlan:
    """What pressing Export would do, in numbers and sentences the page shows."""

    checkpoints: tuple[Checkpoint, ...] = ()
    checkpoint_key: str = ""
    format: str = DEFAULT_FORMAT
    precision: str = DEFAULT_PRECISION
    imgsz_h: int = 0
    imgsz_w: int = 0
    deepstream: bool = False
    extra_text: str = ""
    exports: tuple[ExportInfo, ...] = ()

    # -- what was chosen --------------------------------------------------- #

    @property
    def checkpoint(self) -> Checkpoint | None:
        for candidate in self.checkpoints:
            if candidate.key == self.checkpoint_key:
                return candidate
        return self.checkpoints[0] if self.checkpoints else None

    @property
    def fmt(self) -> Format | None:
        return FORMATS_BY_KEY.get(self.format)

    @property
    def extra(self) -> ExtraArgs:
        return parse_extra_args(self.extra_text, deepstream=self.deepstream)

    @property
    def args(self) -> dict[str, object]:
        """Everything the user asked for, the toggle included.

        The toggle's pairs go in first, so that a mistake which let a typed
        ``nms`` through would be visible as the user's value winning rather
        than invisible as the app's.
        """
        values: dict[str, object] = dict(DEEPSTREAM_ARGS) if self.deepstream else {}
        values.update(self.extra.values)
        return values

    @property
    def qat(self) -> bool:
        checkpoint = self.checkpoint
        return checkpoint is not None and checkpoint.qat

    # -- where it lands ---------------------------------------------------- #

    @property
    def output_name(self) -> str:
        """``bottles_best_fp32.onnx``: which run, which epoch, what precision.

        Derived rather than typed. Six weeks later the only thing anyone knows
        about a file in ``exports/`` is its name, and a name the user invented
        on the day carries none of the three facts that matter.
        """
        checkpoint = self.checkpoint
        fmt = self.fmt
        if checkpoint is None or fmt is None:
            return ""
        stem = f"{_safe(checkpoint.run.name)}_{checkpoint.which}_{self.precision}"
        return stem + fmt.suffix

    def output_path(self, project: Project) -> Path:
        return project.exports_dir / self.output_name

    @property
    def existing_export(self) -> ExportInfo | None:
        """The artifact this export would replace, if there is one."""
        name = self.output_name
        if not name:
            return None
        for export in self.exports:
            if export.name == name:
                return export
        return None

    # -- verdicts ---------------------------------------------------------- #

    def problems(self) -> list[str]:
        """Why Export is disabled, phrased for the person who has to fix it."""
        if not self.checkpoints:
            return [
                tr(
                    "There is no trained model in this project yet. Train one in Step 5, "
                    "or copy a run into the project's runs folder."
                )
            ]

        found: list[str] = []
        checkpoint = self.checkpoint
        fmt = self.fmt

        if fmt is None:
            found.append(tr("Choose a format to export to."))
        if self.precision not in PRECISIONS_BY_KEY:
            found.append(tr("Choose a precision to export at."))

        if checkpoint is not None and fmt is not None and self.precision in PRECISIONS_BY_KEY:
            found.extend(self._precision_problems(checkpoint, fmt))

        trained = checkpoint.run.imgsz if checkpoint is not None else 0
        for value, out_of_range, off_grid, too_large in (
            (self.imgsz_w, *_WIDTH_PROBLEMS),
            (self.imgsz_h, *_HEIGHT_PROBLEMS),
        ):
            value = int(value)
            if not MIN_IMGSZ <= value <= MAX_IMGSZ:
                found.append(tr(out_of_range, low=MIN_IMGSZ, high=MAX_IMGSZ))
            elif value % STRIDE:
                found.append(
                    tr(off_grid, stride=STRIDE, value=value, nearest=nearest_on_grid(value))
                )
            elif trained > 0 and value > trained:
                found.append(
                    tr(too_large, run=checkpoint.run.name, trained=trained, value=value)
                )

        found.extend(self.extra.problems)
        return found

    def _precision_problems(self, checkpoint: Checkpoint, fmt: Format) -> list[str]:
        """The combinations that cannot exist, as opposed to the unwise ones."""
        found: list[str] = []
        if not fmt.supports(self.precision):
            label = PRECISIONS_BY_KEY[self.precision].label
            can = ", ".join(
                p.label for p in PRECISIONS if fmt.supports(p.key) and p.key != self.precision
            )
            found.append(
                tr(
                    "{format} cannot be exported at {precision}. It supports {supported}.",
                    format=fmt.label,
                    precision=label,
                    supported=can,
                )
            )

        if checkpoint.qat:
            # The case the specification names. It is refused here as well as
            # caught in the worker, because the app already knows the answer --
            # the run recorded that it was trained for INT8 -- and letting
            # someone wait through a failing export to be told so is a worse
            # way to say it.
            if self.precision != INT8:
                found.append(
                    tr(
                        '"{run}" was trained for INT8, so its quantisers '
                        "are already part of the model. Choose INT8; exporting it at "
                        "{precision} throws the training away "
                        "and usually fails outright.",
                        run=checkpoint.run.name,
                        precision=PRECISIONS_BY_KEY[self.precision].label,
                    )
                )
            if not fmt.carries_qat:
                found.append(
                    tr(
                        "{format} cannot carry an INT8 QAT model's quantisers. "
                        "Export it as {formats}.",
                        format=fmt.label,
                        formats=_either([f.label for f in QAT_FORMATS]),
                    )
                )
        return found

    def warnings(self) -> list[str]:
        """Things worth saying that are not reasons to refuse."""
        found: list[str] = []
        checkpoint = self.checkpoint
        fmt = self.fmt

        existing = self.existing_export
        if existing is not None:
            found.append(
                tr(
                    '"{name}" already exists and will be replaced. Nothing else '
                    "in the exports folder is touched.",
                    name=existing.name,
                )
            )

        if fmt is not None and fmt.key == "engine":
            found.append(
                tr(
                    "A TensorRT engine is built for this exact GPU and driver. Copying it "
                    "to another machine will not work -- export there instead."
                )
            )

        if self.deepstream and fmt is not None and fmt.key not in DEEPSTREAM_FORMATS:
            found.append(
                tr(
                    "The DeepStream settings only mean something to ONNX and TensorRT; "
                    "{format} will ignore them.",
                    format=fmt.label,
                )
            )

        for key in self.extra.unknown:
            found.append(
                tr(
                    '"{key}" is not a setting this version of the app knows about. If your '
                    "Ultralytics does not know it either, the export will refuse it before "
                    "it starts.",
                    key=key,
                )
            )
        return found

    @property
    def is_runnable(self) -> bool:
        return not self.problems()

    def summary(self) -> str:
        checkpoint = self.checkpoint
        fmt = self.fmt
        if checkpoint is None:
            return tr("Nothing to export yet.")
        if fmt is None:
            return tr("{checkpoint} -- choose a format.", checkpoint=checkpoint.caption())
        precision = PRECISIONS_BY_KEY.get(self.precision)
        label = precision.label if precision else self.precision
        return tr(
            "{run} ({which}) to {format}, {precision}, {width}x{height} -> {output}",
            run=checkpoint.run.name,
            which=checkpoint.which,
            format=fmt.label,
            precision=label,
            width=int(self.imgsz_w),
            height=int(self.imgsz_h),
            output=self.output_name,
        )

    def job(self, project: Project) -> dict:
        """The job payload the worker is given."""
        checkpoint = self.checkpoint
        return {
            "project": str(project.root),
            "weights": str(checkpoint.path) if checkpoint else "",
            "run": checkpoint.run.name if checkpoint else "",
            "which": checkpoint.which if checkpoint else "",
            "exports": str(project.exports_dir),
            "output": self.output_name,
            "format": self.format,
            "precision": self.precision,
            "imgsz_hw": [int(self.imgsz_h), int(self.imgsz_w)],
            "deepstream": bool(self.deepstream),
            "qat": self.qat,
            "args": dict(self.args),
            "schema": EXPORT_SCHEMA,
        }


def build_plan(
    *,
    project: Project | None = None,
    runs: tuple[RunInfo, ...] | None = None,
    exports: tuple[ExportInfo, ...] | None = None,
    checkpoint_key: str = "",
    fmt: str = DEFAULT_FORMAT,
    precision: str = DEFAULT_PRECISION,
    imgsz_h: int = 0,
    imgsz_w: int = 0,
    deepstream: bool = False,
    extra_text: str = "",
) -> ExportPlan:
    available = checkpoints(list_runs(project) if runs is None else tuple(runs))
    return ExportPlan(
        checkpoints=available,
        checkpoint_key=checkpoint_key,
        format=fmt,
        precision=precision,
        imgsz_h=imgsz_h,
        imgsz_w=imgsz_w,
        deepstream=deepstream,
        extra_text=extra_text,
        exports=list_exports(project) if exports is None else tuple(exports),
    )


#: The image-size refusals, one set per side, so that each is a whole sentence
#: rather than "Height" or "Width" dropped into a shared one.
_HEIGHT_PROBLEMS = (
    N_("Height has to be between {low} and {high} pixels."),
    N_(
        "Height has to be a multiple of {stride}. {value} is not -- "
        "the nearest that is, is {nearest}."
    ),
    N_(
        '"{run}" was trained at {trained} pixels, so the height can be at most '
        "{trained}; {value} would only make it slower."
    ),
)
_WIDTH_PROBLEMS = (
    N_("Width has to be between {low} and {high} pixels."),
    N_(
        "Width has to be a multiple of {stride}. {value} is not -- "
        "the nearest that is, is {nearest}."
    ),
    N_(
        '"{run}" was trained at {trained} pixels, so the width can be at most '
        "{trained}; {value} would only make it slower."
    ),
)


def _either(labels: list[str]) -> str:
    """``["ONNX", "TensorRT"]`` -> ``"ONNX or TensorRT"``."""
    if len(labels) < 2:
        return "".join(labels)
    return tr("{names} or {last}", names=", ".join(labels[:-1]), last=labels[-1])


_UNSAFE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f\s]+')


def _safe(name: str) -> str:
    """A run's name, made fit for the middle of a filename.

    Run names are already checked to be legal folder names in Step 5, so this
    is only closing the gap between the two: a space is perfectly good in a
    folder and a nuisance in a filename that will be typed into a shell.
    """
    return _UNSAFE_NAME.sub("_", str(name).strip()) or "model"


# --------------------------------------------------------------------------- #
# Remembering what was asked for
# --------------------------------------------------------------------------- #


def settings_path(project: Project) -> Path:
    return project.internal_dir / SETTINGS_FILENAME


def read_settings(project: Project | None) -> dict:
    """The settings last used in this project, with defaults filled in.

    The image size is the one place this differs from Step 5's: there is no
    sensible default until a checkpoint is chosen, because the right answer is
    whatever that run was trained at. Zero means "no preference recorded", and
    the page fills it from the run.
    """
    record = _read_json(settings_path(project)) if project is not None else None
    record = record if isinstance(record, dict) else {}

    fmt = str(record.get("format") or DEFAULT_FORMAT)
    if fmt not in FORMATS_BY_KEY:
        fmt = DEFAULT_FORMAT
    precision = str(record.get("precision") or DEFAULT_PRECISION)
    if precision not in PRECISIONS_BY_KEY:
        precision = DEFAULT_PRECISION
    return {
        "format": fmt,
        "precision": precision,
        "imgsz_h": _clamped_int(record.get("imgsz_h"), 0, 0, MAX_IMGSZ),
        "imgsz_w": _clamped_int(record.get("imgsz_w"), 0, 0, MAX_IMGSZ),
        "deepstream": bool(record.get("deepstream", False)),
        "args": str(record.get("args") or ""),
    }


def write_settings(project: Project, plan: ExportPlan) -> None:
    payload = {
        "schema": EXPORT_SCHEMA,
        "format": plan.format,
        "precision": plan.precision,
        "imgsz_h": int(plan.imgsz_h),
        "imgsz_w": int(plan.imgsz_w),
        "deepstream": bool(plan.deepstream),
        "args": plan.extra_text,
        "updated_at": time.time(),
    }
    path = settings_path(project)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # A project on a read-only share still exports; it just forgets.
        pass


def _clamped_int(value, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return min(high, max(low, number))


def _read_json(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None
