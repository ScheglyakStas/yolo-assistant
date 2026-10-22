"""What Step 5 is about to run, worked out before anything is loaded.

Training is the step that costs hours rather than minutes, and the one where a
mistake is most expensive to discover late. So everything that can be decided
without torch is decided here, in the GUI process: which checkpoint, how many
epochs, what size, what the extra settings mean, whether the name collides with
a run that already exists, and what is already sitting in ``runs/``.

Three of those deserve a note.

**The runs folder is the state.** "Has this dataset been trained?" is answered
by looking for weights under ``runs/<name>/weights/``, never by a flag. Users
delete runs between sessions and copy weights in from elsewhere, and a recorded
``trained: true`` becomes a lie the moment they do.

**A name is chosen, not generated.** Ultralytics answers a name collision by
silently appending a number -- ``train`` becomes ``train2`` -- which is a
reasonable default for a command line and a bad one here, because the user then
looks for their results in a folder they did not name. The app checks the name
itself and either offers a free one or asks before overwriting, and the worker
is then told to use exactly the folder it was given.

**Image size is one number.** Ultralytics trains on squares and rejects a
height/width pair outright, so a second field here would only be a number the
trainer never sees. The shape the model runs at is chosen in Step 6, where it
genuinely can differ from a square, and that step caps both sides at the size
recorded here.

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

from .i18n import N_, tr, trn
from .project import DatasetStatus, Project

#: Bump when the per-run record's shape changes incompatibly.
TRAIN_SCHEMA = 1

#: Written into every run folder: what this run was asked for, and how it ended.
#: Ultralytics writes its own ``args.yaml`` beside it, which records the settings
#: it understood; this one records the settings the *app* offered, which is
#: where the QAT flag lives.
RUN_RECORD_FILENAME = "umla.json"

#: The app's copy of the last settings used, so that coming back tomorrow does
#: not mean retyping them. Like ``autolabel.json`` it records what was asked
#: for, never that anything finished.
SETTINGS_FILENAME = "train.json"

WEIGHTS_DIRNAME = "weights"
BEST_WEIGHTS = "best.pt"
LAST_WEIGHTS = "last.pt"

DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640

#: YOLO's stride. A side that is not a multiple of it cannot be laid out on the
#: detection grid, and Ultralytics silently rounds it -- so the app refuses
#: instead, because a model trained at 641 that reports 640 is a confusing way
#: to find out.
STRIDE = 32
MIN_IMGSZ = 64
MAX_IMGSZ = 4096


def nearest_on_grid(value: float) -> int:
    """The multiple of :data:`STRIDE` closest to ``value``, never below the minimum."""
    return max(MIN_IMGSZ, round(value / STRIDE) * STRIDE)

#: Enough to be a real run, small enough that a first attempt is not an
#: afternoon. Ultralytics' own floor is 1.
MIN_EPOCHS = 1
MAX_EPOCHS = 10_000


@dataclass(frozen=True)
class Model:
    """One checkpoint the user can start training from."""

    key: str
    """What Ultralytics is given: a checkpoint name it fetches on first use."""

    label: str
    params_m: float
    coco_map: float
    note: str

    @property
    def caption(self) -> str:
        """What the selector shows: the model's name and nothing else.

        The note and the COCO figures are the tooltip instead. A dropdown whose
        every line carries a sentence is read as prose rather than scanned as a
        list, and the thing being chosen here -- which of thirteen checkpoints --
        is exactly the case where scanning is what the user wants.
        """
        return self.label

    @property
    def detail(self) -> str:
        """The tooltip: what the caption no longer says, plus the figures."""
        note = tr(self.note)
        return tr(
            "{note}\n{params_m:g}M parameters, {coco_map:g} mAP50-95 on COCO "
            "before any of your own training.",
            note=f"{note[:1].upper()}{note[1:]}",
            params_m=self.params_m,
            coco_map=self.coco_map,
        )


#: Three families, newest first. YOLO26 is where a new project should start and
#: is the default; YOLO11 is the previous generation and still what most
#: tutorials and deployment recipes are written against; YOLOv8 is here because
#: deployment stacks pin to it -- DeepStream configuration files in particular
#: are written against its output layout, and Step 6 has a DeepStream toggle for
#: exactly those users.
#:
#: The figures are Ultralytics' published COCO numbers, and they are here to make
#: the trade visible: x is nowhere near the accuracy jump over n that it is the
#: time and memory jump.
#:
#: YOLO26 is not simply a better YOLO11. It has no NMS -- the head predicts one
#: box per object directly -- which is why it is faster on edge hardware at the
#: same accuracy, and which also means it drops the DFL loss. Step 5 shows
#: whatever losses the trainer reports, so that costs nothing here, but an
#: export in Step 6 with NMS settings on an end-to-end model is a contradiction,
#: and that is the thing to remember about this family.
MODELS: tuple[Model, ...] = (
    Model("yolo26n.pt", "YOLO26n", 2.4, 40.9, N_("fastest, smallest, no NMS. Start here.")),
    Model("yolo26s.pt", "YOLO26s", 9.5, 48.6, N_("no NMS; a good default once n works.")),
    Model("yolo26m.pt", "YOLO26m", 20.4, 53.1, N_("no NMS; noticeably slower to train.")),
    Model("yolo26l.pt", "YOLO26l", 24.8, 55.0, N_("no NMS; wants a card with room to spare.")),
    Model("yolo26x.pt", "YOLO26x", 55.7, 57.5, N_("no NMS; the most accurate, and the slowest.")),
    Model("yolo11n.pt", "YOLO11n", 2.6, 39.5, N_("previous generation; fastest, smallest.")),
    Model("yolo11s.pt", "YOLO11s", 9.4, 47.0, N_("previous generation; a good all-rounder.")),
    Model("yolo11m.pt", "YOLO11m", 20.1, 51.5, N_("previous generation; slower to train.")),
    Model("yolo11l.pt", "YOLO11l", 25.3, 53.4, N_("previous generation; wants a bigger card.")),
    Model("yolo11x.pt", "YOLO11x", 56.9, 54.7, N_("previous generation; the slowest of them.")),
    Model("yolov8n.pt", "YOLOv8n", 3.2, 37.3, N_("older; pick it if your deployment needs v8.")),
    Model("yolov8s.pt", "YOLOv8s", 11.2, 44.9, N_("older; pick it if your deployment needs v8.")),
    Model("yolov8m.pt", "YOLOv8m", 25.9, 50.2, N_("older; pick it if your deployment needs v8.")),
)

#: The first of them, which is the newest and smallest: the model most likely to
#: get someone through the whole workflow once before they start tuning it.
DEFAULT_MODEL = MODELS[0].key

MODELS_BY_KEY = {model.key: model for model in MODELS}


# --------------------------------------------------------------------------- #
# Extra arguments
# --------------------------------------------------------------------------- #

#: Settings the app decides, which the user therefore cannot pass. Each is here
#: because letting it through would break something the app promises: ``data``
#: and ``model`` are the whole point of the wizard, ``project``/``name`` decide
#: where Step 6 looks for weights, and ``resume`` silently ignores every other
#: argument on the line.
RESERVED_ARGS: frozenset[str] = frozenset(
    {
        "model",
        "data",
        "device",
        "project",
        "name",
        "exist_ok",
        "epochs",
        "imgsz",
        "resume",
        "task",
        "mode",
        "save_dir",
    }
)

#: Ultralytics' own argument names, as of the pinned release. This list is
#: advisory: a name that is not on it produces a warning rather than a refusal,
#: because Step 1's update button can move the environment to a release that
#: added one, and a stale list must not block a setting that now exists. The
#: authoritative check is in the worker, against the defaults of whichever
#: Ultralytics is actually installed, and it happens before training starts.
KNOWN_ARGS: frozenset[str] = frozenset(
    """
    time patience batch cache workers pretrained optimizer verbose seed
    deterministic single_cls rect cos_lr close_mosaic amp fraction profile
    freeze multi_scale overlap_mask mask_ratio dropout val split save
    save_period save_json save_txt save_conf save_crop plots conf iou max_det
    half dnn augment agnostic_nms classes retina_masks embed nbs
    lr0 lrf momentum weight_decay warmup_epochs warmup_momentum warmup_bias_lr
    box cls dfl pose kobj label_smoothing
    hsv_h hsv_s hsv_v degrees translate scale shear perspective flipud fliplr
    bgr mosaic mixup cutmix copy_paste copy_paste_mode auto_augment erasing
    line_width compile
    """.split()
)

#: Deliberately no "1" and no "0". Ultralytics is happy with either sense of
#: those -- ``save=0`` is falsy however it arrives -- while ``batch=1`` is a
#: real and common setting that would otherwise be read as ``batch=True``.
_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}
_NONE = {"none", "null", ""}


@dataclass(frozen=True)
class ExtraArgs:
    """The "additional arguments" box, parsed.

    Kept as three lists rather than an exception because the page shows all of
    the trouble at once: a user who typed three pairs and got two wrong should
    not have to fix them one run at a time.
    """

    values: dict[str, object] = field(default_factory=dict)
    problems: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.problems


def parse_extra_args(text: str) -> ExtraArgs:
    """``"batch=8, patience=20"`` -> ``{"batch": 8, "patience": 20}``.

    Values are typed by what they look like, because Ultralytics cares: it
    tests ``rect`` for truth, and the string ``"false"`` is true. ``true``,
    ``false`` and ``none`` are therefore recognised words rather than text, and
    a number is a number.
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
                    'value, like "batch=8".',
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
                    "what is trained, on what, and where the results go.",
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
    # A quoted string keeps its contents and loses the quotes, so that a value
    # which looks like a number can still be passed as text.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


# --------------------------------------------------------------------------- #
# Runs already on disk
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunInfo:
    """One folder under ``runs/``, described by what is in it right now."""

    name: str
    path: Path
    best: Path | None = None
    last: Path | None = None
    epochs: int = 0
    finished: bool = False
    qat: bool = False
    imgsz: int = 0
    """The square side it was trained at; 0 when nothing on disk says."""

    model: str = ""
    best_fitness: float = 0.0
    modified_at: float = 0.0

    @property
    def has_weights(self) -> bool:
        return self.best is not None or self.last is not None

    @property
    def weights(self) -> Path | None:
        """The file Step 6 would export: the best epoch, or the only one there is."""
        return self.best or self.last

    def caption(self) -> str:
        if not self.has_weights:
            return tr("{name} - started, but no weights were saved", name=self.name)
        if self.finished:
            detail = tr("{name} - finished", name=self.name)
        else:
            detail = tr("{name} - stopped part-way", name=self.name)
        if self.epochs:
            detail += trn(", {n} epoch", ", {n} epochs", self.epochs)
        if self.best_fitness:
            detail += tr(", best mAP50-95 {fitness:.4f}", fitness=self.best_fitness)
        if self.qat:
            detail += ", INT8 QAT"
        return detail


def list_runs(project: Project | None) -> tuple[RunInfo, ...]:
    """Every training run in the project, newest first.

    Read off the filesystem every time it is asked for. A run folder the user
    deleted in a file manager has to disappear from the app without the app
    being restarted, and a weights file they copied in by hand has to appear.
    """
    if project is None:
        return ()
    try:
        entries = [entry for entry in project.runs_dir.iterdir() if entry.is_dir()]
    except OSError:
        return ()
    runs = [_read_run(entry) for entry in entries]
    runs.sort(key=lambda run: (-run.modified_at, run.name.casefold()))
    return tuple(runs)


def _read_run(path: Path) -> RunInfo:
    weights = path / WEIGHTS_DIRNAME
    best = weights / BEST_WEIGHTS
    last = weights / LAST_WEIGHTS
    record = _read_json(path / RUN_RECORD_FILENAME) or {}
    try:
        modified = path.stat().st_mtime
    except OSError:
        modified = 0.0
    return RunInfo(
        name=path.name,
        path=path,
        best=best if best.is_file() else None,
        last=last if last.is_file() else None,
        epochs=int(record.get("epochs_completed") or record.get("epochs") or 0),
        finished=bool(record.get("finished_at")),
        qat=bool(record.get("qat")),
        imgsz=_trained_imgsz(path, record),
        model=str(record.get("model") or ""),
        best_fitness=float(record.get("best_fitness") or 0.0),
        modified_at=modified,
    )


_ARGS_IMGSZ = re.compile(r"^imgsz:\s*(\d+)\s*$", re.MULTILINE)


def _trained_imgsz(path: Path, record: dict) -> int:
    """The square side a run was trained at, or 0 if nothing says.

    The app's own record first. Failing that, Ultralytics' ``args.yaml``, which
    every run has -- including one copied in from another machine -- and which
    states it on a line of its own; a single line does not need a YAML parser.
    """
    try:
        imgsz = int(record.get("imgsz") or 0)
    except (TypeError, ValueError):
        imgsz = 0
    if imgsz <= 0:
        # Runs from before the size became a single number recorded a pair.
        try:
            imgsz = max(int(v) for v in list(record.get("imgsz_hw") or [])[:2])
        except (TypeError, ValueError):
            imgsz = 0
    if imgsz <= 0:
        try:
            match = _ARGS_IMGSZ.search((path / "args.yaml").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            match = None
        imgsz = int(match.group(1)) if match else 0
    return imgsz if imgsz > 0 else 0


def trained_runs(project: Project | None) -> tuple[RunInfo, ...]:
    """Runs that left a model behind. What "Step 5 is done" means."""
    return tuple(run for run in list_runs(project) if run.has_weights)


def has_trained_model(project: Project | None) -> bool:
    return bool(trained_runs(project))


#: Windows reserves these as device names, whatever extension follows.
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)

_UNSAFE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def name_problem(name: str) -> str:
    """Why this output name cannot be a folder, or an empty string.

    Checked here rather than left to the filesystem because the failure would
    otherwise arrive minutes into a run, from inside Ultralytics, as an OSError
    about a path the user never typed.
    """
    name = str(name).strip()
    if not name:
        return tr("Give this run a name; its results are written to a folder of that name.")
    if _UNSAFE_NAME.search(name):
        return tr(
            "A run name cannot contain any of  < > : \" / \\ | ? *  -- it has to work "
            "as a folder name."
        )
    if name != name.strip(" ."):
        return tr("A run name cannot start or end with a space or a full stop.")
    if name.split(".")[0].lower() in _RESERVED_NAMES:
        return tr('"{name}" is a name Windows reserves for a device. Pick another.', name=name)
    if len(name) > 100:
        return tr("That name is too long; keep it under 100 characters.")
    return ""


def suggest_name(project: Project | None, *, base: str = "train") -> str:
    """A run name that is free right now.

    Ultralytics would answer a collision by quietly appending a number, which
    leaves the user's results in a folder nobody named. The app picks the free
    name up front so that what is on the button is what appears on disk.
    """
    taken = {run.name.casefold() for run in list_runs(project)}
    if base.casefold() not in taken:
        return base
    index = 2
    while f"{base}{index}".casefold() in taken:
        index += 1
    return f"{base}{index}"


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainPlan:
    """What pressing Start would do, in numbers and sentences the page shows."""

    status: DatasetStatus
    model: str = DEFAULT_MODEL
    name: str = "train"
    epochs: int = DEFAULT_EPOCHS
    imgsz: int = DEFAULT_IMGSZ
    qat: bool = False
    extra_text: str = ""
    runs: tuple[RunInfo, ...] = ()

    @property
    def extra(self) -> ExtraArgs:
        return parse_extra_args(self.extra_text)

    @property
    def existing_run(self) -> RunInfo | None:
        """The run this one would write into, if there is one."""
        wanted = self.name.strip().casefold()
        for run in self.runs:
            if run.name.casefold() == wanted:
                return run
        return None

    @property
    def trained(self) -> tuple[RunInfo, ...]:
        return tuple(run for run in self.runs if run.has_weights)

    @property
    def is_complete(self) -> bool:
        """Whether Step 6 is reachable without training anything now."""
        return bool(self.trained)

    def problems(self) -> list[str]:
        """Why Start is disabled, phrased for the person who has to fix it."""
        found: list[str] = []
        if not self.status.is_prepared:
            return [tr("No prepared dataset. Finish Step 2 first.")]
        if not self.status.fully_labelled:
            missing = self.status.total_images - self.status.total_labels
            return [
                trn(
                    "{n:,} photo(s) still have no labels. Finish autolabeling in "
                    "Step 3 first -- a photo with no label file is not in the training set.",
                    "{n:,} photo(s) still have no labels. Finish autolabeling in "
                    "Step 3 first -- a photo with no label file is not in the training set.",
                    missing,
                )
            ]
        if self.model not in MODELS_BY_KEY:
            found.append(tr("Choose a model to start training from."))

        name_trouble = name_problem(self.name)
        if name_trouble:
            found.append(name_trouble)

        if not MIN_EPOCHS <= int(self.epochs) <= MAX_EPOCHS:
            found.append(
                tr(
                    "The number of epochs has to be between {low} and {high:,}.",
                    low=MIN_EPOCHS,
                    high=MAX_EPOCHS,
                )
            )

        value = int(self.imgsz)
        if not MIN_IMGSZ <= value <= MAX_IMGSZ:
            found.append(
                tr(
                    "The image size has to be between {low} and {high} pixels.",
                    low=MIN_IMGSZ,
                    high=MAX_IMGSZ,
                )
            )
        elif value % STRIDE:
            found.append(
                tr(
                    "The image size has to be a multiple of {stride}. {value} is not -- "
                    "the nearest that is, is {nearest}.",
                    stride=STRIDE,
                    value=value,
                    nearest=nearest_on_grid(value),
                )
            )

        found.extend(self.extra.problems)
        return found

    def warnings(self) -> list[str]:
        """Things worth saying that are not reasons to refuse."""
        found: list[str] = []
        existing = self.existing_run
        if existing is not None and existing.has_weights:
            found.append(
                tr(
                    '"{name}" already holds a trained model. Starting this run '
                    "replaces it -- rename this one to keep both.",
                    name=existing.name,
                )
            )
        elif existing is not None:
            found.append(
                tr(
                    '"{name}" already exists as a folder, left over from a run '
                    "that saved nothing. Its contents will be replaced.",
                    name=existing.name,
                )
            )
        if self.qat:
            found.append(
                tr(
                    "A model trained this way has to be exported as INT8 in Step 6; "
                    "FP16 and FP32 will be refused. Training is also a little slower, "
                    "because mixed precision is switched off for the run."
                )
            )
        for key in self.extra.unknown:
            found.append(
                tr(
                    '"{key}" is not a setting this version of the app knows about. If your '
                    "Ultralytics does not know it either, the run will refuse it before "
                    "training starts.",
                    key=key,
                )
            )
        return found

    @property
    def is_runnable(self) -> bool:
        return not self.problems()

    def summary(self) -> str:
        if not self.status.is_prepared:
            return tr("No prepared dataset yet.")
        images = self.status.total_images
        in_classes = trn("in {n} class", "in {n} classes", len(self.status.classes))
        if not self.status.fully_labelled:
            return trn(
                "{n:,} photos {in_classes} -- but not all of them are labelled yet.",
                "{n:,} photos {in_classes} -- but not all of them are labelled yet.",
                images,
                in_classes=in_classes,
            )
        model = MODELS_BY_KEY.get(self.model)
        label = model.label if model else self.model
        return trn(
            "{model} on {n:,} photos {in_classes}, {epochs} at {imgsz} pixels.",
            "{model} on {n:,} photos {in_classes}, {epochs} at {imgsz} pixels.",
            images,
            model=label,
            in_classes=in_classes,
            epochs=trn("{n} epoch", "{n} epochs", self.epochs),
            imgsz=self.imgsz,
        )

    def job(self, project: Project) -> dict:
        """The job payload the worker is given."""
        return {
            "project": str(project.root),
            "data": str(project.data_yaml),
            "runs": str(project.runs_dir),
            "name": self.name.strip(),
            "model": self.model,
            "epochs": int(self.epochs),
            "imgsz": int(self.imgsz),
            "qat": bool(self.qat),
            "args": dict(self.extra.values),
            "overwrite": True,
            "schema": TRAIN_SCHEMA,
        }


def build_plan(
    status: DatasetStatus,
    *,
    project: Project | None = None,
    model: str = DEFAULT_MODEL,
    name: str = "train",
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    qat: bool = False,
    extra_text: str = "",
    runs: tuple[RunInfo, ...] | None = None,
) -> TrainPlan:
    return TrainPlan(
        status=status,
        model=model,
        name=name,
        epochs=epochs,
        imgsz=imgsz,
        qat=qat,
        extra_text=extra_text,
        runs=list_runs(project) if runs is None else tuple(runs),
    )


# --------------------------------------------------------------------------- #
# Remembering what was asked for
# --------------------------------------------------------------------------- #


def settings_path(project: Project) -> Path:
    return project.internal_dir / SETTINGS_FILENAME


def read_settings(project: Project | None) -> dict:
    """The settings last used in this project, with defaults filled in.

    Only ever used to save the user some typing, so an unreadable file is worth
    no error at all -- it costs one re-entered epoch count.
    """
    record = _read_json(settings_path(project)) if project is not None else None
    record = record if isinstance(record, dict) else {}

    model = str(record.get("model") or DEFAULT_MODEL)
    if model not in MODELS_BY_KEY:
        model = DEFAULT_MODEL
    imgsz = record.get("imgsz")
    if imgsz is None:
        # Written before the size was a single number: the longer of the pair
        # is what that run actually trained at.
        sides = [_clamped_int(record.get(key), 0, 0, MAX_IMGSZ) for key in ("imgsz_h", "imgsz_w")]
        imgsz = max(sides) or None
    return {
        "model": model,
        "epochs": _clamped_int(record.get("epochs"), DEFAULT_EPOCHS, MIN_EPOCHS, MAX_EPOCHS),
        "imgsz": _clamped_int(imgsz, DEFAULT_IMGSZ, MIN_IMGSZ, MAX_IMGSZ),
        "qat": bool(record.get("qat", False)),
        "args": str(record.get("args") or ""),
    }


def write_settings(project: Project, plan: TrainPlan) -> None:
    payload = {
        "schema": TRAIN_SCHEMA,
        "model": plan.model,
        "epochs": int(plan.epochs),
        "imgsz": int(plan.imgsz),
        "qat": bool(plan.qat),
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
        # A project on a read-only share still trains; it just forgets.
        pass


def _clamped_int(value, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return min(high, max(low, number))


def _read_json(path: Path) -> dict | None:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None
