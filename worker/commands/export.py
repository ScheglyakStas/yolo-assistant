"""Convert a trained checkpoint into a deployment format, and say what happened.

Ultralytics already knows how to export. Four things it does not do are this
file.

**It writes where the checkpoint is.** ``model.export()`` drops its output
beside ``best.pt``, inside the run folder, which means Step 7 would have to go
looking through every run for files that are not weights. The artifact is moved
into the project's ``exports/`` folder under a name that says what it is, and a
small record is written beside it -- because an ``.onnx`` does not know which
run it came from or what precision was asked for, and both are the first things
anyone needs.

**It has no idea what this app calls a precision.** The 8.4 line renamed the
export flags mid-series -- ``half``/``int8`` became ``quantize`` -- so the
spelling is decided here, against the defaults of whichever Ultralytics is
actually installed, rather than pinned to whichever one was current when this
was written.

**It fails at a QAT model in the same voice it fails at everything else.** A
model trained with INT8 quantisation-aware training carries fake-quantisation
in its graph, and asking for an FP16 or FP32 export of one is a contradiction
that surfaces from somewhere deep in the exporter as a shape or a dtype
complaint. The specification asks for that to arrive as "choose INT8", so the
failure is caught here and re-raised with the remedy attached.

**It cannot be interrupted.** An export is one long call with no callbacks --
a TensorRT engine build is minutes of it -- so there is no cooperative
cancellation point to offer. Cancelling this job is the app's ladder
terminating the process, which is safe: the only thing lost is a partial file
in the run folder, and nothing has been moved into ``exports/`` yet.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from worker.channel import Channel
from worker.i18n import N_, tr, trn
from worker.protocol import Phase
from worker.util.errors import (
    ExportFailed,
    GpuUnavailable,
    ModelUnavailable,
    QatFormatMismatch,
)

#: Kept in step with the GUI's ``core/export.py``; the worker cannot import it,
#: having only ``worker`` on its path.
RECORD_SUFFIX = ".umla.json"

FP32 = "fp32"
FP16 = "fp16"
INT8 = "int8"

#: Fixed by the specification, whatever the user asked for. ``opset`` 19 is the
#: newest every runtime the app cares about reads, and simplification folds the
#: constant arithmetic that tracing leaves behind -- both are wanted by every
#: ONNX-derived format and ignored by the rest.
FIXED_ARGS: dict[str, object] = {"simplify": True, "opset": 19}

#: What the GUI's DeepStream toggle contributes, by name. Kept in step with
#: ``core/export.DEEPSTREAM_ARGS``; see :func:`_drop_unsupported` for why the
#: worker has to be able to tell these apart from what the user typed.
DEEPSTREAM_ARG_NAMES = frozenset({"nms", "agnostic_nms"})

#: Formats built out of an ONNX graph, which are the ones :data:`FIXED_ARGS`
#: mean anything to. Passing them elsewhere is harmless -- Ultralytics carries
#: unused config through -- but saying which is which keeps the log honest.
ONNX_DERIVED = frozenset({"onnx", "engine", "openvino"})

#: A build this size is not an export gone wrong, but it is worth a line: a
#: TensorRT engine that comes out at a few hundred kilobytes usually means the
#: graph collapsed to almost nothing.
SUSPICIOUSLY_SMALL_BYTES = 256 * 1024

#: The size refusal, one whole sentence per side.
_HEIGHT_OFF_GRID = N_(
    "Height has to be a multiple of 32; {value} is not. YOLO lays its "
    "detections out on a grid of that spacing."
)
_WIDTH_OFF_GRID = N_(
    "Width has to be a multiple of 32; {value} is not. YOLO lays its "
    "detections out on a grid of that spacing."
)


def run(channel: Channel, job: dict) -> dict:
    weights = Path(job["weights"]).expanduser()
    exports_root = Path(
        job.get("exports") or (Path(job["project"]).expanduser() / "exports")
    ).expanduser()

    fmt = str(job.get("format") or "onnx")
    precision = str(job.get("precision") or FP32)
    imgsz_hw = _pair(job.get("imgsz_hw"), 640)
    qat = bool(job.get("qat"))
    extra = dict(job.get("args") or {})
    output_name = str(job.get("output") or "").strip()

    # Both of these are checked in the GUI as the user types, and both are
    # checked again here, because the command is also runnable from a shell and
    # because the run folder can change under a page that is already open.
    if not weights.is_file():
        raise ExportFailed(
            tr("{weights} is not there.", weights=weights),
            hint=tr(
                "The run it belongs to has been deleted or renamed. Go back to "
                "Step 6 and pick a checkpoint that is still on disk."
            ),
        )
    for off_grid, value in ((_HEIGHT_OFF_GRID, imgsz_hw[0]), (_WIDTH_OFF_GRID, imgsz_hw[1])):
        if value % 32:
            raise ExportFailed(
                tr(off_grid, value=value),
                hint=tr(
                    "The nearest size that works is {nearest}.",
                    nearest=max(32, round(value / 32) * 32),
                ),
            )

    channel.progress(Phase.SETUP, 0, 4, unit="steps")

    device = _resolve_device(fmt)
    extra = _drop_unsupported(extra, job, channel)
    kwargs = {
        **extra,
        **FIXED_ARGS,
        "format": fmt,
        "imgsz": list(imgsz_hw),
        "device": device,
    }
    kwargs.update(_precision_kwargs(precision))

    channel.ready(format=fmt, precision=precision, imgsz=list(imgsz_hw), device=str(device))
    channel.log(
        tr(
            "Exporting {weights} to {format} at {precision}, "
            "{width}x{height}, on device {device}.",
            weights=weights.name,
            format=fmt,
            precision=precision,
            width=imgsz_hw[1],
            height=imgsz_hw[0],
            device=device,
        )
    )
    if fmt in ONNX_DERIVED:
        channel.log(tr("Graph simplification is on and the ONNX opset is 19."))
    channel.progress(Phase.SETUP, 1, 4, unit="steps")

    model = load_yolo(str(weights), channel=channel)
    channel.progress(Phase.EXPORT, 2, 4, unit="steps")
    channel.log(
        tr(
            "Converting. There is no progress to report from inside an export -- a "
            "TensorRT engine build in particular is several quiet minutes."
        )
    )

    started = time.monotonic()
    try:
        produced = model.export(**kwargs)
    except Exception as exc:  # noqa: BLE001 - exporters raise whatever they like
        raise _explain(exc, qat=qat, precision=precision, fmt=fmt) from exc
    elapsed = time.monotonic() - started

    source = _produced_path(produced, weights)
    channel.progress(Phase.EXPORT, 3, 4, unit="steps")

    destination = _move_into(source, exports_root, output_name or source.name)
    record = {
        "schema": int(job.get("schema", 1)),
        "format": fmt,
        "precision": precision,
        "imgsz_hw": list(imgsz_hw),
        "deepstream": bool(job.get("deepstream")),
        "qat": qat,
        "run": str(job.get("run") or ""),
        "which": str(job.get("which") or ""),
        "weights": str(weights),
        "args": extra,
        "elapsed_s": round(elapsed, 2),
        "exported_at": time.time(),
    }
    _write_json(destination.with_name(destination.name + RECORD_SUFFIX), record)

    size = _size_of(destination)
    if 0 < size < SUSPICIOUSLY_SMALL_BYTES:
        channel.warning(
            "export",
            tr(
                "{name} is only {kilobytes:.0f} KB, which is small for a "
                "detector. Test it in Step 7 before deploying it.",
                name=destination.name,
                kilobytes=size / 1024,
            ),
        )

    channel.artifact("export", str(destination), format=fmt, precision=precision)
    channel.progress(Phase.EXPORT, 4, 4, unit="steps")
    channel.log(
        tr(
            "Wrote {path} ({size}) in {duration}.",
            path=destination,
            size=_size(size),
            duration=_duration(elapsed),
        )
    )

    return {
        "export": str(destination),
        "name": destination.name,
        "format": fmt,
        "precision": precision,
        "imgsz_hw": list(imgsz_hw),
        "size_bytes": size,
        "elapsed_s": round(elapsed, 2),
    }


# --------------------------------------------------------------------------- #
# What the exporter is asked for
# --------------------------------------------------------------------------- #


def _precision_kwargs(precision: str) -> dict[str, object]:
    """How to spell the requested precision to the Ultralytics that is here.

    The 8.4 line renamed these mid-series: ``half=True``/``int8=True`` became a
    single ``quantize="fp16"``/``"int8"``. Both spellings are alive in the wild
    -- Step 1's update button moves the environment between releases -- and
    getting it wrong is the quiet kind of wrong: an unrecognised keyword is
    carried through the config untouched, the export succeeds, and what comes
    out is a full-precision model that the app has labelled INT8.

    So the spelling is read off the installed defaults rather than assumed.
    FP32 needs no keyword under either scheme, which is why it is the only one
    that cannot get this wrong.
    """
    if precision == FP32:
        return {}
    defaults = _default_cfg()
    if "quantize" in defaults:
        return {"quantize": precision}
    return {"int8": True} if precision == INT8 else {"half": True}


def _drop_unsupported(values: dict, job: dict, channel: Channel) -> dict:
    """Check the settings against the Ultralytics that is installed.

    Two different things happen to a name this release does not know, because
    two different people are responsible for them. A setting the *user* typed
    is a typo, and stopping before anything is converted is the whole point of
    checking. A setting the *DeepStream toggle* contributed is the app's own
    shorthand: the user ticked a box, and refusing the export because one of
    the pairs behind that box was renamed upstream would charge the app's
    problem to them. Those are dropped, with a line saying which.
    """
    if not values:
        return {}
    defaults = _default_cfg()
    if not defaults:
        return dict(values)

    # The GUI folds the toggle's pairs in with the user's, so which is which is
    # decided by the name. It refuses a typed "nms" while the toggle is on, so
    # the two sets cannot overlap.
    ours = DEEPSTREAM_ARG_NAMES if job.get("deepstream") else frozenset()

    kept: dict[str, object] = {}
    dropped: list[str] = []
    unknown: list[str] = []
    for key, value in values.items():
        if key in defaults:
            kept[key] = value
        elif key in ours:
            dropped.append(key)
        else:
            unknown.append(key)

    if unknown:
        raise ExportFailed(
            tr(
                "Ultralytics has no export setting called {names}. Check the spelling "
                "in the additional settings box.",
                names=_quoted(unknown),
            ),
            hint=tr("The full list of settings is at https://docs.ultralytics.com/usage/cfg/"),
        )
    if dropped:
        channel.warning(
            "export",
            trn(
                "This Ultralytics has no {names}, so the DeepStream toggle could not "
                "set it. The export itself is unaffected.",
                "This Ultralytics has no {names}, so the DeepStream toggle could not "
                "set them. The export itself is unaffected.",
                len(dropped),
                names=_quoted(dropped),
            ),
            hint=tr(
                "The exported file may still work in DeepStream. If it does not, "
                "report this: the app's DeepStream settings need updating for this "
                "Ultralytics release."
            ),
        )
    return kept


def _default_cfg() -> dict:
    """Ultralytics' own defaults, or an empty mapping if they cannot be read.

    An older or newer layout is not worth failing an export over: the check
    this feeds is a courtesy, and the exporter itself has the final say.
    """
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except Exception:  # noqa: BLE001 - a moved module is not a reason to stop
        return {}
    return dict(DEFAULT_CFG_DICT) if isinstance(DEFAULT_CFG_DICT, dict) else {}


def _resolve_device(fmt: str) -> object:
    """Which device to export on. Set by the app, as the specification says.

    Unlike training, falling back to the processor is usually correct here: an
    ONNX export is a trace and a graph rewrite and does not care. TensorRT does
    care -- it compiles kernels for the card in the machine and cannot be built
    without one -- so that is the one case where the absence of a GPU is an
    error rather than a slower route to the same file.
    """
    import torch

    if torch.cuda.is_available():
        return 0
    if fmt == "engine":
        raise GpuUnavailable(
            tr(
                "A TensorRT engine is compiled for the GPU in this machine, and PyTorch "
                "cannot see one. Export to ONNX instead, which runs anywhere, or build "
                "the engine on the machine it will run on."
            )
        )
    return "cpu"


# --------------------------------------------------------------------------- #
# What came out
# --------------------------------------------------------------------------- #


def _produced_path(produced, weights: Path) -> Path:
    """Where Ultralytics says it wrote, checked rather than trusted.

    ``export()`` returns the path as a string in every release the app
    supports, but it has returned other things before, and an export that
    "succeeded" and left nothing findable has to be reported as a failure here
    rather than as a confusing move error two lines later.
    """
    candidate = Path(str(produced)).expanduser() if produced else None
    if candidate is not None and candidate.exists():
        return candidate
    if produced:
        raise ExportFailed(
            tr(
                "The export finished but did not leave a file behind -- it reported "
                "{produced}. Look in {folder} for a partial one.",
                produced=repr(produced),
                folder=weights.parent,
            )
        )
    raise ExportFailed(
        tr(
            "The export finished but did not leave a file behind. Look in {folder} "
            "for a partial one.",
            folder=weights.parent,
        )
    )


def _move_into(source: Path, exports_root: Path, name: str) -> Path:
    """Put the artifact in ``exports/`` under the name the app chose.

    A move, not a copy: a TensorRT engine is hundreds of megabytes and leaving
    a second one inside the run folder is how a project quietly fills a disk.
    """
    exports_root.mkdir(parents=True, exist_ok=True)
    destination = exports_root / name
    if destination.resolve() == source.resolve():
        return destination

    try:
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        shutil.move(str(source), str(destination))
    except OSError as exc:
        raise ExportFailed(
            tr(
                "{name} could not be moved into {folder}: {error}. Close "
                "anything that has the previous export of this name open.",
                name=source.name,
                folder=exports_root,
                error=exc,
            ),
        ) from exc
    return destination


def _explain(exc: Exception, *, qat: bool, precision: str, fmt: str) -> Exception:
    """Turn whatever the exporter raised into something with a remedy attached.

    The QAT case is the one the specification calls out. A model trained with
    quantisation-aware training carries fake-quantisation nodes, and every
    exporter's complaint about them is about a dtype or a node type rather than
    about quantisation -- so the app has to supply the connection.
    """
    if qat and precision != INT8:
        return QatFormatMismatch(
            tr(
                "This model was trained for INT8, and exporting it at "
                "{precision} failed: {error}",
                precision=precision.upper(),
                error=exc,
            )
        )
    if qat and fmt not in ("onnx", "engine"):
        return QatFormatMismatch(
            tr(
                "This model was trained for INT8, and {format} cannot carry its "
                "quantisers: {error}",
                format=fmt,
                error=exc,
            ),
            hint=tr("Export it as ONNX or TensorRT instead."),
        )
    return ExportFailed(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


def load_yolo(model_id: str, *, channel: Channel | None = None):
    """Load a checkpoint. Replaced wholesale in tests, as in ``train.py``."""
    # A missing ultralytics is deliberately left to propagate: the entry point
    # turns it into "the managed environment is missing 'ultralytics'", which
    # names both the problem and the Repair button that fixes it.
    from ultralytics import YOLO

    try:
        return YOLO(model_id)
    except Exception as exc:  # noqa: BLE001 - a bad checkpoint arrives as many types
        raise ModelUnavailable(
            tr("{model} could not be loaded: {error}", model=model_id, error=exc),
            hint=tr(
                "The file may be truncated or from an incompatible Ultralytics "
                "release. Train the model again in Step 5."
            ),
        ) from exc


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _pair(value, fallback: int) -> list[int]:
    try:
        height, width = (int(v) for v in list(value)[:2])
    except (TypeError, ValueError):
        return [fallback, fallback]
    return [height, width]


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


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    if minutes:
        return tr("{minutes}m {seconds:02d}s", minutes=minutes, seconds=secs)
    return tr("{seconds}s", seconds=secs)


def _quoted(keys) -> str:
    return ", ".join(f'"{key}"' for key in sorted(keys))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)
