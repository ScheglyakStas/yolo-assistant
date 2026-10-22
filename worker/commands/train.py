"""Train a YOLO detector on the prepared dataset, and narrate it while it runs.

Ultralytics already knows how to train. What it does not do is tell a GUI what
is happening in a form the GUI can render, stop when a user presses a button, or
put its results where the next step expects to find them. That is this file.

Four things in here are decisions rather than plumbing.

**Progress is counted in batches and reported in epochs.** A bar that only moves
once an epoch is a bar that does not move for eleven minutes, and one labelled
in batches means nothing to a user who chose a number of epochs. So the bar
advances per batch, over the whole run, and says which epoch it is in.

**The estimate is an average over epochs, weighted towards the recent ones.**
The first epoch of a real run pays for the checkpoint download, cudnn's
autotuning, CUDA's warmup and a cold page cache, and is routinely three times
the cost of the second. A plain mean is still apologising for it halfway
through the run, so this is an exponential moving average and it says nothing
at all until one epoch has finished.

**Cancelling raises out of a callback.** Ultralytics has a cooperative stop
flag, but it is only read between epochs -- and an epoch can be twenty minutes,
while the app's cancel ladder starts signalling after five seconds. Raising from
the per-batch callback unwinds the training loop immediately through ordinary
exception handling, and what is on disk afterwards is the last checkpoint
Ultralytics saved: a real, loadable model from the last completed epoch.

**The run folder is named, not negotiated.** Ultralytics answers a name
collision by appending a number, so a user who asks for ``bottles`` and already
has one gets their results in ``bottles2`` and goes looking in the wrong place.
The name is checked here and passed with ``exist_ok=True``, so the folder that
appears is the folder that was asked for.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from worker.channel import Channel
from worker.i18n import tr, trn
from worker.protocol import Phase
from worker.util import gpu_memory, toolchain
from worker.util.errors import DatasetError, GpuUnavailable, ModelUnavailable, QatUnavailable

#: Kept in step with the GUI's ``core/training.py``; the worker cannot import
#: it, having only ``worker`` on its path.
RUN_RECORD_FILENAME = "umla.json"
WEIGHTS_DIRNAME = "weights"
PREPARED_FILENAME = "prepared.json"
CLASSES_FILENAME = "classes.txt"
LABEL_SUFFIX = ".txt"

#: Files that mark a directory under ``runs/`` as one of ours, and therefore
#: safe to replace when the user has agreed to overwrite it. A folder holding
#: none of them is something else that happens to share the name, and deleting
#: it would be the app losing someone's data.
RUN_MARKERS = ("args.yaml", RUN_RECORD_FILENAME, WEIGHTS_DIRNAME, "results.csv")

#: A dataset smaller than this trains, but the accuracy it reports is mostly
#: noise, and saying so now is cheaper than saying it after two hours.
SMALL_DATASET = 50


def run(channel: Channel, job: dict) -> dict:
    project = Path(job["project"]).expanduser()
    dataset = project / "dataset"
    data_yaml = Path(job.get("data") or (dataset / "data.yaml")).expanduser()
    runs_root = Path(job.get("runs") or (project / "runs")).expanduser()

    name = str(job.get("name") or "train").strip()
    model_id = str(job.get("model") or "yolo11n.pt")
    epochs = int(job.get("epochs") or 100)
    imgsz = int(job.get("imgsz") or 640)
    imgsz_hw = _pair(job.get("imgsz_hw"), imgsz)
    qat = bool(job.get("qat"))
    extra = dict(job.get("args") or {})
    overwrite = bool(job.get("overwrite", True))

    _check_dataset(dataset, data_yaml, channel)
    if imgsz % 32:
        raise DatasetError(
            tr(
                "The image size has to be a multiple of 32; {imgsz} is not. "
                "YOLO lays its detections out on a grid of that spacing.",
                imgsz=imgsz,
            )
        )

    run_dir = _prepare_run_dir(runs_root, name, overwrite=overwrite)

    # Checked before anything is downloaded or loaded. A run that is going to
    # fail for want of a toolkit should fail in the first two seconds, not after
    # the checkpoint and the first epoch.
    if qat:
        # Before anything imports Model Optimizer, which is what fixes where
        # torch will look for the CUDA Toolkit.
        kernel_toolchain = _prepare_kernel_toolchain()
        _require_qat_toolkit()
        _warn_if_kernels_cannot_build(channel, kernel_toolchain)

    _reject_unknown_args(extra)

    channel.progress(Phase.SETUP, 0, 1, unit="steps")

    record = {
        "schema": int(job.get("schema", 1)),
        "model": model_id,
        "epochs": epochs,
        "imgsz": imgsz,
        "imgsz_hw": list(imgsz_hw),
        "qat": qat,
        "args": extra,
        "data": str(data_yaml),
        "started_at": time.time(),
    }
    # Written before the device is resolved and before the checkpoint is
    # fetched, for the same reason Step 3 writes its prompts first: a run that
    # fails or is interrupted on the way to its first epoch still leaves behind
    # what it was asked to do, which is what Step 6 reads and what a support
    # request needs.
    _write_json(run_dir / RUN_RECORD_FILENAME, record)

    device = _resolve_device(str(job.get("device") or "auto"))
    # Before anything is allocated on the card; see worker/util/gpu_memory.py.
    cap = gpu_memory.cap_to_free_memory(device)
    if cap is not None:
        channel.log(
            tr(
                "Training may use up to {gb:.1f} GB of GPU memory. If a batch needs "
                "more, the batch size is halved rather than letting Windows move the "
                "overflow into much slower system memory.",
                gb=cap / 1024**3,
            )
        )
    channel.progress(Phase.DOWNLOAD, 0, None, unit="files")
    channel.log(
        tr("Loading {model}. The first run downloads it, which takes a moment.", model=model_id)
    )
    model = load_yolo(model_id, channel=channel)

    kwargs = {
        **extra,
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "device": device,
        "project": str(runs_root),
        "name": name,
        "exist_ok": True,
    }
    if qat:
        # Ultralytics' own QAT, not a callback of the app's. It inserts the
        # quantisers before the EMA copy is taken, and that copy is what gets
        # validated and saved: quantising the live model from a callback leaves
        # best.pt a plain float model with quantize=None in its metadata, which
        # the exporter then calibrates like any other.
        kwargs["quantize"] = 8
        # Mixed precision and fake quantisation fight: the quantisers are
        # calibrated against fp32 activation ranges and autocast then feeds them
        # fp16 ones. Forced off here rather than left to the user, because the
        # failure is a quietly worse model rather than an error.
        kwargs["amp"] = False
        channel.log(
            tr(
                "INT8 QAT: Ultralytics calibrates the quantisers on a few training batches "
                "before the first epoch. Export the result as INT8 in Step 6."
            )
        )

    reporter = _Reporter(channel, epochs=epochs)
    for event, handler in reporter.callbacks().items():
        model.add_callback(event, handler)

    channel.ready(epochs=epochs, model=model_id, imgsz=imgsz, run=str(run_dir), device=str(device))
    channel.log(
        trn(
            "Training {model} for {n} epochs at {imgsz}px on device {device}.",
            "Training {model} for {n} epochs at {imgsz}px on device {device}.",
            epochs,
            model=model_id,
            imgsz=imgsz,
            device=device,
        )
    )
    channel.progress(Phase.SETUP, 1, 1, unit="steps")

    started = time.monotonic()
    model.train(**kwargs)
    elapsed = time.monotonic() - started

    # Where Ultralytics actually wrote, which is what the next step has to be
    # told about. It should equal run_dir -- exist_ok=True is what makes that
    # true -- but reporting a guessed path would be a bug nobody notices until
    # Step 6 finds no weights.
    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", run_dir) or run_dir)
    weights = save_dir / WEIGHTS_DIRNAME
    best = weights / "best.pt"
    last = weights / "last.pt"

    # What Step 6 would export: the best epoch, or the only checkpoint there is
    # when a run was stopped before Ultralytics had chosen one.
    exportable = best if best.is_file() else (last if last.is_file() else None)

    record.update(
        {
            "finished_at": time.time(),
            "elapsed_s": round(elapsed, 2),
            "epochs_completed": reporter.epochs_done,
            "best_epoch": reporter.best_epoch,
            "best_fitness": reporter.best_fitness,
            "metrics": reporter.last_metrics,
            "run": str(save_dir),
            "weights": str(exportable) if exportable else "",
        }
    )
    _write_json(save_dir / RUN_RECORD_FILENAME, record)

    if best.is_file():
        channel.artifact("weights", str(best), which="best", epoch=reporter.best_epoch)
    if last.is_file():
        channel.artifact("weights", str(last), which="last")

    if exportable is None:
        # Training "succeeded" and produced nothing to export. Almost always
        # save=false in the additional arguments.
        channel.warning(
            "train",
            tr(
                "Training finished but no weights file was written, so there is nothing "
                "to export in Step 6."
            ),
            hint=tr('Remove "save=false" from the additional settings and train again.'),
        )

    channel.log(
        trn(
            "Finished {n} epoch in {duration}.",
            "Finished {n} epochs in {duration}.",
            reporter.epochs_done,
            duration=_duration(elapsed),
        )
    )
    return {
        "project": str(project),
        "run": str(save_dir),
        "name": save_dir.name,
        "weights": str(exportable) if exportable else "",
        "epochs": reporter.epochs_done,
        "requested_epochs": epochs,
        "best_epoch": reporter.best_epoch,
        "best_fitness": reporter.best_fitness,
        "metrics": reporter.last_metrics,
        "imgsz": imgsz,
        "qat": qat,
        "elapsed_s": round(elapsed, 2),
    }


# --------------------------------------------------------------------------- #
# What is being trained on
# --------------------------------------------------------------------------- #


def _check_dataset(dataset: Path, data_yaml: Path, channel: Channel) -> None:
    """Refuse a dataset that cannot train, and say which one it is.

    Ultralytics' own message for a missing label is a warning buried in a
    scanning bar, and the run then trains happily on whatever is left. An empty
    training set is not a warning here.
    """
    if not (dataset / PREPARED_FILENAME).is_file():
        raise DatasetError(
            tr("{folder} does not hold a prepared dataset. Run Step 2 again.", folder=dataset)
        )
    if not data_yaml.is_file():
        raise DatasetError(
            tr(
                "{file} is missing, so there is nothing to point the trainer at. "
                "Run Step 2 again.",
                file=data_yaml,
            )
        )

    classes = _class_names(dataset)
    labels_root = dataset / "labels"
    labelled = 0
    for class_dir in sorted(p for p in labels_root.glob("*") if p.is_dir()):
        labelled += sum(
            1 for entry in class_dir.iterdir()
            if entry.is_file() and entry.suffix.lower() == LABEL_SUFFIX
        )
    if not labelled:
        raise DatasetError(
            tr(
                "Not one photo has a label file, so there is nothing for the model to "
                "learn from. Run autolabeling in Step 3 first."
            )
        )
    if labelled < SMALL_DATASET:
        channel.warning(
            "train",
            trn(
                "Only {n} labelled photo(s) {in_classes}. A model "
                "trained on this few will report an accuracy that is mostly luck.",
                "Only {n} labelled photo(s) {in_classes}. A model "
                "trained on this few will report an accuracy that is mostly luck.",
                labelled,
                in_classes=trn("in {n} class(es)", "in {n} class(es)", len(classes)),
            ),
            hint=tr("More photos is the single biggest improvement available here."),
        )
    channel.log(
        trn(
            "Training on {n:,} labelled photos {in_classes}.",
            "Training on {n:,} labelled photos {in_classes}.",
            labelled,
            in_classes=trn("in {n} classes", "in {n} classes", len(classes)),
        )
    )


def _pair(value, fallback: int) -> list[int]:
    """A height/width pair from the job, however incomplete it arrives.

    The pair is the one thing in the record Ultralytics never sees, so Step 6
    reads it back off this file and a malformed one would default the export to
    nothing at all.
    """
    try:
        height, width = (int(v) for v in list(value)[:2])
    except (TypeError, ValueError):
        return [fallback, fallback]
    return [height, width]


def _class_names(dataset: Path) -> list[str]:
    try:
        text = (dataset / CLASSES_FILENAME).read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(
            tr("{file} cannot be read: {error}", file=dataset / CLASSES_FILENAME, error=exc)
        ) from exc
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if not names:
        raise DatasetError(
            tr("{file} is empty. Run Step 2 again.", file=dataset / CLASSES_FILENAME)
        )
    return names


def _prepare_run_dir(runs_root: Path, name: str, *, overwrite: bool) -> Path:
    """The folder this run writes into, emptied if the user agreed to that."""
    if not name or name != name.strip(" .") or set(name) & set('<>:"/\\|?*'):
        raise DatasetError(
            tr('"{name}" cannot be used as a folder name for this run.', name=name)
        )

    run_dir = runs_root / name
    if run_dir.exists():
        if not overwrite:
            raise DatasetError(
                tr(
                    '"{name}" already exists. Choose another name, or agree to replace it.',
                    name=name,
                )
            )
        if not any((run_dir / marker).exists() for marker in RUN_MARKERS) and any(
            run_dir.iterdir()
        ):
            raise DatasetError(
                tr(
                    "{folder} is not empty and does not look like a training run this app "
                    "made, so it will not be deleted. Choose another name for this run.",
                    folder=run_dir,
                )
            )
        try:
            shutil.rmtree(run_dir)
        except OSError as exc:
            # Almost always a file open in another program -- a weights file
            # being copied, or the folder itself open in a file manager.
            raise DatasetError(
                tr(
                    "{folder} could not be emptied: {error}. Close anything that has a "
                    "file in that folder open, or choose another name for this run.",
                    folder=run_dir,
                    error=exc,
                )
            ) from exc
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _reject_unknown_args(values: dict) -> None:
    """Check the extra settings against the Ultralytics that is installed.

    The GUI has its own list and warns from it, but that list ages: Step 1 can
    move the environment to a release that added a setting, and a stale list
    must not block one that now exists. This is the version that cannot be
    stale, and it runs before anything is downloaded.
    """
    if not values:
        return
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except Exception:  # noqa: BLE001 - an older layout is not worth failing over
        return
    unknown = sorted(key for key in values if key not in DEFAULT_CFG_DICT)
    if unknown:
        raise DatasetError(
            tr(
                "Ultralytics has no setting called {names}. Check the spelling in the "
                "additional settings box.",
                names=", ".join(f'"{key}"' for key in unknown),
            ),
            hint=tr("The full list of settings is at https://docs.ultralytics.com/usage/cfg/"),
        )


def _resolve_device(device: str) -> object:
    """Which device to train on, refusing to fall back to the CPU by accident.

    A CPU fallback is not a degraded run, it is a different activity: the same
    hundred epochs go from twenty minutes to somewhere past a day. Someone who
    genuinely wants it can ask for it by name.
    """
    if device not in ("", "auto"):
        return device

    # A missing torch is left to propagate: the entry point turns it into "the
    # managed environment is missing 'torch'", which names the Repair button
    # that fixes it rather than the line that noticed.
    import torch

    if not torch.cuda.is_available():
        raise GpuUnavailable(
            tr(
                "PyTorch cannot see a GPU, and training on the processor would take days "
                "rather than minutes."
            )
        )
    return 0


# --------------------------------------------------------------------------- #
# Narrating the run
# --------------------------------------------------------------------------- #


class _Reporter:
    """Turns Ultralytics' callbacks into the app's protocol messages.

    Everything is read with ``getattr``. The trainer is not a documented
    interface, its attributes have moved between releases, and a missing one
    must cost a line of the display rather than the run.
    """

    def __init__(self, channel: Channel, *, epochs: int) -> None:
        self._channel = channel
        self._epochs = max(1, int(epochs))
        self._batches_per_epoch = 0
        self._batches_this_epoch = 0
        self._last_epoch_started: int | None = None
        self._epoch_started = 0.0
        self._ema_epoch_s: float | None = None

        self.epochs_done = 0
        self.best_epoch = 0
        self.best_fitness = 0.0
        self.last_metrics: dict[str, float] = {}

    def callbacks(self) -> dict:
        return {
            "on_pretrain_routine_end": self.on_pretrain_routine_end,
            "on_train_start": self.on_train_start,
            "on_train_epoch_start": self.on_train_epoch_start,
            "on_train_batch_end": self.on_train_batch_end,
            "on_fit_epoch_end": self.on_fit_epoch_end,
            "on_train_end": self.on_train_end,
        }

    # -- setup ------------------------------------------------------------- #

    def on_pretrain_routine_end(self, trainer) -> None:
        self._channel.check_cancelled()

    def on_train_start(self, trainer) -> None:
        self._channel.check_cancelled()
        self._batches_per_epoch = _loader_length(trainer)
        self._epochs = int(getattr(trainer, "epochs", self._epochs) or self._epochs)
        if self._batches_per_epoch:
            self._channel.log(
                trn(
                    "{n} batches per epoch, {total:,} in total.",
                    "{n} batches per epoch, {total:,} in total.",
                    self._batches_per_epoch,
                    total=self._batches_per_epoch * self._epochs,
                )
            )

    # -- per epoch --------------------------------------------------------- #

    def on_train_epoch_start(self, trainer) -> None:
        """An epoch begins -- or the first one begins again, with a smaller batch.

        When the first epoch runs out of memory, Ultralytics halves the batch,
        rebuilds the dataloader and restarts that epoch without calling
        ``on_train_start`` again. The loader is therefore re-measured here: half
        the batch is twice the batches, and a bar still counting against the old
        total reaches its end halfway through the run and sits there.
        """
        self._channel.check_cancelled()
        epoch = self._epoch_of(trainer)
        if epoch == self._last_epoch_started:
            batch = getattr(trainer, "batch_size", None)
            if batch:
                self._channel.log(
                    tr(
                        "Out of GPU memory; restarting epoch {epoch} with batch size {batch}.",
                        epoch=epoch,
                        batch=batch,
                    )
                )
            else:
                self._channel.log(
                    tr(
                        "Out of GPU memory; restarting epoch {epoch} with a smaller batch.",
                        epoch=epoch,
                    )
                )
        self._last_epoch_started = epoch
        batches = _loader_length(trainer)
        if batches:
            self._batches_per_epoch = batches
        self._batches_this_epoch = 0
        self._epoch_started = time.monotonic()
        self._emit_progress(trainer)

    def on_train_batch_end(self, trainer) -> None:
        # The only cancellation point that runs often enough to matter: an epoch
        # can be twenty minutes and the app starts signalling after five seconds.
        self._channel.check_cancelled()
        self._batches_this_epoch += 1
        self._emit_progress(trainer)

    def on_fit_epoch_end(self, trainer) -> None:
        """One epoch is over and its validation metrics exist."""
        self._channel.check_cancelled()
        epoch = self._epoch_of(trainer)
        self.epochs_done = max(self.epochs_done, epoch)

        elapsed = time.monotonic() - self._epoch_started if self._epoch_started else 0.0
        if elapsed > 0:
            self._ema_epoch_s = (
                elapsed if self._ema_epoch_s is None else 0.6 * self._ema_epoch_s + 0.4 * elapsed
            )

        values = _collect_metrics(trainer)
        fitness = _as_float(getattr(trainer, "fitness", None))
        best = _as_float(getattr(trainer, "best_fitness", None))
        # Ultralytics updates best_fitness during validation, so the two being
        # equal is how "this was the best epoch so far" is spelled.
        is_best = fitness is not None and best is not None and abs(fitness - best) < 1e-12
        if is_best:
            self.best_epoch = epoch
            self.best_fitness = round(float(fitness), 6)

        if values:
            self.last_metrics = values
            self._channel.metrics(epoch, values, best=is_best)
        self._channel.log(_epoch_line(epoch, self._epochs, values, is_best))
        self._emit_progress(trainer)

    def on_train_end(self, trainer) -> None:
        self._channel.progress(Phase.FINAL_VAL, 1, 1, unit="steps")

    # -- internals --------------------------------------------------------- #

    def _epoch_of(self, trainer) -> int:
        """Ultralytics counts epochs from zero; users do not."""
        try:
            return int(getattr(trainer, "epoch", self.epochs_done)) + 1
        except (TypeError, ValueError):
            return self.epochs_done + 1

    def _emit_progress(self, trainer) -> None:
        epoch = self._epoch_of(trainer)
        detail = tr("epoch {epoch} of {total}", epoch=min(epoch, self._epochs), total=self._epochs)
        if self._batches_per_epoch:
            # Worked out from the epoch rather than counted up over the run, so
            # the batches of a first epoch abandoned to an out-of-memory restart
            # are not still on the bar after it starts again.
            total = self._batches_per_epoch * self._epochs
            done = (epoch - 1) * self._batches_per_epoch + self._batches_this_epoch
            done = min(max(0, done), total)
            self._channel.progress(
                Phase.TRAIN,
                done,
                total,
                unit="batches",
                eta_s=self._eta(done, total),
                detail=detail,
            )
        else:
            # No dataloader length to work from -- report in epochs, which is
            # coarse but never wrong.
            self._channel.progress(
                Phase.TRAIN,
                max(0, epoch - 1),
                self._epochs,
                unit="epochs",
                eta_s=self._eta(epoch - 1, self._epochs),
                detail=detail,
            )

    def _eta(self, done: int, total: int) -> float | None:
        """Seconds left, or nothing at all until one epoch has been timed.

        An estimate made from a partial first epoch is wrong by the cost of the
        download and the warmup, which is exactly when the user is watching it.
        """
        if self._ema_epoch_s is None or total <= 0 or done >= total:
            return None
        return round(self._ema_epoch_s * self._epochs * (1 - done / total), 1)


def _loader_length(trainer) -> int:
    try:
        return len(trainer.train_loader)
    except (AttributeError, TypeError):
        return 0


def _collect_metrics(trainer) -> dict[str, float]:
    """Everything numeric this epoch produced, under Ultralytics' own names.

    The names are kept as they come -- ``metrics/mAP50-95(B)`` and the rest --
    because they are what the user will see again in ``results.csv`` and in
    every Ultralytics tutorial. The page is what makes them readable.
    """
    values: dict[str, float] = {}

    losses = getattr(trainer, "tloss", None)
    label_loss_items = getattr(trainer, "label_loss_items", None)
    if losses is not None and callable(label_loss_items):
        try:
            values.update(_numeric(label_loss_items(losses, prefix="train")))
        except Exception:  # noqa: BLE001 - a display nicety, never the run
            pass

    values.update(_numeric(getattr(trainer, "metrics", None)))
    values.update(_numeric(getattr(trainer, "lr", None)))
    # Not ``trainer.fitness``: for a detector it is mAP50-95 under another
    # name, and that is already in ``metrics``.
    return values


def _numeric(mapping) -> dict[str, float]:
    if not isinstance(mapping, dict):
        return {}
    found = {}
    for key, value in mapping.items():
        number = _as_float(value)
        if number is not None:
            found[str(key)] = round(number, 6)
    return found


def _as_float(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        # torch tensors and numpy scalars both answer float() when they hold one
        # number, and raise when they do not -- which is the check we want.
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):  # NaN or infinity
        return None
    return number


def _epoch_line(epoch: int, epochs: int, values: dict, best: bool) -> str:
    parts = [tr("Epoch {epoch}/{total}", epoch=epoch, total=epochs)]
    for key, label in (("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")):
        if key in values:
            parts.append(f"{label} {values[key]:.4f}")
    if best:
        parts.append(tr("(best so far)"))
    return "  ".join(parts)


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# --------------------------------------------------------------------------- #
# INT8 quantisation-aware training
# --------------------------------------------------------------------------- #


def _require_qat_toolkit() -> None:
    """Fail now, not after the first epoch, if QAT cannot happen.

    Ultralytics does the quantisation-aware training itself (``quantize=8`` at
    train time), on top of NVIDIA's Model Optimizer, which the app installs with
    everything else. Both halves are checked. Model Optimizer is optional -- an
    environment that cannot import it is perfectly good for Steps 2 to 7 and
    only this toggle stops working -- and an Ultralytics from before native QAT
    would carry ``quantize`` through training untouched. Either way the failure
    has to be caught here and said out loud. The alternative is training an
    ordinary model, labelling it quantised, and having that discovered, if ever,
    as a calibration prompt at export or several points of accuracy lost on the
    device it was exported for.
    """
    try:
        import modelopt.torch.quantization  # noqa: F401
    except ImportError as exc:
        raise QatUnavailable(
            tr(
                "INT8 quantisation-aware training needs the NVIDIA TensorRT Model "
                "Optimizer, and this environment cannot import it ({error}).",
                error=exc,
            )
        ) from exc
    try:
        from ultralytics.utils.torch_utils import prepare_qat  # noqa: F401
    except ImportError as exc:
        raise QatUnavailable(
            tr(
                "This version of Ultralytics cannot do INT8 quantisation-aware training. "
                "Update it in Step 1."
            )
        ) from exc


def _prepare_kernel_toolchain() -> dict:
    """On Windows, find the compilers Model Optimizer's INT8 kernel is built with.

    Linux has a C++ compiler where torch looks for one. Windows has neither of
    the two it needs where torch looks -- see ``worker/util/toolchain.py`` --
    so they are found here and put where torch will look. It has to happen
    before Model Optimizer is imported, which is why it is separate from the
    warning: that waits until :func:`_require_qat_toolkit` has said whether
    QAT can happen at all. Returns what was found, ``{}`` where nothing needed
    finding.
    """
    if os.name != "nt":
        return {}
    try:
        import torch
    except Exception:  # noqa: BLE001 - _require_qat_toolkit reports a broken torch
        return {}
    torch_cuda = torch.version.cuda
    if not torch_cuda:
        # A CPU build of torch: Model Optimizer does not try to compile anything.
        return {}
    return {
        "torch_cuda": torch_cuda,
        "msvc": toolchain.activate_msvc(),
        "cuda_toolkit": toolchain.select_cuda_home(torch_cuda) is not None,
    }


def _warn_if_kernels_cannot_build(channel: Channel, kernel_toolchain: dict) -> None:
    """Say now if QAT is about to run several times slower than it should.

    Model Optimizer's fake-quantisation has a CUDA kernel, compiled on first use
    through ``torch.utils.cpp_extension``. That needs ninja -- and it looks for
    it by *running* ``ninja --version``, not by importing it, so a ninja sitting
    in the environment's script directory is invisible unless that directory is
    on PATH. On Windows it also needs the Microsoft C++ compiler and a CUDA
    Toolkit, which the app cannot install. When it cannot build, Model Optimizer
    catches the failure and falls back to a pure-PyTorch implementation: the run
    is correct and stays on the GPU, it is simply much slower, in a warning
    buried among a few hundred lines of Ultralytics output.

    A slow run that looks identical to a fast one is the kind of thing nobody
    ever investigates, so it is said here, once, in the app's own voice.
    """
    if kernel_toolchain.get("msvc") is False:
        channel.warning(
            "train",
            tr(
                "The INT8 quantisation kernels cannot be compiled: on Windows they need "
                "the Microsoft C++ compiler, and it was not found. Training will still be "
                "correct, but the quantised layers fall back to a slower implementation."
            ),
            hint=tr(
                "Install the Visual Studio Build Tools with the \"Desktop development "
                "with C++\" workload, then start the run again. The app finds the "
                "compiler by itself; nothing needs to be added to PATH."
            ),
        )
    if kernel_toolchain.get("cuda_toolkit") is False:
        version = kernel_toolchain["torch_cuda"].split(".")[0] + ".x"
        channel.warning(
            "train",
            tr(
                "The INT8 quantisation kernels cannot be compiled: on Windows they need "
                "the NVIDIA CUDA Toolkit {version}, and it was not found. Training will "
                "still be correct, but the quantised layers fall back to a slower "
                "implementation.",
                version=version,
            ),
            hint=tr(
                "Install CUDA Toolkit {version} from NVIDIA, then start the run again. "
                "It can sit beside any other version: the app picks the one that "
                "matches.",
                version=version,
            ),
        )
    if shutil.which("ninja"):
        return
    channel.warning(
        "train",
        tr(
            "ninja is not on this environment's PATH, so the INT8 quantisation kernels "
            "cannot be compiled. Training will still be correct, but the quantised "
            "layers fall back to a slower implementation."
        ),
        hint=tr(
            "This is the app's mistake, not a missing package. Report it with the "
            "log file if this run is slower than you expect."
        ),
    )


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


def load_yolo(model_id: str, *, channel: Channel | None = None):
    """Load an Ultralytics checkpoint. Replaced wholesale in tests.

    A module-level function rather than an import at the top of the file for the
    same reason Step 3's is: the tests have no GPU, no torch and no appetite for
    a checkpoint download, and everything worth testing here is what happens
    around the model rather than inside it.
    """
    # A missing ultralytics is deliberately left to propagate: the entry point
    # turns it into "the managed environment is missing 'ultralytics'", which
    # names both the problem and the Repair button that fixes it.
    from ultralytics import YOLO

    try:
        return YOLO(model_id)
    except Exception as exc:  # noqa: BLE001 - a download failure arrives as many types
        raise ModelUnavailable(
            tr("{model} could not be loaded: {error}", model=model_id, error=exc)
        ) from exc


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)
