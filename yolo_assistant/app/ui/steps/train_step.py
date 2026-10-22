"""Step 5: train a detector on the labelled photos.

This is the longest-running thing the app does, and the page is shaped by two
consequences of that.

**Every refusal happens before the button is pressed.** An image size that is
not a multiple of 32, a misspelt setting, a run name that is not a legal folder
-- each of those would otherwise surface minutes in, from inside Ultralytics, as
a message about something the user never typed. They are checked as the user
types instead, and Start stays disabled with the reason underneath it.

**While it runs, the page answers "should I let this carry on?"** That is not
the same question as "what is it doing", and it is not answered by a progress
bar. It is answered by the last epoch's accuracy next to the best epoch's, and
by the GPU meters -- a card at 30% utilisation is being starved by the
dataloader, and an hour spent finding that out is an hour well spent.

Training is also the one step that can be skipped outright: a project that
already holds a trained model can go straight to Step 6, which is why the runs
already on disk are listed here rather than only in Step 6.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core import project as project_model
from ...core import runtime
from ...core import training as training_model
from ...core.i18n import N_, tr, trn
from ...core.job import JobRunner, JobState
from ...core.project import Project
from ...core.training import MODELS, TrainPlan
from ..common.doc_link import doc_link
from ..common.log_console import LogConsole
from ..common.metrics_panel import MetricsPanel
from ..common.param_form import ParamForm, apply_help
from ..common.progress_header import ProgressHeader
from ..common.status_row import BAD, BUSY, OK, UNKNOWN, WARN, StatusRow
from ..common.telemetry_panel import TelemetryPanel
from .base import StepPage

TRAIN_DOCS_URL = "https://docs.ultralytics.com/modes/train"


class TrainStep(StepPage):
    title = N_("Training")
    subtitle = N_(
        "Train a YOLO detector on your labelled photos. It runs on your own GPU and can "
        "take hours; you can stop at any point and keep the best epoch."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._runner = JobRunner(self)
        self._project: Project | None = None
        self._status = project_model.status(None)
        self._runs: tuple = ()
        self._plan = TrainPlan(status=self._status)
        self._adopted: str | None = None
        self._failure = ""

        self._build_ui()
        self._wire()
        self._adopt_project()

    # -- construction ------------------------------------------------------ #

    def _build_ui(self) -> None:
        heading = QLabel(f"<h2>{tr(self.title)}</h2>")
        self._docs_link = doc_link(TRAIN_DOCS_URL, tr("Ultralytics training guide"))
        subtitle = QLabel(tr(self.subtitle))
        subtitle.setWordWrap(True)

        settings_box = QGroupBox(tr("Training"))
        form = ParamForm(settings_box)

        self._model = QComboBox()
        for model in MODELS:
            self._model.addItem(model.caption, model.key)
            self._model.setItemData(
                self._model.count() - 1, model.detail, Qt.ItemDataRole.ToolTipRole
            )
        form.add_row("train.model", tr("Model"), self._model)

        self._name = QLineEdit()
        self._name.setPlaceholderText(tr("a name for this run, used as its folder"))
        form.add_row("train.output_name", tr("Output name"), self._name)

        self._epochs = QSpinBox()
        self._epochs.setRange(training_model.MIN_EPOCHS, training_model.MAX_EPOCHS)
        self._epochs.setValue(training_model.DEFAULT_EPOCHS)
        form.add_row("train.epochs", tr("Epochs"), self._epochs)

        # One number: Ultralytics trains on squares. The shape the model runs
        # at is Step 6's question.
        self._imgsz = QSpinBox()
        self._imgsz.setRange(training_model.MIN_IMGSZ, training_model.MAX_IMGSZ)
        self._imgsz.setSingleStep(training_model.STRIDE)
        self._imgsz.setValue(training_model.DEFAULT_IMGSZ)
        self._imgsz.setSuffix(" px")
        form.add_row("train.imgsz", tr("Image size"), self._imgsz)

        self._qat = QCheckBox(tr("Train for INT8 (quantisation-aware training)"))
        form.add_row("train.quantize", "INT8 QAT", self._qat)

        self._extra = QLineEdit()
        self._extra.setPlaceholderText(tr("for example:  batch=8, patience=20"))
        form.add_row("train.additional_args", tr("Additional settings"), self._extra)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._problems = QLabel("")
        self._problems.setWordWrap(True)
        self._problems.setVisible(False)
        self._warnings = QLabel("")
        self._warnings.setWordWrap(True)
        self._warnings.setVisible(False)

        self._runs_row = StatusRow(tr("Trained models"))
        apply_help(self._runs_row.action, "train.runs")

        self._start_button = QPushButton(tr("Start training"))
        apply_help(self._start_button, "train.start")
        self._cancel_button = QPushButton(tr("Stop"))
        self._cancel_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self._start_button)
        buttons.addWidget(self._cancel_button)
        buttons.addStretch(1)

        self._progress = ProgressHeader()
        self._metrics = MetricsPanel()
        self._telemetry = TelemetryPanel()
        self._console = LogConsole()

        live = QHBoxLayout()
        live.setContentsMargins(0, 0, 0, 0)
        live.addWidget(self._metrics, 3)
        live.addWidget(self._telemetry, 2)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self._summary)
        top_layout.addWidget(settings_box)
        top_layout.addWidget(self._problems)
        top_layout.addWidget(self._warnings)
        top_layout.addWidget(self._runs_row)
        top_layout.addLayout(buttons)
        top_layout.addWidget(self._progress)
        top_layout.addLayout(live, 1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self._console)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        header = QHBoxLayout()
        header.addWidget(heading)
        header.addStretch(1)
        header.addWidget(self._docs_link)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(subtitle)
        layout.addWidget(splitter, 1)

    def _wire(self) -> None:
        self._start_button.clicked.connect(self._start)
        self._cancel_button.clicked.connect(self._runner.cancel)
        self._runs_row.action.clicked.connect(self._open_runs_folder)

        # None of these change what is on disk, so none of them rescan it. The
        # additional-settings box in particular fires per keystroke, and the
        # scan behind it walks every class folder in the dataset.
        self._model.currentIndexChanged.connect(lambda _index: self.refresh())
        self._name.textChanged.connect(lambda _text: self.refresh())
        self._epochs.valueChanged.connect(lambda _value: self.refresh())
        self._imgsz.valueChanged.connect(lambda _value: self.refresh())
        self._qat.toggled.connect(lambda _checked: self.refresh())
        self._extra.textChanged.connect(lambda _text: self.refresh())

        self._runner.state_changed.connect(self._on_state)
        self._runner.progress.connect(self._progress.update_progress)
        self._runner.metrics.connect(self._metrics.add_epoch)
        self._runner.log_line.connect(self._console.append_line)
        self._runner.artifact.connect(self._on_artifact)
        self._runner.warned.connect(
            lambda code, msg, hint: self._console.append_line(msg, "warning")
        )
        self._runner.failed.connect(self._on_failed)
        self._runner.finished.connect(self._on_finished)
        self._runner.stalled.connect(
            lambda quiet: self._console.append_line(
                tr("No output for {quiet:.0f}s -- the run may be stuck.", quiet=quiet), "warning"
            )
        )

    # -- state ------------------------------------------------------------- #

    def _adopt_project(self) -> None:
        """Take up whichever project the earlier steps left current.

        Settings are restored only when the project changes, so that coming back
        to this page does not undo an epoch count the user just typed.
        """
        self._project = project_model.current()

        root = str(self._project.root) if self._project else None
        if root == self._adopted:
            self._rescan()
            return
        self._adopted = root
        self._failure = ""

        settings = training_model.read_settings(self._project)
        for widget, value in (
            (self._epochs, settings["epochs"]),
            (self._imgsz, settings["imgsz"]),
        ):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        index = self._model.findData(settings["model"])
        if index >= 0:
            self._model.blockSignals(True)
            self._model.setCurrentIndex(index)
            self._model.blockSignals(False)
        for widget, value in ((self._qat, settings["qat"]),):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)
        self._extra.blockSignals(True)
        self._extra.setText(settings["args"])
        self._extra.blockSignals(False)

        # A name that is free right now, rather than the one used last time:
        # offering a name that would replace an existing run as the default is
        # how results get lost.
        self._name.blockSignals(True)
        self._name.setText(training_model.suggest_name(self._project))
        self._name.blockSignals(False)

        # Last, so that the plan is built from the restored settings rather than
        # from whatever the previous project left in the widgets.
        self._rescan()

    def _rescan(self) -> None:
        """Re-read what is on disk, and tell the wizard if the gates moved.

        Kept apart from :meth:`refresh` because the two have very different
        costs. Reading the dataset's state walks every class folder counting
        photos and label files, which is milliseconds on a small dataset and
        noticeable on a large one -- and nothing the user types on this page can
        change the answer. Typing repaints; arriving, finishing and failing
        rescan.
        """
        self._status = project_model.status(self._project)
        self._runs = training_model.list_runs(self._project)
        self.refresh()
        self.gates_changed.emit()

    def refresh(self) -> None:
        self._plan = training_model.build_plan(
            self._status,
            project=self._project,
            runs=self._runs,
            model=str(self._model.currentData() or training_model.DEFAULT_MODEL),
            name=self._name.text(),
            epochs=self._epochs.value(),
            imgsz=self._imgsz.value(),
            qat=self._qat.isChecked(),
            extra_text=self._extra.text(),
        )
        self._paint_summary()
        self._paint_runs_row()
        self._paint_buttons()

    def _paint_summary(self) -> None:
        self._summary.setText(self._plan.summary())
        _paint_notes(self._problems, self._plan.problems(), "#c0392b", "x")
        _paint_notes(self._warnings, self._plan.warnings(), "#b9770e", "!")

    def _paint_runs_row(self) -> None:
        self._runs_row.set_action(
            tr("Open folder"), visible=self._project is not None, enabled=self._project is not None
        )
        if self._busy:
            return  # the progress line says something more useful
        if self._failure:
            self._runs_row.set_state(BAD, _escape(self._failure))
            return
        if self._project is None or not self._status.is_prepared:
            self._runs_row.set_state(UNKNOWN, tr("No prepared dataset yet."))
            return

        trained = self._plan.trained
        if not trained:
            started = len(self._plan.runs)
            if started:
                self._runs_row.set_state(
                    WARN,
                    trn(
                        "{n} run(s) were started here but none saved a model. "
                        "Training has to finish at least one epoch to leave something "
                        "Step 6 can export.",
                        "{n} run(s) were started here but none saved a model. "
                        "Training has to finish at least one epoch to leave something "
                        "Step 6 can export.",
                        started,
                    ),
                )
            else:
                self._runs_row.set_state(
                    UNKNOWN, tr("Nothing has been trained in this project yet.")
                )
            return

        newest = _escape(trained[0].caption())
        others = len(trained) - 1
        if others:
            text = trn(
                "{run} and {n} other. Step 6 can export any of them, so "
                "training again is optional.",
                "{run} and {n} others. Step 6 can export any of them, so "
                "training again is optional.",
                others,
                run=newest,
            )
        else:
            text = tr(
                "{run}. Step 6 can export any of them, so training again is optional.",
                run=newest,
            )
        self._runs_row.set_state(OK, text)

    def _paint_buttons(self) -> None:
        busy = self._busy
        self._start_button.setEnabled(not busy and self._plan.is_runnable)
        epochs = self._plan.epochs
        self._start_button.setText(
            trn("Start training - {n} epoch", "Start training - {n} epochs", epochs)
        )
        self._cancel_button.setEnabled(busy)
        for widget in (
            self._model,
            self._name,
            self._epochs,
            self._imgsz,
            self._qat,
            self._extra,
        ):
            widget.setEnabled(not busy)

    @property
    def _busy(self) -> bool:
        return self._runner.is_active()

    # -- running ----------------------------------------------------------- #

    def _start(self) -> None:
        if self._busy or self._project is None or not self._plan.is_runnable:
            return
        if not self._confirm_overwrite():
            return

        self._console.clear()
        self._console.append_line(runtime.describe_python(), "info")
        self._failure = ""
        self._metrics.clear()
        self._metrics.set_total_epochs(self._plan.epochs)
        self._progress.start()

        try:
            spec = runtime.job_spec(
                "train",
                self._plan.job(self._project),
                project=self._project,
                log_name=f"train-{self._plan.name.strip() or 'run'}",
            )
        except runtime.EnvironmentMissing as exc:
            self._progress.stop(tr("Not started"))
            QMessageBox.warning(self, tr("Environment not ready"), str(exc))
            self._rescan()
            return

        training_model.write_settings(self._project, self._plan)
        self._runner.start(spec)
        self._paint_buttons()

    def _confirm_overwrite(self) -> bool:
        """Ask before replacing a run that already holds a trained model."""
        existing = self._plan.existing_run
        if existing is None or not existing.has_weights:
            return True
        answer = QMessageBox.question(
            self,
            tr("Replace this run?"),
            tr(
                '"{name}" already holds a trained model, and starting this run '
                "deletes it along with everything else in that folder.\n\n"
                "Give this run a different name to keep both.\n\nReplace it?",
                name=existing.name,
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return int(answer) == int(QMessageBox.StandardButton.Yes)

    def _open_runs_folder(self) -> None:
        if self._project is None:
            return
        self._project.runs_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._project.runs_dir)))

    # -- slots ------------------------------------------------------------- #

    def _on_state(self, state: JobState) -> None:
        self._paint_buttons()
        if state is JobState.RUNNING:
            self._runs_row.set_state(BUSY, tr("Training..."))
        elif state is JobState.CANCELLING:
            self._progress.stop(tr("Stopping..."))
            self._runs_row.set_state(BUSY, tr("Stopping..."))

    def _on_artifact(self, kind: str, path: str, meta: dict) -> None:
        if kind == "weights":
            self._console.append_line(tr("Saved {path}", path=path), "info")

    def _on_failed(self, code: str, msg: str, hint: str, traceback: str) -> None:
        self._console.append_line(f"[{code}] {msg}", "error")
        if hint:
            self._console.append_line(hint, "warning")
        self._failure = f"{msg} {hint}".strip()
        self._runs_row.set_state(BAD, _escape(self._failure))
        QMessageBox.warning(self, tr("Training failed"), f"{msg}\n\n{hint}".strip())

    def _on_finished(self, ok: bool, payload: dict) -> None:
        if ok:
            self._failure = ""
            epochs = int(payload.get("epochs") or 0)
            # Ultralytics' fitness for a detector is mAP50-95 alone, so it is
            # reported under the name the results table uses.
            best_map = float(payload.get("best_fitness") or 0.0)
            if best_map:
                done = trn(
                    "Trained {n} epoch - best mAP50-95 {best:.4f}",
                    "Trained {n} epochs - best mAP50-95 {best:.4f}",
                    epochs,
                    best=best_map,
                )
            else:
                done = trn("Trained {n} epoch", "Trained {n} epochs", epochs)
            self._progress.stop(done)
            # The name is free again only once this run is on disk, so the next
            # one gets a fresh suggestion rather than an offer to overwrite.
            self._name.blockSignals(True)
            self._name.setText(training_model.suggest_name(self._project))
            self._name.blockSignals(False)
        elif payload.get("cancelled"):
            self._progress.stop(tr("Stopped"))
            self._console.append_line(
                tr(
                    "Stopped. The last epoch Ultralytics saved is still in the run folder "
                    "and can be exported in Step 6."
                ),
                "warning",
            )
        else:
            self._progress.stop(tr("Failed"))
        self._rescan()

    # -- StepPage ---------------------------------------------------------- #

    def can_enter(self) -> bool:
        if runtime.worker_python() is None:
            return False
        project = project_model.current()
        if project is None or not project.prepared_file.is_file():
            return False
        # Checking labels in Step 4 is optional; having them is not. A photo
        # with no label file is simply not in the training set, so a half
        # labelled dataset would train on a silently smaller one.
        return project_model.status(project).fully_labelled

    def is_complete(self) -> bool:
        """Whether Step 6 has something to export.

        Not "the user pressed Start": a project whose weights were copied in
        from another machine is as trained as one this app produced, and a run
        that was stopped after ten of a hundred epochs still left a usable model.
        """
        return training_model.has_trained_model(self._project or project_model.current())

    def blocked_reason(self) -> str:
        if runtime.worker_python() is None:
            return tr(
                "Step 1 has to finish setting up the Python environment before "
                "training can run."
            )
        status = project_model.status(project_model.current())
        if not status.is_prepared:
            return tr("Prepare your photos in Step 2 first; there is nothing to train on yet.")
        missing = status.total_images - status.total_labels
        return trn(
            "{n:,} photo(s) still have no labels. Finish autolabeling in Step 3 "
            "first -- a photo with no label file is left out of training entirely.",
            "{n:,} photo(s) still have no labels. Finish autolabeling in Step 3 "
            "first -- a photo with no label file is left out of training entirely.",
            missing,
        )

    def on_enter(self) -> None:
        if self._busy:
            return
        # Runs appear and disappear between visits: Step 6 is not the only thing
        # that writes here, and users delete folders in a file manager.
        self._adopt_project()

    def is_busy(self) -> bool:
        return self._busy

    def shutdown(self) -> None:
        if self._runner.is_active():
            self._runner.cancel()
        self._telemetry.shutdown()

    # -- for later steps --------------------------------------------------- #

    @property
    def project(self) -> Project | None:
        return self._project


def _paint_notes(label: QLabel, lines: list[str], colour: str, marker: str) -> None:
    if not lines:
        label.setVisible(False)
        return
    joined = f"<br>{marker} ".join(_escape(line) for line in lines)
    label.setText(f"<span style='color:{colour}'>{marker} {joined}</span>")
    label.setVisible(True)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
