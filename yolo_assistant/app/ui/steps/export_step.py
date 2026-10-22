"""Step 6: convert a trained checkpoint into something deployable.

The page is shaped by what makes export hard, which is not the waiting -- ONNX
is seconds, and even a TensorRT engine is minutes next to Step 5's hours. It is
that a wrong export looks exactly like a right one. A file appears, it is the
size a model should be, and it is only on the target device, or in Step 7, that
it turns out to be full precision when INT8 was asked for, or built for a GPU
that is not the one it will run on.

So the page spends its effort on saying what the file will be before it exists:
which checkpoint, which format, what precision, what size, and under what name.
The combinations that cannot exist are refused as they are picked -- an INT8
ONNX, an INT8 QAT model at FP16 -- and the ones that merely deserve a second
thought are said out loud and left alone.

Like Step 5, this step can be skipped: a project that already holds an export
can go straight to Step 7, which is why what is in ``exports/`` is listed here
rather than only there.
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
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core import export as export_model
from ...core import project as project_model
from ...core import runtime
from ...core import training as training_model
from ...core.export import FORMATS, PRECISIONS, ExportPlan
from ...core.i18n import N_, tr, trn
from ...core.job import JobRunner, JobState
from ...core.project import Project
from ..common.doc_link import doc_link
from ..common.log_console import LogConsole
from ..common.param_form import ParamForm, apply_help
from ..common.progress_header import ProgressHeader
from ..common.status_row import BAD, BUSY, OK, UNKNOWN, StatusRow
from .base import StepPage

EXPORT_DOCS_URL = "https://docs.ultralytics.com/modes/export"


class ExportStep(StepPage):
    title = N_("Export")
    subtitle = N_(
        "Convert a trained model into a deployment format. Minutes at most, and the "
        "result is a file you can run outside this app."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._runner = JobRunner(self)
        self._project: Project | None = None
        self._runs: tuple = ()
        self._exports: tuple = ()
        self._plan = ExportPlan()
        self._adopted: str | None = None
        self._failure = ""
        #: Set while the height/width boxes are being filled in from a run, so
        #: that doing so does not count as the user overriding the run's size.
        self._following_run = True

        self._build_ui()
        self._wire()
        self._adopt_project()

    # -- construction ------------------------------------------------------ #

    def _build_ui(self) -> None:
        heading = QLabel(f"<h2>{tr(self.title)}</h2>")
        self._docs_link = doc_link(EXPORT_DOCS_URL, tr("Ultralytics export guide"))
        subtitle = QLabel(tr(self.subtitle))
        subtitle.setWordWrap(True)

        settings_box = QGroupBox(tr("Export"))
        form = ParamForm(settings_box)

        self._checkpoint = QComboBox()
        # Filled in after the page is first shown, so a combo sized to its
        # contents would size itself to being empty and elide every run name.
        self._checkpoint.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._checkpoint.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow
        )
        self._checkpoint.setMinimumContentsLength(18)
        form.add_row("export.model", tr("Model"), self._checkpoint)

        self._format = QComboBox()
        for fmt in FORMATS:
            self._format.addItem(fmt.caption, fmt.key)
            self._format.setItemData(
                self._format.count() - 1, fmt.detail, Qt.ItemDataRole.ToolTipRole
            )
        form.add_row("export.format", tr("Format"), self._format)

        self._precision = QComboBox()
        for precision in PRECISIONS:
            self._precision.addItem(precision.caption, precision.key)
            self._precision.setItemData(
                self._precision.count() - 1, tr(precision.note), Qt.ItemDataRole.ToolTipRole
            )
        form.add_row("export.quantize", tr("Precision"), self._precision)

        form.add_row("export.imgsz", tr("Image size"), self._build_imgsz())

        self._deepstream = QCheckBox(tr("Add the settings a DeepStream pipeline expects"))
        form.add_row("export.deepstream", "DeepStream", self._deepstream)

        self._extra = QLineEdit()
        self._extra.setPlaceholderText(tr("for example:  dynamic=true, batch=1"))
        form.add_row("export.additional_args", tr("Additional settings"), self._extra)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        # The summary is escaped but has no tags, so auto-detection would show
        # the entities themselves -- "-&gt;" for the arrow.
        self._summary.setTextFormat(Qt.TextFormat.RichText)
        self._problems = QLabel("")
        self._problems.setWordWrap(True)
        self._problems.setVisible(False)
        self._warnings = QLabel("")
        self._warnings.setWordWrap(True)
        self._warnings.setVisible(False)

        self._exports_row = StatusRow(tr("Exported files"))
        apply_help(self._exports_row.action, "export.exports")

        self._start_button = QPushButton(tr("Export"))
        apply_help(self._start_button, "export.start")
        self._cancel_button = QPushButton(tr("Stop"))
        self._cancel_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self._start_button)
        buttons.addWidget(self._cancel_button)
        buttons.addStretch(1)

        self._progress = ProgressHeader()
        self._console = LogConsole()

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self._summary)
        top_layout.addWidget(settings_box)
        top_layout.addWidget(self._problems)
        top_layout.addWidget(self._warnings)
        top_layout.addWidget(self._exports_row)
        top_layout.addLayout(buttons)
        top_layout.addWidget(self._progress)
        top_layout.addStretch(1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self._console)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        header = QHBoxLayout()
        header.addWidget(heading)
        header.addStretch(1)
        header.addWidget(self._docs_link)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(subtitle)
        layout.addWidget(splitter, 1)

    def _build_imgsz(self) -> QWidget:
        """Height and width side by side. Unlike Step 5, there are two.

        Ultralytics trains on squares and exports at whatever shape it is
        given, so this is where the model's shape is chosen -- a model that
        will see 1920x1080 frames does not want a square input. Both boxes are
        capped at the chosen run's training size; see :meth:`_apply_cap`.
        """
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)

        self._imgsz_h = QSpinBox()
        self._imgsz_w = QSpinBox()
        for spin in (self._imgsz_h, self._imgsz_w):
            spin.setRange(training_model.MIN_IMGSZ, training_model.MAX_IMGSZ)
            spin.setSingleStep(training_model.STRIDE)
            spin.setValue(training_model.DEFAULT_IMGSZ)
            spin.setSuffix(" px")
            apply_help(spin, "export.imgsz")

        for label_text, spin in ((tr("Height"), self._imgsz_h), (tr("Width"), self._imgsz_w)):
            label = QLabel(label_text)
            label.setBuddy(spin)
            row.addWidget(label)
            row.addWidget(spin)
        row.addStretch(1)
        return container

    def _wire(self) -> None:
        self._start_button.clicked.connect(self._start)
        self._cancel_button.clicked.connect(self._runner.cancel)
        self._exports_row.action.clicked.connect(self._open_exports_folder)

        # Picking a different checkpoint re-offers that run's own image size,
        # which is the only one of these that changes another field.
        self._checkpoint.currentIndexChanged.connect(self._on_checkpoint_changed)
        self._format.currentIndexChanged.connect(lambda _index: self.refresh())
        self._precision.currentIndexChanged.connect(lambda _index: self.refresh())
        self._imgsz_h.valueChanged.connect(self._on_imgsz_changed)
        self._imgsz_w.valueChanged.connect(self._on_imgsz_changed)
        self._deepstream.toggled.connect(lambda _checked: self.refresh())
        self._extra.textChanged.connect(lambda _text: self.refresh())

        self._runner.state_changed.connect(self._on_state)
        self._runner.progress.connect(self._progress.update_progress)
        self._runner.log_line.connect(self._console.append_line)
        self._runner.artifact.connect(self._on_artifact)
        self._runner.warned.connect(
            lambda code, msg, hint: self._console.append_line(msg, "warning")
        )
        self._runner.failed.connect(self._on_failed)
        self._runner.finished.connect(self._on_finished)
        self._runner.stalled.connect(
            lambda quiet: self._console.append_line(
                tr(
                    "No output for {quiet:.0f}s. An engine build is quiet for minutes at "
                    "a time, so this is not necessarily trouble.",
                    quiet=quiet,
                ),
                "warning",
            )
        )

    # -- state ------------------------------------------------------------- #

    def _adopt_project(self) -> None:
        """Take up whichever project the earlier steps left current."""
        self._project = project_model.current()

        root = str(self._project.root) if self._project else None
        if root == self._adopted:
            self._rescan()
            return
        self._adopted = root
        self._failure = ""

        settings = export_model.read_settings(self._project)
        index = self._format.findData(settings["format"])
        if index >= 0:
            self._set_index(self._format, index)
        index = self._precision.findData(settings["precision"])
        if index >= 0:
            self._set_index(self._precision, index)

        self._deepstream.blockSignals(True)
        self._deepstream.setChecked(settings["deepstream"])
        self._deepstream.blockSignals(False)
        self._extra.blockSignals(True)
        self._extra.setText(settings["args"])
        self._extra.blockSignals(False)

        # A recorded size means the user overrode the run's last time, so it is
        # restored and the run stops deciding. Zero means they never did, and
        # the chosen checkpoint's own size is the right default.
        self._following_run = not (settings["imgsz_h"] and settings["imgsz_w"])
        if not self._following_run:
            self._set_imgsz(settings["imgsz_h"], settings["imgsz_w"])

        self._rescan()

    def _rescan(self) -> None:
        """Re-read what is on disk, and tell the wizard if the gates moved.

        Runs and exports both appear and disappear between visits -- Step 5
        writes runs, this page writes exports, and users delete both in a file
        manager -- so neither is cached across a visit.
        """
        self._runs = training_model.list_runs(self._project)
        self._exports = export_model.list_exports(self._project)
        self._repopulate_checkpoints()
        self.refresh()
        self.gates_changed.emit()

    def _repopulate_checkpoints(self) -> None:
        """Rebuild the checkpoint list, keeping the selection if it survived."""
        wanted = self._checkpoint.currentData() or ""
        available = export_model.checkpoints(self._runs)

        self._checkpoint.blockSignals(True)
        self._checkpoint.clear()
        for checkpoint in available:
            self._checkpoint.addItem(checkpoint.caption(), checkpoint.key)
        index = self._checkpoint.findData(wanted)
        self._checkpoint.setCurrentIndex(max(0, index))
        self._checkpoint.blockSignals(False)

        self._apply_cap()
        if self._following_run:
            self._offer_the_runs_size()

    def _offer_the_runs_size(self) -> None:
        """Default the size to 16:9 at whatever the chosen run was trained at.

        A run trained at 1024 is offered 1024x576. A run with nothing on record
        (weights copied in from elsewhere) is offered the same shape at the
        ordinary default size.
        """
        checkpoint = self._selected_checkpoint()
        self._set_imgsz(*export_model.default_imgsz(checkpoint.run if checkpoint else None))

    def _apply_cap(self) -> None:
        """Neither side may be larger than the chosen run was trained at.

        The model has never seen anything bigger, so a larger input only costs
        time. Lowering the maximum also lowers a value above it, which is the
        point: a size kept from another run is brought within this one's.
        """
        checkpoint = self._selected_checkpoint()
        cap = export_model.max_imgsz(checkpoint.run if checkpoint else None)
        for spin in (self._imgsz_h, self._imgsz_w):
            spin.blockSignals(True)
            spin.setMaximum(cap)
            spin.blockSignals(False)

    def _selected_checkpoint(self):
        key = str(self._checkpoint.currentData() or "")
        for candidate in export_model.checkpoints(self._runs):
            if candidate.key == key:
                return candidate
        return None

    def _on_checkpoint_changed(self, _index: int) -> None:
        self._apply_cap()
        if self._following_run:
            self._offer_the_runs_size()
        self.refresh()

    def _on_imgsz_changed(self, _value: int) -> None:
        # Typing a size is how the user says they want this one rather than the
        # run's, and it has to survive picking a different checkpoint.
        self._following_run = False
        self.refresh()

    def refresh(self) -> None:
        self._plan = export_model.build_plan(
            project=self._project,
            runs=self._runs,
            exports=self._exports,
            checkpoint_key=str(self._checkpoint.currentData() or ""),
            fmt=str(self._format.currentData() or export_model.DEFAULT_FORMAT),
            precision=str(self._precision.currentData() or export_model.DEFAULT_PRECISION),
            imgsz_h=self._imgsz_h.value(),
            imgsz_w=self._imgsz_w.value(),
            deepstream=self._deepstream.isChecked(),
            extra_text=self._extra.text(),
        )
        self._paint_summary()
        self._paint_exports_row()
        self._paint_buttons()

    def _paint_summary(self) -> None:
        self._summary.setText(_escape(self._plan.summary()))
        _paint_notes(self._problems, self._plan.problems(), "#c0392b", "x")
        _paint_notes(self._warnings, self._plan.warnings(), "#b9770e", "!")

    def _paint_exports_row(self) -> None:
        self._exports_row.set_action(
            tr("Open folder"), visible=self._project is not None, enabled=self._project is not None
        )
        if self._busy:
            return  # the progress line says something more useful
        if self._failure:
            self._exports_row.set_state(BAD, _escape(self._failure))
            return
        if not self._exports:
            self._exports_row.set_state(
                UNKNOWN, tr("Nothing has been exported from this project yet.")
            )
            return

        newest = _escape(self._exports[0].caption())
        others = len(self._exports) - 1
        if others:
            detail = trn(
                "{newest} and {n} other. Step 7 can test any of them, so "
                "exporting again is optional.",
                "{newest} and {n} others. Step 7 can test any of them, so "
                "exporting again is optional.",
                others,
                newest=newest,
            )
        else:
            detail = tr(
                "{newest}. Step 7 can test any of them, so exporting again is optional.",
                newest=newest,
            )
        self._exports_row.set_state(OK, detail)

    def _paint_buttons(self) -> None:
        busy = self._busy
        self._start_button.setEnabled(not busy and self._plan.is_runnable)
        name = self._plan.output_name
        self._start_button.setText(tr("Export to {name}", name=name) if name else tr("Export"))
        self._cancel_button.setEnabled(busy)
        for widget in (
            self._checkpoint,
            self._format,
            self._precision,
            self._imgsz_h,
            self._imgsz_w,
            self._deepstream,
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
        self._progress.start()

        try:
            spec = runtime.job_spec(
                "export",
                self._plan.job(self._project),
                project=self._project,
                log_name=f"export-{self._plan.output_name or 'model'}",
            )
        except runtime.EnvironmentMissing as exc:
            self._progress.stop(tr("Not started"))
            QMessageBox.warning(self, tr("Environment not ready"), str(exc))
            self._rescan()
            return

        export_model.write_settings(self._project, self._plan)
        self._runner.start(spec)
        self._paint_buttons()

    def _confirm_overwrite(self) -> bool:
        """Ask before replacing an export that is already there.

        Cheaper to redo than a training run, and still worth asking about: the
        one being replaced may be the file someone has already deployed from,
        and an engine build is not nothing.
        """
        existing = self._plan.existing_export
        if existing is None:
            return True
        answer = QMessageBox.question(
            self,
            tr("Replace this export?"),
            tr(
                '"{name}" already exists and will be replaced.\n\n'
                "Nothing else in the exports folder is touched.\n\nReplace it?",
                name=existing.name,
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return int(answer) == int(QMessageBox.StandardButton.Yes)

    def _open_exports_folder(self) -> None:
        if self._project is None:
            return
        self._project.exports_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._project.exports_dir)))

    # -- slots ------------------------------------------------------------- #

    def _on_state(self, state: JobState) -> None:
        self._paint_buttons()
        if state is JobState.RUNNING:
            self._exports_row.set_state(BUSY, tr("Exporting..."))
        elif state is JobState.CANCELLING:
            self._progress.stop(tr("Stopping..."))
            self._exports_row.set_state(BUSY, tr("Stopping..."))

    def _on_artifact(self, kind: str, path: str, meta: dict) -> None:
        if kind == "export":
            self._console.append_line(tr("Wrote {path}", path=path), "info")

    def _on_failed(self, code: str, msg: str, hint: str, traceback: str) -> None:
        self._console.append_line(f"[{code}] {msg}", "error")
        if hint:
            self._console.append_line(hint, "warning")
        self._failure = f"{msg} {hint}".strip()
        self._exports_row.set_state(BAD, _escape(self._failure))
        # The specification asks for the QAT mismatch in particular to be put
        # in front of the user rather than left in the log, and the remedy is
        # one control away, so the dialog names it.
        title = (
            tr("Export this model as INT8")
            if code == "qat_format_mismatch"
            else tr("Export failed")
        )
        QMessageBox.warning(self, title, f"{msg}\n\n{hint}".strip())

    def _on_finished(self, ok: bool, payload: dict) -> None:
        if ok:
            self._failure = ""
            name = str(payload.get("name") or "")
            self._progress.stop(
                tr("Exported {name}", name=name) if name else tr("Exported the model")
            )
        elif payload.get("cancelled"):
            self._progress.stop(tr("Stopped"))
            self._console.append_line(
                tr("Stopped. Nothing was written to the exports folder."), "warning"
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
        return training_model.has_trained_model(project)

    def is_complete(self) -> bool:
        """Whether Step 7 has something to test.

        Not "the user pressed Export": a file dropped into ``exports/`` by hand
        is as exportable as one this app produced, which is also why a single
        one is enough -- the specification says so in as many words.
        """
        return export_model.has_export(self._project or project_model.current())

    def blocked_reason(self) -> str:
        if runtime.worker_python() is None:
            return tr(
                "Step 1 has to finish setting up the Python environment before "
                "anything can be exported."
            )
        project = project_model.current()
        if project is None or not project.prepared_file.is_file():
            return tr("Prepare your photos in Step 2 first; there is nothing to export yet.")
        return tr(
            "There is no trained model to export. Finish training in Step 5, or copy "
            "a run into the project's runs folder."
        )

    def on_enter(self) -> None:
        if self._busy:
            return
        self._adopt_project()

    def is_busy(self) -> bool:
        return self._busy

    def shutdown(self) -> None:
        if self._runner.is_active():
            self._runner.cancel()

    # -- for later steps --------------------------------------------------- #

    @property
    def project(self) -> Project | None:
        return self._project

    # -- internals --------------------------------------------------------- #

    @staticmethod
    def _set_index(combo: QComboBox, index: int) -> None:
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _set_imgsz(self, height: int, width: int) -> None:
        for spin, value in ((self._imgsz_h, height), (self._imgsz_w, width)):
            spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(False)


def _paint_notes(label: QLabel, lines: list[str], colour: str, marker: str) -> None:
    if not lines:
        label.setVisible(False)
        return
    joined = f"<br>{marker} ".join(_escape(line) for line in lines)
    label.setText(f"<span style='color:{colour}'>{marker} {joined}</span>")
    label.setVisible(True)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
