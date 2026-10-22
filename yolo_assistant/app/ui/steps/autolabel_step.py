"""Step 3: describe each class in a sentence, and let OWLv2 draw the boxes.

The hard part of this page is not running the model. It is that the user is
about to commit to something that takes minutes or hours, on the strength of a
sentence they have never written before, and whose result they cannot see until
it is over. Three things follow from that.

**The prompt column is the page.** Everything else has a sensible default. The
sentences are filled in from the class names so that Start works immediately,
and they are kept in the dataset so that coming back tomorrow does not mean
typing them again.

**It says exactly how much work it is about to do.** Photos already labelled are
skipped, so the number on the button is the number of photos that will actually
go through the model -- which is also why stopping and resuming costs nothing.

**It reports the failure that does not look like one.** A prompt that matches
nothing produces an empty label file for every photo: no error, no crash, and a
dataset that teaches the model that its own subject is not there. The share of
photos where nothing was found is reported per class, loudly.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core import autolabel as autolabel_model
from ...core import project as project_model
from ...core import runtime
from ...core.autolabel import MODELS, Plan
from ...core.i18n import N_, tr, trn
from ...core.job import JobRunner, JobState
from ...core.project import Project
from ..common.log_console import LogConsole
from ..common.param_form import ParamForm, apply_help, help_texts
from ..common.progress_header import ProgressHeader
from ..common.status_row import BAD, BUSY, OK, UNKNOWN, WARN, StatusRow
from ..common.telemetry_panel import TelemetryPanel
from .base import StepPage

CLASS_COLUMN, PHOTOS_COLUMN, LABELLED_COLUMN, PROMPT_COLUMN = range(4)


class AutolabelStep(StepPage):
    title = N_("Autolabeling")
    subtitle = N_(
        "Describe what to look for in each class, in plain English, and OWLv2 finds it "
        "and draws the boxes. It runs on your own GPU; nothing leaves this computer."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._runner = JobRunner(self)
        self._project: Project | None = None
        self._status = project_model.status(None)
        self._prompts: dict[str, str] = {}
        self._plan = Plan()
        self._adopted: str | None = None
        self._painted: tuple = ()
        self._failure = ""
        self._loading = False

        self._build_ui()
        self._wire()
        self._adopt_project()
        self.refresh()

    # -- construction ------------------------------------------------------ #

    def _build_ui(self) -> None:
        heading = QLabel(f"<h2>{tr(self.title)}</h2>")
        subtitle = QLabel(tr(self.subtitle))
        subtitle.setWordWrap(True)

        classes_box = QGroupBox(tr("Classes"))
        self._summary = QLabel(tr("No prepared dataset yet."))
        self._summary.setWordWrap(True)
        self._problems = QLabel("")
        self._problems.setWordWrap(True)
        self._problems.setVisible(False)
        self._table = self._build_table()

        classes_layout = QVBoxLayout(classes_box)
        classes_layout.addWidget(self._summary)
        classes_layout.addWidget(self._problems)
        classes_layout.addWidget(self._table, 1)

        settings_box = QGroupBox(tr("Settings"))
        form = ParamForm(settings_box)

        self._model = QComboBox()
        for model in MODELS:
            self._model.addItem(model.caption, model.key)
        form.add_row("autolabel.model", tr("Model"), self._model)

        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.01, 0.95)
        self._threshold.setSingleStep(0.05)
        self._threshold.setDecimals(2)
        self._threshold.setValue(autolabel_model.DEFAULT_THRESHOLD)
        form.add_row("autolabel.threshold", tr("Confidence needed"), self._threshold)

        self._max_detections = QSpinBox()
        self._max_detections.setRange(1, 100)
        self._max_detections.setValue(autolabel_model.DEFAULT_MAX_DETECTIONS)
        form.add_row(
            "autolabel.max_detections", tr("Most boxes per photo"), self._max_detections
        )

        self._relabel = QCheckBox(tr("Label everything again, replacing existing labels"))
        form.add_row("autolabel.relabel", tr("Redo"), self._relabel)

        self._labels_row = StatusRow(tr("Labels"))

        self._start_button = QPushButton(tr("Start autolabeling"))
        apply_help(self._start_button, "autolabel.start")
        self._cancel_button = QPushButton(tr("Cancel"))
        self._cancel_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self._start_button)
        buttons.addWidget(self._cancel_button)
        buttons.addStretch(1)

        self._progress = ProgressHeader()
        self._telemetry = TelemetryPanel()
        self._console = LogConsole()

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(classes_box, 1)
        top_layout.addWidget(settings_box)
        top_layout.addWidget(self._labels_row)
        top_layout.addLayout(buttons)
        top_layout.addWidget(self._progress)
        top_layout.addWidget(self._telemetry)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self._console)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(subtitle)
        layout.addWidget(splitter, 1)

    def _build_table(self) -> QTableWidget:
        table = QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(
            [tr("Class"), tr("Photos"), tr("Labelled"), tr("What to look for")]
        )
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # Only the prompt cells are editable; the counts come off the filesystem
        # and typing over them would be meaningless.
        table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        header = table.horizontalHeader()
        header.setSectionResizeMode(CLASS_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(PHOTOS_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(LABELLED_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(PROMPT_COLUMN, QHeaderView.ResizeMode.Stretch)
        prompt_help = help_texts()["autolabel.prompt"]
        table.horizontalHeaderItem(PROMPT_COLUMN).setToolTip(prompt_help)
        table.setMinimumHeight(150)
        return table

    def _wire(self) -> None:
        self._start_button.clicked.connect(self._start)
        self._cancel_button.clicked.connect(self._runner.cancel)
        self._table.itemChanged.connect(self._on_item_changed)
        self._relabel.toggled.connect(lambda _checked: self.refresh())
        self._threshold.valueChanged.connect(lambda _value: self.refresh())
        self._max_detections.valueChanged.connect(lambda _value: self.refresh())
        self._model.currentIndexChanged.connect(lambda _index: self.refresh())

        self._runner.state_changed.connect(self._on_state)
        self._runner.progress.connect(self._progress.update_progress)
        self._runner.log_line.connect(self._console.append_line)
        self._runner.warned.connect(
            lambda code, msg, hint: self._console.append_line(msg, "warning")
        )
        self._runner.failed.connect(self._on_failed)
        self._runner.finished.connect(self._on_finished)
        self._runner.stalled.connect(
            lambda quiet: self._console.append_line(
                tr("No output for {quiet:.0f}s -- the model may be stuck.", quiet=quiet),
                "warning",
            )
        )

    # -- state ------------------------------------------------------------- #

    def _adopt_project(self) -> None:
        """Take up whichever project Step 2 left current.

        Settings are restored only when the project changes, so that returning
        to this page does not quietly undo a threshold the user just moved.
        """
        self._project = project_model.current()
        self._status = project_model.status(self._project)

        root = str(self._project.root) if self._project else None
        record = autolabel_model.read_record(self._project)
        stored = autolabel_model.prompts_from(record)

        if root != self._adopted:
            self._adopted = root
            self._prompts = {}
            # Whatever went wrong belonged to the project we have just left.
            self._failure = ""
            settings = autolabel_model.settings_from(record)
            for widget, value in (
                (self._threshold, settings["threshold"]),
                (self._max_detections, settings["max_detections"]),
            ):
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)
            index = self._model.findData(settings["model"])
            if index >= 0:
                self._model.blockSignals(True)
                self._model.setCurrentIndex(index)
                self._model.blockSignals(False)

        # What the user typed this session wins over what was written last
        # time, which in turn beats the suggestion made from the class name.
        for stat in self._status.classes:
            if stat.slug in self._prompts:
                continue
            self._prompts[stat.slug] = (
                stored.get(stat.slug) or autolabel_model.suggest_prompt(stat.name)
            )

    def refresh(self) -> None:
        self._status = project_model.status(self._project)
        self._plan = autolabel_model.build_plan(
            self._status,
            self._prompts,
            relabel=self._relabel.isChecked(),
            threshold=self._threshold.value(),
            max_detections=self._max_detections.value(),
            model=str(self._model.currentData() or autolabel_model.DEFAULT_MODEL),
        )
        self._paint_table()
        self._paint_summary()
        self._paint_labels_row()
        self._paint_buttons()
        self.gates_changed.emit()

    def _paint_table(self) -> None:
        # Rebuilding the rows closes whatever cell the user is editing, and
        # refresh() runs on every keystroke-committed prompt and every spinbox
        # click. Repaint only when a row would actually read differently.
        signature = tuple(
            (e.slug, e.name, e.images, e.labels, e.prompt) for e in self._plan.classes
        )
        if signature == getattr(self, "_painted", None):
            return
        self._painted = signature

        prompt_help = help_texts()["autolabel.prompt"]
        self._loading = True
        try:
            self._table.setRowCount(len(self._plan.classes))
            for row, entry in enumerate(self._plan.classes):
                name = QTableWidgetItem(entry.name)
                name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self._table.setItem(row, CLASS_COLUMN, name)

                photos = QTableWidgetItem(f"{entry.images:,}")
                photos.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                photos.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self._table.setItem(row, PHOTOS_COLUMN, photos)

                labelled = QTableWidgetItem(
                    tr("all") if entry.is_labelled else f"{entry.labels:,}"
                )
                labelled.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                labelled.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self._table.setItem(row, LABELLED_COLUMN, labelled)

                prompt = QTableWidgetItem(entry.prompt)
                prompt.setToolTip(prompt_help)
                # The slug travels with the row: sorting or re-reading the
                # table must never re-associate a sentence with another class.
                prompt.setData(Qt.ItemDataRole.UserRole, entry.slug)
                self._table.setItem(row, PROMPT_COLUMN, prompt)
        finally:
            self._loading = False

    def _paint_summary(self) -> None:
        self._summary.setText(self._plan.summary())
        problems = self._plan.problems()
        if problems and self._plan.classes and not self._plan.is_complete:
            self._problems.setText(
                "<span style='color:#b9770e'>! "
                + "<br>! ".join(_escape(p) for p in problems)
                + "</span>"
            )
            self._problems.setVisible(True)
        else:
            self._problems.setVisible(False)

    def _paint_labels_row(self) -> None:
        if self._busy:
            return  # the progress line says something more useful
        if self._failure:
            # A run that failed leaves counts that are perfectly true and
            # completely misleading: "none of the photos are labelled yet" is
            # not what the user needs to read after OWLv2 refused to load.
            self._labels_row.set_state(BAD, _escape(self._failure))
            return
        if self._project is None or not self._status.is_prepared:
            self._labels_row.set_state(UNKNOWN, tr("No prepared dataset yet."))
            return
        total = self._plan.total_images
        labelled = self._plan.total_labels
        if self._plan.is_complete:
            self._labels_row.set_state(
                OK,
                trn(
                    "All {n:,} photos have labels. Step 4 lets you check them, or you "
                    "can go straight to Step 5.",
                    "All {n:,} photos have labels. Step 4 lets you check them, or you "
                    "can go straight to Step 5.",
                    total,
                ),
            )
        elif labelled:
            self._labels_row.set_state(
                WARN,
                trn(
                    "{labelled:,} of {n:,} photos are labelled.",
                    "{labelled:,} of {n:,} photos are labelled.",
                    total,
                    labelled=labelled,
                )
                + " "
                + trn(
                    "{n:,} still to do.",
                    "{n:,} still to do.",
                    total - labelled,
                ),
            )
        else:
            self._labels_row.set_state(
                UNKNOWN,
                trn(
                    "None of the {n:,} photos are labelled yet.",
                    "None of the {n:,} photos are labelled yet.",
                    total,
                ),
            )

    def _paint_buttons(self) -> None:
        busy = self._busy
        self._start_button.setEnabled(not busy and self._plan.is_runnable)
        self._start_button.setText(
            tr("Start autolabeling")
            if not self._plan.todo
            else trn("Label {n:,} photos", "Label {n:,} photos", self._plan.todo)
        )
        self._cancel_button.setEnabled(busy)
        for widget in (
            self._table,
            self._model,
            self._threshold,
            self._max_detections,
            self._relabel,
        ):
            widget.setEnabled(not busy)

    @property
    def _busy(self) -> bool:
        return self._runner.is_active()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or item.column() != PROMPT_COLUMN:
            return
        slug = item.data(Qt.ItemDataRole.UserRole)
        if slug:
            self._prompts[str(slug)] = item.text().strip()
            self.refresh()

    # -- running ----------------------------------------------------------- #

    def _start(self) -> None:
        if self._busy or self._project is None or not self._plan.is_runnable:
            return
        if not self._confirm_relabel():
            return

        self._console.clear()
        self._console.append_line(runtime.describe_python(), "info")
        self._failure = ""
        self._progress.start()
        self._labels_row.set_state(BUSY, tr("Loading the model..."))

        try:
            spec = runtime.job_spec(
                "autolabel",
                {
                    "project": str(self._project.root),
                    "classes": self._plan.job_classes(),
                    "threshold": self._plan.threshold,
                    "max_detections": self._plan.max_detections,
                    "iou": autolabel_model.DEFAULT_IOU,
                    "relabel": self._plan.relabel,
                    "model": self._plan.model,
                    "schema": autolabel_model.AUTOLABEL_SCHEMA,
                },
                project=self._project,
                log_name="autolabel",
            )
        except runtime.EnvironmentMissing as exc:
            self._progress.stop(tr("Not started"))
            QMessageBox.warning(self, tr("Environment not ready"), str(exc))
            self.refresh()
            return

        self._runner.start(spec)
        self._paint_buttons()

    def _confirm_relabel(self) -> bool:
        """Ask before overwriting labels, including corrections made by hand."""
        if not self._plan.relabel or not self._plan.total_labels:
            return True
        answer = QMessageBox.question(
            self,
            tr("Label everything again"),
            trn(
                "{n:,} label file(s) already exist and will be "
                "replaced, including any corrections you made in Step 4.\n\n"
                "Your photos are not touched.\n\nContinue?",
                "{n:,} label file(s) already exist and will be "
                "replaced, including any corrections you made in Step 4.\n\n"
                "Your photos are not touched.\n\nContinue?",
                self._plan.total_labels,
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return int(answer) == int(QMessageBox.StandardButton.Yes)

    # -- slots ------------------------------------------------------------- #

    def _on_state(self, state: JobState) -> None:
        self._paint_buttons()
        if state is JobState.RUNNING:
            self._labels_row.set_state(BUSY, tr("Labelling..."))
        elif state is JobState.CANCELLING:
            self._progress.stop(tr("Stopping..."))
            self._labels_row.set_state(BUSY, tr("Stopping..."))

    def _on_failed(self, code: str, msg: str, hint: str, traceback: str) -> None:
        self._console.append_line(f"[{code}] {msg}", "error")
        if hint:
            self._console.append_line(hint, "warning")
        self._failure = f"{msg} {hint}".strip()
        self._labels_row.set_state(BAD, _escape(self._failure))
        QMessageBox.warning(self, tr("Autolabeling failed"), f"{msg}\n\n{hint}".strip())

    def _on_finished(self, ok: bool, payload: dict) -> None:
        if ok:
            self._failure = ""
            boxes = int(payload.get("boxes") or 0)
            labelled = int(payload.get("labelled") or 0)
            self._progress.stop(
                trn("Labelled {n:,} photos", "Labelled {n:,} photos", labelled)
                + " - "
                + trn("{n:,} boxes", "{n:,} boxes", boxes)
            )
        elif payload.get("cancelled"):
            self._progress.stop(tr("Stopped"))
            self._console.append_line(
                tr(
                    "Stopped. The photos already labelled are kept; starting again carries "
                    "on with the rest."
                ),
                "warning",
            )
        else:
            self._progress.stop(tr("Failed"))
        self.refresh()

    # -- StepPage ---------------------------------------------------------- #

    def can_enter(self) -> bool:
        if runtime.worker_python() is None:
            return False
        # Read from disk rather than from this page's own copy: Step 2 may have
        # prepared a dataset since this page was last looked at, and the wizard
        # asks this question far more often than the page is opened.
        project = project_model.current()
        return project is not None and project.prepared_file.is_file()

    def is_complete(self) -> bool:
        return self._plan.is_complete

    def blocked_reason(self) -> str:
        if runtime.worker_python() is None:
            return tr(
                "Step 1 has to finish setting up the Python environment before "
                "autolabeling can run."
            )
        return tr(
            "Prepare your photos in Step 2 first; autolabeling works on the copy it makes."
        )

    def on_enter(self) -> None:
        if self._busy:
            return
        # The labels may have changed since this page was last open -- Step 4
        # writes corrections, and users delete things between sessions.
        self._adopt_project()
        self.refresh()

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


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
