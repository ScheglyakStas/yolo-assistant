"""Step 2: turn a folder of photos into a dataset.

The user has a folder of photos and no idea what YOLO wants. This page's job is
to stand between those two facts, and it does three things to earn its place.

**It shows what it found before it does anything.** Choosing a folder scans it
and reports what is in there -- the classes, the counts, the names that have to
be spelled differently on disk, the files that are not photos. Preparation is a
copy of potentially thousands of files; the moment to discover that a folder has
one class in it instead of four is before that, not after.

**It never touches the originals.** Everything is copied into a project folder,
which the page shows and lets the user move. That is also why the source folder
and the project folder are both on screen at once: "where did my photos go" has
a visible answer.

**It says what would be lost.** Preparing again over an existing dataset throws
away every label made in Steps 3 and 4, which can be an afternoon's work. The
confirmation names the number.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core import dataset as dataset_scan
from ...core import project as project_model
from ...core import runtime
from ...core.i18n import N_, tr, trn
from ...core.job import JobRunner, JobState
from ...core.project import Project
from ..common.log_console import LogConsole
from ..common.param_form import ParamForm, apply_help
from ..common.progress_header import ProgressHeader
from ..common.status_row import BAD, BUSY, OK, UNKNOWN, WARN, StatusRow
from .base import StepPage

#: Share of each class held back to measure accuracy, as a percentage.
DEFAULT_VAL_PERCENT = 20


class _ScanThread(QThread):
    """Walks the photo folder off the GUI thread.

    A folder of 40,000 photos on a network share takes seconds to list, and a
    window that stops repainting while the user waits looks broken.
    """

    scanned = Signal(object)  # dataset_scan.SourceScan

    def __init__(self, root: Path, parent=None) -> None:
        super().__init__(parent)
        self._root = root

    def run(self) -> None:  # noqa: D102 - QThread entry point
        self.scanned.emit(dataset_scan.scan(self._root))


class DatasetStep(StepPage):
    title = N_("Dataset preparation")
    subtitle = N_(
        "Point the app at a folder with one sub-folder per class. It copies the photos "
        "into a project folder, renames them, and lays them out the way YOLO expects. "
        "Your originals are not modified."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._runner = JobRunner(self)
        self._scan: dataset_scan.SourceScan | None = None
        self._scan_thread: _ScanThread | None = None
        self._scanned_root: Path | None = None
        self._project: Project | None = None
        self._status = project_model.status(None)

        self._build_ui()
        self._wire()
        self._adopt_current_project()
        self.refresh()

    # -- construction ------------------------------------------------------ #

    def _build_ui(self) -> None:
        heading = QLabel(f"<h2>{tr(self.title)}</h2>")
        subtitle = QLabel(tr(self.subtitle))
        subtitle.setWordWrap(True)

        source_box = QGroupBox(tr("Photos"))
        # No parent: this form is one part of the group box's layout rather than
        # the whole of it, and a QFormLayout installed on the box would be.
        source_form = ParamForm()
        self._source_edit = QLineEdit()
        self._source_edit.setPlaceholderText(tr("Choose the folder holding your class folders"))
        self._source_browse = QPushButton(tr("Browse..."))
        source_form.add_row(
            "dataset.source_folder",
            tr("Photo folder"),
            _with_button(self._source_edit, self._source_browse),
        )

        self._summary = QLabel(tr("No folder chosen yet."))
        self._summary.setWordWrap(True)
        self._problems = QLabel("")
        self._problems.setWordWrap(True)
        self._problems.setVisible(False)
        self._table = self._build_table()

        source_layout = QVBoxLayout()
        source_layout.addLayout(source_form)
        source_layout.addWidget(self._summary)
        source_layout.addWidget(self._problems)
        source_layout.addWidget(self._table, 1)
        source_box.setLayout(source_layout)

        project_box = QGroupBox(tr("Project"))
        project_form = ParamForm(project_box)
        self._project_edit = QLineEdit()
        self._project_edit.setPlaceholderText(tr("Where the prepared dataset will be written"))
        self._project_browse = QPushButton(tr("Browse..."))
        project_form.add_row(
            "dataset.project_folder",
            tr("Project folder"),
            _with_button(self._project_edit, self._project_browse),
        )

        self._val_split = QSpinBox()
        self._val_split.setRange(0, 50)
        self._val_split.setValue(DEFAULT_VAL_PERCENT)
        self._val_split.setSuffix(" %")
        project_form.add_row("dataset.val_split", tr("Held back for checking"), self._val_split)

        self._dataset_row = StatusRow(tr("Prepared dataset"))
        self._dataset_row.action.setVisible(False)

        self._prepare_button = QPushButton(tr("Prepare dataset"))
        apply_help(self._prepare_button, "dataset.prepare")
        self._cancel_button = QPushButton(tr("Cancel"))
        self._cancel_button.setEnabled(False)
        self._open_button = QPushButton(tr("Open project folder"))
        apply_help(self._open_button, "dataset.open_folder")

        buttons = QHBoxLayout()
        buttons.addWidget(self._prepare_button)
        buttons.addWidget(self._cancel_button)
        buttons.addStretch(1)
        buttons.addWidget(self._open_button)

        self._progress = ProgressHeader()
        self._console = LogConsole()

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(source_box, 1)
        top_layout.addWidget(project_box)
        top_layout.addWidget(self._dataset_row)
        top_layout.addLayout(buttons)
        top_layout.addWidget(self._progress)

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
        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels([tr("Class"), tr("Photos"), tr("Stored as")])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        table.setMinimumHeight(120)
        return table

    def _wire(self) -> None:
        self._source_browse.clicked.connect(self._choose_source)
        self._project_browse.clicked.connect(self._choose_project)
        self._source_edit.editingFinished.connect(self._on_source_typed)
        self._project_edit.editingFinished.connect(self._on_project_typed)
        self._prepare_button.clicked.connect(self._prepare)
        self._cancel_button.clicked.connect(self._runner.cancel)
        self._open_button.clicked.connect(self._open_project_folder)

        self._runner.state_changed.connect(self._on_state)
        self._runner.progress.connect(self._progress.update_progress)
        self._runner.log_line.connect(self._console.append_line)
        self._runner.warned.connect(
            lambda code, msg, hint: self._console.append_line(msg, "warning")
        )
        self._runner.failed.connect(self._on_failed)
        self._runner.finished.connect(self._on_finished)

    # -- state ------------------------------------------------------------- #

    def _adopt_current_project(self) -> None:
        """Pick up where the last session left off, if that folder is still there."""
        existing = project_model.current()
        if existing is None:
            return
        self._project = existing
        self._project_edit.setText(str(existing.root))
        record = project_model.read_prepared(existing)
        source = (record or {}).get("source")
        if source:
            self._source_edit.setText(str(source))
            if Path(source).is_dir():
                # Scan it again rather than trusting the record: the point of
                # returning to this page is usually that the photos changed.
                self.start_scan(Path(source))
        if record and record.get("val_split") is not None:
            self._val_split.setValue(int(round(float(record["val_split"]) * 100)))

    def refresh(self) -> None:
        self._status = project_model.status(self._project)
        self._paint_dataset_row()
        self._paint_buttons()
        self.gates_changed.emit()

    def _paint_dataset_row(self) -> None:
        if self._busy:
            return  # the stage message is more useful than a stale count
        status = self._status
        if self._project is None:
            self._dataset_row.set_state(
                UNKNOWN, tr("Choose where the prepared dataset should go.")
            )
            return

        state = {
            project_model.PREPARED: OK,
            project_model.PARTIAL: WARN,
            project_model.STALE: WARN,
            project_model.ABSENT: UNKNOWN,
        }.get(status.state, UNKNOWN)
        # The row renders rich text, so anything from disk -- a path, a warning
        # the worker wrote -- is escaped before it gets there.
        detail = _escape(status.message)
        if status.is_prepared:
            detail = tr(
                "{status} In {folder}.",
                status=detail,
                folder=_escape(str(self._project.dataset_dir)),
            )
            for warning in status.warnings:
                detail += f"<br><span style='color:#b9770e'>{_escape(warning)}</span>"
                state = WARN
        self._dataset_row.set_state(state, detail)

    def _paint_buttons(self) -> None:
        busy = self._busy
        ready = bool(self._scan and self._scan.is_usable) and bool(self._project_edit.text().strip())
        self._prepare_button.setEnabled(not busy and ready)
        self._prepare_button.setText(
            tr("Prepare again") if self._status.is_prepared else tr("Prepare dataset")
        )
        self._cancel_button.setEnabled(busy)
        self._open_button.setEnabled(
            self._project is not None and self._project.dataset_dir.exists()
        )
        for widget in (
            self._source_edit,
            self._source_browse,
            self._project_edit,
            self._project_browse,
            self._val_split,
        ):
            widget.setEnabled(not busy)

    @property
    def _busy(self) -> bool:
        return self._runner.is_active()

    # -- choosing folders -------------------------------------------------- #

    def _choose_source(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, tr("Choose the folder holding your class folders"), self._source_edit.text()
        )
        if chosen:
            self._source_edit.setText(chosen)
            # Choosing the same folder again is how a user asks for a re-scan
            # after adding photos to it, so it must not be treated as a no-op.
            self._scanned_root = None
            self._on_source_typed()

    def _choose_project(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, tr("Choose where to put the project"), self._project_edit.text()
        )
        if chosen:
            self._project_edit.setText(chosen)
            self._on_project_typed()

    def _on_source_typed(self) -> None:
        text = self._source_edit.text().strip()
        if not text:
            return
        source = Path(text).expanduser()
        # Suggest a project folder, but never overwrite one the user typed: the
        # destination drive is often a deliberate choice about free space.
        if not self._project_edit.text().strip():
            self._project_edit.setText(str(project_model.default_root_for(source)))
            self._on_project_typed()
        # editingFinished also fires when the field merely loses focus, and
        # walking forty thousand photos on a network share to learn nothing new
        # is not free.
        if source != self._scanned_root:
            self.start_scan(source)

    def _on_project_typed(self) -> None:
        text = self._project_edit.text().strip()
        self._project = Project(Path(text)) if text else None
        self.refresh()

    def start_scan(self, source: Path) -> None:
        if self._scan_thread is not None and self._scan_thread.isRunning():
            return
        self._summary.setText(tr("Looking through {folder}...", folder=source))
        self._problems.setVisible(False)
        self._scanned_root = source
        self._scan_thread = _ScanThread(source, self)
        self._scan_thread.scanned.connect(self._on_scanned)
        self._scan_thread.start()

    def _on_scanned(self, scan: dataset_scan.SourceScan) -> None:
        self._scan = scan
        self._summary.setText(scan.summary())
        self._paint_problems(scan)
        self._paint_table(scan)
        self.refresh()

    def _paint_problems(self, scan: dataset_scan.SourceScan) -> None:
        if not scan.problems:
            self._problems.setVisible(False)
            return
        lines = [
            f"<span style='color:{'#c0392b' if problem.is_error else '#b9770e'}'>"
            f"{'✗' if problem.is_error else '!'} {_escape(problem.text)}</span>"
            for problem in scan.problems
        ]
        self._problems.setText("<br>".join(lines))
        self._problems.setVisible(True)

    def _paint_table(self, scan: dataset_scan.SourceScan) -> None:
        self._table.setRowCount(0)
        for cls in scan.classes:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(cls.name))
            count = QTableWidgetItem(tr("{count:,}", count=cls.count))
            count.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, 1, count)
            self._table.setItem(
                row, 2, QTableWidgetItem(f"{cls.slug}/{cls.slug}_000001.jpg")
            )

    # -- preparing --------------------------------------------------------- #

    def _prepare(self) -> None:
        if self._busy or self._scan is None or self._project is None:
            return
        if not self._scan.is_usable:
            return
        if not self._confirm_replacement():
            return

        self._console.clear()
        self._console.append_line(runtime.describe_python(), "info")
        self._progress.start()
        self._dataset_row.set_state(BUSY, tr("Copying and checking photos..."))

        try:
            spec = runtime.job_spec(
                "prepare_dataset",
                {
                    "source": str(self._scan.root),
                    "dest": str(self._project.root),
                    "classes": self._scan.job_classes(),
                    "val_split": self._val_split.value() / 100.0,
                    "schema": project_model.PROJECT_SCHEMA,
                },
                project=self._project,
                log_name="prepare_dataset",
            )
        except runtime.EnvironmentMissing as exc:
            self._progress.stop(tr("Not started"))
            QMessageBox.warning(self, tr("Environment not ready"), str(exc))
            self.refresh()
            return

        self._runner.start(spec)
        self._paint_buttons()

    def _confirm_replacement(self) -> bool:
        """Ask before throwing away labels. Preparing again destroys them."""
        if not self._status.is_prepared:
            return True
        labels = self._status.total_labels
        paragraphs = [
            trn(
                "{folder} already holds a prepared dataset with {n:,} photos. "
                "Preparing again replaces it completely.",
                "{folder} already holds a prepared dataset with {n:,} photos. "
                "Preparing again replaces it completely.",
                self._status.total_images,
                folder=self._project.dataset_dir,
            )
        ]
        if labels:
            paragraphs.append(
                trn(
                    "{n:,} label file(s) from Steps 3 and 4 will be deleted with it.",
                    "{n:,} label file(s) from Steps 3 and 4 will be deleted with it.",
                    labels,
                )
            )
        paragraphs += [tr("Your original photos are not touched."), tr("Continue?")]
        answer = QMessageBox.question(
            self,
            tr("Prepare again"),
            "\n\n".join(paragraphs),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return int(answer) == int(QMessageBox.StandardButton.Yes)

    def _open_project_folder(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        if self._project is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._project.root)))

    # -- slots ------------------------------------------------------------- #

    def _on_state(self, state: JobState) -> None:
        self._paint_buttons()
        if state is JobState.CANCELLING:
            self._progress.stop(tr("Stopping..."))
            self._dataset_row.set_state(BUSY, tr("Stopping..."))

    def _on_failed(self, code: str, msg: str, hint: str, traceback: str) -> None:
        self._console.append_line(f"[{code}] {msg}", "error")
        if hint:
            self._console.append_line(hint, "warning")
        self._dataset_row.set_state(BAD, msg)
        QMessageBox.warning(self, tr("Preparation failed"), f"{msg}\n\n{hint}".strip())

    def _on_finished(self, ok: bool, payload: dict) -> None:
        if ok:
            self._progress.stop(tr("Dataset ready"))
            if self._project is not None:
                # Only now: a project that failed to prepare is not one later
                # steps should open on the next launch.
                project_model.set_current(self._project)
        elif payload.get("cancelled"):
            self._progress.stop(tr("Cancelled"))
            self._console.append_line(
                tr(
                    "Preparation was cancelled. The partly-copied dataset is incomplete; "
                    "preparing again starts from a clean folder."
                ),
                "warning",
            )
        else:
            self._progress.stop(tr("Failed"))
        self.refresh()

    # -- StepPage ---------------------------------------------------------- #

    def can_enter(self) -> bool:
        return runtime.worker_python() is not None

    def is_complete(self) -> bool:
        return self._status.is_prepared and self._status.total_images > 0

    def blocked_reason(self) -> str:
        return tr(
            "Step 1 has to finish setting up the Python environment before photos "
            "can be prepared."
        )

    def on_enter(self) -> None:
        if not self._busy:
            # The user may have deleted the folder, or labelled some of it in
            # another session, since this page was last looked at.
            self.refresh()

    def is_busy(self) -> bool:
        return self._busy

    def shutdown(self) -> None:
        if self._runner.is_active():
            self._runner.cancel()
        if self._scan_thread is not None and self._scan_thread.isRunning():
            self._scan_thread.wait(5000)

    # -- for later steps --------------------------------------------------- #

    @property
    def project(self) -> Project | None:
        return self._project


def _with_button(edit: QLineEdit, button: QPushButton) -> QWidget:
    """A line edit and its Browse button as one form field.

    The tooltip is applied to the container by ``ParamForm``, and Qt propagates
    it to both children -- so hovering anywhere on the row explains the row.
    """
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit, 1)
    layout.addWidget(button)
    return container


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
