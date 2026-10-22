"""Development harness: drives a simulated training run.

No machine learning happens here. The page exists so the parts every later step
depends on -- framing, throttling, the live metrics table, the progress
estimate, the telemetry panel and above all the cancel ladder -- can be
exercised on any machine, in seconds, with no GPU and no managed environment.

It is reachable via ``--fake-worker`` and is not part of the shipped workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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

from ...core.i18n import N_, tr, trn
from ...core.job import JobRunner, JobSpec, JobState
from ..common.log_console import LogConsole
from ..common.param_form import ParamForm
from ..common.progress_header import ProgressHeader
from ..common.telemetry_panel import TelemetryPanel
from .base import StepPage

REPO_ROOT = Path(__file__).resolve().parents[4]

#: Columns of the live metrics table, in display order.
METRIC_COLUMNS = [
    ("metrics/mAP50(B)", N_("mAP50")),
    ("metrics/mAP50-95(B)", N_("mAP50-95")),
    ("metrics/precision(B)", N_("Precision")),
    ("metrics/recall(B)", N_("Recall")),
    ("train/box_loss", N_("Box loss")),
    ("train/cls_loss", N_("Cls loss")),
]


class DemoStep(StepPage):
    title = N_("Simulated run")
    subtitle = N_(
        "Exercises the application's own machinery against a simulated training run. "
        "No model is trained and no GPU is required."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._runner = JobRunner(self)
        self._completed = False

        self._build_ui()
        self._wire_runner()
        self._update_buttons()

    # -- construction ------------------------------------------------------ #

    def _build_ui(self) -> None:
        heading = QLabel(f"<h2>{tr(self.title)}</h2>")
        subtitle = QLabel(tr(self.subtitle))
        subtitle.setWordWrap(True)

        params = QGroupBox(tr("Simulation settings"))
        form = ParamForm(params)

        self._epochs = QSpinBox()
        self._epochs.setRange(1, 500)
        self._epochs.setValue(10)
        form.add_row("demo.epochs", tr("Epochs"), self._epochs)

        self._batches = QSpinBox()
        self._batches.setRange(1, 5000)
        self._batches.setValue(60)
        form.add_row("demo.batches_per_epoch", tr("Batches per epoch"), self._batches)

        self._epoch_seconds = QDoubleSpinBox()
        self._epoch_seconds.setRange(0.1, 120.0)
        self._epoch_seconds.setSingleStep(0.5)
        self._epoch_seconds.setValue(2.0)
        self._epoch_seconds.setSuffix(" s")
        form.add_row("demo.epoch_seconds", tr("Seconds per epoch"), self._epoch_seconds)

        self._ignore_cancel = QCheckBox(tr("Ignore the request to stop"))
        form.add_row("demo.ignore_cancel", tr("Unresponsive worker"), self._ignore_cancel)

        self._children = QSpinBox()
        self._children.setRange(0, 8)
        self._children.setValue(0)
        form.add_row("demo.children", tr("Extra helper processes"), self._children)

        self._start_button = QPushButton(tr("Start"))
        self._start_button.clicked.connect(self._start)
        self._cancel_button = QPushButton(tr("Cancel"))
        self._cancel_button.clicked.connect(self._cancel)

        buttons = QHBoxLayout()
        buttons.addWidget(self._start_button)
        buttons.addWidget(self._cancel_button)
        buttons.addStretch(1)

        self._progress = ProgressHeader()
        self._telemetry = TelemetryPanel()
        self._table = self._build_table()
        self._console = LogConsole()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(params)
        left_layout.addLayout(buttons)
        left_layout.addWidget(self._progress)
        left_layout.addWidget(self._telemetry)
        left_layout.addWidget(self._table, 1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(left)
        splitter.addWidget(self._console)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(subtitle)
        layout.addWidget(splitter, 1)

    def _build_table(self) -> QTableWidget:
        table = QTableWidget(0, 1 + len(METRIC_COLUMNS))
        table.setHorizontalHeaderLabels([tr("Epoch"), *[tr(label) for _, label in METRIC_COLUMNS]])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    def _wire_runner(self) -> None:
        self._runner.state_changed.connect(self._on_state)
        self._runner.progress.connect(self._progress.update_progress)
        self._runner.metrics.connect(self._on_metrics)
        self._runner.log_line.connect(self._console.append_line)
        self._runner.artifact.connect(
            lambda kind, path, meta: self._console.append_line(f"[{kind}] {path}", "info")
        )
        self._runner.warned.connect(
            lambda code, msg, hint: self._console.append_line(f"[{code}] {msg} {hint}", "warning")
        )
        self._runner.failed.connect(self._on_failed)
        self._runner.finished.connect(self._on_finished)
        self._runner.stalled.connect(
            lambda quiet: self._console.append_line(
                tr("No output for {quiet:.0f}s -- the worker may be stuck.", quiet=quiet),
                "warning",
            )
        )

    # -- actions ----------------------------------------------------------- #

    def _start(self) -> None:
        self._table.setRowCount(0)
        self._console.clear()
        self._completed = False
        self._progress.start()

        spec = JobSpec(
            command="fake_train",
            job={
                "epochs": self._epochs.value(),
                "batches_per_epoch": self._batches.value(),
                "epoch_seconds": self._epoch_seconds.value(),
                "ignore_cancel": self._ignore_cancel.isChecked(),
                # An unresponsive worker is only a meaningful test if it also
                # resists the signal that follows.
                "ignore_sigterm": self._ignore_cancel.isChecked(),
                "children": self._children.value(),
            },
            python=sys.executable,
            worker_dir=REPO_ROOT,
            cwd=REPO_ROOT,
            log_path=REPO_ROOT / "logs" / "demo.log",
        )
        self._runner.start(spec)

    def _cancel(self) -> None:
        self._runner.cancel()

    # -- slots ------------------------------------------------------------- #

    def _on_state(self, state: JobState) -> None:
        self._update_buttons()
        if state is JobState.CANCELLING:
            self._progress.stop(tr("Stopping..."))

    def _on_metrics(self, epoch: int, values: dict, is_best: bool) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(str(epoch)))
        for column, (key, _label) in enumerate(METRIC_COLUMNS, start=1):
            value = values.get(key)
            item = QTableWidgetItem("--" if value is None else f"{value:.4f}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, column, item)
        if is_best:
            for column in range(self._table.columnCount()):
                item = self._table.item(row, column)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
        self._table.scrollToBottom()

    def _on_failed(self, code: str, msg: str, hint: str, traceback: str) -> None:
        self._console.append_line(f"[{code}] {msg}", "error")
        if hint:
            self._console.append_line(hint, "warning")

    def _on_finished(self, ok: bool, payload: dict) -> None:
        self._completed = bool(ok)
        if ok:
            epochs = int(payload.get("epochs", 0) or 0)
            self._progress.stop(trn("Finished - {n} epochs", "Finished - {n} epochs", epochs))
        elif payload.get("cancelled"):
            self._progress.stop(tr("Cancelled"))
        else:
            self._progress.stop(tr("Failed"))
            QMessageBox.warning(
                self,
                tr("Simulated run failed"),
                tr("The simulated worker stopped unexpectedly. The log below has the details."),
            )
        self._update_buttons()
        self.gates_changed.emit()

    # -- state ------------------------------------------------------------- #

    def _update_buttons(self) -> None:
        active = self._runner.is_active()
        self._start_button.setEnabled(not active)
        self._cancel_button.setEnabled(active)
        self._start_button.setText(tr("Running...") if active else tr("Start"))

    def is_complete(self) -> bool:
        return self._completed

    def is_busy(self) -> bool:
        return self._runner.is_active()

    def shutdown(self) -> None:
        if self._runner.is_active():
            self._runner.cancel()
        self._telemetry.shutdown()
