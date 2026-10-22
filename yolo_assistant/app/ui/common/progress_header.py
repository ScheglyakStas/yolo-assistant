"""Progress with an honest time estimate.

Two things it refuses to do: show a single bar for a run made of phases with
different per-unit costs, and estimate from a global mean. The first epoch of a
real run includes weight downloads, cudnn autotuning and CUDA warmup, so a
global mean says "4 hours remaining" for the first several minutes. The worker
sends an EMA-based estimate; this widget just renders it, and says nothing at
all until one arrives.

The clock, though, is this widget's own. A worker that is inside one long call
-- fetching a 1.3 GB model, building a TensorRT engine, waiting on a cold
dataloader -- reports nothing for minutes at a time, and an elapsed time that
only advances when a message arrives stops at "1s" and makes a working
application look hung. So the elapsed time ticks here, and the estimate counts
down from when it was last reported rather than freezing at the value it had.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from ...core.i18n import N_, tr, trn

PHASE_LABELS = {
    "setup": N_("Preparing"),
    "download": N_("Downloading"),
    "warmup": N_("Warming up"),
    "train": N_("Training"),
    "val": N_("Validating"),
    "final_val": N_("Final validation"),
    "export": N_("Exporting"),
    "infer": N_("Running inference"),
    "preprocess": N_("Preparing images"),
    "label": N_("Labelling"),
    "install": N_("Installing"),
}


class ProgressHeader(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._started: float | None = None
        self._eta: float | None = None
        self._eta_at: float = 0.0

        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._paint_times)

        self._phase = QLabel(tr("Idle"))
        font = self._phase.font()
        font.setBold(True)
        self._phase.setFont(font)

        self._counts = QLabel("")
        self._times = QLabel("")

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._phase)
        top.addStretch(1)
        top.addWidget(self._counts)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addWidget(self._times)
        bottom.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self._bar)
        layout.addLayout(bottom)

    def start(self) -> None:
        self._started = time.monotonic()
        self._eta = None
        self._bar.setValue(0)
        self._phase.setText(tr("Starting..."))
        self._counts.setText("")
        self._paint_times()
        self._clock.start()

    def stop(self, message: str = "") -> None:
        self._clock.stop()
        # The elapsed time is left on screen at the value it reached: how long
        # the run took is the first thing anyone asks afterwards.
        self._started = None
        self._eta = None
        if message:
            self._phase.setText(message)

    def update_progress(self, data: dict) -> None:
        phase = str(data.get("phase", ""))
        known = PHASE_LABELS.get(phase)
        label = tr(known) if known else phase.replace("_", " ").capitalize() or tr("Working")
        # An optional free-text detail, for producers whose phase alone is too
        # coarse -- "Installing" versus "Installing - Downloading torch (3.2 / 9.8 GB)".
        detail = str(data.get("detail", "")).strip()
        self._phase.setText(f"{label} - {detail}" if detail else label)

        current = int(data.get("current") or 0)
        total = data.get("total")
        unit = str(data.get("unit", "items"))

        if total:
            total = int(total)
            self._bar.setRange(0, 1000)
            self._bar.setValue(int(1000 * min(1.0, current / total)))
            self._counts.setText(f"{current} / {_amount(total, unit)}")
        else:
            self._bar.setRange(0, 0)  # indeterminate
            self._counts.setText(_amount(current, unit) if current else "")

        eta = data.get("eta_s")
        if eta is not None:
            self._eta = float(eta)
            self._eta_at = time.monotonic()
        self._paint_times()

    def _paint_times(self) -> None:
        if self._started is None:
            return
        now = time.monotonic()
        parts = [tr("Elapsed {duration}", duration=_duration(now - self._started))]
        # Stay silent rather than guess: an estimate shown too early is worse
        # than none at all.
        if self._eta is not None:
            parts.append(
                tr("remaining {duration}", duration=_duration(self._eta - (now - self._eta_at)))
            )
        self._times.setText("  ·  ".join(parts))


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return tr("{hours}h {minutes:02d}m", hours=hours, minutes=minutes)
    if minutes:
        return tr("{minutes}m {seconds:02d}s", minutes=minutes, seconds=secs)
    return tr("{seconds}s", seconds=secs)


def _amount(n: int, unit: str) -> str:
    """``n`` and its unit, with the noun agreeing with ``n`` in every language.

    ``unit`` is a protocol name rather than display text; one not listed here
    is shown as it was sent.
    """
    if unit == "photos":
        return trn("{n} photos", "{n} photos", n)
    if unit == "steps":
        return trn("{n} steps", "{n} steps", n)
    if unit == "checks":
        return trn("{n} checks", "{n} checks", n)
    if unit == "batches":
        return trn("{n} batches", "{n} batches", n)
    if unit == "devices":
        return trn("{n} devices", "{n} devices", n)
    if unit == "epochs":
        return trn("{n} epochs", "{n} epochs", n)
    if unit == "frames":
        return trn("{n} frames", "{n} frames", n)
    if unit == "files":
        return trn("{n} files", "{n} files", n)
    if unit == "packages":
        return trn("{n} packages", "{n} packages", n)
    if unit == "items":
        return trn("{n} items", "{n} items", n)
    return f"{n} {unit}"
