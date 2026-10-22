"""What the last epoch achieved, next to the best epoch so far.

A single column of numbers cannot answer the question anyone actually has in
front of a training run, which is "is this still getting better?". Epoch 40
reporting mAP50-95 of 0.61 means one thing when the best so far is 0.61 and
quite another when the best was 0.68 at epoch 22 -- the second is a run that
peaked twenty epochs ago and is now overfitting, and it is worth stopping. So
both columns are here, and the epoch each came from is named.

Ultralytics' own metric names are kept as the key and shortened for display.
They are what appears in ``results.csv``, in every tutorial and in every answer
the user will find when they search for one, so the app should not invent a
private vocabulary for them -- but ``metrics/mAP50-95(B)`` is also not a thing
to put in front of a beginner as it stands. The accuracy figures keep their
English names in every language, because those names are the search terms; the
losses are descriptions rather than terms, and are translated.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import N_, tr, trn

NAME_COLUMN, LAST_COLUMN, BEST_COLUMN = range(3)

#: Display name and one-line explanation per metric, in the order they matter.
#: A name marked with ``N_`` is translated; one that is not is a term of art and
#: is shown in English whatever the language -- see the module docstring.
#: The explanation is the tooltip: these are numbers a beginner is meant to act
#: on, and "mAP50-95" tells them nothing on its own.
#:
#: This is an allow-list, and that is the point of it. The trainer reports a
#: dozen or more numbers per epoch and which dozen depends on the model: an
#: end-to-end head drops the DFL loss and adds L1 and L2 terms of its own, and
#: a future one will report something else again. Rendering whatever turns up
#: gives a table that changes shape with the checkpoint and fills with terms
#: nobody chose to explain, so only what is listed here is shown. Everything
#: else is in ``results.csv`` in the run folder, under Ultralytics' own names.
KNOWN_METRICS: tuple[tuple[str, str, str], ...] = (
    (
        "metrics/mAP50-95(B)",
        "mAP50-95",
        N_(
            "The headline accuracy figure, averaged over ten strictnesses of overlap. "
            "This is the number to watch: if it has stopped rising, more epochs will "
            "not help. It is also what Ultralytics ranks epochs by, so the epoch with "
            "the highest is the one saved as best.pt."
        ),
    ),
    (
        "metrics/mAP50(B)",
        "mAP50",
        N_(
            "Accuracy counting a box as correct when it overlaps the true one by half. "
            "Always higher than mAP50-95, and more forgiving about exactly where the "
            "box sits."
        ),
    ),
    (
        "metrics/precision(B)",
        "Precision",
        N_(
            "Of the objects the model reported, the share that were really there. Low "
            "precision means it is inventing detections."
        ),
    ),
    (
        "metrics/recall(B)",
        "Recall",
        N_(
            "Of the objects that were there, the share the model found. Low recall "
            "means it is missing them."
        ),
    ),
    (
        "train/box_loss",
        N_("Box loss"),
        N_(
            "How far the predicted boxes are from the true ones, on the training "
            "photos. It should fall and then flatten."
        ),
    ),
    (
        "train/cls_loss",
        N_("Class loss"),
        N_(
            "How wrong the model is about which class each object belongs to, on the "
            "training photos."
        ),
    ),
)

_ORDER = {key: index for index, (key, _label, _help) in enumerate(KNOWN_METRICS)}
_LABELS = {key: label for key, label, _help in KNOWN_METRICS}
_HELP = {key: help_text for key, _label, help_text in KNOWN_METRICS}
#: The names shown exactly as written above, never translated.
_TERMS = frozenset({"mAP50-95", "mAP50", "Precision", "Recall"})


class MetricsPanel(QGroupBox):
    """The last epoch's numbers, the best epoch's, and which epoch each was."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(tr("Results"), parent)

        self._heading = QLabel("")
        self._heading.setWordWrap(True)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels([tr("Metric"), tr("Last epoch"), tr("Best epoch")])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setAlternatingRowColors(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(NAME_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(LAST_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(BEST_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setMinimumHeight(140)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.addWidget(self._heading)
        layout.addWidget(self._table, 1)

        self.clear()

    # -- state ------------------------------------------------------------- #

    def clear(self) -> None:
        self._last: dict[str, float] = {}
        self._best: dict[str, float] = {}
        self._last_epoch = 0
        self._best_epoch = 0
        self._total_epochs = 0
        self._table.setRowCount(0)
        self._heading.setText(tr("No epoch has finished yet."))

    def set_total_epochs(self, epochs: int) -> None:
        self._total_epochs = max(0, int(epochs))
        self._paint_heading()

    def add_epoch(self, epoch: int, values: dict, best: bool = False) -> None:
        """Record one finished epoch. ``best`` is Ultralytics' own verdict."""
        numbers = {}
        for key, value in (values or {}).items():
            try:
                numbers[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if not numbers:
            return
        self._last = numbers
        self._last_epoch = int(epoch)
        if best or not self._best:
            self._best = dict(numbers)
            self._best_epoch = int(epoch)
        self._paint()

    # -- painting ---------------------------------------------------------- #

    def _paint(self) -> None:
        self._paint_heading()
        # Only what this epoch actually reported, so a row never sits at "-"
        # for a whole run because the model does not produce that number.
        keys = sorted(
            (key for key in set(self._last) | set(self._best) if key in _ORDER),
            key=lambda key: _ORDER[key],
        )
        self._table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            label = _LABELS[key]
            name = QTableWidgetItem(label if label in _TERMS else tr(label))
            name.setToolTip(tr(_HELP[key]))
            self._table.setItem(row, NAME_COLUMN, name)
            for column, source in ((LAST_COLUMN, self._last), (BEST_COLUMN, self._best)):
                item = QTableWidgetItem(_format(source.get(key)))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self._table.setItem(row, column, item)

    def _paint_heading(self) -> None:
        if not self._last_epoch:
            self._heading.setText(tr("No epoch has finished yet."))
            return
        if self._total_epochs:
            sentences = [
                tr(
                    "Epoch {epoch} of {total} finished.",
                    epoch=self._last_epoch,
                    total=self._total_epochs,
                )
            ]
        else:
            sentences = [tr("Epoch {epoch} finished.", epoch=self._last_epoch)]
        if self._best_epoch and self._best_epoch != self._last_epoch:
            sentences.append(
                trn(
                    "The best so far was epoch {best}, {n} epoch ago.",
                    "The best so far was epoch {best}, {n} epochs ago.",
                    self._last_epoch - self._best_epoch,
                    best=self._best_epoch,
                )
            )
        elif self._best_epoch:
            sentences.append(tr("It is the best so far."))
        self._heading.setText(" ".join(sentences))


def _format(value) -> str:
    """Four decimals for everything on the table.

    The accuracy figures are shares of one and the losses are small numbers a
    little above it, so one width suits both -- and four places is where two
    consecutive epochs stop looking identical.
    """
    if value is None:
        return "-"
    return f"{float(value):.4f}"
