"""How fast the model ran, in the four numbers that answer different questions.

The specification asks for inference time per frame, in milliseconds, in real
time. That number alone is not quite enough to act on, and the other three cost
nothing to carry beside it:

* **This frame** is what the specification asks for, and it flickers -- a
  single frame's timing moves by a millisecond or two frame to frame, and
  reading a moving number is how people end up quoting the wrong one.
* **Average** is the figure to quote, and the one to compare against a
  datasheet.
* **Slowest 5%** is the one that decides whether a camera feed keeps up. A
  model averaging 20 ms with occasional 90 ms frames drops frames; one that
  averages 25 ms and never exceeds 30 does not, and the average alone says the
  first is the better model.
* **Frames a second** is the end-to-end rate, decoding included, which is
  always lower than 1000/inference and is what the user actually sees. Showing
  only the inference figure invites "8 ms means 125 frames a second", which it
  does not.

Startup is reported separately, and never folded into any of them: see
``worker/commands/infer.py`` for why.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QWidget

from ...core.i18n import N_, tr

#: ``key in the worker's stats``, label, suffix, tooltip.
FIELDS: tuple[tuple[str, str, str, str], ...] = (
    (
        "inference_ms",
        N_("This frame"),
        "ms",
        N_(
            "How long the model itself took on the frame on screen. This is the number "
            "the specification asks for, and the one that moves about most."
        ),
    ),
    (
        "inference_ms_mean",
        N_("Average"),
        "ms",
        N_(
            "The mean over every frame measured so far, and the figure worth quoting. "
            "Getting the model ready is not counted in it."
        ),
    ),
    (
        "inference_ms_p95",
        N_("Slowest 5%"),
        "ms",
        N_(
            "Nineteen frames out of twenty were faster than this. It is what decides "
            "whether a live feed keeps up, which the average alone cannot tell you."
        ),
    ),
    (
        "fps",
        N_("Frames a second"),
        "",
        N_(
            "The end-to-end rate, including reading and decoding each frame. Always "
            "lower than the inference time alone suggests, and what you actually see."
        ),
    ),
)


class SpeedPanel(QGroupBox):
    """Four numbers, updated from each preview frame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(tr("Speed"), parent)
        self._values: dict[str, QLabel] = {}

        grid = QGridLayout(self)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)
        for column, (key, label, suffix, help_text) in enumerate(FIELDS):
            caption = QLabel(tr(label))
            caption.setToolTip(tr(help_text))
            value = QLabel("--")
            value.setToolTip(tr(help_text))
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            font = value.font()
            font.setBold(True)
            font.setPointSizeF(max(11.0, font.pointSizeF() * 1.2))
            value.setFont(font)
            value.setProperty("suffix", suffix)
            grid.addWidget(caption, 0, column)
            grid.addWidget(value, 1, column)
            self._values[key] = value

        self._footnote = QLabel("")
        self._footnote.setWordWrap(True)
        grid.addWidget(self._footnote, 2, 0, 1, len(FIELDS))

    # -- contents ---------------------------------------------------------- #

    def update_stats(self, stats: dict, timings: dict | None = None) -> None:
        """Paint one sample. ``timings`` is this frame's, ``stats`` the run's."""
        merged = dict(stats or {})
        if timings:
            merged.update(timings)
        for key, label in self._values.items():
            value = merged.get(key)
            label.setText(_format(value, str(label.property("suffix") or "")))

    def set_footnote(self, text: str) -> None:
        self._footnote.setText(text)

    def clear(self) -> None:
        for label in self._values.values():
            label.setText("--")
        self._footnote.setText("")


def _format(value, suffix: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not number:
        return "--"
    text = f"{number:.1f}" if number < 100 else f"{number:.0f}"
    return f"{text} {suffix}".strip()
