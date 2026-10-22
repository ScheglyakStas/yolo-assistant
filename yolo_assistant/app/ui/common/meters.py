"""Small labelled bar used for VRAM, power and any other bounded quantity."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from ...core.i18n import tr


class Meter(QWidget):
    """A caption, a bar and a value, on one line.

    Reports "not supported" rather than zero when a field is unavailable --
    power and temperature genuinely are on some laptop and vGPU parts, and a
    zero there would read as a broken GPU rather than a missing sensor.
    """

    def __init__(self, caption: str, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._caption = QLabel(caption)
        self._caption.setMinimumWidth(78)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(12)

        self._value = QLabel("--")
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._value.setMinimumWidth(112)
        font = self._value.font()
        font.setStyleHint(font.StyleHint.Monospace)
        font.setFamily("monospace")
        self._value.setFont(font)

        layout.addWidget(self._caption)
        layout.addWidget(self._bar, 1)
        layout.addWidget(self._value)

    def set_value(self, fraction: float | None, text: str) -> None:
        if fraction is None:
            self._bar.setEnabled(False)
            self._bar.setValue(0)
        else:
            self._bar.setEnabled(True)
            self._bar.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        self._value.setText(text)

    def set_unsupported(self) -> None:
        self.set_value(None, tr("not supported"))

    def clear(self) -> None:
        self.set_value(None, "--")
