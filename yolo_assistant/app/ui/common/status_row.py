"""A requirement, its current state, and the button that fixes it.

Step 1 is the only screen most users will see fail, so a failed check has to
carry its own remedy rather than sending the user to a manual. Each row is
therefore a state, a sentence written for someone who does not know what CUDA
is, and -- where the app can do something about it -- an action.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ...core.i18n import tr

OK = "ok"
BUSY = "busy"
WARN = "warn"
BAD = "bad"
UNKNOWN = "unknown"

_MARKERS = {
    OK: ("✓", "#1e8449"),
    BUSY: ("⋯", "#2471a3"),
    WARN: ("!", "#b9770e"),
    BAD: ("✗", "#c0392b"),
    UNKNOWN: ("–", "#7f8c8d"),
}


class StatusRow(QWidget):
    """One requirement line: marker, title, explanation, optional action."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._marker = QLabel()
        self._marker.setFixedWidth(18)
        self._marker.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        marker_font = self._marker.font()
        marker_font.setBold(True)
        self._marker.setFont(marker_font)

        self._title = QLabel(title)
        title_font = self._title.font()
        title_font.setBold(True)
        self._title.setFont(title_font)

        self._detail = QLabel("")
        self._detail.setWordWrap(True)
        self._detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self._detail.setOpenExternalLinks(True)

        self.action = QPushButton("")
        self.action.setVisible(False)

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(1)
        text.addWidget(self._title)
        text.addWidget(self._detail)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.addWidget(self._marker)
        layout.addLayout(text, 1)
        layout.addWidget(self.action, 0, Qt.AlignmentFlag.AlignTop)

        self.set_state(UNKNOWN, tr("Checking..."))

    def set_state(self, state: str, detail: str) -> None:
        marker, colour = _MARKERS.get(state, _MARKERS[UNKNOWN])
        self._marker.setText(marker)
        self._marker.setStyleSheet(f"color: {colour};")
        self._detail.setText(detail)
        self.state = state

    def set_action(self, text: str, *, visible: bool = True, enabled: bool = True) -> QPushButton:
        self.action.setText(text)
        self.action.setVisible(visible)
        self.action.setEnabled(enabled)
        return self.action
