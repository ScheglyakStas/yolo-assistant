"""Raw worker output, bounded.

Ultralytics, TensorRT and pip are all extremely chatty, and a training run can
emit tens of thousands of lines. A ring buffer keeps memory flat; the full
stream is on disk in the job's log file regardless.
"""

from __future__ import annotations

from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit, QWidget

LEVEL_COLOURS = {
    "stderr": "#c0392b",
    "error": "#c0392b",
    "warning": "#b9770e",
    "info": "#2c3e50",
    "stdout": "#555555",
}


class LogConsole(QPlainTextEdit):
    def __init__(self, parent: QWidget | None = None, *, max_lines: int = 2000) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_lines)  # Qt's own ring buffer
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = QFont("monospace")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        self.setFont(font)

    def append_line(self, text: str, level: str = "info") -> None:
        if not text:
            return
        # Follow the tail only when the user is already at the bottom, so
        # scrolling back to read something is not yanked away.
        scrollbar = self.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4

        colour = LEVEL_COLOURS.get(level, LEVEL_COLOURS["info"])
        escaped = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        self.appendHtml(f'<span style="color:{colour}; white-space:pre">{escaped}</span>')

        if at_bottom:
            self.moveCursor(QTextCursor.MoveOperation.End)
            scrollbar.setValue(scrollbar.maximum())
