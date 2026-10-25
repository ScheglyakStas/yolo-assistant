"""Raw worker output, bounded.

Ultralytics, TensorRT and pip are all extremely chatty, and a training run can
emit tens of thousands of lines. A ring buffer keeps memory flat; the full
stream is on disk in the job's log file regardless.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtGui import QColor, QFont, QPalette, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit, QWidget

LEVEL_COLOURS = {
    "stderr": "#c0392b",
    "error": "#c0392b",
    "warning": "#b9770e",
    "info": "#2c3e50",
    "stdout": "#555555",
}

#: The same on a dark theme, where the two near-black ones would vanish.
DARK_LEVEL_COLOURS = {**LEVEL_COLOURS, "info": "#ffffff", "stdout": "#ffffff"}

#: Each line's level is kept on its block, so a theme switch can recolour it.
_LEVELS = list(LEVEL_COLOURS)


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
        self._dark = self._is_dark()

    def _is_dark(self) -> bool:
        return self.palette().color(QPalette.ColorRole.Base).lightness() < 128

    def _colour(self, level: str) -> str:
        colours = DARK_LEVEL_COLOURS if self._dark else LEVEL_COLOURS
        return colours.get(level, colours["info"])

    def append_line(self, text: str, level: str = "info") -> None:
        if not text:
            return
        # Follow the tail only when the user is already at the bottom, so
        # scrolling back to read something is not yanked away.
        scrollbar = self.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4

        colour = self._colour(level)
        escaped = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        self.appendHtml(f'<span style="color:{colour}; white-space:pre">{escaped}</span>')
        self.document().lastBlock().setUserState(
            _LEVELS.index(level) if level in LEVEL_COLOURS else _LEVELS.index("info")
        )

        if at_bottom:
            self.moveCursor(QTextCursor.MoveOperation.End)
            scrollbar.setValue(scrollbar.maximum())

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and self._is_dark() != self._dark:
            self._dark = not self._dark
            self._recolour()

    def _recolour(self) -> None:
        """Repaint the lines already shown in the current theme's colours."""
        block = self.document().firstBlock()
        while block.isValid():
            state = block.userState()
            if 0 <= state < len(_LEVELS):
                fmt = QTextCharFormat()
                fmt.setForeground(QColor(self._colour(_LEVELS[state])))
                cursor = QTextCursor(block)
                cursor.movePosition(
                    QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor
                )
                cursor.mergeCharFormat(fmt)
            block = block.next()
