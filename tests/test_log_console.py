"""The log console stays readable on a dark theme.

Its info and stdout colours are near-black, which is right on a light
background and invisible on a dark one. The theme can also change while the
app is open, so lines already shown must follow it.
"""

from __future__ import annotations

import pytest
from PySide6.QtGui import QColor, QPalette

from yolo_assistant.app.ui.common.log_console import LogConsole


def themed(base: str, text: str) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Base, QColor(base))
    palette.setColor(QPalette.ColorRole.Text, QColor(text))
    return palette


LIGHT = ("#ffffff", "#000000")
DARK = ("#1e1e1e", "#ffffff")


def colour_of_line(console: LogConsole, line: int) -> str:
    block = console.document().findBlockByNumber(line)
    fragment = next(iter(block)).fragment()
    return fragment.charFormat().foreground().color().name()


@pytest.fixture
def console(qapp):
    widget = LogConsole()
    widget.setPalette(themed(*LIGHT))
    return widget


def test_lines_are_dark_on_a_light_theme(console):
    console.append_line("Collecting torch", "stdout")
    console.append_line("Step 1 finished", "info")

    assert colour_of_line(console, 0) == "#555555"
    assert colour_of_line(console, 1) == "#2c3e50"


@pytest.mark.parametrize("level", ["stdout", "info"])
def test_lines_are_white_on_a_dark_theme(console, level):
    console.setPalette(themed(*DARK))
    console.append_line("Collecting torch", level)

    assert colour_of_line(console, 0) == "#ffffff"


def test_switching_theme_recolours_what_is_already_shown(console):
    console.append_line("Collecting torch", "stdout")
    console.append_line("No space left on device", "error")

    console.setPalette(themed(*DARK))
    assert colour_of_line(console, 0) == "#ffffff"
    assert colour_of_line(console, 1) == "#c0392b", "errors stay red"

    console.setPalette(themed(*LIGHT))
    assert colour_of_line(console, 0) == "#555555"
