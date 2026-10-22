"""Give the Windows build the look the app has on Ubuntu.

Qt draws with each platform's own style and font unless told otherwise, so the
same window is Fusion and Ubuntu Sans on Ubuntu 24.04 but the native Windows 11
style and Segoe UI on Windows, and its ``QFont("monospace")`` log console falls
back to Courier New there. On Windows this switches to Fusion and the Ubuntu
Sans fonts bundled in ``resources/fonts``; elsewhere the desktop's settings stand.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


FONT_DIR = Path(__file__).resolve().parents[1] / "resources" / "fonts"
STYLE = "Fusion"
UI_FAMILY = "Ubuntu Sans"
MONO_FAMILY = "Ubuntu Sans Mono"
# GNOME's default interface font is "Ubuntu Sans 11".
UI_POINT_SIZE = 11


def apply(app: QApplication) -> None:
    """Restyle ``app`` on Windows; call before any widget is built."""
    if sys.platform != "win32":
        return
    app.setStyle(STYLE)
    families = load_fonts()
    if UI_FAMILY in families:
        app.setFont(QFont(UI_FAMILY, UI_POINT_SIZE))
    if MONO_FAMILY in families:
        # The log console and meters ask for the generic "monospace", which
        # Windows has no font for.
        QFont.insertSubstitution("monospace", MONO_FAMILY)


def load_fonts() -> set[str]:
    """Register every bundled font with Qt and return the families that loaded.

    A file that fails to load is skipped, so the app falls back to the system
    font rather than failing to start; ``--self-check`` reports it.
    """
    families: set[str] = set()
    for path in sorted(FONT_DIR.glob("*.ttf")):
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id != -1:
            families.update(QFontDatabase.applicationFontFamilies(font_id))
    return families
