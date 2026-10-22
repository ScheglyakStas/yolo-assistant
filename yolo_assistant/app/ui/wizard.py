"""The wizard shell: a step rail, a stack of pages, and the gating between them.

Navigation is driven entirely by each page's ``can_enter``/``is_complete``
predicates, re-evaluated whenever any page emits ``gates_changed``. Nothing here
records that a step "was done" -- see ``steps/base.py`` for why.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QSize, Qt
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..core import language
from ..core.i18n import N_, tr
from .steps.base import StepPage

LOCKED_HINT = N_("(locked)")

#: Shown in Help -> About. The AGPL obliges a distributor to tell recipients
#: what licence they have and how to get the source, so this is a compliance
#: requirement rather than decoration.
ABOUT_TEXT = N_("""<h3>YOLO Assistant</h3>
<p>A guided interface for training, exporting and testing YOLO models.</p>
<p>This program is free software, licensed under the
<a href="https://www.gnu.org/licenses/agpl-3.0.html">GNU Affero General Public
License, version 3</a>. It comes with absolutely no warranty. The full licence
text ships with the application in the file <code>LICENSE</code>.</p>
<p>It builds on <a href="https://github.com/ultralytics/ultralytics">Ultralytics</a>,
which is also AGPL-3.0.</p>
<p>You may redistribute it under the terms of that licence, and you are entitled
to the complete corresponding source code.</p>""")


class WizardWindow(QMainWindow):
    def __init__(self, pages: list[StepPage]) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Assistant")
        self.resize(1180, 820)

        self._pages = pages
        self._rail = QListWidget()
        self._rail.setSpacing(2)
        self._rail.currentRowChanged.connect(self._on_rail_changed)

        self._stack = QStackedWidget()
        for index, page in enumerate(pages):
            item = QListWidgetItem(f"{index + 1}.  {tr(page.title)}")
            item.setSizeHint(QSize(0, 34))
            self._rail.addItem(item)
            self._stack.addWidget(page)
            page.gates_changed.connect(self.refresh_gates)

        self._rail.setFixedWidth(self._rail_width())

        self._back = QPushButton(tr("Back"))
        self._back.clicked.connect(lambda: self._goto(self._stack.currentIndex() - 1))
        self._next = QPushButton(tr("Next"))
        self._next.clicked.connect(lambda: self._goto(self._stack.currentIndex() + 1))

        self._blocked = QLabel("")
        self._blocked.setWordWrap(True)

        footer = QHBoxLayout()
        footer.addWidget(self._blocked, 1)
        footer.addWidget(self._back)
        footer.addWidget(self._next)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._stack, 1)
        right_layout.addLayout(footer)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self._rail)
        layout.addWidget(right, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        self._build_menu()
        self._rail.setCurrentRow(0)
        self.refresh_gates()

    def _rail_width(self) -> int:
        """Wide enough for the longest step name, ticked or locked.

        Fixed rather than measured once, because the tick and "(locked)" come
        and go; at least 220 px, which the English always fitted in.
        """
        metrics = self._rail.fontMetrics()
        widest = max(
            metrics.horizontalAdvance(f"{index + 1}.  ✓  {tr(page.title)}  {tr(LOCKED_HINT)}")
            for index, page in enumerate(self._pages)
        )
        return max(220, widest + 32)

    def _build_menu(self) -> None:
        language_menu = self.menuBar().addMenu(tr("&Language"))
        self._language_group = QActionGroup(self)
        self._language_group.setExclusive(True)
        # The saved choice, not the one in effect: after choosing Polski and
        # answering "Later", the menu should already say Polski.
        chosen = language.saved() or language.current()
        for code, name in language.available().items():
            action = language_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(code == chosen)
            action.setData(code)
            self._language_group.addAction(action)
            action.triggered.connect(lambda _checked=False, code=code: self.choose_language(code))

        help_menu = self.menuBar().addMenu(tr("&Help"))
        about = help_menu.addAction(tr("&About"))
        about.triggered.connect(self._show_about)
        licence = help_menu.addAction(tr("&Licence"))
        licence.triggered.connect(self._show_licence)

    def _show_about(self) -> None:
        QMessageBox.about(self, tr("About YOLO Assistant"), tr(ABOUT_TEXT))

    def _show_licence(self) -> None:
        try:
            text = licence_file().read_text(encoding="utf-8")
        except OSError:
            text = tr(
                "The LICENSE file could not be found. See https://www.gnu.org/licenses/agpl-3.0.html"
            )
        box = QMessageBox(self)
        box.setWindowTitle(tr("Licence"))
        box.setText(
            tr("YOLO Assistant is licensed under the GNU Affero General Public License v3.")
        )
        box.setDetailedText(text)
        box.exec()

    # -- language ---------------------------------------------------------- #

    def choose_language(self, code: str) -> None:
        """Remember ``code`` and offer to restart into it.

        The question is asked in the language just chosen: someone who picked
        Polski from a menu they could barely read should not then face a dialog
        in the language they were escaping.
        """
        previous = language.saved() or language.current()
        language.save(code)
        if code == previous or code == language.current():
            # No change, or back to the language already on screen.
            return

        busy = any(page.is_busy() for page in self._pages)
        shown = language.current()
        language.apply(code)
        try:
            title = tr("Language")
            if busy:
                text = tr(
                    "The new language takes effect the next time the app starts. "
                    "Something is still running, so restart the app once it has finished."
                )
            else:
                text = tr(
                    "The new language takes effect the next time the app starts. "
                    "Restart it now?"
                )
            restart_label, later_label = tr("Restart now"), tr("Later")
        finally:
            language.apply(shown)

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle(title)
        box.setText(text)
        if busy:
            box.addButton(QMessageBox.StandardButton.Ok)
            box.exec()
            return
        restart = box.addButton(restart_label, QMessageBox.ButtonRole.AcceptRole)
        later = box.addButton(later_label, QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(later)
        box.exec()
        if box.clickedButton() is restart:
            self.restart()

    def restart(self) -> None:
        """Close as on a normal exit, then start a fresh copy of the app."""
        program, arguments = restart_command(sys.argv)
        self.close()
        QProcess.startDetached(program, arguments)
        QApplication.quit()

    # -- navigation -------------------------------------------------------- #

    def _goto(self, index: int) -> None:
        if not 0 <= index < len(self._pages):
            return
        if not self._pages[index].can_enter():
            self.statusBar().showMessage(self._pages[index].blocked_reason(), 6000)
            return
        self._rail.setCurrentRow(index)

    def _on_rail_changed(self, index: int) -> None:
        if not 0 <= index < len(self._pages):
            return
        page = self._pages[index]
        if not page.can_enter():
            # Bounce back rather than showing a page the user cannot use.
            self.statusBar().showMessage(page.blocked_reason(), 6000)
            self._rail.setCurrentRow(self._stack.currentIndex())
            return
        previous = self._stack.currentWidget()
        if isinstance(previous, StepPage) and previous is not page:
            previous.on_leave()
        self._stack.setCurrentIndex(index)
        page.on_enter()
        self.refresh_gates()

    def refresh_gates(self) -> None:
        current = self._stack.currentIndex()
        for index, page in enumerate(self._pages):
            item = self._rail.item(index)
            unlocked = page.can_enter()
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsEnabled
                if unlocked
                else item.flags() & ~Qt.ItemFlag.ItemIsEnabled
            )
            marker = "✓  " if page.is_complete() else ""
            suffix = "" if unlocked else f"  {tr(LOCKED_HINT)}"
            item.setText(f"{index + 1}.  {marker}{tr(page.title)}{suffix}")

        self._back.setEnabled(current > 0 and self._pages[current - 1].can_enter())
        can_advance = current + 1 < len(self._pages) and self._pages[current + 1].can_enter()
        self._next.setEnabled(can_advance)
        if current + 1 < len(self._pages) and not can_advance:
            self._blocked.setText(self._pages[current + 1].blocked_reason())
        else:
            self._blocked.setText("")

    # -- lifecycle --------------------------------------------------------- #

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        for page in self._pages:
            page.shutdown()
        super().closeEvent(event)


def licence_file() -> Path:
    """The AGPL text: at the repository root, or the bundle's in a frozen build."""
    return Path(__file__).resolve().parents[3] / "LICENSE"


def restart_command(argv: list[str]) -> tuple[str, list[str]]:
    """The program and arguments that start this app again.

    A one-off ``--language`` is dropped: it would override the choice the
    restart is for.
    """
    arguments: list[str] = []
    skip = False
    for argument in argv[1:]:
        if skip:
            skip = False
        elif argument == "--language":
            skip = True
        elif not argument.startswith("--language="):
            arguments.append(argument)
    if getattr(sys, "frozen", False):
        return sys.executable, arguments
    # Not argv[0]: run as "-m", that is main.py's path, whose relative imports
    # fail when it is run as a script.
    return sys.executable, ["-m", "yolo_assistant.app.main", *arguments]
