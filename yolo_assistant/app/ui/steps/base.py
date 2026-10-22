"""Base class for wizard pages.

Gating is expressed as two predicates rather than stored flags. The reasoning is
in the plan: users delete folders, move weights and clear ``runs/`` between
sessions, so a stored ``step3_complete = true`` becomes a lie while a recomputed
``labelled == total`` cannot. Pages therefore answer from whatever they can
observe right now.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from ...core.i18n import N_, tr


class StepPage(QWidget):
    """One step of the guided workflow."""

    #: Emitted whenever something happened that may change which steps are
    #: reachable. The wizard re-evaluates every page's predicates on this.
    gates_changed = Signal()

    #: Short name shown in the step rail. Marked with ``N_`` and translated
    #: where it is shown, because a class body runs before the language is set.
    title: str = N_("Step")
    #: One line under the heading explaining what this step is for. Marked the
    #: same way.
    subtitle: str = ""

    def can_enter(self) -> bool:
        """Whether the user is allowed to open this page at all."""
        return True

    def is_complete(self) -> bool:
        """Whether this step's work has been done."""
        return False

    def blocked_reason(self) -> str:
        """Why :meth:`can_enter` is false, phrased for a beginner."""
        return tr("An earlier step has to be finished first.")

    def is_busy(self) -> bool:
        """Whether a job is running that quitting the app would stop."""
        return False

    def on_enter(self) -> None:
        """Called when the page becomes visible."""

    def on_leave(self) -> None:
        """Called when the user navigates away."""

    def on_app_start(self) -> None:
        """Called once per launch, after the main window is first shown."""

    def shutdown(self) -> None:
        """Called once when the application is closing."""
