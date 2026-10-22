"""Shared test setup.

The one non-obvious thing here is the Qt application fixture. Qt allows exactly
one application object per process, and a ``QCoreApplication`` is not enough to
construct a widget -- doing so aborts the interpreter outright rather than
raising. Since some test modules need only the event loop and others need real
widgets, whichever ran first used to decide whether the rest of the suite
survived. One session-wide ``QApplication``, shared under both names the test
modules ask for, removes that ordering dependency.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def qt_app():
    from PySide6.QtWidgets import QApplication

    instance = QApplication.instance()
    if instance is None:
        instance = QApplication(sys.argv[:1])
    return instance


@pytest.fixture(scope="session")
def app(qt_app):
    return qt_app


@pytest.fixture(scope="session")
def qapp(qt_app):
    """Overrides pytest-qt's fixture of the same name, for the reason above."""
    return qt_app
