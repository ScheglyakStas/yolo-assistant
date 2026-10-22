"""Application entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from .. import __version__
from .core import i18n, language, redirection_guard
from .core.i18n import tr
from .core.paths import bundle_dir, ensure_dirs
from .ui import platform_look
from .ui.steps.base import StepPage
from .ui.wizard import WizardWindow


ICON_FILE = Path(__file__).resolve().parent / "resources" / "icon.svg"


def build_pages(*, fake_worker: bool) -> list[StepPage]:
    """Assemble the wizard's pages.

    ``--fake-worker`` appends the simulated-run page, which needs no GPU and no
    managed environment and so keeps the shell testable on any machine.
    """
    from .ui.steps.autolabel_step import AutolabelStep
    from .ui.steps.dataset_step import DatasetStep
    from .ui.steps.export_step import ExportStep
    from .ui.steps.inference_step import InferenceStep
    from .ui.steps.system_step import SystemStep
    from .ui.steps.train_step import TrainStep
    from .ui.steps.verify_step import VerifyStep

    pages: list[StepPage] = [
        SystemStep(),
        DatasetStep(),
        AutolabelStep(),
        VerifyStep(),
        TrainStep(),
        ExportStep(),
        InferenceStep(),
    ]
    if fake_worker:
        from .ui.steps.demo_step import DemoStep

        pages.append(DemoStep())
    return pages


def install_qt_translations(app: QApplication, code: str) -> None:
    """Put Qt's own text -- Yes and No, the folder chooser, number formats -- in the same language.

    Qt ships these catalogues with PySide6. One that is missing leaves those
    buttons in English, which is untidy but not worth refusing to start over.
    """
    if code == i18n.SOURCE_LANGUAGE:
        return
    # Decimal commas in spin boxes. English keeps following the system, as it
    # always has.
    QLocale.setDefault(QLocale(code))
    directory = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    translator = QTranslator(app)
    if translator.load(f"qtbase_{code}", directory):
        app.installTranslator(translator)


def warn_if_python_is_unreachable(window: QWidget) -> None:
    """Say so if Windows started this copy of the app unable to run a worker.

    See ``core/redirection_guard.py``. Before the first page asks for a worker,
    so that the first thing the user reads is the cause rather than a uv error.
    """
    if not redirection_guard.is_enforced():
        return
    QMessageBox.warning(
        window,
        tr("Start the app again"),
        tr(
            "Windows started this window with a restriction that stops it from "
            "running its own Python, so setup, autolabeling and training would all "
            "fail. This happens when the app is opened from the last page of an "
            "older installer.\n\nClose this window and open YOLO "
            "Assistant from the Start menu."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yolo-assistant")
    parser.add_argument(
        "--fake-worker",
        action="store_true",
        help="Add the simulated-run page, which needs no GPU and no managed environment.",
    )
    parser.add_argument(
        "--language",
        choices=sorted(language.available()),
        help="Start in this language, this time only. The Language menu changes it for good.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Check that this installation has everything it needs, then exit.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.fake_worker and bundle_dir() is not None:
        # The simulated run executes the worker under the GUI's own Python, and
        # an installed build has none -- only the executable PyInstaller made.
        parser.error("--fake-worker needs a source checkout; the installed app cannot run it.")

    if args.self_check:
        from .self_check import run

        return run()

    # Create the data layout up front, so "where does this app write?" has a
    # visible answer from the first launch rather than only after a setup run.
    ensure_dirs()

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Assistant")
    app.setOrganizationName("YOLO Assistant")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(QIcon(str(ICON_FILE)))
    # Matches the installed .desktop file, which is how a Wayland compositor
    # finds the icon and groups the windows under it.
    app.setDesktopFileName("yolo-assistant")
    # Before any widget is built, so every one picks up the style and font.
    platform_look.apply(app)

    # Before any page is built: text is translated as it is created.
    code = language.choose(QLocale.system().name(), args.language)
    language.apply(code)
    install_qt_translations(app, code)

    pages = build_pages(fake_worker=args.fake_worker)
    window = WizardWindow(pages)
    window.show()
    warn_if_python_is_unreachable(window)
    for page in pages:
        page.on_app_start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
