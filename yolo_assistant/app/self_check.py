"""``--self-check``: is everything this build needs actually inside it?

A frozen build fails in ways a source checkout cannot. A data file left out of
the PyInstaller spec is a crash on the first hover over a parameter; a Qt
plugin left out is a video that never plays in Step 7; a ``uv`` that lost its
executable bit is a Step 1 that cannot start. None of that shows until a user
reaches it, so ``packaging/build.py`` runs this against every build before it
packages one, and a user can run it to answer "is my installation intact?".

Each check is independent and reports a line, so one failure does not hide the
rest. Nothing is started that the app would start on launch -- no probes, no
network -- and the window is built but never shown.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable

from .. import __version__


def run() -> int:
    """Run every check, print one line each, and return the exit status."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([sys.argv[0], "-platform", "offscreen"])
    checks: list[tuple[str, Callable[[], str]]] = [
        ("hover explanations", _tooltips),
        ("translations", _translations),
        ("icon", _icon),
        ("fonts", _fonts),
        ("licence", _licence),
        ("environment requirements", _requirements),
        ("worker source", _worker),
        ("bundled uv", _uv),
        ("video playback", _multimedia),
        ("pages", _pages),
        ("managed environment", _managed_environment),
    ]
    failures = 0
    _say(f"YOLO Assistant {__version__} self-check")
    for name, check in checks:
        try:
            detail = check()
        except Exception as exc:  # noqa: BLE001 - every failure is a finding
            failures += 1
            _say(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            _say(f"ok   {name}: {detail}")
    _say("all checks passed" if not failures else f"{failures} check(s) failed")
    del app
    return 1 if failures else 0


def _say(line: str) -> None:
    # A windowed Windows build has no stdout at all; the exit status still counts.
    if sys.stdout is not None:
        print(line, flush=True)


def _tooltips() -> str:
    from .core import i18n, language
    from .ui.common.param_form import HELP_FILE, HelpTexts, translated_help_file

    count = len(HelpTexts(HELP_FILE).keys())
    for code in language.available():
        if code != i18n.SOURCE_LANGUAGE:
            HelpTexts(translated_help_file(code))
    return f"{count} parameters"


def _translations() -> str:
    from .core import i18n, language

    codes = [code for code in language.available() if code != i18n.SOURCE_LANGUAGE]
    for code in codes:
        catalogue = language.CATALOG_DIR / f"{code}.po"
        if "msgstr" not in catalogue.read_text(encoding="utf-8"):
            raise ValueError(f"{catalogue} has no translations in it")
    return ", ".join(codes)


def _icon() -> str:
    from PySide6.QtGui import QIcon

    from .main import ICON_FILE

    icon = QIcon(str(ICON_FILE))
    if icon.isNull() or icon.pixmap(64, 64).isNull():
        raise RuntimeError(f"{ICON_FILE} did not render; is the SVG plugin bundled?")
    return ICON_FILE.name


def _fonts() -> str:
    from .ui import platform_look

    families = platform_look.load_fonts()
    missing = {platform_look.UI_FAMILY, platform_look.MONO_FAMILY} - families
    if missing:
        raise FileNotFoundError(f"{', '.join(sorted(missing))} not in {platform_look.FONT_DIR}")
    return ", ".join(sorted(families))


def _licence() -> str:
    from .ui import wizard

    path = wizard.licence_file()
    if "GNU AFFERO GENERAL PUBLIC LICENSE" not in path.read_text(encoding="utf-8"):
        raise ValueError(f"{path} is not the AGPL")
    return str(path)


def _requirements() -> str:
    from .core.bootstrap import venv_manager

    path = venv_manager.BASE_REQUIREMENTS
    if "ultralytics" not in path.read_text(encoding="utf-8"):
        raise ValueError(f"{path} does not mention ultralytics")
    return str(path)


def _worker() -> str:
    from .core.paths import worker_source_dir

    root = worker_source_dir()
    for name in ("__main__.py", "protocol.py", "channel.py", "commands/train.py"):
        if not (root / "worker" / name).is_file():
            raise FileNotFoundError(root / "worker" / name)
    return str(root)


def _uv() -> str:
    from .core.bootstrap import uv_manager
    from .core.paths import bundle_dir

    bundled = uv_manager.bundled_uv()
    if bundled is None:
        if bundle_dir() is None:
            return "not applicable to a source checkout"
        return "not bundled; Step 1 will download it"
    completed = subprocess.run(
        [str(bundled), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    version = completed.stdout.strip()
    if uv_manager.UV_VERSION not in version:
        raise ValueError(f"expected uv {uv_manager.UV_VERSION}, got {version!r}")
    return version


def _multimedia() -> str:
    from PySide6.QtMultimedia import QMediaFormat

    formats = QMediaFormat().supportedFileFormats(QMediaFormat.ConversionMode.Decode)
    if not formats:
        raise RuntimeError("Qt has no media backend, so Step 7 cannot play its videos")
    return f"{len(formats)} container formats"


def _pages() -> str:
    from PySide6.QtCore import QCoreApplication, QEvent

    from .core import language
    from .main import build_pages
    from .ui.wizard import WizardWindow

    language.apply("en")
    pages = build_pages(fake_worker=False)
    window = WizardWindow(pages)
    # Torn down as closing the app would, and now rather than whenever the
    # event loop next runs: a page left half-alive can fire into whatever the
    # process does next, which in the test suite is the next test.
    for page in pages:
        page.shutdown()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return f"{len(pages)} steps"


def _managed_environment() -> str:
    """Start ``worker probe`` in the managed environment, the way a step would.

    This is the check a frozen build most needs and a source checkout cannot
    exercise: the worker is the extracted copy, and the environment it starts
    in is the one :func:`build_worker_env` scrubbed of PyInstaller's library
    paths. When that scrubbing misses something, torch loads the bundle's
    libraries instead of its own, and the symptom is a crash in a native
    library rather than an error message.
    """
    from .core import protocol
    from .core.bootstrap import venv_manager
    from .core.paths import cache_dir, worker_source_dir
    from .core.runtime import managed_environment
    from .core.worker_env import build_worker_env

    found = managed_environment()
    if found is None:
        return "none set up yet, so not checked"
    python = found[0].python
    completed = subprocess.run(
        [str(python), "-u", "-m", "worker", "probe",
         "--job", json.dumps({"packages": list(venv_manager.PACKAGE_NAMES)})],
        env=build_worker_env(worker_source_dir(), python=python, cache_dir=cache_dir()),
        cwd=worker_source_dir(),
        capture_output=True,
        timeout=600,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    decoder = protocol.FrameDecoder(require_record_sep=True)
    results = [m for m in decoder.feed(completed.stdout) if m.type is protocol.MsgType.RESULT]
    if completed.returncode != 0 or not results or not results[-1].data.get("ok"):
        tail = completed.stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
        raise RuntimeError(
            f"worker probe exited with {completed.returncode} in {python}: " + " | ".join(tail)
        )
    info = results[-1].data.get("payload") or {}
    missing = [name for name, version in (info.get("packages") or {}).items() if version is None]
    detail = f"torch {info.get('torch')}, GPU {'visible' if info.get('cuda_available') else 'NOT visible'}"
    return detail + (f", cannot import {', '.join(missing)}" if missing else "")
