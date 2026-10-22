"""Which language the interface is in, and remembering it between launches.

The mechanism -- :func:`~.i18n.tr` and the catalogues -- is in ``i18n.py``,
which has to stay free of anything app-specific because the worker carries a
copy of it. This module is the app's side: where the choice is saved, what the
first launch defaults to, and where the catalogues ship.

A change takes effect at the next start. Text is translated as it is built, and
much of it is built once, so switching in place would leave half the window in
the old language until each page happened to repaint.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import i18n
from .paths import settings_file

#: Where the ``.po`` catalogues ship. Inside the package, so a frozen build
#: carries them as data and a worker can be pointed at the same files.
CATALOG_DIR = Path(__file__).resolve().parents[1] / "resources" / "i18n"


def available() -> dict[str, str]:
    """Code -> the language's name for itself, in menu order."""
    return dict(i18n.LANGUAGES)


def saved() -> str | None:
    """The language the user chose last time, if they ever chose one."""
    try:
        data = json.loads(settings_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    code = data.get("language") if isinstance(data, dict) else None
    return code if code in i18n.LANGUAGES else None


def save(code: str) -> None:
    """Remember ``code`` for the next launch, keeping any other settings."""
    if code not in i18n.LANGUAGES:
        raise ValueError(f"unknown language {code!r}")
    path = settings_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data["language"] = code
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def from_locale(name: str | None) -> str:
    """The offered language matching a locale name such as ``pl_PL``, else English."""
    code = (name or "").replace("-", "_").split("_", 1)[0].lower()
    return code if code in i18n.LANGUAGES else i18n.SOURCE_LANGUAGE


def choose(system_locale: str | None, override: str | None = None) -> str:
    """The language to start in: an explicit override, the saved choice, or the system's."""
    if override in i18n.LANGUAGES:
        return override
    return saved() or from_locale(system_locale)


def apply(code: str) -> None:
    """Put the app -- and every worker it launches from now on -- in ``code``."""
    i18n.set_language(code, CATALOG_DIR)


def current() -> str:
    return i18n.language()
