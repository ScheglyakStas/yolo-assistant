"""Is there a newer Ultralytics, and is it safe to take?

The spec asks for an update button. The honest version of that button has to
answer two questions, not one: *is there a newer release*, and *is it one this
application has been tested against*. The 8.4 line renamed export arguments
mid-series -- ``half`` and ``int8`` became ``quantize`` -- so "newest" and
"works with this app" are genuinely different things, and an update that breaks
Step 6 for a beginner is worse than no update button at all.

So an out-of-range release is still offered, and still installable, but it is
labelled as untested rather than recommended.

PyPI is queried from the GUI process: it needs no venv, which means the button
works even when the environment is missing, and the answer can be compared
against whatever ``ready.json`` recorded.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .i18n import tr

PYPI_URL = "https://pypi.org/pypi/ultralytics/json"

#: The range this application's Step 5 and Step 6 code has been written
#: against. Widen it deliberately, after testing the export matrix.
TESTED_RANGE = ">=8.4.154,<8.5"


@dataclass(frozen=True)
class UpdateInfo:
    installed: str | None = None
    latest: str | None = None
    error: str = ""

    @property
    def available(self) -> bool:
        """Whether ``latest`` is genuinely newer than what is installed."""
        if not self.installed or not self.latest:
            return False
        try:
            return Version(self.latest) > Version(self.installed)
        except InvalidVersion:
            return self.latest != self.installed

    @property
    def latest_is_tested(self) -> bool:
        return _in_range(self.latest)

    @property
    def installed_is_tested(self) -> bool:
        return _in_range(self.installed)

    def summary(self) -> str:
        """One line for the status row."""
        if self.error:
            return tr("Could not check for updates: {error}", error=self.error)
        if not self.installed:
            return tr("Set up the environment before checking for updates.")
        if not self.available:
            return tr("Ultralytics {installed} is the newest release.", installed=self.installed)
        if self.latest_is_tested:
            return tr(
                "Ultralytics {latest} is available (currently {installed}).",
                latest=self.latest,
                installed=self.installed,
            )
        return tr(
            "Ultralytics {latest} is available (currently {installed}), but it is "
            "outside the range this app was tested with ({tested}). Updating may break "
            "training or export.",
            latest=self.latest,
            installed=self.installed,
            tested=TESTED_RANGE,
        )


def latest_release(*, timeout: float = 15.0, opener=None) -> tuple[str | None, str]:
    """Ask PyPI for the newest Ultralytics. Returns ``(version, error)``."""
    try:
        fetch = opener or urllib.request.urlopen
        with fetch(PYPI_URL, timeout=timeout) as response:  # noqa: S310 - fixed https host
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return None, str(exc) or type(exc).__name__
    version = ((payload or {}).get("info") or {}).get("version")
    return (str(version) if version else None), ""


def check(installed: str | None, *, timeout: float = 15.0, opener=None) -> UpdateInfo:
    latest, error = latest_release(timeout=timeout, opener=opener)
    return UpdateInfo(installed=installed, latest=latest, error=error)


def _in_range(version: str | None) -> bool:
    if not version:
        return False
    try:
        return Version(version) in SpecifierSet(TESTED_RANGE)
    except (InvalidVersion, InvalidSpecifier):
        return False
