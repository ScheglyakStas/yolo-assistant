"""Turn uv's output into progress, without pretending to more precision than it has.

Parsing another tool's console output is normally a mistake, and everywhere else
in this app it is banned -- training metrics come from callbacks, never stdout.
uv is the exception for one unavoidable reason: it is not our process and has no
other channel. What makes it tolerable is that uv detects a non-tty and switches
to plain, stable, newline-terminated lines with no carriage returns::

    Resolved 137 packages in 648ms
    Downloading torch (821.2MiB)
     Downloaded sympy
    Prepared 137 packages in 3.95s
    Installed 137 packages in 69ms
     + torch==2.14.0

The denominator is the resolved package count, which is known early and never
changes. Completed downloads are the numerator, and packages served from cache
are never downloaded at all -- so the bar legitimately stops short, and is
snapped to full when ``Prepared`` arrives. A bar that jumps forward at the end
is honest; one that slides backwards because the total grew is not, which is why
the announced-download count is deliberately not used as the total.

Everything here is pure text in, dataclass out, so it is tested without a
network, a GPU or Qt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..i18n import tr

RESOLVED_RE = re.compile(r"^Resolved (\d+) packages?")
DOWNLOADING_RE = re.compile(r"^\s*Downloading (\S+)")
#: The size is not always the first parenthesised group: uv's Python downloads
#: read "Downloading cpython-3.12.14-linux-x86_64-gnu (download) (32.6MiB)".
SIZE_RE = re.compile(r"\(([\d.]+)\s*([KMG]i?B)\)")
DOWNLOADED_RE = re.compile(r"^\s+Downloaded (\S+)")
PREPARED_RE = re.compile(r"^Prepared (\d+) packages?")
INSTALLED_RE = re.compile(r"^(?:Installed|Uninstalled|Audited) (\d+) packages?")
ADDED_RE = re.compile(r"^\s*\+ (\S+?)==(\S+)")
ERROR_RE = re.compile(r"^\s*error:\s*(.+)", re.IGNORECASE)
PYTHON_INSTALLED_RE = re.compile(r"^Installed Python (\S+)")

_UNITS = {"KB": 1e3, "KiB": 1024, "MB": 1e6, "MiB": 1024**2, "GB": 1e9, "GiB": 1024**3}


@dataclass
class UvProgress:
    """Running state of one uv invocation."""

    resolved: int = 0
    downloaded: int = 0
    in_flight: dict[str, float] = field(default_factory=dict)
    downloaded_bytes: float = 0.0
    announced_bytes: float = 0.0
    prepared: bool = False
    installed: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    python_version: str = ""

    @property
    def current(self) -> int:
        return self.resolved if self.prepared else self.downloaded

    @property
    def total(self) -> int | None:
        return self.resolved or None

    def describe(self) -> str:
        """A one-line summary for the progress header."""
        if self.prepared:
            return tr("Installing")
        if self.in_flight:
            biggest = max(self.in_flight, key=lambda name: self.in_flight[name])
            return tr("Downloading {package}", package=biggest)
        if self.resolved:
            return tr("Preparing packages")
        return tr("Resolving packages")

    def describe_bytes(self) -> str:
        if not self.announced_bytes:
            return ""
        # Scale to the larger figure so a 40 MB install does not read
        # "0.0 / 0.0 GB" and a 10 GB one does not read "9800 / 10240 MB".
        if self.announced_bytes >= 1024**3:
            return tr(
                "{done:.1f} / {total:.1f} GB",
                done=self.downloaded_bytes / 1024**3,
                total=self.announced_bytes / 1024**3,
            )
        return tr(
            "{done:.0f} / {total:.0f} MB",
            done=self.downloaded_bytes / 1024**2,
            total=self.announced_bytes / 1024**2,
        )


class UvOutputParser:
    """Feed it lines; ask it for state."""

    def __init__(self) -> None:
        self.state = UvProgress()
        self._sizes: dict[str, float] = {}

    def feed(self, line: str) -> bool:
        """Consume one line. Returns whether anything observable changed."""
        text = line.rstrip("\r\n")
        if not text.strip():
            return False
        state = self.state

        match = RESOLVED_RE.match(text)
        if match:
            state.resolved = int(match.group(1))
            return True

        match = DOWNLOADED_RE.match(text)
        if match:
            name = match.group(1)
            state.downloaded += 1
            size = state.in_flight.pop(name, self._sizes.get(name, 0.0))
            state.downloaded_bytes += size
            return True

        match = DOWNLOADING_RE.match(text)
        if match:
            name = match.group(1)
            size_match = SIZE_RE.search(text)
            size = _bytes(*size_match.groups()) if size_match else 0.0
            state.in_flight[name] = size
            self._sizes[name] = size
            state.announced_bytes += size
            return True

        if PREPARED_RE.match(text):
            state.prepared = True
            return True

        match = ADDED_RE.match(text)
        if match:
            state.installed[match.group(1)] = match.group(2)
            return True

        match = PYTHON_INSTALLED_RE.match(text)
        if match:
            state.python_version = match.group(1)
            return True

        match = ERROR_RE.match(text)
        if match:
            state.errors.append(match.group(1).strip())
            return True

        return bool(INSTALLED_RE.match(text))

    def failure_hint(self) -> str:
        """Turn uv's error text into something a beginner can act on."""
        joined = " ".join(self.state.errors).lower()
        if not joined:
            return ""
        if "no solution" in joined or "cannot be installed" in joined:
            return tr(
                "The package versions could not be reconciled. This usually means the "
                "selected CUDA backend has no build for one of the requirements -- try "
                "the next backend down."
            )
        if "network" in joined or "connect" in joined or "timed out" in joined or "dns" in joined:
            return tr(
                "The download could not reach the package index. Check the network "
                "connection, or a proxy or firewall that blocks pypi.org and "
                "download.pytorch.org."
            )
        if "no space" in joined or "disk" in joined:
            return tr(
                "There is not enough free disk space. The environment needs roughly 10-14 GB."
            )
        if "permission" in joined or "denied" in joined or "os error 5" in joined:
            return tr(
                "A file could not be written. Antivirus software sometimes locks files "
                "mid-install; excluding the application's data folder usually fixes it."
            )
        return ""


def _bytes(value: str | None, unit: str | None) -> float:
    if not value or not unit:
        return 0.0
    try:
        return float(value) * _UNITS.get(unit, 1.0)
    except ValueError:
        return 0.0
