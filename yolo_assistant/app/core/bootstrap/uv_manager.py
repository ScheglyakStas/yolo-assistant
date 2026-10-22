"""Obtain ``uv``, and build the command lines that use it.

``uv`` solves the problem that otherwise has no clean answer: *with which
Python do we create the Python environment?* It is a single static binary that
needs no Python at all, and ``uv python install`` fetches a self-contained
CPython into our own data directory -- no administrator rights, no PATH
changes, and nothing added to or removed from the user's system.

Alternatives considered and rejected: Windows' embeddable zip has no pip and
does not support venvs; the python.org installer needs UAC and mutates the
machine; micromamba is ~100 MB and Anaconda's default channel carries real
commercial-use terms.

The binary is downloaded over HTTPS and its SHA-256 checked against the pinned
value below before anything is executed. That check is not optional: this
module's whole job is to fetch a file and then run it.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..i18n import tr
from ..paths import bundle_dir, cache_dir, pythons_dir, tools_dir

#: Pinned. Bumping it means replacing every checksum below -- they are published
#: as ``<asset>.sha256`` next to each release asset.
UV_VERSION = "0.12.15"
UV_RELEASE_URL = "https://github.com/astral-sh/uv/releases/download"


@dataclass(frozen=True)
class UvAsset:
    name: str
    sha256: str

    @property
    def url(self) -> str:
        return f"{UV_RELEASE_URL}/{UV_VERSION}/{self.name}"


#: Keyed by ``(os.name, normalised machine)``.
UV_ASSETS: dict[tuple[str, str], UvAsset] = {
    ("posix", "x86_64"): UvAsset(
        "uv-x86_64-unknown-linux-gnu.tar.gz",
        "f97935763c04be3e692460a7aaeaaab8fc3b78fcf8b389da820b38ae7423a638",
    ),
    ("posix", "aarch64"): UvAsset(
        "uv-aarch64-unknown-linux-gnu.tar.gz",
        "0e9a3499b0587d449c9ff684c0160da607826e4af1cee220bc87f378702d3e08",
    ),
    ("nt", "x86_64"): UvAsset(
        "uv-x86_64-pc-windows-msvc.zip",
        "477bd99a84e34891f2bd4c9152ddeb74e971accccbc59c0f0301f11f08a32d46",
    ),
    ("nt", "aarch64"): UvAsset(
        "uv-aarch64-pc-windows-msvc.zip",
        "a37c8e96cb1260488c8510b64c848533a3a82a2fdf9e905de7c2700ceebf6437",
    ),
}

EXE_NAME = "uv.exe" if os.name == "nt" else "uv"


class UvUnavailable(RuntimeError):
    """Raised when uv is neither present nor obtainable."""


class UvCancelled(UvUnavailable):
    """Raised when the caller asked to stop mid-download."""


def normalised_machine() -> str:
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        return "x86_64"
    if machine in ("arm64", "aarch64"):
        return "aarch64"
    return machine


def asset_for_this_machine() -> UvAsset:
    key = (os.name, normalised_machine())
    try:
        return UV_ASSETS[key]
    except KeyError:
        raise UvUnavailable(
            tr(
                "No uv build is pinned for {system} {machine}. "
                "Install uv manually and make sure it is on PATH.",
                system=platform.system(),
                machine=platform.machine(),
            )
        ) from None


def managed_uv() -> Path:
    """Where a downloaded uv is kept."""
    return tools_dir() / EXE_NAME


def bundled_uv() -> Path | None:
    """uv shipped inside the application directory, if the build included one."""
    root = bundle_dir()
    if root is None:
        return None
    candidate = root / "tools" / EXE_NAME
    return candidate if candidate.exists() else None


def find_uv() -> Path | None:
    """Locate a usable uv without downloading one.

    Order matters: a uv we shipped or installed is a known version, whereas one
    on PATH is whatever the user happens to have. The user's is still preferred
    over a download -- it saves ~20 MB and usually works -- but it is reported
    so an odd failure can be traced to it.
    """
    for candidate in (bundled_uv(), managed_uv()):
        if candidate is not None and candidate.exists():
            return candidate
    found = shutil.which("uv")
    return Path(found) if found else None


def download_uv(
    *,
    progress=None,
    should_stop=None,
    chunk_size: int = 1 << 16,
    timeout: float = 60.0,
) -> Path:
    """Fetch, verify and install uv. Returns the executable's path.

    ``progress`` is called as ``progress(downloaded_bytes, total_bytes)``; total
    is ``0`` when the server does not say.

    ``should_stop`` is polled once per chunk. urllib offers no way to interrupt
    a read from outside, so abandoning the download has to be cooperative --
    and it has to be possible, because otherwise closing the window during this
    download leaves a thread writing into a destroyed object.
    """
    asset = asset_for_this_machine()
    tools_dir().mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="uv-download-") as tmp:
        archive = Path(tmp) / asset.name
        digest = hashlib.sha256()
        with urllib.request.urlopen(asset.url, timeout=timeout) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            with archive.open("wb") as handle:
                while True:
                    if should_stop is not None and should_stop():
                        raise UvCancelled("The uv download was stopped.")
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total)

        actual = digest.hexdigest()
        if actual != asset.sha256:
            raise UvUnavailable(
                tr(
                    "The downloaded uv archive does not match its published checksum "
                    "(expected {expected}..., got {actual}...). "
                    "Nothing was installed.",
                    expected=asset.sha256[:16],
                    actual=actual[:16],
                )
            )

        extracted = _extract(archive, Path(tmp) / "unpacked")
        destination = managed_uv()
        # Replace atomically: a half-written binary that still runs is worse
        # than one that is missing.
        staged = destination.with_suffix(destination.suffix + ".new")
        shutil.copy2(extracted, staged)
        staged.chmod(staged.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.replace(staged, destination)
        return destination


def _extract(archive: Path, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(into)
    else:
        with tarfile.open(archive) as bundle:
            # filter="data" refuses absolute paths and traversal entries. The
            # archive is checksum-verified, but this costs nothing.
            bundle.extractall(into, filter="data")
    for candidate in into.rglob(EXE_NAME):
        return candidate
    raise UvUnavailable(tr("The uv archive did not contain {name}.", name=EXE_NAME))


# -- command lines --------------------------------------------------------- #


def uv_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a uv invocation.

    uv's cache goes in our data directory so that uninstalling the app really
    removes several gigabytes of wheels, and ``UV_PYTHON_INSTALL_DIR`` keeps the
    managed interpreters there too rather than in a shared location another
    tool might garbage-collect.
    """
    env = dict(os.environ if base is None else base)
    env["UV_CACHE_DIR"] = str(cache_dir() / "uv")
    env["UV_PYTHON_INSTALL_DIR"] = str(pythons_dir())
    # Never silently fall back to a system interpreter we did not choose.
    env["UV_PYTHON_DOWNLOADS"] = "automatic"
    env["UV_NO_PROGRESS"] = "0"
    return env


def python_install_argv(uv: Path, version: str) -> list[str]:
    return [str(uv), "python", "install", version]


def python_find_argv(uv: Path, version: str) -> list[str]:
    return [str(uv), "python", "find", version]


def venv_argv(uv: Path, venv: Path, version: str) -> list[str]:
    return [str(uv), "venv", "--python", version, str(venv)]


def pip_install_argv(
    uv: Path,
    venv_python: Path,
    *,
    requirements: Path | None = None,
    packages: list[str] | None = None,
    torch_backend: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    argv = [str(uv), "pip", "install", "--python", str(venv_python)]
    if torch_backend:
        # uv sends torch, torchvision and the rest of the PyTorch ecosystem to
        # download.pytorch.org/whl/<backend>, and everything else to PyPI.
        #
        # Not ``--index-url <torch index> --extra-index-url <PyPI>``, which looks
        # equivalent and is not: in uv an *extra* index outranks ``--index-url``,
        # so with first-index matching PyPI was asked for torch first and always
        # answered. On Linux that happens to be a CUDA build; on Windows it is
        # the CPU-only wheel, and the GPU silently disappears.
        argv += ["--torch-backend", torch_backend]
    if requirements is not None:
        argv += ["-r", str(requirements)]
    argv += list(packages or [])
    argv += list(extra_args or [])
    return argv


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
