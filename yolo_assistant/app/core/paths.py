"""Where the application keeps everything it owns.

One module so that "what does this app write, and where?" has a single answer --
which matters for an uninstaller, for a support request, and for tests, which
set ``UMLA_DATA_DIR`` and get a completely isolated installation.

Nothing here touches the user's home beyond these roots, and nothing is written
next to the application itself: a frozen app in ``C:\\Program Files`` cannot
write beside its executable, and on Linux the app may be read-only.
"""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

APP_DIRNAME = "YOLOAssistant" if os.name == "nt" else "yolo-assistant"

#: Set this to relocate the entire installation. Tests rely on it.
DATA_DIR_ENV = "UMLA_DATA_DIR"


def data_root() -> Path:
    """Root of everything the app installs: uv, Pythons, venvs, caches, logs."""
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        # LOCALAPPDATA, not APPDATA: these are machine-specific, often many
        # gigabytes, and must never be pushed through a roaming profile.
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    elif sys.platform == "darwin":  # pragma: no cover - not a target platform
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / APP_DIRNAME


def tools_dir() -> Path:
    """The bundled/downloaded uv binary lives here."""
    return data_root() / "tools"


def pythons_dir() -> Path:
    """Interpreters uv installs for us, outside any system Python."""
    return data_root() / "pythons"


def venvs_dir() -> Path:
    """Managed environments, one directory per fingerprint."""
    return data_root() / "venvs"


def projects_dir() -> Path:
    """Default home for prepared datasets, when the user picks no other place.

    Unlike everything else under the data root this holds the *user's* data, and
    it can be tens of gigabytes. Step 2 therefore offers it as a default rather
    than imposing it, and will happily put a project on another drive.
    """
    return data_root() / "projects"


def state_file() -> Path:
    """Which project the user was last working on.

    Deliberately a pointer and nothing else. It records where to look, never
    what was found there: a project's state is always read back off disk.
    """
    return data_root() / "state.json"


def settings_file() -> Path:
    """The user's preferences -- at present only the interface language.

    Kept apart from :func:`state_file`, which points at a project and is
    rewritten whenever the user opens another one.
    """
    return data_root() / "settings.json"


def cache_dir() -> Path:
    """Model and wheel caches: Hugging Face, torch hub, uv's own cache."""
    return data_root() / "cache"


def logs_dir() -> Path:
    return data_root() / "logs"


def workers_dir() -> Path:
    """Where the ``worker`` package is extracted from a frozen build.

    It cannot be imported out of PyInstaller's archive -- the venv's interpreter
    has no idea what ``sys._MEIPASS`` is -- so it ships as loose source and is
    unpacked here, keyed by build hash so an upgrade cannot half-overwrite the
    copy a running job is importing.
    """
    return data_root() / "workers"


def bundle_dir() -> Path | None:
    """Where a frozen build keeps its bundled files, or ``None`` from source."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return None


def worker_source_dir() -> Path:
    """The directory to put on a worker's ``PYTHONPATH``.

    In a source checkout that is the repository root, where ``worker/`` sits.
    In a frozen build it is the extracted copy, because ``worker/`` cannot be
    imported out of PyInstaller's archive by an interpreter that has never
    heard of ``sys._MEIPASS``.
    """
    bundle = bundle_dir()
    if bundle is None:
        return Path(__file__).resolve().parents[3]
    return _extracted_worker(bundle, workers_dir())


@functools.cache
def _extracted_worker(bundle: Path, workers: Path) -> Path:
    """Copy the bundled ``worker/`` to ``workers/<hash>`` once, and return that.

    The hash is of the source itself, so a reinstall of the same build reuses
    the copy and an upgrade gets a directory of its own rather than rewriting
    files a running job may be importing. The copy is assembled under a
    temporary name and renamed into place, so a second instance starting at
    the same moment, or a crash halfway, never leaves a directory that exists
    but is missing half its modules.

    Cached per root because ``UMLA_DATA_DIR`` can change under tests.
    """
    source = bundle / "worker"
    files = sorted(
        path for path in source.rglob("*.py") if "__pycache__" not in path.parts
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    target = workers / digest.hexdigest()[:16]
    if (target / "worker" / "__main__.py").exists():
        return target

    workers.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{target.name}.", dir=workers))
    try:
        for path in files:
            copy = staging / "worker" / path.relative_to(source)
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, copy)
        try:
            os.replace(staging, target)
        except OSError:
            # Either another instance got there first, and its copy is
            # identical, or something removed files from a finished copy --
            # antivirus quarantine, a user tidying up -- and it is replaced.
            if not (target / "worker" / "__main__.py").exists():
                shutil.rmtree(target, ignore_errors=True)
                os.replace(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def ensure_dirs() -> None:
    for path in (tools_dir(), pythons_dir(), venvs_dir(), cache_dir(), logs_dir(),
                 projects_dir()):
        path.mkdir(parents=True, exist_ok=True)
