"""Build the environment a worker subprocess runs in.

This is small and boring and will save days. PyInstaller rewrites the process
environment so that its own bundled shared libraries win -- and children inherit
that. Spawn the managed venv's Python from a frozen GUI without scrubbing, and
torch loads PyInstaller's libstdc++/OpenSSL/Qt instead of its own. The failure
is rarely a clean ImportError; it is a segfault, or worse, something that half
works.

Kept free of Qt so it can be unit-tested directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import i18n

#: Variables PyInstaller sets (or that a bundled app otherwise pollutes) which
#: must not reach a child running a different interpreter.
LEAKED_VARS: tuple[str, ...] = (
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "LIBPATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "DYLD_INSERT_LIBRARIES",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "QT_PLUGIN_PATH",
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "QML2_IMPORT_PATH",
    "QML_IMPORT_PATH",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "TCL_LIBRARY",
    "TK_LIBRARY",
    "TKPATH",
    "GIO_MODULE_DIR",
    "GST_PLUGIN_PATH",
    "GST_PLUGIN_SYSTEM_PATH",
    "MATPLOTLIBDATA",
    "_MEIPASS2",
)

#: PyInstaller stashes the pre-launch value of a scrubbed variable here.
#: Restoring from it is more correct than simply deleting, because the user may
#: legitimately have had one set.
ORIG_SUFFIX = "_ORIG"


def build_worker_env(
    worker_dir: str | os.PathLike[str],
    *,
    python: str | os.PathLike[str] | None = None,
    project_dir: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    base: dict[str, str] | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return a clean environment for a worker subprocess.

    ``worker_dir`` becomes the sole entry on ``PYTHONPATH`` -- the worker package
    is shipped as loose source precisely so the venv's interpreter can import it.

    ``python`` is the interpreter that will be launched, and giving it puts that
    environment's console scripts on ``PATH``. See :func:`scripts_dir`.
    """
    env = dict(os.environ if base is None else base)

    for name in LEAKED_VARS:
        original = env.pop(f"{name}{ORIG_SUFFIX}", None)
        if original:
            env[name] = original
        else:
            env.pop(name, None)

    # Anything PyInstaller marks as its own.
    for key in [k for k in env if k.startswith("_PYI_")]:
        env.pop(key, None)

    if python is not None:
        env["PATH"] = _with_scripts_dir(env.get("PATH", ""), scripts_dir(python))
        # What activation sets, and what tools use to recognise that they are
        # inside an environment rather than guessing from sys.prefix.
        env["VIRTUAL_ENV"] = str(Path(python).parent.parent)

    env["PYTHONPATH"] = str(Path(worker_dir))
    # A stray ~/.local/lib must not shadow the managed venv.
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    # Ultralytics plots results at the end of training and will otherwise reach
    # for a GUI backend inside a headless subprocess.
    env["MPLBACKEND"] = "Agg"

    if project_dir is not None:
        # Keep Ultralytics' settings per project rather than in a surprise
        # location in the user's home, so two projects cannot contaminate
        # each other and a frozen app writes only where we expect.
        env["YOLO_CONFIG_DIR"] = str(Path(project_dir) / ".ultralytics")
    if cache_dir is not None:
        cache = Path(cache_dir)
        env["HF_HOME"] = str(cache / "huggingface")
        env["TORCH_HOME"] = str(cache / "torch")

    # The worker writes its own error messages, so it has to be told which
    # language the user is reading.
    env.update(i18n.environment())

    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def scripts_dir(python: str | os.PathLike[str]) -> Path:
    """Where an environment's console scripts live, given its interpreter.

    The same directory the interpreter is in, on both platforms: a venv is
    ``<env>/bin/python`` with ``<env>/bin/ninja`` beside it, and
    ``<env>\\Scripts\\python.exe`` with ``<env>\\Scripts\\ninja.exe`` beside that.
    """
    return Path(python).parent


def _with_scripts_dir(path: str, scripts: Path) -> str:
    """``scripts`` first on ``PATH``, without duplicating it.

    Launching ``<venv>/bin/python`` directly does *not* put that venv on
    ``PATH`` -- only ``activate`` does that, and the app never activates
    anything. The interpreter finds its own packages regardless, so for a long
    time nothing noticed; what breaks is a library that shells out to a console
    script its own wheel installed.

    ``ninja`` is the case that found this. Step 5's INT8 QAT reaches
    ``torch.utils.cpp_extension.load``, which checks for ninja by *running*
    ``ninja --version`` rather than by importing anything -- so a ninja pip put
    in ``<venv>/bin`` is installed, importable, and still invisible. The
    extension then fails to build and quantisation silently falls back to a
    slower pure-PyTorch path. Adding ninja to the requirements does not help,
    because it was never missing.
    """
    scripts = Path(scripts)
    entries = [entry for entry in path.split(os.pathsep) if entry]
    # Compared as paths, so that a trailing separator or a different spelling of
    # the same directory does not add it twice.
    entries = [entry for entry in entries if Path(entry) != scripts]
    return os.pathsep.join([str(scripts), *entries])
