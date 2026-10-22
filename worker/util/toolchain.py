"""Make the compilers Model Optimizer's INT8 kernel needs reachable on Windows.

Model Optimizer compiles its fake-quantisation kernel on first use, through
``torch.utils.cpp_extension``. On Linux that needs a C++ compiler the system
already has. On Windows it needs two things torch's wheels do not carry, and
neither is found the way torch looks for it:

**The Microsoft C++ compiler.** torch runs ``cl`` straight off ``PATH``. Visual
Studio and its Build Tools never put it there: only a "Developer Command
Prompt" does, by running ``vcvars64.bat``, and the app is not launched from
one. So a machine with the compiler installed fails exactly like a machine
without it. This finds the installation the way Visual Studio's own tools do,
with ``vswhere``, runs ``vcvars64.bat`` once and copies the environment it
leaves behind into this process.

**The CUDA Toolkit.** torch takes ``CUDA_HOME`` or ``CUDA_PATH`` and refuses
outright when that toolkit's major version is not the one torch was built
with. The NVIDIA installer points ``CUDA_PATH`` at whichever toolkit was
installed last, and leaves one ``CUDA_PATH_V<major>_<minor>`` per toolkit, so
a machine with a matching toolkit installed beside a newer one is told it has
the wrong one. This picks the matching one and says so in ``CUDA_HOME``. CUDA 13 also needs ``cl`` told
to use its standard preprocessor, which nothing upstream asks for.

All of it has to happen before ``torch.utils.cpp_extension`` is imported: it reads
``CUDA_HOME`` once, into a module constant. Neither torch nor Ultralytics
imports it; Model Optimizer does.

Kept free of torch so it can be tested anywhere.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import MutableMapping
from pathlib import Path

#: The Visual Studio component that holds the x64 compiler. Asking vswhere for
#: it skips an installation that has the IDE but not the C++ workload.
VC_TOOLS_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"

#: Where the NVIDIA installer puts every toolkit, one ``v<major>.<minor>`` each.
DEFAULT_CUDA_ROOT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")

#: vcvars64.bat is a few seconds on a warm machine, and much longer on a cold
#: one with antivirus inspecting every batch file it calls.
VCVARS_TIMEOUT_S = 180


def activate_msvc(environ: MutableMapping[str, str] | None = None) -> bool:
    """Put ``cl`` on ``PATH`` if Visual Studio has one. True if it is there now."""
    environ = os.environ if environ is None else environ
    if shutil.which("cl", path=environ.get("PATH")):
        return True
    vcvars = find_vcvars(environ)
    if vcvars is None:
        return False
    after = _environment_after(vcvars, environ)
    if not after:
        return False
    environ.update({key: value for key, value in after.items() if environ.get(key) != value})
    # vcvars puts its own directories in front, and Visual Studio carries a
    # ninja of its own among them. The environment's script directory goes back
    # first, so the ninja that runs is still the one the requirements pin.
    environ["PATH"] = _first_on_path(environ.get("PATH", ""), Path(sys.executable).parent)
    return bool(shutil.which("cl", path=environ.get("PATH")))


def find_vcvars(environ: MutableMapping[str, str] | None = None) -> Path | None:
    """The newest Visual Studio's ``vcvars64.bat`` with the C++ compiler in it."""
    environ = os.environ if environ is None else environ
    program_files = environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    try:
        result = subprocess.run(
            [str(vswhere), "-latest", "-products", "*", "-prerelease",
             "-requires", VC_TOOLS_COMPONENT, "-property", "installationPath", "-utf8"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in result.stdout.splitlines():
        if line.strip():
            candidate = Path(line.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if candidate.is_file():
                return candidate
    return None


def _environment_after(vcvars: Path, environ: MutableMapping[str, str]) -> dict[str, str]:
    """The environment ``vcvars`` leaves behind, or ``{}`` if it failed.

    Read back by a Python started in the same ``cmd`` rather than with ``set``:
    JSON is unambiguous about a value with a newline or a character outside the
    console's code page in it, and ``set``'s output is neither.
    """
    command = (
        f'cmd.exe /d /c ""{vcvars}" >nul 2>&1 && "{sys.executable}" -I -c '
        f'"import json,os;print(json.dumps(dict(os.environ)))""'
    )
    try:
        result = subprocess.run(
            command, capture_output=True, env=dict(environ), timeout=VCVARS_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    try:
        after = json.loads(result.stdout.decode("utf-8", errors="replace").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {}
    return {str(key): str(value) for key, value in after.items()} if isinstance(after, dict) else {}


def select_cuda_home(
    torch_cuda: str | None, environ: MutableMapping[str, str] | None = None
) -> Path | None:
    """Point ``CUDA_HOME`` at a toolkit torch will accept, and return it.

    That is one with ``nvcc`` in it and the same major version as
    ``torch_cuda``. ``None`` means there is none, and ``CUDA_HOME`` is left
    alone, so that torch's own message about it is not made any worse.
    """
    environ = os.environ if environ is None else environ
    if not torch_cuda:
        return None
    major = torch_cuda.split(".")[0]
    for home in _cuda_candidates(environ):
        if not (home / "bin" / "nvcc.exe").is_file():
            continue
        version = cuda_version(home)
        if version is None or version.split(".")[0] == major:
            environ["CUDA_HOME"] = str(home)
            if int(major) >= 13:
                _use_standard_preprocessor(environ)
            return home
    return None


def _use_standard_preprocessor(environ: MutableMapping[str, str]) -> None:
    """Have nvcc ask ``cl`` for its standard-conforming preprocessor.

    CUDA 13's own headers stop with ``#error`` under MSVC's traditional one,
    which is still ``cl``'s default, and neither torch nor Model Optimizer
    passes the switch. nvcc adds ``NVCC_APPEND_FLAGS`` to every command line it
    is given, which is the one way in that does not involve either of them.
    """
    flag = "-Xcompiler=/Zc:preprocessor"
    current = environ.get("NVCC_APPEND_FLAGS", "")
    if "/Zc:preprocessor" not in current:
        environ["NVCC_APPEND_FLAGS"] = f"{current} {flag}".strip()


def _cuda_candidates(environ: MutableMapping[str, str]) -> list[Path]:
    """Toolkits in the order to try them: what the user chose, then newest first."""
    chosen = [Path(environ[key]) for key in ("CUDA_HOME", "CUDA_PATH") if environ.get(key)]
    versioned = [
        Path(value) for key, value in environ.items()
        if key.upper().startswith("CUDA_PATH_V") and value
    ]
    if DEFAULT_CUDA_ROOT.is_dir():
        versioned += [path for path in DEFAULT_CUDA_ROOT.glob("v*.*") if path.is_dir()]
    versioned.sort(key=lambda home: _version_key(cuda_version(home)), reverse=True)
    seen: set[str] = set()
    ordered = []
    for home in chosen + versioned:
        key = os.path.normcase(str(home))
        if key not in seen:
            seen.add(key)
            ordered.append(home)
    return ordered


def cuda_version(home: Path) -> str | None:
    """A toolkit's version from its ``version.json``, or failing that its folder name."""
    try:
        data = json.loads((home / "version.json").read_text(encoding="utf-8"))
        return str(data["cuda"]["version"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    match = re.fullmatch(r"v(\d+\.\d+)", home.name)
    return match.group(1) if match else None


def _version_key(version: str | None) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version or ""))


def _first_on_path(path: str, directory: Path) -> str:
    entries = [entry for entry in path.split(os.pathsep) if entry and Path(entry) != directory]
    return os.pathsep.join([str(directory), *entries])
