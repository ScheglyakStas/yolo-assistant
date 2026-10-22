"""Finding the Microsoft C++ compiler and a CUDA Toolkit for Model Optimizer on Windows.

Nothing here needs Visual Studio or a toolkit installed: both are directories
with a few files in them, as far as the search is concerned, and are built in
``tmp_path``. The one test that runs ``cmd.exe`` uses a batch file of its own
in place of ``vcvars64.bat``.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from worker.util import toolchain


@pytest.fixture(autouse=True)
def no_default_cuda_root(monkeypatch, tmp_path):
    """Keep a real toolkit on the machine running the tests out of the answers."""
    monkeypatch.setattr(toolchain, "DEFAULT_CUDA_ROOT", tmp_path / "no-default-root")


def make_toolkit(root: Path, version: str, *, nvcc: bool = True) -> Path:
    home = root / f"v{'.'.join(version.split('.')[:2])}"
    (home / "bin").mkdir(parents=True)
    if nvcc:
        (home / "bin" / "nvcc.exe").write_bytes(b"")
    (home / "version.json").write_text(
        json.dumps({"cuda": {"name": "CUDA SDK", "version": version}}), encoding="utf-8"
    )
    return home


def make_program(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    program = directory / (f"{name}.exe" if os.name == "nt" else name)
    program.write_bytes(b"")
    program.chmod(program.stat().st_mode | stat.S_IEXEC)
    return program


# -- the CUDA Toolkit ------------------------------------------------------------ #


def test_the_toolkit_matching_torch_wins_over_the_newest(tmp_path):
    """The installer points CUDA_PATH at the last one installed; torch refuses a
    different major version outright."""
    old = make_toolkit(tmp_path, "12.9.1")
    new = make_toolkit(tmp_path, "13.4.2")
    environ = {"CUDA_PATH": str(new), "CUDA_PATH_V12_9": str(old), "CUDA_PATH_V13_4": str(new)}

    assert toolchain.select_cuda_home("12.8", environ) == old
    assert environ["CUDA_HOME"] == str(old)


def test_the_users_own_choice_is_kept_when_it_fits(tmp_path):
    chosen = make_toolkit(tmp_path, "13.3.0")
    newer = make_toolkit(tmp_path, "13.4.2")
    environ = {"CUDA_PATH": str(chosen), "CUDA_PATH_V13_4": str(newer)}

    assert toolchain.select_cuda_home("13.2", environ) == chosen


def test_a_toolkit_without_nvcc_is_passed_over(tmp_path):
    broken = make_toolkit(tmp_path, "13.4.2", nvcc=False)
    working = make_toolkit(tmp_path, "13.3.0")
    environ = {"CUDA_PATH": str(broken), "CUDA_PATH_V13_3": str(working)}

    assert toolchain.select_cuda_home("13.2", environ) == working


def test_no_matching_toolkit_leaves_cuda_home_alone(tmp_path):
    other = make_toolkit(tmp_path, "12.9.1")
    environ = {"CUDA_PATH": str(other)}

    assert toolchain.select_cuda_home("13.2", environ) is None
    assert "CUDA_HOME" not in environ


def test_the_default_install_folder_is_searched_too(tmp_path, monkeypatch):
    """CUDA_PATH is lost by anything that rebuilds the environment from scratch."""
    root = tmp_path / "CUDA"
    home = make_toolkit(root, "13.4.2")
    monkeypatch.setattr(toolchain, "DEFAULT_CUDA_ROOT", root)

    assert toolchain.select_cuda_home("13.2", {}) == home


def test_cuda_13_gets_cls_standard_preprocessor(tmp_path):
    """CUDA 13's headers #error out under cl's traditional preprocessor."""
    home = make_toolkit(tmp_path, "13.4.2")
    environ = {"CUDA_PATH": str(home), "NVCC_APPEND_FLAGS": "-lineinfo"}

    toolchain.select_cuda_home("13.2", environ)
    toolchain.select_cuda_home("13.2", environ)

    assert environ["NVCC_APPEND_FLAGS"] == "-lineinfo -Xcompiler=/Zc:preprocessor"


def test_cuda_12_is_compiled_as_it_always_was(tmp_path):
    home = make_toolkit(tmp_path, "12.9.1")
    environ = {"CUDA_PATH": str(home)}

    toolchain.select_cuda_home("12.8", environ)

    assert "NVCC_APPEND_FLAGS" not in environ


def test_a_toolkit_version_comes_from_its_folder_when_version_json_is_missing(tmp_path):
    home = tmp_path / "v12.6"
    home.mkdir()

    assert toolchain.cuda_version(home) == "12.6"
    assert toolchain.cuda_version(tmp_path) is None


# -- the Microsoft C++ compiler -------------------------------------------------- #


def test_a_compiler_already_on_path_is_used_as_it_is(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    make_program(bin_dir, "cl")
    monkeypatch.setattr(toolchain, "find_vcvars", lambda *a: pytest.fail("ran vswhere"))
    environ = {"PATH": str(bin_dir)}

    assert toolchain.activate_msvc(environ) is True
    assert environ == {"PATH": str(bin_dir)}


def test_without_visual_studio_there_is_no_compiler(tmp_path):
    environ = {"PATH": str(tmp_path / "empty"), "ProgramFiles(x86)": str(tmp_path)}

    assert toolchain.find_vcvars(environ) is None
    assert toolchain.activate_msvc(environ) is False


def test_the_environment_vcvars_leaves_is_taken_over(tmp_path, monkeypatch):
    msvc_bin = tmp_path / "msvc" / "bin"
    make_program(msvc_bin, "cl")
    scripts = Path(sys.executable).parent
    before = os.pathsep.join([str(scripts), str(tmp_path / "other")])
    after = {
        # What vcvars does: its own directories in front of what was there.
        "PATH": os.pathsep.join([str(msvc_bin), before]),
        "INCLUDE": r"C:\VS\include",
        "LIB": r"C:\VS\lib",
    }
    monkeypatch.setattr(toolchain, "find_vcvars", lambda *a: tmp_path / "vcvars64.bat")
    monkeypatch.setattr(toolchain, "_environment_after", lambda *a: after)
    environ = {"PATH": before}

    assert toolchain.activate_msvc(environ) is True
    assert environ["INCLUDE"] == r"C:\VS\include"
    assert environ["LIB"] == r"C:\VS\lib"
    entries = environ["PATH"].split(os.pathsep)
    # The environment's own scripts stay first, so its pinned ninja is the one run.
    assert Path(entries[0]) == scripts
    assert str(msvc_bin) in entries


def test_a_failed_vcvars_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "find_vcvars", lambda *a: tmp_path / "vcvars64.bat")
    monkeypatch.setattr(toolchain, "_environment_after", lambda *a: {})
    environ = {"PATH": str(tmp_path)}

    assert toolchain.activate_msvc(environ) is False
    assert environ == {"PATH": str(tmp_path)}


@pytest.mark.skipif(os.name != "nt", reason="runs cmd.exe")
def test_a_batch_files_environment_is_read_back_through_cmd(tmp_path):
    """The quoting is the part that can only be checked for real: both the batch
    file and the interpreter can live under paths with spaces in them."""
    folder = tmp_path / "Visual Studio" / "VC"
    folder.mkdir(parents=True)
    vcvars = folder / "vcvars64.bat"
    vcvars.write_text(
        '@echo this is noise\r\n@set "UMLA_TOOLCHAIN_TEST=a b & c"\r\n'
        '@set "PATH=C:\\umla-fake-msvc;%PATH%"\r\n',
        encoding="ascii",
    )

    after = toolchain._environment_after(vcvars, dict(os.environ))

    assert after["UMLA_TOOLCHAIN_TEST"] == "a b & c"
    assert after["PATH"].split(os.pathsep)[0] == r"C:\umla-fake-msvc"
