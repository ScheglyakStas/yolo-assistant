"""Environment scrubbing for worker subprocesses."""

import os

import pytest

from yolo_assistant.app.core.worker_env import (
    LEAKED_VARS,
    build_worker_env,
    scripts_dir,
)


@pytest.fixture
def frozen_env():
    """Roughly what PyInstaller leaves in the environment of a frozen app."""
    return {
        "PATH": "/usr/bin",
        "HOME": "/home/user",
        "LD_LIBRARY_PATH": "/tmp/_MEI123/lib",
        "LD_LIBRARY_PATH_ORIG": "/usr/local/lib",
        "QT_PLUGIN_PATH": "/tmp/_MEI123/PySide6/plugins",
        "SSL_CERT_FILE": "/tmp/_MEI123/certifi/cacert.pem",
        "PYTHONHOME": "/tmp/_MEI123",
        "PYTHONPATH": "/tmp/_MEI123",
        "_MEIPASS2": "/tmp/_MEI123",
        "_PYI_APPLICATION_HOME_DIR": "/tmp/_MEI123",
    }


def test_bundled_library_paths_do_not_reach_the_child(frozen_env):
    """The failure this prevents is torch loading PyInstaller's libstdc++."""
    env = build_worker_env("/opt/worker", base=frozen_env)
    for name in ("QT_PLUGIN_PATH", "SSL_CERT_FILE", "PYTHONHOME"):
        assert name not in env, f"{name} leaked into the worker environment"
    assert not [k for k in env if k.startswith("_PYI_")]
    assert "_MEIPASS2" not in env


def test_original_value_is_restored_not_just_deleted(frozen_env):
    """A user may legitimately have had LD_LIBRARY_PATH set before launch."""
    env = build_worker_env("/opt/worker", base=frozen_env)
    assert env["LD_LIBRARY_PATH"] == "/usr/local/lib"
    assert "LD_LIBRARY_PATH_ORIG" not in env


def test_unrelated_variables_are_preserved(frozen_env):
    env = build_worker_env("/opt/worker", base=frozen_env)
    assert env["HOME"] == "/home/user"
    assert env["PATH"] == "/usr/bin"


# --------------------------------------------------------------------------- #
# The environment's own console scripts
# --------------------------------------------------------------------------- #


def test_the_environments_scripts_come_first_on_path(frozen_env):
    """Launching <venv>/bin/python does not put <venv>/bin on PATH; only
    activation does, and the app never activates anything. Everything that
    shells out to a console script its own wheel installed depends on this --
    ninja, which torch runs rather than imports to build CUDA extensions, is
    how the gap was found."""
    env = build_worker_env("/opt/worker", python="/venvs/cu128/bin/python", base=frozen_env)

    assert env["PATH"].split(os.pathsep) == ["/venvs/cu128/bin", "/usr/bin"]


def test_the_scripts_directory_is_not_added_twice(frozen_env):
    """Restarting a job must not grow PATH one entry at a time."""
    once = build_worker_env("/opt/worker", python="/venvs/cu128/bin/python", base=frozen_env)
    twice = build_worker_env("/opt/worker", python="/venvs/cu128/bin/python", base=once)

    assert once["PATH"] == twice["PATH"]


def test_an_empty_inherited_path_is_still_valid(frozen_env):
    env = build_worker_env(
        "/opt/worker", python="/venvs/cu128/bin/python", base={"PATH": ""}
    )

    assert env["PATH"] == "/venvs/cu128/bin"


def test_the_environment_is_announced_the_way_activation_would(frozen_env):
    env = build_worker_env("/opt/worker", python="/venvs/cu128/bin/python", base=frozen_env)

    assert env["VIRTUAL_ENV"] == "/venvs/cu128"


def test_path_is_left_alone_without_an_interpreter(frozen_env):
    """The bootstrap runs uv, which is not inside the managed environment and
    must not have its scripts shadowing the system's."""
    env = build_worker_env("/opt/worker", base=frozen_env)

    assert env["PATH"] == "/usr/bin"
    assert "VIRTUAL_ENV" not in env


def test_scripts_live_beside_the_interpreter_on_both_platforms():
    """``bin`` here, ``Scripts`` on Windows -- in both cases the interpreter's
    own directory, which is why the function needs no platform test.

    The Windows half is checked through ``PureWindowsPath`` rather than through
    ``scripts_dir`` itself, because a Linux ``Path`` cannot parse a Windows path
    at all: it reads the whole thing as a single filename with no parent.
    """
    from pathlib import PureWindowsPath

    assert scripts_dir("/venvs/cu128/bin/python").name == "bin"
    assert PureWindowsPath(r"C:\venvs\cu128\Scripts\python.exe").parent.name == "Scripts"


def test_pythonpath_points_only_at_the_worker(frozen_env):
    """The worker ships as loose source; this is how the venv's python finds it."""
    env = build_worker_env("/opt/worker", base=frozen_env)
    assert env["PYTHONPATH"] == "/opt/worker"
    assert env["PYTHONNOUSERSITE"] == "1"


def test_ultralytics_is_kept_inside_the_project(frozen_env):
    env = build_worker_env("/opt/worker", project_dir="/data/proj", base=frozen_env)
    assert env["YOLO_CONFIG_DIR"] == "/data/proj/.ultralytics"
    assert env["MPLBACKEND"] == "Agg"


def test_caches_are_redirected(frozen_env):
    env = build_worker_env("/opt/worker", cache_dir="/var/cache/umla", base=frozen_env)
    assert env["HF_HOME"] == "/var/cache/umla/huggingface"
    assert env["TORCH_HOME"] == "/var/cache/umla/torch"


def test_extra_overrides_win(frozen_env):
    env = build_worker_env("/opt/worker", base=frozen_env, extra={"CUDA_VISIBLE_DEVICES": "1"})
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


@pytest.mark.parametrize("name", LEAKED_VARS)
def test_every_declared_leak_is_actually_scrubbed(name):
    env = build_worker_env("/opt/worker", base={name: "/tmp/_MEI999"})
    assert env.get(name) != "/tmp/_MEI999"
