"""Choosing the interpreter a worker job runs in."""

from __future__ import annotations

import sys

import pytest

from yolo_assistant.app.core import runtime
from yolo_assistant.app.core.paths import DATA_DIR_ENV
from yolo_assistant.app.core.project import Project


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "appdata"))
    monkeypatch.delenv(runtime.PYTHON_ENV, raising=False)
    return tmp_path


def test_no_environment_means_no_interpreter(isolated, monkeypatch):
    monkeypatch.setattr(runtime, "managed_environment", lambda: None)
    assert runtime.worker_python() is None


def test_the_override_wins(isolated, monkeypatch):
    monkeypatch.setenv(runtime.PYTHON_ENV, sys.executable)
    assert str(runtime.worker_python()) == sys.executable
    assert runtime.PYTHON_ENV in runtime.describe_python()


def test_a_job_without_an_environment_refuses_rather_than_falling_back(isolated, monkeypatch):
    monkeypatch.setattr(runtime, "managed_environment", lambda: None)

    # Falling back to the GUI's own Python would work right up to the first
    # "import torch", and the error would then name a module instead of a
    # missing environment.
    with pytest.raises(runtime.EnvironmentMissing, match="Step 1"):
        runtime.job_spec("probe", {})


def test_a_job_is_pointed_at_the_project_and_its_log(isolated, monkeypatch):
    monkeypatch.setenv(runtime.PYTHON_ENV, sys.executable)
    project = Project(isolated / "p")

    spec = runtime.job_spec("prepare_dataset", {"a": 1}, project=project, log_name="prepare")

    assert spec.python == sys.executable
    assert spec.project_dir == project.root
    assert spec.log_path == project.internal_dir / "logs" / "prepare.log"
    # Never the project folder: a stray file written beside the working
    # directory would land in the user's dataset.
    assert spec.cwd != project.root
    assert (spec.worker_dir / "worker" / "__main__.py").exists()


def test_the_first_usable_environment_is_chosen(isolated, monkeypatch):
    from yolo_assistant.app.core.bootstrap import venv_manager
    from yolo_assistant.app.core.bootstrap.planner import BACKENDS

    broken = venv_manager.EnvSpec(backend=BACKENDS[0])
    working = venv_manager.EnvSpec(backend=BACKENDS[1])
    monkeypatch.setattr(
        venv_manager,
        "discover",
        lambda: [
            (broken, venv_manager.VenvStatus(venv_manager.INCOMPLETE, "half built")),
            (working, venv_manager.VenvStatus(venv_manager.READY, "fine")),
        ],
    )

    spec, _status = runtime.managed_environment()
    assert spec is working
