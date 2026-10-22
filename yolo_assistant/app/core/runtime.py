"""Which interpreter a worker job runs in, and how to point a job at a project.

Every step from 2 onwards launches worker commands, and every one of them needs
the same three facts: the managed environment's Python, where the ``worker``
package can be imported from, and where this job's output and log belong. Each
page working that out for itself is how they end up subtly disagreeing, so it is
worked out once, here.

The interpreter is found by looking at the filesystem rather than by asking
Step 1, because the user may never have opened Step 1 in this session -- the
environment was built last week and the app remembers nothing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .bootstrap import venv_manager
from .bootstrap.venv_manager import EnvSpec, VenvStatus
from .i18n import tr
from .job import JobSpec
from .paths import cache_dir, logs_dir, worker_source_dir
from .project import Project

#: Point this at any interpreter to run worker commands under it instead of the
#: managed environment. Intended for development -- the commands that need only
#: Pillow run perfectly well under the GUI's own Python -- and for recovering a
#: machine whose managed environment will not build.
PYTHON_ENV = "UMLA_WORKER_PYTHON"


def managed_environment() -> tuple[EnvSpec, VenvStatus] | None:
    """The environment worker jobs should run in, or ``None`` if there isn't one.

    When several exist -- a rebuild onto a newer CUDA backend leaves the old one
    in place until the new one proves itself -- the first usable one wins, and
    ``discover`` returns them in the planner's preference order.
    """
    for spec, status in venv_manager.discover():
        if status.is_usable:
            return spec, status
    return None


def worker_python() -> Path | None:
    """The interpreter to launch, honouring :data:`PYTHON_ENV`."""
    override = os.environ.get(PYTHON_ENV)
    if override:
        return Path(override)
    found = managed_environment()
    return found[0].python if found is not None else None


def job_spec(
    command: str,
    job: dict[str, Any],
    *,
    project: Project | None = None,
    log_name: str | None = None,
    **extra: Any,
) -> JobSpec:
    """A :class:`JobSpec` for one managed-environment command.

    Raises :class:`EnvironmentMissing` rather than falling back to the GUI's own
    interpreter: that would appear to work, right up to the first ``import
    torch``, and the error would then name a missing module instead of a missing
    environment.
    """
    python = worker_python()
    if python is None:
        raise EnvironmentMissing(
            tr(
                "The Python environment this app uses has not been set up yet. "
                "Finish Step 1 first."
            )
        )

    log_path = None
    if log_name:
        log_path = (
            project.log_path(log_name) if project is not None else logs_dir() / f"{log_name}.log"
        )

    return JobSpec(
        command=command,
        job=job,
        python=str(python),
        worker_dir=worker_source_dir(),
        project_dir=project.root if project is not None else None,
        cache_dir=cache_dir(),
        # Never the project folder: a worker that writes a stray file beside its
        # working directory would drop it into the user's dataset.
        cwd=worker_source_dir(),
        log_path=log_path,
        **extra,
    )


class EnvironmentMissing(RuntimeError):
    """No interpreter is available to run worker commands in."""


def describe_python() -> str:
    """One line for the log pane, so a surprising interpreter is visible."""
    python = worker_python()
    if python is None:
        return tr("No managed Python environment.")
    if os.environ.get(PYTHON_ENV):
        return tr("Using {python} (from {variable}).", python=python, variable=PYTHON_ENV)
    return tr("Using the managed environment at {python}.", python=python)


def gui_python() -> str:
    """The GUI's own interpreter. Used by tests and the simulated-run page."""
    return sys.executable
