"""Updating Ultralytics in place, with the plan shown first.

The hazard being guarded against is not a failed update -- it is a *successful*
one that quietly swaps the CUDA build of PyTorch for a CPU build from PyPI,
because Ultralytics only asks for ``torch>=1.8``. The user would find out at
their next training run, with no obvious connection to the button they pressed.
So: the resolution is shown before anything is installed, the CUDA index stays
primary throughout, and the environment has to prove itself again afterwards.
"""

import os
import stat
import sys
import time

import pytest

from yolo_assistant.app.core import paths
from yolo_assistant.app.core.bootstrap import uv_manager, venv_manager as vm
from yolo_assistant.app.core.bootstrap.updater import UpdateController

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="the stand-in interpreter is a POSIX symlink"
)

FAKE_UV = '''#!/usr/bin/env python3
import os, pathlib, sys

argv = sys.argv[1:]
log = pathlib.Path(os.environ["FAKE_UV_LOG"])
log.write_text(log.read_text() + " ".join(argv) + "\\n")
if os.environ.get("FAKE_UV_FAIL"):
    print("error: No solution found when resolving dependencies:", file=sys.stderr)
    sys.exit(1)
if "--dry-run" in argv:
    print("Resolved 137 packages in 210ms")
    print("Would install 1 package")
    print(" + ultralytics==8.4.200")
else:
    print("Resolved 137 packages in 210ms")
    print("Prepared 1 package in 2s")
    print("Installed 1 package in 30ms")
    print(" + ultralytics==8.4.200")
'''

PROBE = {
    "torch": "2.14.0+cu132",
    "cuda_build": "13.2",
    "cuda_available": True,
    "ultralytics": "8.4.154",
    "devices": [{"index": 0, "name": "RTX 5070 Ti"}],
    # What the probe reported it could import. An environment whose packages
    # were never checked reads as incomplete, which is a different test.
    "packages": {"json": "unknown"},
}


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    calls = tmp_path / "uv-calls.log"
    calls.write_text("")
    monkeypatch.setenv("FAKE_UV_LOG", str(calls))

    binary = tmp_path / "uv"
    binary.write_text(FAKE_UV)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(uv_manager, "find_uv", lambda: binary)
    # The verify stage runs the real probe under this test run's interpreter,
    # which has no torch. Check for something it does have.
    monkeypatch.setattr(vm, "REQUIRED_PACKAGES", (("json", "these tests"),))
    monkeypatch.setattr(vm, "OPTIONAL_PACKAGES", ())
    monkeypatch.setattr(vm, "PACKAGE_NAMES", ("json",))

    spec = vm.spec_for("cu132")
    spec.python.parent.mkdir(parents=True)
    os.symlink(sys.executable, spec.python)
    vm.write_ready(spec, PROBE)
    return spec, calls


def drive(app, controller, action, spec, timeout=60.0):
    box = {"done": None, "plan": None, "logs": [], "failures": []}
    controller.finished.connect(lambda ok, status: box.__setitem__("done", (ok, status)))
    controller.plan_ready.connect(lambda lines: box.__setitem__("plan", lines))
    controller.log_line.connect(lambda text, level: box["logs"].append(text))
    controller.failed.connect(lambda c, m, h: box["failures"].append((c, m, h)))

    action(spec, "ultralytics==8.4.200")
    deadline = time.monotonic() + timeout
    while box["done"] is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    return box


def test_the_preview_shows_the_resolution_and_changes_nothing(app, environment):
    spec, calls = environment
    controller = UpdateController()
    box = drive(app, controller, controller.preview, spec)

    assert box["done"] == (True, None)
    assert any("+ ultralytics==8.4.200" in line for line in box["plan"])
    assert "--dry-run" in calls.read_text()
    # Untouched: the environment is still the one it was before.
    assert vm.status(spec).state == vm.READY
    assert vm.status(spec).ready.ultralytics == "8.4.154"


def test_the_preview_does_not_run_a_probe(app, environment):
    spec, calls = environment
    controller = UpdateController()
    drive(app, controller, controller.preview, spec)
    assert calls.read_text().count("pip install") == 1


def test_applying_reinstalls_and_reverifies(app, environment):
    spec, calls = environment
    controller = UpdateController()
    box = drive(app, controller, controller.apply, spec)

    ok, status = box["done"]
    assert ok is True, box["failures"]
    assert "--dry-run" not in calls.read_text()
    assert status.state in (vm.READY, vm.DEGRADED)
    # ready.json was rewritten from a fresh probe, not left at its old contents.
    assert status.ready.installed_at >= 0


def test_the_cuda_index_stays_primary_during_an_update(app, environment):
    """Otherwise an Ultralytics bump can satisfy torch from PyPI and drop the GPU."""
    spec, calls = environment
    controller = UpdateController()
    drive(app, controller, controller.apply, spec)

    line = next(l for l in calls.read_text().splitlines() if l.startswith("pip install"))
    args = line.split()
    assert args[args.index("--torch-backend") + 1] == "cu132"


def test_the_environment_is_marked_unproven_while_it_is_being_changed(app, environment):
    spec, _calls = environment
    controller = UpdateController()
    states = []
    controller.busy_changed.connect(lambda busy: states.append(vm.status(spec).state))

    drive(app, controller, controller.apply, spec)
    assert states[0] != vm.READY, "ready.json survived the start of an update"


def test_a_failed_update_reports_a_code_and_leaves_it_unproven(app, environment, monkeypatch):
    spec, _calls = environment
    monkeypatch.setenv("FAKE_UV_FAIL", "1")
    controller = UpdateController()
    box = drive(app, controller, controller.apply, spec)

    ok, status = box["done"]
    assert ok is False and status is None
    assert box["failures"][0][0] == "dependency_install_failed"
    assert vm.status(spec).state == vm.INCOMPLETE


def test_an_update_without_uv_says_so_instead_of_crashing(app, environment, monkeypatch):
    spec, _calls = environment
    monkeypatch.setattr(uv_manager, "find_uv", lambda: None)
    controller = UpdateController()
    box = drive(app, controller, controller.preview, spec)

    assert box["done"] == (False, None)
    assert box["failures"][0][0] == "venv_broken"
    assert "Run environment setup first" in box["failures"][0][2]
