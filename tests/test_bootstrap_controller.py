"""The five bootstrap stages, driven end to end against a stand-in for uv.

The fake uv below mimics the one behaviour that matters for ordering: real uv
*refuses to create an environment in a directory that already exists*. An
earlier version of the controller wrote the generated ``requirements.txt`` into
the environment directory before asking uv to create it, and every first-time
setup failed at stage three. These tests exist so that cannot come back.

The final stage runs the real ``worker probe`` through a symlinked interpreter,
so the sequence is exercised all the way to ``ready.json``.
"""

import os
import stat
import sys
import time

import pytest

from yolo_assistant.app.core import paths
from yolo_assistant.app.core.bootstrap import uv_manager, venv_manager as vm
from yolo_assistant.app.core.bootstrap.controller import BootstrapController

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="the stand-in interpreter is a POSIX symlink"
)

FAKE_UV = '''#!/usr/bin/env python3
"""Enough of uv to drive the controller, including its refusal to overwrite."""
import os, pathlib, sys

argv = sys.argv[1:]
log = pathlib.Path(os.environ["FAKE_UV_LOG"])
log.write_text(log.read_text() + " ".join(argv) + "\\n")

if argv[:2] == ["python", "install"]:
    print("Downloading cpython-3.12.14-linux-x86_64-gnu (download) (32.6MiB)")
    print(" Downloaded cpython-3.12.14-linux-x86_64-gnu (download)")
    print("Installed Python 3.12.14 in 3.16s")
elif argv[:1] == ["venv"]:
    target = pathlib.Path(argv[-1])
    if target.exists():
        print("error: Failed to create virtual environment", file=sys.stderr)
        print("  cause: A directory already exists at: %s" % target, file=sys.stderr)
        sys.exit(2)
    (target / "bin").mkdir(parents=True)
    os.symlink(os.environ["FAKE_UV_PYTHON"], target / "bin" / "python")
    print("Creating virtual environment at: %s" % target)
elif argv[:2] == ["pip", "install"]:
    requirements = pathlib.Path(argv[argv.index("-r") + 1])
    if not requirements.exists():
        print("error: No such file: %s" % requirements, file=sys.stderr)
        sys.exit(2)
    print("Resolved 4 packages in 200ms")
    print("Downloading torch (3000.0MiB)")
    print(" Downloaded torch")
    print("Prepared 4 packages in 1s")
    print("Installed 4 packages in 1s")
else:
    print("error: unexpected arguments %r" % argv, file=sys.stderr)
    sys.exit(2)
'''


@pytest.fixture
def fake_uv(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    calls = tmp_path / "uv-calls.log"
    calls.write_text("")
    monkeypatch.setenv("FAKE_UV_LOG", str(calls))
    monkeypatch.setenv("FAKE_UV_PYTHON", sys.executable)

    binary = tmp_path / "uv"
    binary.write_text(FAKE_UV)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(uv_manager, "find_uv", lambda: binary)

    # The stand-in interpreter is this test run's own, which has no torch and no
    # ultralytics, so the verify stage would correctly report an environment
    # that cannot do the work. Point the check at a module every interpreter
    # has: what is under test here is the sequence, not the package list.
    monkeypatch.setattr(vm, "REQUIRED_PACKAGES", (("json", "these tests"),))
    monkeypatch.setattr(vm, "OPTIONAL_PACKAGES", ())
    monkeypatch.setattr(vm, "PACKAGE_NAMES", ("json",))
    return calls


def run(app, controller, spec, timeout=90.0):
    box = {"done": None, "stages": [], "logs": [], "failures": [], "progress": []}
    controller.finished.connect(lambda ok, status: box.__setitem__("done", (ok, status)))
    controller.stage_changed.connect(lambda key, *_rest: box["stages"].append(key))
    controller.log_line.connect(lambda text, level: box["logs"].append((level, text)))
    controller.failed.connect(lambda c, m, h: box["failures"].append((c, m, h)))
    controller.progress.connect(box["progress"].append)

    controller.start(spec)
    deadline = time.monotonic() + timeout
    while box["done"] is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    return box


def test_a_first_time_setup_runs_every_stage_and_writes_ready(app, fake_uv):
    spec = vm.spec_for("cpu")
    box = run(app, BootstrapController(), spec)

    assert box["done"] is not None, "the bootstrap never finished"
    ok, status = box["done"]
    assert ok is True, box["failures"]
    assert box["stages"] == ["uv", "python", "venv", "packages", "verify"]
    assert status.state == vm.READY

    ready = status.ready
    assert ready.backend == "cpu" and ready.fingerprint == spec.fingerprint
    assert ready.installed_at > 0


def test_an_environment_missing_a_package_does_not_finish_as_ready(app, fake_uv, monkeypatch):
    """Controller to probe to ready.json to status, over one missing import.

    This is the scipy bug in miniature: the install reports success, and the
    only thing standing between that and a failure three steps later is the
    verify stage actually importing what the later steps import.
    """
    monkeypatch.setattr(
        vm, "REQUIRED_PACKAGES", (("json", "these tests"), ("no_such_module", "autolabeling"))
    )
    monkeypatch.setattr(vm, "PACKAGE_NAMES", ("json", "no_such_module"))
    spec = vm.spec_for("cpu")

    box = run(app, BootstrapController(), spec)

    assert box["done"][0] is True, "the install itself did not fail"
    status = box["done"][1]
    assert status.state == vm.INCOMPLETE
    assert "no_such_module" in status.message and "autolabeling" in status.message
    assert any("cannot be imported" in text for _lvl, text in box["logs"])


def test_the_requirements_file_is_written_where_uv_can_read_it(app, fake_uv):
    """The regression: it must land inside the environment, but only after uv
    has created it."""
    spec = vm.spec_for("cpu")
    box = run(app, BootstrapController(), spec)
    assert box["done"][0] is True, box["failures"]

    text = spec.requirements_file.read_text()
    lines = [line.strip() for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert any(line.startswith("ultralytics") for line in lines) and "torch" in text

    order = fake_uv.read_text().splitlines()
    venv_line = next(i for i, line in enumerate(order) if line.startswith("venv "))
    pip_line = next(i for i, line in enumerate(order) if line.startswith("pip install"))
    assert venv_line < pip_line


def test_debris_from_a_failed_attempt_is_cleared(app, fake_uv):
    """uv will not create an environment over an existing directory, so a
    directory with no interpreter in it has to be removed first."""
    spec = vm.spec_for("cpu")
    spec.path.mkdir(parents=True)
    (spec.path / "half-written.txt").write_text("junk")

    box = run(app, BootstrapController(), spec)
    assert box["done"][0] is True, box["failures"]
    assert not (spec.path / "half-written.txt").exists()
    assert any("Removing an unusable environment" in text for _lvl, text in box["logs"])


def test_an_existing_interpreter_is_reused_rather_than_rebuilt(app, fake_uv):
    """Otherwise repairing an interrupted install would re-download everything."""
    spec = vm.spec_for("cpu")
    assert run(app, BootstrapController(), spec)["done"][0] is True

    fake_uv.write_text("")
    box = run(app, BootstrapController(), spec)
    assert box["done"][0] is True, box["failures"]
    assert not any(line.startswith("venv ") for line in fake_uv.read_text().splitlines())
    assert any("Reusing the existing environment" in text for _lvl, text in box["logs"])


def test_repairing_a_cpu_torch_forces_the_gpu_build_in(app, fake_uv):
    """The CPU wheel satisfies ``torch>=2.12`` as well as the CUDA one does, so a
    plain re-run of the install changes nothing -- which is exactly what three
    clicks of Repair did on a real machine before this."""
    spec = vm.spec_for("cu132")
    assert run(app, BootstrapController(), spec)["done"] is not None
    site = spec.path / "lib" / "python3.12" / "site-packages" / "torch"
    site.mkdir(parents=True)
    (site / "version.py").write_text("cuda: Optional[str] = None\n")

    fake_uv.write_text("")
    box = run(app, BootstrapController(), spec)

    pip = next(l for l in fake_uv.read_text().splitlines() if l.startswith("pip install"))
    assert "--torch-backend cu132" in pip
    assert "--reinstall-package torch" in pip and "--reinstall-package torchvision" in pip
    assert any("CPU-only PyTorch" in text for _lvl, text in box["logs"])


def test_the_environment_stops_claiming_to_be_ready_while_it_is_being_changed(app, fake_uv):
    spec = vm.spec_for("cpu")
    assert run(app, BootstrapController(), spec)["done"][0] is True
    assert vm.status(spec).state == vm.READY

    seen = []
    controller = BootstrapController()
    controller.stage_changed.connect(lambda *_a: seen.append(vm.status(spec).state))
    run(app, controller, spec)
    assert seen and seen[0] != vm.READY, "ready.json survived the start of a rebuild"


def test_a_failing_stage_reports_a_code_and_stops(app, fake_uv, monkeypatch):
    spec = vm.spec_for("cpu")
    # Make the package install fail by pointing it at a requirements file that
    # will not exist, the way a resolution failure would.
    monkeypatch.setattr(vm, "write_requirements", lambda _spec, _text: _spec.requirements_file)

    box = run(app, BootstrapController(), spec)
    ok, status = box["done"]
    assert ok is False and status is None
    assert box["failures"], "a failed install must report a coded error"
    code, _msg, _hint = box["failures"][0]
    assert code == "dependency_install_failed"
    assert box["stages"] == ["uv", "python", "venv", "packages"], "it must not carry on"
    assert vm.status(spec).state == vm.INCOMPLETE


def test_progress_keeps_moving_while_uv_says_nothing(app, fake_uv):
    """torch is a single three-gigabyte wheel and uv prints nothing for its whole
    download. Without a heartbeat the window would look frozen for minutes."""
    controller = BootstrapController()
    seen = []
    controller.progress.connect(seen.append)

    controller._active = True
    controller._emit_progress({"phase": "install", "current": 23, "total": 74})
    before = len(seen)
    controller._on_tick()
    controller._on_tick()
    assert len(seen) == before + 2
    assert seen[-1] == seen[before - 1], "the heartbeat must repeat, not invent, progress"

    controller._set_active(False)
    controller._on_tick()
    assert len(seen) == before + 2, "the heartbeat must stop when the job does"


def test_progress_is_reported_during_the_long_stage(app, fake_uv):
    box = run(app, BootstrapController(), vm.spec_for("cpu"))
    assert box["done"][0] is True
    install = [p for p in box["progress"] if p.get("phase") == "install"]
    assert install, "the package stage reported no progress at all"
    assert any("torch" in str(p.get("detail", "")) for p in install)
    assert install[-1]["current"] == install[-1]["total"] == 4
