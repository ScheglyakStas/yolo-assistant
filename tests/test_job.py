"""JobRunner integration tests.

These drive a real subprocess over a real loopback socket. They are the reason
M0 exists: if cancellation, tree reaping and crash reporting are not right here,
every later step inherits the bug.
"""

import sys
import time
from pathlib import Path

import psutil
import pytest

from yolo_assistant.app.core import job as job_mod
from yolo_assistant.app.core.job import JobRunner, JobSpec, JobState, kill_tree

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(app):
    r = JobRunner()
    yield r
    if r.is_active():
        proc = r._proc
        if proc is not None and proc.poll() is None:
            kill_tree(proc.pid)


class Collector:
    """Records everything a runner emits."""

    def __init__(self, runner: JobRunner):
        self.states, self.metrics, self.progress = [], [], []
        self.logs, self.errors, self.artifacts = [], [], []
        self.finished = None
        self.hello = None
        runner.state_changed.connect(self.states.append)
        runner.hello.connect(lambda d: setattr(self, "hello", d))
        runner.metrics.connect(lambda e, v, b: self.metrics.append((e, v, b)))
        runner.progress.connect(self.progress.append)
        runner.artifact.connect(lambda k, p_, m: self.artifacts.append((k, p_, m)))
        runner.log_line.connect(lambda t, lv: self.logs.append((lv, t)))
        runner.failed.connect(lambda c, m, h, tb: self.errors.append((c, m, h)))
        runner.finished.connect(lambda ok, payload: setattr(self, "finished", (ok, payload)))


def pump(app, predicate, timeout=30.0):
    """Spin the event loop until ``predicate`` holds or we run out of patience."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


def spec(**job) -> JobSpec:
    return JobSpec(
        command="fake_train",
        job=job,
        python=sys.executable,
        worker_dir=ROOT,
        cwd=ROOT,
    )


def test_happy_path_streams_metrics_and_result(app, runner):
    events = Collector(runner)
    runner.start(spec(epochs=3, batches_per_epoch=8, epoch_seconds=0.2, seed=7))

    assert pump(app, lambda: events.finished is not None), "job never finished"
    ok, payload = events.finished
    assert ok is True
    assert runner.state is JobState.SUCCEEDED
    assert payload["epochs"] == 3

    assert [e for e, _, _ in events.metrics] == [1, 2, 3]
    assert events.hello["versions"]["python"] == sys.version.split()[0]
    assert any(k == "weights" for k, _, _ in events.artifacts)

    # mAP improves, so the first epoch is never the best and the last one is.
    assert events.metrics[0][2] is True  # first epoch is best so far
    maps = [v["metrics/mAP50-95(B)"] for _, v, _ in events.metrics]
    assert maps[-1] > maps[0]


def test_progress_is_monotonic_within_a_phase(app, runner):
    events = Collector(runner)
    runner.start(spec(epochs=2, batches_per_epoch=40, epoch_seconds=0.6))
    assert pump(app, lambda: events.finished is not None)

    batches = [p["current"] for p in events.progress if p["phase"] in ("train", "warmup")]
    assert batches == sorted(batches), "coalescing reordered progress"
    assert len(batches) < 80, "throttling did not coalesce"
    assert batches[-1] <= 80


def test_cooperative_cancel_stops_promptly(app, runner):
    events = Collector(runner)
    runner.start(spec(epochs=50, batches_per_epoch=20, epoch_seconds=1.0))

    assert pump(app, lambda: runner.state is JobState.RUNNING), "worker never connected"
    assert pump(app, lambda: len(events.progress) > 2)

    started = time.monotonic()
    runner.cancel()
    assert pump(app, lambda: events.finished is not None, timeout=10.0)
    elapsed = time.monotonic() - started

    ok, payload = events.finished
    assert ok is False
    assert payload.get("cancelled") is True
    assert runner.state is JobState.CANCELLED
    # It stopped because it was asked, not because the ladder escalated.
    assert elapsed < job_mod.GRACE_COOPERATIVE_S, f"took {elapsed:.1f}s, ladder must have escalated"
    assert not any("killing the process tree" in text for _, text in events.logs)


def test_kill_ladder_reaps_an_unresponsive_tree(app, runner, monkeypatch):
    """A worker ignoring both the cancel frame and SIGTERM, with grandchildren."""
    monkeypatch.setattr(job_mod, "GRACE_COOPERATIVE_S", 0.5)
    monkeypatch.setattr(job_mod, "GRACE_SIGNAL_S", 0.5)

    events = Collector(runner)
    runner.start(
        spec(
            epochs=50,
            batches_per_epoch=20,
            epoch_seconds=1.0,
            ignore_cancel=True,
            ignore_sigterm=True,
            children=2,
        )
    )
    assert pump(app, lambda: runner.state is JobState.RUNNING)
    assert pump(app, lambda: len(events.progress) > 2)

    worker_pid = runner._proc.pid
    descendants = []
    assert pump(app, lambda: len(psutil.Process(worker_pid).children()) == 2)
    descendants = [c.pid for c in psutil.Process(worker_pid).children(recursive=True)]
    assert len(descendants) == 2

    runner.cancel()
    assert pump(app, lambda: events.finished is not None, timeout=20.0)

    assert runner.state is JobState.CANCELLED
    assert any("killing the process tree" in text for _, text in events.logs), (
        "escalation never reached the tree kill"
    )
    # Nothing survived -- the point of the whole exercise.
    for pid in [worker_pid, *descendants]:
        assert not _alive(pid), f"pid {pid} survived the kill ladder"


def test_a_signal_that_cannot_be_delivered_still_reaps_the_tree(app, runner, monkeypatch):
    """The installed Windows app's normal case, simulated.

    A console control event reaches only processes sharing the sender's
    console, and a windowed app has none, so the signal rung fails there every
    time. Falling back to ``terminate()`` used to end the worker alone: the
    kill rung then found it already gone, and its dataloader workers ran on.
    """
    monkeypatch.setattr(job_mod, "GRACE_COOPERATIVE_S", 0.5)

    def undeliverable(*_args):
        raise OSError("no console to deliver the event to")

    monkeypatch.setattr(job_mod.os, "killpg", undeliverable)

    events = Collector(runner)
    runner.start(
        spec(epochs=50, batches_per_epoch=20, epoch_seconds=1.0, ignore_cancel=True, children=2)
    )
    assert pump(app, lambda: runner.state is JobState.RUNNING)
    worker_pid = runner._proc.pid
    assert pump(app, lambda: len(psutil.Process(worker_pid).children()) == 2)
    descendants = [c.pid for c in psutil.Process(worker_pid).children(recursive=True)]

    runner.cancel()
    assert pump(app, lambda: events.finished is not None, timeout=20.0)

    assert runner.state is JobState.CANCELLED
    for pid in [worker_pid, *descendants]:
        assert not _alive(pid), f"pid {pid} survived an undeliverable signal"


def test_crash_without_result_is_reported_with_a_code(app, runner):
    events = Collector(runner)
    runner.start(spec(epochs=5, batches_per_epoch=10, epoch_seconds=0.2, crash_after_batches=3))

    assert pump(app, lambda: events.finished is not None)
    ok, payload = events.finished
    assert ok is False
    assert runner.state is JobState.FAILED
    assert payload["exit_code"] == 3
    assert events.errors, "a silent crash must still produce a coded error"
    code, msg, hint = events.errors[0]
    assert code == "unknown"
    assert "out of memory" in hint.lower()


def test_second_job_while_running_is_refused(app, runner):
    runner.start(spec(epochs=10, batches_per_epoch=10, epoch_seconds=0.5))
    assert pump(app, lambda: runner.state is JobState.RUNNING)
    with pytest.raises(RuntimeError, match="already running"):
        runner.start(spec(epochs=1))
    runner.cancel()
    assert pump(app, lambda: not runner.is_active(), timeout=10.0)


def test_kill_tree_returns_count(app):
    import subprocess

    parent = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess,sys,time\n"
         "subprocess.Popen([sys.executable,'-c','import time\\nwhile True: time.sleep(3600)'])\n"
         "time.sleep(3600)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)
    killed = kill_tree(parent.pid)
    assert killed >= 2, f"expected parent + child, killed {killed}"
    parent.wait(timeout=5)


def _alive(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


# --------------------------------------------------------------------------- #
# The environment the worker actually runs in
# --------------------------------------------------------------------------- #


def test_a_worker_can_run_a_console_script_from_its_own_environment(app, tmp_path):
    """The whole point of putting the environment's scripts on PATH.

    A library that shells out to a console script its own wheel installed --
    torch running ``ninja --version`` to decide whether it can build a CUDA
    extension is the case that found this -- looks the script up on PATH, not
    through Python's import machinery. Launching ``<venv>/bin/python`` directly
    does not put ``<venv>/bin`` on PATH, so before this the script was installed,
    importable, and still not there.

    Driven against a stand-in layout rather than a real venv: what matters is
    that a script sitting beside the interpreter is reachable by bare name.
    """
    import os
    import shutil
    import stat
    import subprocess
    import sys

    from yolo_assistant.app.core.worker_env import build_worker_env

    fake_env = tmp_path / "venv"
    scripts = fake_env / ("Scripts" if os.name == "nt" else "bin")
    scripts.mkdir(parents=True)

    interpreter = scripts / Path(sys.executable).name
    # A copy, not a symlink: a symlink would resolve back to the real
    # interpreter's own directory and prove nothing about PATH.
    shutil.copy2(sys.executable, interpreter)

    script = scripts / ("umla-probe.bat" if os.name == "nt" else "umla-probe")
    script.write_text(
        "@echo found\n" if os.name == "nt" else "#!/bin/sh\necho found\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    def which_umla_probe(env) -> str:
        found = subprocess.run(
            [str(interpreter), "-c", "import shutil; print(shutil.which('umla-probe'))"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert found.returncode == 0, found.stderr
        return found.stdout.strip()

    base = dict(os.environ)
    assert (
        which_umla_probe(build_worker_env(tmp_path, python=interpreter, base=base))
        == str(script)
    )
    # And the bug this fixed: the interpreter finds its own packages either way,
    # which is why nothing noticed for so long.
    assert which_umla_probe(build_worker_env(tmp_path, base=base)) == "None"
